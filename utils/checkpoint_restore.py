#!/usr/bin/env python3
"""
正确的 Checkpoint 恢复工具
专家建议：绝不要用 .ply 初始化，只用 Task 1 的 checkpoint 恢复完整状态
"""
import torch


def restore_from_task1_checkpoint(gaussians, ckpt_path, opt, device):
    """
    从 Task1 checkpoint 恢复完整 Gaussians
    
    正确顺序：
    1. 先建立优化器 param groups（与 Task1 相同的超参）
    2. 恢复参数（包含 xyz/scale/rot/opacity/SH 等）
    3. 恢复优化器状态（若 shape 完全一致）
    4. 绝对不要在此之后再做 init_* 或 reset_* 之类会覆盖参数的操作
    
    Args:
        gaussians: GaussianModel 实例
        ckpt_path: checkpoint 文件路径
        opt: OptimizationParams
        device: 设备
    
    Returns:
        (gaussians, iteration): 恢复后的模型和迭代数
    """
    print(f"[RESTORE] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 检查 checkpoint 格式
    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_params, iteration = ckpt
        print(f"[RESTORE] Checkpoint format: (model_params, iteration={iteration})")
        
        # 解包 model_params（15个元素）
        if len(model_params) != 15:
            raise ValueError(f"Expected 15 elements in model_params, got {len(model_params)}")
        
        (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity,
         embeddings, appearance_embeddings, appearance_mlp, max_radii2D,
         xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = model_params
        
        # 1) 先建立优化器 param groups（与 Task1 相同的超参）
        gaussians.training_setup(opt)
        print(f"[RESTORE] Training setup completed")
        
        # 2) 手动恢复参数（不使用 restore，避免优化器冲突）
        gaussians.active_sh_degree = active_sh_degree
        gaussians._xyz = xyz
        gaussians._features_dc = features_dc
        gaussians._features_rest = features_rest
        gaussians._scaling = scaling
        gaussians._rotation = rotation
        gaussians._opacity = opacity
        gaussians._embeddings = embeddings
        gaussians.appearance_embeddings = appearance_embeddings
        gaussians.appearance_mlp = appearance_mlp
        gaussians.max_radii2D = max_radii2D
        gaussians.xyz_gradient_accum = xyz_gradient_accum
        gaussians.denom = denom
        gaussians.spatial_lr_scale = spatial_lr_scale
        
        print(f"[RESTORE] Parameters restored: {gaussians.get_xyz.shape[0]} points")
        
        # 3) 尝试恢复优化器状态（若 shape 完全一致）
        try:
            gaussians.optimizer.load_state_dict(opt_dict)
            print(f"[RESTORE] ✅ Optimizer state loaded")
        except Exception as e:
            print(f"[RESTORE] ⚠️  Optimizer state not loaded: {e}. Proceed with fresh optimizer.")
        
        # 4) 诊断信息
        print(f"[RESTORE] 诊断信息:")
        print(f"  - 点数: {gaussians.get_xyz.shape[0]}")
        opacity_activated = torch.sigmoid(gaussians._opacity)
        print(f"  - opacity (activated): min={opacity_activated.min().item():.6f}, "
              f"max={opacity_activated.max().item():.6f}, mean={opacity_activated.mean().item():.6f}")
        print(f"  - features_dc: min={gaussians._features_dc.min().item():.6f}, "
              f"max={gaussians._features_dc.max().item():.6f}")
        
        return gaussians, iteration
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")


def ensure_filter_3d(gaussians, n_pts, device):
    """
    filter_3D 尺寸守护 + 空掩码回退
    
    正确做法：
    - 若维度不匹配 → 重建为全 0（禁用过滤，允许全部点渲染）
    - filter_3D 的形状必须是 (N, 1)，dtype 是 float32
    - filter_3D = 0 表示禁用过滤（coef ≈ 1）
    
    Args:
        gaussians: GaussianModel 实例
        n_pts: 点数
        device: 设备
    """
    if (not hasattr(gaussians, "filter_3D")) or (gaussians.filter_3D is None) \
       or (gaussians.filter_3D.shape != (n_pts, 1)):
        print(f"[FILTER] Rebuilding filter_3D -> zeros (n={n_pts}, shape=(n,1))")
        print(f"[FILTER] filter_3D=0 表示禁用过滤，允许全部点渲染")
        gaussians.filter_3D = torch.zeros(n_pts, 1, dtype=torch.float32, device=device)


@torch.no_grad()
def safe_compute_3d_filter(gaussians, cameras):
    """
    安全地计算 3D filter
    
    如果失败，保持 filter_3D = 0（禁用过滤）
    
    Args:
        gaussians: GaussianModel 实例
        cameras: 相机列表
    """
    try:
        if hasattr(gaussians, 'compute_3D_filter'):
            gaussians.compute_3D_filter(cameras=cameras)
            # 确保是 float32
            if gaussians.filter_3D.dtype != torch.float32:
                gaussians.filter_3D = gaussians.filter_3D.float()
            print(f"[FILTER] compute_3D_filter completed")
            print(f"[FILTER] filter_3D: mean={gaussians.filter_3D.mean().item():.4f}, "
                  f"max={gaussians.filter_3D.max().item():.4f}")
    except Exception as e:
        print(f"[FILTER] ⚠️  compute_3D_filter failed: {e}. Keep filter_3D=0 (disabled).")
        # 保持 filter_3D = 0（禁用过滤）
        n_pts = gaussians.get_xyz.shape[0]
        gaussians.filter_3D = torch.zeros(n_pts, 1, dtype=torch.float32, device=gaussians.get_xyz.device)
    
    # 统计信息
    if gaussians.filter_3D is not None:
        print(f"[FILTER] ✅ filter_3D shape: {gaussians.filter_3D.shape}, "
              f"dtype: {gaussians.filter_3D.dtype}")


def unpack_render_dict(out, background, device):
    """
    兼容不同 renderer 的输出字典
    
    统一取 rgb 和 alpha，并固定合成顺序
    
    Args:
        out: 渲染器输出字典
        background: 背景颜色 tensor
        device: 设备
    
    Returns:
        (rgb, alpha, comp): Linear RGB, alpha, 合成图
    """
    # 尝试不同 key 取 rgb（linear, NOT sRGB）
    rgb = out.get("render", None)
    if rgb is None:
        rgb = out.get("rgb", None)
    if rgb is None:
        rgb = out.get("color", None)
    assert rgb is not None, "Renderer output missing 'render/rgb/color'."
    
    # 兼容 alpha / accumulation / rendered_alpha
    alpha = out.get("alpha", None)
    if alpha is None:
        alpha = out.get("accumulation", None)
    if alpha is None:
        alpha = out.get("rendered_alpha", None)
    if alpha is None:
        alpha = out.get("A", None)
    
    if alpha is None:
        # 若真的没有，给一个保底 alpha=1（避免全黑/全灰），同时打 WARNING
        print("[WARN] Renderer output missing alpha. Use fallback alpha=1.")
        # 根据 rgb 的形状创建 alpha
        if rgb.dim() == 3 and rgb.shape[0] == 3:  # (3,H,W)
            alpha = torch.ones((1, rgb.shape[1], rgb.shape[2]), device=rgb.device)
        else:
            alpha = torch.ones_like(rgb[..., :1])
    
    # 渲染器返回 (3,H,W) 格式，转换为 (H,W,3)
    if rgb.dim() == 3 and rgb.shape[0] == 3:  # (3,H,W)
        rgb = rgb.permute(1, 2, 0)  # -> (H,W,3)
    
    # alpha 通常是 (1,H,W)，转换为 (H,W,1)
    if alpha.dim() == 3 and alpha.shape[0] == 1:  # (1,H,W)
        alpha = alpha.permute(1, 2, 0)  # -> (H,W,1)
    elif alpha.dim() == 2:  # (H,W)
        alpha = alpha.unsqueeze(-1)  # -> (H,W,1)
    
    # 合成
    bg_expanded = background.view(1, 1, 3).to(device)  # (1,1,3)
    comp = rgb * alpha + bg_expanded * (1.0 - alpha)  # (H,W,3)
    
    return rgb, alpha, comp


@torch.no_grad()
def sanity_render_and_assert(gaussians, camera, pipeline, background, device,
                             render_func, alpha_thr=0.02, rgb_thr=1e-3):
    """
    Sanity 渲染门卫（训练前/每阶段首迭代）
    
    只要一进训练循环，就先渲染一帧做"出光检查"。不过线，立刻中断。
    
    Args:
        gaussians: GaussianModel 实例
        camera: 相机对象
        pipeline: PipelineParams
        background: 背景颜色 tensor
        device: 设备
        render_func: 渲染函数
        alpha_thr: alpha 阈值
        rgb_thr: rgb 阈值
    
    Returns:
        (rgb_max, alpha_mean): RGB 最大值和 alpha 平均值
    
    Raises:
        RuntimeError: 如果渲染失败
    """
    out = render_func(camera, gaussians, pipeline, background, kernel_size=0.1)
    rgb, alpha, comp = unpack_render_dict(out, background=background, device=device)
    
    rgb_max = float(comp.max().item())
    alpha_mean = float(alpha.mean().item())
    
    print(f"[SANITY] rgb_max={rgb_max:.6f}, alpha_mean={alpha_mean:.6f}")
    
    if not (rgb_max > rgb_thr and alpha_mean > alpha_thr):
        raise RuntimeError(
            f"[SANITY] ❌ Sanity failed: rgb_max={rgb_max:.3e}, alpha_mean={alpha_mean:.3e} "
            f"(thr={rgb_thr},{alpha_thr}). Check checkpoint restore / filter_3D / camera near/far."
        )
    
    print(f"[SANITY] ✅ 通过检查")
    return rgb_max, alpha_mean
