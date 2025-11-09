#!/usr/bin/env python3
"""
Checkpoint 和渲染工具函数
专家建议的三个关键函数，可复用于所有训练脚本
"""
import torch
import numpy as np


def load_ckpt_strict(gaussians, ckpt_path, opt, device):
    """
    从 checkpoint 严格恢复完整 Gaussians 状态
    
    Gaussian Splatting 的 checkpoint 格式是: (model_params_tuple, iteration)
    其中 model_params_tuple 包含15个元素：
    (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity,
     embeddings, appearance_embeddings, appearance_mlp, max_radii2D, 
     xyz_gradient_accum, denom, optimizer_state_dict, spatial_lr_scale)
    
    Args:
        gaussians: GaussianModel 实例
        ckpt_path: checkpoint 文件路径 (.pth)
        opt: OptimizationParams
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        (gaussians, iteration): 恢复后的模型和迭代数
    
    Raises:
        ValueError: 如果 checkpoint 格式不正确
    
    Example:
        >>> gaussians, iteration = load_ckpt_strict(
        ...     gaussians, 'output/chkpnt5000.pth', opt, 'cuda'
        ... )
        [CHECKPOINT] Loading from output/chkpnt5000.pth
        [CHECKPOINT] ✅ 恢复完成，点数: 123456
    """
    print(f"[CHECKPOINT] Loading from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 检查 checkpoint 格式
    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_params, iteration = ckpt
        print(f"[CHECKPOINT] Checkpoint format: (model_params, iteration={iteration})")
        
        # 解包 model_params（15个元素）
        if len(model_params) == 15:
            (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity,
             embeddings, appearance_embeddings, appearance_mlp, max_radii2D,
             xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = model_params
            
            # 手动恢复参数（不使用 restore，避免优化器状态不匹配）
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
            
            # 尝试加载优化器状态（如果失败就跳过）
            try:
                gaussians.optimizer.load_state_dict(opt_dict)
                print(f"[CHECKPOINT] ✅ 优化器状态已加载")
            except Exception as e:
                print(f"[CHECKPOINT] ⚠️  优化器状态加载失败（参数组不匹配），将使用新的优化器: {e}")
                # 不加载优化器状态，使用新初始化的优化器
            
            print(f"[CHECKPOINT] ✅ 恢复完成，点数: {gaussians.get_xyz.shape[0]}")
            return gaussians, iteration
        else:
            raise ValueError(f"Unexpected model_params length: {len(model_params)}, expected 15")
    else:
        raise ValueError(
            f"Unexpected checkpoint format. "
            f"Expected (model_params, iteration) tuple, got {type(ckpt)}"
        )


def rebuild_filter_if_needed(gaussians, cameras):
    """
    重建 filter_3D，确保尺寸和设备匹配
    
    正确的禁用策略：filter_3D = 0（而不是 100），否则 opacity 会被整体衰减到几乎 0。
    filter_3D 的正确形状是 (N, 1)，dtype 是 float32
    
    物理意义：
    - filter_3D = 0 → det2 ≈ det1 → coef ≈ 1 → opacity 不变
    - filter_3D = 100 → det2 >> det1 → coef ≈ 0 → opacity ≈ 0（全黑）
    
    专家建议：直接设为0，不要调用compute_3D_filter()，避免计算出非零值导致全黑
    
    Args:
        gaussians: GaussianModel 实例
        cameras: 相机列表
    
    Example:
        >>> rebuild_filter_if_needed(gaussians, train_cameras)
        [FILTER] 设置filter_3D为0: N=123456, device=cuda:0
        [FILTER] ✅ filter_3D已设置，mean=0.0000, max=0.0000
    """
    N = gaussians.get_xyz.shape[0]
    dev = gaussians.get_xyz.device
    
    # 直接设置为0（禁用过滤）
    print(f"[FILTER] 设置filter_3D为0: N={N}, device={dev}")
    gaussians.filter_3D = torch.zeros(N, 1, dtype=torch.float32, device=dev)
    
    filter_mean = float(gaussians.filter_3D.mean().item())
    filter_max = float(gaussians.filter_3D.max().item())
    print(f"[FILTER] ✅ filter_3D已设置，mean={filter_mean:.4f}, max={filter_max:.4f}")


@torch.no_grad()
def sanity_render_and_assert(render_func, gaussians, camera, pipe, background, color_calib=None):
    """
    Sanity 渲染硬闸门
    
    渲染器并不保证返回 'alpha'；本函数在缺失时回退到 depth/图像强度/相机 mask。
    
    回退策略：
    1. 首选 alpha
    2. 其次 accumulation（别名）
    3. 再次 depth > 0（前景掩码）
    4. 最后 rgb 强度 > eps（近似覆盖率）
    5. 与相机 mask 取交集（如有）
    
    Args:
        render_func: 渲染函数（如 gaussian_renderer.render）
        gaussians: GaussianModel 实例
        camera: 相机对象
        pipe: PipelineParams
        background: 背景颜色 tensor
        color_calib: 颜色校准层（可选）
    
    Returns:
        (rgb_max, alpha_mean): RGB 最大值和 alpha 平均值
    
    Raises:
        RuntimeError: 如果渲染失败（rgb_max < 1e-3 或 alpha_mean < 0.02）
    
    Example:
        >>> rgb_max, alpha_mean = sanity_render_and_assert(
        ...     render, gaussians, camera, pipe, background
        ... )
        [SANITY] rgb_max=0.987654, alpha_mean=0.678901
        [SANITY] ✅ 通过检查
    """
    out = render_func(camera, gaussians, pipe, background, kernel_size=0.1)
    
    rgb_lin = out.get("render", None)
    if rgb_lin is None:
        raise RuntimeError("[SANITY] ❌ 渲染器未返回 'render' 键")
    
    # 1) 首选 alpha/accumulation
    alpha = out.get("alpha", None)
    if alpha is None:
        alpha = out.get("accumulation", None)  # 兼容别名
    
    # 2) 其次，用深度>0 作为前景掩码
    if alpha is None and "render_depth" in out:
        d = out["render_depth"]
        alpha = (d > 0).float()[None, ...]  # (1,H,W)
    
    # 3) 再次，用图像强度 > eps 近似覆盖率
    if alpha is None:
        alpha = (rgb_lin.sum(dim=0, keepdim=True) > 1e-6).float()
    
    # 4) 与相机 mask 取交集（如有）
    if hasattr(camera, "original_mask") and camera.original_mask is not None:
        cmask = camera.original_mask.to(rgb_lin.device).float()[None, ...]
        if cmask.shape[-2:] != alpha.shape[-2:]:
            cmask = torch.nn.functional.interpolate(
                cmask, size=alpha.shape[-2:], mode="nearest"
            )
        alpha = alpha * cmask
    
    rgb_max = float(rgb_lin.max().item())
    alpha_mean = float(alpha.mean().item())
    
    print(f"[SANITY] rgb_max={rgb_max:.6f}, alpha_mean={alpha_mean:.6f}")
    
    if not (rgb_max > 1e-3 and alpha_mean > 2e-2):
        raise RuntimeError(
            f"[SANITY] ❌ Sanity failed: rgb_max={rgb_max:.3e}, alpha_mean={alpha_mean:.3e}"
        )
    
    print(f"[SANITY] ✅ 通过检查")
    return rgb_max, alpha_mean


@torch.no_grad()
def quick_render_sanity(render_out, tag="debug"):
    """
    快速检查渲染输出是否正常（用于调试）
    
    Args:
        render_out: dict with "render" and "alpha" keys
        tag: 标签用于日志
    
    Returns:
        bool: True if render is valid, False if ~0
    
    Example:
        >>> ok = quick_render_sanity(render_pkg, tag="iter1")
        [iter1] render stats: min=0.00123, max=0.98765, mean=0.45678, coverage=67.89%
        [iter1] alpha stats: min=0.00000, max=1.00000, mean=0.67890
    """
    rgb = render_out.get("render", None)
    alpha = render_out.get("alpha", None)
    
    if rgb is None:
        print(f"[{tag}] ERROR: no render in output")
        return False
    
    if rgb.ndim == 4:  # [B,3,H,W]
        rgb = rgb[0]
    rgb_np = rgb.detach().float().cpu().clamp(0, 1).numpy()
    
    H, W = rgb_np.shape[-2:]
    cov = (rgb_np.sum(axis=0) > 1e-6).mean()
    print(f"[{tag}] render stats: min={rgb_np.min():.5f}, max={rgb_np.max():.5f}, "
          f"mean={rgb_np.mean():.5f}, coverage={(cov*100):.2f}% (sum>1e-6)")
    
    if alpha is not None:
        if alpha.ndim == 3:  # [1,H,W]
            a = alpha[0] if alpha.shape[0] == 1 else alpha
        elif alpha.ndim == 4:  # [B,1,H,W]
            a = alpha[0,0]
        else:
            a = alpha
        a = a.detach().float().cpu().numpy()
        print(f"[{tag}] alpha stats: min={a.min():.5f}, max={a.max():.5f}, "
              f"mean={a.mean():.5f}")
    
    # 判定是否"没光"
    if rgb_np.max() < 1e-4:
        print(f"[{tag}] ❌ render ~ 0 -> 上游 3DGS 没出光。"
              f"优先检查：模型加载/半径/过滤/相机。")
        return False
    return True


def pick_visual_camera(cameras, gaussians, pipe, background, tries=12, cov_thresh=0.05):
    """
    选一个覆盖度足够的相机来做可视化
    
    Args:
        cameras: 相机列表
        gaussians: GaussianModel
        pipe: PipelineParams
        background: 背景颜色
        tries: 尝试次数
        cov_thresh: 覆盖度阈值
    
    Returns:
        (camera, render_pkg, coverage): 最佳相机、渲染结果、覆盖度
    
    Example:
        >>> camera, render_pkg, coverage = pick_visual_camera(
        ...     train_cameras, gaussians, pipe, background
        ... )
        >>> print(f"选择相机: {camera.image_name}, 覆盖度: {coverage:.2%}")
    """
    import random
    from gaussian_renderer import render
    
    idxs = list(range(len(cameras)))
    random.shuffle(idxs)
    best = (None, None, -1.0)
    
    for i in idxs[:tries]:
        cam = cameras[i]
        pkg = render(cam, gaussians, pipe, background, kernel_size=0.1)
        alpha = pkg.get("alpha", None)
        cov = float((alpha > 0.01).float().mean().item()) if alpha is not None else 1.0
        if cov > best[2]:
            best = (cam, pkg, cov)
        if cov >= cov_thresh:
            return cam, pkg, cov
    return best  # 退化返回覆盖度最高的那一帧
