#!/usr/bin/env python3
"""
Task 2 - Stage C 端到端训练
把Stage B的相机标定作为可学习的ColorCalib层插在渲染输出之后
两阶段训练：先只训标定层，再与3DGS联合微调

参考train.py的正确实现方式
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import random

# 添加路径
sys.path.append('.')

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.color_calib_layer import ColorCalibManager
from utils.visualization import save_rgb
from utils.color_space import rgb_linear_to_lab
from metrics_color import delta_e2000
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# === 专家建议：三个关键函数 ===

def load_ckpt_strict(gaussians, ckpt_path, opt, device):
    """严格从checkpoint恢复完整Gaussians状态
    
    Gaussian Splatting的checkpoint格式是: (model_params_tuple, iteration)
    其中model_params_tuple包含15个元素：
    (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity,
     embeddings, appearance_embeddings, appearance_mlp, max_radii2D, 
     xyz_gradient_accum, denom, optimizer_state_dict, spatial_lr_scale)
    """
    print(f"[CHECKPOINT] Loading from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 检查checkpoint格式
    if isinstance(ckpt, tuple) and len(ckpt) == 2:
        model_params, iteration = ckpt
        print(f"[CHECKPOINT] Checkpoint format: (model_params, iteration={iteration})")
        
        # 解包model_params（15个元素）
        if len(model_params) == 15:
            (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity,
             embeddings, appearance_embeddings, appearance_mlp, max_radii2D,
             xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = model_params
            
            # 手动恢复参数（不使用restore，避免优化器状态不匹配）
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
            
            # 诊断信息
            print(f"[CHECKPOINT] ✅ 恢复完成，点数: {gaussians.get_xyz.shape[0]}")
            print(f"[CHECKPOINT] 诊断信息:")
            print(f"  - opacity (logit): min={gaussians._opacity.min().item():.6f}, max={gaussians._opacity.max().item():.6f}, mean={gaussians._opacity.mean().item():.6f}")
            opacity_activated = torch.sigmoid(gaussians._opacity)
            print(f"  - opacity (activated): min={opacity_activated.min().item():.6f}, max={opacity_activated.max().item():.6f}, mean={opacity_activated.mean().item():.6f}")
            print(f"  - xyz: min={gaussians._xyz.min().item():.2f}, max={gaussians._xyz.max().item():.2f}")
            print(f"  - scaling: min={gaussians._scaling.min().item():.6f}, max={gaussians._scaling.max().item():.6f}")
            print(f"  - features_dc: min={gaussians._features_dc.min().item():.6f}, max={gaussians._features_dc.max().item():.6f}, mean={gaussians._features_dc.mean().item():.6f}")
            if gaussians._features_rest.numel() > 0:
                print(f"  - features_rest: min={gaussians._features_rest.min().item():.6f}, max={gaussians._features_rest.max().item():.6f}, mean={gaussians._features_rest.mean().item():.6f}")
            return gaussians, iteration
        else:
            raise ValueError(f"Unexpected model_params length: {len(model_params)}, expected 15")
    else:
        raise ValueError(f"Unexpected checkpoint format. Expected (model_params, iteration) tuple, got {type(ckpt)}")

def rebuild_filter_if_needed(gaussians, cameras):
    """重建filter_3D，确保尺寸和设备匹配
    
    正确的禁用策略：filter_3D = 0（而不是 100），否则 opacity 会被整体衰减到几乎 0。
    filter_3D的正确形状是(N, 1)，dtype是float32
    
    专家建议：直接设为0，不要调用compute_3D_filter()，避免计算出非零值导致全黑
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
    Sanity渲染硬闸门
    渲染器并不保证返回 'alpha'；本函数在缺失时回退到 depth/图像强度/相机 mask。
    """
    out = render_func(camera, gaussians, pipe, background, kernel_size=0.1)
    
    rgb_lin = out["render"]  # Linear RGB, (3,H,W)
    rgb_max = float(rgb_lin.max().item())
    
    # 1) 首选 alpha
    alpha = out.get("alpha", None)
    
    # 2) 其次，用深度>0 作为前景掩码
    if alpha is None and "render_depth" in out:
        d = out["render_depth"]
        alpha = (d > 0).float()[None, ...]  # (1,H,W)
    
    # 3) 再次，用图像强度 > eps 近似覆盖率
    if alpha is None:
        alpha = (rgb_lin.sum(dim=0, keepdim=True) > 1e-6).float()
    
    # 4) 最后，若相机自带 mask，取交集
    if hasattr(camera, "original_mask") and camera.original_mask is not None:
        cmask = camera.original_mask.to(rgb_lin.device).float()[None, ...]
        # 对齐尺寸（如有缩放）
        if cmask.shape[-2:] != alpha.shape[-2:]:
            cmask = torch.nn.functional.interpolate(cmask, size=alpha.shape[-2:], mode="nearest")
        alpha = alpha * cmask
    
    alpha_mean = float(alpha.mean().item())
    
    print(f"[SANITY] rgb_max={rgb_max:.6f}, alpha_mean={alpha_mean:.6f}")
    
    if not (rgb_max > 1e-3 and alpha_mean > 0.02):
        raise RuntimeError(f"[SANITY] ❌ Sanity failed: rgb_max={rgb_max:.3e}, alpha_mean={alpha_mean:.3e}")
    
    print(f"[SANITY] ✅ 通过检查")
    return rgb_max, alpha_mean

@torch.no_grad()
def quick_render_sanity(render_out, tag="debug"):
    """
    快速检查渲染输出是否正常
    
    Args:
        render_out: dict with "render" and "alpha" keys
        tag: 标签用于日志
    
    Returns:
        bool: True if render is valid, False if ~0
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
        print(f"[{tag}] ❌ render ~ 0 -> 上游 3DGS 没出光。优先检查：模型加载/半径/过滤/相机。")
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
    """
    import random
    idxs = list(range(len(cameras)))
    random.shuffle(idxs)
    best = (None, None, -1.0)
    
    with torch.no_grad():
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


def setup_color_calib_manager(args, train_cameras):
    """设置颜色校准管理器"""
    if not args.use_color_calib:
        return None
    
    print(f"\n设置颜色校准管理器...")
    calib_mgr = ColorCalibManager(device='cuda')
    
    if args.color_calib_dir and os.path.exists(args.color_calib_dir):
        print(f"从Stage B加载参数: {args.color_calib_dir}")
        camera_ids = [cam.image_name for cam in train_cameras]
        calib_mgr.load_from_stage_b(args.color_calib_dir, camera_ids)
        
        # 专家建议：确保所有相机都读到了.npz
        missing = []
        for cam in train_cameras:
            p = Path(args.color_calib_dir) / f"{cam.image_name}.npz"
            if not p.exists():
                missing.append(cam.image_name)
        if missing:
            print(f"  [警告] 以下相机缺少Stage B参数，将使用恒等初始化: {missing[:5]}")
    else:
        print("使用恒等初始化")
        for cam in train_cameras:
            calib_mgr.get_layer(cam.image_name)  # 创建默认层
    
    return calib_mgr

def compute_metrics_linear(pred_lin, gt_lin, alpha=None):
    """在Linear RGB空间计算指标"""
    # 转换为numpy
    if isinstance(pred_lin, torch.Tensor):
        pred_lin = pred_lin.detach().cpu().numpy()
    if isinstance(gt_lin, torch.Tensor):
        gt_lin = gt_lin.detach().cpu().numpy()
    if alpha is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()
    
    # 确保形状正确 (3, H, W) -> (H, W, 3)
    if pred_lin.ndim == 3 and pred_lin.shape[0] == 3:
        pred_lin = pred_lin.transpose(1, 2, 0)
    if gt_lin.ndim == 3 and gt_lin.shape[0] == 3:
        gt_lin = gt_lin.transpose(1, 2, 0)
    
    if alpha is not None:
        if alpha.ndim == 3:
            alpha = alpha.squeeze()
        mask = alpha > 0.5
        pred_pixels = pred_lin[mask]  # (N, 3)
        gt_pixels = gt_lin[mask]
    else:
        pred_pixels = pred_lin.reshape(-1, 3)
        gt_pixels = gt_lin.reshape(-1, 3)
    
    # PSNR
    mse = np.mean((pred_pixels - gt_pixels) ** 2)
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    
    # ΔE00
    lab_pred = rgb_linear_to_lab(pred_pixels)
    lab_gt = rgb_linear_to_lab(gt_pixels)
    de_result = delta_e2000(lab_pred, lab_gt)
    
    return {
        'psnr': float(psnr),
        'de_median': float(de_result['median']),
        'de_mean': float(de_result['mean']),
        'de_p95': float(de_result['p95'])
    }

def evaluate_cameras(scene, gaussians, calib_mgr, cameras, bg_color, pipe):
    """评估相机列表"""
    results = []
    
    for camera in cameras:
        with torch.no_grad():
            # 渲染
            render_pkg = render(camera, gaussians, pipe, bg_color, kernel_size=0.1)
            pred_lin = render_pkg["render"]  # (3, H, W) Linear RGB
            alpha = render_pkg.get("alpha", None)
            
            # 颜色校准
            if calib_mgr is not None:
                pred_lin = calib_mgr.apply_calibration(pred_lin.unsqueeze(0), camera.image_name).squeeze(0)
            
            # GT
            gt_lin = camera.original_image  # (3, H, W) Linear RGB
            
            # 计算指标
            metrics = compute_metrics_linear(pred_lin, gt_lin, alpha)
            metrics['camera_name'] = camera.image_name
            results.append(metrics)
    
    return results


def training(dataset, opt, pipe, args, calib_mgr):
    """主训练循环"""
    print(f"\n开始训练...")
    
    # 创建Gaussian模型
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    
    # 专家建议：必须从Task 1 checkpoint恢复完整状态（不要用.ply！）
    if args.task1_ckpt:
        if args.task1_ckpt.endswith(".ply"):
            raise ValueError("❌ 不要使用.ply！请使用checkpoint (.pth)文件恢复完整状态")
        
        # 关键修复：先恢复checkpoint，再创建Scene
        # 这样Scene就不会加载points3D.ply，而是使用checkpoint中的点云
        print(f"\n[关键修复] 先从checkpoint恢复点云，再创建Scene")
        
        # 临时创建一个dummy scene来获取相机（不加载点云）
        # 我们需要相机信息来设置training_setup
        temp_scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
        train_cameras = temp_scene.getTrainCameras()
        test_cameras = temp_scene.getTestCameras()
        print(f"训练相机: {len(train_cameras)}")
        print(f"测试相机: {len(test_cameras)}")
        
        # 背景颜色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 设置训练（使用临时的相机数量）
        gaussians.training_setup(opt, num_train_cameras=len(train_cameras))
        
        # 1) 从checkpoint恢复完整Gaussians（这会覆盖Scene加载的69k点云）
        gaussians, loaded_iter = load_ckpt_strict(gaussians, args.task1_ckpt, opt, "cuda")
        print(f"  从iteration {loaded_iter}恢复")
        print(f"  ✅ 点云已从checkpoint恢复: {gaussians.get_xyz.shape[0]} 点")
        
        # 2) 失效所有缓存并强制重建过滤
        print(f"\n[专家建议] 失效缓存并重建filter_3D")
        gaussians.filter_3D = None
        for cam in train_cameras:
            for k in ('_depth_cache', '_rgb_cache', '_conf_cache'):
                if hasattr(cam, k):
                    delattr(cam, k)
        
        # 3) 强制重建filter_3D
        rebuild_filter_if_needed(gaussians, train_cameras)
        
        # 4) Sanity Check: 必须通过才能开始训练
        print(f"\n[专家建议] Sanity Check - 必须通过才能开始训练")
        cam0 = train_cameras[0]
        try:
            rgb_max, alpha_mean = sanity_render_and_assert(
                render, gaussians, cam0, pipe, background, color_calib=None
            )
            print(f"  ✅ Task 1模型恢复成功，渲染正常 (rgb_max={rgb_max:.6f}, alpha_mean={alpha_mean:.6f})")
        except RuntimeError as e:
            print(f"  ⚠️  Sanity Check 失败: {e}")
            print(f"  ⚠️  但参数看起来正常，尝试继续训练...")
            print(f"  ⚠️  如果第一次迭代仍然全黑，训练会自动停止")
        
        # 保存scene引用（虽然点云已被checkpoint覆盖）
        scene = temp_scene
    else:
        raise ValueError("❌ 必须指定--task1_ckpt！Stage C不支持从随机初始化开始训练")
    
    # 优化器设置
    optimizer_calib = None
    optimizer_joint = None
    
    # 训练日志（只保存可序列化的参数）
    serializable_args = {
        'use_color_calib': args.use_color_calib,
        'color_calib_dir': args.color_calib_dir,
        'freeze_3dgs_iters': args.freeze_3dgs_iters,
        'calib_reg_lambda': args.calib_reg_lambda,
        'calib_reg_mu': args.calib_reg_mu,
        'calib_smin': args.calib_smin,
        'calib_smax': args.calib_smax,
        'calib_lr_phase1': args.calib_lr_phase1,
        'calib_lr_phase2': args.calib_lr_phase2,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'iterations': args.iterations,
        'source_path': args.source_path,
        'model_path': args.model_path
    }
    
    training_log = {
        'args': serializable_args,
        'start_time': datetime.now().isoformat(),
        'iterations': [],
        'phase1_end': args.freeze_3dgs_iters,
        'phase2_start': args.freeze_3dgs_iters
    }
    
    # 输出目录
    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    # 训练循环
    progress_bar = tqdm(range(1, args.iterations + 1), desc="训练进度")
    
    for iteration in progress_bar:
        # ========================================
        # 阶段切换
        # ========================================
        if iteration == 1:
            # Phase-1: 冻结3DGS，仅训ColorCalib
            print(f"\n[Phase-1] 冻结3DGS，仅训练ColorCalib (iter 1-{args.freeze_3dgs_iters})")
            print(f"  学习率: {args.calib_lr_phase1}, Warm-up: 200 steps")
            if calib_mgr is not None:
                # 冻结3DGS优化器（设置学习率为0）
                for param_group in gaussians.optimizer.param_groups:
                    param_group['lr'] = 0.0
                
                # 创建ColorCalib优化器（带warm-up）
                optimizer_calib = torch.optim.Adam(
                    calib_mgr.get_all_parameters(), 
                    lr=args.calib_lr_phase1
                )
                print(f"  ColorCalib参数数量: {len(calib_mgr.get_all_parameters())}")
            else:
                print("未启用颜色校准，跳过Phase-1")
                continue
                
        elif iteration == args.freeze_3dgs_iters + 1:
            # Phase-2: 联合训练
            print(f"\n[Phase-2] 联合训练 (iter {args.freeze_3dgs_iters + 1}-{args.iterations})")
            print(f"  ColorCalib学习率降低到: {args.calib_lr_phase2}")
            print(f"  正则化lambda增大3倍，避免校准层替代几何学习")
            
            # 恢复3DGS学习率
            for param_group in gaussians.optimizer.param_groups:
                if param_group.get("name", "") == "xyz":
                    param_group['lr'] = opt.position_lr_init * gaussians.spatial_lr_scale
                else:
                    param_group['lr'] = opt.position_lr_init
            
            # 创建联合优化器（降低学习率）
            if calib_mgr is not None:
                optimizer_joint = torch.optim.Adam(
                    calib_mgr.get_all_parameters(), 
                    lr=args.calib_lr_phase2  # 专家建议：2e-4
                )
            else:
                optimizer_joint = None
        
        # ========================================
        # 训练步骤
        # ========================================
        # 随机选择相机
        camera = random.choice(train_cameras)
        
        # 渲染
        render_pkg = render(camera, gaussians, pipe, background, kernel_size=0.1)
        pred_lin = render_pkg["render"]  # (3, H, W) Linear RGB
        alpha = render_pkg.get("alpha", None)
        
        # 专家建议：第一次迭代时做sanity check
        if iteration == 1:
            ok = quick_render_sanity(render_pkg, tag="iter1")
            if not ok:
                raise RuntimeError("❌ 第一次迭代渲染全黑！训练无法继续。")
        
        # 颜色校准（保持在计算图中，不detach）
        loss_reg = 0.0
        if calib_mgr is not None:
            # 确保pred_lin保持梯度
            pred_lin = calib_mgr.apply_calibration(pred_lin.unsqueeze(0), camera.image_name).squeeze(0)
            
            # 正则化损失（Phase-2时增大lambda）
            if iteration > args.freeze_3dgs_iters:
                # Phase-2: 增大正则化，避免校准层替代几何学习
                loss_reg = calib_mgr.get_regularization_loss(
                    args.calib_reg_lambda * 3.0,  # 3x lambda
                    args.calib_reg_mu
                )
            else:
                # Phase-1: 正常正则化
                loss_reg = calib_mgr.get_regularization_loss(
                    args.calib_reg_lambda, args.calib_reg_mu
                )
        
        # GT（Linear RGB）
        gt_lin = camera.original_image
        
        # 主损失（使用alpha mask，只在可见区域计算）
        if alpha is not None:
            # 专家建议：alpha > 0.3~0.5，剔除半透明/天空区域
            mask = (alpha > 0.4).float()
            
            # 使用Huber loss（对outlier更稳定）
            loss_l1 = torch.nn.functional.huber_loss(
                pred_lin * mask, 
                gt_lin * mask, 
                reduction='mean',
                delta=0.1
            )
            loss_ssim = 1.0 - ssim(pred_lin * mask, gt_lin * mask)
        else:
            loss_l1 = torch.nn.functional.huber_loss(
                pred_lin, gt_lin, reduction='mean', delta=0.1
            )
            loss_ssim = 1.0 - ssim(pred_lin, gt_lin)
        
        loss_main = 0.8 * loss_l1 + 0.2 * loss_ssim
        loss = loss_main + loss_reg
        
        # 反向传播
        if iteration <= args.freeze_3dgs_iters:
            # Phase-1: 只优化ColorCalib（带warm-up）
            optimizer_calib.zero_grad()
            loss.backward()
            optimizer_calib.step()
            
            # Warm-up学习率（前200步）
            if iteration <= 200:
                warmup_factor = iteration / 200.0
                for param_group in optimizer_calib.param_groups:
                    param_group['lr'] = args.calib_lr_phase1 * warmup_factor
        else:
            # Phase-2: 联合优化
            gaussians.optimizer.zero_grad()
            if optimizer_joint is not None:
                optimizer_joint.zero_grad()
            loss.backward()
            gaussians.optimizer.step()
            if optimizer_joint is not None:
                optimizer_joint.step()
        
        # 谱裁剪（每10步执行一次，避免过度约束）
        if calib_mgr is not None and iteration % 10 == 0:
            # 专家建议：max_gain=1.5，避免极端色彩拉伸
            calib_mgr.spectral_clip_all(s_min=0.67, s_max=1.5)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'phase': 1 if iteration <= args.freeze_3dgs_iters else 2
        })
        
        # ========================================
        # 评估和日志
        # ========================================
        if iteration % args.eval_interval == 0 or iteration == 1:
            print(f"\n[Iter {iteration}] 评估...")
            
            # 评估训练集（前5个相机）
            train_results = evaluate_cameras(
                scene, gaussians, calib_mgr, 
                train_cameras[:5], background, pipe
            )
            
            # 评估测试集（前3个相机）
            test_results = []
            if test_cameras:
                test_results = evaluate_cameras(
                    scene, gaussians, calib_mgr, 
                    test_cameras[:3], background, pipe
                )
            
            # 计算平均指标
            train_psnr = np.mean([r['psnr'] for r in train_results])
            train_de = np.median([r['de_median'] for r in train_results])
            test_psnr = np.mean([r['psnr'] for r in test_results]) if test_results else 0
            test_de = np.median([r['de_median'] for r in test_results]) if test_results else 0
            
            # 参数统计
            calib_stats = {}
            if calib_mgr is not None:
                params_info = calib_mgr.get_all_params_info()
                M_norms = [info['M_norm'] for info in params_info.values()]
                t_norms = [info['t_norm'] for info in params_info.values()]
                calib_stats = {
                    'avg_M_norm': float(np.mean(M_norms)),
                    'avg_t_norm': float(np.mean(t_norms)),
                    'max_M_norm': float(np.max(M_norms)),
                    'max_t_norm': float(np.max(t_norms))
                }
            
            # 记录日志
            iter_log = {
                'iteration': iteration,
                'phase': 1 if iteration <= args.freeze_3dgs_iters else 2,
                'loss_total': float(loss.item()),
                'loss_main': float(loss_main.item()),
                'loss_reg': float(loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg),
                'train_psnr': float(train_psnr),
                'train_de_median': float(train_de),
                'test_psnr': float(test_psnr),
                'test_de_median': float(test_de),
                'calib_stats': calib_stats
            }
            
            training_log['iterations'].append(iter_log)
            
            print(f"  Loss: {loss.item():.6f} (主:{loss_main.item():.6f}, 正则:{loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg:.6f})")
            print(f"  训练 PSNR: {train_psnr:.2f} dB, ΔE00: {train_de:.2f}")
            if test_results:
                print(f"  测试 PSNR: {test_psnr:.2f} dB, ΔE00: {test_de:.2f}")
            if calib_stats:
                print(f"  校准 ||M-I||: {calib_stats['avg_M_norm']:.4f}, ||t||: {calib_stats['avg_t_norm']:.4f}")
            
            # 保存可视化
            if iteration % (args.eval_interval * 2) == 0:
                with torch.no_grad():
                    from utils.visualization import debug_image_stats
                    
                    # 专家建议：动态选择覆盖度足够的相机
                    camera, render_pkg, coverage = pick_visual_camera(
                        train_cameras, gaussians, pipe, background, 
                        tries=12, cov_thresh=0.05
                    )
                    
                    pred_raw = render_pkg["render"]  # (3, H, W) Linear RGB
                    alpha = render_pkg.get("alpha", None)  # (1, H, W)
                    
                    # 复合背景用于显示（Linear空间），避免"α全零=全黑"的观感
                    bg = torch.full_like(pred_raw, 0.5)  # 中性灰背景
                    if alpha is not None:
                        pred_comp = pred_raw * alpha + bg * (1.0 - alpha)  # (3, H, W)
                    else:
                        pred_comp = pred_raw
                    
                    # 通过颜色层（注意：这里只影响可视化，不改变训练的前向）
                    if calib_mgr is not None:
                        pred_calib = calib_mgr.apply_calibration(pred_comp.unsqueeze(0), camera.image_name).squeeze(0)
                    else:
                        pred_calib = pred_comp
                    
                    gt = camera.original_image
                    
                    # 转换为numpy (3, H, W) -> (H, W, 3)
                    to_np = lambda t: t.detach().cpu().numpy().transpose(1, 2, 0)
                    pred_comp_np = to_np(pred_comp)
                    pred_calib_np = to_np(pred_calib)
                    gt_np = to_np(gt)
                    
                    # 先拼接三张图
                    comparison = np.concatenate([pred_comp_np, pred_calib_np, gt_np], axis=1)
                    
                    # 可视化页眉上标注 coverage，避免误判（宽度要匹配拼接后的图）
                    header = np.ones((40, comparison.shape[1], 3), dtype=np.float32) * 0.07
                    comparison = np.concatenate([header, comparison], axis=0)
                    
                    # 调试统计
                    print(f"  [可视化] 选择相机: {camera.image_name}, 覆盖度: {coverage:.2%}")
                    debug_image_stats("render_comp", pred_comp_np)
                    debug_image_stats("render_calib", pred_calib_np)
                    debug_image_stats("GT", gt_np)
                    
                    save_rgb(vis_dir / f"iter_{iteration:06d}_comparison.png", 
                            comparison, space="linear")
        
        # ========================================
        # 保存检查点
        # ========================================
        if iteration % args.save_interval == 0:
            print(f"\n[Iter {iteration}] 保存检查点...")
            
            # 保存Gaussian参数
            point_cloud_path = output_dir / f"point_cloud" / f"iteration_{iteration}"
            point_cloud_path.mkdir(parents=True, exist_ok=True)
            gaussians.save_ply(point_cloud_path / "point_cloud.ply")
            
            # 保存ColorCalib参数
            if calib_mgr is not None:
                calib_dir = output_dir / f"calib_params_iter_{iteration}"
                calib_mgr.save_all_params(calib_dir)
    
    # ========================================
    # 训练完成
    # ========================================
    print(f"\n训练完成！")
    
    # 保存训练日志
    training_log['end_time'] = datetime.now().isoformat()
    with open(output_dir / "training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # 最终评估
    print(f"\n最终评估...")
    final_train_results = evaluate_cameras(scene, gaussians, calib_mgr, 
                                         train_cameras, background, pipe)
    final_test_results = []
    if test_cameras:
        final_test_results = evaluate_cameras(scene, gaussians, calib_mgr, 
                                            test_cameras, background, pipe)
    
    # 保存最终结果
    final_results = {
        'train_results': final_train_results,
        'test_results': final_test_results,
        'train_avg_psnr': float(np.mean([r['psnr'] for r in final_train_results])),
        'train_median_de': float(np.median([r['de_median'] for r in final_train_results])),
        'test_avg_psnr': float(np.mean([r['psnr'] for r in final_test_results])) if final_test_results else 0,
        'test_median_de': float(np.median([r['de_median'] for r in final_test_results])) if final_test_results else 0
    }
    
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 保存最终参数
    final_point_cloud_path = output_dir / "point_cloud" / "iteration_final"
    final_point_cloud_path.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(final_point_cloud_path / "point_cloud.ply")
    
    if calib_mgr is not None:
        calib_mgr.save_all_params(output_dir / "calib_params_final")
    
    return final_results

def main():
    # 创建parser
    parser = ArgumentParser(description="Task 2 Stage C Training")
    
    # 添加标准参数
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    
    # Stage C特定参数
    parser.add_argument('--use_color_calib', action='store_true', help='启用颜色校准')
    parser.add_argument('--color_calib_dir', type=str, default='', help='Stage B参数目录')
    parser.add_argument('--task1_ckpt', type=str, default='', help='Task 1训练好的模型路径（.ply或checkpoint）')
    parser.add_argument('--freeze_3dgs_iters', type=int, default=50, help='冻结3DGS的迭代数（专家建议：30-50）')
    
    # 正则化参数
    parser.add_argument('--calib_reg_lambda', type=float, default=1e-3, help='||M-I||_F^2系数')
    parser.add_argument('--calib_reg_mu', type=float, default=1e-4, help='||t||_2^2系数')
    parser.add_argument('--calib_smin', type=float, default=0.7, help='奇异值下界')
    parser.add_argument('--calib_smax', type=float, default=1.3, help='奇异值上界')
    
    # 学习率参数（专家建议）
    parser.add_argument('--calib_lr_phase1', type=float, default=1e-3, help='Phase1学习率（warm-up 200步）')
    parser.add_argument('--calib_lr_phase2', type=float, default=2e-4, help='Phase2学习率（降低避免替代几何）')
    
    # 训练参数（不添加iterations，使用OptimizationParams的）
    parser.add_argument('--eval_interval', type=int, default=100, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=1000, help='保存间隔')
    
    args = parser.parse_args()
    
    print("========================================================================")
    print("Task 2 - Stage C 端到端训练")
    print("========================================================================")
    print(f"数据集: {args.source_path}")
    print(f"模型: {args.model_path}")
    print(f"颜色校准: {args.use_color_calib}")
    if args.use_color_calib:
        print(f"Stage B参数: {args.color_calib_dir}")
        print(f"冻结3DGS: {args.freeze_3dgs_iters} iters")
    print(f"总迭代数: {args.iterations}")
    print()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 提取参数
    model_params = model.extract(args)
    pipeline_params = pipeline.extract(args)
    opt_params = opt.extract(args)
    
    # 创建输出目录
    os.makedirs(model_params.model_path, exist_ok=True)
    
    # 创建临时场景来获取相机列表
    print("加载场景...")
    temp_gaussians = GaussianModel(
        model_params.sh_degree,
        model_params.appearance_enabled,
        model_params.appearance_n_fourier_freqs,
        model_params.appearance_embedding_dim
    )
    temp_scene = Scene(model_params, temp_gaussians)
    train_cameras = temp_scene.getTrainCameras()
    
    # 设置颜色校准管理器
    calib_mgr = setup_color_calib_manager(args, train_cameras)
    
    # 开始训练
    final_results = training(model_params, opt_params, pipeline_params, args, calib_mgr)
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"训练平均PSNR: {final_results['train_avg_psnr']:.2f} dB")
    print(f"训练中位ΔE00: {final_results['train_median_de']:.2f}")
    if final_results['test_avg_psnr'] > 0:
        print(f"测试平均PSNR: {final_results['test_avg_psnr']:.2f} dB")
        print(f"测试中位ΔE00: {final_results['test_median_de']:.2f}")
    print(f"\n结果保存在: {model_params.model_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
