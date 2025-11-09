#!/usr/bin/env python3
"""
Task 2 Stage C - 最小修改版本
基于 Task 1 的训练逻辑，只插入 ColorCalib 层

核心原则：
1. 100% 复用 Task 1 的 Scene/Camera/Renderer 逻辑
2. 只在渲染输出后插入 ColorCalib
3. 不修改任何归一化/裁剪/渲染参数
4. 保持 Linear RGB 空间
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

# 添加路径
sys.path.append('.')

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.color_calib_layer import ColorCalibManager
from metrics_color import delta_e2000
from utils.color_space import rgb_linear_to_lab
from utils.visualization import save_rgb, linear_to_srgb_safe
from argparse import ArgumentParser, Namespace

def sanity_check(image, render_pkg, tag=""):
    """
    Sanity Check 硬闸门
    确保渲染输出在合理范围内
    """
    with torch.no_grad():
        rgb_max = float(image.max().item())
        rgb_min = float(image.min().item())
        rgb_mean = float(image.mean().item())
        
        alpha = render_pkg.get("alpha", None)
        if alpha is None:
            alpha = render_pkg.get("accumulation", None)
        alpha_mean = float(alpha.mean().item()) if alpha is not None else -1.0
        
        print(f"[Sanity{tag}] rgb: min={rgb_min:.4f}, max={rgb_max:.4f}, mean={rgb_mean:.4f}, alpha_mean={alpha_mean:.4f}")
        
        # 硬闸门检查
        if rgb_max < 1e-3:
            raise RuntimeError(f"❌ rgb_max={rgb_max:.2e} 太小（可能黑屏/坐标口径错）")
        if rgb_max > 4.0:
            raise RuntimeError(f"❌ rgb_max={rgb_max:.2e} 太大（可能HDR未归一/near-far口径错）")
        if alpha_mean > 0 and not (0.05 <= alpha_mean <= 0.99):
            print(f"⚠️  alpha_mean={alpha_mean:.3f} 异常（不透明度口径）")
        
        return rgb_max, alpha_mean

def training(dataset, opt, pipe, args):
    """
    训练主函数
    """
    print("=" * 80)
    print("Task 2 Stage C - 最小修改版本")
    print("=" * 80)
    print(f"数据集: {dataset.source_path}")
    print(f"模型: {dataset.model_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"颜色校准: {args.use_color_calib}")
    if args.use_color_calib:
        print(f"Stage B参数: {args.color_calib_dir}")
        print(f"冻结3DGS: {args.freeze_3dgs_iters} iters")
    print(f"总迭代数: {args.iterations}")
    print()
    
    # 创建输出目录
    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    # 创建 Gaussians（使用 Task 1 的参数）
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree,
        appearance_enabled=False,
        appearance_n_fourier_freqs=4,
        appearance_embedding_dim=32
    )
    
    # 创建 Scene（使用 Task 1 的方式）
    print("加载 Scene...")
    scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    
    # 获取相机
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    print(f"训练相机: {len(train_cameras)}")
    print(f"测试相机: {len(test_cameras)}")
    
    # 从 checkpoint 恢复
    print(f"\n从 checkpoint 恢复...")
    checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    
    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        model_params, iteration = checkpoint
        print(f"Checkpoint iteration: {iteration}")
        
        # 使用 restore 方法恢复
        gaussians.restore(model_params, opt)
        print(f"✅ 从 checkpoint 恢复完成，点数: {gaussians.get_xyz.shape[0]}")
    else:
        raise ValueError("❌ Checkpoint 格式错误，期望 (model_params, iteration) tuple")
    
    # 设置训练
    gaussians.training_setup(opt)
    
    # 背景色（使用 Task 1 的设置）
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 初始化 ColorCalib 管理器
    calib_mgr = None
    calib_optimizer = None
    
    if args.use_color_calib:
        print(f"\n初始化 ColorCalib...")
        camera_ids = [cam.image_name for cam in train_cameras]
        calib_mgr = ColorCalibManager(camera_ids=camera_ids, device='cuda')
        
        # 从 Stage B 加载参数
        if args.color_calib_dir and os.path.exists(args.color_calib_dir):
            loaded_count = 0
            for cam_id in camera_ids:
                npz_path = os.path.join(args.color_calib_dir, f'{cam_id}.npz')
                if os.path.exists(npz_path):
                    loaded_count += 1
            
            if loaded_count > 0:
                calib_mgr.load_from_stage_b(args.color_calib_dir)
                print(f"✅ 从 Stage B 加载了 {loaded_count}/{len(camera_ids)} 个相机的参数")
            else:
                print(f"⚠️  Stage B 参数目录为空，使用恒等初始化")
        else:
            print(f"✅ 使用恒等初始化 {len(camera_ids)} 个相机")
        
        # 创建 ColorCalib 优化器
        calib_optimizer = torch.optim.Adam(
            calib_mgr.get_all_parameters(),
            lr=args.calib_lr_phase1
        )
    
    # Sanity Check（训练前）
    print(f"\n[Sanity Check] 训练前渲染测试...")
    test_cam = train_cameras[0]
    with torch.no_grad():
        render_pkg = render(test_cam, gaussians, pipe, background)
        test_image = render_pkg["render"]
        
        # 应用 ColorCalib（如果有）
        if calib_mgr is not None:
            test_image = calib_mgr.apply(
                test_image.unsqueeze(0),
                test_cam.image_name
            ).squeeze(0)
        
        try:
            sanity_check(test_image, render_pkg, tag=" 初始")
            print("✅ Sanity Check 通过，可以开始训练")
        except RuntimeError as e:
            print(f"❌ Sanity Check 失败: {e}")
            print("请检查 checkpoint 和渲染配置")
            return None
    
    # 如果只是 Sanity Check，就退出
    if args.iterations == 0:
        print("\n✅ Sanity Check 完成，退出")
        return None
    
    # 训练循环
    print(f"\n开始训练...")
    phase = 1
    
    # Phase-1: 冻结 3DGS
    if calib_mgr is not None:
        print(f"\n[Phase-1] 冻结 3DGS，只训练 ColorCalib (iter 1-{args.freeze_3dgs_iters})")
        print(f"  ColorCalib 学习率: {args.calib_lr_phase1}")
        for param in gaussians.parameters():
            param.requires_grad = False
    
    progress_bar = tqdm(range(1, args.iterations + 1), desc="训练进度")
    
    for iteration in progress_bar:
        # 阶段切换
        if iteration == args.freeze_3dgs_iters + 1 and calib_mgr is not None:
            phase = 2
            print(f"\n[Phase-2] 联合训练 (iter {args.freeze_3dgs_iters + 1}-{args.iterations})")
            print(f"  ColorCalib 学习率降低到: {args.calib_lr_phase2}")
            
            # 解冻 3DGS
            for param in gaussians.parameters():
                param.requires_grad = True
            
            # 降低 ColorCalib 学习率
            for param_group in calib_optimizer.param_groups:
                param_group['lr'] = args.calib_lr_phase2
        
        # 随机选择相机
        viewpoint_cam = train_cameras[torch.randint(0, len(train_cameras), (1,)).item()]
        
        # 渲染（使用 Task 1 的方式）
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]  # Linear RGB, (3,H,W)
        
        # 插入 ColorCalib（关键！）
        if calib_mgr is not None:
            image = calib_mgr.apply(
                image.unsqueeze(0),
                viewpoint_cam.image_name
            ).squeeze(0)
        
        # 计算 loss（保持 Linear RGB）
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 添加正则化
        if calib_mgr is not None:
            reg_loss = calib_mgr.compute_regularization(
                lambda_frob=args.calib_reg_lambda,
                lambda_bias=args.calib_reg_mu
            )
            loss = loss + reg_loss
        
        # 反向传播
        loss.backward()
        
        # 优化器步进
        if phase == 1:
            # Phase-1: 只更新 ColorCalib
            if calib_optimizer is not None:
                calib_optimizer.step()
                calib_optimizer.zero_grad()
        else:
            # Phase-2: 更新 3DGS 和 ColorCalib
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()
            if calib_optimizer is not None:
                calib_optimizer.step()
                calib_optimizer.zero_grad()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'phase': phase
        })
        
        # 评估
        if iteration % args.eval_interval == 0:
            with torch.no_grad():
                print(f"\n[Iter {iteration}] 评估...")
                
                # 计算训练集指标
                train_psnr_list = []
                train_de_list = []
                
                for cam in train_cameras[:5]:  # 只评估前5个相机
                    render_pkg = render(cam, gaussians, pipe, background)
                    img = render_pkg["render"]
                    
                    if calib_mgr is not None:
                        img = calib_mgr.apply(img.unsqueeze(0), cam.image_name).squeeze(0)
                    
                    gt = cam.original_image.cuda()
                    
                    # PSNR
                    train_psnr_list.append(psnr(img, gt).item())
                    
                    # ΔE00
                    img_lab = rgb_linear_to_lab(img)
                    gt_lab = rgb_linear_to_lab(gt)
                    de = delta_e2000(img_lab, gt_lab)
                    train_de_list.append(de.mean().item())
                
                avg_psnr = np.mean(train_psnr_list)
                avg_de = np.mean(train_de_list)
                
                print(f"  Loss: {loss.item():.6f}")
                print(f"  训练 PSNR: {avg_psnr:.2f} dB")
                print(f"  训练 ΔE00: {avg_de:.2f}")
                
                if calib_mgr is not None:
                    # 计算校准参数统计
                    M_norms = []
                    t_norms = []
                    for layer in calib_mgr.layers.values():
                        M = layer.M.detach()
                        t = layer.t.detach()
                        M_norms.append(torch.norm(M - torch.eye(3, device=M.device), p='fro').item())
                        t_norms.append(torch.norm(t, p=2).item())
                    
                    print(f"  校准 ||M-I||: {np.mean(M_norms):.4f}")
                    print(f"  校准 ||t||: {np.mean(t_norms):.4f}")
                
                # 保存可视化
                if iteration % (args.eval_interval * 2) == 0:
                    vis_cam = train_cameras[0]
                    render_pkg = render(vis_cam, gaussians, pipe, background)
                    vis_img = render_pkg["render"]
                    
                    if calib_mgr is not None:
                        vis_img = calib_mgr.apply(vis_img.unsqueeze(0), vis_cam.image_name).squeeze(0)
                    
                    vis_gt = vis_cam.original_image.cuda()
                    
                    # 转 sRGB 保存
                    vis_img_np = vis_img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    vis_gt_np = vis_gt.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                    
                    vis_img_srgb = linear_to_srgb_safe(vis_img_np)
                    vis_gt_srgb = linear_to_srgb_safe(vis_gt_np)
                    
                    # 拼接保存
                    vis_combined = np.concatenate([vis_gt_srgb, vis_img_srgb], axis=1)
                    save_path = vis_dir / f"iter_{iteration:06d}_comparison.png"
                    save_rgb(torch.from_numpy(vis_combined.transpose(2, 0, 1)), str(save_path))
    
    print(f"\n✅ 训练完成！")
    print(f"输出目录: {output_dir}")
    
    return None

def main():
    # 参数解析
    parser = ArgumentParser(description="Task 2 Stage C Training")
    
    # 基本参数
    parser.add_argument('-s', '--source_path', type=str, required=True, help='数据集路径')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='模型输出路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='Task 1 checkpoint 路径')
    
    # ColorCalib 参数
    parser.add_argument('--use_color_calib', action='store_true', help='启用颜色校准')
    parser.add_argument('--color_calib_dir', type=str, default='', help='Stage B 参数目录')
    parser.add_argument('--freeze_3dgs_iters', type=int, default=100, help='Phase-1 冻结 3DGS 的迭代数')
    
    # 训练参数
    parser.add_argument('--iterations', type=int, default=3000, help='总迭代数')
    parser.add_argument('--eval_interval', type=int, default=100, help='评估间隔')
    
    # ColorCalib 超参数
    parser.add_argument('--calib_lr_phase1', type=float, default=1e-3, help='Phase-1 ColorCalib 学习率')
    parser.add_argument('--calib_lr_phase2', type=float, default=2e-4, help='Phase-2 ColorCalib 学习率')
    parser.add_argument('--calib_reg_lambda', type=float, default=3e-3, help='||M-I||_F^2 正则化系数')
    parser.add_argument('--calib_reg_mu', type=float, default=1e-4, help='||t||_2^2 正则化系数')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建参数对象（使用 Namespace 方式，类似 render_task1_model.py）
    model_params = Namespace(
        source_path=args.source_path,
        model_path=args.model_path,
        images="images",
        resolution=-1,
        white_background=False,
        eval=False,
        sh_degree=3,
        data_device="cuda"
    )
    
    pipe_params = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False
    )
    
    opt_params = Namespace(
        iterations=args.iterations,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002
    )
    
    # 开始训练
    training(model_params, opt_params, pipe_params, args)

if __name__ == "__main__":
    main()
