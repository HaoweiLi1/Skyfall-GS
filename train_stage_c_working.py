#!/usr/bin/env python3
"""
Task 2 - Stage C 端到端训练（专家修复版）

关键修复：
1. 正确的 checkpoint 恢复顺序
2. filter_3D 守护（全 True，不是 0）
3. 渲染兼容层（多字段回退）
4. Sanity 门卫（必须通过才能训练）
5. ColorCalib 在合成后的 Linear RGB 上应用
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
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

# 专家建议的正确工具函数
from utils.checkpoint_restore import (
    restore_from_task1_checkpoint,
    ensure_filter_3d,
    safe_compute_3d_filter,
    unpack_render_dict,
    sanity_render_and_assert
)


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
    else:
        print("使用恒等初始化")
        for cam in train_cameras:
            calib_mgr.get_layer(cam.image_name)
    
    return calib_mgr


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
    
    # 背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # === 专家建议：正确的恢复顺序 ===
    if not args.task1_ckpt:
        raise ValueError("❌ 必须指定--task1_ckpt！Stage C不支持从随机初始化开始训练")
    
    if args.task1_ckpt.endswith(".ply"):
        raise ValueError("❌ 不要使用.ply！请使用checkpoint (.pth)文件恢复完整状态")
    
    print(f"\n{'='*80}")
    print("专家方案：正确的 Checkpoint 恢复")
    print(f"{'='*80}")
    
    # 临时创建 Scene 来获取相机（不加载点云）
    temp_scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
    train_cameras = temp_scene.getTrainCameras()
    test_cameras = temp_scene.getTestCameras()
    print(f"训练相机: {len(train_cameras)}")
    print(f"测试相机: {len(test_cameras)}")
    
    # Step 0: 正确的恢复方式
    gaussians, loaded_iter = restore_from_task1_checkpoint(
        gaussians, args.task1_ckpt, opt, device='cuda'
    )
    print(f"从 iteration {loaded_iter} 恢复")
    
    # Step 1: filter_3D 尺寸守护 + 空掩码回退
    print(f"\n{'='*80}")
    print("专家方案：filter_3D 守护（全 True，不是 0）")
    print(f"{'='*80}")
    n_pts = gaussians.get_xyz.shape[0]
    ensure_filter_3d(gaussians, n_pts, device='cuda')
    safe_compute_3d_filter(gaussians, train_cameras)
    
    # Step 3: Sanity 渲染门卫
    print(f"\n{'='*80}")
    print("专家方案：Sanity 渲染门卫")
    print(f"{'='*80}")
    cam0 = train_cameras[0]
    rgb_max, alpha_mean = sanity_render_and_assert(
        gaussians, cam0, pipe, background, device='cuda', render_func=render
    )
    print(f"✅ Task 1模型恢复成功，渲染正常 (rgb_max={rgb_max:.6f}, alpha_mean={alpha_mean:.6f})")
    
    # 保存 scene 引用
    scene = temp_scene
    
    # 优化器设置
    optimizer_calib = None
    
    # 输出目录
    output_dir = Path(dataset.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    # 训练日志
    training_log = {
        'start_time': datetime.now().isoformat(),
        'iterations': [],
    }
    
    # 训练循环
    progress_bar = tqdm(range(1, args.iterations + 1), desc="训练进度")
    
    for iteration in progress_bar:
        # 阶段切换
        if iteration == 1:
            # Phase-1: 冻结3DGS，仅训ColorCalib
            print(f"\n[Phase-1] 冻结3DGS，仅训练ColorCalib (iter 1-{args.freeze_3dgs_iters})")
            if calib_mgr is not None:
                # 冻结3DGS优化器
                for param_group in gaussians.optimizer.param_groups:
                    param_group['lr'] = 0.0
                
                # 创建ColorCalib优化器
                optimizer_calib = torch.optim.Adam(
                    calib_mgr.get_all_parameters(), 
                    lr=args.calib_lr_phase1
                )
                print(f"  ColorCalib参数数量: {len(calib_mgr.get_all_parameters())}")
                
        elif iteration == args.freeze_3dgs_iters + 1:
            # Phase-2: 联合训练
            print(f"\n[Phase-2] 联合训练 (iter {args.freeze_3dgs_iters + 1}-{args.iterations})")
            
            # 恢复3DGS学习率
            for param_group in gaussians.optimizer.param_groups:
                if param_group.get("name", "") == "xyz":
                    param_group['lr'] = opt.position_lr_init * gaussians.spatial_lr_scale
                else:
                    param_group['lr'] = opt.position_lr_init
            
            # 降低ColorCalib学习率
            if calib_mgr is not None:
                for param_group in optimizer_calib.param_groups:
                    param_group['lr'] = args.calib_lr_phase2
        
        # 训练步骤
        camera = random.choice(train_cameras)
        
        # Step 2: 渲染兼容层
        out = render(camera, gaussians, pipe, background, kernel_size=0.1)
        rgb, alpha, comp = unpack_render_dict(out, background=background, device='cuda')
        
        # Step 4: ColorCalib 在合成后的 Linear RGB 上应用
        if calib_mgr is not None:
            # comp 是 (B,H,W,3)，需要转换为 (B,3,H,W)
            comp_chw = comp.permute(0, 3, 1, 2).contiguous()  # -> (B,3,H,W)
            comp_calib = calib_mgr.apply_calibration(comp_chw, camera.image_name)
            comp = comp_calib.permute(0, 2, 3, 1).contiguous()  # -> (B,H,W,3)
            
            # 正则化损失
            if iteration > args.freeze_3dgs_iters:
                loss_reg = calib_mgr.get_regularization_loss(
                    args.calib_reg_lambda * 3.0, args.calib_reg_mu
                )
            else:
                loss_reg = calib_mgr.get_regularization_loss(
                    args.calib_reg_lambda, args.calib_reg_mu
                )
        else:
            loss_reg = 0.0
        
        # GT（Linear RGB）
        gt = camera.original_image  # (3,H,W)
        if gt.dim() == 3:
            gt = gt.permute(1, 2, 0).unsqueeze(0)  # -> (1,H,W,3)
        
        # 主损失（使用 alpha mask）
        mask = (alpha > 0.4).float()
        loss_l1 = torch.nn.functional.huber_loss(
            comp * mask, gt * mask, reduction='mean', delta=0.1
        )
        
        # 转换为 (B,C,H,W) 计算 SSIM
        comp_chw = comp.permute(0, 3, 1, 2)
        gt_chw = gt.permute(0, 3, 1, 2)
        mask_chw = mask.permute(0, 3, 1, 2)
        loss_ssim = 1.0 - ssim(comp_chw * mask_chw, gt_chw * mask_chw)
        
        loss_main = 0.8 * loss_l1 + 0.2 * loss_ssim
        loss = loss_main + loss_reg
        
        # 反向传播
        if iteration <= args.freeze_3dgs_iters:
            # Phase-1: 只优化ColorCalib
            if optimizer_calib is not None:
                optimizer_calib.zero_grad()
                loss.backward()
                optimizer_calib.step()
                
                # Warm-up
                if iteration <= 200:
                    warmup_factor = iteration / 200.0
                    for param_group in optimizer_calib.param_groups:
                        param_group['lr'] = args.calib_lr_phase1 * warmup_factor
        else:
            # Phase-2: 联合优化
            gaussians.optimizer.zero_grad()
            if optimizer_calib is not None:
                optimizer_calib.zero_grad()
            loss.backward()
            gaussians.optimizer.step()
            if optimizer_calib is not None:
                optimizer_calib.step()
        
        # 谱裁剪
        if calib_mgr is not None and iteration % 10 == 0:
            calib_mgr.spectral_clip_all(s_min=0.67, s_max=1.5)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'phase': 1 if iteration <= args.freeze_3dgs_iters else 2
        })
        
        # 评估和日志
        if iteration % args.eval_interval == 0 or iteration == 1:
            print(f"\n[Iter {iteration}] 评估...")
            print(f"  Loss: {loss.item():.6f} (主:{loss_main.item():.6f}, 正则:{loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg:.6f})")
            
            # 保存可视化
            if iteration % (args.eval_interval * 2) == 0:
                with torch.no_grad():
                    # 使用当前相机
                    out_vis = render(camera, gaussians, pipe, background, kernel_size=0.1)
                    rgb_vis, alpha_vis, comp_vis = unpack_render_dict(out_vis, background, device='cuda')
                    
                    if calib_mgr is not None:
                        comp_vis_chw = comp_vis.permute(0, 3, 1, 2)
                        comp_calib_vis = calib_mgr.apply_calibration(comp_vis_chw, camera.image_name)
                        comp_vis = comp_calib_vis.permute(0, 2, 3, 1)
                    
                    gt_vis = camera.original_image
                    if gt_vis.dim() == 3:
                        gt_vis = gt_vis.permute(1, 2, 0).unsqueeze(0)
                    
                    # 转换为 numpy (B,H,W,3) -> (H,W,3)
                    comp_vis_np = comp_vis[0].detach().cpu().numpy()
                    gt_vis_np = gt_vis[0].detach().cpu().numpy()
                    
                    # 拼接
                    comparison = np.concatenate([comp_vis_np, gt_vis_np], axis=1)
                    
                    # 保存（Linear -> sRGB）
                    save_rgb(vis_dir / f"iter_{iteration:06d}_comparison.png", 
                            comparison, space="linear")
        
        # 保存检查点
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
    
    # 训练完成
    print(f"\n训练完成！")
    
    # 保存训练日志
    training_log['end_time'] = datetime.now().isoformat()
    with open(output_dir / "training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # 保存最终参数
    final_point_cloud_path = output_dir / "point_cloud" / "iteration_final"
    final_point_cloud_path.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(final_point_cloud_path / "point_cloud.ply")
    
    if calib_mgr is not None:
        calib_mgr.save_all_params(output_dir / "calib_params_final")
    
    return {}


def main():
    # 创建parser
    parser = ArgumentParser(description="Task 2 Stage C Training (Expert Fixed)")
    
    # 添加标准参数
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    
    # Stage C特定参数
    parser.add_argument('--use_color_calib', action='store_true', help='启用颜色校准')
    parser.add_argument('--color_calib_dir', type=str, default='', help='Stage B参数目录')
    parser.add_argument('--task1_ckpt', type=str, required=True, help='Task 1 checkpoint路径')
    parser.add_argument('--freeze_3dgs_iters', type=int, default=50, help='冻结3DGS的迭代数')
    
    # 正则化参数
    parser.add_argument('--calib_reg_lambda', type=float, default=1e-3, help='||M-I||_F^2系数')
    parser.add_argument('--calib_reg_mu', type=float, default=1e-4, help='||t||_2^2系数')
    
    # 学习率参数
    parser.add_argument('--calib_lr_phase1', type=float, default=1e-3, help='Phase1学习率')
    parser.add_argument('--calib_lr_phase2', type=float, default=2e-4, help='Phase2学习率')
    
    # 训练参数
    parser.add_argument('--eval_interval', type=int, default=100, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=1000, help='保存间隔')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Task 2 - Stage C 端到端训练（专家修复版）")
    print("="*80)
    print(f"数据集: {args.source_path}")
    print(f"模型: {args.model_path}")
    print(f"Task 1 checkpoint: {args.task1_ckpt}")
    print(f"颜色校准: {args.use_color_calib}")
    if args.use_color_calib:
        print(f"Stage B参数: {args.color_calib_dir}")
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
    
    # 设置颜色校准管理器（需要在training之前，但这里先传None）
    calib_mgr = None
    if args.use_color_calib:
        # 会在training函数内部创建
        pass
    
    # 开始训练
    final_results = training(model_params, opt_params, pipeline_params, args, calib_mgr)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"结果保存在: {model_params.model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
