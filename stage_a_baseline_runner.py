#!/usr/bin/env python3
"""
Stage A 基线执行器
使用Task 1训练好的模型，对所有训练相机应用Reinhard颜色迁移
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import cv2

# 添加项目路径
sys.path.append('.')

# 导入专家提供的模块
from stage_a_reinhard import reinhard_color_transfer_linear, srgb_to_linear, linear_to_srgb, create_mask
from metrics_color import compute_color_metrics

# 导入3DGS相关模块
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, get_combined_args

def load_trained_model(model_path, iteration):
    """加载训练好的模型"""
    print(f"[Stage A] 加载模型: {model_path} @ iteration {iteration}")
    
    # 读取保存的配置
    cfg_file = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"配置文件不存在: {cfg_file}")
    
    with open(cfg_file, 'r') as f:
        cfg_str = f.read()
    
    # 手动解析配置（简单的键值对提取）
    import re
    cfg_dict = {}
    # 提取source_path
    match = re.search(r"source_path='([^']+)'", cfg_str)
    if match:
        cfg_dict['source_path'] = match.group(1)
    
    # 提取其他参数
    for key in ['sh_degree', 'appearance_enabled', 'appearance_n_fourier_freqs', 'appearance_embedding_dim',
                'images', 'resolution', 'white_background', 'data_device', 'eval', 'load_allres']:
        match = re.search(rf"{key}=([^,\)]+)", cfg_str)
        if match:
            value = match.group(1).strip().strip("'\"")
            # 转换类型
            if value in ['True', 'False']:
                cfg_dict[key] = value == 'True'
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                cfg_dict[key] = int(value)
            else:
                cfg_dict[key] = value
    
    # 创建高斯模型
    gaussians = GaussianModel(
        sh_degree=cfg_dict.get('sh_degree', 0),
        appearance_enabled=cfg_dict.get('appearance_enabled', False),
        appearance_n_fourier_freqs=cfg_dict.get('appearance_n_fourier_freqs', 0),
        appearance_embedding_dim=cfg_dict.get('appearance_embedding_dim', 0)
    )
    
    # 创建模型参数对象
    class ModelArgs:
        def __init__(self, cfg_dict, model_path):
            self.model_path = model_path
            self.source_path = cfg_dict.get('source_path', '')
            self.images = cfg_dict.get('images', 'images')
            self.resolution = cfg_dict.get('resolution', -1)
            self.white_background = cfg_dict.get('white_background', False)
            self.data_device = cfg_dict.get('data_device', 'cuda')
            self.eval = cfg_dict.get('eval', False)
            self.load_allres = cfg_dict.get('load_allres', False)
    
    model_args = ModelArgs(cfg_dict, model_path)
    
    # 加载场景
    scene = Scene(model_args, gaussians, load_iteration=iteration, shuffle=False)
    
    print(f"[Stage A] 模型加载完成，高斯数量: {len(gaussians._xyz)}")
    return scene, gaussians

def render_camera(camera, gaussians, pipeline_args, background, kernel_size=0.1):
    """渲染单个相机"""
    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipeline_args, background, kernel_size=kernel_size)
        image = render_pkg["render"]
        alpha = render_pkg.get("alpha", None)
    return image, alpha

def process_single_camera(camera, gaussians, pipeline_args, background, output_dir, cam_id):
    """处理单个相机的颜色迁移"""
    print(f"[Stage A] 处理相机 {cam_id}...")
    
    # 渲染
    render_image, alpha = render_camera(camera, gaussians, pipeline_args, background)
    
    # 转换为numpy (CHW -> HWC)
    render_np = render_image.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    if alpha is not None:
        alpha_np = alpha.detach().cpu().numpy().squeeze()  # (H, W)
    else:
        alpha_np = np.ones(render_np.shape[:2])
    
    # GT图像（已经是Linear RGB，Task 1已修复）
    gt_image = camera.original_image.cuda()
    gt_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    
    # 创建有效像素mask
    mask = create_mask(alpha_np, render_np)
    valid_pixels = np.sum(mask)
    print(f"  有效像素: {valid_pixels} / {mask.size} ({100*valid_pixels/mask.size:.1f}%)")
    
    if valid_pixels < 1000:
        print(f"  ⚠️  有效像素太少，跳过相机 {cam_id}")
        return None
    
    # 计算原始指标
    original_metrics = compute_color_metrics(render_np, gt_np, mask)
    print(f"  原始 - PSNR: {original_metrics['psnr']:.2f} dB, ΔE00: {original_metrics['delta_e00']['median']:.2f}")
    
    # 应用Reinhard颜色迁移
    calibrated_np = reinhard_color_transfer_linear(render_np, gt_np, mask)
    
    # 计算校正后指标
    calibrated_metrics = compute_color_metrics(calibrated_np, gt_np, mask)
    print(f"  校正后 - PSNR: {calibrated_metrics['psnr']:.2f} dB, ΔE00: {calibrated_metrics['delta_e00']['median']:.2f}")
    
    # 计算提升
    psnr_improvement = calibrated_metrics['psnr'] - original_metrics['psnr']
    delta_e_improvement = original_metrics['delta_e00']['median'] - calibrated_metrics['delta_e00']['median']
    print(f"  提升 - PSNR: +{psnr_improvement:.2f} dB, ΔE00: -{delta_e_improvement:.2f}")
    
    # 保存对比图
    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    # 转换为sRGB用于保存
    render_srgb = linear_to_srgb(render_np)
    gt_srgb = linear_to_srgb(gt_np)
    calibrated_srgb = linear_to_srgb(calibrated_np)
    
    # 创建对比图
    comparison = np.concatenate([render_srgb, gt_srgb, calibrated_srgb], axis=1)
    Image.fromarray(comparison).save(vis_dir / f"cam_{cam_id:03d}_comparison.png")
    
    # 保存ΔE误差图
    delta_e_map = calibrated_metrics['delta_e00']['map']
    delta_e_vis = np.clip(delta_e_map / 10.0 * 255, 0, 255).astype(np.uint8)
    delta_e_colored = cv2.applyColorMap(delta_e_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(vis_dir / f"cam_{cam_id:03d}_delta_e.png"), delta_e_colored)
    
    return {
        'camera_id': cam_id,
        'valid_pixels': int(valid_pixels),
        'original_psnr': float(original_metrics['psnr']),
        'calibrated_psnr': float(calibrated_metrics['psnr']),
        'psnr_improvement': float(psnr_improvement),
        'original_delta_e00_median': float(original_metrics['delta_e00']['median']),
        'calibrated_delta_e00_median': float(calibrated_metrics['delta_e00']['median']),
        'delta_e_improvement': float(delta_e_improvement)
    }

def main():
    parser = argparse.ArgumentParser(description="Stage A 基线执行器")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--iteration', type=int, default=3000, help='迭代数')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    args = parser.parse_args()
    
    print("========================================================================")
    print("Task 2 - Stage A 基线执行器")
    print("========================================================================")
    print(f"模型: {args.model_path}")
    print(f"迭代: {args.iteration}")
    print(f"输出: {args.output}")
    print("")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    scene, gaussians = load_trained_model(args.model_path, args.iteration)
    
    # 设置渲染参数
    class PipelineArgs:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
    
    pipeline_args = PipelineArgs()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 获取训练相机
    train_cameras = scene.getTrainCameras()
    print(f"[Stage A] 处理 {len(train_cameras)} 个训练相机")
    
    # 处理所有相机
    results = []
    for i, camera in enumerate(train_cameras):
        try:
            result = process_single_camera(
                camera, gaussians, pipeline_args, 
                background, args.output, i
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ❌ 相机 {i} 处理失败: {e}")
            continue
    
    if not results:
        print("❌ 没有成功处理的相机")
        return
    
    # 计算整体统计
    avg_psnr_improvement = np.mean([r['psnr_improvement'] for r in results])
    avg_delta_e00_median = np.mean([r['calibrated_delta_e00_median'] for r in results])
    
    summary = {
        'num_cameras': len(results),
        'avg_psnr_improvement': float(avg_psnr_improvement),
        'avg_delta_e00_median': float(avg_delta_e00_median),
        'per_camera_results': results
    }
    
    # 保存结果
    with open(output_dir / "results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("")
    print("========================================================================")
    print("Stage A 完成")
    print("========================================================================")
    print(f"处理相机数: {len(results)}")
    print(f"平均PSNR提升: {avg_psnr_improvement:.2f} dB")
    print(f"平均ΔE00: {avg_delta_e00_median:.2f}")
    
    # Gate T2-1验证
    psnr_ok = avg_psnr_improvement >= 1.0
    delta_e_ok = avg_delta_e00_median <= 4.0
    
    print("")
    print("Gate T2-1验证:")
    print(f"  PSNR提升≥1.0dB: {'✅' if psnr_ok else '❌'} ({avg_psnr_improvement:.2f} dB)")
    print(f"  ΔE00≤4.0: {'✅' if delta_e_ok else '❌'} ({avg_delta_e00_median:.2f})")
    
    if psnr_ok and delta_e_ok:
        print("  🎉 Gate T2-1 通过！可以进入Stage B")
    else:
        print("  ⚠️  Gate T2-1 未通过，需要调整参数")
    
    print(f"")
    print(f"结果已保存: {output_dir / 'results.json'}")
    print(f"对比图: {output_dir / 'vis/'}")

if __name__ == "__main__":
    main()
