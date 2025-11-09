#!/usr/bin/env python3
"""
Task 2 - Step 2 评估（Stage A + Stage B）
在Step 1的基础上，添加Stage B闭式解评估
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append('.')

from stage_a_reinhard import (
    srgb_to_linear, linear_to_srgb, 
    reinhard_color_transfer_linear
)
from stage_b_robust_solver import (
    solve_affine_color_calib,
    solve_global_then_per_camera,
    apply_color_calib,
    save_calib_params
)
from utils.color_space import rgb_linear_to_lab
from utils.sampling import sample_pairs
from utils.visualization import save_rgb, debug_image_stats
from metrics_color import delta_e2000

def load_image_linear(image_path):
    """加载图像并转换到Linear RGB"""
    img = Image.open(image_path).convert('RGB')
    img_srgb = np.array(img).astype(np.float32) / 255.0
    img_linear = srgb_to_linear(img_srgb)
    return img_linear

def compute_metrics(render_lin, gt_lin, mask=None):
    """计算PSNR和ΔE00指标"""
    if mask is not None:
        render_pixels = render_lin[mask]
        gt_pixels = gt_lin[mask]
    else:
        render_pixels = render_lin.reshape(-1, 3)
        gt_pixels = gt_lin.reshape(-1, 3)
    
    # PSNR (Linear RGB空间)
    mse = np.mean((render_pixels - gt_pixels) ** 2)
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    
    # ΔE00 (Linear RGB → Lab)
    lab_render = rgb_linear_to_lab(render_pixels.reshape(-1, 3))
    lab_gt = rgb_linear_to_lab(gt_pixels.reshape(-1, 3))
    
    de_result = delta_e2000(lab_render, lab_gt)
    
    return {
        'psnr': float(psnr),
        'de_median': float(de_result['median']),
        'de_mean': float(de_result['mean']),
        'de_p95': float(de_result['p95'])
    }

def create_alpha_mask(image_shape, threshold=0.5):
    """创建alpha mask"""
    return np.ones(image_shape[:2], dtype=bool)

def collect_pixels_for_stage_b(image_pairs, n_samples=20000):
    """
    为Stage B收集像素样本
    
    Args:
        image_pairs: List[(render_path, gt_path, img_id)]
        n_samples: 每张图采样的像素数
    
    Returns:
        render_pixels_list: List[(N,3)]
        gt_pixels_list: List[(N,3)]
        image_ids: List[str]
    """
    print("\n收集Stage B训练样本...")
    
    render_pixels_list = []
    gt_pixels_list = []
    image_ids = []
    
    for render_path, gt_path, img_id in tqdm(image_pairs, desc="采样像素"):
        render_lin = load_image_linear(render_path)
        gt_lin = load_image_linear(gt_path)
        alpha_mask = create_alpha_mask(render_lin.shape)
        
        # 使用专家提供的采样函数
        render_pixels, gt_pixels = sample_pairs(
            render_lin.transpose(2,0,1), 
            gt_lin.transpose(2,0,1), 
            alpha_mask, 
            n=n_samples
        )
        
        if render_pixels is not None and len(render_pixels) > 100:
            render_pixels_list.append(render_pixels)
            gt_pixels_list.append(gt_pixels)
            image_ids.append(img_id)
            print(f"  {img_id}: {len(render_pixels):,} 像素")
    
    return render_pixels_list, gt_pixels_list, image_ids

def evaluate_image_pair(render_path, gt_path, output_dir, img_id, stage_b_params=None):
    """评估单对图像"""
    # 加载图像
    render_lin = load_image_linear(render_path)
    gt_lin = load_image_linear(gt_path)
    
    H, W = render_lin.shape[:2]
    alpha_mask = create_alpha_mask(render_lin.shape)
    
    # 基线1: Task 1修复后
    baseline1_metrics = compute_metrics(render_lin, gt_lin, alpha_mask)
    
    # Stage A: 统计匹配
    stage_a_calibrated = reinhard_color_transfer_linear(
        render_lin, gt_lin, alpha_mask
    )
    stage_a_metrics = compute_metrics(stage_a_calibrated, gt_lin, alpha_mask)
    stage_a_psnr_gain = stage_a_metrics['psnr'] - baseline1_metrics['psnr']
    stage_a_de_gain = baseline1_metrics['de_median'] - stage_a_metrics['de_median']
    
    # Stage B: 闭式解（如果有参数）
    stage_b_metrics = None
    stage_b_psnr_gain = 0
    stage_b_de_gain = 0
    
    if stage_b_params is not None:
        M, t = stage_b_params
        stage_b_calibrated = apply_color_calib(render_lin, M, t)
        stage_b_metrics = compute_metrics(stage_b_calibrated, gt_lin, alpha_mask)
        stage_b_psnr_gain = stage_b_metrics['psnr'] - baseline1_metrics['psnr']
        stage_b_de_gain = baseline1_metrics['de_median'] - stage_b_metrics['de_median']
    
    # 保存可视化
    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建对比图（所有图像都是Linear RGB）
    if stage_b_params is not None:
        comparison = np.concatenate([render_lin, gt_lin, stage_a_calibrated, stage_b_calibrated], axis=1)
    else:
        comparison = np.concatenate([render_lin, gt_lin, stage_a_calibrated], axis=1)
    
    # 使用专家提供的稳健保存函数（Linear -> sRGB -> uint8）
    save_rgb(vis_dir / f"{img_id}_comparison.png", comparison, space="linear")
    
    return {
        'image_id': img_id,
        'baseline1': baseline1_metrics,
        'stage_a': stage_a_metrics,
        'task2_stage_a_psnr_gain': float(stage_a_psnr_gain),
        'task2_stage_a_de_gain': float(stage_a_de_gain),
        'stage_b': stage_b_metrics,
        'task2_stage_b_psnr_gain': float(stage_b_psnr_gain),
        'task2_stage_b_de_gain': float(stage_b_de_gain)
    }

def main():
    parser = argparse.ArgumentParser(description="Task 2 Step 2 评估")
    parser.add_argument('--render_dir', type=str, required=True, help='渲染图像目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT图像目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--n_samples', type=int, default=20000, help='每张图采样像素数')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Task 2 - Step 2 评估（Stage A + Stage B）")
    print("=" * 80)
    print(f"渲染图像: {args.render_dir}")
    print(f"GT图像: {args.gt_dir}")
    print(f"输出: {args.output}")
    print()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找图像对
    render_dir = Path(args.render_dir)
    gt_dir = Path(args.gt_dir)
    
    render_files = sorted(render_dir.glob("*.png"))
    if not render_files:
        print(f"❌ 未找到渲染图像: {render_dir}")
        return
    
    print(f"找到 {len(render_files)} 个渲染图像")
    
    # 准备图像对
    image_pairs = []
    for render_file in render_files:
        img_id = render_file.stem
        gt_file = gt_dir / render_file.name
        
        if not gt_file.exists():
            print(f"⚠️  跳过 {img_id}（未找到GT）")
            continue
        
        image_pairs.append((render_file, gt_file, img_id))
    
    if not image_pairs:
        print("❌ 没有有效的图像对")
        return
    
    print(f"有效图像对: {len(image_pairs)}\n")
    
    # ========================================
    # Stage B: 求解全局校准参数
    # ========================================
    render_pixels_list, gt_pixels_list, image_ids = collect_pixels_for_stage_b(
        image_pairs, n_samples=args.n_samples
    )
    
    if not render_pixels_list:
        print("❌ 没有有效的像素样本")
        return
    
    # 求解全局参数
    M_global, t_global, M_cameras, t_cameras = solve_global_then_per_camera(
        render_pixels_list,
        gt_pixels_list,
        reg_lambda=1e-2,
        reg_mu=1e-3,
        huber_delta=0.02,
        camera_reg_scale=2.0
    )
    
    # 保存参数
    params_dir = output_dir / "stage_b_params"
    params_dir.mkdir(exist_ok=True)
    save_calib_params(M_global, t_global, params_dir / "global.npz")
    
    for i, (img_id, M_cam, t_cam) in enumerate(zip(image_ids, M_cameras, t_cameras)):
        save_calib_params(M_cam, t_cam, params_dir / f"{img_id}.npz")
    
    # ========================================
    # 评估所有图像对
    # ========================================
    print("\n评估所有图像对...")
    results = []
    
    for render_file, gt_file, img_id in tqdm(image_pairs, desc="评估进度"):
        try:
            # 使用全局参数（简化版）
            result = evaluate_image_pair(
                render_file, gt_file, args.output, img_id, 
                stage_b_params=(M_global, t_global)
            )
            results.append(result)
        except Exception as e:
            print(f"  ❌ {img_id} 处理失败: {e}")
            continue
    
    if not results:
        print("\n❌ 没有成功处理的图像")
        return
    
    # ========================================
    # 计算整体统计
    # ========================================
    baseline1_psnr_list = [r['baseline1']['psnr'] for r in results]
    baseline1_de_list = [r['baseline1']['de_median'] for r in results]
    
    stage_a_psnr_list = [r['stage_a']['psnr'] for r in results]
    stage_a_de_list = [r['stage_a']['de_median'] for r in results]
    task2_a_psnr_gain_list = [r['task2_stage_a_psnr_gain'] for r in results]
    task2_a_de_gain_list = [r['task2_stage_a_de_gain'] for r in results]
    
    stage_b_results = [r for r in results if r['stage_b'] is not None]
    if stage_b_results:
        stage_b_psnr_list = [r['stage_b']['psnr'] for r in stage_b_results]
        stage_b_de_list = [r['stage_b']['de_median'] for r in stage_b_results]
        task2_b_psnr_gain_list = [r['task2_stage_b_psnr_gain'] for r in stage_b_results]
        task2_b_de_gain_list = [r['task2_stage_b_de_gain'] for r in stage_b_results]
    
    summary = {
        'num_images': len(results),
        'baseline1': {
            'avg_psnr': float(np.mean(baseline1_psnr_list)),
            'median_psnr': float(np.median(baseline1_psnr_list)),
            'avg_de_median': float(np.mean(baseline1_de_list)),
            'median_de_median': float(np.median(baseline1_de_list))
        },
        'stage_a': {
            'avg_psnr': float(np.mean(stage_a_psnr_list)),
            'median_psnr': float(np.median(stage_a_psnr_list)),
            'avg_de_median': float(np.mean(stage_a_de_list)),
            'median_de_median': float(np.median(stage_a_de_list)),
            'avg_psnr_gain': float(np.mean(task2_a_psnr_gain_list)),
            'median_psnr_gain': float(np.median(task2_a_psnr_gain_list)),
            'avg_de_gain': float(np.mean(task2_a_de_gain_list)),
            'median_de_gain': float(np.median(task2_a_de_gain_list))
        },
        'per_image_results': results
    }
    
    if stage_b_results:
        summary['stage_b'] = {
            'avg_psnr': float(np.mean(stage_b_psnr_list)),
            'median_psnr': float(np.median(stage_b_psnr_list)),
            'avg_de_median': float(np.mean(stage_b_de_list)),
            'median_de_median': float(np.median(stage_b_de_list)),
            'avg_psnr_gain': float(np.mean(task2_b_psnr_gain_list)),
            'median_psnr_gain': float(np.median(task2_b_psnr_gain_list)),
            'avg_de_gain': float(np.mean(task2_b_de_gain_list)),
            'median_de_gain': float(np.median(task2_b_de_gain_list))
        }
    
    # 保存结果
    with open(output_dir / "step2_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ========================================
    # 打印最终报告
    # ========================================
    print("\n" + "=" * 80)
    print("Task 2 - Step 2 评估完成")
    print("=" * 80)
    print(f"处理图像数: {len(results)}")
    print()
    
    print("【基线1】Task 1修复后:")
    print(f"  平均PSNR: {summary['baseline1']['avg_psnr']:.2f} dB")
    print(f"  中位PSNR: {summary['baseline1']['median_psnr']:.2f} dB")
    print(f"  平均ΔE00: {summary['baseline1']['avg_de_median']:.2f}")
    print(f"  中位ΔE00: {summary['baseline1']['median_de_median']:.2f}")
    print()
    
    print("【Task 2纯增益】Stage A统计匹配:")
    print(f"  平均PSNR增益: {summary['stage_a']['avg_psnr_gain']:+.2f} dB")
    print(f"  中位PSNR增益: {summary['stage_a']['median_psnr_gain']:+.2f} dB")
    print(f"  平均ΔE00改善: {summary['stage_a']['avg_de_gain']:+.2f}")
    print(f"  中位ΔE00改善: {summary['stage_a']['median_de_gain']:+.2f}")
    print(f"  最终中位ΔE00: {summary['stage_a']['median_de_median']:.2f}")
    print()
    
    if 'stage_b' in summary:
        print("【Task 2纯增益】Stage B闭式解:")
        print(f"  平均PSNR增益: {summary['stage_b']['avg_psnr_gain']:+.2f} dB")
        print(f"  中位PSNR增益: {summary['stage_b']['median_psnr_gain']:+.2f} dB")
        print(f"  平均ΔE00改善: {summary['stage_b']['avg_de_gain']:+.2f}")
        print(f"  中位ΔE00改善: {summary['stage_b']['median_de_gain']:+.2f}")
        print(f"  最终中位ΔE00: {summary['stage_b']['median_de_median']:.2f}")
        print()
        
        # Gate T2-2 验证
        print("【Gate T2-2 验证】Stage B闭式解:")
        gate_psnr_gain = summary['stage_b']['median_psnr_gain']
        gate_de_median = summary['stage_b']['median_de_median']
        
        gate_psnr_pass = gate_psnr_gain >= 0.5
        gate_de_pass = gate_de_median <= 5.0
        
        print(f"  ✓ PSNR增益 ≥ 0.5 dB: {'✅ 通过' if gate_psnr_pass else '❌ 未通过'} ({gate_psnr_gain:+.2f} dB)")
        print(f"  ✓ ΔE00中位数 ≤ 5.0: {'✅ 通过' if gate_de_pass else '❌ 未通过'} ({gate_de_median:.2f})")
        
        if gate_psnr_pass and gate_de_pass:
            print("\n  🎉 Gate T2-2 通过！")
        else:
            print("\n  ⚠️  Gate T2-2 未通过")
    
    print()
    print(f"结果已保存: {output_dir / 'step2_results.json'}")
    print(f"对比图: {output_dir / 'vis/'}")
    print(f"Stage B参数: {output_dir / 'stage_b_params/'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
