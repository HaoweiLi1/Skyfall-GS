#!/usr/bin/env python3
"""
Task 2 - Step 0 + Step 1 统一评估
按照专家要求正确分账Task 1修复收益与Task 2纯颜色增益

基线划分：
- 基线0: 原始（sRGB混用 + BGR错误）
- 基线1: Task 1修复后（Linear + BGR正确）← Task 2的起点
- 基线2: +Stage A（统计匹配）
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
from utils.color_space import rgb_linear_to_lab
from utils.visualization import save_rgb, debug_image_stats
from metrics_color import delta_e2000

def load_image_linear(image_path):
    """加载图像并转换到Linear RGB（Task 1修复后的正确路径）"""
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
    """创建alpha mask（简化版，全图有效）"""
    return np.ones(image_shape[:2], dtype=bool)

def evaluate_image_pair(render_path, gt_path, output_dir, img_id):
    """
    评估单对图像，分账统计
    
    返回：
    - baseline1: Task 1修复后（Linear + BGR正确）
    - stage_a: +Stage A统计匹配
    """
    print(f"\n处理: {img_id}")
    
    # 加载图像（Task 1修复后的正确路径）
    render_lin = load_image_linear(render_path)
    gt_lin = load_image_linear(gt_path)
    
    H, W = render_lin.shape[:2]
    print(f"  尺寸: {H}×{W}")
    
    # 创建mask
    alpha_mask = create_alpha_mask(render_lin.shape)
    
    # ========================================
    # 基线1: Task 1修复后（这是Task 2的起点）
    # ========================================
    baseline1_metrics = compute_metrics(render_lin, gt_lin, alpha_mask)
    print(f"  基线1（Task 1修复后）:")
    print(f"    PSNR: {baseline1_metrics['psnr']:.2f} dB")
    print(f"    ΔE00: {baseline1_metrics['de_median']:.2f}")
    
    # ========================================
    # Stage A: 统计匹配
    # ========================================
    stage_a_calibrated = reinhard_color_transfer_linear(
        render_lin, gt_lin, alpha_mask
    )
    stage_a_metrics = compute_metrics(stage_a_calibrated, gt_lin, alpha_mask)
    
    # 计算Task 2纯增益
    stage_a_psnr_gain = stage_a_metrics['psnr'] - baseline1_metrics['psnr']
    stage_a_de_gain = baseline1_metrics['de_median'] - stage_a_metrics['de_median']
    
    print(f"  Stage A（统计匹配）:")
    print(f"    PSNR: {stage_a_metrics['psnr']:.2f} dB (Task 2增益: {stage_a_psnr_gain:+.2f} dB)")
    print(f"    ΔE00: {stage_a_metrics['de_median']:.2f} (Task 2改善: {stage_a_de_gain:+.2f})")
    
    # ========================================
    # 保存可视化
    # ========================================
    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建对比图（所有图像都是Linear RGB）
    comparison = np.concatenate([
        render_lin,           # 渲染（基线1）
        gt_lin,               # GT
        stage_a_calibrated    # Stage A校正
    ], axis=1)
    
    # 使用专家提供的稳健保存函数（Linear -> sRGB -> uint8）
    save_rgb(vis_dir / f"{img_id}_comparison.png", comparison, space="linear")
    
    # 计算ΔE map
    lab_render = rgb_linear_to_lab(render_lin.reshape(-1, 3)).reshape(H, W, 3)
    lab_gt = rgb_linear_to_lab(gt_lin.reshape(-1, 3)).reshape(H, W, 3)
    lab_stage_a = rgb_linear_to_lab(stage_a_calibrated.reshape(-1, 3)).reshape(H, W, 3)
    
    de_baseline = delta_e2000(lab_render.reshape(-1, 3), lab_gt.reshape(-1, 3))['map'].reshape(H, W)
    de_stage_a = delta_e2000(lab_stage_a.reshape(-1, 3), lab_gt.reshape(-1, 3))['map'].reshape(H, W)
    
    # 保存ΔE map（使用热力图）
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(de_baseline, cmap='hot', vmin=0, vmax=10)
    axes[0].set_title(f'Baseline1 ΔE00 (median={baseline1_metrics["de_median"]:.2f})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(de_stage_a, cmap='hot', vmin=0, vmax=10)
    axes[1].set_title(f'Stage A ΔE00 (median={stage_a_metrics["de_median"]:.2f})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(vis_dir / f"{img_id}_delta_e_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'image_id': img_id,
        'baseline1': baseline1_metrics,
        'stage_a': stage_a_metrics,
        'task2_stage_a_psnr_gain': float(stage_a_psnr_gain),
        'task2_stage_a_de_gain': float(stage_a_de_gain)
    }

def main():
    parser = argparse.ArgumentParser(description="Task 2 Step 0 + Step 1 评估")
    parser.add_argument('--render_dir', type=str, required=True, help='渲染图像目录（Task 1输出）')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT图像目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Task 2 - Step 0 + Step 1 统一评估")
    print("正确分账Task 1修复收益与Task 2纯颜色增益")
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
    
    # 处理所有图像对
    results = []
    for render_file, gt_file, img_id in tqdm(image_pairs, desc="评估进度"):
        try:
            result = evaluate_image_pair(render_file, gt_file, args.output, img_id)
            results.append(result)
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
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
    
    task2_psnr_gain_list = [r['task2_stage_a_psnr_gain'] for r in results]
    task2_de_gain_list = [r['task2_stage_a_de_gain'] for r in results]
    
    summary = {
        'num_images': len(results),
        'baseline1_task1_fixed': {
            'avg_psnr': float(np.mean(baseline1_psnr_list)),
            'median_psnr': float(np.median(baseline1_psnr_list)),
            'avg_de_median': float(np.mean(baseline1_de_list)),
            'median_de_median': float(np.median(baseline1_de_list))
        },
        'stage_a_after_calibration': {
            'avg_psnr': float(np.mean(stage_a_psnr_list)),
            'median_psnr': float(np.median(stage_a_psnr_list)),
            'avg_de_median': float(np.mean(stage_a_de_list)),
            'median_de_median': float(np.median(stage_a_de_list))
        },
        'task2_pure_gain_stage_a': {
            'avg_psnr_gain': float(np.mean(task2_psnr_gain_list)),
            'median_psnr_gain': float(np.median(task2_psnr_gain_list)),
            'avg_de_gain': float(np.mean(task2_de_gain_list)),
            'median_de_gain': float(np.median(task2_de_gain_list))
        },
        'per_image_results': results
    }
    
    # 保存结果
    with open(output_dir / "step0_step1_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ========================================
    # 打印最终报告
    # ========================================
    print("\n" + "=" * 80)
    print("Task 2 - Step 0 + Step 1 评估完成")
    print("=" * 80)
    print(f"处理图像数: {len(results)}")
    print()
    
    print("【基线1】Task 1修复后（Linear + BGR正确）← Task 2的起点:")
    print(f"  平均PSNR: {summary['baseline1_task1_fixed']['avg_psnr']:.2f} dB")
    print(f"  中位PSNR: {summary['baseline1_task1_fixed']['median_psnr']:.2f} dB")
    print(f"  平均ΔE00: {summary['baseline1_task1_fixed']['avg_de_median']:.2f}")
    print(f"  中位ΔE00: {summary['baseline1_task1_fixed']['median_de_median']:.2f}")
    print()
    
    print("【Task 2纯增益】Stage A统计匹配:")
    print(f"  平均PSNR增益: {summary['task2_pure_gain_stage_a']['avg_psnr_gain']:+.2f} dB")
    print(f"  中位PSNR增益: {summary['task2_pure_gain_stage_a']['median_psnr_gain']:+.2f} dB")
    print(f"  平均ΔE00改善: {summary['task2_pure_gain_stage_a']['avg_de_gain']:+.2f}")
    print(f"  中位ΔE00改善: {summary['task2_pure_gain_stage_a']['median_de_gain']:+.2f}")
    print()
    
    print("【Stage A最终结果】:")
    print(f"  平均PSNR: {summary['stage_a_after_calibration']['avg_psnr']:.2f} dB")
    print(f"  中位PSNR: {summary['stage_a_after_calibration']['median_psnr']:.2f} dB")
    print(f"  平均ΔE00: {summary['stage_a_after_calibration']['avg_de_median']:.2f}")
    print(f"  中位ΔE00: {summary['stage_a_after_calibration']['median_de_median']:.2f}")
    print()
    
    # ========================================
    # Gate T2-1 验证
    # ========================================
    print("【Gate T2-1 验证】Stage A基线验证:")
    
    gate_psnr_gain = summary['task2_pure_gain_stage_a']['median_psnr_gain']
    gate_de_median = summary['stage_a_after_calibration']['median_de_median']
    
    gate_psnr_pass = gate_psnr_gain >= 1.0
    gate_de_pass = gate_de_median <= 4.0
    
    print(f"  ✓ PSNR增益 ≥ 1.0 dB: {'✅ 通过' if gate_psnr_pass else '❌ 未通过'} ({gate_psnr_gain:+.2f} dB)")
    print(f"  ✓ ΔE00中位数 ≤ 4.0: {'✅ 通过' if gate_de_pass else '❌ 未通过'} ({gate_de_median:.2f})")
    
    if gate_psnr_pass and gate_de_pass:
        print("\n  🎉 Gate T2-1 通过！")
    else:
        print("\n  ⚠️  Gate T2-1 未通过")
    
    print()
    print(f"结果已保存: {output_dir / 'step0_step1_results.json'}")
    print(f"对比图: {output_dir / 'vis/'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
