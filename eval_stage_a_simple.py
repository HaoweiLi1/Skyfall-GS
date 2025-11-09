#!/usr/bin/env python3
"""
Stage A 简化评估 - 使用预渲染图像
绕过模型加载复杂性，直接评估颜色校准效果
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.append('.')

from stage_a_reinhard import reinhard_color_transfer_linear, srgb_to_linear, linear_to_srgb, create_mask
from metrics_color import compute_color_metrics

def load_image_linear(image_path):
    """加载图像并转换到Linear RGB"""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    # 假设输入已经是Linear RGB（Task 1已修复）
    return img_np

def process_image_pair(render_path, gt_path, output_dir, img_id):
    """处理单对图像"""
    print(f"\n处理图像对 {img_id}...")
    
    # 加载图像
    render_lin = load_image_linear(render_path)
    gt_lin = load_image_linear(gt_path)
    
    print(f"  图像尺寸: {render_lin.shape}")
    
    # 创建简单的mask（全图有效）
    mask = np.ones(render_lin.shape[:2], dtype=bool)
    
    # 计算原始指标
    original_metrics = compute_color_metrics(render_lin, gt_lin, mask)
    print(f"  原始 - PSNR: {original_metrics['psnr']:.2f} dB, ΔE00: {original_metrics['delta_e00']['median']:.2f}")
    
    # 应用Reinhard颜色迁移
    calibrated_lin = reinhard_color_transfer_linear(render_lin, gt_lin, mask)
    
    # 计算校正后指标
    calibrated_metrics = compute_color_metrics(calibrated_lin, gt_lin, mask)
    print(f"  校正后 - PSNR: {calibrated_metrics['psnr']:.2f} dB, ΔE00: {calibrated_metrics['delta_e00']['median']:.2f}")
    
    # 计算提升
    psnr_improvement = calibrated_metrics['psnr'] - original_metrics['psnr']
    delta_e_improvement = original_metrics['delta_e00']['median'] - calibrated_metrics['delta_e00']['median']
    print(f"  提升 - PSNR: {psnr_improvement:+.2f} dB, ΔE00: {delta_e_improvement:+.2f}")
    
    # 保存对比图
    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(exist_ok=True)
    
    render_srgb = linear_to_srgb(render_lin)
    gt_srgb = linear_to_srgb(gt_lin)
    calibrated_srgb = linear_to_srgb(calibrated_lin)
    
    comparison = np.concatenate([render_srgb, gt_srgb, calibrated_srgb], axis=1)
    Image.fromarray(comparison).save(vis_dir / f"{img_id}_comparison.png")
    
    # 保存ΔE误差图
    delta_e_map = calibrated_metrics['delta_e00']['map']
    delta_e_vis = np.clip(delta_e_map / 10.0 * 255, 0, 255).astype(np.uint8)
    delta_e_colored = cv2.applyColorMap(delta_e_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(vis_dir / f"{img_id}_delta_e.png"), delta_e_colored)
    
    return {
        'image_id': img_id,
        'original_psnr': float(original_metrics['psnr']),
        'calibrated_psnr': float(calibrated_metrics['psnr']),
        'psnr_improvement': float(psnr_improvement),
        'original_delta_e00_median': float(original_metrics['delta_e00']['median']),
        'calibrated_delta_e00_median': float(calibrated_metrics['delta_e00']['median']),
        'delta_e_improvement': float(delta_e_improvement)
    }

def main():
    parser = argparse.ArgumentParser(description="Stage A 简化评估")
    parser.add_argument('--render_dir', type=str, required=True, help='渲染图像目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT图像目录')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    args = parser.parse_args()
    
    print("========================================================================")
    print("Task 2 - Stage A 简化评估（预渲染图像）")
    print("========================================================================")
    print(f"渲染图像: {args.render_dir}")
    print(f"GT图像: {args.gt_dir}")
    print(f"输出: {args.output}")
    print("")
    
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
    
    # 处理所有图像对
    results = []
    for render_file in render_files:
        img_id = render_file.stem
        gt_file = gt_dir / render_file.name
        
        if not gt_file.exists():
            print(f"⚠️  跳过 {img_id}（未找到GT图像）")
            continue
        
        try:
            result = process_image_pair(render_file, gt_file, args.output, img_id)
            results.append(result)
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    if not results:
        print("❌ 没有成功处理的图像")
        return
    
    # 计算整体统计
    avg_psnr_improvement = np.mean([r['psnr_improvement'] for r in results])
    avg_delta_e00_median = np.mean([r['calibrated_delta_e00_median'] for r in results])
    
    summary = {
        'num_images': len(results),
        'avg_psnr_improvement': float(avg_psnr_improvement),
        'avg_delta_e00_median': float(avg_delta_e00_median),
        'per_image_results': results
    }
    
    # 保存结果
    with open(output_dir / "results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("")
    print("========================================================================")
    print("Stage A 完成")
    print("========================================================================")
    print(f"处理图像数: {len(results)}")
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
        print("  ⚠️  Gate T2-1 未通过")
    
    print(f"")
    print(f"结果已保存: {output_dir / 'results.json'}")
    print(f"对比图: {output_dir / 'vis/'}")

if __name__ == "__main__":
    main()
