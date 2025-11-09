#!/usr/bin/env python3
"""
创建模拟的渲染图像（添加颜色偏移）
用于测试Task 2的颜色校准效果
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

from stage_a_reinhard import srgb_to_linear, linear_to_srgb

def add_color_shift(image_linear, shift_type='warm'):
    """
    添加颜色偏移到Linear RGB图像
    
    Args:
        image_linear: (H, W, 3) Linear RGB
        shift_type: 'warm', 'cool', 'green', 'magenta'
    
    Returns:
        shifted image (Linear RGB)
    """
    if shift_type == 'warm':
        # 暖色调：增加红色，减少蓝色
        M = np.array([
            [1.15, 0.05, 0.00],
            [0.02, 1.00, 0.02],
            [0.00, 0.00, 0.85]
        ])
        t = np.array([0.02, 0.00, -0.02])
    elif shift_type == 'cool':
        # 冷色调：减少红色，增加蓝色
        M = np.array([
            [0.85, 0.00, 0.05],
            [0.00, 1.00, 0.00],
            [0.05, 0.05, 1.15]
        ])
        t = np.array([-0.02, 0.00, 0.02])
    elif shift_type == 'green':
        # 绿色偏移
        M = np.array([
            [0.95, 0.00, 0.00],
            [0.05, 1.10, 0.00],
            [0.00, 0.00, 0.95]
        ])
        t = np.array([0.00, 0.02, 0.00])
    elif shift_type == 'magenta':
        # 洋红偏移
        M = np.array([
            [1.05, 0.00, 0.05],
            [0.00, 0.95, 0.00],
            [0.00, 0.00, 1.05]
        ])
        t = np.array([0.01, -0.01, 0.01])
    else:
        # 默认：轻微暖色
        M = np.array([
            [1.10, 0.03, 0.00],
            [0.01, 1.00, 0.01],
            [0.00, 0.00, 0.90]
        ])
        t = np.array([0.01, 0.00, -0.01])
    
    # 应用变换
    H, W = image_linear.shape[:2]
    pixels = image_linear.reshape(-1, 3)
    shifted_pixels = (pixels @ M.T) + t
    shifted = shifted_pixels.reshape(H, W, 3)
    
    return np.clip(shifted, 0, 1)

def process_image(input_path, output_path, shift_type='warm'):
    """处理单张图像"""
    # 加载图像
    img = Image.open(input_path).convert('RGB')
    img_srgb = np.array(img).astype(np.float32) / 255.0
    
    # 转换到Linear RGB
    img_linear = srgb_to_linear(img_srgb)
    
    # 添加颜色偏移
    shifted_linear = add_color_shift(img_linear, shift_type)
    
    # 转换回sRGB并保存
    shifted_srgb = linear_to_srgb(shifted_linear)
    Image.fromarray(shifted_srgb).save(output_path)

def main():
    parser = argparse.ArgumentParser(description="创建模拟的渲染图像")
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录（GT）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录（模拟渲染）')
    parser.add_argument('--shift_type', type=str, default='warm', 
                       choices=['warm', 'cool', 'green', 'magenta'],
                       help='颜色偏移类型')
    parser.add_argument('--num_images', type=int, default=None, help='处理图像数量（None=全部）')
    args = parser.parse_args()
    
    print("=" * 80)
    print("创建模拟的渲染图像（添加颜色偏移）")
    print("=" * 80)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"颜色偏移: {args.shift_type}")
    print()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有PNG图像
    image_files = sorted(input_dir.glob("*.png"))
    
    if args.num_images:
        image_files = image_files[:args.num_images]
    
    print(f"找到 {len(image_files)} 个图像")
    print()
    
    # 处理所有图像
    for img_file in tqdm(image_files, desc="处理图像"):
        output_path = output_dir / img_file.name
        process_image(img_file, output_path, args.shift_type)
    
    print()
    print(f"✅ 完成！处理了 {len(image_files)} 个图像")
    print(f"输出: {output_dir}")

if __name__ == "__main__":
    main()
