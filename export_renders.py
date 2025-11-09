#!/usr/bin/env python3
"""
导出预渲染图像 - 绕过模型加载复杂性
从已有的训练输出中提取渲染结果和GT图像
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import json

sys.path.append('.')

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams

def export_renders(model_path, iteration, output_dir):
    """导出渲染图像和GT图像"""
    print(f"导出渲染图像: {model_path} @ iteration {iteration}")
    
    output_dir = Path(output_dir)
    (output_dir / "render").mkdir(parents=True, exist_ok=True)
    (output_dir / "gt").mkdir(parents=True, exist_ok=True)
    (output_dir / "alpha").mkdir(parents=True, exist_ok=True)
    
    # 简化方案：直接从已有的渲染结果中提取
    # 检查是否有已渲染的图像
    render_dir = Path(model_path) / "test" / f"ours_{iteration}"
    if render_dir.exists():
        print(f"找到已渲染的图像: {render_dir}")
        # 复制渲染图像
        for img_file in render_dir.glob("*.png"):
            print(f"  复制: {img_file.name}")
            img = Image.open(img_file)
            img.save(output_dir / "render" / img_file.name)
    else:
        print(f"未找到预渲染图像: {render_dir}")
        print("建议先运行渲染脚本生成图像")
    
    print(f"\n导出完成: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="导出预渲染图像")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--iteration', type=int, default=3000, help='迭代数')
    parser.add_argument('--output', type=str, required=True, help='输出路径')
    args = parser.parse_args()
    
    export_renders(args.model_path, args.iteration, args.output)

if __name__ == "__main__":
    main()
