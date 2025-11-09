#!/usr/bin/env python3
"""
渲染Task 1训练好的模型，用于Task 2评估
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

# 添加路径
sys.path.append('.')

from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import Namespace

def load_model(model_path, source_path):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 创建参数
    args = Namespace(
        model_path=model_path,
        source_path=source_path,
        images="images",
        resolution=-1,
        white_background=False,
        eval=True,
        sh_degree=3,
        data_device="cuda"
    )
    
    # 创建Gaussian模型（使用训练时的参数）
    gaussians = GaussianModel(
        sh_degree=3,
        appearance_enabled=False,
        appearance_n_fourier_freqs=4,
        appearance_embedding_dim=32
    )
    
    # 加载场景
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)
    
    return scene, gaussians

def render_all_cameras(scene, gaussians, output_dir):
    """渲染所有训练相机"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取训练相机
    train_cameras = scene.getTrainCameras()
    print(f"找到 {len(train_cameras)} 个训练相机")
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    results = []
    
    for idx, camera in enumerate(tqdm(train_cameras, desc="渲染相机")):
        # 渲染
        with torch.no_grad():
            render_pkg = render(camera, gaussians, bg_color)
            rendering = render_pkg["render"]  # (3, H, W) Linear RGB
        
        # 转换为numpy (H, W, 3)
        image_linear = rendering.permute(1, 2, 0).cpu().numpy()
        
        # 保存为NPZ（Linear RGB）
        cam_name = camera.image_name
        npz_path = output_dir / f"{cam_name}_linear.npz"
        np.savez(npz_path, image=image_linear)
        
        # 同时保存sRGB PNG用于可视化
        from stage_a_reinhard import linear_to_srgb
        image_srgb = linear_to_srgb(image_linear)
        image_uint8 = (np.clip(image_srgb, 0, 1) * 255).astype(np.uint8)
        png_path = output_dir / f"{cam_name}.png"
        Image.fromarray(image_uint8).save(png_path)
        
        results.append({
            'camera_name': cam_name,
            'npz_path': str(npz_path),
            'png_path': str(png_path),
            'shape': image_linear.shape
        })
    
    print(f"\n渲染完成！")
    print(f"  Linear RGB (NPZ): {output_dir}/*_linear.npz")
    print(f"  sRGB PNG: {output_dir}/*.png")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="渲染Task 1模型")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--source_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    
    print("=" * 80)
    print("渲染Task 1训练模型")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"数据集: {args.source_path}")
    print(f"输出: {args.output}")
    print()
    
    # 加载模型
    scene, gaussians = load_model(args.model_path, args.source_path)
    
    # 渲染所有相机
    results = render_all_cameras(scene, gaussians, args.output)
    
    print(f"\n✅ 完成！渲染了 {len(results)} 个相机")

if __name__ == "__main__":
    main()
