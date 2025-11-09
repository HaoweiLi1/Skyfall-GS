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
from argparse import Namespace, ArgumentParser
from arguments import OptimizationParams
from utils.render_api import get_final_image_and_alpha, sanity_check_render_output

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
    
    # 加载场景（不加载点云，我们会从 checkpoint 加载）
    scene = Scene(args, gaussians, load_iteration=None, shuffle=False)
    
    # 从 checkpoint 加载
    checkpoint_path = os.path.join(model_path, "chkpnt30000.pth")
    if os.path.exists(checkpoint_path):
        print(f"从 checkpoint 加载: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
            model_params, iteration = checkpoint
            # 解包参数
            (active_sh_degree, xyz, features_dc, features_rest, scaling, rotation, opacity,
             embeddings, appearance_embeddings, appearance_mlp, max_radii2D,
             xyz_gradient_accum, denom, opt_dict, spatial_lr_scale) = model_params
            
            # 手动恢复参数（不恢复优化器）
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
            
            # 初始化 filter_3D
            n_pts = gaussians.get_xyz.shape[0]
            gaussians.filter_3D = torch.zeros(n_pts, 1, dtype=torch.float32, device='cuda')
            
            print(f"✅ 从 checkpoint 恢复，iteration={iteration}, 点数={gaussians.get_xyz.shape[0]}")
        else:
            raise ValueError("Checkpoint 格式错误")
    else:
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    
    return scene, gaussians

def render_all_cameras(scene, gaussians, output_dir):
    """渲染所有训练相机"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取训练相机
    train_cameras = scene.getTrainCameras()
    print(f"找到 {len(train_cameras)} 个训练相机")
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # 创建 pipeline 参数
    from argparse import Namespace
    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False
    )
    
    results = []
    
    for idx, camera in enumerate(tqdm(train_cameras, desc="渲染相机")):
        # 渲染
        with torch.no_grad():
            render_pkg = render(camera, gaussians, pipe, bg_color, kernel_size=0.1)
            
            # 使用统一的 API 获取最终图像（带守门员检查）
            comp_rgb, alpha = get_final_image_and_alpha(render_pkg)
            
            # Sanity Check（只对第一个相机）
            if idx == 0:
                sanity_check_render_output(comp_rgb, alpha, camera, iteration=0)
                
                # 同时检查 GT
                gt = camera.original_image
                print(f"  GT 图像: min={gt.min().item():.6f}, max={gt.max().item():.6f}, mean={gt.mean().item():.6f}")
            
            # 转换为numpy (H, W, 3)
            image_linear = comp_rgb.permute(1, 2, 0).cpu().numpy()
        
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
