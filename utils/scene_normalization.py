"""
场景尺度归一化 - 统一缩放点云、相机、Near/Far

专家建议：卫星场景extent≈1570.85远超标准3DGS范围（<100），
必须统一归一化到~100量级，确保投影半径在1-2px范围。
"""

import torch
import numpy as np

def normalize_scene_scale(point_cloud, cameras, target_extent=100.0, verbose=True):
    """
    统一归一化场景尺度（完整相似变换）
    
    专家公式：
    - 点云：X' = s * (X - μ)
    - 相机（w2c）：t' = s * t + s * (R @ μ)
    - 相机（c2w）：t'_cw = s * (t_cw - μ)
    
    Args:
        point_cloud: BasicPointCloud对象
        cameras: 相机列表（train + test）
        target_extent: 目标场景extent（默认100.0）
        verbose: 是否打印详细信息
    
    Returns:
        scale_factor: 缩放因子
        normalized_point_cloud: 归一化后的点云
        normalized_cameras: 归一化后的相机列表
    """
    
    # 计算场景中心和extent
    points = point_cloud.points
    scene_center = points.mean(axis=0)  # μ
    current_extent = np.linalg.norm(points - scene_center, axis=1).max()
    
    # 计算缩放因子
    scale_factor = target_extent / current_extent
    
    if verbose:
        print("=" * 80)
        print("场景尺度归一化（完整相似变换）")
        print("=" * 80)
        print(f"  场景中心μ: [{scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f}]")
        print(f"  当前extent: {current_extent:.2f}")
        print(f"  目标extent: {target_extent:.2f}")
        print(f"  缩放因子s: {scale_factor:.6f}")
    
    # 1. 归一化点云：X' = s * (X - μ)
    normalized_points = scale_factor * (points - scene_center)
    
    # 创建归一化后的点云对象
    from utils.graphics_utils import BasicPointCloud
    normalized_point_cloud = BasicPointCloud(
        points=normalized_points,
        colors=point_cloud.colors,
        normals=point_cloud.normals
    )
    
    if verbose:
        print(f"  ✅ 点云归一化: {len(points)} points")
        print(f"     原始范围: [{points.min():.2f}, {points.max():.2f}]")
        print(f"     归一化后: [{normalized_points.min():.2f}, {normalized_points.max():.2f}]")
    
    # 2. 归一化相机（完整相似变换）
    normalized_cameras = []
    for cam in cameras:
        # 复制相机对象
        import copy
        norm_cam = copy.deepcopy(cam)
        
        # 专家公式：w2c格式 t' = s * t + s * (R @ μ)
        if hasattr(norm_cam, 'T') and hasattr(norm_cam, 'R'):
            R = norm_cam.R if isinstance(norm_cam.R, np.ndarray) else np.array(norm_cam.R)
            t = norm_cam.T if isinstance(norm_cam.T, np.ndarray) else np.array(norm_cam.T)
            
            # 完整相似变换
            t_new = scale_factor * t + scale_factor * (R @ scene_center)
            norm_cam.T = t_new
            
            # 验证：计算相机中心 C' = s * (C - μ)
            C_old = -R.T @ t
            C_new = -R.T @ t_new
            C_expected = scale_factor * (C_old - scene_center)
            
            # 断言检查（容差1e-4）
            if not np.allclose(C_new, C_expected, atol=1e-4):
                print(f"[WARN] Camera {cam.uid if hasattr(cam, 'uid') else '?'}: C' mismatch!")
                print(f"  C_new: {C_new}")
                print(f"  C_expected: {C_expected}")
        
        # 归一化Near/Far
        if hasattr(norm_cam, 'znear'):
            norm_cam.znear = norm_cam.znear * scale_factor
        if hasattr(norm_cam, 'zfar'):
            norm_cam.zfar = norm_cam.zfar * scale_factor
        
        normalized_cameras.append(norm_cam)
    
    if verbose:
        print(f"  ✅ 相机归一化: {len(cameras)} cameras")
        if len(cameras) > 0:
            # 统计Near/Far范围
            znears = [cam.znear for cam in normalized_cameras if hasattr(cam, 'znear')]
            zfars = [cam.zfar for cam in normalized_cameras if hasattr(cam, 'zfar')]
            if znears and zfars:
                print(f"     Near范围: [{min(znears):.2f}, {max(znears):.2f}]")
                print(f"     Far范围: [{min(zfars):.2f}, {max(zfars):.2f}]")
    
    if verbose:
        print("=" * 80)
    
    return scale_factor, normalized_point_cloud, normalized_cameras


def apply_scale_to_gaussians(gaussians, scale_factor, verbose=True):
    """
    将缩放因子应用到已初始化的高斯模型
    
    Args:
        gaussians: GaussianModel对象
        scale_factor: 缩放因子
        verbose: 是否打印详细信息
    """
    with torch.no_grad():
        # 缩放位置
        gaussians._xyz.data *= scale_factor
        
        # 缩放尺度（在log空间）
        log_scale_factor = torch.log(torch.tensor(scale_factor, device=gaussians._xyz.device))
        gaussians._scaling.data += log_scale_factor
        
        if verbose:
            print(f"  ✅ 高斯参数归一化:")
            print(f"     xyz范围: [{gaussians._xyz.min().item():.2f}, {gaussians._xyz.max().item():.2f}]")
            print(f"     scaling范围: [{gaussians.get_scaling.min().item():.4f}, {gaussians.get_scaling.max().item():.4f}]")


def verify_normalization(gaussians, cameras, target_extent=100.0, verbose=True):
    """
    验证归一化是否正确
    
    Args:
        gaussians: GaussianModel对象
        cameras: 相机列表
        target_extent: 目标extent
        verbose: 是否打印详细信息
    
    Returns:
        bool: 是否通过验证
    """
    with torch.no_grad():
        # 检查点云extent
        points = gaussians.get_xyz.cpu().numpy()
        current_extent = np.linalg.norm(points - points.mean(axis=0), axis=1).max()
        
        extent_ok = abs(current_extent - target_extent) / target_extent < 0.1  # 10%容差
        
        # 检查投影半径（简单估计）
        # 选择第一个相机，计算平均投影半径
        if len(cameras) > 0:
            cam = cameras[0]
            
            # 计算点到相机的距离
            if hasattr(cam, 'camera_center'):
                cam_center = cam.camera_center
            else:
                # 从w2c恢复相机中心
                R = cam.R
                T = cam.T
                cam_center = -R.T @ T
            
            cam_center_torch = torch.tensor(cam_center, device=gaussians._xyz.device, dtype=torch.float32)
            distances = torch.norm(gaussians._xyz - cam_center_torch, dim=1)
            
            # 估计投影半径（简化）
            # radius_2d ≈ scale_3d * focal / distance
            focal = cam.image_width / 2.0  # 简化估计
            scales = gaussians.get_scaling.mean(dim=1)
            proj_radii = scales * focal / (distances + 1e-6)
            
            avg_proj_radius = proj_radii.mean().item()
            proj_radius_ok = 0.5 < avg_proj_radius < 5.0  # 期望在0.5-5px范围
        else:
            proj_radius_ok = False
            avg_proj_radius = 0.0
        
        if verbose:
            print("=" * 80)
            print("归一化验证")
            print("=" * 80)
            print(f"  Extent: {current_extent:.2f} (目标: {target_extent:.2f}) {'✅' if extent_ok else '❌'}")
            print(f"  平均投影半径: {avg_proj_radius:.2f} px {'✅' if proj_radius_ok else '❌'}")
            print("=" * 80)
        
        return extent_ok and proj_radius_ok
