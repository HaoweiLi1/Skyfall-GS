"""
自适应近远裁剪平面计算
按照专家建议：基于点云和相机分布自动计算合理的znear/zfar
"""
import numpy as np
import torch


def compute_adaptive_clip_planes(points_3d, camera_centers, percentile_near=10, percentile_far=99):
    """
    计算自适应的近远裁剪平面
    
    Args:
        points_3d: (N, 3) numpy array of 3D points
        camera_centers: (M, 3) numpy array of camera centers
        percentile_near: 近平面百分位数（默认10%）
        percentile_far: 远平面百分位数（默认99%）
    
    Returns:
        znear: 近裁剪平面距离
        zfar: 远裁剪平面距离
    """
    # 计算所有相机到所有点的距离
    all_distances = []
    
    for cam_center in camera_centers:
        # 计算相机到点云的距离
        dists = np.linalg.norm(points_3d - cam_center, axis=1)
        all_distances.extend(dists)
    
    all_distances = np.array(all_distances)
    
    # 使用稳健的百分位数
    p_near = np.percentile(all_distances, percentile_near)
    p_far = np.percentile(all_distances, percentile_far)
    
    # 应用安全边界
    znear = max(1.0, p_near * 0.5)  # 小一档，保证不过裁
    zfar = p_far * 1.5  # 大一档，保证覆盖
    
    # 约束 far/near 比例，避免投影精度问题
    max_ratio = 1e5
    if zfar / znear > max_ratio:
        # 调整使比例不超过max_ratio
        zfar = znear * max_ratio
    
    print(f"Adaptive clip planes computed:")
    print(f"  Distance distribution: p10={p_near:.2f}, p99={p_far:.2f}")
    print(f"  znear: {znear:.2f} (p{percentile_near} * 0.5)")
    print(f"  zfar: {zfar:.2f} (p{percentile_far} * 1.5)")
    print(f"  far/near ratio: {zfar/znear:.2f}")
    
    return znear, zfar


def compute_depth_range_from_cameras(points_3d, cameras):
    """
    从相机列表计算深度范围
    
    Args:
        points_3d: (N, 3) numpy array
        cameras: list of Camera objects with R, T attributes
    
    Returns:
        znear, zfar
    """
    all_depths = []
    
    for camera in cameras:
        # 转换到相机坐标系
        R = np.array(camera.R)
        T = np.array(camera.T)
        
        # points_cam = R @ points_3d.T + T[:, None]
        # 注意：R可能已经是转置的（glm约定）
        points_cam = (R @ points_3d.T).T + T
        
        # 深度是Z坐标
        depths = points_cam[:, 2]
        
        # 只考虑前方的点
        valid_depths = depths[depths > 0]
        if len(valid_depths) > 0:
            all_depths.extend(valid_depths)
    
    if len(all_depths) == 0:
        print("[WARN] No valid depths found, using default clip planes")
        return 0.01, 100.0
    
    all_depths = np.array(all_depths)
    
    # 使用百分位数
    p10 = np.percentile(all_depths, 10)
    p99 = np.percentile(all_depths, 99)
    
    znear = max(0.01, p10 * 0.5)
    zfar = p99 * 1.5
    
    # 约束比例
    if zfar / znear > 1e5:
        zfar = znear * 1e5
    
    print(f"Depth-based clip planes:")
    print(f"  Depth range: p10={p10:.2f}, p99={p99:.2f}")
    print(f"  znear: {znear:.2f}")
    print(f"  zfar: {zfar:.2f}")
    
    return znear, zfar
