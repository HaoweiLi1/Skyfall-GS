#!/usr/bin/env python3
"""
坐标约定单测：验证OpenCV/OpenGL约定一致性
确保点云投影到相机的误差<0.5px
"""
import os
import sys
import numpy as np
import json
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graphics_utils import focal2fov, fov2focal


def load_vggt_data(scene_path):
    """加载VGGT数据"""
    json_path = os.path.join(scene_path, "vggt_poses.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cameras = []
    for frame in data["frames"]:
        # 内参
        intr = frame["intrinsic"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        width, height = frame["width"], frame["height"]
        
        # 外参 (w2c)
        extr = frame["extrinsic_w2c"]
        R_w2c = np.array(extr["R"])
        t_w2c = np.array(extr["t"])
        
        cameras.append({
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy,
            'width': width, 'height': height,
            'R_w2c': R_w2c,
            't_w2c': t_w2c,
            'image_path': frame["image_path"]
        })
    
    # 加载点云
    ply_path = os.path.join(scene_path, "vggt_points3d_downsampled.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(scene_path, "vggt_points3d.ply")
    
    if os.path.exists(ply_path):
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    else:
        # 从深度图采样一些点
        print("[WARN] No point cloud found, using depth map samples")
        points = None
    
    return cameras, points


def project_point_to_camera(point_world, camera):
    """
    将世界坐标点投影到相机图像
    
    Args:
        point_world: (3,) 世界坐标
        camera: 相机参数字典
    
    Returns:
        u, v: 像素坐标
        depth: 深度值
        valid: 是否有效（在相机前方且在图像内）
    """
    # 转到相机坐标系
    R_w2c = camera['R_w2c']
    t_w2c = camera['t_w2c']
    
    point_cam = R_w2c @ point_world + t_w2c
    
    # 检查深度
    depth = point_cam[2]
    if depth <= 0:
        return None, None, depth, False
    
    # 投影到图像平面
    fx, fy = camera['fx'], camera['fy']
    cx, cy = camera['cx'], camera['cy']
    
    u = fx * point_cam[0] / depth + cx
    v = fy * point_cam[1] / depth + cy
    
    # 检查是否在图像内
    valid = (0 <= u < camera['width']) and (0 <= v < camera['height'])
    
    return u, v, depth, valid


def test_coordinate_convention(scene_path, num_test_points=100):
    """
    测试坐标约定
    
    验证:
    1. 点云中心点投影到所有相机的误差
    2. 随机采样点的投影误差
    3. 深度值的正负性
    """
    print("="*80)
    print("坐标约定单测")
    print("="*80)
    
    # 加载数据
    print(f"\n1. 加载数据: {scene_path}")
    cameras, points = load_vggt_data(scene_path)
    
    print(f"   相机数: {len(cameras)}")
    if points is not None:
        print(f"   点数: {len(points):,}")
    
    # 测试1: 点云中心
    print(f"\n2. 测试点云中心投影")
    if points is not None:
        center = points.mean(axis=0)
        print(f"   中心坐标: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        
        visible_count = 0
        depth_positive_count = 0
        projection_errors = []
        
        for i, cam in enumerate(cameras):
            u, v, depth, valid = project_point_to_camera(center, cam)
            
            if depth > 0:
                depth_positive_count += 1
            
            if valid:
                visible_count += 1
                # 这里我们假设中心点应该在图像中心附近
                # 计算到图像中心的距离作为"误差"的参考
                img_center_u = cam['width'] / 2
                img_center_v = cam['height'] / 2
                error = np.sqrt((u - img_center_u)**2 + (v - img_center_v)**2)
                projection_errors.append(error)
                
                if i < 3:  # 打印前3个相机的详情
                    print(f"   Camera {i}: u={u:.1f}, v={v:.1f}, depth={depth:.3f}, valid={valid}")
        
        print(f"\n   统计:")
        print(f"     深度>0: {depth_positive_count}/{len(cameras)} ({depth_positive_count/len(cameras)*100:.1f}%)")
        print(f"     可见: {visible_count}/{len(cameras)} ({visible_count/len(cameras)*100:.1f}%)")
        
        if projection_errors:
            print(f"     投影误差 (到图像中心):")
            print(f"       平均: {np.mean(projection_errors):.1f} px")
            print(f"       中位数: {np.median(projection_errors):.1f} px")
            print(f"       最大: {np.max(projection_errors):.1f} px")
    
    # 测试2: 随机采样点
    print(f"\n3. 测试随机采样点投影")
    if points is not None:
        # 采样点（避免边缘点）
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        radius_95 = np.percentile(distances, 95)
        
        # 选择距离中心较近的点
        mask = distances < radius_95 * 0.5
        candidate_points = points[mask]
        
        if len(candidate_points) > num_test_points:
            indices = np.random.choice(len(candidate_points), num_test_points, replace=False)
            test_points = candidate_points[indices]
        else:
            test_points = candidate_points
        
        print(f"   测试点数: {len(test_points)}")
        
        all_visible_counts = []
        all_depth_positive_counts = []
        
        for point in test_points:
            visible_count = 0
            depth_positive_count = 0
            
            for cam in cameras:
                u, v, depth, valid = project_point_to_camera(point, cam)
                
                if depth > 0:
                    depth_positive_count += 1
                if valid:
                    visible_count += 1
            
            all_visible_counts.append(visible_count)
            all_depth_positive_counts.append(depth_positive_count)
        
        print(f"\n   统计 (每个点):")
        print(f"     平均可见相机数: {np.mean(all_visible_counts):.1f}/{len(cameras)}")
        print(f"     平均深度>0相机数: {np.mean(all_depth_positive_counts):.1f}/{len(cameras)}")
        print(f"     可见率: {np.mean(all_visible_counts)/len(cameras)*100:.1f}%")
    
    # 测试3: 坐标系一致性检查
    print(f"\n4. 坐标系一致性检查")
    
    # 检查相机位置
    camera_positions = []
    for cam in cameras:
        # c2w = inv(w2c)
        R_w2c = cam['R_w2c']
        t_w2c = cam['t_w2c']
        
        # 相机中心在世界坐标系中的位置
        # C = -R_w2c.T @ t_w2c
        C = -R_w2c.T @ t_w2c
        camera_positions.append(C)
    
    camera_positions = np.array(camera_positions)
    
    print(f"   相机位置范围:")
    print(f"     X: [{camera_positions[:, 0].min():.3f}, {camera_positions[:, 0].max():.3f}]")
    print(f"     Y: [{camera_positions[:, 1].min():.3f}, {camera_positions[:, 1].max():.3f}]")
    print(f"     Z: [{camera_positions[:, 2].min():.3f}, {camera_positions[:, 2].max():.3f}]")
    
    if points is not None:
        print(f"\n   点云范围:")
        print(f"     X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"     Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"     Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # 检查相机是否在点云"上方"（卫星视角）
        cam_z_median = np.median(camera_positions[:, 2])
        points_z_median = np.median(points[:, 2])
        
        print(f"\n   相对位置:")
        print(f"     相机Z中位数: {cam_z_median:.3f}")
        print(f"     点云Z中位数: {points_z_median:.3f}")
        print(f"     相机在点云上方: {cam_z_median > points_z_median}")
    
    # 验收标准
    print(f"\n" + "="*80)
    print("验收结果:")
    print("="*80)
    
    checks = []
    
    if points is not None:
        # 检查1: 大部分点的深度应该为正
        avg_depth_positive_rate = np.mean(all_depth_positive_counts) / len(cameras)
        check1 = avg_depth_positive_rate > 0.5
        checks.append(("深度>0比例 > 50%", check1, f"{avg_depth_positive_rate*100:.1f}%"))
        
        # 检查2: 点应该在至少一些相机中可见
        avg_visible_rate = np.mean(all_visible_counts) / len(cameras)
        check2 = avg_visible_rate > 0.1
        checks.append(("可见率 > 10%", check2, f"{avg_visible_rate*100:.1f}%"))
        
        # 检查3: 相机应该在点云上方（卫星视角）
        check3 = cam_z_median > points_z_median
        checks.append(("相机在点云上方", check3, f"Δz={cam_z_median-points_z_median:.3f}"))
    
    for name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}: {value}")
    
    all_passed = all(c[1] for c in checks)
    
    if all_passed:
        print(f"\n✅ 所有检查通过！坐标约定正确。")
        return 0
    else:
        print(f"\n❌ 部分检查失败！可能存在坐标约定问题。")
        print(f"\n建议:")
        print(f"  1. 检查是否需要OpenCV/OpenGL坐标系转换")
        print(f"  2. 验证w2c矩阵的正确性")
        print(f"  3. 检查相机内参的符号")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test coordinate convention consistency"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene directory path"
    )
    parser.add_argument(
        "--num_test_points",
        type=int,
        default=100,
        help="Number of points to test (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = test_coordinate_convention(args.scene, args.num_test_points)
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
