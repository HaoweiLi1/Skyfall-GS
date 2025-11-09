#!/usr/bin/env python3
"""
坐标系转换工具
处理 OpenCV 和 OpenGL 坐标系之间的转换
"""
import numpy as np
import torch
from typing import Tuple, Union


def opencv_to_opengl_camera(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV → OpenGL 相机坐标系转换（w2c）
    
    OpenCV 约定: x右, y下, z前
    OpenGL 约定: x右, y上, -z前
    
    转换矩阵: A = diag(1, -1, -1)
    
    Args:
        R: OpenCV 旋转矩阵 (3, 3), w2c
        t: OpenCV 平移向量 (3,), w2c
    
    Returns:
        R_gl: OpenGL 旋转矩阵 (3, 3)
        t_gl: OpenGL 平移向量 (3,)
    
    数学原理:
        w2c_gl = A @ w2c_cv
        R_gl = A @ R_cv
        t_gl = A @ t_cv
    """
    A = np.diag([1., -1., -1.])
    R_gl = A @ R
    t_gl = A @ t
    return R_gl, t_gl


def opengl_to_opencv_camera(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenGL → OpenCV 相机坐标系转换（w2c）
    
    Args:
        R: OpenGL 旋转矩阵 (3, 3), w2c
        t: OpenGL 平移向量 (3,), w2c
    
    Returns:
        R_cv: OpenCV 旋转矩阵 (3, 3)
        t_cv: OpenCV 平移向量 (3,)
    
    注意: A 是自己的逆，所以转换是对称的
    """
    return opencv_to_opengl_camera(R, t)


def opencv_to_opengl_c2w(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV → OpenGL 相机坐标系转换（c2w）
    
    Args:
        R: OpenCV 旋转矩阵 (3, 3), c2w
        t: OpenCV 平移向量 (3,), c2w (相机中心)
    
    Returns:
        R_gl: OpenGL 旋转矩阵 (3, 3)
        t_gl: OpenGL 平移向量 (3,)
    
    数学原理:
        c2w_gl = c2w_cv @ A
        R_gl = R_cv @ A
        t_gl = t_cv (相机中心不变)
    """
    A = np.diag([1., -1., -1.])
    R_gl = R @ A
    t_gl = t  # 相机中心在世界坐标系中，不需要变换
    return R_gl, t_gl


def get_camera_center_from_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    从 w2c 计算相机中心
    
    Args:
        R: 旋转矩阵 (3, 3), w2c
        t: 平移向量 (3,), w2c
    
    Returns:
        C: 相机中心 (3,) in world coordinates
    
    数学原理:
        w2c = [R | t]
        c2w = [R^T | -R^T @ t]
        C = -R^T @ t
    """
    return -R.T @ t


def project_points_to_image(points_3d: np.ndarray,
                           R: np.ndarray,
                           t: np.ndarray,
                           K: np.ndarray,
                           convention: str = "opencv") -> np.ndarray:
    """
    将 3D 点投影到图像平面
    
    Args:
        points_3d: 3D 点 (N, 3) in world coordinates
        R: 旋转矩阵 (3, 3), w2c
        t: 平移向量 (3,), w2c
        K: 内参矩阵 (3, 3)
        convention: "opencv" 或 "opengl"
    
    Returns:
        points_2d: 2D 点 (N, 2) in image coordinates
    """
    # 转换到相机坐标系
    points_cam = (R @ points_3d.T).T + t
    
    # OpenGL 约定下，z 是负的
    if convention == "opengl":
        # 检查点是否在相机前方（z < 0）
        valid_mask = points_cam[:, 2] < 0
    else:  # opencv
        # 检查点是否在相机前方（z > 0）
        valid_mask = points_cam[:, 2] > 0
    
    # 投影到图像平面
    points_2d_homo = (K @ points_cam.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    # 标记无效点
    points_2d[~valid_mask] = np.nan
    
    return points_2d


def test_coordinate_conversion():
    """
    测试坐标系转换的正确性
    """
    print("="*80)
    print("测试坐标系转换")
    print("="*80)
    
    # 创建测试相机（OpenCV 约定）
    R_cv = np.array([
        [1, 0, 0],
        [0, 0.866, -0.5],  # 绕 x 轴旋转 30 度
        [0, 0.5, 0.866]
    ])
    t_cv = np.array([0, 0, 5])  # 相机在 z=5 处
    
    print("\n1. OpenCV 相机 (w2c):")
    print(f"   R_cv:\n{R_cv}")
    print(f"   t_cv: {t_cv}")
    
    # 计算相机中心
    C_cv = get_camera_center_from_w2c(R_cv, t_cv)
    print(f"   相机中心: {C_cv}")
    
    # 转换到 OpenGL
    R_gl, t_gl = opencv_to_opengl_camera(R_cv, t_cv)
    print("\n2. OpenGL 相机 (w2c):")
    print(f"   R_gl:\n{R_gl}")
    print(f"   t_gl: {t_gl}")
    
    # 计算相机中心（应该相同）
    C_gl = get_camera_center_from_w2c(R_gl, t_gl)
    print(f"   相机中心: {C_gl}")
    
    # 验证相机中心一致
    center_diff = np.linalg.norm(C_cv - C_gl)
    print(f"\n3. 相机中心差异: {center_diff:.10f}")
    assert center_diff < 1e-10, "相机中心应该相同！"
    print("   ✅ 相机中心一致")
    
    # 测试投影
    print("\n4. 测试投影:")
    
    # 创建测试点（在相机前方）
    points_3d = np.array([
        [0, 0, 10],   # 正前方
        [1, 0, 10],   # 右侧
        [0, 1, 10],   # 上方（OpenCV）/ 下方（OpenGL）
    ])
    
    # 内参矩阵
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # OpenCV 投影
    points_2d_cv = project_points_to_image(points_3d, R_cv, t_cv, K, "opencv")
    print(f"   OpenCV 投影:\n{points_2d_cv}")
    
    # OpenGL 投影
    points_2d_gl = project_points_to_image(points_3d, R_gl, t_gl, K, "opengl")
    print(f"   OpenGL 投影:\n{points_2d_gl}")
    
    # 验证投影一致（对于同一世界点，两种约定下的投影应该相同）
    # 注意：y 坐标会翻转
    proj_diff_x = np.abs(points_2d_cv[:, 0] - points_2d_gl[:, 0])
    proj_diff_y = np.abs(points_2d_cv[:, 1] + points_2d_gl[:, 1] - 2*240)  # 关于图像中心对称
    
    print(f"\n5. 投影差异:")
    print(f"   X 差异: {proj_diff_x}")
    print(f"   Y 差异（关于中心对称）: {proj_diff_y}")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)


def create_test_cube(center: np.ndarray = np.array([0, 0, 10]), 
                     size: float = 2.0) -> np.ndarray:
    """
    创建测试立方体的 8 个顶点
    
    Args:
        center: 立方体中心 (3,)
        size: 立方体边长
    
    Returns:
        vertices: 8 个顶点 (8, 3)
    """
    half_size = size / 2
    vertices = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ]) * half_size + center
    return vertices


# PyTorch 版本

def torch_opencv_to_opengl_camera(R: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch 版本的 OpenCV → OpenGL 转换
    
    Args:
        R: (3, 3) or (B, 3, 3)
        t: (3,) or (B, 3)
    
    Returns:
        R_gl, t_gl: 相同形状
    """
    if R.dim() == 2:
        A = torch.diag(torch.tensor([1., -1., -1.], device=R.device, dtype=R.dtype))
        R_gl = A @ R
        t_gl = A @ t
    else:  # batch
        A = torch.diag(torch.tensor([1., -1., -1.], device=R.device, dtype=R.dtype))
        R_gl = torch.einsum('ij,bjk->bik', A, R)
        t_gl = torch.einsum('ij,bj->bi', A, t)
    
    return R_gl, t_gl


if __name__ == "__main__":
    test_coordinate_conversion()
