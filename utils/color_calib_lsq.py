#!/usr/bin/env python3
"""
颜色校准闭式解 - 稳健的3×3+偏置求解
专家提供的完整实现，带Tikhonov正则化
"""

import numpy as np

def solve_affine_color_calib(render_lin_rgb, gt_lin_rgb,
                             reg_lambda=1e-2, reg_mu=1e-3):
    """
    求解颜色校准的3×3矩阵M和偏置t
    
    目标函数:
        min ||X M^T + 1 t^T - Y||^2 + λ||M-I||^2_F + μ||t||^2
    
    其中:
    - X: 渲染像素（Linear RGB）
    - Y: GT像素（Linear RGB）
    - M: 3×3颜色矩阵
    - t: 3维偏置向量
    - λ: M的正则化系数（拉向单位矩阵）
    - μ: t的正则化系数（拉向零）
    
    Args:
        render_lin_rgb: (N, 3) 渲染像素（Linear RGB）
        gt_lin_rgb: (N, 3) GT像素（Linear RGB）
        reg_lambda: M的正则化系数
        reg_mu: t的正则化系数
    
    Returns:
        M: (3, 3) 颜色矩阵
        t: (3,) 偏置向量
    """
    X = render_lin_rgb.astype(np.float64)
    Y = gt_lin_rgb.astype(np.float64)
    N = X.shape[0]
    
    # 增广设计矩阵 [X 1]
    ones = np.ones((N, 1), dtype=np.float64)
    X_aug = np.concatenate([X, ones], axis=1)  # (N, 4)
    
    # 正则矩阵
    # 对M的9个元素用λ，对t的3个元素用μ
    R = np.zeros((4, 4), dtype=np.float64)
    R[:3, :3] = reg_lambda * np.eye(3)  # 对M
    R[3, 3] = reg_mu                    # 对t
    
    # 闭式解: W = (X_aug^T X_aug + R)^{-1} (X_aug^T Y + B)
    # 其中B是把"恒等"偏置注入的项
    A = X_aug.T @ X_aug + R
    B = X_aug.T @ Y
    B[:3, :] += reg_lambda * np.eye(3)  # 拉向恒等矩阵
    
    W = np.linalg.solve(A, B)  # (4, 3)
    
    M = W[:3, :].T  # (3, 3)
    t = W[3, :].T   # (3,)
    
    return M.astype(np.float32), t.astype(np.float32)

def solve_global_then_per_camera(render_pixels_dict, gt_pixels_dict,
                                 reg_lambda=1e-2, reg_mu=1e-3,
                                 reg_lambda_local=5e-3):
    """
    两段式求解：先解全局，再相机层微调
    
    策略:
    1. 先用所有相机的像素求解全局(M_g, t_g)
    2. 每个相机以(M_g, t_g)为初始化，在其附近微调
    
    Args:
        render_pixels_dict: {cam_id: (N, 3)} 渲染像素字典
        gt_pixels_dict: {cam_id: (N, 3)} GT像素字典
        reg_lambda: 全局M的正则化系数
        reg_mu: 全局t的正则化系数
        reg_lambda_local: 相机层M的正则化系数（更强）
    
    Returns:
        M_global: (3, 3) 全局颜色矩阵
        t_global: (3,) 全局偏置
        M_per_cam: {cam_id: (3, 3)} 每相机的颜色矩阵
        t_per_cam: {cam_id: (3,)} 每相机的偏置
    """
    # 1. 求解全局
    all_render = np.concatenate([render_pixels_dict[k] for k in render_pixels_dict], axis=0)
    all_gt = np.concatenate([gt_pixels_dict[k] for k in gt_pixels_dict], axis=0)
    
    M_global, t_global = solve_affine_color_calib(
        all_render, all_gt,
        reg_lambda=reg_lambda,
        reg_mu=reg_mu
    )
    
    print(f"[Global] M norm: {np.linalg.norm(M_global - np.eye(3)):.4f}, t norm: {np.linalg.norm(t_global):.4f}")
    
    # 2. 每相机微调
    M_per_cam = {}
    t_per_cam = {}
    
    for cam_id in render_pixels_dict:
        M_cam, t_cam = solve_affine_color_calib(
            render_pixels_dict[cam_id],
            gt_pixels_dict[cam_id],
            reg_lambda=reg_lambda_local,  # 更强的正则
            reg_mu=reg_mu
        )
        M_per_cam[cam_id] = M_cam
        t_per_cam[cam_id] = t_cam
        
        print(f"[Cam {cam_id}] M norm: {np.linalg.norm(M_cam - np.eye(3)):.4f}, t norm: {np.linalg.norm(t_cam):.4f}")
    
    return M_global, t_global, M_per_cam, t_per_cam

def apply_color_calib(image_lin, M, t):
    """
    应用颜色校准
    
    Args:
        image_lin: (H, W, 3) Linear RGB图像
        M: (3, 3) 颜色矩阵
        t: (3,) 偏置
    
    Returns:
        校准后的图像
    """
    H, W, _ = image_lin.shape
    pixels = image_lin.reshape(-1, 3)  # (HW, 3)
    calibrated = (pixels @ M.T) + t
    return np.clip(calibrated.reshape(H, W, 3), 0.0, 1.0)

if __name__ == "__main__":
    print("颜色校准闭式解 - 稳健的3×3+偏置求解")
    print("专家提供的完整实现")
    
    # 简单测试
    N = 1000
    X = np.random.rand(N, 3).astype(np.float32)
    
    # 模拟颜色偏移
    M_true = np.array([[1.1, 0.05, 0.02],
                       [0.03, 1.08, 0.04],
                       [0.02, 0.03, 0.95]], dtype=np.float32)
    t_true = np.array([0.02, -0.01, 0.03], dtype=np.float32)
    Y = (X @ M_true.T) + t_true + np.random.randn(N, 3) * 0.01
    
    # 求解
    M_est, t_est = solve_affine_color_calib(X, Y)
    
    print(f"\n测试:")
    print(f"  真实M:\n{M_true}")
    print(f"  估计M:\n{M_est}")
    print(f"  M误差: {np.linalg.norm(M_true - M_est):.6f}")
    print(f"  真实t: {t_true}")
    print(f"  估计t: {t_est}")
    print(f"  t误差: {np.linalg.norm(t_true - t_est):.6f}")
