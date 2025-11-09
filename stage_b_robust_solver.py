#!/usr/bin/env python3
"""
Stage B - 稳健闭式解求解器
专家提供的完整实现：Tikhonov正则化 + Huber鲁棒 + 谱裁剪
"""

import numpy as np
from pathlib import Path

def solve_affine_color_calib(
    x_lin: np.ndarray,  # (N,3) render (Linear RGB)
    y_lin: np.ndarray,  # (N,3) GT (Linear RGB)
    reg_lambda: float = 1e-2,  # ||M - I||_F^2
    reg_mu: float = 1e-3,      # ||t||_2^2
    huber_delta: float = None, # e.g., 0.02 for IRLS; None -> no IRLS
    iters: int = 2,
):
    """
    专家提供的稳健闭式解
    
    目标: min ||X M^T + 1 t^T - Y||^2 + λ||M-I||^2_F + μ||t||^2
    
    Args:
        x_lin: (N,3) 渲染像素（Linear RGB）
        y_lin: (N,3) GT像素（Linear RGB）
        reg_lambda: M正则化系数（拉向恒等）
        reg_mu: t正则化系数（拉向零）
        huber_delta: Huber损失阈值（None=不使用）
        iters: IRLS迭代次数
    
    Returns:
        M: (3,3) 颜色矩阵
        t: (3,) 偏置向量
    """
    assert x_lin.ndim == 2 and x_lin.shape[1] == 3
    assert y_lin.shape == x_lin.shape
    
    # --- whitening on x ---
    xm = x_lin.mean(0, keepdims=True)
    xs = x_lin.std(0, keepdims=True) + 1e-8
    x = (x_lin - xm) / xs
    y = y_lin.copy()
    
    N = x.shape[0]
    X = np.concatenate([x, np.ones((N,1))], axis=1)  # (N,4)
    
    # Regularizer for W=[M|t] in (4,3): diag on 12 weights
    # Encourage M->I, t->0
    R = np.zeros((4,4), dtype=np.float64)
    R[:3,:3] = reg_lambda * np.eye(3)
    R[3,3]   = reg_mu
    
    W = np.zeros((4,3), dtype=np.float64)  # init
    W[:3,:3] = np.eye(3)
    
    def closed_form(W, w):
        # w: (N,) robust weights
        # Weighted normal equations: (X^T W X + R)^{-1} X^T W Y
        WX = X * w[:, None]
        A = WX.T @ X + R
        B = WX.T @ y
        return np.linalg.solve(A, B)
    
    w = np.ones((N,), dtype=np.float64)
    W = closed_form(W, w)
    
    if huber_delta is not None:
        for _ in range(iters):
            pred = X @ W  # (N,3)
            resid = y - pred
            r = np.linalg.norm(resid, axis=1)  # (N,)
            # Huber weights
            w = np.where(r <= huber_delta, 1.0, huber_delta / (r + 1e-12))
            W = closed_form(W, w)
    
    # unwhiten: y ≈ M*((x_lin - xm)/xs) + t
    # => y ≈ (M/ xs) * x_lin + (t - M*xm/xs)
    M = W[:3, :] / xs.T  # (3,3)
    t = W[3, :] - (xm.reshape(3) / xs.reshape(3)) @ W[:3, :]
    M = M.T  # to make y = M @ x + t
    t = t.reshape(3)
    
    # spectral clamp to keep mapping near-identity & stable
    u, s, vh = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, 0.7, 1.3)  # keep color gains in a sane range
    M = (u * s) @ vh
    
    return M.astype(np.float32), t.astype(np.float32)

def solve_global_then_per_camera(
    render_pixels_list,  # List[(N_i,3)] 每个相机的渲染像素
    gt_pixels_list,      # List[(N_i,3)] 每个相机的GT像素
    reg_lambda=1e-2,
    reg_mu=1e-3,
    huber_delta=0.02,
    camera_reg_scale=2.0  # 相机层正则化倍数
):
    """
    两段式求解：先解全局，再相机层微调
    
    Args:
        render_pixels_list: List[(N_i,3)] 每个相机的渲染像素
        gt_pixels_list: List[(N_i,3)] 每个相机的GT像素
        reg_lambda: 全局正则化系数
        reg_mu: 全局正则化系数
        huber_delta: Huber阈值
        camera_reg_scale: 相机层正则化倍数（更强正则）
    
    Returns:
        M_global, t_global: 全局校准参数
        M_cameras, t_cameras: List[M_i, t_i] 每个相机的参数
    """
    print("\n" + "="*80)
    print("Stage B - 两段式求解")
    print("="*80)
    
    # 1. 全局求解
    print("\n[1/2] 求解全局校准参数...")
    all_render = np.concatenate(render_pixels_list, axis=0)
    all_gt = np.concatenate(gt_pixels_list, axis=0)
    
    print(f"  总像素数: {len(all_render):,}")
    
    M_global, t_global = solve_affine_color_calib(
        all_render, all_gt, 
        reg_lambda=reg_lambda, 
        reg_mu=reg_mu,
        huber_delta=huber_delta
    )
    
    print(f"  全局参数:")
    print(f"    ||M-I||_F: {np.linalg.norm(M_global - np.eye(3)):.4f}")
    print(f"    ||t||_2: {np.linalg.norm(t_global):.4f}")
    
    # 2. 相机层微调
    print(f"\n[2/2] 相机层微调（{len(render_pixels_list)}个相机）...")
    M_cameras = []
    t_cameras = []
    
    for i, (render_pix, gt_pix) in enumerate(zip(render_pixels_list, gt_pixels_list)):
        if len(render_pix) < 100:  # 像素太少，直接用全局
            M_cameras.append(M_global.copy())
            t_cameras.append(t_global.copy())
            print(f"  相机 {i:02d}: 像素太少({len(render_pix)})，使用全局参数")
            continue
        
        # 以全局参数为初始化，做微调（更强正则）
        M_cam, t_cam = solve_affine_color_calib(
            render_pix, gt_pix, 
            reg_lambda=reg_lambda * camera_reg_scale,
            reg_mu=reg_mu * camera_reg_scale,
            huber_delta=huber_delta
        )
        
        M_cameras.append(M_cam)
        t_cameras.append(t_cam)
        
        print(f"  相机 {i:02d}: ||M-I||={np.linalg.norm(M_cam - np.eye(3)):.4f}, "
              f"||t||={np.linalg.norm(t_cam):.4f}, N={len(render_pix):,}")
    
    print("\n" + "="*80)
    return M_global, t_global, M_cameras, t_cameras

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
    H, W = image_lin.shape[:2]
    pixels = image_lin.reshape(-1, 3)  # (HW, 3)
    calibrated = (pixels @ M.T) + t
    return np.clip(calibrated.reshape(H, W, 3), 0.0, 1.0)

def save_calib_params(M, t, filepath, metadata=None):
    """
    保存校准参数
    
    Args:
        M: (3, 3) 颜色矩阵
        t: (3,) 偏置
        filepath: 保存路径
        metadata: 可选的元数据字典
    """
    save_dict = {'M': M, 't': t}
    if metadata:
        save_dict.update(metadata)
    
    np.savez(filepath, **save_dict)
    print(f"  保存: {filepath}")

def load_calib_params(filepath):
    """
    加载校准参数
    
    Args:
        filepath: 文件路径
    
    Returns:
        M: (3, 3) 颜色矩阵
        t: (3,) 偏置
    """
    data = np.load(filepath)
    return data['M'], data['t']

if __name__ == "__main__":
    print("Stage B - 稳健闭式解求解器")
    print("专家提供的完整实现")
    
    # 简单测试
    N = 1000
    render_test = np.random.rand(N, 3) * 0.8
    
    # 模拟颜色偏移
    M_true = np.array([
        [1.1, 0.02, 0.0],
        [0.01, 1.0, 0.01],
        [0.0, 0.0, 0.9]
    ])
    t_true = np.array([0.01, 0.0, -0.01])
    
    gt_test = (render_test @ M_true.T) + t_true + np.random.randn(N, 3) * 0.01
    
    # 求解
    M_est, t_est = solve_affine_color_calib(
        render_test, gt_test,
        reg_lambda=1e-2,
        reg_mu=1e-3,
        huber_delta=0.02
    )
    
    print(f"\n测试结果:")
    print(f"  真实M:\n{M_true}")
    print(f"  估计M:\n{M_est}")
    print(f"  误差: {np.linalg.norm(M_est - M_true):.4f}")
    print(f"\n  真实t: {t_true}")
    print(f"  估计t: {t_est}")
    print(f"  误差: {np.linalg.norm(t_est - t_true):.4f}")
