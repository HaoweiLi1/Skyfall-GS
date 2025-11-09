#!/usr/bin/env python3
"""
颜色度量工具
实现 CIEDE2000 色差计算和其他颜色相关度量
"""
import numpy as np
from utils.color_space import rgb_linear_to_lab


def delta_e2000(lab1, lab2, mask=None):
    """
    计算 CIEDE2000 色差
    
    Args:
        lab1: Lab 颜色数组，形状 (N, 3) 或 (H, W, 3)
        lab2: Lab 颜色数组，形状 (N, 3) 或 (H, W, 3)
        mask: 可选的掩码，形状 (N,) 或 (H, W)
    
    Returns:
        dict: {
            'mean': float,
            'median': float,
            'p95': float,
            'map': ndarray (如果输入是图像)
        }
    
    Reference:
        Sharma, G., Wu, W., & Dalal, E. N. (2005).
        The CIEDE2000 color-difference formula: Implementation notes,
        supplementary test data, and mathematical observations.
        Color Research & Application, 30(1), 21-30.
    """
    # 确保输入是 numpy 数组
    if not isinstance(lab1, np.ndarray):
        lab1 = np.array(lab1)
    if not isinstance(lab2, np.ndarray):
        lab2 = np.array(lab2)
    
    # 保存原始形状
    original_shape = lab1.shape
    is_image = len(original_shape) == 3
    
    # 展平为 (N, 3)
    if is_image:
        H, W = original_shape[:2]
        lab1 = lab1.reshape(-1, 3)
        lab2 = lab2.reshape(-1, 3)
        if mask is not None:
            mask = mask.reshape(-1)
    
    # 提取 L*, a*, b*
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    
    # 计算 C* (chroma)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    
    # 计算 G
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    
    # 计算 a'
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    # 计算 C'
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    C_bar_prime = (C1_prime + C2_prime) / 2.0
    
    # 计算 h'
    h1_prime = np.arctan2(b1, a1_prime) * 180 / np.pi
    h1_prime = np.where(h1_prime < 0, h1_prime + 360, h1_prime)
    
    h2_prime = np.arctan2(b2, a2_prime) * 180 / np.pi
    h2_prime = np.where(h2_prime < 0, h2_prime + 360, h2_prime)
    
    # 计算 ΔL', ΔC', ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    # 计算 Δh'
    delta_h_prime = np.zeros_like(h1_prime)
    mask_both_zero = (C1_prime == 0) | (C2_prime == 0)
    mask_diff_le_180 = np.abs(h2_prime - h1_prime) <= 180
    mask_diff_gt_180_h2_le_h1 = (np.abs(h2_prime - h1_prime) > 180) & (h2_prime <= h1_prime)
    mask_diff_gt_180_h2_gt_h1 = (np.abs(h2_prime - h1_prime) > 180) & (h2_prime > h1_prime)
    
    delta_h_prime[mask_both_zero] = 0
    delta_h_prime[mask_diff_le_180 & ~mask_both_zero] = h2_prime[mask_diff_le_180 & ~mask_both_zero] - h1_prime[mask_diff_le_180 & ~mask_both_zero]
    delta_h_prime[mask_diff_gt_180_h2_le_h1] = h2_prime[mask_diff_gt_180_h2_le_h1] - h1_prime[mask_diff_gt_180_h2_le_h1] + 360
    delta_h_prime[mask_diff_gt_180_h2_gt_h1] = h2_prime[mask_diff_gt_180_h2_gt_h1] - h1_prime[mask_diff_gt_180_h2_gt_h1] - 360
    
    # 计算 ΔH'
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(delta_h_prime * np.pi / 360)
    
    # 计算 L̄', C̄', H̄'
    L_bar_prime = (L1 + L2) / 2.0
    
    # 计算 H̄'
    H_bar_prime = np.zeros_like(h1_prime)
    mask_sum_ge_360 = (h1_prime + h2_prime) >= 360
    mask_sum_lt_360 = (h1_prime + h2_prime) < 360
    
    H_bar_prime[mask_both_zero] = h1_prime[mask_both_zero] + h2_prime[mask_both_zero]
    H_bar_prime[mask_diff_le_180 & ~mask_both_zero] = (h1_prime[mask_diff_le_180 & ~mask_both_zero] + h2_prime[mask_diff_le_180 & ~mask_both_zero]) / 2.0
    H_bar_prime[mask_diff_gt_180_h2_le_h1 & mask_sum_lt_360] = (h1_prime[mask_diff_gt_180_h2_le_h1 & mask_sum_lt_360] + h2_prime[mask_diff_gt_180_h2_le_h1 & mask_sum_lt_360] + 360) / 2.0
    H_bar_prime[mask_diff_gt_180_h2_gt_h1 & mask_sum_ge_360] = (h1_prime[mask_diff_gt_180_h2_gt_h1 & mask_sum_ge_360] + h2_prime[mask_diff_gt_180_h2_gt_h1 & mask_sum_ge_360] - 360) / 2.0
    
    # 计算 T
    T = (1 - 0.17 * np.cos((H_bar_prime - 30) * np.pi / 180) +
         0.24 * np.cos(2 * H_bar_prime * np.pi / 180) +
         0.32 * np.cos((3 * H_bar_prime + 6) * np.pi / 180) -
         0.20 * np.cos((4 * H_bar_prime - 63) * np.pi / 180))
    
    # 计算 S_L, S_C, S_H
    S_L = 1 + (0.015 * (L_bar_prime - 50)**2) / np.sqrt(20 + (L_bar_prime - 50)**2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    
    # 计算 R_T
    delta_theta = 30 * np.exp(-((H_bar_prime - 275) / 25)**2)
    R_C = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    R_T = -R_C * np.sin(2 * delta_theta * np.pi / 180)
    
    # 计算 ΔE00
    k_L = k_C = k_H = 1.0
    delta_E = np.sqrt(
        (delta_L_prime / (k_L * S_L))**2 +
        (delta_C_prime / (k_C * S_C))**2 +
        (delta_H_prime / (k_H * S_H))**2 +
        R_T * (delta_C_prime / (k_C * S_C)) * (delta_H_prime / (k_H * S_H))
    )
    
    # 应用掩码
    if mask is not None:
        delta_E = delta_E[mask]
    
    # 计算统计量
    result = {
        'mean': float(np.mean(delta_E)),
        'median': float(np.median(delta_E)),
        'p95': float(np.percentile(delta_E, 95)),
    }
    
    # 如果是图像，返回 map
    if is_image:
        if mask is not None:
            delta_E_map = np.zeros(H * W)
            delta_E_map[mask] = delta_E
            delta_E_map = delta_E_map.reshape(H, W)
        else:
            delta_E_map = delta_E.reshape(H, W)
        result['map'] = delta_E_map
    
    return result


def psnr_linear(pred_lin, gt_lin, mask=None, eps=1e-12):
    """
    在 Linear RGB 空间计算 PSNR
    
    Args:
        pred_lin: 预测图像，Linear RGB，形状 (H, W, 3) 或 (N, 3)
        gt_lin: GT 图像，Linear RGB，形状 (H, W, 3) 或 (N, 3)
        mask: 可选的掩码，形状 (H, W) 或 (N,)
        eps: 防止除零的小值
    
    Returns:
        float: PSNR 值（dB）
    """
    # 确保输入是 numpy 数组
    if not isinstance(pred_lin, np.ndarray):
        pred_lin = np.array(pred_lin)
    if not isinstance(gt_lin, np.ndarray):
        gt_lin = np.array(gt_lin)
    
    # 应用掩码
    if mask is not None:
        if len(pred_lin.shape) == 3:  # 图像
            pred_lin = pred_lin[mask]
            gt_lin = gt_lin[mask]
        else:  # 像素
            pred_lin = pred_lin[mask]
            gt_lin = gt_lin[mask]
    
    # 计算 MSE
    mse = np.mean((pred_lin - gt_lin) ** 2)
    
    # 计算 PSNR
    psnr = 10.0 * np.log10(1.0 / max(mse, eps))
    
    return float(psnr)


def compute_color_metrics(pred_lin, gt_lin, mask=None):
    """
    计算完整的颜色度量
    
    Args:
        pred_lin: 预测图像，Linear RGB，形状 (H, W, 3) 或 (N, 3)
        gt_lin: GT 图像，Linear RGB，形状 (H, W, 3) 或 (N, 3)
        mask: 可选的掩码，形状 (H, W) 或 (N,)
    
    Returns:
        dict: {
            'psnr': float,
            'delta_e00': {
                'mean': float,
                'median': float,
                'p95': float,
                'map': ndarray (如果输入是图像)
            }
        }
    """
    # 计算 PSNR
    psnr = psnr_linear(pred_lin, gt_lin, mask)
    
    # 转换到 Lab 空间
    if len(pred_lin.shape) == 3:  # 图像
        H, W = pred_lin.shape[:2]
        pred_pixels = pred_lin.reshape(-1, 3)
        gt_pixels = gt_lin.reshape(-1, 3)
        if mask is not None:
            mask_flat = mask.reshape(-1)
        else:
            mask_flat = None
    else:  # 像素
        pred_pixels = pred_lin
        gt_pixels = gt_lin
        mask_flat = mask
    
    # 转换到 Lab
    lab_pred = rgb_linear_to_lab(pred_pixels)
    lab_gt = rgb_linear_to_lab(gt_pixels)
    
    # 计算 ΔE00
    de_result = delta_e2000(lab_pred, lab_gt, mask_flat)
    
    return {
        'psnr': psnr,
        'delta_e00': de_result
    }
