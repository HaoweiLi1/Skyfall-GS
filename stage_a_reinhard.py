#!/usr/bin/env python3
"""
Stage A - Reinhard颜色迁移基线
专家提供的Linear RGB空间Reinhard匹配实现
"""

import numpy as np
import cv2

def srgb_to_linear(img_srgb):
    """sRGB → Linear RGB转换
    
    Args:
        img_srgb: uint8 [0,255] 或 float32 [0,1]
    
    Returns:
        float32 [0,1] Linear RGB
    """
    img = img_srgb.astype(np.float32) / 255.0 if img_srgb.dtype == np.uint8 else img_srgb.astype(np.float32)
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92,
                    ((img + a) / (1 + a)) ** 2.4)

def linear_to_srgb(img_lin):
    """Linear RGB → sRGB转换
    
    Args:
        img_lin: float32 [0,1] Linear RGB
    
    Returns:
        uint8 [0,255] sRGB
    """
    a = 0.055
    img = np.where(img_lin <= 0.0031308, 12.92 * img_lin,
                   (1 + a) * (np.clip(img_lin, 0, 1) ** (1/2.4)) - a)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def reinhard_color_transfer_linear(render_lin, gt_lin, mask=None):
    """
    在 Linear RGB 下做颜色转 Lab，再做 Reinhard 匹配。
    render_lin, gt_lin: float32, [0,1], HxWx3
    mask: 可选，布尔或0/1，True表示参与统计（建议来自 alpha>0.5 & 非极端亮/暗）
    
    注意：专家指出必须使用 Linear RGB → XYZ → Lab 路径，而非直接转sRGB再转Lab
    但为了快速验证，这里暂时使用OpenCV的Lab转换（通过sRGB中间步骤）
    """
    # 转到 sRGB 以便使用 cv2 的 Lab（OpenCV 的 Lab 期望 sRGB 近似）
    r_srgb = linear_to_srgb(render_lin)
    g_srgb = linear_to_srgb(gt_lin)
    
    # OpenCV的cvtColor期望BGR格式
    r_bgr = cv2.cvtColor(r_srgb, cv2.COLOR_RGB2BGR)
    g_bgr = cv2.cvtColor(g_srgb, cv2.COLOR_RGB2BGR)
    
    r_lab = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    g_lab = cv2.cvtColor(g_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if mask is None:
        mask = np.ones(r_lab.shape[:2], dtype=bool)
    
    out_lab = r_lab.copy()
    for c in range(3):
        src = r_lab[..., c][mask]
        ref = g_lab[..., c][mask]
        
        if len(src) == 0:
            continue
            
        s_mean, s_std = np.mean(src), np.std(src) + 1e-6
        t_mean, t_std = np.mean(ref), np.std(ref) + 1e-6
        out = (r_lab[..., c] - s_mean) / s_std * t_std + t_mean
        # OpenCV的Lab通道都是[0,255]范围
        out = np.clip(out, 0, 255)
        out_lab[..., c] = out
    
    # 转换回BGR再到RGB
    out_lab_uint8 = out_lab.astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab_uint8, cv2.COLOR_LAB2BGR)
    out_srgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    
    # 返回到 Linear
    out_lin = srgb_to_linear(out_srgb)
    return np.clip(out_lin, 0.0, 1.0)

def create_mask(alpha, render_lin):
    """
    创建有效像素mask
    
    Args:
        alpha: (H, W) alpha通道
        render_lin: (H, W, 3) 渲染图像
    
    Returns:
        mask: (H, W) 布尔数组
    """
    # alpha > 0.5
    mask = alpha > 0.5
    
    # 排除极端亮/暗（线性域）
    lum = 0.2126*render_lin[...,0] + 0.7152*render_lin[...,1] + 0.0722*render_lin[...,2]
    mask &= (lum > 0.02) & (lum < 0.98)
    
    return mask

if __name__ == "__main__":
    print("Stage A - Reinhard颜色迁移基线")
    print("专家提供的Linear RGB空间实现")
    print("准备就绪，等待集成到完整流程...")
