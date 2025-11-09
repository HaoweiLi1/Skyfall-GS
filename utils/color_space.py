#!/usr/bin/env python3
"""
颜色空间转换工具 - Linear RGB ↔ XYZ ↔ Lab（D65标准）
专家提供的完整实现，确保数值精度和标准一致性
"""

import numpy as np

# sRGB linear ↔ XYZ (D65) 转换矩阵
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float64)

_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)

# D65 white point
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

def rgb_linear_to_xyz(rgb):
    """
    Linear RGB → XYZ (D65)
    
    Args:
        rgb: (..., 3) Linear RGB in [0, +inf)
    
    Returns:
        xyz: (..., 3) XYZ
    """
    return np.tensordot(rgb, _M_RGB2XYZ.T, axes=1)

def xyz_to_rgb_linear(xyz):
    """
    XYZ → Linear RGB (D65)
    
    Args:
        xyz: (..., 3) XYZ
    
    Returns:
        rgb: (..., 3) Linear RGB
    """
    return np.tensordot(xyz, _M_XYZ2RGB.T, axes=1)

def _f_lab(t):
    """
    CIE Lab f函数（标准实现）
    
    Args:
        t: 归一化的XYZ值
    
    Returns:
        f(t)
    """
    eps = 216/24389
    kappa = 24389/27
    t = np.maximum(t, 1e-12)  # 数值保护
    return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

def rgb_linear_to_lab(rgb):
    """
    Linear RGB → Lab (D65)
    
    这是正确的转换路径：Linear RGB → XYZ → Lab
    不要使用 Linear RGB → sRGB → Lab（会引入误差）
    
    Args:
        rgb: (..., 3) Linear RGB in [0, +inf)
    
    Returns:
        lab: (..., 3) Lab (L: [0, 100], a,b: [-128, 127])
    """
    xyz = rgb_linear_to_xyz(rgb)
    x = xyz[..., 0] / _Xn
    y = xyz[..., 1] / _Yn
    z = xyz[..., 2] / _Zn
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)

def _f_lab_inv(t):
    """
    CIE Lab f函数的逆函数
    
    Args:
        t: f(t)的值
    
    Returns:
        t
    """
    eps = 216/24389
    kappa = 24389/27
    delta = 6/29
    return np.where(t > delta, t**3, (116*t - 16) / kappa)

def lab_to_rgb_linear(lab):
    """
    Lab → Linear RGB (D65)
    
    Args:
        lab: (..., 3) Lab
    
    Returns:
        rgb: (..., 3) Linear RGB
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    x = _f_lab_inv(fx) * _Xn
    y = _f_lab_inv(fy) * _Yn
    z = _f_lab_inv(fz) * _Zn
    
    xyz = np.stack([x, y, z], axis=-1)
    return xyz_to_rgb_linear(xyz)

def srgb_to_linear_np(img_srgb):
    """
    sRGB → Linear RGB转换（numpy版本）
    
    Args:
        img_srgb: uint8 [0,255] 或 float32 [0,1]
    
    Returns:
        float32 [0,1] Linear RGB
    """
    img = img_srgb.astype(np.float32) / 255.0 if img_srgb.dtype == np.uint8 else img_srgb.astype(np.float32)
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92,
                    ((img + a) / (1 + a)) ** 2.4)

def linear_to_srgb_np(img_lin):
    """
    Linear RGB → sRGB转换（numpy版本）
    
    Args:
        img_lin: float32 [0,1] Linear RGB
    
    Returns:
        float32 [0,1] sRGB
    """
    a = 0.055
    img = np.where(img_lin <= 0.0031308, 12.92 * img_lin,
                   (1 + a) * (np.clip(img_lin, 0, 1) ** (1/2.4)) - a)
    return np.clip(img, 0, 1)

def process_image_for_training(image):
    """
    处理图像用于训练（Task 1修复后的正确路径）
    
    Args:
        image: PIL Image 或 numpy array (sRGB)
    
    Returns:
        numpy array (Linear RGB, float32)
    """
    if hasattr(image, 'convert'):  # PIL Image
        image = np.array(image.convert('RGB'))
    
    # sRGB → Linear RGB
    return srgb_to_linear_np(image)

if __name__ == "__main__":
    print("颜色空间转换工具 - Linear RGB ↔ XYZ ↔ Lab（D65标准）")
    print("专家提供的完整实现")
    
    # 简单测试
    rgb = np.array([0.5, 0.3, 0.2])
    xyz = rgb_linear_to_xyz(rgb)
    lab = rgb_linear_to_lab(rgb)
    rgb_back = lab_to_rgb_linear(lab)
    
    print(f"\n测试:")
    print(f"  RGB: {rgb}")
    print(f"  XYZ: {xyz}")
    print(f"  Lab: {lab}")
    print(f"  RGB (往返): {rgb_back}")
    print(f"  误差: {np.max(np.abs(rgb - rgb_back)):.6f}")
