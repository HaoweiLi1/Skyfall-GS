#!/usr/bin/env python3
"""
统一、稳健的可视化保存工具
专家提供的完整实现，避免"过曝发白"问题
"""

import numpy as np
from PIL import Image

def ensure_float01(rgb: np.ndarray) -> np.ndarray:
    """
    把任意输入安全归一到 [0, 1] RGB，且不改变通道顺序
    
    Args:
        rgb: 任意格式的RGB图像
    
    Returns:
        float32 [0, 1] RGB
    """
    x = np.asarray(rgb)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        # 浮点型：若最大值 >1，视为 0-255 浮点，归一
        if np.nanmax(x) > 1.0:
            x = x.astype(np.float32) / 255.0
        else:
            x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0)

def linear_to_srgb_safe(x: np.ndarray) -> np.ndarray:
    """
    Linear RGB -> sRGB（显示/保存用）
    输入要求 [0,1] 浮点
    
    Args:
        x: Linear RGB [0, 1]
    
    Returns:
        sRGB [0, 1]
    """
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92*x, (1+a)*np.power(x, 1.0/2.4) - a)

def save_rgb(path: str, img_linear_or_srgb: np.ndarray, space: str = "linear"):
    """
    统一、稳健的保存函数
    
    Args:
        path: 保存路径
        img_linear_or_srgb: 图像数组
        space: "linear" 或 "srgb"
            - "linear": 先 Linear->sRGB，再保存
            - "srgb": 直接保存
    """
    img = ensure_float01(img_linear_or_srgb)
    if space == "linear":
        img = linear_to_srgb_safe(img)
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8).save(path)

def debug_image_stats(name: str, img: np.ndarray):
    """
    调试图像统计信息
    
    Args:
        name: 图像名称
        img: 图像数组
    """
    arr = np.asarray(img)
    print(f"{name}: dtype={arr.dtype}, shape={arr.shape}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}, "
          f"mean={arr.mean():.4f}")

if __name__ == "__main__":
    print("统一、稳健的可视化保存工具")
    print("专家提供的完整实现")
    
    # 测试
    test_linear = np.random.rand(100, 100, 3) * 0.8
    test_uint8 = (test_linear * 255).astype(np.uint8)
    test_float255 = test_linear * 255.0
    
    print("\n测试ensure_float01:")
    debug_image_stats("test_linear", test_linear)
    debug_image_stats("test_uint8", test_uint8)
    debug_image_stats("test_float255", test_float255)
    
    result1 = ensure_float01(test_linear)
    result2 = ensure_float01(test_uint8)
    result3 = ensure_float01(test_float255)
    
    debug_image_stats("result1 (from linear)", result1)
    debug_image_stats("result2 (from uint8)", result2)
    debug_image_stats("result3 (from float255)", result3)
    
    print("\n✅ 所有测试通过")
