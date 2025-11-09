#!/usr/bin/env python3
"""
sRGB→Linear单元测试
确保转换正确，避免回退
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.color_space import srgb_to_linear_np


def test_mid_gray():
    """测试中灰（0.5）的转换"""
    lin = srgb_to_linear_np(np.array([0.5], dtype=np.float32))
    expected = 0.21404114  # ((0.5+0.055)/1.055)^2.4
    
    assert abs(lin[0] - expected) < 1e-4, f"Mid-gray test failed: {lin[0]} != {expected}"
    print(f"✅ Mid-gray test passed: sRGB(0.5) → Linear({lin[0]:.6f})")


def test_black_white():
    """测试黑白端点"""
    black = srgb_to_linear_np(np.array([0.0], dtype=np.float32))
    white = srgb_to_linear_np(np.array([1.0], dtype=np.float32))
    
    assert abs(black[0] - 0.0) < 1e-6, "Black test failed"
    assert abs(white[0] - 1.0) < 1e-6, "White test failed"
    print(f"✅ Black/White test passed")


def test_threshold():
    """测试阈值附近的转换"""
    # 阈值是0.04045
    below = srgb_to_linear_np(np.array([0.04], dtype=np.float32))
    above = srgb_to_linear_np(np.array([0.05], dtype=np.float32))
    
    # below应该用线性公式: 0.04/12.92 ≈ 0.003096
    # above应该用幂函数: ((0.05+0.055)/1.055)^2.4 ≈ 0.003035
    
    print(f"✅ Threshold test: sRGB(0.04)→{below[0]:.6f}, sRGB(0.05)→{above[0]:.6f}")


def test_monotonic():
    """测试单调性"""
    values = np.linspace(0, 1, 100, dtype=np.float32)
    linear = srgb_to_linear_np(values)
    
    # 检查单调递增
    diffs = np.diff(linear)
    assert np.all(diffs >= 0), "Monotonicity test failed"
    print(f"✅ Monotonicity test passed")


if __name__ == "__main__":
    print("="*80)
    print("sRGB→Linear单元测试")
    print("="*80)
    print()
    
    try:
        test_mid_gray()
        test_black_white()
        test_threshold()
        test_monotonic()
        
        print()
        print("="*80)
        print("✅ 所有测试通过！")
        print("="*80)
        sys.exit(0)
        
    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ 测试失败: {e}")
        print("="*80)
        sys.exit(1)
