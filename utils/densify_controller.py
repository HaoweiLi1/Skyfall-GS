#!/usr/bin/env python3
"""
Densification三重闸门控制器
基于专家反馈实现
"""
import numpy as np
import torch
from collections import deque


class DensifyController:
    """
    Densification三重闸门控制器
    
    三重闸门:
    1. 覆盖率门: 平均覆盖率 ≥ 25%
    2. 半径门: 平均屏幕半径 ∈ [0.8, 3.0] 像素
    3. 趋势门: PSNR斜率 ≥ +0.02 dB/100iter
    """
    
    def __init__(self, window_size=100, coverage_threshold=0.25, 
                 radius_range=(0.8, 3.0), psnr_slope_threshold=0.02):
        self.window_size = window_size
        self.coverage_threshold = coverage_threshold
        self.radius_range = radius_range
        self.psnr_slope_threshold = psnr_slope_threshold
        
        # 指标历史
        self.metrics_history = deque(maxlen=window_size)
        
        # 状态
        self.last_densify_iter = -1
        self.densify_cooldown = 500  # 失败后的冷却期
        
        print(f"[DensifyController] Initialized:")
        print(f"  Coverage threshold: {coverage_threshold:.1%}")
        print(f"  Radius range: {radius_range}")
        print(f"  PSNR slope threshold: {psnr_slope_threshold:.3f} dB/100iter")
        print(f"  Window size: {window_size}")
    
    def update_metrics(self, iteration, psnr, coverage_ratio, mean_radius_px):
        """
        更新指标历史
        
        Args:
            iteration: 当前迭代
            psnr: PSNR值 (dB)
            coverage_ratio: 像素覆盖率 [0, 1]
            mean_radius_px: 平均屏幕半径 (像素)
        """
        self.metrics_history.append({
            'iteration': iteration,
            'psnr': psnr,
            'coverage': coverage_ratio,
            'mean_radius': mean_radius_px
        })
    
    def can_densify(self, iteration, min_iter=800):
        """
        检查是否可以进行densification
        
        Args:
            iteration: 当前迭代
            min_iter: 最小允许densify的迭代数
        
        Returns:
            (can_densify, reason): (bool, str)
        """
        # 基础检查
        if iteration < min_iter:
            return False, f"Too early (iter {iteration} < {min_iter})"
        
        if len(self.metrics_history) < self.window_size:
            return False, f"Insufficient history ({len(self.metrics_history)} < {self.window_size})"
        
        # 冷却期检查
        if iteration - self.last_densify_iter < self.densify_cooldown:
            return False, f"In cooldown period"
        
        # 获取窗口内的指标
        window_metrics = list(self.metrics_history)
        
        # 1. 覆盖率门
        avg_coverage = np.mean([m['coverage'] for m in window_metrics])
        if avg_coverage < self.coverage_threshold:
            return False, f"Coverage too low ({avg_coverage:.1%} < {self.coverage_threshold:.1%})"
        
        # 2. 半径门
        avg_radius = np.mean([m['mean_radius'] for m in window_metrics])
        if not (self.radius_range[0] <= avg_radius <= self.radius_range[1]):
            return False, f"Radius out of range ({avg_radius:.2f} not in {self.radius_range})"
        
        # 3. 趋势门
        psnr_values = [m['psnr'] for m in window_metrics]
        psnr_slope = (psnr_values[-1] - psnr_values[0])  # dB per window_size iters
        psnr_slope_per_100 = psnr_slope * (100.0 / self.window_size)
        
        if psnr_slope_per_100 < self.psnr_slope_threshold:
            return False, f"PSNR slope too low ({psnr_slope_per_100:.3f} < {self.psnr_slope_threshold:.3f} dB/100iter)"
        
        return True, f"All gates passed (cov={avg_coverage:.1%}, rad={avg_radius:.2f}, slope={psnr_slope_per_100:.3f})"
    
    def record_densify_attempt(self, iteration, success=True):
        """
        记录densification尝试
        
        Args:
            iteration: 迭代次数
            success: 是否成功
        """
        self.last_densify_iter = iteration
        
        if success:
            print(f"[ITER {iteration}] Densification successful")
        else:
            print(f"[ITER {iteration}] Densification failed, entering cooldown")
            # 失败后延长冷却期
            self.densify_cooldown = min(1000, int(self.densify_cooldown * 1.2))
    
    def get_status(self):
        """
        获取当前状态摘要
        """
        if len(self.metrics_history) == 0:
            return "No metrics available"
        
        if len(self.metrics_history) >= self.window_size:
            window_metrics = list(self.metrics_history)
            avg_coverage = np.mean([m['coverage'] for m in window_metrics])
            avg_radius = np.mean([m['mean_radius'] for m in window_metrics])
            psnr_slope = (window_metrics[-1]['psnr'] - window_metrics[0]['psnr']) * (100.0 / self.window_size)
            
            return (f"Coverage: {avg_coverage:.1%} (≥{self.coverage_threshold:.1%}), "
                   f"Radius: {avg_radius:.2f} ({self.radius_range}), "
                   f"PSNR slope: {psnr_slope:.3f} (≥{self.psnr_slope_threshold:.3f}) dB/100iter")
        else:
            return f"Building history: {len(self.metrics_history)}/{self.window_size}"


def compute_coverage_ratio(rendered_image, threshold=0.01):
    """
    计算像素覆盖率
    
    Args:
        rendered_image: 渲染图像 tensor [C, H, W]
        threshold: 认为被覆盖的最小值
    
    Returns:
        coverage_ratio: 覆盖率 [0, 1]
    """
    if rendered_image.dim() == 3:
        # 取RGB的最大值
        max_vals = rendered_image.max(dim=0)[0]  # [H, W]
    else:
        max_vals = rendered_image
    
    covered_pixels = (max_vals > threshold).float().sum()
    total_pixels = max_vals.numel()
    
    return (covered_pixels / total_pixels).item()


def compute_mean_radius_px(radii):
    """
    计算平均屏幕半径（像素）
    
    Args:
        radii: 高斯的屏幕半径 tensor
    
    Returns:
        mean_radius: 平均半径（像素）
    """
    if radii is None or radii.numel() == 0:
        return 1.0
    
    # Convert to float if needed
    if radii.dtype in [torch.int32, torch.int64]:
        radii = radii.float()
    
    valid_radii = radii[radii > 0]
    if len(valid_radii) > 0:
        return valid_radii.mean().item()
    
    return 1.0
