#!/usr/bin/env python3
"""
学习率调度器 - Warmup + 指数衰减
基于专家反馈实现
"""
import torch


def apply_lr_schedule(optimizer, iteration, warmup_iters=500, base_lr=2e-5, final_lr=5e-6, total_iters=3000):
    """
    应用学习率调度：Warmup + 指数衰减
    
    Args:
        optimizer: 优化器
        iteration: 当前迭代次数（0-based）
        warmup_iters: warmup迭代数
        base_lr: 基础学习率
        final_lr: 最终学习率
        total_iters: 总迭代数
    
    Returns:
        当前学习率
    """
    if iteration < warmup_iters:
        # Warmup阶段：线性增长
        scale = (iteration + 1) / warmup_iters
        lr_xyz = base_lr * scale
    else:
        # 指数衰减阶段
        t = (iteration - warmup_iters) / max(1, (total_iters - warmup_iters))
        lr_xyz = final_lr + (base_lr - final_lr) * (0.5 ** (t * 6))
    
    # 应用到优化器的xyz参数组
    for param_group in optimizer.param_groups:
        if param_group.get("name", "") == "xyz":
            param_group["lr"] = lr_xyz
    
    return lr_xyz
