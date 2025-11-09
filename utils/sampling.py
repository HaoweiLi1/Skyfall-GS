#!/usr/bin/env python3
"""
稳健采样工具 - α掩码 + 分层采样 + 分位裁剪
专家提供的完整实现，确保采样质量
"""

import numpy as np

def sample_pairs(render_lin, gt_lin, alpha, n=20000, seed=0):
    """
    稳健的像素对采样策略
    
    特点:
    1. 只取α > 0.5的像素
    2. 按亮度分层采样（暗/中/亮各1/3）
    3. 剔除top/bottom 1%亮度分位的异常值
    
    Args:
        render_lin: (3, H, W) 或 (H, W, 3) 渲染图像（Linear RGB）
        gt_lin: (3, H, W) 或 (H, W, 3) GT图像（Linear RGB）
        alpha: (H, W) alpha通道
        n: 采样数量
        seed: 随机种子
    
    Returns:
        R: (N, 3) 渲染像素
        G: (N, 3) GT像素
    """
    rng = np.random.default_rng(seed)
    
    # 统一格式到 (H, W, 3)
    if render_lin.shape[0] == 3:
        render_lin = render_lin.transpose(1, 2, 0)
    if gt_lin.shape[0] == 3:
        gt_lin = gt_lin.transpose(1, 2, 0)
    
    # α掩码
    mask = (alpha > 0.5).reshape(-1)
    
    # 展平
    R = render_lin.reshape(-1, 3)
    G = gt_lin.reshape(-1, 3)
    
    # 计算亮度（使用GT的亮度进行分层）
    L = (0.2126*G[:,0] + 0.7152*G[:,1] + 0.0722*G[:,2])
    
    # 应用mask
    R, G, L = R[mask], G[mask], L[mask]
    
    if len(R) == 0:
        return None, None
    
    # 分层索引（暗/中/亮）
    q1, q2 = np.quantile(L, [0.33, 0.66])
    idx_dark = np.where(L <= q1)[0]
    idx_mid = np.where((L > q1) & (L <= q2))[0]
    idx_bright = np.where(L > q2)[0]
    
    def pick(idx, k):
        if len(idx) == 0:
            return np.array([], dtype=int)
        k = min(k, len(idx))
        return rng.choice(idx, size=k, replace=False)
    
    # 每层采样1/3
    k = n // 3
    sel = np.concatenate([
        pick(idx_dark, k),
        pick(idx_mid, k),
        pick(idx_bright, n - 2*k)
    ])
    
    # 分位裁剪（剔除top/bottom 1%）
    norms = np.linalg.norm(G[sel], axis=1)
    lo, hi = np.quantile(norms, [0.01, 0.99])
    ok = (norms >= lo) & (norms <= hi)
    
    return R[sel][ok], G[sel][ok]

def create_robust_mask(alpha, render_lin, gt_lin=None):
    """
    创建稳健的像素mask
    
    规则:
    1. α > 0.5
    2. 排除极端亮/暗（亮度 ∈ [0.02, 0.98]）
    3. 可选：排除GT中的异常值
    
    Args:
        alpha: (H, W) alpha通道
        render_lin: (H, W, 3) 渲染图像
        gt_lin: (H, W, 3) 可选的GT图像
    
    Returns:
        mask: (H, W) 布尔mask
    """
    # α掩码
    mask = alpha > 0.5
    
    # 排除极端亮/暗（使用渲染图像的亮度）
    lum = 0.2126*render_lin[...,0] + 0.7152*render_lin[...,1] + 0.0722*render_lin[...,2]
    mask &= (lum > 0.02) & (lum < 0.98)
    
    # 可选：排除GT中的异常值
    if gt_lin is not None:
        gt_lum = 0.2126*gt_lin[...,0] + 0.7152*gt_lin[...,1] + 0.0722*gt_lin[...,2]
        mask &= (gt_lum > 0.02) & (gt_lum < 0.98)
    
    return mask

if __name__ == "__main__":
    print("稳健采样工具 - α掩码 + 分层采样 + 分位裁剪")
    print("专家提供的完整实现")
    
    # 简单测试
    H, W = 256, 256
    render_lin = np.random.rand(H, W, 3).astype(np.float32)
    gt_lin = np.random.rand(H, W, 3).astype(np.float32)
    alpha = np.random.rand(H, W).astype(np.float32)
    
    R, G = sample_pairs(render_lin, gt_lin, alpha, n=1000)
    
    if R is not None:
        print(f"\n测试:")
        print(f"  输入图像: {H}x{W}")
        print(f"  采样像素: {len(R)}")
        print(f"  渲染范围: [{R.min():.3f}, {R.max():.3f}]")
        print(f"  GT范围: [{G.min():.3f}, {G.max():.3f}]")
    else:
        print("\n测试: 无有效像素")
