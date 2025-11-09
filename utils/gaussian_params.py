"""
Gaussian parameter utilities for safe numerical operations.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def inv_sigmoid(p: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid (logit) with numerical stability."""
    p = torch.clamp(p, 1e-6, 1. - 1e-6)
    return torch.log(p / (1. - p))


@torch.jit.script
def safe_opacity(opacity_like: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Ensure opacity is in valid range [eps, 1-eps].
    
    If input is already in [0,1], just clamp.
    Otherwise, apply sigmoid first then clamp.
    """
    if (opacity_like.min() >= 0.) and (opacity_like.max() <= 1.):
        x = opacity_like
    else:
        x = torch.sigmoid(opacity_like)
    return torch.clamp(x, eps, 1. - eps)


@torch.jit.script
def positive_scale_param(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Ensure scale/covariance parameters are positive."""
    return F.softplus(raw) + eps


def tensor_stats(x: torch.Tensor):
    """Get basic statistics of a tensor."""
    x = x.detach()
    return {
        "min": float(torch.nanmin(x).item()),
        "max": float(torch.nanmax(x).item()),
        "mean": float(torch.nanmean(x).item()),
        "isfinite": bool(torch.isfinite(x).all().item()),
    }


def print_stats(name: str, x: torch.Tensor, hist: bool = False):
    """Print tensor statistics for debugging."""
    s = tensor_stats(x)
    print(f"[STAT] {name}: min={s['min']:.6f} max={s['max']:.6f} mean={s['mean']:.6f} finite={s['isfinite']}")
    if hist:
        try:
            import numpy as np
            h, b = torch.histogram(x.nan_to_num(), bins=16)
            print(f"[STAT] {name} hist bins={b.cpu().numpy()}")
            print(f"[STAT] counts={h.cpu().numpy()}")
        except Exception:
            pass
