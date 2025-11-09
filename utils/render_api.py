"""
统一的渲染输出 API
防止拿错渲染缓冲区（radiance accumulation vs final composited image）
"""
import torch


def get_final_image_and_alpha(out: dict):
    """
    从渲染器输出中提取最终图像和 alpha
    
    Args:
        out: 渲染器返回的字典
        
    Returns:
        (image, alpha) 其中 image 为已经 premultiplied & composited 的最终图像，
        范围应在 [0,1] (Linear RGB)
        
    Raises:
        KeyError: 如果找不到最终图像键
        RuntimeError: 如果图像范围异常（可能拿错了缓冲区）
    """
    # 常见最终图像键的优先级（不同分支命名不同）
    img = None
    for k in ["image", "render", "rgb", "composited"]:
        if k in out:
            img = out[k]
            break
    
    if img is None:
        raise KeyError(
            f"[RenderAPI] Cannot find final image key in outputs. "
            f"Available keys: {list(out.keys())}"
        )
    
    # 常见 alpha/accum 键
    alpha = None
    for k in ["alpha", "render_alpha", "accumulation", "acc"]:
        if k in out:
            alpha = out[k]
            break
    
    if alpha is None:
        # 没 alpha 也可以工作，但要明确提示
        print("[RenderAPI] WARN: alpha/accumulation not found; proceeding without alpha map.")
    
    # 类型转换
    if img.dtype != torch.float32:
        img = img.float()
    if alpha is not None and alpha.dtype != torch.float32:
        alpha = alpha.float()
    
    # 强规则：若 img 的 99.9 分位数 > 1.2，直接报错
    # 这说明你拿到的不是最终 composited image，而是某个累加缓冲
    with torch.no_grad():
        img_clamped = img.clamp_min(0)
        if img_clamped.numel() > 0:
            q = torch.quantile(img_clamped.flatten(), 0.999)
            if q.item() > 1.2:
                # 打印诊断信息
                print(f"\n[RenderAPI] ❌ 渲染输出异常！")
                print(f"  p99.9 = {q.item():.3f} (应该 <= 1.0)")
                print(f"  min = {img.min().item():.3f}")
                print(f"  max = {img.max().item():.3f}")
                print(f"  mean = {img.mean().item():.3f}")
                print(f"  可用的键: {list(out.keys())}")
                print(f"\n  你可能拿到了错误的缓冲区（如 radiance accumulation）")
                print(f"  而不是最终的 composited image！")
                
                raise RuntimeError(
                    f"[RenderAPI] Final image buffer out-of-range (p99.9={q.item():.3f} > 1.2). "
                    f"You're likely reading a non-composited buffer. Keys={list(out.keys())}"
                )
    
    return img, alpha


def sanity_check_render_output(img, alpha, camera=None, iteration=0):
    """
    Sanity Check：验证渲染输出的合理性
    
    Args:
        img: 渲染图像 (C,H,W) 或 (H,W,C)
        alpha: Alpha 通道
        camera: 相机对象（可选，用于打印信息）
        iteration: 迭代次数（可选）
    """
    with torch.no_grad():
        rgb_min = img.min().item()
        rgb_max = img.max().item()
        rgb_mean = img.mean().item()
        
        alpha_mean = alpha.mean().item() if alpha is not None else -1.0
        
        cam_name = camera.image_name if camera is not None else "unknown"
        
        print(f"\n[Sanity Check] Iter {iteration}, Camera: {cam_name}")
        print(f"  RGB: min={rgb_min:.6f}, max={rgb_max:.6f}, mean={rgb_mean:.6f}")
        print(f"  Alpha: mean={alpha_mean:.6f}")
        
        # 硬闸门
        if rgb_max > 1.02 or rgb_min < -0.02:
            raise RuntimeError(
                f"❌ Sanity failed: RGB out of range [{rgb_min:.3g}, {rgb_max:.3g}]. "
                f"Expected [0, 1]."
            )
        
        if rgb_max < 1e-3:
            raise RuntimeError(
                f"❌ Sanity failed: RGB max={rgb_max:.2e} too small (likely all black). "
                f"Check checkpoint restore / camera-point cloud alignment."
            )
        
        if alpha_mean > 0 and not (0.05 <= alpha_mean <= 0.99):
            print(f"  ⚠️  Alpha mean={alpha_mean:.3f} 异常（但可能正常）")
        
        print(f"  ✅ Sanity Check 通过！")
        
        return rgb_max, alpha_mean
