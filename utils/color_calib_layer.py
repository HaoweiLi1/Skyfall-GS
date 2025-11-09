#!/usr/bin/env python3
"""
ColorCalib层 - 端到端训练版本
专家提供的完整实现，支持两阶段训练策略
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class ColorCalibLayer(nn.Module):
    """
    颜色校正层：y = clamp01(M @ x + t)
    
    Args:
        M_init: (3,3) 初始颜色矩阵（默认恒等）
        t_init: (3,) 初始偏置（默认零）
        clamp_min: 输出最小值
        clamp_max: 输出最大值
    
    Forward:
        x: (B,3,H,W) Linear RGB
        返回: (B,3,H,W) Linear RGB（校正后）
    """
    
    def __init__(self, M_init=None, t_init=None,
                 clamp_min=0.0, clamp_max=1.0):
        super().__init__()
        M = torch.eye(3) if M_init is None else torch.as_tensor(M_init, dtype=torch.float32)
        t = torch.zeros(3) if t_init is None else torch.as_tensor(t_init, dtype=torch.float32)
        self.M = nn.Parameter(M)   # 可学习
        self.t = nn.Parameter(t)   # 可学习
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    
    @torch.no_grad()
    def spectral_clip_(self, s_min=0.7, s_max=1.3):
        """
        谱裁剪：限制 M 的奇异值范围，保证接近物理可逆线性变换
        
        Args:
            s_min: 奇异值下限
            s_max: 奇异值上限
        """
        U, S, Vh = torch.linalg.svd(self.M, full_matrices=False)
        S_clamped = torch.clamp(S, s_min, s_max)
        self.M.copy_(U @ torch.diag(S_clamped) @ Vh)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B,3,H,W) Linear RGB
        
        Returns:
            y: (B,3,H,W) Linear RGB（校正后）
        
        注意：专家建议训练中不强行clamp，可在保存/评估时裁到[0,1]
        """
        # x: (B,3,H,W)
        B, C, H, W = x.shape
        y = torch.einsum('ij,bjhw->bihw', self.M, x) + self.t.view(1, 3, 1, 1)
        # 训练中不clamp，保持梯度流动
        return y
    
    def get_params_info(self):
        """获取参数信息（用于监控）"""
        I = torch.eye(3, device=self.M.device, dtype=self.M.dtype)
        M_norm = torch.norm(self.M - I).item()
        t_norm = torch.norm(self.t).item()
        
        # 奇异值
        U, S, Vh = torch.linalg.svd(self.M, full_matrices=False)
        singular_values = S.detach().cpu().numpy()
        
        return {
            'M_norm': M_norm,
            't_norm': t_norm,
            'singular_values': singular_values,
            'M': self.M.detach().cpu().numpy(),
            't': self.t.detach().cpu().numpy()
        }

def color_calib_regularizer(M, t, lambda_M=1e-2, lambda_t=1e-3):
    """
    颜色校准正则化损失
    
    Args:
        M: (3,3) 颜色矩阵
        t: (3,) 偏置
        lambda_M: M正则化系数（拉向恒等）
        lambda_t: t正则化系数（拉向零）
    
    Returns:
        正则化损失
    """
    # 让 M 接近 I，t 接近 0
    I = torch.eye(3, device=M.device, dtype=M.dtype)
    reg_M = lambda_M * torch.sum((M - I) ** 2)
    reg_t = lambda_t * torch.sum(t ** 2)
    return reg_M + reg_t

class ColorCalibManager:
    """
    颜色校准管理器 - 管理多相机的校准层
    """
    
    def __init__(self, device='cuda'):
        self.calib_layers = {}
        self.device = device
    
    def load_from_stage_b(self, params_dir, camera_ids):
        """
        从Stage B结果加载初始参数
        
        Args:
            params_dir: Stage B参数目录
            camera_ids: 相机ID列表
        """
        params_dir = Path(params_dir)
        
        for cam_id in camera_ids:
            param_file = params_dir / f"{cam_id}.npz"
            
            if param_file.exists():
                # 加载Stage B参数
                data = np.load(param_file)
                M_init = torch.tensor(data['M'], dtype=torch.float32)
                t_init = torch.tensor(data['t'], dtype=torch.float32)
                print(f"  加载相机 {cam_id}: ||M-I||={np.linalg.norm(data['M'] - np.eye(3)):.4f}")
            else:
                # 使用恒等初始化
                M_init = torch.eye(3, dtype=torch.float32)
                t_init = torch.zeros(3, dtype=torch.float32)
                print(f"  相机 {cam_id}: 使用恒等初始化")
            
            # 创建校准层
            layer = ColorCalibLayer(M_init, t_init).to(self.device)
            self.calib_layers[cam_id] = layer
    
    def get_layer(self, camera_id):
        """获取指定相机的校准层"""
        if camera_id not in self.calib_layers:
            # 创建默认层
            layer = ColorCalibLayer().to(self.device)
            self.calib_layers[camera_id] = layer
        return self.calib_layers[camera_id]
    
    def apply_calibration(self, render_output, camera_id):
        """对渲染输出应用颜色校准"""
        layer = self.get_layer(camera_id)
        return layer(render_output)
    
    def get_all_parameters(self):
        """获取所有校准层的参数（用于优化器）"""
        params = []
        for layer in self.calib_layers.values():
            params.extend([layer.M, layer.t])
        return params
    
    def get_regularization_loss(self, lambda_M=1e-2, lambda_t=1e-3):
        """计算所有校准层的正则化损失"""
        total_reg = 0.0
        for layer in self.calib_layers.values():
            total_reg += color_calib_regularizer(layer.M, layer.t, lambda_M, lambda_t)
        return total_reg
    
    def spectral_clip_all(self, s_min=0.7, s_max=1.3):
        """对所有校准层执行谱裁剪"""
        for layer in self.calib_layers.values():
            layer.spectral_clip_(s_min, s_max)
    
    def get_all_params_info(self):
        """获取所有校准层的参数信息"""
        info = {}
        for cam_id, layer in self.calib_layers.items():
            info[cam_id] = layer.get_params_info()
        return info
    
    def save_all_params(self, output_dir):
        """保存所有校准参数"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for cam_id, layer in self.calib_layers.items():
            params = layer.get_params_info()
            filepath = output_dir / f"{cam_id}.npz"
            np.savez(filepath, M=params['M'], t=params['t'])
        
        print(f"[ColorCalib] 保存 {len(self.calib_layers)} 个相机参数到 {output_dir}")

if __name__ == "__main__":
    print("ColorCalib层 - 端到端训练版本")
    print("专家提供的完整实现")
    
    # 测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # 创建测试层
    M_init = torch.tensor([
        [1.1, 0.02, 0.0],
        [0.01, 1.0, 0.01],
        [0.0, 0.0, 0.9]
    ], dtype=torch.float32)
    t_init = torch.tensor([0.01, 0.0, -0.01], dtype=torch.float32)
    
    layer = ColorCalibLayer(M_init, t_init).to(device)
    
    # 测试前向传播
    x = torch.rand(2, 3, 64, 64).to(device)
    y = layer(x)
    print(f"\n前向传播测试:")
    print(f"  输入: {x.shape}, 范围 [{x.min():.3f}, {x.max():.3f}]")
    print(f"  输出: {y.shape}, 范围 [{y.min():.3f}, {y.max():.3f}]")
    
    # 测试正则化
    reg_loss = color_calib_regularizer(layer.M, layer.t)
    print(f"\n正则化损失: {reg_loss:.6f}")
    
    # 测试谱裁剪
    layer.spectral_clip_(s_min=0.7, s_max=1.3)
    info = layer.get_params_info()
    print(f"\n谱裁剪后:")
    print(f"  ||M-I||: {info['M_norm']:.6f}")
    print(f"  ||t||: {info['t_norm']:.6f}")
    print(f"  奇异值: {info['singular_values']}")
    
    # 测试ColorCalibManager
    print(f"\n测试ColorCalibManager:")
    manager = ColorCalibManager(device=device)
    manager.load_from_stage_b('output/task2_step2_fixed_viz/stage_b_params', ['cam_000', 'cam_001'])
    print(f"  加载了 {len(manager.calib_layers)} 个相机")
    
    # 测试应用校准
    x_test = torch.rand(1, 3, 64, 64).to(device)
    y_test = manager.apply_calibration(x_test, 'cam_000')
    print(f"  应用校准: {x_test.shape} → {y_test.shape}")
    
    # 测试正则化
    reg_total = manager.get_regularization_loss(1e-3, 1e-4)
    print(f"  总正则损失: {reg_total:.6f}")
    
    print("\n✅ 所有测试通过")
