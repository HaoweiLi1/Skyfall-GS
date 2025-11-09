#!/usr/bin/env python3
"""提取 Task 1 的 Sim(3) 归一化参数"""

import numpy as np
import json
import os
from plyfile import PlyData

# 读取原始点云
ply_path = 'data/datasets_JAX/JAX_068/points3D.ply'
print(f"读取点云: {ply_path}")
plydata = PlyData.read(ply_path)
vertices = plydata['vertex']
positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

# 计算归一化参数（与 Scene 初始化逻辑一致）
# 注意：scene_normalization.py 使用 max_dist 作为 current_extent，不是 max_dist * 2
center = positions.mean(axis=0)
max_dist = np.linalg.norm(positions - center, axis=1).max()
current_extent = max_dist  # 这里是关键！
target_extent = 100.0
scale = target_extent / current_extent

print(f'\n原始点云统计:')
print(f'  点数: {len(positions)}')
print(f'  中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]')
print(f'  current_extent (max_dist): {current_extent:.2f}')
print(f'  归一化scale: {scale:.6f}')
print(f'  归一化后extent: {target_extent:.2f}')

# 保存到 JSON
sim3 = {
    'center': center.tolist(),
    'scale': float(scale),
    'extent_before': float(current_extent),
    'extent_after': float(target_extent)
}

output_path = 'outputs/JAX/JAX_068/chkpnt30000.sim3.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(sim3, f, indent=2)

print(f'\n✅ Sim(3) 参数已保存到: {output_path}')
print(f'\n内容:')
print(json.dumps(sim3, indent=2))
