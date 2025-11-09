#!/usr/bin/env python3
"""
修复下采样点云颜色缺失问题
确保下采样后的PLY文件包含RGB颜色信息
"""
import open3d as o3d
import numpy as np
import sys
import argparse
from pathlib import Path


def fix_downsample_with_color(input_path, output_path, voxel_size, verify=True):
    """
    下采样点云并确保保留颜色信息
    
    Args:
        input_path: 输入PLY路径
        output_path: 输出PLY路径
        voxel_size: 体素大小
        verify: 是否验证输出文件
    """
    print("="*80)
    print("修复下采样点云颜色缺失")
    print("="*80)
    
    # 1. 加载原始点云
    print(f"\n1. 加载原始点云: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    
    if not pcd.has_points():
        raise ValueError("Empty point cloud!")
    
    print(f"   点数: {len(pcd.points):,}")
    print(f"   有颜色: {pcd.has_colors()}")
    
    # 2. 检查并添加颜色（如果缺失）
    if not pcd.has_colors():
        print("\n⚠️  原始点云没有颜色，添加灰色兜底...")
        colors = np.full((len(pcd.points), 3), 0.5, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("   ✅ 已添加灰色")
    else:
        colors = np.asarray(pcd.colors)
        print(f"   颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
    
    # 3. 体素下采样
    print(f"\n2. 体素下采样 (voxel_size={voxel_size})")
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    print(f"   下采样后点数: {len(pcd_ds.points):,}")
    print(f"   压缩比: {len(pcd.points) / len(pcd_ds.points):.2f}x")
    
    # 4. 验证下采样后仍有颜色
    if not pcd_ds.has_colors():
        print("\n⚠️  下采样后丢失颜色，从原点云回填...")
        # 使用KD树从原点云找最近邻的颜色
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        colors_ds = []
        
        for i, pt in enumerate(pcd_ds.points):
            if i % 10000 == 0:
                print(f"   回填进度: {i}/{len(pcd_ds.points)}")
            _, idx, _ = pcd_tree.search_knn_vector_3d(pt, 1)
            colors_ds.append(np.asarray(pcd.colors)[idx[0]])
        
        pcd_ds.colors = o3d.utility.Vector3dVector(np.asarray(colors_ds))
        print("   ✅ 颜色回填完成")
    else:
        colors_ds = np.asarray(pcd_ds.colors)
        print(f"   ✅ 下采样保留了颜色")
        print(f"   颜色范围: [{colors_ds.min():.3f}, {colors_ds.max():.3f}]")
    
    # 5. 保存
    print(f"\n3. 保存到: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    ok = o3d.io.write_point_cloud(
        output_path, 
        pcd_ds,
        write_ascii=False,  # 二进制格式更小
        print_progress=True
    )
    
    if not ok:
        raise RuntimeError("Write failed!")
    
    print("   ✅ 保存成功")
    
    # 6. 验证读回
    if verify:
        print(f"\n4. 验证读回...")
        pcd_chk = o3d.io.read_point_cloud(output_path)
        
        checks = {
            "有点": pcd_chk.has_points(),
            "有颜色": pcd_chk.has_colors(),
            "点数匹配": len(pcd_chk.points) == len(pcd_ds.points),
        }
        
        print("   验证结果:")
        for name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"     {status} {name}: {result}")
        
        if not all(checks.values()):
            raise RuntimeError("Verification failed!")
        
        # 检查PLY文件头
        print(f"\n5. 检查PLY文件头...")
        with open(output_path, 'rb') as f:
            header_lines = []
            for _ in range(20):  # 读前20行
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
        
        has_rgb = any('red' in line or 'green' in line or 'blue' in line 
                     for line in header_lines)
        
        print("   PLY头部:")
        for line in header_lines[:10]:
            print(f"     {line}")
        
        if has_rgb:
            print("   ✅ PLY文件包含RGB字段")
        else:
            print("   ❌ PLY文件缺少RGB字段")
            raise RuntimeError("PLY file has no RGB fields!")
    
    print("\n" + "="*80)
    print("✅ 修复完成！")
    print("="*80)
    print(f"\n统计:")
    print(f"  原始点数: {len(pcd.points):,}")
    print(f"  下采样点数: {len(pcd_ds.points):,}")
    print(f"  压缩比: {len(pcd.points) / len(pcd_ds.points):.2f}x")
    print(f"  有颜色: ✅")
    
    return pcd_ds


def main():
    parser = argparse.ArgumentParser(
        description="Fix downsampled point cloud color loss"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input PLY file path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output PLY file path"
    )
    parser.add_argument(
        "--voxel_size", 
        type=float, 
        default=0.02,
        help="Voxel size for downsampling (default: 0.02)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output file after saving"
    )
    
    args = parser.parse_args()
    
    try:
        fix_downsample_with_color(
            args.input,
            args.output,
            args.voxel_size,
            args.verify
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
