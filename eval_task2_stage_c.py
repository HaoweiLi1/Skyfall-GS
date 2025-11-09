#!/usr/bin/env python3
"""
Task 2 - Stage C 评估脚本
评估端到端训练的效果，验证Gate T2-3
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('.')

def load_training_log(log_path):
    """加载训练日志"""
    with open(log_path, 'r') as f:
        return json.load(f)

def load_final_results(results_path):
    """加载最终结果"""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_training_curves(training_log, output_dir):
    """绘制训练曲线"""
    iterations = [iter_log['iteration'] for iter_log in training_log['iterations']]
    train_psnr = [iter_log['train_psnr'] for iter_log in training_log['iterations']]
    train_de = [iter_log['train_de_median'] for iter_log in training_log['iterations']]
    test_psnr = [iter_log['test_psnr'] for iter_log in training_log['iterations']]
    test_de = [iter_log['test_de_median'] for iter_log in training_log['iterations']]
    
    loss_total = [iter_log['loss_total'] for iter_log in training_log['iterations']]
    loss_main = [iter_log['loss_main'] for iter_log in training_log['iterations']]
    loss_reg = [iter_log['loss_reg'] for iter_log in training_log['iterations']]
    
    # 校准参数统计
    M_norms = [iter_log['calib_stats'].get('avg_M_norm', 0) for iter_log in training_log['iterations']]
    t_norms = [iter_log['calib_stats'].get('avg_t_norm', 0) for iter_log in training_log['iterations']]
    
    phase1_end = training_log.get('phase1_end', 500)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PSNR曲线
    axes[0, 0].plot(iterations, train_psnr, 'b-', label='训练', linewidth=2)
    if any(p > 0 for p in test_psnr):
        axes[0, 0].plot(iterations, test_psnr, 'r-', label='测试', linewidth=2)
    axes[0, 0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase切换')
    axes[0, 0].set_xlabel('迭代数')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title('PSNR曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ΔE00曲线
    axes[0, 1].plot(iterations, train_de, 'b-', label='训练', linewidth=2)
    if any(d > 0 for d in test_de):
        axes[0, 1].plot(iterations, test_de, 'r-', label='测试', linewidth=2)
    axes[0, 1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase切换')
    axes[0, 1].set_xlabel('迭代数')
    axes[0, 1].set_ylabel('ΔE00')
    axes[0, 1].set_title('ΔE00曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[0, 2].plot(iterations, loss_total, 'k-', label='总损失', linewidth=2)
    axes[0, 2].plot(iterations, loss_main, 'b-', label='主损失', linewidth=1)
    axes[0, 2].plot(iterations, loss_reg, 'r-', label='正则损失', linewidth=1)
    axes[0, 2].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase切换')
    axes[0, 2].set_xlabel('迭代数')
    axes[0, 2].set_ylabel('损失')
    axes[0, 2].set_title('损失曲线')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # ||M-I||曲线
    axes[1, 0].plot(iterations, M_norms, 'g-', linewidth=2)
    axes[1, 0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase切换')
    axes[1, 0].set_xlabel('迭代数')
    axes[1, 0].set_ylabel('||M-I||_F')
    axes[1, 0].set_title('颜色矩阵偏离恒等')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ||t||曲线
    axes[1, 1].plot(iterations, t_norms, 'orange', linewidth=2)
    axes[1, 1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase切换')
    axes[1, 1].set_xlabel('迭代数')
    axes[1, 1].set_ylabel('||t||_2')
    axes[1, 1].set_title('偏置向量范数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # PSNR vs ΔE00散点图
    axes[1, 2].scatter(train_de, train_psnr, c=iterations, cmap='viridis', alpha=0.7)
    axes[1, 2].set_xlabel('ΔE00')
    axes[1, 2].set_ylabel('PSNR (dB)')
    axes[1, 2].set_title('PSNR vs ΔE00')
    cbar = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar.set_label('迭代数')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {output_dir / 'training_curves.png'}")

def evaluate_gate_t2_3(final_results, baseline_results=None):
    """评估Gate T2-3"""
    print("\n" + "=" * 80)
    print("Gate T2-3 验证")
    print("=" * 80)
    
    train_psnr = final_results['train_avg_psnr']
    train_de = final_results['train_median_de']
    test_psnr = final_results['test_avg_psnr']
    test_de = final_results['test_median_de']
    
    print(f"训练结果:")
    print(f"  平均PSNR: {train_psnr:.2f} dB")
    print(f"  中位ΔE00: {train_de:.2f}")
    
    if test_psnr > 0:
        print(f"测试结果:")
        print(f"  平均PSNR: {test_psnr:.2f} dB")
        print(f"  中位ΔE00: {test_de:.2f}")
    
    # Gate T2-3标准
    gate_results = {
        'psnr_gain_sufficient': False,
        'de_acceptable': False,
        'overall_pass': False
    }
    
    if baseline_results:
        baseline_psnr = baseline_results.get('baseline_psnr', 0)
        psnr_gain = train_psnr - baseline_psnr
        gate_results['psnr_gain'] = psnr_gain
        gate_results['psnr_gain_sufficient'] = psnr_gain >= 0.5
        
        print(f"\nGate T2-3 验证:")
        print(f"  基线PSNR: {baseline_psnr:.2f} dB")
        print(f"  PSNR增益: {psnr_gain:+.2f} dB")
        print(f"  ✓ PSNR增益 ≥ 0.5 dB: {'✅ 通过' if gate_results['psnr_gain_sufficient'] else '❌ 未通过'}")
    else:
        print(f"\nGate T2-3 验证（无基线对比）:")
        gate_results['psnr_gain_sufficient'] = train_psnr >= 30.0  # 假设阈值
        print(f"  ✓ PSNR ≥ 30.0 dB: {'✅ 通过' if gate_results['psnr_gain_sufficient'] else '❌ 未通过'}")
    
    gate_results['de_acceptable'] = train_de <= 3.5
    print(f"  ✓ ΔE00 ≤ 3.5: {'✅ 通过' if gate_results['de_acceptable'] else '❌ 未通过'} ({train_de:.2f})")
    
    gate_results['overall_pass'] = gate_results['psnr_gain_sufficient'] and gate_results['de_acceptable']
    
    if gate_results['overall_pass']:
        print(f"\n  🎉 Gate T2-3 通过！")
    else:
        print(f"\n  ⚠️  Gate T2-3 未通过")
    
    return gate_results

def generate_report(training_log, final_results, gate_results, output_dir):
    """生成评估报告"""
    report = {
        'evaluation_date': training_log.get('end_time', 'unknown'),
        'training_args': training_log.get('args', {}),
        'phase1_end': training_log.get('phase1_end', 500),
        'total_iterations': training_log['args'].get('iterations', 3000),
        'final_results': final_results,
        'gate_t2_3': gate_results,
        'training_summary': {
            'num_train_cameras': len(final_results['train_results']),
            'num_test_cameras': len(final_results['test_results']),
            'final_train_psnr': final_results['train_avg_psnr'],
            'final_train_de': final_results['train_median_de'],
            'final_test_psnr': final_results['test_avg_psnr'],
            'final_test_de': final_results['test_median_de']
        }
    }
    
    # 保存报告
    with open(output_dir / "stage_c_evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n评估报告已保存: {output_dir / 'stage_c_evaluation_report.json'}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Task 2 Stage C 评估")
    parser.add_argument('--result_dir', type=str, required=True, help='训练结果目录')
    parser.add_argument('--baseline_results', type=str, help='基线结果文件（可选）')
    parser.add_argument('--output', type=str, help='输出目录（默认为result_dir）')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output) if args.output else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("========================================================================")
    print("Task 2 - Stage C 评估")
    print("========================================================================")
    print(f"结果目录: {result_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 加载训练日志
    log_path = result_dir / "training_log.json"
    if not log_path.exists():
        print(f"❌ 训练日志不存在: {log_path}")
        return
    
    training_log = load_training_log(log_path)
    print(f"✅ 加载训练日志: {len(training_log['iterations'])} 个记录")
    
    # 加载最终结果
    results_path = result_dir / "final_results.json"
    if not results_path.exists():
        print(f"❌ 最终结果不存在: {results_path}")
        return
    
    final_results = load_final_results(results_path)
    print(f"✅ 加载最终结果: {len(final_results['train_results'])} 训练相机, {len(final_results['test_results'])} 测试相机")
    
    # 加载基线结果（可选）
    baseline_results = None
    if args.baseline_results and os.path.exists(args.baseline_results):
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        print(f"✅ 加载基线结果")
    
    # 绘制训练曲线
    print(f"\n绘制训练曲线...")
    plot_training_curves(training_log, output_dir)
    
    # 评估Gate T2-3
    gate_results = evaluate_gate_t2_3(final_results, baseline_results)
    
    # 生成报告
    report = generate_report(training_log, final_results, gate_results, output_dir)
    
    print("\n" + "=" * 80)
    print("Stage C 评估完成")
    print("=" * 80)
    print(f"训练曲线: {output_dir / 'training_curves.png'}")
    print(f"评估报告: {output_dir / 'stage_c_evaluation_report.json'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
