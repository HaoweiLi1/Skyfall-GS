#!/bin/bash
# Task 2 - 完整评估流水线
# 一键运行所有评估步骤

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "Task 2 - 颜色校准与色域对齐 - 完整评估流水线"
echo "========================================================================"
echo ""

# 配置
RENDER_DIR="output/simulated_renders_warm"
GT_DIR="data/datasets_JAX/JAX_068/images"
OUTPUT_BASE="output/task2_full_pipeline"
N_SAMPLES=10000

echo "配置:"
echo "  渲染图像: $RENDER_DIR"
echo "  GT图像: $GT_DIR"
echo "  输出目录: $OUTPUT_BASE"
echo "  采样数: $N_SAMPLES"
echo ""

# 检查输入目录
if [ ! -d "$RENDER_DIR" ]; then
    echo "❌ 错误: 渲染图像目录不存在: $RENDER_DIR"
    echo "请先运行: python create_simulated_renders.py"
    exit 1
fi

if [ ! -d "$GT_DIR" ]; then
    echo "❌ 错误: GT图像目录不存在: $GT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

echo "========================================================================"
echo "Step 1 - Stage A评估（Reinhard统计匹配）"
echo "========================================================================"
echo ""

python eval_task2_step0_step1.py \
    --render_dir "$RENDER_DIR" \
    --gt_dir "$GT_DIR" \
    --output "$OUTPUT_BASE/step1_stage_a"

echo ""
echo "✅ Step 1完成"
echo ""

echo "========================================================================"
echo "Step 2 - Stage B评估（稳健闭式解）"
echo "========================================================================"
echo ""

python eval_task2_step2.py \
    --render_dir "$RENDER_DIR" \
    --gt_dir "$GT_DIR" \
    --output "$OUTPUT_BASE/step2_stage_b" \
    --n_samples "$N_SAMPLES"

echo ""
echo "✅ Step 2完成"
echo ""

echo "========================================================================"
echo "评估完成！"
echo "========================================================================"
echo ""
echo "结果位置:"
echo "  Step 1: $OUTPUT_BASE/step1_stage_a/"
echo "  Step 2: $OUTPUT_BASE/step2_stage_b/"
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_BASE/step1_stage_a/step0_step1_results.json"
echo "  cat $OUTPUT_BASE/step2_stage_b/step2_results.json"
echo ""
echo "对比图:"
echo "  ls $OUTPUT_BASE/step1_stage_a/vis/"
echo "  ls $OUTPUT_BASE/step2_stage_b/vis/"
echo ""
echo "Stage B参数:"
echo "  ls $OUTPUT_BASE/step2_stage_b/stage_b_params/"
echo ""
echo "🎉 Task 2评估流水线完成！"
