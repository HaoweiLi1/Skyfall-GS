#!/bin/bash
################################################################################
# Stage C Sanity Check
# 只渲染一帧，验证渲染输出是否正常
################################################################################

set -e

echo "========================================================================"
echo "Stage C Sanity Check"
echo "========================================================================"

# 进入 Skyfall-GS 目录
cd "$(dirname "$0")"

# 检查 sim3.json 是否存在
if [ ! -f "outputs/JAX/JAX_068/chkpnt30000.sim3.json" ]; then
    echo "❌ sim3.json 不存在，先运行 extract_sim3.py"
    echo "sim3.json 应该已经存在，如果不存在请手动运行: python extract_sim3.py"
    exit 1
fi

echo "✅ sim3.json 存在"

# 激活环境并运行
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skyfall-gs

# Sanity Check
python train_task2_stage_c.py \
    --source_path data/datasets_JAX/JAX_068 \
    --model_path output/task2_stage_c_sanity \
    --task1_ckpt outputs/JAX/JAX_068/chkpnt30000.pth \
    --use_color_calib \
    --use_ckpt_sim3 \
    --sim3_json outputs/JAX/JAX_068/chkpnt30000.sim3.json \
    --sanity_only

echo ""
echo "========================================================================"
echo "Sanity Check 完成"
echo "========================================================================"
echo "检查输出:"
echo "  - rgb_max 应该在 [0.3, 1.5]"
echo "  - alpha_mean 应该 > 0.2"
echo "  - 可视化图像: output/task2_stage_c_sanity/sanity_check.png"
echo "========================================================================"
