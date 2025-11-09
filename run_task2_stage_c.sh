#!/bin/bash
# Task 2 - Stage C 端到端训练一键脚本

set -e  # 遇到错误立即退出

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate skyfall-gs

echo "========================================================================"
echo "Task 2 - Stage C 端到端训练"
echo "========================================================================"
echo ""

# 配置
SOURCE_PATH="data/datasets_JAX/JAX_068"
MODEL_PATH="output/task2_stage_c_final"  # 输出目录
CALIB_DIR="output/task2_step2_fixed_viz/stage_b_params"  # Stage B参数
TASK1_CKPT="output/task1_final_test/chkpnt5000.pth"  # Task 1 checkpoint（不要用.ply！）
ITERATIONS=3000
FREEZE_ITERS=50  # 专家建议：30-50 iter

echo "配置:"
echo "  数据集: $SOURCE_PATH"
echo "  输出目录: $MODEL_PATH"
echo "  Stage B参数: $CALIB_DIR"
echo "  Task 1 checkpoint: $TASK1_CKPT"
echo "  总迭代数: $ITERATIONS"
echo "  冻结3DGS: $FREEZE_ITERS iters (专家建议：30-50)"
echo ""

# 检查输入
if [ ! -d "$SOURCE_PATH" ]; then
    echo "❌ 错误: 数据集目录不存在: $SOURCE_PATH"
    exit 1
fi

if [ ! -d "$CALIB_DIR" ]; then
    echo "❌ 错误: Stage B参数目录不存在: $CALIB_DIR"
    echo "请先运行: bash run_task2_full_pipeline.sh"
    exit 1
fi

if [ ! -f "$TASK1_CKPT" ]; then
    echo "❌ 错误: Task 1 checkpoint不存在: $TASK1_CKPT"
    echo "请先运行Task 1训练"
    exit 1
fi

echo "✅ Task 1 checkpoint存在: $TASK1_CKPT"

# 创建输出目录
mkdir -p "$MODEL_PATH"

echo "========================================================================"
echo "Phase 1 - 训练准备"
echo "========================================================================"
echo ""

# 检查环境
echo "检查环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "========================================================================"
echo "Phase 2 - 开始训练"
echo "========================================================================"
echo ""

# 开始训练（专家建议：加载Task 1模型）
python train_task2_stage_c.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --use_color_calib \
    --color_calib_dir "$CALIB_DIR" \
    --task1_ckpt "$TASK1_CKPT" \
    --freeze_3dgs_iters "$FREEZE_ITERS" \
    --iterations "$ITERATIONS" \
    --calib_reg_lambda 1e-3 \
    --calib_reg_mu 1e-4 \
    --calib_lr_phase1 1e-3 \
    --calib_lr_phase2 2e-4 \
    --eval_interval 100 \
    --save_interval 1000

echo ""
echo "✅ 训练完成"
echo ""

echo "========================================================================"
echo "Phase 3 - 评估结果"
echo "========================================================================"
echo ""

# 评估结果
python eval_task2_stage_c.py \
    --result_dir "$MODEL_PATH" \
    --output "$MODEL_PATH/evaluation"

echo ""
echo "✅ 评估完成"
echo ""

echo "========================================================================"
echo "Stage C 完成！"
echo "========================================================================"
echo ""
echo "结果位置:"
echo "  训练结果: $MODEL_PATH/"
echo "  评估结果: $MODEL_PATH/evaluation/"
echo ""
echo "查看结果:"
echo "  cat $MODEL_PATH/final_results.json"
echo "  cat $MODEL_PATH/evaluation/stage_c_evaluation_report.json"
echo ""
echo "可视化:"
echo "  ls $MODEL_PATH/vis/"
echo "  ls $MODEL_PATH/evaluation/training_curves.png"
echo ""
echo "🎉 Task 2 - Stage C 端到端训练完成！"
