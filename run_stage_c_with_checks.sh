#!/bin/bash
################################################################################
# Task 2 Stage C - 带检查的执行脚本
# 
# 功能：
# 1. 执行前置检查（代码、数据、参数）
# 2. 运行 Stage C 训练
# 3. 实时监控输出
################################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Task 2 Stage C - 执行前检查"
echo "========================================================================"

# 确保在正确的目录
if [ ! -f "train_task2_stage_c.py" ]; then
    echo -e "${YELLOW}当前目录: $(pwd)${NC}"
    echo -e "${YELLOW}脚本需要在 Skyfall-GS 目录下运行${NC}"
    if [ -f "Skyfall-GS/train_task2_stage_c.py" ]; then
        echo -e "${GREEN}检测到 Skyfall-GS 目录，自动切换...${NC}"
        cd Skyfall-GS
    else
        echo -e "${RED}❌ 找不到 train_task2_stage_c.py${NC}"
        echo "请在 Skyfall-GS 目录下运行此脚本"
        exit 1
    fi
fi

# ============================================================================
# 1. 代码检查
# ============================================================================
echo -e "\n${YELLOW}[1/5] 代码检查...${NC}"

# 1.1 检查脚本存在
if [ ! -f "train_task2_stage_c.py" ]; then
    echo -e "${RED}❌ train_task2_stage_c.py 不存在${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 脚本文件存在${NC}"

# 1.2 检查关键函数
if ! grep -q "def load_ckpt_strict" train_task2_stage_c.py; then
    echo -e "${RED}❌ 缺少 load_ckpt_strict() 函数${NC}"
    exit 1
fi
if ! grep -q "def rebuild_filter_if_needed" train_task2_stage_c.py; then
    echo -e "${RED}❌ 缺少 rebuild_filter_if_needed() 函数${NC}"
    exit 1
fi
if ! grep -q "def sanity_render_and_assert" train_task2_stage_c.py; then
    echo -e "${RED}❌ 缺少 sanity_render_and_assert() 函数${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 三个关键函数都存在${NC}"

# 1.3 检查没有 filter_3D=100 的旧代码（只检查实际赋值语句）
if grep -rn 'filter_3D\s*=\s*torch\.\(full\|ones\).*100' *.py utils/*.py 2>/dev/null; then
    echo -e "${RED}❌ 发现 filter_3D=100 的旧代码！${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 没有 filter_3D=100 的旧代码（已检查实际赋值语句）${NC}"

# ============================================================================
# 2. 数据检查
# ============================================================================
echo -e "\n${YELLOW}[2/5] 数据检查...${NC}"

# 2.1 检查 Task 1 checkpoint
# 自动查找最新的 checkpoint
TASK1_CKPT=""
if [ -f "output/task1_final_test/chkpnt5000.pth" ]; then
    TASK1_CKPT="output/task1_final_test/chkpnt5000.pth"
elif [ -d "outputs/JAX/JAX_068" ]; then
    # 查找最新的 checkpoint
    LATEST_CKPT=$(ls -t outputs/JAX/JAX_068/chkpnt*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        TASK1_CKPT="$LATEST_CKPT"
        echo -e "${YELLOW}自动检测到 checkpoint: $TASK1_CKPT${NC}"
    fi
fi

if [ -z "$TASK1_CKPT" ] || [ ! -f "$TASK1_CKPT" ]; then
    echo -e "${RED}❌ Task 1 checkpoint 不存在${NC}"
    echo "请先运行 Task 1 训练，或手动设置 TASK1_CKPT 变量"
    echo "可用的 checkpoint:"
    find outputs -name "chkpnt*.pth" 2>/dev/null | head -5
    exit 1
fi
echo -e "${GREEN}✅ Task 1 checkpoint 存在: $TASK1_CKPT${NC}"

# 2.2 检查 Stage B 参数
STAGE_B_DIR=""
if [ -d "output/task2_step2_results/stage_b_params" ]; then
    STAGE_B_DIR="output/task2_step2_results/stage_b_params"
elif [ -d "outputs/task2_step2_results/stage_b_params" ]; then
    STAGE_B_DIR="outputs/task2_step2_results/stage_b_params"
fi

if [ -z "$STAGE_B_DIR" ] || [ ! -d "$STAGE_B_DIR" ]; then
    echo -e "${YELLOW}⚠️  Stage B 参数目录不存在${NC}"
    echo "将使用恒等初始化（不加载 Stage B 参数）"
    STAGE_B_DIR=""
else
    NUM_PARAMS=$(ls "$STAGE_B_DIR"/*.npz 2>/dev/null | wc -l)
    if [ "$NUM_PARAMS" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Stage B 参数目录为空${NC}"
        echo "将使用恒等初始化"
        STAGE_B_DIR=""
    else
        echo -e "${GREEN}✅ Stage B 参数存在: $NUM_PARAMS 个文件 ($STAGE_B_DIR)${NC}"
    fi
fi

# 2.3 检查数据集
DATASET_PATH="data/datasets_JAX/JAX_068"
if [ ! -d "$DATASET_PATH/images" ]; then
    echo -e "${RED}❌ 数据集不存在: $DATASET_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 数据集存在: $DATASET_PATH${NC}"

# ============================================================================
# 3. 参数设置
# ============================================================================
echo -e "\n${YELLOW}[3/5] 参数设置...${NC}"

# 输出目录
OUTPUT_DIR="output/task2_stage_c"
mkdir -p "$OUTPUT_DIR"

# 训练参数（专家建议）
FREEZE_ITERS=50          # 冻结 3DGS 的迭代数（30-50）
CALIB_LR_P1=1e-3         # Phase-1 学习率（带 warm-up）
CALIB_LR_P2=2e-4         # Phase-2 学习率（降低避免替代几何）
REG_LAMBDA=1e-3          # ||M-I||_F^2 系数
REG_MU=1e-4              # ||t||_2^2 系数
TOTAL_ITERS=3000         # 总迭代数
EVAL_INTERVAL=100        # 评估间隔
SAVE_INTERVAL=1000       # 保存间隔

echo "  输出目录: $OUTPUT_DIR"
echo "  冻结迭代数: $FREEZE_ITERS"
echo "  Phase-1 学习率: $CALIB_LR_P1"
echo "  Phase-2 学习率: $CALIB_LR_P2"
echo "  总迭代数: $TOTAL_ITERS"

# ============================================================================
# 4. 环境检查
# ============================================================================
echo -e "\n${YELLOW}[4/5] 环境检查...${NC}"

# 4.1 检查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python 未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python: $(python --version)${NC}"

# 4.2 检查 CUDA
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}❌ CUDA 不可用${NC}"
    exit 1
fi
echo -e "${GREEN}✅ CUDA 可用${NC}"

# 4.3 检查依赖
if ! python -c "import numpy, torch, tqdm" 2>/dev/null; then
    echo -e "${RED}❌ 缺少必要的 Python 包${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 依赖完整${NC}"

# ============================================================================
# 5. 用户确认
# ============================================================================
echo -e "\n${YELLOW}[5/5] 用户确认...${NC}"
echo "即将开始训练，参数如下："
echo "  数据集: $DATASET_PATH"
echo "  Task 1 checkpoint: $TASK1_CKPT"
echo "  Stage B 参数: $STAGE_B_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  总迭代数: $TOTAL_ITERS"
echo ""
read -p "确认开始训练？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消训练"
    exit 0
fi

# ============================================================================
# 6. 开始训练
# ============================================================================
echo -e "\n${GREEN}========================================================================"
echo "开始训练 Stage C"
echo "========================================================================${NC}"

# 构建命令
CMD="python train_task2_stage_c.py \
    --source_path $DATASET_PATH \
    --model_path $OUTPUT_DIR \
    --use_color_calib \
    --color_calib_dir $STAGE_B_DIR \
    --task1_ckpt $TASK1_CKPT \
    --freeze_3dgs_iters $FREEZE_ITERS \
    --calib_lr_phase1 $CALIB_LR_P1 \
    --calib_lr_phase2 $CALIB_LR_P2 \
    --calib_reg_lambda $REG_LAMBDA \
    --calib_reg_mu $REG_MU \
    --iterations $TOTAL_ITERS \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL"

echo "执行命令:"
echo "$CMD"
echo ""

# 创建日志文件
LOG_FILE="$OUTPUT_DIR/training_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo ""

# 执行训练（同时输出到终端和日志文件）
$CMD 2>&1 | tee "$LOG_FILE"

# ============================================================================
# 7. 训练后检查
# ============================================================================
echo -e "\n${GREEN}========================================================================"
echo "训练完成，执行后检查"
echo "========================================================================${NC}"

# 7.1 检查 Sanity Check 是否通过
if grep -q "SANITY.*✅.*通过检查" "$LOG_FILE"; then
    echo -e "${GREEN}✅ Sanity Check 通过${NC}"
else
    echo -e "${RED}❌ Sanity Check 未通过${NC}"
    echo "请检查日志: $LOG_FILE"
fi

# 7.2 检查是否有 NaN
if grep -q "NaN\|nan\|inf" "$LOG_FILE"; then
    echo -e "${YELLOW}⚠️  训练过程中出现 NaN/Inf${NC}"
fi

# 7.3 检查输出文件
if [ -f "$OUTPUT_DIR/final_results.json" ]; then
    echo -e "${GREEN}✅ 最终结果已保存${NC}"
    python -c "
import json
with open('$OUTPUT_DIR/final_results.json') as f:
    results = json.load(f)
    print(f'  训练 PSNR: {results[\"train_avg_psnr\"]:.2f} dB')
    print(f'  训练 ΔE00: {results[\"train_median_de\"]:.2f}')
    if results['test_avg_psnr'] > 0:
        print(f'  测试 PSNR: {results[\"test_avg_psnr\"]:.2f} dB')
        print(f'  测试 ΔE00: {results[\"test_median_de\"]:.2f}')
"
else
    echo -e "${RED}❌ 最终结果文件不存在${NC}"
fi

# 7.4 检查可视化
NUM_VIS=$(ls "$OUTPUT_DIR/vis"/*.png 2>/dev/null | wc -l)
if [ "$NUM_VIS" -gt 0 ]; then
    echo -e "${GREEN}✅ 可视化文件: $NUM_VIS 个${NC}"
else
    echo -e "${YELLOW}⚠️  没有可视化文件${NC}"
fi

echo -e "\n${GREEN}========================================================================"
echo "全部完成！"
echo "========================================================================${NC}"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo ""
