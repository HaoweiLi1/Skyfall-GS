#!/bin/bash
# 测试专家修复版本

python train_task2_stage_c_fixed.py \
    --source_path data/datasets_JAX/JAX_068 \
    --model_path output/task2_stage_c_expert_fix \
    --use_color_calib \
    --task1_ckpt outputs/JAX/JAX_068/chkpnt30000.pth \
    --freeze_3dgs_iters 50 \
    --iterations 100 \
    --eval_interval 50 \
    --save_interval 100
