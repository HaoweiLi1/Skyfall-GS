#!/bin/bash
# 诊断测试 - 只运行到 Sanity Check

python train_task2_stage_c.py \
    --source_path data/datasets_JAX/JAX_068 \
    --model_path output/task2_stage_c_diagnose \
    --use_color_calib \
    --task1_ckpt outputs/JAX/JAX_068/chkpnt30000.pth \
    --freeze_3dgs_iters 50 \
    --iterations 1 \
    --eval_interval 1 \
    --save_interval 1000 2>&1 | tee diagnose.log
