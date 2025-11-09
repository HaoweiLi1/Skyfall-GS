#!/bin/bash
# 快速测试 Stage C（只运行100次迭代）

python train_task2_stage_c.py \
    --source_path data/datasets_JAX/JAX_068 \
    --model_path output/task2_stage_c_test \
    --use_color_calib \
    --task1_ckpt outputs/JAX/JAX_068/chkpnt30000.pth \
    --freeze_3dgs_iters 50 \
    --iterations 100 \
    --eval_interval 50 \
    --save_interval 100
