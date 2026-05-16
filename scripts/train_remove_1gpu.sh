#!/bin/bash
# 物体移除任务训练脚本（单卡）
# 从项目根目录运行：bash scripts/train_remove_1gpu.sh

CUDA_VISIBLE_DEVICES=0 \
USE_FLASH_ATTENTION=1 \
PYTHONPATH=. \
python -u -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=12346 \
    main.py \
    --base=configs/train/seva_edit_mmdit_remove.yaml \
    --no_date
