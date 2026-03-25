#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# 强制在脚本运行环境中指定临时目录
export TMPDIR=/data4/shengyingying/tmp

# 参数设置
EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=1e-5
CLASSES=2
RUN_NAME="v4" 

# 💡 建议：如果要对比实验，把 --load 行注释掉
# LOAD_PATH="./checkpoints/checkpoint_epoch100.pth"

echo "🚀 开始启动 U-Net 训练任务: $RUN_NAME"
echo "👉 使用显卡: GPU $CUDA_VISIBLE_DEVICES | LR: $LEARNING_RATE"

python train.py \
    --run-name $RUN_NAME \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --classes $CLASSES \
    --amp \
    --bilinear \
    # --load $LOAD_PATH  # <--- 对比实验时先注释掉