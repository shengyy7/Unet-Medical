#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

# 参数设置
EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=1e-5  # 增加学习率变量
CLASSES=2

echo "🚀 开始启动 U-Net 训练任务..."
echo "👉 使用显卡: GPU $CUDA_VISIBLE_DEVICES"

python train.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --classes $CLASSES \
    --amp \
    --bilinear  # 建议开启，防止转置卷积带来的棋盘效应