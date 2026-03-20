# 创建一个专门存放预测结果的文件夹
mkdir -p ./results/test_predictions

# 执行预测
python predict.py \
    --model ./checkpoints/checkpoint_epoch100.pth \
    --input ./data_split/test/imgs/*.jpg \
    --output ./results/test_predictions/ \
    --classes 2 \
    --bilinear