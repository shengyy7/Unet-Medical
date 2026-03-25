# 创建一个专门存放预测结果的文件夹
mkdir -p ./results/test_predictions_v3_2

# 执行预测
python predict.py \
    --model ./checkpoints/v3/best_model.pth \
    --input ./data_split/test/imgs/*.jpg \
    --output ./results/test_predictions_v3_2/ \
    --classes 2 \
    --bilinear