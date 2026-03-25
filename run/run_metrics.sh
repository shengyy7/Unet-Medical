#!/bin/bash

# 请修改为你的实际路径
PRED_PATH="./results/test_predictions_v3_2"      # predict.py 生成的结果文件夹
GT_PATH="./data_split/test/masks" # 真实的测试集标签文件夹

python ./scripts/compute_metrics.py --pred_dir $PRED_PATH --gt_dir $GT_PATH