import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm

def calculate_dice_iou(pred_mask, gt_mask):
    """计算单张图片的 Dice 和 IoU"""
    # 确保是布尔类型
    pred = pred_mask > 0
    gt = gt_mask > 0
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    
    return dice, iou

def main(args):
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    dice_list = []
    iou_list = []

    print(f"开始评估，共找到 {len(pred_files)} 张预测图...")

    for filename in tqdm(pred_files):
        # 1. 读取预测图
        pred_path = os.path.join(args.pred_dir, filename)
        pred_img = np.array(Image.open(pred_path).convert('L'))

        # 2. 读取对应的真实标签 (假设文件名一致)
        gt_path = os.path.join(args.gt_dir, filename)
        if not os.path.exists(gt_path):
            print(f"跳过：未找到对应的真实标签 {gt_path}")
            continue
        gt_img = np.array(Image.open(gt_path).convert('L'))

        # 3. 计算指标
        dice, iou = calculate_dice_iou(pred_img, gt_img)
        dice_list.append(dice)
        iou_list.append(iou)

    if len(dice_list) > 0:
        print("\n" + "="*30)
        print(f"评估结果 ({len(dice_list)} 张图片):")
        print(f"平均 Dice 系数: {np.mean(dice_list):.4f}")
        print(f"平均 IoU (Jaccard): {np.mean(iou_list):.4f}")
        print("="*30)
    else:
        print("错误：没有完成任何有效的对比，请检查文件名和路径。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='预测结果掩码图所在的文件夹')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实标签(Ground Truth)所在的文件夹')
    args = parser.parse_args()
    main(args)