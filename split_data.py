import os
import pandas as pd
import random
import shutil
import re
from pathlib import Path

# ================= 配置区 =================
# 指向你当前的原始数据目录（请确保这两个文件夹路径正确）
img_dir = Path('./data/imgs')
mask_dir = Path('./data/masks')
# 指向你的 Excel 诊断表（csv格式）
excel_path = "吻合口良恶性狭窄诊断version.1.xlsx"
# 输出目录
output_root = Path('./data_split')

# 划分比例 (训练:验证:测试 = 7:1:2)
split_ratios = {'train': 0.7, 'val': 0.1, 'test': 0.2}
seed = 42
# ==========================================

# 1. 扫描所有图片并按病人 ID 分组
print("🔍 正在扫描文件并解析病人 ID...")
all_images = list(img_dir.glob('*.jpg'))
patient_to_images = {}

for img_path in all_images:
    # 适配你的命名：case_1_1(2).jpg -> 提取出 "1"
    match = re.search(r'case_(\d+)_', img_path.name)
    if match:
        case_id = int(match.group(1))
        if case_id not in patient_to_images:
            patient_to_images[case_id] = []
        patient_to_images[case_id].append(img_path.name)

if not patient_to_images:
    print("❌ 错误：在 ./data/imgs 中没找到符合 'case_ID_' 命名规律的图片！")
    exit()

# 2. 读取 Excel 标签并进行分层 (按病人 ID)
df = pd.read_excel(excel_path, sheet_name='Sheet1')
malignant_patients = []
benign_patients = []

for case_id in patient_to_images.keys():
    # 从 Excel 找对应的 rucurrence 标签 (1为恶性, 0为良性)
    row = df[df['case'] == case_id]
    if not row.empty:
        label = row['rucurrence'].iloc[0]
        if label == 1:
            malignant_patients.append(case_id)
        else:
            benign_patients.append(case_id)
    else:
        # 如果 Excel 里没这个 case，默认分到良性组（或者你可以打印出来检查）
        benign_patients.append(case_id)

print(f"📊 统计：恶性病例 {len(malignant_patients)} 个，良性病例 {len(benign_patients)} 个")

# 3. 分层随机划分 ID
def split_list(lst, ratios, seed):
    random.seed(seed)
    random.shuffle(lst)
    n = len(lst)
    train_end = int(n * ratios['train'])
    val_end = train_end + int(n * ratios['val'])
    return lst[:train_end], lst[train_end:val_end], lst[val_end:]

m_train, m_val, m_test = split_list(malignant_patients, split_ratios, seed)
b_train, b_val, b_test = split_list(benign_patients, split_ratios, seed)

final_split = {
    'train': m_train + b_train,
    'val': m_val + b_val,
    'test': m_test + b_test
}

# 4. 拷贝文件到新目录
print("🚚 正在搬运文件...")
for split_name, ids in final_split.items():
    img_out = output_root / split_name / 'imgs'
    mask_out = output_root / split_name / 'masks'
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    
    for case_id in ids:
        for img_name in patient_to_images[case_id]:
            # 拷贝原图
            shutil.copy(img_dir / img_name, img_out / img_name)
            # 拷贝掩码 (处理你的后缀逻辑)
            # 假设 mask 是 case_1_1(2)_mask.png
            mask_name = img_name.replace('.jpg', '_mask.png')
            if (mask_dir / mask_name).exists():
                shutil.copy(mask_dir / mask_name, mask_out / mask_name)
            else:
                # 兼容性处理：如果你的 mask 叫 case_1_1(2)_mask.jpg 之类的
                potential_mask = list(mask_dir.glob(img_name.split('.')[0] + '_mask.*'))
                if potential_mask:
                    shutil.copy(potential_mask[0], mask_out / potential_mask[0].name)

print(f"✅ 划分完成！请查看 {output_root} 目录。")
print(f"提示：训练集包含 {len(final_split['train'])} 个病人，测试集包含 {len(final_split['test'])} 个病人。")