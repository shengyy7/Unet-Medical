import os
from pathlib import Path

# ================= 配置区域 =================
# 1. 你的检查点目录 (根据你的 train.py 设置)
CHECKPOINT_DIR = Path('./checkpoints/v2_adamw_patience15') 

# 2. 你想保留的 Epoch 编号 (比如最好的在 85 轮，最后的是 100 轮)
# 请修改下面的数字
BEST_EPOCH = 2
LAST_EPOCH = 100 

# 3. 是否开启“只看预览不删除”模式
# 先保持 True 运行一次，确认输出列表正确后，再改为 False 执行删除
DRY_RUN = False
# ===========================================

def manual_cleanup():
    if not CHECKPOINT_DIR.exists():
        print(f"错误：找不到目录 {CHECKPOINT_DIR}")
        return

    # 定义要保留的文件名模式
    # 根据 train.py 中的 torch.save(..., 'checkpoint_epoch{}.pth'.format(epoch))
    keep_filenames = [
        f'checkpoint_epoch{BEST_EPOCH}.pth',
        f'checkpoint_epoch{LAST_EPOCH}.pth'
    ]

    print(f"--- 开始清理检查点目录: {CHECKPOINT_DIR} ---")
    print(f"计划保留的文件: {keep_filenames}")
    
    deleted_count = 0
    kept_count = 0

    # 遍历目录下所有文件
    for file_path in CHECKPOINT_DIR.glob('*.pth'):
        if file_path.name in keep_filenames:
            print(f"[保留] {file_path.name}")
            kept_count += 1
        else:
            if DRY_RUN:
                print(f"[预演删除] {file_path.name}")
            else:
                try:
                    file_path.unlink() # 执行物理删除
                    print(f"[已删除] {file_path.name}")
                except Exception as e:
                    print(f"[失败] 无法删除 {file_path.name}: {e}")
            deleted_count += 1

    print("\n--- 清理总结 ---")
    print(f"保留文件数: {kept_count}")
    if DRY_RUN:
        print(f"待删除文件数 (预演): {deleted_count}")
        print("\n提示：当前为【预览模式】，没有文件被真正删除。")
        print("确认无误后，请将脚本中的 DRY_RUN = True 改为 False 再运行。")
    else:
        print(f"实际删除文件数: {deleted_count}")

if __name__ == "__main__":
    manual_cleanup()