import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import traceback

import tempfile
print(f"当前程序使用的临时目录是: {tempfile.gettempdir()}")

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')
# 修改为接收两个路径
dir_train_img = Path('./data_split/train/imgs/')
dir_train_mask = Path('./data_split/train/masks/')
dir_val_img = Path('./data_split/val/imgs/')
dir_val_mask = Path('./data_split/val/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        dir_checkpoint, 
        run_name,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        # val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-4,# 从最初的1e-8到1e-4到1e-6到1e-4
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    #直接分别创建训练集和验证集
    try:
        train_set = CarvanaDataset(dir_train_img, dir_train_mask, img_scale)
        val_set = CarvanaDataset(dir_val_img, dir_val_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        train_set = BasicDataset(dir_train_img, dir_train_mask, img_scale)
        val_set =   BasicDataset(dir_val_img, dir_val_mask, img_scale)

    n_train = len(train_set)
    n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow',name=run_name, anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            split_strategy="physical_712", save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    # # 【修改：优化器换成 AdamW 并增加权重衰减防止过拟合】
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # # 【修改：将 patience 改为 15，给模型更多探索时间】
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15)

    # 1. 适当调大 weight_decay（例如 1e-4），防止模型后期过度拟合训练集噪声
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    # 2. 使用余弦退火，T_max 设置为总 Epoch 数
    # 这会让 LR 呈曲线下降，直到最后一个 Epoch 降至最小值
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    

    # --- 新增：初始化最高分数变量 ---
    best_val_score = -float('inf')

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += 1.2*dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += 1.2*dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        

                        logging.info('Validation Dice score: {}'.format(val_score))

                        # 判断并保存最好的模型
                        if val_score > best_val_score:
                            best_val_score = val_score
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = train_set.mask_values

                            # 固定名称，直接覆盖旧的最好模型
                            save_path = str(dir_checkpoint / 'best_model.pth')
                            torch.save(state_dict, save_path)
                            logging.info(f'New best model saved! Dice: {val_score:.4f}')
                        try:
                            img_numpy = images[0].cpu().permute(1, 2, 0).numpy()
                            # 对于 2 维掩码 [H, W]，直接转 Numpy 即可
                            true_mask_numpy = true_masks[0].cpu().numpy()
                            pred_mask_numpy = masks_pred.argmax(dim=1)[0].cpu().numpy()
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(img_numpy),
                                'masks': {
                                    'true': wandb.Image(true_mask_numpy),
                                    'pred': wandb.Image(pred_mask_numpy),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception as e:
                            print(f"___ 捕获到错误: {e} ___")
                            traceback.print_exc() # 这会打印完整的错误链路

        scheduler.step()
        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True) # 自动创建新文件夹
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values

            last_path = str(dir_checkpoint / 'last_model.pth')
            torch.save(state_dict, last_path)
            logging.info(f'Last model updated at epoch {epoch}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
    #                     help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--run-name', type=str, default='v1_base', help='本次实验的唯一名称')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    checkpoint_path = Path('./checkpoints') / args.run_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            dir_checkpoint=checkpoint_path, # 需要修改 train_model 的接收参数
            run_name=args.run_name,
            img_scale=args.scale,
            # val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            # val_percent=args.val / 100,
            amp=args.amp,
            dir_checkpoint=checkpoint_path, # 需要修改 train_model 的接收参数
            run_name=args.run_name
        )
