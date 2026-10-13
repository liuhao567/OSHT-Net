import fix_startup          # 修复 Python 3.10.19 regex bug, 必须最早!
import os
import torch               # 💡 必须先 import 纯粹的 torch
import torchvision         # 💡 再 import torchvision 顶层
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
# ... 后续保留其他 import
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import config
from models import overlock_xt_bridge_scsa

# ==========================================
# 1. 核心配置区
# ==========================================
CONFIG = {
    'data_dir': os.path.join(config.BRACS_ROOT, 'train'),  # ImageFolder: class folders directly below
    'save_dir': os.path.join(config.ROOT, 'ssl_ckp'),
    'img_size': config.MAE_CFG['img_size'],
    'patch_size': config.MAE_CFG['patch_size'],
    'mask_ratio': config.MAE_CFG['mask_ratio'],
    'batch_size': config.MAE_CFG['batch_size'],
    'epochs': config.MAE_CFG['epochs'],
    'base_lr': config.MAE_CFG['base_lr'],
    'weight_decay': config.MAE_CFG['weight_decay'],
    'num_workers': config.MAE_CFG['num_workers'],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 病理学专属数据增强 (剔除旋转预测的毒瘤)
# ==========================================
pathology_ssl_transforms = T.Compose([
    T.RandomResizedCrop(CONFIG['img_size'], scale=(0.4, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    
    # 💡 核心修复：提前将其转换为 PyTorch Tensor (浮点型)
    T.ToTensor(), 
    
    # 转换成 Tensor 后，ColorJitter 就不再走 PIL/Numpy 通道，完美支持负数 Hue 偏移
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
    
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. 掩码生成器 (Mask Generator)
# ==========================================
class RandomMaskGenerator:
    def __init__(self, input_size=224, patch_size=16, mask_ratio=0.6):
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.num_mask = int(self.num_patches * mask_ratio)
        self.patch_size = patch_size

    def __call__(self, x):
        B = x.shape[0]
        # 随机生成噪声进行 argsort，得到打乱的索引
        noise = torch.rand(B, self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask_idx = ids_shuffle[:, :self.num_mask] 
        
        # 1表示被遮挡，0表示保留
        mask = torch.zeros(B, self.num_patches, device=x.device)
        mask.scatter_(dim=1, index=mask_idx, value=1.0)
        
        # 将 1D 掩码拉伸成原始图像的 2D 形状
        mask_grid = mask.view(B, 1, self.grid_size, self.grid_size)
        mask_img = F.interpolate(mask_grid, scale_factor=self.patch_size, mode='nearest')
        return mask_img, mask

# ==========================================
# 4. Bridge SCSA MAE 架构封装
# ==========================================
class OverlockBridgeMAE(nn.Module):
    def __init__(self, encoder, encoder_dim=336): # XT 的最终输出是 336 维
        super().__init__()
        self.encoder = encoder
        
        # 可学习的 Mask Token，用于在图像级直接替换被遮住的像素
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 解码器：将 OverLoCK 输出的 7x7 特征图逐级上采样回 224x224
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(encoder_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 28x28 -> 56x56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 112x112 -> 224x224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, mask_img):
        # 1. 图像级屏蔽：被遮盖处用 mask_token，保留处用原图
        masked_x = x * (1 - mask_img) + self.mask_token * mask_img
        
        # 2. 前向传播：触发你的 ContextBridgeSCSA，逼迫桥接模块在信息残缺时强行对齐宏微观特征
        features, ctx_cls = self.encoder.forward_features(masked_x)
        
        # 3. 仅利用带有空间结构的主分支 features 进行重建
        pred = self.decoder(features)
        
        return pred

# ==========================================
# 5. 像素归一化重建损失 (核心 Trick)
# ==========================================
def calculate_mae_loss(imgs, pred, mask, patch_size=16):
    p = patch_size
    h = w = imgs.shape[2] // p
    
    # 目标图像切块并进行方差-均值归一化，逼迫模型学习高频纹理而非单纯的光照颜色
    target = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    target = torch.einsum('nchpwq->nhwpqc', target).reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1.e-6)**.5
    
    # 预测图像切块
    pred_patches = pred.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    pred_patches = torch.einsum('nchpwq->nhwpqc', pred_patches).reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    
    # 计算 MSE 损失
    loss = (pred_patches - target) ** 2
    loss = loss.mean(dim=-1)  # [B, num_patches]
    
    # 仅计算被 Mask 掉的区域损失 (模型只需负责把看不见的地方猜出来)
    loss = (loss * mask).sum() / mask.sum() 
    return loss

# ==========================================
# 6. 主训练循环
# ==========================================
def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # 加载无监督数据集 (忽略标签)
    dataset = ImageFolder(CONFIG['data_dir'], transform=pathology_ssl_transforms)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                            num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    
    print(f"🔥 BRACS MAE 自监督预训练启动 | 总计 WSI/ROI 切片: {len(dataset)} 张")
    
    
    encoder = overlock_xt_bridge_scsa(pretrained=False, num_classes=0)
    
   
    model = OverlockBridgeMAE(encoder=encoder, encoder_dim=420).to(DEVICE)
    
   
    actual_lr = CONFIG['base_lr'] * CONFIG['batch_size'] / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, betas=(0.9, 0.95), weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    mask_generator = RandomMaskGenerator(CONFIG['img_size'], CONFIG['patch_size'], CONFIG['mask_ratio'])
    scaler = GradScaler()
    
    best_loss = float('inf')
    start_epoch = 1

    # ---- Resume support ----
    ckpt_path = os.path.join(CONFIG['save_dir'], "overlock_xt_bridge_mae_best.pth")
    resume_path = os.path.join(CONFIG['save_dir'], "resume_checkpoint.pth")

    # If a previous best encoder exists, use it as initialization and restart
    # from epoch 1 with a fresh optimizer (delete ssl_ckp/ to train from scratch).
    if os.path.exists(ckpt_path):
        print(f"Loading existing encoder weights: {ckpt_path}")
        model.encoder.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)

    # Full resume (optimizer / scaler / epoch) is only done from resume_checkpoint.pth.
    if os.path.exists(resume_path):
        resume = torch.load(resume_path, map_location=DEVICE)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        start_epoch = resume['epoch'] + 1
        best_loss = resume['best_loss']
        for _ in range(resume['epoch']):
            scheduler.step()
        print(f"Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [MAE SSL]")
        
        for imgs, _ in progress:
            imgs = imgs.to(DEVICE)
            
           
            mask_img, mask_1d = mask_generator(imgs)
            
            optimizer.zero_grad()
            with autocast():
                
                preds = model(imgs, mask_img)
                loss = calculate_mae_loss(imgs, preds, mask_1d, patch_size=CONFIG['patch_size'])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress.set_postfix({'MSE': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"📈 Epoch {epoch:03d} | 平均重建损失 (MSE): {avg_loss:.4f}")
        
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(CONFIG['save_dir'], "overlock_xt_bridge_mae_best.pth")
            
            
            torch.save(model.encoder.state_dict(), save_path)
            print(f"💾 最佳 Encoder 权重已更新并保存至: {save_path}")
        
       
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, resume_path)
            print(f"🔄 Resume checkpoint 已保存 (epoch {epoch})")

if __name__ == '__main__':
    main()