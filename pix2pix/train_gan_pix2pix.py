import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Import các thành phần Đạt đã tạo
from generator import UNetGenerator
from discriminator import PatchDiscriminator
from dataset import LungDataset

# --- CẤU HÌNH HỆ THỐNG ---
DATA_ROOT = r"D:\UNIVERSITY\LAB\Simu\processed_data"
CHECKPOINT_DIR = r"D:\UNIVERSITY\LAB\Simu\checkpoints"
RESULT_DIR = r"D:\UNIVERSITY\LAB\Simu\results\pix2pix"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KHỞI TẠO MÔ HÌNH
netG = UNetGenerator(in_c=1, out_c=1).to(device)
netD = PatchDiscriminator(in_c=2).to(device) # Discriminator nhận concat(A, B)

# Optimizer theo bài báo (lr=0.0002, beta1=0.5)
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterionGAN = nn.BCEWithLogitsLoss()
criterionL1 = nn.L1Loss()
L1_LAMBDA = 100

# DataLoader
dataset = LungDataset(DATA_ROOT)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Bắt đầu huấn luyện trên {device}...")

for epoch in range(1, 201):
    for i, batch in enumerate(loader):
        real_mask = batch["A"].to(device)
        real_img = batch["B"].to(device)

        # TỐI ƯU DISCRIMINATOR
        optD.zero_grad()
        # Ảnh giả từ Generator
        fake_img = netG(real_mask)
        
        # Đánh giá ảnh thật
        real_pair = torch.cat([real_mask, real_img], dim=1)
        pred_real = netD(real_pair)
        loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))

        # Đánh giá ảnh giả
        fake_pair = torch.cat([real_mask, fake_img.detach()], dim=1)
        pred_fake = netD(fake_pair)
        loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optD.step()

        # TỐI ƯU GENERATOR
        optG.zero_grad()
        # Đánh lừa Discriminator
        fake_pair_G = torch.cat([real_mask, fake_img], dim=1)
        pred_g_fake = netD(fake_pair_G)
        loss_G_GAN = criterionGAN(pred_g_fake, torch.ones_like(pred_g_fake))
        
        # Loss L1 để giữ cấu trúc giải phẫu
        loss_G_L1 = criterionL1(fake_img, real_img) * L1_LAMBDA
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optG.step()

    # Log kết quả và lưu ảnh mỗi 10 epoch
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/200] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")
        # Lưu ảnh vào thư mục results/pix2pix cho gọn
        img_path = os.path.join(RESULT_DIR, f"epoch_{epoch}.png")
        plt.imsave(img_path, fake_img[0].detach().cpu().squeeze(), cmap='gray')

# LƯU TRỌNG SỐ SAU CÙNG
torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, "pix2pix_final.pth"))
print(f"File của bạn nằm ở: {CHECKPOINT_DIR}\pix2pix_final.pth")