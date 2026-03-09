import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Import các module từ cùng thư mục
from generator_ccgan import UNetGeneratorCCGAN
from discriminator_ccgan import DiscriminatorCCGAN
from dataset_ccgan import LungDatasetCCGAN

# Đường dẫn
DATA_ROOT = r"D:\UNIVERSITY\LAB\Simu\processed_data"
PIX2PIX_PATH = r"D:\UNIVERSITY\LAB\Simu\checkpoints\pix2pix_final.pth"
CHECKPOINT_DIR = r"D:\UNIVERSITY\LAB\Simu\checkpoints_cgan"
RESULT_DIR = r"D:\UNIVERSITY\LAB\Simu\results_cgan"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình
netG = UNetGeneratorCCGAN(in_c=1, out_c=1, num_classes=2).to(device)
netD = DiscriminatorCCGAN(in_c=2, num_classes=2).to(device)

# Nạp trọng số từ Pix2pix (Transfer Learning)
if os.path.exists(PIX2PIX_PATH):
    print(f"Đang nạp tiền bối Pix2Pix từ: {PIX2PIX_PATH}")
    # strict=False vì Generator mới có thêm lớp nhãn (label_emb)
    netG.load_state_dict(torch.load(PIX2PIX_PATH), strict=False)
    print("Đã nạp xong, giờ sẽ học thêm về bệnh lý.")

# Optimizer và Loss
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterionGAN = nn.BCEWithLogitsLoss()
criterionL1 = nn.L1Loss()

# DataLoader
dataset = LungDatasetCCGAN(DATA_ROOT)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Huấn luyện (200 vòng)
print(f"Bắt đầu huấn luyện cCGAN trên {device}...")

for epoch in range(1, 201):
    for i, batch in enumerate(loader):
        masks, imgs, labels = batch["A"].to(device), batch["B"].to(device), batch["L"].to(device)
        
        # --- TRAIN DISCRIMINATOR ---
        optD.zero_grad()
        fake_imgs = netG(masks, labels)
        
        real_pair = torch.cat([masks, imgs], dim=1)
        fake_pair = torch.cat([masks, fake_imgs.detach()], dim=1)
        
        # Sửa lỗi kích thước: Dùng ones_like/zeros_like trên pred
        pred_real = netD(real_pair, labels)
        loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
        
        pred_fake = netD(fake_pair, labels)
        loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optD.step()

        # Train generator
        optG.zero_grad()
        # Generator cố gắng đánh lừa Discriminator với nhãn tương ứng
        pred_g_fake = netD(torch.cat([masks, fake_imgs], 1), labels)
        loss_G_GAN = criterionGAN(pred_g_fake, torch.ones_like(pred_g_fake))
        
        # Loss L1 giữ cấu trúc phổi (lambda=100)
        loss_G_L1 = criterionL1(fake_imgs, imgs) * 100
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optG.step()

    # Lưu ảnh kiểm tra và in thông tin mỗi 10 epoch
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/200] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")
        img_path = os.path.join(RESULT_DIR, f"cgan_epoch_{epoch}.png")
        plt.imsave(img_path, fake_imgs[0].detach().cpu().squeeze(), cmap='gray')

# Lưu mô hình cuối cùng
torch.save(netG.state_dict(), os.path.join(CHECKPOINT_DIR, "cgan_final.pth"))
print(f"Đã lưu file cgan_final.pth tại: {CHECKPOINT_DIR}")
