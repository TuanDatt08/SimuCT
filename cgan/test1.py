import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys
import random

# Đường dẫn đến các file .pth và dữ liệu
# Thêm thư mục cha để Python tìm thấy generator.py của Pix2Pix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from generator import UNetGenerator # Pix2Pix
    from generator_ccgan import UNetGeneratorCCGAN # cCGAN
except ImportError:
    print("generator.py và generator_ccgan.py nằm sai thư mục")
    sys.exit()

# Đường dẫn đến các file .pth và dữ liệu
PATH_P2P = r"D:\UNIVERSITY\LAB\Simu\checkpoints\pix2pix_final.pth"
PATH_CGAN = r"D:\UNIVERSITY\LAB\Simu\checkpoints_cgan\cgan_final.pth"
DATA_ROOT = r"D:\UNIVERSITY\LAB\Simu\processed_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nạp mô hình Pix2Pix và cCGAN
print("Loading mô hình Pix2Pix và cCGAN...")
model_p2p = UNetGenerator(1, 1).to(DEVICE)
model_p2p.load_state_dict(torch.load(PATH_P2P))
model_p2p.eval()

model_cgan = UNetGeneratorCCGAN(1, 1, num_classes=2).to(DEVICE)
model_cgan.load_state_dict(torch.load(PATH_CGAN))
model_cgan.eval()

#  TỰ ĐỘNG CHỌN MẪU BỆNH LÝ (LABEL 1) 
def get_random_sample():
    label_path = os.path.join(DATA_ROOT, "1") # Ưu tiên bốc ca bệnh để thấy sự khác biệt
    pids = [p for p in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, p))]
    pid = random.choice(pids)
    p_path = os.path.join(label_path, pid)
    masks = [f for f in os.listdir(p_path) if f.startswith('mask_')]
    m_name = random.choice(masks)
    i_name = m_name.replace('mask_', 'img_')
    return os.path.join(p_path, m_name), os.path.join(p_path, i_name), pid

mask_path, img_path, pid = get_random_sample()
mask_np = np.load(mask_path).astype(np.float32)
real_np = np.load(img_path).astype(np.float32)

# Tiền xử lý dữ liệu
mask_res = cv2.resize(mask_np, (256, 256))
mask_tensor = torch.from_numpy(mask_res / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
label_tensor = torch.tensor([1]).to(DEVICE) # Vẽ bệnh lý mã 59

# AI THỰC HIỆN VẼ 
with torch.no_grad():
    # Pix2Pix vẽ phổi khỏe
    fake_p2p = model_p2p(mask_tensor).squeeze().cpu().numpy()
    # cCGAN vẽ phổi bệnh
    fake_cgan = model_cgan(mask_tensor, label_tensor).squeeze().cpu().numpy()

# Chuyển đổi dải giá trị từ [-1, 1] về [0, 1] để hiển thị
fake_p2p = (fake_p2p + 1) / 2.0
fake_cgan = (fake_cgan + 1) / 2.0
real_norm = (real_np - real_np.min()) / (real_np.max() - real_np.min() + 1e-5)

# VẼ BẢNG SO SÁNH 4 CỘT
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
images = [mask_res, real_norm, fake_p2p, fake_cgan]
titles = ["1. Mask (Input)", "2. Real CT (Ground Truth)", 
          "3. Pix2Pix (Healthy)", "4. cCGAN (Pathology 59)"]

for i in range(4):
    # Sửa lỗi mask đen bằng cách ép dải hiển thị
    v_max = 1.0 if i == 0 else images[i].max()
    axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=max(1.0, v_max))
    axes[i].set_title(titles[i], fontsize=12)
    axes[i].axis('off')

plt.suptitle(f"Kết quả so sánh mô hình - Bệnh nhân: {pid}", fontsize=16)
plt.tight_layout()
plt.savefig(f"final_comparison_{pid}.png", dpi=300)
print(f"Đã lưu bảng so sánh 4 cột cho bệnh nhân {pid}!")
plt.show()