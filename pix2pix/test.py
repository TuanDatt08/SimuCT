import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from generator import UNetGenerator  # Nạp kiến trúc từ file của Đạt

# CẤU HÌNH ĐƯỜNG DẪN
MODEL_PATH = r"D:\UNIVERSITY\LAB\Simu\checkpoints\pix2pix_final.pth"
DATA_ROOT = r"D:\UNIVERSITY\LAB\Simu\processed_data"
SAVE_DIR = r"D:\UNIVERSITY\LAB\Simu\results\test_automated"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 5  # Số lượng bệnh nhân Đạt muốn thử tài ngẫu nhiên

# NẠP MÔ HÌNH (TRÍ TUỆ EPOCH 200) ---
model = UNetGenerator(in_c=1, out_c=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(f"✅ Đã nạp mô hình từ {MODEL_PATH}")

# QUÉT TỰ ĐỘNG TOÀN BỘ DỮ LIỆU ---
all_pairs = []
for label in ['0', '1']:  # Quét cả nhóm phổi thường và phổi bệnh
    label_path = os.path.join(DATA_ROOT, label)
    if not os.path.exists(label_path): continue
    
    for pid in os.listdir(label_path):
        pid_path = os.path.join(label_path, pid)
        if not os.path.isdir(pid_path): continue
        
        # Tìm các cặp img-mask
        imgs = [f for f in os.listdir(pid_path) if f.startswith('img_')]
        for img_name in imgs:
            mask_name = img_name.replace('img_', 'mask_')
            if os.path.exists(os.path.join(pid_path, mask_name)):
                all_pairs.append({
                    'pid': pid,
                    'label': label,
                    'img_path': os.path.join(pid_path, img_name),
                    'mask_path': os.path.join(pid_path, mask_name)
                })

print(f"🔍 Tìm thấy tổng cộng {len(all_pairs)} cặp ảnh trong dữ liệu.")

# CHỌN NGẪU NHIÊN VÀ VẼ ẢNH
selected_samples = random.sample(all_pairs, min(NUM_SAMPLES, len(all_pairs)))

for idx, sample in enumerate(selected_samples):
    # Đọc file .npy
    mask_np = np.load(sample['mask_path']).astype(np.float32)
    real_np = np.load(sample['img_path']).astype(np.float32)
    
    # Tiền xử lý (Resize về 256x256 và chuẩn hóa)
    mask_res = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy(mask_res / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # AI thực hiện vẽ (Inference)
    with torch.no_grad():
        fake_tensor = model(mask_tensor)
        fake_img = fake_tensor.squeeze().cpu().numpy()

    # TẠO BẢNG SO SÁNH 3 CỘT
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mask_res, cmap='gray', vmin=0, vmax=max(1, mask_res.max()))
    axes[0].axis('off')
    
    axes[1].imshow(real_np, cmap='gray')
    axes[1].set_title("2. Real CT (Original)")
    axes[1].axis('off')
    
    axes[2].imshow(fake_img, cmap='gray')
    axes[2].set_title("3. Pix2Pix (AI Generated)")
    axes[2].axis('off')
    
    # Lưu kết quả
    plt.tight_layout()
    save_name = f"test_{sample['label']}_{sample['pid']}_{idx}.png"
    plt.savefig(os.path.join(SAVE_DIR, save_name))
    plt.close()
    print(f"Đã lưu kết quả tại: {save_name}")

print(f"\nHoàn tất! Vào thư mục {SAVE_DIR} để kiểm tra kết quả.")