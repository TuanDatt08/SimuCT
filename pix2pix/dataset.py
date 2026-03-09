import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        # Quét thư mục 0 (bình thường) và 1 (bệnh lý mã 59)
        for label in ['0', '1']:
            path = os.path.join(root, label)
            if not os.path.exists(path): continue
            for pid in os.listdir(path):
                p_path = os.path.join(path, pid)
                if not os.path.isdir(p_path): continue
                # Tìm tất cả file ảnh, sau đó tìm mask tương ứng
                imgs = [f for f in os.listdir(p_path) if f.startswith('img_')]
                for f in imgs:
                    m_file = f.replace('img_', 'mask_')
                    if os.path.exists(os.path.join(p_path, m_file)):
                        self.samples.append((os.path.join(p_path, m_file), os.path.join(p_path, f)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load dữ liệu npy
        mask = np.load(self.samples[idx][0]).astype(np.float32)
        img = np.load(self.samples[idx][1]).astype(np.float32)

        # Resize chuẩn hóa kích thước để tránh lỗi stack tensor
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Chuẩn hóa pixel
        mask = mask / 255.0  # Về [0, 1]
        # Đưa ảnh CT về [-1, 1] để khớp với hàm Tanh của Generator
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = img * 2 - 1

        return {
            "A": torch.from_numpy(mask).unsqueeze(0).float(), 
            "B": torch.from_numpy(img).unsqueeze(0).float()
        }