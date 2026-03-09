import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class LungDatasetCCGAN(Dataset):
    def __init__(self, root):
        self.samples = []
        for label in ['0', '1']: # 0: Thường, 1: Bệnh 59
            path = os.path.join(root, label)
            for pid in os.listdir(path):
                p_path = os.path.join(path, pid)
                imgs = [f for f in os.listdir(p_path) if f.startswith('img_')]
                for f in imgs:
                    m_file = f.replace('img_', 'mask_')
                    if os.path.exists(os.path.join(p_path, m_file)):
                        # Lưu thêm nhãn (int)
                        self.samples.append((os.path.join(p_path, m_file), 
                                           os.path.join(p_path, f), 
                                           int(label)))

    def __getitem__(self, idx):
        mask = np.load(self.samples[idx][0]).astype(np.float32)
        img = np.load(self.samples[idx][1]).astype(np.float32)
        label = self.samples[idx][2]

        mask = cv2.resize(mask, (256, 256)) / 255.0
        img = (cv2.resize(img, (256, 256)) - 127.5) / 127.5 # Chuẩn hóa về [-1, 1]

        return {
            "A": torch.from_numpy(mask).unsqueeze(0).float(),
            "B": torch.from_numpy(img).unsqueeze(0).float(),
            "L": torch.tensor(label).long() # Trả về nhãn L
        }
    def __len__(self): return len(self.samples)