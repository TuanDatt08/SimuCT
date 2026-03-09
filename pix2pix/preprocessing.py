import os
import pydicom
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# ĐƯỜNG DẪN
ROOT_DIR = r"D:\UNIVERSITY\LAB\New folder"
CSV_PATH = r"D:\UNIVERSITY\LAB\Simu\nlst_780_ctab_idc_20210527.csv"
OUTPUT_BASE = r"D:\UNIVERSITY\LAB\Simu\processed_data"

def get_advanced_lung_mask(image):
    """Tạo mặt nạ phổi chuẩn theo bài báo"""
    img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_indices = np.argsort(areas)[-2:] + 1
        mask = np.zeros_like(labels, dtype=np.uint8)
        for idx in largest_indices: mask[labels == idx] = 255
    return mask

def run_vinh_recursive_scan():
    # Đọc danh sách PID từ CSV để đối chiếu
    df = pd.read_csv(CSV_PATH)
    emphysema_dict = dict(zip(df['pid'].astype(str), df['sct_ab_desc']))
    
    print("🔍 Đang tìm kiếm mã bệnh nhân trong 'New folder'...")
    
    # Tìm tất cả các thư mục trùng với PID trong CSV
    patient_data = {}
    for root, dirs, _ in os.walk(ROOT_DIR):
        for d in dirs:
            if d in emphysema_dict:
                patient_data[d] = os.path.join(root, d)

    print(f"Đã tìm thấy {len(patient_data)} mã bệnh nhân hợp lệ.")

    # Quét sâu vào bên trong để lấy file .dcm
    success_count = 0
    for pid, p_path in tqdm(patient_data.items(), desc=" Đang xử lý DICOM"):
        all_dcm = []
        for r, _, f_list in os.walk(p_path): # "Chui sâu" vào các thư mục Ngày chụp/ID Series
            for f in f_list:
                if f.lower().endswith('.dcm'):
                    all_dcm.append(os.path.join(r, f))
        
        if len(all_dcm) < 5: continue

        # Xác định nhãn bệnh
        label = "1" if emphysema_dict[pid] == 59 else "0"
        save_dir = os.path.join(OUTPUT_BASE, label, pid)
        os.makedirs(save_dir, exist_ok=True)

        # Lấy 5 lát cắt trung tâm
        all_dcm.sort() # Sắp xếp để lấy đúng thứ tự lát cắt
        mid = len(all_dcm) // 2
        for i, fpath in enumerate(all_dcm[mid-2 : mid+3]):
            try:
                ds = pydicom.dcmread(fpath)
                img = ds.pixel_array
                mask_img = get_advanced_lung_mask(img)
                np.save(os.path.join(save_dir, f"img_{i}.npy"), img)
                np.save(os.path.join(save_dir, f"mask_{i}.npy"), mask_img)
            except: continue
        success_count += 1

    print(f"\nHoàn tất! Đã xử lý xong {success_count} ca. Tổng cộng {success_count*5*2} file .npy đã sẵn sàng!")

if __name__ == "__main__":
    run_vinh_recursive_scan()