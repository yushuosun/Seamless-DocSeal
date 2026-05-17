!pip install ultralytics -q

import os
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm
import yaml

# ================= 1. 配置真实 Kaggle 路径 =================
# 你的印章切片路径（用 *.* 兼容 jpg 和 png）
STAMP_DIR = "/kaggle/input/datasets/yushuosun/seal-dataset/only_stamps/words_under_stamps/seal_0/red/*.*" 
BASE_DIR = "/kaggle/working/yolo_dataset"

# 创建 YOLO 所需的目录结构
for split in ['train', 'val']:
    os.makedirs(os.path.join(BASE_DIR, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, f"labels/{split}"), exist_ok=True)

stamp_paths =[p for p in glob(STAMP_DIR) if p.endswith(('.jpg', '.jpeg', '.png'))]

if not stamp_paths:
    raise ValueError(f"❌ 未找到印章图片，请检查数据集是否正确挂载到 {STAMP_DIR}")
else:
    print(f"✅ 成功找到 {len(stamp_paths)} 张印章切片，准备生成合成训练数据...")

# ================= 2. 自动合成数据 (Copy-Paste) =================
def generate_synthetic_data(num_samples=1000):
    for i in tqdm(range(num_samples), desc="合成虚拟文档中"):
        # 1. 创建模拟文档背景 (宽 800-1000, 高 1000-1400)
        bg_w = random.randint(800, 1000)
        bg_h = random.randint(1000, 1400)
        # 模拟纸张颜色（白纸偏一点灰度）
        bg_color = random.randint(235, 255) 
        bg_img = np.full((bg_h, bg_w, 3), bg_color, dtype=np.uint8)
        
        # 2. 随机选一张印章
        stamp_path = random.choice(stamp_paths)
        stamp_img = cv2.imread(stamp_path)
        if stamp_img is None: continue
            
        sh, sw = stamp_img.shape[:2]
        
        # 3. 数据增强：随机缩放 (模拟不同尺寸的印章)
        scale = random.uniform(0.7, 1.3)
        new_w, new_h = int(sw * scale), int(sh * scale)
        stamp_img = cv2.resize(stamp_img, (new_w, new_h))
        sh, sw = stamp_img.shape[:2]
        
        if sw >= bg_w or sh >= bg_h: continue
            
        # 4. 随机确定印章在文档上的位置
        x1 = random.randint(0, bg_w - sw)
        y1 = random.randint(0, bg_h - sh)
        x2, y2 = x1 + sw, y1 + sh
        
        # 将印章贴到背景上
        bg_img[y1:y2, x1:x2] = stamp_img
        
        # 5. 计算 YOLO 格式的中心点坐标标签 (归一化到 0~1)
        cx = (x1 + x2) / 2.0 / bg_w
        cy = (y1 + y2) / 2.0 / bg_h
        w = sw / bg_w
        h = sh / bg_h
        
        # 6. 按 8:2 划分训练集和验证集
        split = 'train' if random.random() < 0.8 else 'val'
        
        # 7. 写入图片和标签
        img_name = f"synth_doc_{i:05d}.jpg"
        label_name = f"synth_doc_{i:05d}.txt"
        
        cv2.imwrite(os.path.join(BASE_DIR, f"images/{split}", img_name), bg_img)
        with open(os.path.join(BASE_DIR, f"labels/{split}", label_name), "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n") # 类别0: seal

# 生成 1500 张合成数据用来训练（几秒钟就能生成完）
generate_synthetic_data(1500)

# ================= 3. 创建 YOLO 配置文件 yaml =================
yaml_data = {
    'path': BASE_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': ['seal']
}
with open('/kaggle/working/seal_data.yaml', 'w') as f:
    yaml.dump(yaml_data, f)
    
print("✅ 数据集构建完毕！已生成 seal_data.yaml。请运行下一个 Cell 开始训练。")

