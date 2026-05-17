# 🪪 Seamless-DocSeal: End-to-End Seal Detection and Removal Framework

**基于 YOLO 与局部扩散模型的高效文档印章检测与去除框架**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [中文](#chinese)

---

<a id="english"></a>
## 📖 Introduction

Removing seals (stamps) from document images and restoring the underlying text is a challenging task in Document Image Analysis (DIA). Directly feeding high-resolution document images (e.g., 4K scans) into Diffusion Models (like DocDiff) often leads to **Out-Of-Memory (OOM)** issues and unexpected degradation of original non-seal regions.

We propose an efficient **"Detect → Local Diffuse → Smart Paste"** pipeline:

1. **Accurate Detection** — YOLO11n trained on mask-only synthetic data (bIoU 0.92, mAP50 0.94, 3.3ms/img)
2. **Local Diffusion** — only the seal crops go through DocDiff, padded to multiples of 8
3. **Smart Paste** — use `|out − in| > 25` diff mask to paste back only modified pixels, preserving the background 100%

### ✨ Key Features

- **🚀 Ultra-Low VRAM** — avoid full-image diffusion, 10x+ faster inference
- **🖼️ Pixel-Perfect Background** — non-seal regions stay 100% untouched
- **🎨 Multi-color Routing** — independent weights for red / blue / black stamps
- **🧠 Counter-intuitive Finding** — mask-only synthesis (clean shape, simple rendering) generalizes better than realistic textured synthesis. *Shape prior > texture prior.*

### 🏆 Results (synthetic holdout)

| Metric | Value |
|---|---|
| Black stamp PSNR (full image / stamp region) | 46 / 54 dB |
| YOLO11n bIoU / mAP50 / Recall@0.5 | 0.92 / 0.94 / 0.94 |
| DocDiff inference per crop | ~5s (DDIM 100 steps) |

---

<a id="chinese"></a>
## 📖 项目简介 (中文)

在文档图像处理中，去除印章并还原被遮挡的底字是一项极具挑战的任务。如果将高分辨率的文档原图（如 4K 扫描件）直接送入扩散模型（如 DocDiff）处理，极易导致 **显存溢出 (OOM)**，且会破坏文档中未盖章区域的原始画质。

本项目采用 **"定位裁剪 → 局部扩散 → 智能贴回"** 框架：

```
原图(含章) ─┬─ YOLO11n 检测 → bbox
            │
            ├─ crop + 颜色路由
            │   ├─ 红章: DocDiff 原始权重
            │   ├─ 蓝章: 蓝→红色相平移 → DocDiff
            │   └─ 黑章: DocDiff (mask_only 微调权重)
            │
            └─ DocDiff 输出 → 智能 paste → 干净文档
```

### ✨ 核心特性

- **🚀 极低显存** — 避免全图扩散，速度提升 10 倍以上
- **🖼️ 完美保留底字** — 用 `|out − in|` 差分 mask 决定贴回像素，未修改区域 100% 保留
- **🎨 多色独立路由** — 红/蓝/黑章各用各权重，互不影响
- **🧠 反直觉发现** — mask-only 合成（干净形状 + 简单纯黑渲染）泛化反而比"真实纹理章合成"好。*形状先验 > 纹理先验。*

---

## 🧰 预训练模型 (Pre-trained Models)

权重未随仓库发布（376 MB），用项目脚本自行训练（详见下方训练流水线），或使用 DocDiff 官方红章权重。

| 文件 | 作用 | 来源 |
|---|---|---|
| `seal_init.pth` / `seal_denoiser.pth` | 红/蓝章 DocDiff | [DocDiff 官方](https://github.com/Royalvice/DocDiff) |
| `seal_init_black_mask_only_long_ema.pth` / `seal_denoiser_black_mask_only_long_ema.pth` | 黑章微调权重 (PSNR 46/54) | `finetune_black_realistic_long.py` |
| `yolo_runs/.../best.pt` | YOLO11n 章检测器 | `yolo_train_eval.py train` |
| `stamp_segmenter.pth` | UNet segmenter (备选) | `seg_unet.py train` |

**外部资源**：
- Kaggle 训练 notebook: <https://www.kaggle.com/code/yushuosun/yolo-training>
- 印章/文档数据集: <https://www.kaggle.com/datasets/yushuosun/seal-dataset>

---

## 🚀 快速开始 (Quick Start)

```bash
git clone https://github.com/yushuosun/Seamless-DocSeal.git
cd Seamless-DocSeal
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

CUDA 推荐 PyTorch 2.0+，本项目在 RTX 5060 + CUDA 12.x 上开发。

### 端到端推理

```bash
python full_pipeline_yolo_docdiff.py \
  --dir2 /path/to/test_images \
  --out_root /path/to/output \
  --docdiff_init seal_init_black_mask_only_long_ema.pth \
  --docdiff_den  seal_denoiser_black_mask_only_long_ema.pth \
  --yolo_weight  yolo_runs/black_mask_only_v1/weights/best.pt \
  --conf 0.25
```

每张图输出：
- `<stem>.png` — 去章结果
- `<stem>_compare.jpg` — input+bbox / output / |out-in| 三列对比
- `<stem>_bbox.txt` — 检测 bbox

---

## 🛠️ 训练流水线

三步：合成训练数据 → 训 detector → 训 DocDiff 黑章微调。

### Step 1: 合成训练数据

```bash
python synth_black_mask_only_v3.py --n 6000 --workers 4
```
（章 mask 模板池 + 干净文档页池路径在脚本里改）

输出 `synth_black_mask_only_v3/` 含 `input/ gt/ stamp_mask/ meta.csv`（各 6000 张）。约 8-12 min。

### Step 2: 训 YOLO 黑章检测器

```bash
python yolo_prepare_data.py --symlink                                  # 转 YOLO 格式
python yolo_train_eval.py train --epochs 50 --imgsz 640 --batch 16     # 训练
python yolo_train_eval.py eval --n 20                                  # 评估
```
参考结果：bIoU 0.92 / mAP50 0.94 / recall 0.94。

也可用 `generate_dataset.py` + `train_seal_detector.py` 在 Kaggle 训红章 YOLO（早期脚本，保留）。

### Step 3: 微调 DocDiff 处理黑章

从 DocDiff 官方红章权重 fine-tune：

```bash
python finetune_black_realistic_long.py \
  --iters 120000 --batch 8 --lr 5e-6 \
  --image_size 128 --holdout 80 \
  --ema_decay 0.9997 --stamp_weight 5.5 \
  --stamp_bias 0.88 --eval_every 1000
```
参考结果：合成 holdout PSNR ~44-46 / stamp_psnr ~53-54。

---

## 🧪 关键设计决策

| 决策 | 原因 |
|---|---|
| **mask-only 合成 > 真实章合成** | 真实裁剪章带原文档文字，Otsu 分不开 → 模型学到"擦章+顺带擦周围文字"。mask 模板 = 干净形状 → 泛化更好。**形状先验 > 纹理先验。** |
| **YOLO11n > seg_unet > CV detector** | CV (HSV+形态学) bIoU 0.118 全失效；seg_unet 0.895 需后处理过滤文字段 FP；YOLO11n **0.920 + 3.3ms** 天然无文字段 FP。 |
| **颜色路由独立** | 红/蓝/黑章各用独立权重文件，新训黑章不影响红章能力。 |
| **智能 paste** | `|DocDiff_out - input| > 25` 差分 mask，避免误改文档背景。 |
| **数据增强不旋转文档** | 真实文档基本正向，`±8°` 旋转是分布外样本。 |

---

## 📁 仓库结构

```
.
├── README.md  LICENSE  requirements.txt  .gitignore
│
├── kaggle_docdiff_multicolor.py         核心推理类 DocDiffCropRunner
├── full_pipeline_yolo_docdiff.py        生产 pipeline (YOLO + DocDiff)
├── color_normalize.py                   蓝/黑章颜色归一化
├── seal_detector_v2.py                  CV detector (红/蓝)
│
├── yolo_prepare_data.py                 YOLO 数据格式转换
├── yolo_train_eval.py                   YOLO 训练 / 评估
├── seg_unet.py                          UNet segmenter (备用)
├── finetune_black_realistic_long.py     DocDiff 黑章微调
├── synth_black_mask_only_v3.py          黑章数据合成
│
├── eval_holdout.py                      端到端 PSNR 评估
├── eval_black_detection.py              detector IoU 评估
├── make_compare_sheet.py                可视化工具
│
├── DocDiff/                             DocDiff 库 (MIT)
│   ├── model/  schedule/  src/  utils/
│   └── data/  demo/  conf.yml
│
├── generate_dataset.py                  Kaggle 红章 YOLO 数据合成
├── train_seal_detector.py               Kaggle 红章 YOLO 训练
│
└── experiments/                         开发过程脚本 (保留作记录)
    ├── synth_black_real_v2.py           真实章合成 (已证文字污染)
    └── ... (其他早期实验)
```

---

## 📦 不在仓库中的资源

| 项 | 大小 | 获取方式 |
|---|---|---|
| 训练权重 (`*.pth`) | ~376 MB | 自行训练（或 DocDiff 官方红章权重） |
| 合成数据集 | ~10 GB | `synth_black_mask_only_v3.py` 生成 |
| YOLO runs (`yolo_runs/`) | 中等 | 训练时自动生成 |

---

## 🙏 致谢

- [DocDiff](https://github.com/Royalvice/DocDiff) — diffusion-based document enhancement backbone (MIT)
- [Ultralytics YOLO11](https://docs.ultralytics.com/) — detector framework
- 章 mask 数据集来自 SealData 公开数据

## 📜 License

MIT License (本仓库主代码)。DocDiff 子目录采用其原始 MIT 协议（见 `DocDiff/LICENSE`）。
