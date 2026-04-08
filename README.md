# Seamless-DocSeal
End-to-End Seal Detection and Removal Framework
# 🪪 Seamless-DocSeal: End-to-End Seal Detection and Removal Framework
**基于 YOLOv8 与局部扩散模型的高效文档印章检测与去除框架**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](#english) | [中文](#chinese)

---

<a id="english"></a>
## 📖 Introduction
Removing seals (stamps) from document images and restoring the underlying text is a challenging task in Document Image Analysis (DIA). Directly feeding high-resolution document images (e.g., 4K scans) into Diffusion Models (like DocDiff) often leads to **Out-Of-Memory (OOM)** issues and unexpected degradation of original non-seal regions.

To address this, we propose an efficient **"Detect - Local Diffuse - Stitch"** pipeline:
1. **Accurate Detection**: Utilizing a custom-trained YOLOv8s model to accurately locate seals.
2. **Local Processing**: Cropping the seal regions (padded to multiples of 8) and feeding ONLY the patches into the DocDiff diffusion model.
3. **Seamless Stitching**: Replacing the restored patches back into the original high-resolution document.

### ✨ Key Features
- **🚀 Ultra-Low VRAM Consumption**: By avoiding full-image diffusion processing, this pipeline reduces VRAM usage and speeds up inference by >10x.
- **🖼️ High-Fidelity Preservation**: Non-seal regions remain 100% untouched, ensuring absolute maximum PSNR/SSIM metrics for the document background.
- **📦 Ready-to-Use Weights**: We release our robust `SealDet-YOLOv8s` model, trained on heavily augmented synthetic datasets (robust to text occlusion, various colors, and shapes).

## 🧰 Pre-trained Models
| Model | Description | Download |
| --- | --- | --- |
| `SealDet-YOLOv8s.pt` | YOLOv8s fine-tuned for seal/stamp detection on complex documents. | [Link]() (Update your link) |

*(If this project helps your research, please consider citing our work!)*

---

<a id="chinese"></a>
## 📖 项目简介 (中文)
在文档图像处理中，去除印章（公章）并还原被遮挡的底字是一项极具挑战的任务。如果将高分辨率的文档原图（如 4K 扫描件）直接送入扩散模型（如 DocDiff）处理，极易导致 **显存溢出 (OOM)**，且会破坏文档中未盖章区域的原始画质。

为此，本项目提出了一种高效的 **“定位裁剪 - 局部扩散 - 像素级缝合”** 框架：
1. **精准定位裁剪**：使用针对印章特殊微调的 YOLOv8s 模型，精准框选印章区域并裁剪（自动规整为 8 的倍数）。
2. **局部无损去印**：仅将印章切片送入 DocDiff 扩散模型进行去印和底字还原运算。
3. **完美无缝还原**：将处理干净的切片精准贴回高分辨率原图。

### ✨ 核心优势
- **🚀 极低显存消耗**：避免了全图扩散运算，将显存压力降至最低，处理速度提升 10 倍以上。
- **🖼️ 完美画质保留**：文档中未盖章区域的像素得到 100% 保留，在 PSNR/SSIM 等学术评测指标上具有巨大优势。
- **📦 开箱即用的权重**：开源了基于大量“遮挡/复杂背景”合成数据训练的印章检测预训练模型 `SealDet-YOLOv8s`，具备极强的抗干扰与泛化能力。

## 🧰 预训练模型 (Model Zoo)
| 模型名称 | 说明 | 下载地址 |
| --- | --- | --- |
| `SealDet-YOLOv8s.pt` | 针对复杂文档抗干扰训练的印章目标检测大模型 | [点击下载]() (请填入你的链接) |
| `seal_init.pth` | DocDiff 初始预测权重 | (DocDiff官方提供) |
| `seal_denoiser.pth` | DocDiff 去噪扩散权重 | (DocDiff官方提供) |

## 🚀 快速开始 (Quick Start)

**1. 环境配置**
```bash
git clone https://github.com/yushuosun/Seamless-DocSeal.git
cd Seamless-DocSeal
pip install -r requirements.txt

kaggle地址:
https://www.kaggle.com/code/yushuosun/yolo-training
