# Seamless-DocSeal

文档黑章（也支持红/蓝章）去除流水线。
基于 [DocDiff](https://github.com/Royalvice/DocDiff) 扩展，加入：
- **YOLO11n 章检测器**（黑章 bIoU 0.92 / mAP50 0.94）
- **mask-only 合成训练**：用 mask 模板渲染合成数据微调 DocDiff，黑章合成 holdout PSNR ~46/54 dB
- **智能 paste**：基于 `|out − in|` 差分 mask 局部贴回，不破坏文档背景

```
原图(含章) ─┬─ YOLO 检测 → bbox
            │
            ├─ crop + 颜色路由
            │   ├─ 红章: DocDiff (原始权重)
            │   ├─ 蓝章: 色相平移 → DocDiff
            │   └─ 黑章: DocDiff (mask_only 微调权重)
            │
            └─ DocDiff 输出 → 智能 paste → 干净文档
```

---

## 环境

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

CUDA 推荐 PyTorch 2.0+。本项目在 RTX 5060 + CUDA 12.x 上开发。

---

## 权重

权重未随仓库发布，请按下面任一方式获取：

**方案 A: 用项目内提供的脚本自己训（详细见下）**

**方案 B: 使用 DocDiff 官方红章权重（仅红章有效）**
- 从 [DocDiff 官方](https://github.com/Royalvice/DocDiff) 获取 `init.pth` / `denoiser.pth`，放到 `DocDiff/checksave/`

需要的权重文件（放在 `DocDiff/checksave/`）：

| 文件 | 作用 | 来源 |
|---|---|---|
| `seal_init.pth` / `seal_denoiser.pth` | 红/蓝章 DocDiff (原始) | DocDiff 官方 |
| `seal_init_black_mask_only_long_ema.pth` / `seal_denoiser_black_mask_only_long_ema.pth` | 黑章微调权重 | 用本仓库脚本训 |
| `stamp_segmenter.pth` | UNet segmenter (可选 detector) | `python seg_unet.py train` |

YOLO 权重 (`yolo_runs/best.pt`) 用 `yolo_train_eval.py` 训。

---

## 端到端推理

```bash
python full_pipeline_yolo_docdiff.py \
  --dir2 /path/to/test_images \
  --out_root /path/to/output \
  --docdiff_init  seal_init_black_mask_only_long_ema.pth \
  --docdiff_den   seal_denoiser_black_mask_only_long_ema.pth \
  --yolo_weight   yolo_runs/black_mask_only_v1/weights/best.pt \
  --conf 0.25
```

每张图输出：
- `<stem>.png` — 去章结果
- `<stem>_compare.jpg` — input+bbox / output / |out-in| 三列对比
- `<stem>_bbox.txt` — 检测到的 bbox

---

## 训练流水线

整个项目分三步：合成训练数据 → 训 detector → 训 DocDiff 黑章微调。

### Step 1: 合成训练数据

需要章 mask 模板池 + 干净文档页池（路径在脚本里改）。

```bash
python synth_black_mask_only_v3.py --n 6000 --workers 4
```

输出：
```
synth_black_mask_only_v3/
├── input/         (6000 png, 带章)
├── gt/            (6000 png, 干净版)
├── stamp_mask/    (6000 png, 章像素 mask)
└── meta.csv
```

约 8-12 min（4 进程，依硬盘速度）。

### Step 2: 训 YOLO 黑章检测器

```bash
# 准备 YOLO 数据格式
python yolo_prepare_data.py --symlink

# 训练 (50 epoch, ~30-50 min on RTX 5060)
python yolo_train_eval.py train --epochs 50 --imgsz 640 --batch 16

# 评估 holdout
python yolo_train_eval.py eval --n 20
```

参考结果：bIoU 0.92 / mAP50 0.94 / recall 0.94。

### Step 3: 微调 DocDiff 处理黑章

从 DocDiff 官方红章权重 fine-tune（需要在 `DocDiff/checksave/` 放 `seal_init.pth` + `seal_denoiser.pth`）：

```bash
python finetune_black_realistic_long.py \
  --iters 120000 --batch 8 --lr 5e-6 \
  --image_size 128 --holdout 80 \
  --ema_decay 0.9997 --stamp_weight 5.5 \
  --stamp_bias 0.88 --eval_every 1000
```

参考结果：合成 holdout PSNR ~44-46 / stamp_psnr ~53-54。

---

## 关键设计决策

1. **mask-only 合成 > 真实章合成**
   真实裁剪章带原文档文字，Otsu 分不开 → 模型学到"擦章 + 顺带擦周围文字" → 弃。
   mask 模板 = 干净章形状，简单纯黑渲染，泛化反而更好。
   *形状先验 > 纹理先验。*

2. **颜色路由独立权重**
   红/蓝/黑章各用各权重，互不影响：
   - 红 → `seal_init.pth`（原始）
   - 蓝 → `seal_init.pth` + `color_normalize.py` 蓝→红
   - 黑 → `seal_init_black_mask_only_long_ema.pth` + `skip_normalize=True`

3. **智能 paste**
   不用 bbox 整片替换，用 `|DocDiff_out - input| > 25` 差分 mask 决定贴回哪些像素 → DocDiff 没动的区域保留原图，避免误改文档背景。

4. **YOLO > seg_unet > CV detector**
   - CV (HSV + 形态学): bIoU 0.118，对真实退化章完全失效
   - seg_unet (10k iter @ 512): bIoU 0.895, ~50ms/张, 需后处理
   - **YOLO11n (50 epoch @ 640): bIoU 0.920, 3.3ms/张, 天然无文字段 FP**

---

## 仓库结构

```
.
├── README.md
├── LICENSE                              MIT
├── requirements.txt
├── .gitignore
│
├── kaggle_docdiff_multicolor.py         核心推理类 DocDiffCropRunner
├── full_pipeline_yolo_docdiff.py        生产 pipeline (YOLO + DocDiff)
├── color_normalize.py                   蓝/黑章颜色归一化
├── seal_detector_v2.py                  CV detector (红/蓝)
│
├── yolo_prepare_data.py                 YOLO 数据格式转换
├── yolo_train_eval.py                   YOLO 训练 / 评估
├── seg_unet.py                          UNet segmenter (备用 detector)
├── finetune_black_realistic_long.py     DocDiff 黑章微调
├── synth_black_mask_only_v3.py          黑章数据合成
│
├── eval_holdout.py                      端到端 PSNR 评估
├── eval_black_detection.py              detector IoU 评估
├── make_compare_sheet.py                可视化工具
│
├── DocDiff/                             DocDiff 库 (MIT, 见 DocDiff/LICENSE)
│   ├── model/  schedule/  src/  utils/
│   └── data/  demo/  conf.yml
│
└── experiments/                         开发过程脚本 (保留作记录)
    ├── diagnose_black_recall.py
    ├── finetune_black.py / finetune_black_v2.py
    ├── synth_black_real_v2.py           ← 真实章合成 (已证文字污染)
    └── ... (其他早期实验)
```

---

## 不在仓库中的资源

| 项 | 大小 | 获取方式 |
|---|---|---|
| 训练权重 (`*.pth`) | ~376 MB | 自行训练 |
| 合成数据集 | ~10 GB | 用 synth 脚本生成 |
| YOLO runs (`yolo_runs/`) | 中等 | 训练时自动生成 |

---

## 致谢

- [DocDiff](https://github.com/Royalvice/DocDiff) — diffusion-based document enhancement backbone
- [Ultralytics YOLO11](https://docs.ultralytics.com/) — detector
- 章 mask 数据集来自 SealData 公开数据

## License

MIT License (本仓库主代码)
DocDiff 子目录采用其原始 MIT 协议（见 `DocDiff/LICENSE`）。
