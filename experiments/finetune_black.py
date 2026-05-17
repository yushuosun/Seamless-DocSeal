"""DocDiff fine-tune for black stamps (skip pseudo-red, learn black->clean directly).

策略 A: 直接在原始黑章 input vs gt 上微调, 不做颜色归一化. 训完后推理时跳过
color_normalize, 直接把黑章 crop 喂给微调后的 DocDiff.

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\finetune_black.py
        [--iters 3000] [--lr 2e-5] [--batch 8]

输出权重:
    DocDiff/checksave/seal_init_black.pth
    DocDiff/checksave/seal_denoiser_black.pth
"""
from __future__ import annotations
import argparse, os, sys, time, math, random
from pathlib import Path

# 确保 import 路径正确 (DocDiff 用相对的 src.* / model.* / schedule.*)
HERE = Path(__file__).parent
DOCDIFF = HERE / "DocDiff"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOCDIFF))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

from DocDiff.model.DocDiff import DocDiff
from DocDiff.schedule.schedule import Schedule
from DocDiff.schedule.diffusionSample import GaussianDiffusion
from DocDiff.src.sobel import Laplacian


# ── 数据 ──────────────────────────────────────────────────────────
class StampPairDataset(Dataset):
    """从 (input, gt) pair 里随机 crop 128x128 patch.

    50% 的 patch 偏向章中心 (用 meta.csv 指引), 50% 完全随机, 兼顾章去除和无章保留.
    """
    def __init__(self, input_dir, gt_dir, meta_csv, image_size=128, stamp_bias=0.7):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.stamp_bias = stamp_bias

        import csv
        self.entries = []
        with open(meta_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fname = row["file"]
                if not (self.input_dir / fname).exists():
                    continue
                if not (self.gt_dir / fname).exists():
                    continue
                self.entries.append({
                    "file": fname,
                    "x": int(row["x"]), "y": int(row["y"]),
                    "w": int(row["stamp_w"]), "h": int(row["stamp_h"]),
                })
        print(f"[dataset] {len(self.entries)} pairs from {self.input_dir}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        inp = Image.open(self.input_dir / e["file"]).convert("RGB")
        gt  = Image.open(self.gt_dir   / e["file"]).convert("RGB")
        W, H = inp.size

        s = self.image_size
        if random.random() < self.stamp_bias:
            # 偏向章区域: 章中心 +- 随机偏移
            cx = e["x"] + e["w"] // 2 + random.randint(-e["w"]//3, e["w"]//3)
            cy = e["y"] + e["h"] // 2 + random.randint(-e["h"]//3, e["h"]//3)
            x1 = max(0, min(W - s, cx - s // 2))
            y1 = max(0, min(H - s, cy - s // 2))
        else:
            x1 = random.randint(0, max(0, W - s))
            y1 = random.randint(0, max(0, H - s))

        inp = inp.crop((x1, y1, x1 + s, y1 + s))
        gt  = gt.crop((x1, y1, x1 + s, y1 + s))

        # 增强: 随机水平翻转 + 轻微旋转 (input 和 gt 同步)
        if random.random() < 0.5:
            inp = TF.hflip(inp); gt = TF.hflip(gt)
        ang = random.uniform(-8, 8)
        if abs(ang) > 0.5:
            inp = TF.rotate(inp, ang, fill=255)
            gt  = TF.rotate(gt,  ang, fill=255)

        return TF.to_tensor(inp), TF.to_tensor(gt)


# ── 训练 ──────────────────────────────────────────────────────────
def make_config():
    return type("Cfg", (), {
        "IMAGE_SIZE": [128, 128],
        "CHANNEL_X": 3, "CHANNEL_Y": 3,
        "MODEL_CHANNELS": 32, "NUM_RESBLOCKS": 1,
        "CHANNEL_MULT": [1, 2, 3, 4],
        "TIMESTEPS": 100, "SCHEDULE": "linear",
        "PRE_ORI": "True", "BETA_LOSS": 50,
        "HIGH_LOW_FREQ": "True",
    })()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\input")
    ap.add_argument("--gt_dir",    default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\gt")
    ap.add_argument("--meta",      default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\meta.csv")
    ap.add_argument("--init_w",    default=str(DOCDIFF / "checksave" / "seal_init.pth"))
    ap.add_argument("--den_w",     default=str(DOCDIFF / "checksave" / "seal_denoiser.pth"))
    ap.add_argument("--out_init",  default=str(DOCDIFF / "checksave" / "seal_init_black.pth"))
    ap.add_argument("--out_den",   default=str(DOCDIFF / "checksave" / "seal_denoiser_black.pth"))
    ap.add_argument("--iters",     type=int, default=3000)
    ap.add_argument("--lr",        type=float, default=2e-5)
    ap.add_argument("--batch",     type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  (torch {torch.__version__})")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    cfg = make_config()
    net = DocDiff(
        input_channels=cfg.CHANNEL_X + cfg.CHANNEL_Y,
        output_channels=cfg.CHANNEL_Y,
        n_channels=cfg.MODEL_CHANNELS,
        ch_mults=cfg.CHANNEL_MULT,
        n_blocks=cfg.NUM_RESBLOCKS,
    ).to(device)

    # 加载预训练 (red) 权重
    print(f"[init weights] {args.init_w}")
    net.init_predictor.load_state_dict(torch.load(args.init_w, map_location=device))
    print(f"[denoiser weights] {args.den_w}")
    net.denoiser.load_state_dict(torch.load(args.den_w, map_location=device))

    schedule = Schedule(cfg.SCHEDULE, cfg.TIMESTEPS)
    diffusion = GaussianDiffusion(net.denoiser, cfg.TIMESTEPS, schedule).to(device)
    high_filter = Laplacian().to(device)
    loss_fn = nn.MSELoss()

    ds = StampPairDataset(args.input_dir, args.gt_dir, args.meta,
                          image_size=cfg.IMAGE_SIZE[0], stamp_bias=0.7)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    Path(args.out_init).parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] iters={args.iters}  batch={args.batch}  lr={args.lr}")
    iteration = 0
    t0 = time.time()
    losses = []
    while iteration < args.iters:
        for img, gt in dl:
            if iteration >= args.iters:
                break
            img = img.to(device, non_blocking=True)
            gt  = gt.to(device,  non_blocking=True)
            net.train()
            opt.zero_grad()

            t = torch.randint(0, cfg.TIMESTEPS, (img.shape[0],), device=device).long()
            init_predict, noise_pred, noisy_image, noise_ref = net(gt, img, t, diffusion)

            # PRE_ORI=True && HIGH_LOW_FREQ=True 同 trainer.py
            residual_high = high_filter(gt - init_predict)
            ddpm_loss = 2 * loss_fn(high_filter(noise_pred), residual_high) \
                        + loss_fn(noise_pred, gt - init_predict)
            low_high_loss = loss_fn(init_predict, gt)
            low_freq_loss = loss_fn(init_predict - high_filter(init_predict),
                                    gt - high_filter(gt))
            pixel_loss = low_high_loss + 2 * low_freq_loss
            loss = ddpm_loss + cfg.BETA_LOSS * pixel_loss / cfg.TIMESTEPS

            loss.backward()
            opt.step()

            losses.append(loss.item())
            if iteration % 50 == 0:
                avg = np.mean(losses[-50:])
                elapsed = time.time() - t0
                ips = (iteration + 1) / elapsed
                eta = (args.iters - iteration) / max(ips, 1e-3)
                print(f"  iter {iteration:5d}/{args.iters}  loss={avg:.4f}  "
                      f"ddpm={ddpm_loss.item():.4f}  pix={pixel_loss.item():.4f}  "
                      f"{ips:.1f} it/s  ETA {eta/60:.1f}m")
            iteration += 1

            if iteration % args.save_every == 0:
                torch.save(net.init_predictor.state_dict(), args.out_init)
                torch.save(net.denoiser.state_dict(), args.out_den)
                print(f"  [saved checkpoint @ iter {iteration}]")

    torch.save(net.init_predictor.state_dict(), args.out_init)
    torch.save(net.denoiser.state_dict(), args.out_den)
    print(f"\n[done] total {time.time()-t0:.1f}s")
    print(f"[saved] {args.out_init}")
    print(f"[saved] {args.out_den}")


if __name__ == "__main__":
    main()
