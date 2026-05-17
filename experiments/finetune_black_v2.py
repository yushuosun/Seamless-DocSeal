"""DocDiff fine-tune v2 - EMA + 章区域加权 loss + 留出验证.

改进:
1. EMA: 维护一份指数平滑权重, 最终保存 EMA 版 (+0.5~1.5 PSNR 经验值)
2. 加权 loss: 用 stamp_mask/ 的像素级 mask, 章像素 loss 权重 ×5
3. 留出验证: 训练时跳过最后 N (默认 20) 张, 每 N iter 报 patch PSNR

数据要求:
    {input_dir, gt_dir, stamp_mask_dir} 同名 PNG
    meta.csv 含列: file, x, y, stamp_w, stamp_h

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\finetune_black_v2.py
        [--iters 8000] [--batch 8] [--lr 2e-5]
        [--input_dir ...] [--gt_dir ...] [--stamp_mask_dir ...]
        [--holdout 20]   # 末尾 20 张不参与训练
        [--ema_decay 0.9995]
        [--stamp_weight 5.0]

输出:
    seal_init_black_v2.pth     (最终步权重)
    seal_denoiser_black_v2.pth
    seal_init_black_v2_ema.pth      (EMA 版)
    seal_denoiser_black_v2_ema.pth
"""
from __future__ import annotations
import argparse, csv, os, sys, time, copy, random
from pathlib import Path

HERE = Path(__file__).parent
DOCDIFF = HERE / "DocDiff"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOCDIFF))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

from DocDiff.model.DocDiff import DocDiff
from DocDiff.schedule.schedule import Schedule
from DocDiff.schedule.diffusionSample import GaussianDiffusion
from DocDiff.src.sobel import Laplacian


# ── 数据 ──────────────────────────────────────────────────────────
class StampPairDataset(Dataset):
    """从 (input, gt, stamp_mask) triplet 里随机 crop 128x128.

    stamp_bias 概率裁剪到章中心附近 (用 meta.csv 的 x,y,w,h),
    其余完全随机 (覆盖无章 patch, 保持模型对纯文档不动手).
    """
    def __init__(self, input_dir, gt_dir, mask_dir, meta_csv, image_size=128,
                 stamp_bias=0.7, holdout_files=None):
        self.input_dir = Path(input_dir)
        self.gt_dir = Path(gt_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.stamp_bias = stamp_bias
        holdout = set(holdout_files or [])

        self.entries = []
        with open(meta_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fname = row["file"]
                if fname in holdout:
                    continue
                if not (self.input_dir / fname).exists():
                    continue
                if not (self.gt_dir / fname).exists():
                    continue
                self.entries.append({
                    "file": fname,
                    "x": int(row["x"]), "y": int(row["y"]),
                    "w": int(row["stamp_w"]), "h": int(row["stamp_h"]),
                })
        print(f"[dataset] {len(self.entries)} pairs (holdout {len(holdout)})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        inp = Image.open(self.input_dir / e["file"]).convert("RGB")
        gt  = Image.open(self.gt_dir   / e["file"]).convert("RGB")
        mask_path = self.mask_dir / e["file"]
        if mask_path.exists():
            mask_pil = Image.open(mask_path).convert("L")
        else:
            mask_pil = None
        W, H = inp.size

        s = self.image_size
        if random.random() < self.stamp_bias:
            cx = e["x"] + e["w"] // 2 + random.randint(-e["w"]//3, e["w"]//3)
            cy = e["y"] + e["h"] // 2 + random.randint(-e["h"]//3, e["h"]//3)
            x1 = max(0, min(W - s, cx - s // 2))
            y1 = max(0, min(H - s, cy - s // 2))
        else:
            x1 = random.randint(0, max(0, W - s))
            y1 = random.randint(0, max(0, H - s))

        inp = inp.crop((x1, y1, x1 + s, y1 + s))
        gt  = gt.crop((x1, y1, x1 + s, y1 + s))
        if mask_pil is not None:
            mask_pil = mask_pil.crop((x1, y1, x1 + s, y1 + s))

        # 增强: 翻转 + 旋转 (input/gt/mask 同步)
        if random.random() < 0.5:
            inp = TF.hflip(inp); gt = TF.hflip(gt)
            if mask_pil is not None: mask_pil = TF.hflip(mask_pil)
        ang = random.uniform(-8, 8)
        if abs(ang) > 0.5:
            inp = TF.rotate(inp, ang, fill=255)
            gt  = TF.rotate(gt,  ang, fill=255)
            if mask_pil is not None: mask_pil = TF.rotate(mask_pil, ang, fill=0)

        inp_t = TF.to_tensor(inp)
        gt_t  = TF.to_tensor(gt)
        if mask_pil is not None:
            mask_t = TF.to_tensor(mask_pil)  # 1xHxW, 0~1
        else:
            mask_t = torch.zeros(1, s, s)
        return inp_t, gt_t, mask_t


# ── 配置 ──────────────────────────────────────────────────────────
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


# ── 留出 patch 验证 ──────────────────────────────────────────────
@torch.no_grad()
def validate(net, holdout_paths, device, n_patches=40, patch=128):
    """在 holdout 图上随机 crop n 个 stamp-centered patch, 用 init_predictor 测 PSNR."""
    if not holdout_paths:
        return None
    net.eval()
    psnrs = []
    rng = random.Random(20251231)
    for inp_path, gt_path, mask_path, meta in holdout_paths[:n_patches]:
        inp = cv2_load_rgb(inp_path)
        gt  = cv2_load_rgb(gt_path)
        if inp is None or gt is None: continue
        H, W, _ = inp.shape
        # crop 章中心 patch
        cx = meta["x"] + meta["w"] // 2 + rng.randint(-meta["w"]//4, meta["w"]//4)
        cy = meta["y"] + meta["h"] // 2 + rng.randint(-meta["h"]//4, meta["h"]//4)
        x1 = max(0, min(W - patch, cx - patch // 2))
        y1 = max(0, min(H - patch, cy - patch // 2))
        inp_p = inp[y1:y1+patch, x1:x1+patch]
        gt_p  = gt[y1:y1+patch, x1:x1+patch]
        if inp_p.shape[:2] != (patch, patch): continue

        t = torch.from_numpy(inp_p).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        init = net.init_predictor(t, torch.zeros((1,), device=device, dtype=torch.long))
        pred = init.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        gt_f = gt_p.astype(np.float32)
        mse = np.mean((pred - gt_f) ** 2)
        if mse > 0:
            psnrs.append(10 * np.log10(255.0 ** 2 / mse))
    net.train()
    return float(np.mean(psnrs)) if psnrs else None


def cv2_load_rgb(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    base = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    ap.add_argument("--input_dir",      default=fr"{base}\input")
    ap.add_argument("--gt_dir",         default=fr"{base}\gt")
    ap.add_argument("--stamp_mask_dir", default=fr"{base}\stamp_mask")
    ap.add_argument("--meta",           default=fr"{base}\meta.csv")
    ap.add_argument("--init_w", default=str(DOCDIFF / "checksave" / "seal_init.pth"))
    ap.add_argument("--den_w",  default=str(DOCDIFF / "checksave" / "seal_denoiser.pth"))
    ap.add_argument("--out_init",     default=str(DOCDIFF / "checksave" / "seal_init_black_v2.pth"))
    ap.add_argument("--out_den",      default=str(DOCDIFF / "checksave" / "seal_denoiser_black_v2.pth"))
    ap.add_argument("--out_init_ema", default=str(DOCDIFF / "checksave" / "seal_init_black_v2_ema.pth"))
    ap.add_argument("--out_den_ema",  default=str(DOCDIFF / "checksave" / "seal_denoiser_black_v2_ema.pth"))

    ap.add_argument("--iters",         type=int,   default=8000)
    ap.add_argument("--lr",            type=float, default=2e-5)
    ap.add_argument("--batch",         type=int,   default=8)
    ap.add_argument("--num_workers",   type=int,   default=2)
    ap.add_argument("--save_every",    type=int,   default=1000)
    ap.add_argument("--val_every",     type=int,   default=1000)
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--ema_decay",     type=float, default=0.9995)
    ap.add_argument("--stamp_weight",  type=float, default=5.0,
                    help="章像素 loss 权重 (1.0 = 不加权)")
    ap.add_argument("--holdout",       type=int,   default=20,
                    help="末尾 N 张作为留出验证, 不参与训练")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
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

    print(f"[init_w] {args.init_w}")
    net.init_predictor.load_state_dict(torch.load(args.init_w, map_location=device))
    print(f"[den_w]  {args.den_w}")
    net.denoiser.load_state_dict(torch.load(args.den_w, map_location=device))

    # EMA 模型
    ema_net = copy.deepcopy(net)
    for p in ema_net.parameters():
        p.requires_grad_(False)
    ema_net.eval()

    schedule = Schedule(cfg.SCHEDULE, cfg.TIMESTEPS)
    diffusion = GaussianDiffusion(net.denoiser, cfg.TIMESTEPS, schedule).to(device)
    high_filter = Laplacian().to(device)

    # 准备 holdout (用 meta 末尾 N 行)
    all_meta_rows = []
    with open(args.meta, encoding="utf-8-sig") as f:
        all_meta_rows = list(csv.DictReader(f))
    all_meta_rows = [r for r in all_meta_rows if (Path(args.input_dir) / r["file"]).exists()]
    holdout_rows = all_meta_rows[-args.holdout:] if args.holdout > 0 else []
    holdout_files = {r["file"] for r in holdout_rows}
    holdout_paths = []
    for r in holdout_rows:
        holdout_paths.append((
            Path(args.input_dir) / r["file"],
            Path(args.gt_dir)    / r["file"],
            Path(args.stamp_mask_dir) / r["file"],
            {"x": int(r["x"]), "y": int(r["y"]), "w": int(r["stamp_w"]), "h": int(r["stamp_h"])},
        ))
    print(f"[holdout] {len(holdout_paths)} val samples (excluded from training)")

    ds = StampPairDataset(args.input_dir, args.gt_dir, args.stamp_mask_dir, args.meta,
                          image_size=cfg.IMAGE_SIZE[0], stamp_bias=0.7,
                          holdout_files=holdout_files)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    Path(args.out_init).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[train] iters={args.iters}  batch={args.batch}  lr={args.lr}  "
          f"ema_decay={args.ema_decay}  stamp_w={args.stamp_weight}")
    iteration = 0
    t0 = time.time()
    losses = []

    def update_ema():
        with torch.no_grad():
            for ep, p in zip(ema_net.parameters(), net.parameters()):
                ep.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)
            for eb, b in zip(ema_net.buffers(), net.buffers()):
                eb.copy_(b)

    def weighted_mse(pred, target, mask_w):
        # mask_w: B x 1 x H x W in [0,1]; 章像素权重 = stamp_weight, 其余 = 1
        w = 1.0 + (args.stamp_weight - 1.0) * mask_w
        return ((pred - target) ** 2 * w).mean()

    while iteration < args.iters:
        for img, gt, mask in dl:
            if iteration >= args.iters:
                break
            img = img.to(device, non_blocking=True)
            gt  = gt.to(device,  non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            net.train()
            opt.zero_grad()

            t = torch.randint(0, cfg.TIMESTEPS, (img.shape[0],), device=device).long()
            init_predict, noise_pred, noisy_image, noise_ref = net(gt, img, t, diffusion)

            # PRE_ORI=True && HIGH_LOW_FREQ=True (照原 trainer)
            residual_high = high_filter(gt - init_predict)
            ddpm_loss_high = 2 * weighted_mse(high_filter(noise_pred), residual_high, mask)
            ddpm_loss_full = weighted_mse(noise_pred, gt - init_predict, mask)
            ddpm_loss = ddpm_loss_high + ddpm_loss_full

            low_high_loss = weighted_mse(init_predict, gt, mask)
            low_freq_loss = weighted_mse(init_predict - high_filter(init_predict),
                                         gt - high_filter(gt), mask)
            pixel_loss = low_high_loss + 2 * low_freq_loss
            loss = ddpm_loss + cfg.BETA_LOSS * pixel_loss / cfg.TIMESTEPS

            loss.backward()
            opt.step()
            update_ema()

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

            if iteration % args.val_every == 0:
                psnr = validate(net, holdout_paths, device)
                psnr_ema = validate(ema_net, holdout_paths, device)
                print(f"  [val @ {iteration}] holdout init-PSNR={psnr:.2f}  ema={psnr_ema:.2f}"
                      if psnr is not None else f"  [val @ {iteration}] no holdout")

            if iteration % args.save_every == 0:
                torch.save(net.init_predictor.state_dict(), args.out_init)
                torch.save(net.denoiser.state_dict(), args.out_den)
                torch.save(ema_net.init_predictor.state_dict(), args.out_init_ema)
                torch.save(ema_net.denoiser.state_dict(), args.out_den_ema)
                print(f"  [saved checkpoint @ iter {iteration}]")

    torch.save(net.init_predictor.state_dict(), args.out_init)
    torch.save(net.denoiser.state_dict(), args.out_den)
    torch.save(ema_net.init_predictor.state_dict(), args.out_init_ema)
    torch.save(ema_net.denoiser.state_dict(), args.out_den_ema)
    print(f"\n[done] total {time.time()-t0:.1f}s")
    print(f"[saved] {args.out_init}")
    print(f"[saved] {args.out_den}")
    print(f"[saved] {args.out_init_ema}")
    print(f"[saved] {args.out_den_ema}")


if __name__ == "__main__":
    main()
