"""轻量 UNet stamp segmenter.

训练: 6000 张 (input, stamp_mask) 对像素级监督.
推理: 全图下采样 -> 预测 mask -> 上采样 -> 阈值 -> 连通块 bbox.

用法:
    # 训练
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\seg_unet.py train [--iters 5000]

    # 评估 (holdout)
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\seg_unet.py eval [--n 20]

    # 单张推理 (debug)
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\seg_unet.py infer <image_path>
"""
from __future__ import annotations
import argparse, csv, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

HERE = Path(__file__).parent
WEIGHT_DIR = HERE / "DocDiff" / "checksave"
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_PATH = WEIGHT_DIR / "stamp_segmenter.pth"


# ── UNet 架构 ─────────────────────────────────────────────────────
def conv_bn_relu(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNetSeg(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.d1 = conv_bn_relu(in_ch, base)
        self.d2 = conv_bn_relu(base, base * 2)
        self.d3 = conv_bn_relu(base * 2, base * 4)
        self.d4 = conv_bn_relu(base * 4, base * 8)
        self.bot = conv_bn_relu(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = conv_bn_relu(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = conv_bn_relu(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = conv_bn_relu(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = conv_bn_relu(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        b = self.bot(self.pool(d4))
        u4 = self.u4(torch.cat([self.up4(b), d4], 1))
        u3 = self.u3(torch.cat([self.up3(u4), d3], 1))
        u2 = self.u2(torch.cat([self.up2(u3), d2], 1))
        u1 = self.u1(torch.cat([self.up1(u2), d1], 1))
        return self.out(u1)  # logits


# ── 数据 ──────────────────────────────────────────────────────────
class SegDataset(Dataset):
    def __init__(self, input_dir, mask_dir, files, img_size=384, augment=True):
        self.input_dir = Path(input_dir)
        self.mask_dir = Path(mask_dir)
        self.files = files
        self.img_size = img_size
        self.augment = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.imread(str(self.input_dir / f))
        mask = cv2.imread(str(self.mask_dir / f), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            # 兜底: 重新随机选一个
            return self.__getitem__((idx + 1) % len(self.files))

        H, W = img.shape[:2]
        s = self.img_size
        # 随机 crop 比例 (60%-100% 短边) + resize 到 s
        if self.augment:
            min_side = min(H, W)
            crop_size = random.randint(int(min_side * 0.6), min_side)
            x = random.randint(0, W - crop_size)
            y = random.randint(0, H - crop_size)
            img = img[y:y+crop_size, x:x+crop_size]
            mask = mask[y:y+crop_size, x:x+crop_size]
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)
            if random.random() < 0.5:
                img = cv2.flip(img, 0); mask = cv2.flip(mask, 0)
            # 颜色抖动
            if random.random() < 0.5:
                a = random.uniform(0.85, 1.15); b = random.randint(-15, 15)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((mask > 30).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


def collect_files(meta_path, input_dir, mask_dir, holdout_n=20):
    rows = []
    with open(meta_path, encoding="utf-8-sig") as f:
        rows = [r["file"] for r in csv.DictReader(f)
                if (Path(input_dir) / r["file"]).exists() and (Path(mask_dir) / r["file"]).exists()]
    train, val = rows[:-holdout_n] if holdout_n > 0 else rows, rows[-holdout_n:] if holdout_n > 0 else []
    return train, val


# ── 损失 ──────────────────────────────────────────────────────────
def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = (p + target).sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def loss_fn(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


# ── 训练 ──────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    train_files, val_files = collect_files(args.meta, args.input_dir, args.mask_dir, args.holdout)
    print(f"[data] train={len(train_files)} val={len(val_files)}")

    train_ds = SegDataset(args.input_dir, args.mask_dir, train_files, img_size=args.img_size, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_ds = SegDataset(args.input_dir, args.mask_dir, val_files, img_size=args.img_size, augment=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    net = UNetSeg(base=args.base).to(device)
    print(f"[model] UNetSeg base={args.base}  params={sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    iteration = 0
    t0 = time.time()
    losses = []
    print(f"\n[train] iters={args.iters}  batch={args.batch}  lr={args.lr}  size={args.img_size}")
    while iteration < args.iters:
        for img, mask in train_dl:
            if iteration >= args.iters: break
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            net.train()
            opt.zero_grad()
            logits = net(img)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            losses.append(loss.item())

            if iteration % 50 == 0:
                avg = np.mean(losses[-50:])
                ips = (iteration + 1) / max(time.time() - t0, 1e-3)
                eta = (args.iters - iteration) / max(ips, 1e-3)
                print(f"  iter {iteration:5d}/{args.iters}  loss={avg:.4f}  {ips:.1f} it/s  ETA {eta/60:.1f}m")
            iteration += 1

            if iteration % args.val_every == 0:
                with torch.no_grad():
                    net.eval()
                    ious = []
                    for vimg, vmask in val_dl:
                        vimg = vimg.to(device); vmask = vmask.to(device)
                        vlogits = net(vimg)
                        vp = (torch.sigmoid(vlogits) > 0.5).float()
                        inter = (vp * vmask).sum(dim=(1,2,3))
                        union = ((vp + vmask) > 0).float().sum(dim=(1,2,3))
                        iou = (inter / union.clamp(min=1)).cpu().numpy().tolist()
                        ious.extend(iou)
                    print(f"  [val @ {iteration}] holdout mIoU = {np.mean(ious):.3f}")

            if iteration % args.save_every == 0:
                torch.save(net.state_dict(), str(WEIGHT_PATH))
                print(f"  [saved] {WEIGHT_PATH}")

    torch.save(net.state_dict(), str(WEIGHT_PATH))
    print(f"\n[done] total {time.time()-t0:.1f}s   saved -> {WEIGHT_PATH}")


# ── 推理 + 评估 ──────────────────────────────────────────────────
@torch.no_grad()
def predict_mask_full(net, img_bgr, device, infer_size=512, thresh=0.5):
    """全图下采样到 infer_size 预测, 上采样回原尺寸."""
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_s = cv2.resize(rgb, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb_s).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    logits = net(t)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask_full = cv2.resize((prob * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    binary = (mask_full > thresh * 255).astype(np.uint8) * 255
    return binary, mask_full


def bboxes_from_mask(mask, min_area=2000, merge_dist_frac=0.10):
    """提取 bbox. 先做形态学闭运算合并碎片, 然后小距离的 bbox 合并成一个章.

    merge_dist_frac: 中心距离 <= 章对角线 * frac 的两个 bbox 视为同一章.
    """
    H, W = mask.shape
    # 闭运算: 把碎片粘合 (基于图尺寸自适应核大小)
    k_size = max(15, min(H, W) // 40)
    if k_size % 2 == 0: k_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    raw = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < min_area: continue
        raw.append([int(x), int(y), int(x + w), int(y + h), int(a)])

    # 合并相近 bbox
    merged = []
    used = [False] * len(raw)
    for i, b in enumerate(raw):
        if used[i]: continue
        cx_i, cy_i = (b[0]+b[2])/2, (b[1]+b[3])/2
        diag_i = float(np.hypot(b[2]-b[0], b[3]-b[1]))
        cluster = [b]
        for j in range(i+1, len(raw)):
            if used[j]: continue
            c = raw[j]
            cx_j, cy_j = (c[0]+c[2])/2, (c[1]+c[3])/2
            diag_j = float(np.hypot(c[2]-c[0], c[3]-c[1]))
            d = float(np.hypot(cx_i - cx_j, cy_i - cy_j))
            if d <= max(diag_i, diag_j) * merge_dist_frac * 2:
                cluster.append(c); used[j] = True
        used[i] = True
        x1 = min(c[0] for c in cluster); y1 = min(c[1] for c in cluster)
        x2 = max(c[2] for c in cluster); y2 = max(c[3] for c in cluster)
        a_total = sum(c[4] for c in cluster)
        merged.append((x1, y1, x2, y2, a_total))

    merged.sort(key=lambda b: -b[4])
    return merged


def eval_holdout(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not WEIGHT_PATH.exists():
        print(f"weight not found: {WEIGHT_PATH}"); return
    net = UNetSeg(base=args.base).to(device)
    net.load_state_dict(torch.load(str(WEIGHT_PATH), map_location=device))
    net.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {'file':<22} {'mIoU':>7} {'bIoU':>7}  cov   over")
    print("  " + "-" * 50)
    mious, bious, covs, overs = [], [], [], []
    for i in range(args.start, args.start + args.n):
        fname = f"sample_{i:04d}.png"
        inp = cv2.imread(os.path.join(args.input_dir, fname))
        gt_mask = cv2.imread(os.path.join(args.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if inp is None or gt_mask is None: continue
        my_mask, prob = predict_mask_full(net, inp, device)

        # mask IoU
        a = (my_mask > 0); b = (gt_mask > 30)
        miou = (a & b).sum() / max(1, (a | b).sum())
        # bbox IoU
        bbs = bboxes_from_mask(my_mask)
        my_bb = bbs[0][:4] if bbs else None
        ys, xs = np.where(b)
        gt_bb = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
        if my_bb and gt_bb:
            iw = max(0, min(my_bb[2], gt_bb[2]) - max(my_bb[0], gt_bb[0]))
            ih = max(0, min(my_bb[3], gt_bb[3]) - max(my_bb[1], gt_bb[1]))
            inter = iw * ih
            ua = (my_bb[2]-my_bb[0])*(my_bb[3]-my_bb[1]) + (gt_bb[2]-gt_bb[0])*(gt_bb[3]-gt_bb[1]) - inter
            biou = inter / max(1, ua)
            cov = inter / max(1, (gt_bb[2]-gt_bb[0])*(gt_bb[3]-gt_bb[1]))
            over = ((my_bb[2]-my_bb[0])*(my_bb[3]-my_bb[1])) / max(1, (gt_bb[2]-gt_bb[0])*(gt_bb[3]-gt_bb[1]))
        else:
            biou, cov, over = 0.0, 0.0, 0.0
        mious.append(miou); bious.append(biou); covs.append(cov); overs.append(over)
        print(f"  {fname:<22} {miou:>7.3f} {biou:>7.3f}  {cov:.2f}  {over:.2f}")

        # 可视化
        prob_color = cv2.applyColorMap(prob, cv2.COLORMAP_HOT)
        my_vis = cv2.cvtColor(my_mask, cv2.COLOR_GRAY2BGR)
        gt_vis = cv2.cvtColor((gt_mask > 30).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        sheet = np.hstack([inp, prob_color, my_vis, gt_vis])
        s = 1600.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (int(sheet.shape[1]*s), int(sheet.shape[0]*s)))
        cv2.imwrite(str(out_dir / f"{fname.replace('.png','')}_panel.jpg"), sheet)

    print("  " + "-" * 50)
    if mious:
        print(f"  {'MEAN':<22} {np.mean(mious):>7.3f} {np.mean(bious):>7.3f}  "
              f"{np.mean(covs):.2f}  {np.mean(overs):.2f}")
    print(f"\n  panels saved -> {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────
def main():
    base_dir = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    pt = sp.add_parser("train")
    pt.add_argument("--input_dir", default=fr"{base_dir}\input")
    pt.add_argument("--mask_dir",  default=fr"{base_dir}\stamp_mask")
    pt.add_argument("--meta",      default=fr"{base_dir}\meta.csv")
    pt.add_argument("--iters",     type=int, default=5000)
    pt.add_argument("--batch",     type=int, default=8)
    pt.add_argument("--lr",        type=float, default=1e-3)
    pt.add_argument("--img_size",  type=int, default=384)
    pt.add_argument("--base",      type=int, default=32)
    pt.add_argument("--num_workers", type=int, default=2)
    pt.add_argument("--val_every", type=int, default=500)
    pt.add_argument("--save_every", type=int, default=500)
    pt.add_argument("--holdout",   type=int, default=20)

    pe = sp.add_parser("eval")
    pe.add_argument("--input_dir", default=fr"{base_dir}\input")
    pe.add_argument("--mask_dir",  default=fr"{base_dir}\stamp_mask")
    pe.add_argument("--out",       default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\seg_eval")
    pe.add_argument("--start",     type=int, default=5980)
    pe.add_argument("--n",         type=int, default=20)
    pe.add_argument("--base",      type=int, default=32)

    pi = sp.add_parser("infer")
    pi.add_argument("path")
    pi.add_argument("--out", default=None)
    pi.add_argument("--base", type=int, default=32)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_holdout(args)
    elif args.cmd == "infer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = UNetSeg(base=args.base).to(device)
        net.load_state_dict(torch.load(str(WEIGHT_PATH), map_location=device))
        net.eval()
        img = cv2.imread(args.path)
        mask, prob = predict_mask_full(net, img, device)
        out = args.out or args.path.replace(".png", "_mask.png").replace(".jpg", "_mask.png")
        cv2.imwrite(out, mask)
        print(f"saved -> {out}")
        print(f"bboxes: {bboxes_from_mask(mask)}")


if __name__ == "__main__":
    main()
