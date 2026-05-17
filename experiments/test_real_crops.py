"""测试 DocDiff (mask_only EMA 权重) 对真实裁剪黑章的擦除效果.

无 GT, 用启发式 + 视觉:
  * 启发式: stamp 区域暗像素占比从 input → output 的下降比例
            (V<=120 & S<=90 的像素数 / 全图像素数)
            理想: 下降 60%+ (大部分章被擦)
            稍好: 下降 30~60%
            差:   下降 <30%
  * 视觉: input | output | diff 三列对比图

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\test_real_crops.py
        [--n 20] [--weight_init seal_init_black_mask_only_long_ema.pth]
"""
from __future__ import annotations
import argparse, os, sys, random
from pathlib import Path
import numpy as np
import cv2
import torch

HERE = Path(__file__).parent
DOCDIFF = HERE / "DocDiff"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOCDIFF))

from kaggle_docdiff_multicolor import DocDiffCropRunner


def dark_pct(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 2] <= 120) & (hsv[:, :, 1] <= 90)).mean())


def run(args):
    stamp_dir = Path(args.stamp_dir)
    files = sorted([f for f in stamp_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    random.seed(args.seed)
    if args.n < len(files):
        files = random.sample(files, args.n)
    print(f"[stamps] picked {len(files)} from {stamp_dir}")
    print(f"[weights] init={args.weight_init}")
    print(f"[weights] den ={args.weight_den}")

    runner = DocDiffCropRunner(
        device="cuda" if torch.cuda.is_available() else "cpu",
        sampler="ddim", ddim_steps=args.ddim_steps,
        weight_init_name=args.weight_init,
        weight_denoiser_name=args.weight_den,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'file':<48} {'darkIn':>7} {'darkOut':>8} {'drop':>7}")
    print("  " + "-" * 74)
    rows = []
    for f in files:
        crop = cv2.imread(str(f))
        if crop is None:
            continue
        h, w = crop.shape[:2]

        # DocDiff 训练分辨率 ~128, 真实 crop 一般 200~600. 先 resize 到合理尺寸再过模型
        # 章对角线缩放到 ~150 px (接近训练分布)
        target_diag = 200
        cur_diag = float(np.hypot(h, w))
        if cur_diag > target_diag * 1.6:
            s = target_diag / cur_diag
            crop_in = cv2.resize(crop, (max(96, int(w * s)), max(96, int(h * s))),
                                 interpolation=cv2.INTER_AREA)
            scaled = True
        else:
            crop_in = crop
            scaled = False

        # 跑 DocDiff
        out_crop_small = runner.run_crop(crop_in)

        # 缩回原尺寸方便对比
        if scaled:
            out_crop = cv2.resize(out_crop_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            out_crop = out_crop_small

        d_in = dark_pct(crop)
        d_out = dark_pct(out_crop)
        drop = 1.0 - d_out / max(1e-6, d_in)

        rows.append({"file": f.name, "d_in": d_in, "d_out": d_out, "drop": drop})

        # 拼图: input | output | diff_heat
        diff = cv2.absdiff(crop, out_crop).max(axis=2)
        diff_heat = cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
        target_h = 320
        s2 = target_h / h
        c1 = cv2.resize(crop, (int(w * s2), target_h))
        c2 = cv2.resize(out_crop, (int(w * s2), target_h))
        c3 = cv2.resize(diff_heat, (int(w * s2), target_h))
        sheet = np.hstack([c1, c2, c3])
        # 标签
        for i, lab in enumerate(["INPUT", "OUTPUT", "|out-in|"]):
            cv2.rectangle(sheet, (i * c1.shape[1], 0), ((i+1) * c1.shape[1], 30), (255, 255, 255), -1)
            cv2.putText(sheet, lab, (i * c1.shape[1] + 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(out_dir / f"{f.stem}_compare.jpg"), sheet)

        tag = "✅" if drop > 0.6 else ("➕" if drop > 0.3 else ("⚠️" if drop > 0.1 else "❌"))
        print(f"  {f.name:<48} {d_in:>7.3f} {d_out:>8.3f} {drop:>+7.1%}  {tag}")

    # 总览
    print("  " + "-" * 74)
    if rows:
        d_in_avg = float(np.mean([r["d_in"] for r in rows]))
        d_out_avg = float(np.mean([r["d_out"] for r in rows]))
        drop_avg = float(np.mean([r["drop"] for r in rows]))
        print(f"  {'MEAN':<48} {d_in_avg:>7.3f} {d_out_avg:>8.3f} {drop_avg:>+7.1%}")
        # 分布
        levels = {"strong (>60%)": 0, "moderate (30-60%)": 0, "weak (10-30%)": 0, "fail (<10%)": 0}
        for r in rows:
            d = r["drop"]
            if d > 0.6: levels["strong (>60%)"] += 1
            elif d > 0.3: levels["moderate (30-60%)"] += 1
            elif d > 0.1: levels["weak (10-30%)"] += 1
            else: levels["fail (<10%)"] += 1
        print("\n  removal strength distribution:")
        for k, v in levels.items():
            print(f"    {k:<22} {v:>3} / {len(rows)}")

    # contact sheet
    panels = []
    for f in files[:10]:
        p = out_dir / f"{f.stem}_compare.jpg"
        if p.exists():
            panels.append(cv2.imread(str(p)))
    if panels:
        max_w = max(p.shape[1] for p in panels)
        padded = [cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for p in panels]
        cv2.imwrite(str(out_dir / "_contact_sheet.jpg"), np.vstack(padded))

    print(f"\n  vis -> {out_dir}")
    print(f"  contact sheet -> {out_dir / '_contact_sheet.jpg'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp_dir",
                    default=r"E:\per\LEARNING\AI_ra\stamp\data\Seal_Dataset\only_stamps\words_under_stamps\seal_0\black")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\real_seal0_black_test")
    ap.add_argument("--weight_init",
                    default="seal_init_black_mask_only_long_ema.pth")
    ap.add_argument("--weight_den",
                    default="seal_denoiser_black_mask_only_long_ema.pth")
    ap.add_argument("--ddim_steps", type=int, default=100)
    args = ap.parse_args()
    run(args)
