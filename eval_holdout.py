"""跑 holdout 20 张 (sample_5980-5999) 全流程, 对比 init 权重 vs EMA 权重 PSNR."""
import argparse, os, sys, csv
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_docdiff_multicolor import DocDiffCropRunner
from seal_detector_v2 import detect_seals
from color_normalize import build_seal_mask


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def run_one(runner, inp_path, gt_path, out_path):
    img = cv2.imread(str(inp_path))
    gt  = cv2.imread(str(gt_path))
    if img is None or gt is None:
        return None
    out_bgr, _ = runner.infer_image(
        img, colors=("black",), detector_margin=32, detector_iou=0.12,
        max_candidates=1, paste_kernel=5, skip_normalize=True,
    )
    cv2.imwrite(str(out_path), out_bgr)
    return psnr(img, gt), psnr(out_bgr, gt)


def main():
    ap = argparse.ArgumentParser()
    base = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    ap.add_argument("--input_dir", default=fr"{base}\input")
    ap.add_argument("--gt_dir",    default=fr"{base}\gt")
    ap.add_argument("--out_dir",   default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\holdout_eval_v2")
    ap.add_argument("--start", type=int, default=5980)
    ap.add_argument("--n",     type=int, default=20)
    ap.add_argument("--variant", choices=["plain", "ema", "both"], default="both")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    files = [f"sample_{i:04d}.png" for i in range(args.start, args.start + args.n)]
    files = [f for f in files if (Path(args.input_dir) / f).exists()]

    variants = []
    if args.variant in ("plain", "both"):
        variants.append(("plain", "seal_init_black_v2.pth", "seal_denoiser_black_v2.pth"))
    if args.variant in ("ema", "both"):
        variants.append(("ema", "seal_init_black_v2_ema.pth", "seal_denoiser_black_v2_ema.pth"))

    all_results = {}
    for tag, w_init, w_den in variants:
        print(f"\n[runner: {tag}]   init={w_init}")
        runner = DocDiffCropRunner(
            device="cuda", sampler="ddim", ddim_steps=100,
            weight_init_name=w_init, weight_denoiser_name=w_den,
        )
        out_sub = Path(args.out_dir) / tag
        out_sub.mkdir(exist_ok=True)
        rows = []
        for f in files:
            r = run_one(runner, Path(args.input_dir) / f, Path(args.gt_dir) / f, out_sub / f)
            if r is None: continue
            in_p, out_p = r
            rows.append((f, in_p, out_p))
            print(f"  {f}: in={in_p:.2f}  out={out_p:.2f}  Δ={out_p-in_p:+.2f}")
        all_results[tag] = rows

    # 汇总
    print("\n" + "=" * 70)
    print(f"{'file':<22}", end="")
    for tag, _, _ in variants: print(f"  in    {tag:<5} Δ", end="")
    print()
    print("-" * 70)
    rep_files = [r[0] for r in all_results[variants[0][0]]]
    for f in rep_files:
        line = f"{f:<22}"
        in_p = next((r[1] for r in all_results[variants[0][0]] if r[0] == f), 0)
        line += f"  {in_p:5.2f}"
        for tag, _, _ in variants:
            out_p = next((r[2] for r in all_results[tag] if r[0] == f), 0)
            line += f"  {out_p:5.2f}  {out_p-in_p:+5.2f}"
        print(line)
    print("-" * 70)
    line = f"{'MEAN':<22}"
    in_avg = np.mean([r[1] for r in all_results[variants[0][0]]])
    line += f"  {in_avg:5.2f}"
    for tag, _, _ in variants:
        out_avg = np.mean([r[2] for r in all_results[tag]])
        line += f"  {out_avg:5.2f}  {out_avg-in_avg:+5.2f}"
    print(line)


if __name__ == "__main__":
    main()
