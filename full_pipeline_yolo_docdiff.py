"""完整黑章去除 pipeline (YOLO 版): YOLO 检测 → DocDiff 擦除 → paste 回原图.

替代 seg_unet 版. YOLO 直接学 bbox, 天然过滤文字段 FP, 不需要形状后处理.

每张图输出:
  - <stem>.png         : 最终去章后的图
  - <stem>_compare.jpg : input+bbox | output | diff_heat
  - <stem>_bbox.txt    : x1 y1 x2 y2 conf

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\full_pipeline_yolo_docdiff.py
        [--yolo_weight code\\stamp_final_v1code\\yolo_runs\\black_mask_only_v1\\weights\\best.pt]
        [--conf 0.25]
        [--docdiff_init seal_init_black_mask_only_long_ema.pth]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import cv2
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "DocDiff"))

from kaggle_docdiff_multicolor import DocDiffCropRunner


def expand_bbox(bb, margin_frac, W, H):
    x1, y1, x2, y2 = bb
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_frac); my = int(bh * margin_frac)
    return (max(0, x1 - mx), max(0, y1 - my),
            min(W, x2 + mx), min(H, y2 + my))


def make_paste_mask(crop_in_bgr, crop_out_bgr, diff_thresh=25, dilate=5):
    diff = cv2.absdiff(crop_in_bgr, crop_out_bgr).max(axis=2)
    paste = (diff > diff_thresh).astype(np.uint8) * 255
    if dilate > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        paste = cv2.dilate(paste, k, iterations=1)
    return paste


def yolo_detect(yolo_model, img_bgr, conf=0.25, iou=0.5, imgsz=640):
    """跑 YOLO 推理, 返回 [(x1,y1,x2,y2,conf), ...]"""
    res = yolo_model.predict(source=img_bgr, conf=conf, iou=iou,
                              imgsz=imgsz, verbose=False)
    out = []
    for r in res:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for b, c in zip(boxes, confs):
            out.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(c)))
    return out


def process_image(img_bgr, yolo_model, docdiff_runner, args):
    H, W = img_bgr.shape[:2]
    bbs = yolo_detect(yolo_model, img_bgr, conf=args.conf, iou=args.nms_iou, imgsz=args.yolo_imgsz)
    if not bbs:
        return img_bgr.copy(), []

    out_bgr = img_bgr.copy()
    for x1, y1, x2, y2, _conf in bbs:
        ex1, ey1, ex2, ey2 = expand_bbox((x1, y1, x2, y2), args.margin_frac, W, H)
        crop = img_bgr[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]

        # 缩到 DocDiff 训练分布尺寸
        target_diag = args.target_diag
        cur_diag = float(np.hypot(ch, cw))
        if cur_diag > target_diag * 1.6:
            s = target_diag / cur_diag
            crop_in = cv2.resize(crop, (max(96, int(cw * s)), max(96, int(ch * s))),
                                 interpolation=cv2.INTER_AREA)
            scaled = True
        else:
            crop_in = crop
            scaled = False

        crop_out_small = docdiff_runner.run_crop(crop_in)
        crop_out = cv2.resize(crop_out_small, (cw, ch), interpolation=cv2.INTER_LINEAR) if scaled else crop_out_small

        paste_mask = make_paste_mask(crop, crop_out,
                                     diff_thresh=args.paste_diff_thresh,
                                     dilate=args.paste_dilate)
        region = out_bgr[ey1:ey2, ex1:ex2].copy()
        region[paste_mask > 0] = crop_out[paste_mask > 0]
        out_bgr[ey1:ey2, ex1:ex2] = region

    return out_bgr, bbs


def run_dir(yolo_model, docdiff_runner, in_dir: Path, out_dir: Path, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    print(f"\n[{in_dir.name}] {len(files)} files")
    print(f"  {'file':<44} {'#bbox':>5} {'darkIn':>7} {'darkOut':>8} {'drop':>7}")
    print("  " + "-" * 80)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  {f.name:<44}  cant read"); continue

        out_bgr, bbs = process_image(img, yolo_model, docdiff_runner, args)

        def dark_pct(im):
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            return float(((hsv[:, :, 2] <= 120) & (hsv[:, :, 1] <= 90)).mean())
        d_in, d_out = dark_pct(img), dark_pct(out_bgr)
        drop = 1.0 - d_out / max(1e-6, d_in)
        tag = "✅" if drop > 0.4 else ("➕" if drop > 0.15 else ("⚠️" if drop > 0.05 else "·"))
        print(f"  {f.name:<44} {len(bbs):>5} {d_in:>7.3f} {d_out:>8.3f} {drop:>+7.1%}  {tag}")

        cv2.imwrite(str(out_dir / f.name), out_bgr)

        # 三列对比
        diff = cv2.absdiff(img, out_bgr).max(axis=2)
        diff_heat = cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
        in_vis = img.copy()
        for x1, y1, x2, y2, conf in bbs:
            cv2.rectangle(in_vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(in_vis, f"{conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        sheet = np.hstack([in_vis, out_bgr, diff_heat])
        for i, lab in enumerate(["INPUT+BBOX", "OUTPUT", "|out-in|"]):
            cv2.rectangle(sheet, (i * img.shape[1], 0), ((i+1) * img.shape[1], 40), (255,255,255), -1)
            cv2.putText(sheet, lab, (i * img.shape[1] + 12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        s = 1800.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (int(sheet.shape[1]*s), int(sheet.shape[0]*s)))
        cv2.imwrite(str(out_dir / f"{f.stem}_compare.jpg"), sheet)

        with open(out_dir / f"{f.stem}_bbox.txt", "w", encoding="utf-8") as ft:
            ft.write(f"# x1 y1 x2 y2 conf\n")
            for x1, y1, x2, y2, c in bbs:
                ft.write(f"{x1} {y1} {x2} {y2} {c:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    base = r"E:\per\LEARNING\AI_ra\stamp"
    ap.add_argument("--dir1", default=fr"{base}\data\test\stamppure_synth10_v2\input")
    ap.add_argument("--dir2", default=fr"{base}\data\test\realdoc_test")
    ap.add_argument("--out_root", default=fr"{base}\output\week6\full_pipeline_yolo")

    # YOLO
    ap.add_argument("--yolo_weight",
        default=fr"{base}\code\stamp_final_v1code\yolo_runs\black_mask_only_v1\weights\best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--yolo_imgsz", type=int, default=640)

    # DocDiff
    ap.add_argument("--docdiff_init", default="seal_init_black_mask_only_long_ema.pth")
    ap.add_argument("--docdiff_den",  default="seal_denoiser_black_mask_only_long_ema.pth")
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--target_diag", type=int, default=180)

    # paste
    ap.add_argument("--margin_frac", type=float, default=0.15)
    ap.add_argument("--paste_diff_thresh", type=int, default=25)
    ap.add_argument("--paste_dilate", type=int, default=5)

    args = ap.parse_args()

    # YOLO 可能在多个位置
    candidates = [Path(args.yolo_weight),
                  Path(args.yolo_weight).parent.parent / "weights" / "best.pt",
                  HERE / "yolo_runs" / "black_mask_only_v1" / "weights" / "best.pt"]
    yolo_w = next((p for p in candidates if p.exists()), None)
    if yolo_w is None:
        print("❌ YOLO 权重找不到, 试过:")
        for c in candidates: print(f"   {c}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    print(f"\n[yolo]    {yolo_w}")
    from ultralytics import YOLO
    yolo_model = YOLO(str(yolo_w))

    print(f"\n[docdiff] init={args.docdiff_init}")
    docdiff_runner = DocDiffCropRunner(
        device=str(device), sampler="ddim", ddim_steps=args.ddim_steps,
        weight_init_name=args.docdiff_init,
        weight_denoiser_name=args.docdiff_den,
    )

    out_root = Path(args.out_root)
    for in_dir_str, sub in [(args.dir1, "synth10_v2"), (args.dir2, "realdoc_test")]:
        in_dir = Path(in_dir_str)
        if not in_dir.exists():
            print(f"\n⚠️  skip (not found): {in_dir}")
            continue
        run_dir(yolo_model, docdiff_runner, in_dir, out_root / sub, args)

    print(f"\n[done] {out_root}")


if __name__ == "__main__":
    main()
