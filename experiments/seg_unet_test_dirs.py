"""用训好的 seg_unet 权重在两个测试目录上跑推理，输出可视化和 bbox.

测试目录:
1. data/test/stamppure_synth10_v2/input  (合成测试)
2. data/test/realdoc_test                 (真实文档测试)

输出每张图:
  - <stem>_panel.jpg : input | prob_heat | mask | overlay (4 列)
  - <stem>_bbox.txt  : bbox 坐标 + 置信度

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\seg_unet_test_dirs.py
        [--weight code\\stamp_final_v1code\\DocDiff\\checksave\\stamp_segmenter.pth]
        [--img_size 512]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import cv2
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from seg_unet import UNetSeg, predict_mask_full, bboxes_from_mask


def overlay(img, mask, color=(0, 0, 255), alpha=0.45):
    out = img.copy()
    m = (mask > 0).astype(np.uint8)
    color_img = np.zeros_like(img); color_img[:] = color
    out = np.where(m[..., None] > 0,
                   (img.astype(np.float32) * (1-alpha) + color_img.astype(np.float32) * alpha).astype(np.uint8),
                   img)
    return out


def run_dir(net, in_dir: Path, out_dir: Path, device, img_size, thresh, min_area):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    print(f"\n[{in_dir.name}] {len(files)} files")
    print(f"  {'file':<40} {'#bbox':>6}  bbox_areas")
    print("  " + "-" * 78)
    summary = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  {f.name:<40}  cant read"); continue
        H, W = img.shape[:2]

        mask, prob = predict_mask_full(net, img, device, infer_size=img_size, thresh=thresh)
        bbs = bboxes_from_mask(mask, min_area=min_area)
        n_bb = len(bbs)
        areas = ", ".join(f"{(b[2]-b[0])}x{(b[3]-b[1])}" for b in bbs[:3]) or "(none)"
        print(f"  {f.name:<40} {n_bb:>6}  {areas}")
        summary.append({"file": f.name, "n_bb": n_bb, "bbs": bbs})

        # ── 可视化拼图 ────────────────────────────────────────
        prob_heat = cv2.applyColorMap(prob, cv2.COLORMAP_HOT)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ov = overlay(img, mask, color=(0, 0, 255), alpha=0.40)
        # 画 bbox
        for x1, y1, x2, y2, _ in bbs:
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 4)
        sheet = np.hstack([img, prob_heat, mask_vis, ov])
        # 标签
        labels = ["INPUT", "PROB_HEAT", "BIN_MASK", "OVERLAY+BBOX"]
        for i, lab in enumerate(labels):
            cv2.rectangle(sheet, (i*W, 0), ((i+1)*W, 40), (255,255,255), -1)
            cv2.putText(sheet, lab, (i*W + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)
        # resize 到合理尺寸
        s = 1800.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (int(sheet.shape[1]*s), int(sheet.shape[0]*s)))
        cv2.imwrite(str(out_dir / f"{f.stem}_panel.jpg"), sheet)

        # bbox txt
        with open(out_dir / f"{f.stem}_bbox.txt", "w", encoding="utf-8") as ftxt:
            ftxt.write(f"# bbox in pixel coords: x1 y1 x2 y2 area\n")
            for b in bbs:
                ftxt.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")

    # 总结
    print("  " + "-" * 78)
    n_with_det = sum(1 for r in summary if r["n_bb"] > 0)
    avg_per_img = sum(r["n_bb"] for r in summary) / max(1, len(summary))
    print(f"  detected stamps in {n_with_det}/{len(summary)} images, avg {avg_per_img:.2f} bbox/img")
    print(f"  panels -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    base = r"E:\per\LEARNING\AI_ra\stamp"
    ap.add_argument("--weight", default=fr"{base}\code\stamp_final_v1code\DocDiff\checksave\stamp_segmenter.pth")
    ap.add_argument("--dir1", default=fr"{base}\data\test\stamppure_synth10_v2\input")
    ap.add_argument("--dir2", default=fr"{base}\data\test\realdoc_test")
    ap.add_argument("--out_root", default=fr"{base}\output\week6\seg_unet_test")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--base_ch", type=int, default=32, help="UNet base 通道数, 默认 32")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--min_area", type=int, default=2000)
    args = ap.parse_args()

    weight_path = Path(args.weight)
    if not weight_path.exists():
        print(f"❌ weight not found: {weight_path}")
        print("   先把 Kaggle 训出的 stamp_segmenter.pth 下载到这个路径")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
    print(f"[weight] {weight_path}")

    net = UNetSeg(base=args.base_ch).to(device)
    net.load_state_dict(torch.load(str(weight_path), map_location=device))
    net.eval()
    print(f"[model] loaded, params={sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    out_root = Path(args.out_root)

    for in_dir_str, sub in [(args.dir1, "synth10_v2"), (args.dir2, "realdoc_test")]:
        in_dir = Path(in_dir_str)
        if not in_dir.exists():
            print(f"\n⚠️  skip (not found): {in_dir}")
            continue
        run_dir(net, in_dir, out_root / sub, device,
                img_size=args.img_size, thresh=args.thresh, min_area=args.min_area)

    print(f"\n[done] all panels under {out_root}")


if __name__ == "__main__":
    main()
