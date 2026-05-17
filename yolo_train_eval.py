"""YOLO 黑章 detector 训练 + eval.

依赖: pip install ultralytics

用法:
    # 训练 (从 yolo11n.pt 预训练 fine-tune)
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\yolo_train_eval.py train
        [--model yolo11n.pt] [--epochs 50] [--imgsz 640] [--batch 16]

    # eval holdout
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\yolo_train_eval.py eval
        [--n 20]

    # 单图推理 (debug)
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\yolo_train_eval.py infer <image_path>
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).parent
RUNS_DIR = HERE / "yolo_runs"


def train(args):
    from ultralytics import YOLO
    base = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    yaml_path = Path(args.dataset) if args.dataset else Path(base) / "yolo_dataset" / "dataset.yaml"
    if not yaml_path.exists():
        print(f"[error] dataset.yaml not found: {yaml_path}")
        print("Run yolo_prepare_data.py first.")
        return

    print(f"[model] {args.model}")
    print(f"[data]  {yaml_path}")
    model = YOLO(args.model)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        # 增强: 章是简单形状, 不需要太激进
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=0.5,  # mosaic 有助小目标
        # 优化
        lr0=0.01, lrf=0.01,
        cos_lr=True,
        patience=15,  # early stop
    )
    # 训完保存 best.pt 到固定位置
    best = RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\n[done] best weights: {best}")
    print("Test with: yolo_train_eval.py eval")


def eval_holdout(args):
    from ultralytics import YOLO
    weight = Path(args.weights) if args.weights else (RUNS_DIR / args.name / "weights" / "best.pt")
    if not weight.exists():
        print(f"[error] weights not found: {weight}"); return
    print(f"[weights] {weight}")
    model = YOLO(str(weight))

    base = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    val_img_dir = Path(base) / "yolo_dataset" / "images" / "val"
    val_lbl_dir = Path(base) / "yolo_dataset" / "labels" / "val"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析 val GT bboxes
    def read_yolo_label(p, w, h):
        out = []
        if not p.exists(): return out
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5: continue
            cx, cy, ww, hh = map(float, parts[1:])
            x1 = (cx - ww/2) * w; y1 = (cy - hh/2) * h
            x2 = (cx + ww/2) * w; y2 = (cy + hh/2) * h
            out.append((int(x1), int(y1), int(x2), int(y2)))
        return out

    def iou(a, b):
        iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        if inter == 0: return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua

    files = sorted(val_img_dir.glob("*.png"))[:args.n]
    print(f"\n  {'file':<22} {'pred':>5}  {'best_IoU':>8}  recall@0.5 cov")
    print("  " + "-" * 56)
    bIoUs, recalls = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None: continue
        H, W = img.shape[:2]
        gt = read_yolo_label(val_lbl_dir / (f.stem + ".txt"), W, H)
        results = model.predict(source=str(f), conf=0.25, iou=0.5, verbose=False)
        preds = []
        for r in results:
            for b in r.boxes.xyxy.cpu().numpy():
                preds.append(tuple(int(v) for v in b))

        if not gt:
            print(f"  {f.name:<22} {len(preds):>5}  no GT"); continue

        # 对每个 GT 找最佳匹配 pred
        best_per_gt = []
        for g in gt:
            ious = [iou(p, g) for p in preds]
            best_per_gt.append(max(ious) if ious else 0.0)
        mean_best = float(np.mean(best_per_gt))
        recall = sum(1 for x in best_per_gt if x >= 0.5) / len(best_per_gt)
        bIoUs.append(mean_best); recalls.append(recall)
        print(f"  {f.name:<22} {len(preds):>5}  {mean_best:>8.3f}  {recall:.2f}")

        # 可视化
        vis = img.copy()
        for g in gt:
            cv2.rectangle(vis, g[:2], g[2:], (0, 255, 0), 4)
        for p in preds:
            cv2.rectangle(vis, p[:2], p[2:], (0, 0, 255), 3)
        s = 1200.0 / vis.shape[1]
        vis = cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s)))
        cv2.imwrite(str(out_dir / f.name), vis)

    print("  " + "-" * 56)
    if bIoUs:
        print(f"  {'MEAN':<22}        {np.mean(bIoUs):>8.3f}  {np.mean(recalls):.2f}")
    print(f"\n  vis -> {out_dir}")


def infer(args):
    from ultralytics import YOLO
    weight = Path(args.weights) if args.weights else (RUNS_DIR / args.name / "weights" / "best.pt")
    model = YOLO(str(weight))
    img = cv2.imread(args.path)
    res = model.predict(source=args.path, conf=0.25, verbose=True)
    for r in res:
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            print(f"  bbox={tuple(int(v) for v in b)} conf={c:.3f}")
            x1, y1, x2, y2 = (int(v) for v in b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    out = args.out or args.path.replace(".png", "_yolo.png").replace(".jpg", "_yolo.jpg")
    cv2.imwrite(out, img)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    pt = sp.add_parser("train")
    pt.add_argument("--model",  default="yolo11n.pt")
    pt.add_argument("--dataset", default=None)
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--imgsz",  type=int, default=640)
    pt.add_argument("--batch",  type=int, default=16)
    pt.add_argument("--name",   default="black_stamp_v1")

    pe = sp.add_parser("eval")
    pe.add_argument("--weights", default=None)
    pe.add_argument("--name",    default="black_stamp_v1")
    pe.add_argument("--n", type=int, default=20)
    pe.add_argument("--out", default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\yolo_eval")

    pi = sp.add_parser("infer")
    pi.add_argument("path")
    pi.add_argument("--weights", default=None)
    pi.add_argument("--name", default="black_stamp_v1")
    pi.add_argument("--out", default=None)

    args = ap.parse_args()
    {"train": train, "eval": eval_holdout, "infer": infer}[args.cmd](args)


if __name__ == "__main__":
    main()
