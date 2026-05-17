"""黑章 detector 评估脚手架.

GT bbox = pixel-level diff(input, gt).max(axis=2) > 15 的紧致包围盒.
这是绝对真实 (像素级), 比 meta.csv 准确.

用法 (默认 5 张):
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\eval_black_detection.py

--n 控制评估张数. --baseline_csv 可选保存 csv 供 grid search 比对.
"""
from __future__ import annotations
import argparse, csv, os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seal_detector_v2 import detect_seals


def real_bbox_from_diff(inp_bgr: np.ndarray, gt_bgr: np.ndarray, thresh: int = 15):
    diff = cv2.absdiff(inp_bgr, gt_bgr).max(axis=2)
    ys, xs = np.where(diff > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def coverage(det, gt) -> float:
    iw = max(0, min(det[2], gt[2]) - max(det[0], gt[0]))
    ih = max(0, min(det[3], gt[3]) - max(det[1], gt[1]))
    ga = (gt[2]-gt[0]) * (gt[3]-gt[1])
    return 0.0 if ga == 0 else (iw*ih) / float(ga)


def overshoot(det, gt) -> float:
    da = (det[2]-det[0]) * (det[3]-det[1])
    ga = (gt[2]-gt[0]) * (gt[3]-gt[1])
    return 0.0 if ga == 0 else da / float(ga)


def evaluate(input_dir, gt_dir, out_dir, n, save_csv=None,
             detector_kwargs=None, quiet=False):
    os.makedirs(out_dir, exist_ok=True)
    detector_kwargs = detector_kwargs or {}
    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".png",".jpg",".jpeg")))[:n]

    rows = []
    panels = []
    for fname in files:
        inp = cv2.imread(os.path.join(input_dir, fname))
        gt_img = cv2.imread(os.path.join(gt_dir, fname))
        if inp is None or gt_img is None:
            continue
        gt_box = real_bbox_from_diff(inp, gt_img)
        if gt_box is None:
            continue

        cands = detect_seals(
            inp,
            colors=("black",),
            margin=detector_kwargs.get("margin", 32),
            iou_thresh=detector_kwargs.get("iou_thresh", 0.12),
            max_candidates=detector_kwargs.get("max_candidates", 4),
        )
        if cands:
            scored = sorted(((iou(c.bbox, gt_box), c) for c in cands), key=lambda t: t[0], reverse=True)
            best_iou, best = scored[0]
        else:
            best_iou, best = 0.0, None

        det_box = best.bbox if best else None
        cov = coverage(det_box, gt_box) if det_box else 0.0
        ovr = overshoot(det_box, gt_box) if det_box else 0.0

        # FP 数 = IoU 跟 GT < 0.1 的额外候选
        fp = sum(1 for c in cands if iou(c.bbox, gt_box) < 0.1)

        rows.append({
            "file": fname, "n_cands": len(cands), "fp": fp,
            "iou": best_iou, "coverage": cov, "overshoot": ovr,
            "gt": gt_box, "det": det_box,
        })

        # 可视化
        vis = inp.copy()
        for c in cands:
            col = (0, 0, 255) if (best is not None and c.bbox == best.bbox) else (0, 165, 255)
            cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), col, 4)
        cv2.rectangle(vis, gt_box[:2], gt_box[2:], (0, 255, 0), 4)
        label = f"{fname}  IoU={best_iou:.2f}  cov={cov:.2f}  over={ovr:.2f}  FP={fp}"
        cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6)
        cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.imwrite(os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_eval.jpg"), vis)
        s = 480.0 / max(vis.shape[:2])
        panels.append(cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s))))

    if panels:
        max_h = max(p.shape[0] for p in panels)
        padded = [cv2.copyMakeBorder(p, 0, max_h-p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255)) for p in panels]
        cv2.imwrite(os.path.join(out_dir, "contact_sheet.jpg"), np.hstack(padded))

    if not quiet:
        print("\n" + "="*78)
        print(f"{'file':<22}{'n':>3}{'fp':>4}{'IoU':>8}{'cov':>8}{'over':>8}   gt(wxh)        det(wxh)")
        for r in rows:
            gw, gh = r["gt"][2]-r["gt"][0], r["gt"][3]-r["gt"][1]
            ds = f"{r['det'][2]-r['det'][0]}x{r['det'][3]-r['det'][1]}" if r["det"] else "MISS"
            print(f"{r['file']:<22}{r['n_cands']:>3}{r['fp']:>4}{r['iou']:>8.3f}{r['coverage']:>8.3f}{r['overshoot']:>8.2f}   {gw}x{gh:<10}  {ds}")
        if rows:
            print("="*78)
            print(f"mean IoU      = {np.mean([r['iou'] for r in rows]):.3f}")
            print(f"mean coverage = {np.mean([r['coverage'] for r in rows]):.3f}")
            print(f"mean overshoot= {np.mean([r['overshoot'] for r in rows]):.2f}")
            print(f"recall@0.5    = {sum(1 for r in rows if r['iou']>=0.5)}/{len(rows)}")
            print(f"recall@0.7    = {sum(1 for r in rows if r['iou']>=0.7)}/{len(rows)}")
            print(f"total FP      = {sum(r['fp'] for r in rows)}")
        print(f"\n输出: {out_dir}\\contact_sheet.jpg")

    if save_csv:
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file","n_cands","fp","iou","coverage","overshoot","gt","det"])
            for r in rows:
                w.writerow([r["file"], r["n_cands"], r["fp"], r["iou"], r["coverage"], r["overshoot"], r["gt"], r["det"]])

    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\input")
    ap.add_argument("--gt_dir",    default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\gt")
    ap.add_argument("--out",       default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\black_detect_eval")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--save_csv", default=None)
    args = ap.parse_args()
    evaluate(args.input_dir, args.gt_dir, args.out, args.n, save_csv=args.save_csv)
