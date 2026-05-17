"""验证 meta.csv 的 (x,y) 是中心还是左上角: 用 input vs gt 的像素差分作为真实 bbox."""
import csv, os, sys
import cv2
import numpy as np

INPUT = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\input"
GT    = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\gt"
META  = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\meta.csv"

with open(META, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

for r in rows[:5]:
    fname = r["file"]
    inp = cv2.imread(os.path.join(INPUT, fname))
    gt  = cv2.imread(os.path.join(GT, fname))
    if inp is None or gt is None:
        print(f"miss {fname}"); continue
    diff = cv2.absdiff(inp, gt).max(axis=2)
    ys, xs = np.where(diff > 15)
    if len(xs) == 0:
        print(f"{fname} no diff"); continue
    rx1, ry1, rx2, ry2 = xs.min(), ys.min(), xs.max(), ys.max()
    real_cx, real_cy = (rx1+rx2)/2, (ry1+ry2)/2
    real_w, real_h = rx2-rx1, ry2-ry1

    mx, my = int(r["x"]), int(r["y"])
    mw, mh = int(r["stamp_w"]), int(r["stamp_h"])
    print(f"{fname}")
    print(f"  meta:   x={mx} y={my} w={mw} h={mh} angle={r['angle']}")
    print(f"  real:   bbox=({rx1},{ry1})-({rx2},{ry2})  w={real_w} h={real_h}  center=({real_cx:.0f},{real_cy:.0f})")
    print(f"  if (x,y)=topleft → diff from real (rx1,ry1): dx={mx-rx1} dy={my-ry1}")
    print(f"  if (x,y)=center  → diff from real center:    dx={mx-real_cx:.0f} dy={my-real_cy:.0f}")
