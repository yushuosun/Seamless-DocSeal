"""导出每张图的 raw_dark_mask / 拟合椭圆 / 最终 mask, 用于人眼诊断."""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from color_normalize import build_seal_mask, _fit_stamp_ellipse, _stamp_ellipse_mask
from seal_detector_v2 import detect_seals

INP = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\input"
DEBUG = r"E:\per\LEARNING\AI_ra\stamp\output\week6\mask_debug"
os.makedirs(DEBUG, exist_ok=True)

for f in sorted(os.listdir(INP))[:5]:
    if not f.endswith(".png"):
        continue
    img = cv2.imread(os.path.join(INP, f))
    cands = detect_seals(img, colors=("black",), max_candidates=1)
    if not cands:
        print(f"{f}: NO CANDIDATE"); continue
    c = cands[0]
    crop = img[c.y1:c.y2, c.x1:c.x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    raw = ((hsv[:,:,2] <= 80) & (hsv[:,:,1] <= 60)).astype(np.uint8) * 255
    raw_loose = ((hsv[:,:,2] <= 110) & (hsv[:,:,1] <= 90)).astype(np.uint8) * 255
    ell = _fit_stamp_ellipse(crop)
    final = build_seal_mask(crop, "black")

    H, W = crop.shape[:2]
    raw_vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    raw_loose_vis = cv2.cvtColor(raw_loose, cv2.COLOR_GRAY2BGR)
    ell_vis = crop.copy()
    if ell is not None:
        cv2.ellipse(ell_vis, ell, (0,255,0), 4)
    final_vis = crop.copy()
    final_vis[final > 0] = (0, 0, 255)

    # 添加文字标签
    def label(img, txt):
        cv2.rectangle(img, (0,0), (img.shape[1], 38), (255,255,255), -1)
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        return img

    label(crop, f"{f} crop {W}x{H}")
    label(raw_vis, "raw dark V<=80")
    label(raw_loose_vis, "raw dark V<=110")
    label(ell_vis, "fitted ellipse" if ell is not None else "fit FAILED")
    label(final_vis, "final mask (red)")

    sheet = np.hstack([crop, raw_vis, raw_loose_vis, ell_vis, final_vis])
    cv2.imwrite(os.path.join(DEBUG, f"{os.path.splitext(f)[0]}_mask_debug.jpg"), sheet)

    raw_pct = (raw>0).sum() / raw.size * 100
    final_pct = (final>0).sum() / final.size * 100
    print(f"{f}: crop={W}x{H}  raw_dark={raw_pct:.1f}%  final={final_pct:.1f}%  "
          f"ellipse={'fit' if ell is not None else 'FAIL'}")

print(f"\nsaved to: {DEBUG}")
