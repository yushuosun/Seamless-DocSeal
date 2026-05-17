"""
黑章漏检诊断脚本 — 对每张图跑一遍黑章检测流水线, 详细打印每个候选 contour 被
哪条过滤踢掉, 让我们能定向调整 seal_detector_v2.py 的阈值。

用法 (Kaggle / 本机):
    INPUT_DIR = "/kaggle/input/datasets/yushuosun/synth-black-text-overlay-300/synth_black_text_overlay_300/input"
    python diagnose_black_recall.py
或直接复制成 notebook cell, 修改 INPUT_DIR 即可。
"""

import os
import sys
import cv2
import numpy as np
from collections import Counter

# ───── 路径配置 ─────────────────────────────────────────────────
INPUT_DIR = "/kaggle/input/datasets/yushuosun/synth-black-text-overlay-300/synth_black_text_overlay_300/input"
N_SAMPLES = 20           # 只看前 N 张, 设 0 表示全部
SAVE_VIS  = "/kaggle/working/black_diag"  # 保存 ink mask + grouped mask + 标注图
os.makedirs(SAVE_VIS, exist_ok=True)

# ───── 复用 detector 内部函数 ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from seal_detector_v2 import (
    build_ink_mask, _group_mask, _circularity, _dark_ratio,
)

# ───── 黑章过滤逻辑 (复刻 _candidate_from_contour 黑章分支) ─────
FILTERS = [
    ("area_ratio_low",   lambda d: d['area_ratio']   < 0.004),
    ("area_ratio_high",  lambda d: d['area_ratio']   > 0.18),
    ("aspect_ratio",     lambda d: not (0.45 <= d['aspect_ratio'] <= 2.2)),
    ("mask_ratio",       lambda d: d['mask_ratio']   < 0.025),
    ("circularity",      lambda d: d['circularity']  < 0.30),
    ("dark_ratio_high",  lambda d: d['dark_ratio']   > 0.55),
]

def diagnose_one(img_bgr):
    """返回 [{contour 诊断字典}, ...]，按面积降序。"""
    H, W = img_bgr.shape[:2]
    page_area = H * W
    ink = build_ink_mask(img_bgr, "black")
    grp = _group_mask(ink, "black", img_bgr.shape)
    contours, _ = cv2.findContours(grp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        margin = 24
        x1 = max(0, x - margin); y1 = max(0, y - margin)
        x2 = min(W, x + w + margin); y2 = min(H, y + h + margin)
        bw, bh = x2 - x1, y2 - y1
        bbox_area = bw * bh
        crop_mask = ink[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        d = {
            'bbox':         (x1, y1, x2, y2),
            'bw': bw, 'bh': bh,
            'area_ratio':   bbox_area / float(page_area),
            'aspect_ratio': bw / float(bh),
            'mask_ratio':   float((crop_mask > 0).sum()) / float(crop_mask.size),
            'circularity':  _circularity(cnt),
            'dark_ratio':   _dark_ratio(gray_crop),
            'contour_area': cv2.contourArea(cnt),
        }
        # 跑过滤链
        rejected_by = [name for name, fn in FILTERS if fn(d)]
        d['rejected_by'] = rejected_by
        d['passed'] = len(rejected_by) == 0
        results.append(d)

    results.sort(key=lambda r: r['contour_area'], reverse=True)
    return ink, grp, results


def fmt(d):
    """打印一行诊断, 标记哪些过滤拒绝。"""
    flags = {f[0]: ('✗' if f[0] in d['rejected_by'] else '✓') for f in FILTERS}
    return (
        f"  {d['bw']}x{d['bh']:<4} "
        f"area={d['area_ratio']:.4f}{flags['area_ratio_low']}{flags['area_ratio_high']} "
        f"asp={d['aspect_ratio']:.2f}{flags['aspect_ratio']} "
        f"mask={d['mask_ratio']:.3f}{flags['mask_ratio']} "
        f"circ={d['circularity']:.2f}{flags['circularity']} "
        f"dark={d['dark_ratio']:.2f}{flags['dark_ratio_high']} "
        f"→ {'✅PASS' if d['passed'] else 'REJECT: ' + ','.join(d['rejected_by'])}"
    )


# ───── 主流程 ───────────────────────────────────────────────────
files = sorted(f for f in os.listdir(INPUT_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
if N_SAMPLES > 0:
    files = files[:N_SAMPLES]

print(f"诊断 {len(files)} 张图\n" + "="*80)

reject_counter = Counter()
total_passed = 0
total_with_candidate = 0
no_contour_files = []

for fname in files:
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"❌ 读不出: {fname}")
        continue

    ink, grp, results = diagnose_one(img)
    n_pass = sum(1 for r in results if r['passed'])
    if n_pass > 0:
        total_passed += 1

    # 只看前 5 个最大 contour, 否则刷屏
    print(f"\n📷 {fname}  H={img.shape[0]} W={img.shape[1]} "
          f"contours={len(results)} passed={n_pass}")

    if not results:
        no_contour_files.append(fname)
        print("   ⚠️ ink mask 闭合后没有任何 contour")
        continue

    total_with_candidate += 1
    for d in results[:5]:
        print(fmt(d))
        for r in d['rejected_by']:
            reject_counter[r] += 1

    # 保存中间产物供检查
    stem = os.path.splitext(fname)[0]
    cv2.imwrite(f"{SAVE_VIS}/{stem}_ink.png", ink)
    cv2.imwrite(f"{SAVE_VIS}/{stem}_grouped.png", grp)
    vis = img.copy()
    for d in results[:5]:
        x1, y1, x2, y2 = d['bbox']
        col = (0, 255, 0) if d['passed'] else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
    cv2.imwrite(f"{SAVE_VIS}/{stem}_vis.png", vis)


# ───── 汇总 ──────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"📊 总结: {total_passed}/{len(files)} 张图至少有 1 个 contour 通过全部过滤")
print(f"        {len(no_contour_files)} 张图连 contour 都没有 (ink mask 太弱)")
if no_contour_files:
    print(f"        没 contour 的图: {no_contour_files[:10]}")

print("\n📈 拒绝原因频次 (越高 = 越是瓶颈):")
for name, _ in FILTERS:
    print(f"   {name:<20} {reject_counter[name]}")

print(f"\n💾 中间产物保存到: {SAVE_VIS}")
print("   - *_ink.png      原始 ink mask (V<=95 & S<=85)")
print("   - *_grouped.png  morphology 闭运算后")
print("   - *_vis.png      标注图: 红=被拒, 绿=通过")
