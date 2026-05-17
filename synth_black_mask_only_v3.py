"""增强 mask-only 合成 v3: 用 mask 模板 + 真实印章纹理变化, 无文字污染.

相对 mask_only_6000 的改进:
- 章不再是纯黑死板, 而是空间变化的墨密度 (模拟真实印泥/印油不均)
- 章颜色微变 (黑/深灰/暗棕)
- 边缘羽化 (扫描软化)
- 局部"墨水断裂"小孔
- 整体压力变化 (整张章浓淡)
- 多尺度 + 多角度

关键: 不用 seal_0/black 真实章, 因为那种章带原始文档文字, 提取不干净会污染训练.
mask 模板保证形状干净, 纹理通过算法补足.

输出 (匹配 mask_only_6000 schema):
    synth_black_mask_only_v3/
        input/      6000 png
        gt/         6000 png
        stamp_mask/ 6000 png
        meta.csv

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\synth_black_mask_only_v3.py
        [--n 6000] [--start 0] [--workers 4] [--preview]
"""
from __future__ import annotations
import argparse, csv, os, random, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# ── 路径 ─────────────────────────────────────────────────────────
DATA_ROOT = Path(r"E:\per\LEARNING\AI_ra\stamp\data")
OUT_ROOT  = DATA_ROOT / "high_quality_stamped_corpus" / "synth_black_mask_only_v3"

PAGE_DIRS = [
    (DATA_ROOT / "Seal_Dataset" / "wholedoc_stamps" / "fake" / "StainDoc_seal" / "train" / "target", 1.0),
    (DATA_ROOT / "high_quality_stamped_corpus" / "background_pdfs" / "dataset" / "training_data" / "images", 0.4),
    (DATA_ROOT / "high_quality_stamped_corpus" / "pdf_screenshots_180dpi" / "pages", 0.6),
]
MASK_DIR = DATA_ROOT / "Seal_Dataset" / "only_stamps" / "SealData(red,mask,pure stamps)" / "masks"

EXCLUDE_DIRS = [DATA_ROOT / "JZW", DATA_ROOT / "test"]


def is_excluded(path: Path) -> bool:
    for ex in EXCLUDE_DIRS:
        try: path.relative_to(ex); return True
        except ValueError: pass
    return False


def collect_pages():
    pool = []
    for d, w in PAGE_DIRS:
        if not d.exists(): continue
        for f in d.iterdir():
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
            if is_excluded(f): continue
            pool.append((f, w))
    return pool


def collect_masks():
    if not MASK_DIR.exists(): return []
    return [f for f in MASK_DIR.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]


# ── mask -> 真实纹理黑章 RGBA ────────────────────────────────────
def render_mask_to_stamp(mask_path: Path, rng: random.Random):
    """从 mask 模板渲染出"真实纹理"黑章 RGBA.

    mask 模板是二值 (黑底白字 或 白底黑字), 我们要把它变成:
      - 形状: mask 内部
      - alpha: 空间变化的墨密度 (有的地方深, 有的地方浅, 有局部断裂)
      - RGB:  深灰/黑色为主, 微小色变 (扫描偏色)
    """
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or min(img.shape) < 50: return None
    # 自动判断黑底白字还是白底黑字
    if img.mean() > 127:
        binary = (img < 100).astype(np.uint8)  # 章是黑色
    else:
        binary = (img > 100).astype(np.uint8)  # 章是白色
    if binary.sum() < 500: return None

    # 紧 crop
    ys, xs = np.where(binary > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pad = 4
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    binary = binary[y1:y2, x1:x2]
    h, w = binary.shape

    # ───────────── 关键: 空间变化的墨密度 ─────────────
    # 1. 整体压力变化 (低频, 一边深一边浅)
    nh1, nw1 = max(4, h // 32), max(4, w // 32)
    pressure = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh1, nw1).astype(np.float32)
    pressure = cv2.resize(pressure, (w, h), interpolation=cv2.INTER_LINEAR)
    pressure = 0.55 + 0.45 * pressure  # 0.55~1.0, 不会太淡

    # 2. 中频不均匀 (印泥分布)
    nh2, nw2 = max(8, h // 18), max(8, w // 18)
    ink_uneven = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh2, nw2).astype(np.float32)
    ink_uneven = cv2.resize(ink_uneven, (w, h), interpolation=cv2.INTER_LINEAR)
    ink_uneven = 0.7 + 0.3 * ink_uneven

    # 3. 高频细节 (墨颗粒感)
    nh3, nw3 = max(16, h // 8), max(16, w // 8)
    grain = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh3, nw3).astype(np.float32)
    grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)
    grain = 0.85 + 0.15 * grain  # 微调

    # 综合 alpha 浓度
    alpha = pressure * ink_uneven * grain
    alpha = alpha * binary  # 只在 mask 内有 alpha

    # 4. 局部断裂 (小孔, 偶发)
    if rng.random() < 0.5:
        nh4, nw4 = max(20, h // 6), max(20, w // 6)
        breaks = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh4, nw4).astype(np.float32)
        breaks = cv2.resize(breaks, (w, h), interpolation=cv2.INTER_LINEAR)
        # 偶发墨水稀薄点 (高斯阈值, 只削减少数区域)
        break_mask = (breaks > 0.88).astype(np.float32)
        alpha = alpha * (1.0 - 0.55 * break_mask)

    # 5. 边缘羽化 (扫描软化)
    if rng.random() < 0.6:
        ks = rng.choice([3, 5])
        alpha8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        alpha8 = cv2.GaussianBlur(alpha8, (ks, ks), 0.4 + rng.random() * 0.3)
        alpha = alpha8.astype(np.float32) / 255.0

    # 6. 形态学小变化 (印戳压力轻重)
    if rng.random() < 0.35:
        a8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if rng.random() < 0.5: a8 = cv2.erode(a8, k)
        else: a8 = cv2.dilate(a8, k)
        alpha = a8.astype(np.float32) / 255.0

    # 7. 整体不透明度
    op = rng.uniform(0.78, 0.96)
    alpha = alpha * op

    # ───────────── RGB: 微色变 (扫描的章不会是绝对黑) ─────────────
    base_v = rng.uniform(15, 50)  # 基础暗度
    # 加一些通道偏色 (扫描偏黄/偏蓝/偏红)
    tint_b = rng.uniform(-5, 10)
    tint_g = rng.uniform(-5, 10)
    tint_r = rng.uniform(-5, 10)

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = base_v + tint_b
    rgb[..., 1] = base_v + tint_g
    rgb[..., 2] = base_v + tint_r
    # alpha 越浓的地方颜色更深; 加一点亮度变化
    brightness_var = (1.0 - alpha * 0.3)
    rgb = rgb * brightness_var[..., None]

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.clip(0, 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).clip(0, 255).astype(np.uint8)
    return rgba


# ── 几何 ─────────────────────────────────────────────────────────
def rotate_rgba(rgba: np.ndarray, angle: float):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(h*sin + w*cos), int(h*cos + w*sin)
    M[0,2] += (nw - w)/2; M[1,2] += (nh - h)/2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))


def scale_rgba(rgba: np.ndarray, target_diag: int):
    h, w = rgba.shape[:2]
    cur = float(np.hypot(h, w))
    s = target_diag / max(1.0, cur)
    nw, nh = max(40, int(w*s)), max(40, int(h*s))
    return cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def composite(page_bgr: np.ndarray, stamp_rgba: np.ndarray, x: int, y: int):
    H, W = page_bgr.shape[:2]
    sh, sw = stamp_rgba.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + sw), min(H, y + sh)
    if x2 <= x1 or y2 <= y1:
        return page_bgr, np.zeros((H, W), dtype=np.uint8)
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    out = page_bgr.copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    sub = stamp_rgba[sy1:sy2, sx1:sx2]
    a = sub[..., 3:4].astype(np.float32) / 255.0
    rgb = sub[..., :3].astype(np.float32)
    blended = a * rgb + (1 - a) * region
    out[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = (a[..., 0] * 255).clip(0, 255).astype(np.uint8)
    return out, mask


def scan_degrade(img: np.ndarray, rng: random.Random):
    out = img.copy()
    if rng.random() < 0.6:
        sigma = rng.uniform(0.8, 2.5)
        n = np.random.RandomState(rng.randint(0, 1<<30)).randn(*out.shape) * sigma
        out = (out.astype(np.float32) + n).clip(0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        sigma = rng.uniform(0.3, 0.6)
        ks = max(3, int(sigma * 4) | 1)
        out = cv2.GaussianBlur(out, (ks, ks), sigma)
    if rng.random() < 0.4:
        a = rng.uniform(0.92, 1.08); b = rng.uniform(-8, 8)
        out = (out.astype(np.float32) * a + b).clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        q = rng.randint(72, 92)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok: out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


# ── 单样本 ──────────────────────────────────────────────────────
def make_one(idx: int, seed: int, page_files: list, mask_files: list, out_dirs: dict):
    rng = random.Random(seed * 1_000_003 + idx)
    np.random.seed((seed * 1_000_003 + idx) & 0xFFFFFFFF)

    # 1. 选页
    page = None
    for _ in range(8):
        page_path, _ = rng.choices(page_files, weights=[w for _, w in page_files], k=1)[0]
        page = cv2.imread(str(page_path))
        if page is not None and min(page.shape[:2]) >= 600:
            break
    if page is None: return None
    H, W = page.shape[:2]
    if max(H, W) > 2400:
        s = 2400.0 / max(H, W)
        page = cv2.resize(page, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        H, W = page.shape[:2]

    # 2. 选 mask 模板 + 渲染章
    rgba, mask_path = None, None
    for _ in range(8):
        mp = rng.choice(mask_files)
        rgba = render_mask_to_stamp(mp, rng)
        if rgba is not None and min(rgba.shape[:2]) > 50:
            mask_path = mp
            break
    if rgba is None: return None

    # 3. 缩放
    diag = float(np.hypot(H, W))
    target = int(rng.uniform(0.16, 0.30) * diag)
    rgba = scale_rgba(rgba, target)

    # 4. 旋转
    angle = rng.uniform(-30, 30)
    rgba = rotate_rgba(rgba, angle)

    sh, sw = rgba.shape[:2]
    if sh > H - 40 or sw > W - 40:
        s = min((H - 40) / sh, (W - 40) / sw, 1.0)
        rgba = scale_rgba(rgba, int(np.hypot(sh, sw) * s))
        sh, sw = rgba.shape[:2]

    # 5. 位置
    mx = max(20, int(W * 0.05))
    my_top = max(40, int(H * 0.10))
    my_bot = max(40, int(H * 0.05))
    x = rng.randint(mx, max(mx + 1, W - sw - mx))
    y = rng.randint(my_top, max(my_top + 1, H - sh - my_bot))

    # 6. 合成
    composited, stamp_mask = composite(page, rgba, x, y)

    # 7. 全图退化 (input/gt 同 seed 保证只差章)
    deg_seed = seed * 1_000_003 + idx + 7777
    rng_a = random.Random(deg_seed)
    rng_b = random.Random(deg_seed)
    input_img = scan_degrade(composited, rng_a)
    gt_img    = scan_degrade(page, rng_b)

    fname = f"sample_{idx:04d}.png"
    cv2.imwrite(str(out_dirs["input"] / fname), input_img)
    cv2.imwrite(str(out_dirs["gt"] / fname), gt_img)
    cv2.imwrite(str(out_dirs["stamp_mask"] / fname), stamp_mask)

    return {
        "file": fname, "gt_file": fname,
        "x": int(x), "y": int(y),
        "stamp_w": int(sw), "stamp_h": int(sh),
        "angle": round(angle, 3),
        "asset_kind": "mask_textured",
        "asset_path": str(mask_path),
        "source_page": str(page_path).replace(str(DATA_ROOT) + os.sep, ""),
        "seed": seed * 1_000_003 + idx,
        "placement": "text",
        "stamp_idx": 0,
    }


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    out_root = OUT_ROOT.parent / "synth_black_mask_only_v3_preview" if args.preview else OUT_ROOT
    out_dirs = {
        "input": out_root / "input",
        "gt": out_root / "gt",
        "stamp_mask": out_root / "stamp_mask",
    }
    for d in out_dirs.values(): d.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "meta.csv"
    n = 6 if args.preview else args.n

    print("[loading resources]")
    page_files = collect_pages()
    print(f"  pages: {len(page_files)}")
    mask_files = collect_masks()
    print(f"  masks: {len(mask_files)}")
    if not page_files or not mask_files:
        print("  ❌ resources missing"); return

    print(f"\n[generating] n={n} start={args.start} workers={args.workers}")
    rows = []
    if args.workers <= 1:
        from tqdm import tqdm
        for i in tqdm(range(n)):
            r = make_one(args.start + i, args.seed, page_files, mask_files, out_dirs)
            if r: rows.append(r)
    else:
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(make_one, args.start + i, args.seed, page_files, mask_files, out_dirs)
                    for i in range(n)]
            for f in tqdm(as_completed(futs), total=len(futs)):
                r = f.result()
                if r: rows.append(r)
        rows.sort(key=lambda r: r["file"])

    if meta_path.exists() and not args.preview:
        with open(meta_path, encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
            existing_fields = list(csv.DictReader(open(meta_path, encoding="utf-8-sig")).fieldnames or [])
        all_fields = sorted(set(existing_fields) | {k for r in rows for k in r.keys()})
        all_rows = existing + [{k: r.get(k, "") for k in all_fields} for r in rows]
    else:
        all_fields = sorted({k for r in rows for k in r.keys()})
        all_rows = rows
    with open(meta_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader(); w.writerows(all_rows)

    if args.preview:
        panels = []
        for r in rows[:6]:
            inp = cv2.imread(str(out_dirs["input"] / r["file"]))
            gt  = cv2.imread(str(out_dirs["gt"] / r["file"]))
            mk  = cv2.imread(str(out_dirs["stamp_mask"] / r["file"]))
            row = np.hstack([cv2.resize(inp, (480, 640)),
                              cv2.resize(gt, (480, 640)),
                              cv2.resize(mk, (480, 640))])
            panels.append(row)
        if panels:
            cv2.imwrite(str(out_root / "preview_sheet.jpg"), np.vstack(panels))
            print(f"\n[preview] {out_root / 'preview_sheet.jpg'}")

    print(f"\n[done] {len(rows)} samples -> {out_root}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
