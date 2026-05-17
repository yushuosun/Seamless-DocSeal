"""真实章合成 (方向 2): 把 seal_0/black 的真实裁剪章贴到干净文档上.

目的: 让 DocDiff 见过"真实印章纹理 + 真实扫描退化"的样本, 缩小 synth-to-real gap.

与之前 mask_only 数据集的区别:
- 章源: 只用 seal_0/black 真实裁剪章 (3767 张, 有真实墨迹/磨损/扫描伪影)
- 渲染: 不再"渲染 mask 为黑墨"(那是合成纹理), 直接保留章原本的灰度分布
- 退化: 用相对克制的退化 (用户原话"章不要太模糊"); 只加印章实际可能的纹理变化

输出结构 (匹配 mask_only_6000 schema, 训练时可二者混用):
    synth_black_real_v2/
        input/      6000 png
        gt/         6000 png
        stamp_mask/ 6000 png
        meta.csv

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\synth_black_real_v2.py
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
OUT_ROOT  = DATA_ROOT / "high_quality_stamped_corpus" / "synth_black_real_v2"

# 文档页池 (排除 JZW 和 test)
PAGE_DIRS = [
    (DATA_ROOT / "Seal_Dataset" / "wholedoc_stamps" / "fake" / "StainDoc_seal" / "train" / "target", 1.0),
    (DATA_ROOT / "high_quality_stamped_corpus" / "background_pdfs" / "dataset" / "training_data" / "images", 0.4),
    (DATA_ROOT / "high_quality_stamped_corpus" / "pdf_screenshots_180dpi" / "pages", 0.6),
]
# 真实黑章源
STAMP_DIR = DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_0" / "black"

# 排除路径 (绝对不参与训练)
EXCLUDE_DIRS = [
    DATA_ROOT / "JZW",
    DATA_ROOT / "test",
]


# ── 资源采集 ─────────────────────────────────────────────────────
def is_excluded(path: Path) -> bool:
    for ex in EXCLUDE_DIRS:
        try:
            path.relative_to(ex); return True
        except ValueError:
            pass
    return False


def collect_pages():
    pool = []
    for d, w in PAGE_DIRS:
        if not d.exists():
            print(f"  [warn] missing pages dir: {d}"); continue
        for f in d.iterdir():
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
            if is_excluded(f): continue
            pool.append((f, w))
    return pool


def collect_stamps():
    if not STAMP_DIR.exists():
        print(f"  [error] stamp dir missing: {STAMP_DIR}"); return []
    return [f for f in STAMP_DIR.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]


# ── 真实章提取 (保留章本身的灰度而不是二值) ───────────────────────
def extract_stamp_rgba(path: Path, rng: random.Random):
    """seal_0/black 的章是"章 + 白底 + 一点周围文字". 我们要剥离背景, 保留章本身.

    思路:
      1. Otsu 找暗像素 = 章笔画 + 章下方文字
      2. 找最大连通块, 假设是章 (中央大圆形块)
      3. 在章的紧致 bbox 内裁剪
      4. alpha 通道 = 章笔画的浓度 (基于灰度反转)
      5. RGB 通道保留原章像素 (而不是统一改为黑) — 这是关键, 保留真实纹理
    """
    img = cv2.imread(str(path))
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (otsu > 0).sum() < 500:
        return None

    # 闭运算填充章内的细微断裂 (但不要 bridge 到周围文字)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k)

    # 找最大连通块 -> 假设是章 (typically dominates after Otsu)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num < 2: return None
    # 排除占满整图的(背景反转), 取最大的内容块
    areas = stats[1:, cv2.CC_STAT_AREA]
    big_idx = 1 + int(np.argmax(areas))
    if stats[big_idx, cv2.CC_STAT_AREA] < 1000:
        return None
    cluster_mask = (labels == big_idx).astype(np.uint8) * 255

    # bbox + dilate 一点保章环外缘
    ys, xs = np.where(cluster_mask > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pad = 6
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(h, y2 + pad), min(w, x2 + pad)

    # 精细 alpha: 章 mask 内, 暗度高 = alpha 高
    bbox_gray = gray[y1:y2, x1:x2]
    bbox_otsu = otsu[y1:y2, x1:x2]
    bbox_rgb = img[y1:y2, x1:x2]

    # alpha = 反相亮度, 限制在章 mask 区域内 (otsu) 才有 alpha
    alpha = (255 - bbox_gray).astype(np.float32)
    # 如果在 otsu mask 外, alpha 衰减到 0; 内部线性映射保留浓度
    alpha = np.where(bbox_otsu > 0, alpha, 0)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    rgba[..., :3] = bbox_rgb
    rgba[..., 3] = alpha
    return rgba


# ── 章退化 (温和, 模拟扫描印章变化) ───────────────────────────────
def degrade_stamp(rgba: np.ndarray, rng: random.Random):
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    info = {}

    # 1. 整体不透明度 0.75-0.95 (章压力变化)
    op = rng.uniform(0.75, 0.95)
    alpha = alpha * op
    info["op"] = round(op, 2)

    # 2. 不均匀墨迹 (低频乘性噪声)
    if rng.random() < 0.6:
        nh, nw = max(4, h // 24), max(4, w // 24)
        ink = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh, nw).astype(np.float32)
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_LINEAR)
        ink = 0.65 + 0.35 * ink
        alpha = alpha * ink
        info["uneven"] = 1

    # 3. 局部断裂 (高频 mask 衰减)
    if rng.random() < 0.35:
        nh, nw = max(8, h // 14), max(8, w // 14)
        cut = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh, nw).astype(np.float32)
        cut = cv2.resize(cut, (w, h), interpolation=cv2.INTER_LINEAR)
        cut = (cut > 0.85).astype(np.float32)  # 偶发断裂点
        alpha = alpha * (1.0 - 0.5 * cut)
        info["break"] = 1

    # 4. 形态学 (压力轻重)
    morph = rng.choice([None, "erode", "dilate", None, None])
    if morph:
        a8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if morph == "erode":
            a8 = cv2.erode(a8, k)
        elif morph == "dilate":
            a8 = cv2.dilate(a8, k)
        alpha = a8.astype(np.float32) / 255.0
        info["morph"] = morph

    # 5. 极轻微模糊 (扫描边缘软化, 不要太强)
    if rng.random() < 0.4:
        sigma = rng.uniform(0.3, 0.7)
        ks = max(3, int(sigma * 4) | 1)
        a8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        a8 = cv2.GaussianBlur(a8, (ks, ks), sigma)
        alpha = a8.astype(np.float32) / 255.0
        # 同时 RGB 也轻度模糊保持一致
        rgb_blur = cv2.GaussianBlur(rgb.astype(np.uint8), (ks, ks), sigma).astype(np.float32)
        rgb = rgb_blur
        info["blur"] = round(sigma, 2)

    # 6. 通道偏色 (扫描偏黄/偏蓝)
    if rng.random() < 0.25:
        tint = (rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(-15, 15))
        rgb = np.clip(rgb + np.array(tint), 0, 255)
        info["tint"] = tuple(round(t, 1) for t in tint)

    out = np.zeros_like(rgba)
    out[..., :3] = rgb.clip(0, 255).astype(np.uint8)
    out[..., 3] = (alpha * 255).clip(0, 255).astype(np.uint8)
    return out, info


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


# ── 全图扫描退化 ─────────────────────────────────────────────────
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
def make_one(idx: int, seed: int, page_files: list, stamp_files: list, out_dirs: dict):
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

    # 2. 选章 + 提取
    rgba, asset_path = None, None
    for _ in range(8):
        sp = rng.choice(stamp_files)
        rgba = extract_stamp_rgba(sp, rng)
        if rgba is not None and min(rgba.shape[:2]) > 60:
            asset_path = sp
            break
    if rgba is None: return None

    # 3. 缩放 (章对角线 = 页面对角线 18-32%)
    diag = float(np.hypot(H, W))
    target = int(rng.uniform(0.18, 0.32) * diag)
    rgba = scale_rgba(rgba, target)

    # 4. 旋转
    angle = rng.uniform(-30, 30)
    rgba = rotate_rgba(rgba, angle)

    # 5. 退化
    rgba, deg = degrade_stamp(rgba, rng)
    sh, sw = rgba.shape[:2]
    if sh > H - 40 or sw > W - 40:
        s = min((H - 40) / sh, (W - 40) / sw, 1.0)
        rgba = scale_rgba(rgba, int(np.hypot(sh, sw) * s))
        sh, sw = rgba.shape[:2]

    # 6. 选位置 (避免页眉页脚)
    mx = max(20, int(W * 0.05))
    my_top = max(40, int(H * 0.10))
    my_bot = max(40, int(H * 0.05))
    x = rng.randint(mx, max(mx + 1, W - sw - mx))
    y = rng.randint(my_top, max(my_top + 1, H - sh - my_bot))

    # 7. 合成
    composited, stamp_mask = composite(page, rgba, x, y)

    # 8. 全图退化 (input/gt 用同 seed 保证只差章)
    deg_seed = (seed * 1_000_003 + idx + 7777)
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
        "asset_kind": "real_black",
        "asset_path": str(asset_path),
        "source_page": str(page_path).replace(str(DATA_ROOT) + os.sep, ""),
        "seed": seed * 1_000_003 + idx,
        "placement": "text",
        "stamp_idx": 0,
        **{f"deg_{k}": str(v) for k, v in deg.items()},
    }


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview", action="store_true", help="只跑 6 张到 _preview/")
    args = ap.parse_args()

    if args.preview:
        out_root = OUT_ROOT.parent / "synth_black_real_v2_preview"
    else:
        out_root = OUT_ROOT
    out_dirs = {
        "input": out_root / "input",
        "gt": out_root / "gt",
        "stamp_mask": out_root / "stamp_mask",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "meta.csv"
    n = 6 if args.preview else args.n

    print("[loading resources]")
    page_files = collect_pages()
    print(f"  pages: {len(page_files)}")
    stamp_files = collect_stamps()
    print(f"  stamps (seal_0/black): {len(stamp_files)}")
    if not page_files or not stamp_files:
        print("  ❌ resources missing"); return

    print(f"\n[generating] n={n} start={args.start} workers={args.workers}")
    rows = []
    if args.workers <= 1:
        from tqdm import tqdm
        for i in tqdm(range(n)):
            r = make_one(args.start + i, args.seed, page_files, stamp_files, out_dirs)
            if r: rows.append(r)
    else:
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(make_one, args.start + i, args.seed, page_files, stamp_files, out_dirs)
                    for i in range(n)]
            for f in tqdm(as_completed(futs), total=len(futs)):
                r = f.result()
                if r: rows.append(r)
        rows.sort(key=lambda r: r["file"])

    # meta
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

    # preview 拼图
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
