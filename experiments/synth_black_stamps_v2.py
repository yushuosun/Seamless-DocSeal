"""黑章合成 v2 - 扩充 3000 张到现有 synth_black_realistic_scan_3000 文件夹.

策略:
- 文档页池: StainDoc target (4502) + background_pdfs (199) + pdf_screenshots (321)
- 章源池: seal_0/black 真实黑章 + seal_1 各色章 (转黑) + masks 模板
- 真实扫描风格退化: 不均匀墨迹 / 局部断裂 / 轻微模糊 / 随机透明度 / 通道偏色 / 噪声
- 输出: 追加 sample_3000.png ~ sample_5999.png 到现有目录
- meta.csv 沿用现有 schema, 追加新行 (不覆盖原 3000 行)

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\synth_black_stamps_v2.py
        [--n 3000] [--start 3000] [--seed 20260428] [--workers 4]
        [--preview]   # 只跑 5 张到 _preview/ 看效果
"""
from __future__ import annotations
import argparse, csv, os, random, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────
DATA_ROOT = Path(r"E:\per\LEARNING\AI_ra\stamp\data")
OUT_ROOT  = DATA_ROOT / "high_quality_stamped_corpus" / "synth_black_realistic_scan_3000"

PAGE_DIRS = [
    (DATA_ROOT / "Seal_Dataset" / "wholedoc_stamps" / "fake" / "StainDoc_seal" / "train" / "target", 1.0),
    (DATA_ROOT / "high_quality_stamped_corpus" / "background_pdfs" / "dataset" / "training_data" / "images", 0.4),
    (DATA_ROOT / "high_quality_stamped_corpus" / "pdf_screenshots_180dpi" / "pages", 0.6),
]

STAMP_DIRS = {
    "real_black": [(DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_0" / "black", 1.0)],
    "real_color": [
        (DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_1" / "0_1", 0.6),
        (DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_1" / "0_2", 0.6),
        (DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_1" / "1_cut", 0.6),
        (DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_1" / "2_cut", 0.6),
        (DATA_ROOT / "Seal_Dataset" / "only_stamps" / "words_under_stamps" / "seal_1" / "seal", 0.8),
    ],
    "mask": [(DATA_ROOT / "Seal_Dataset" / "only_stamps" / "SealData(red,mask,pure stamps)" / "masks", 0.7)],
}

# 排除的目录 (用户说 JZW 留作测试)
JZW_DIR = DATA_ROOT / "JZW"


# ── 资源加载 ─────────────────────────────────────────────────────
def collect(dirs_with_weights, exts=(".png", ".jpg", ".jpeg")):
    pool = []
    for d, w in dirs_with_weights:
        if not d.exists():
            print(f"  [warn] missing: {d}")
            continue
        files = [d / f for f in os.listdir(d) if f.lower().endswith(exts)]
        files = [f for f in files if JZW_DIR not in f.parents]
        pool.extend([(f, w) for f in files])
    return pool


# ── 章提取: 返回 RGBA 章 (alpha = 墨水浓度 0~255) ─────────────────
def extract_real_black(path: Path, rng: random.Random) -> np.ndarray | None:
    """seal_0/black: Otsu 阈值找章笔画."""
    img = cv2.imread(str(path))
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (otsu > 0).sum() < 200:
        return None
    # 紧 crop 到笔画 bbox
    ys, xs = np.where(otsu > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pad = 4
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    crop_alpha = otsu[y1:y2, x1:x2]
    h, w = crop_alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # 墨水浓度: 越暗 alpha 越高 (用 gray 作 soft alpha)
    soft = 255 - gray[y1:y2, x1:x2]
    soft = np.where(crop_alpha > 0, soft, 0)
    rgba[..., 3] = soft
    return rgba


def extract_color_to_black(path: Path, rng: random.Random) -> np.ndarray | None:
    """seal_1 各色章: 检测主色 -> 提取章像素 -> 渲染为黑."""
    img = cv2.imread(str(path))
    if img is None or min(img.shape[:2]) < 50:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    # 多种颜色掩膜, 取最大
    masks = {
        "red":   ((((h_ <= 12) | (h_ >= 160)) & (s_ >= 50) & (v_ >= 40))).astype(np.uint8) * 255,
        "blue":  (((h_ >= 95) & (h_ <= 135) & (s_ >= 50) & (v_ >= 40))).astype(np.uint8) * 255,
        "black": (((v_ <= 100) & (s_ <= 80))).astype(np.uint8) * 255,
        "green": (((h_ >= 35) & (h_ <= 85) & (s_ >= 50) & (v_ >= 40))).astype(np.uint8) * 255,
    }
    best = max(masks.items(), key=lambda kv: kv[1].sum())[1]
    if best.sum() < 500:
        return None
    # 闭运算 + 找最大连通块
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(best, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num < 2: return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    keep = (labels == big).astype(np.uint8) * 255
    ys, xs = np.where(keep > 0)
    if len(xs) < 200: return None
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pad = 4
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    region = best[y1:y2, x1:x2]
    h, w = region.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = region
    return rgba


def extract_mask(path: Path, rng: random.Random) -> np.ndarray | None:
    """mask 模板: 二值 mask 渲染为黑 + 添加墨水纹理."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or min(img.shape) < 50: return None
    # mask 文件白底黑字 OR 黑底白字, 自动识别
    if img.mean() > 127:
        binary = (img < 100).astype(np.uint8) * 255
    else:
        binary = (img > 100).astype(np.uint8) * 255
    if binary.sum() < 500: return None
    ys, xs = np.where(binary > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    pad = 4
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    region = binary[y1:y2, x1:x2]
    h, w = region.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = region
    return rgba


# ── 章退化: 模拟真实扫描 ─────────────────────────────────────────
def degrade_stamp(rgba: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    """对章 alpha 做退化, 返回退化后 RGBA 和 meta dict."""
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    info = {}

    # 1. 局部断裂: 随机噪声乘到 alpha 上
    if rng.random() < 0.5:
        noise_scale = rng.uniform(0.05, 0.2)
        noise_freq = rng.uniform(8, 30)
        nh, nw = max(8, int(h / noise_freq)), max(8, int(w / noise_freq))
        noise = rng.random()  # discard
        noise_lo = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh, nw).astype(np.float32)
        noise_lo = cv2.resize(noise_lo, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = alpha * (1.0 - noise_scale * noise_lo)
        info["break"] = round(noise_scale, 2)

    # 2. 形态学: 腐蚀(墨少)/膨胀(墨厚)
    morph = rng.choice([None, "erode", "dilate", "open"])
    if morph and rng.random() < 0.6:
        ks = rng.choice([3, 3, 5])
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        a8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        if morph == "erode":
            a8 = cv2.erode(a8, k)
        elif morph == "dilate":
            a8 = cv2.dilate(a8, k)
        elif morph == "open":
            a8 = cv2.morphologyEx(a8, cv2.MORPH_OPEN, k)
        alpha = a8.astype(np.float32) / 255.0
        info["morph"] = morph

    # 3. 不均匀墨迹: 多频率柏林状噪声 multiplicative
    if rng.random() < 0.7:
        nh, nw = max(4, h // 30), max(4, w // 30)
        ink = np.random.RandomState(rng.randint(0, 1<<30)).rand(nh, nw).astype(np.float32)
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_LINEAR)
        ink = 0.6 + 0.4 * ink  # 0.6~1.0
        alpha = alpha * ink
        info["uneven"] = 1

    # 4. 模糊: 比较小, "章不要太模糊"
    if rng.random() < 0.4:
        sigma = rng.uniform(0.3, 0.8)
        ks = max(3, int(sigma * 4) | 1)  # 奇数
        a8 = (alpha * 255).clip(0, 255).astype(np.uint8)
        a8 = cv2.GaussianBlur(a8, (ks, ks), sigma)
        alpha = a8.astype(np.float32) / 255.0
        info["blur"] = round(sigma, 2)

    # 5. 整体不透明度
    opacity = rng.uniform(0.65, 0.95)
    alpha = alpha * opacity
    info["opacity"] = round(opacity, 2)

    # 6. 通道偏色 (灰黑低对比 / 偏暗棕)
    rgb_tint = (
        rng.randint(15, 40),
        rng.randint(15, 40),
        rng.randint(15, 40),
    )
    if rng.random() < 0.3:
        # 偏棕
        rgb_tint = (rgb_tint[0], rgb_tint[1] + rng.randint(0, 15), rgb_tint[2] + rng.randint(0, 25))
    info["tint"] = rgb_tint

    out = np.zeros_like(rgba)
    out[..., 0] = rgb_tint[0]
    out[..., 1] = rgb_tint[1]
    out[..., 2] = rgb_tint[2]
    out[..., 3] = (alpha * 255).clip(0, 255).astype(np.uint8)
    return out, info


# ── 旋转 + 缩放 ─────────────────────────────────────────────────
def rotate_rgba(rgba: np.ndarray, angle: float) -> np.ndarray:
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    new_w = int(h*sin + w*cos); new_h = int(h*cos + w*sin)
    M[0,2] += (new_w - w)/2; M[1,2] += (new_h - h)/2
    rot = cv2.warpAffine(rgba, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rot


def scale_to_target(rgba: np.ndarray, target_diag: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    cur = float(np.hypot(h, w))
    s = target_diag / max(1.0, cur)
    new_w = max(40, int(w * s)); new_h = max(40, int(h * s))
    return cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


# ── 合成 ──────────────────────────────────────────────────────────
def composite(page_bgr: np.ndarray, stamp_rgba: np.ndarray, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """把 stamp_rgba 贴到 page (x,y) 左上角. 返回 (composited_bgr, binary_mask)."""
    H, W = page_bgr.shape[:2]
    sh, sw = stamp_rgba.shape[:2]
    x2 = min(W, x + sw); y2 = min(H, y + sh)
    x1 = max(0, x); y1 = max(0, y)
    if x2 <= x1 or y2 <= y1:
        return page_bgr, np.zeros((H, W), dtype=np.uint8)
    sx1 = x1 - x; sy1 = y1 - y
    sx2 = sx1 + (x2 - x1); sy2 = sy1 + (y2 - y1)

    out = page_bgr.copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    stamp = stamp_rgba[sy1:sy2, sx1:sx2]
    a = stamp[..., 3:4].astype(np.float32) / 255.0
    rgb = stamp[..., :3].astype(np.float32)
    blended = a * rgb + (1 - a) * region
    out[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = (a[..., 0] * 255).clip(0, 255).astype(np.uint8)
    return out, mask


# ── 全图扫描风格退化 ─────────────────────────────────────────────
def scan_degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    # 极轻微高斯噪声
    if rng.random() < 0.7:
        sigma = rng.uniform(1.0, 3.0)
        n = np.random.RandomState(rng.randint(0, 1<<30)).randn(*out.shape) * sigma
        out = (out.astype(np.float32) + n).clip(0, 255).astype(np.uint8)
    # 极轻微模糊 (扫描软化)
    if rng.random() < 0.3:
        sigma = rng.uniform(0.3, 0.6)
        ks = max(3, int(sigma * 4) | 1)
        out = cv2.GaussianBlur(out, (ks, ks), sigma)
    # 对比度/亮度抖动
    if rng.random() < 0.4:
        a = rng.uniform(0.92, 1.08)
        b = rng.uniform(-8, 8)
        out = (out.astype(np.float32) * a + b).clip(0, 255).astype(np.uint8)
    # JPEG 压缩
    if rng.random() < 0.5:
        q = rng.randint(72, 92)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


# ── 章源采样 + 提取 ─────────────────────────────────────────────
def sample_stamp(rng: random.Random, stamp_pools: dict) -> tuple[np.ndarray, str, str] | None:
    """加权随机选源, 返回 (rgba, kind, path)."""
    # 三类型权重
    kinds = ["real_black", "real_color", "mask"]
    weights = [0.50, 0.30, 0.20]
    for _ in range(8):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        files = stamp_pools[kind]
        if not files:
            continue
        f, _ = rng.choices(files, weights=[w for _, w in files], k=1)[0]
        if kind == "real_black":
            rgba = extract_real_black(f, rng)
        elif kind == "real_color":
            rgba = extract_color_to_black(f, rng)
        else:
            rgba = extract_mask(f, rng)
        if rgba is not None and rgba.shape[0] > 30 and rgba.shape[1] > 30:
            return rgba, kind, str(f)
    return None


# ── 单样本生成 ──────────────────────────────────────────────────
def make_one(idx: int, seed: int, page_files: list, stamp_pools: dict, out_dirs: dict) -> dict | None:
    rng = random.Random(seed * 1_000_003 + idx)
    np.random.seed((seed * 1_000_003 + idx) & 0xFFFFFFFF)

    # 1. 选页
    for _ in range(8):
        page_path, _ = rng.choices(page_files, weights=[w for _, w in page_files], k=1)[0]
        page = cv2.imread(str(page_path))
        if page is None or min(page.shape[:2]) < 600:
            continue
        break
    else:
        return None
    H, W = page.shape[:2]
    # 太大就缩小到合理尺寸
    if max(H, W) > 2400:
        s = 2400.0 / max(H, W)
        page = cv2.resize(page, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        H, W = page.shape[:2]

    # 2. 选章
    sample = sample_stamp(rng, stamp_pools)
    if sample is None:
        return None
    rgba, kind, asset_path = sample

    # 3. 缩放到合理尺寸 (相对页面)
    diag = float(np.hypot(H, W))
    target_diag = int(rng.uniform(0.18, 0.32) * diag)
    rgba = scale_to_target(rgba, target_diag)

    # 4. 旋转
    angle = rng.uniform(-25, 25)
    rgba = rotate_rgba(rgba, angle)

    # 5. 退化
    rgba, deg = degrade_stamp(rgba, rng)
    sh, sw = rgba.shape[:2]
    if sh > H - 40 or sw > W - 40:
        s = min((H - 40) / sh, (W - 40) / sw, 1.0)
        rgba = scale_to_target(rgba, int(np.hypot(sh, sw) * s))
        sh, sw = rgba.shape[:2]

    # 6. 选位置 (偏向中部, 避免顶部页眉/底部页脚)
    margin_x = max(20, int(W * 0.05))
    margin_y_top = max(40, int(H * 0.10))
    margin_y_bot = max(40, int(H * 0.05))
    x = rng.randint(margin_x, max(margin_x + 1, W - sw - margin_x))
    y = rng.randint(margin_y_top, max(margin_y_top + 1, H - sh - margin_y_bot))

    # 7. 合成
    composited, stamp_mask = composite(page, rgba, x, y)

    # 8. 全图扫描退化 (input 和 gt 都过, 保证只差章)
    rng2 = random.Random(seed * 1_000_003 + idx + 7777)
    deg_seed = rng2.randint(0, 1<<30)
    rng_deg_a = random.Random(deg_seed)
    input_img = scan_degrade(composited, rng_deg_a)
    rng_deg_b = random.Random(deg_seed)
    gt_img    = scan_degrade(page, rng_deg_b)

    # 9. 写盘
    fname = f"sample_{idx:04d}.png"
    cv2.imwrite(str(out_dirs["input"] / fname), input_img)
    cv2.imwrite(str(out_dirs["gt"] / fname), gt_img)
    cv2.imwrite(str(out_dirs["stamp_mask"] / fname), stamp_mask)

    return {
        "angle": round(angle, 3),
        "asset_kind": kind,
        "asset_path": asset_path,
        "file": fname,
        "gt_file": fname,
        "placement": "text",
        "seed": seed * 1_000_003 + idx,
        "source_page": str(page_path).replace(str(DATA_ROOT) + os.sep, ""),
        "stamp_h": sh,
        "stamp_idx": 0,
        "stamp_w": sw,
        "x": x,
        "y": y,
        **{f"deg_{k}": v for k, v in deg.items()},
    }


# ── 主流程 ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--start", type=int, default=3000, help="起始 sample idx (默认 3000 接续现有)")
    ap.add_argument("--seed", type=int, default=20260428)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview", action="store_true", help="只跑 5 张到 _preview/")
    args = ap.parse_args()

    # 输出目录
    if args.preview:
        out_root = OUT_ROOT.parent / "synth_preview_v2"
    else:
        out_root = OUT_ROOT
    out_dirs = {
        "input": out_root / "input",
        "gt": out_root / "gt",
        "stamp_mask": out_root / "stamp_mask",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / ("meta_v2.csv" if args.preview else "meta.csv")
    n = 5 if args.preview else args.n

    # 资源池
    print("[loading resource pools]")
    page_files = collect(PAGE_DIRS)
    print(f"  pages: {len(page_files)}")
    stamp_pools = {k: collect(v) for k, v in STAMP_DIRS.items()}
    for k, v in stamp_pools.items():
        print(f"  stamps[{k}]: {len(v)}")

    # 生成
    print(f"\n[generating] n={n} start_idx={args.start} workers={args.workers}")
    rows = []
    if args.workers <= 1:
        from tqdm import tqdm
        for i in tqdm(range(n)):
            r = make_one(args.start + i, args.seed, page_files, stamp_pools, out_dirs)
            if r: rows.append(r)
    else:
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(make_one, args.start + i, args.seed, page_files, stamp_pools, out_dirs) for i in range(n)]
            for f in tqdm(as_completed(futs), total=len(futs)):
                r = f.result()
                if r: rows.append(r)
        rows.sort(key=lambda r: r["file"])

    # 写 meta (preview 写新 csv, 正式追加)
    if args.preview or not meta_path.exists():
        existing_rows = []
        fieldnames = sorted({k for r in rows for k in r.keys()})
    else:
        with open(meta_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        # 合并字段
        fieldnames = sorted(set(fieldnames) | {k for r in rows for k in r.keys()})

    all_rows = existing_rows + [{k: r.get(k, "") for k in fieldnames} for r in rows]
    with open(meta_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # preview 拼图
    if args.preview:
        sheet_panels = []
        for r in rows[:5]:
            inp = cv2.imread(str(out_dirs["input"] / r["file"]))
            gt = cv2.imread(str(out_dirs["gt"] / r["file"]))
            mask = cv2.imread(str(out_dirs["stamp_mask"] / r["file"]))
            row = np.hstack([cv2.resize(inp, (480, 640)), cv2.resize(gt, (480, 640)), cv2.resize(mask, (480, 640))])
            sheet_panels.append(row)
        if sheet_panels:
            cv2.imwrite(str(out_root / "preview_sheet.jpg"), np.vstack(sheet_panels))
            print(f"\n[preview] {out_root / 'preview_sheet.jpg'}")

    print(f"\n[done] generated {len(rows)} samples -> {out_root}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
