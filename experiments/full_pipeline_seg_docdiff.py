"""完整黑章去除 pipeline: seg_unet 检测 → DocDiff 擦除 → paste 回原图.

替代方案: 不再用 seal_detector_v2 (CV) 找章, 而是用 seg_unet 训好的 bbox.

每张图输出:
  - <stem>.png         : 最终去章后的图
  - <stem>_compare.jpg : input | output | diff_heat 三列
  - <stem>_bbox.txt    : 检测到的 bbox

用法:
    .venv-torch\\Scripts\\python.exe code\\stamp_final_v1code\\full_pipeline_seg_docdiff.py
        [--dir1 ...] [--dir2 ...]
        [--seg_weight ...]
        [--docdiff_init seal_init_black_mask_only_long_ema.pth]
        [--docdiff_den  seal_denoiser_black_mask_only_long_ema.pth]
        [--margin_frac 0.15] [--paste_diff_thresh 25]
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

from seg_unet import UNetSeg, predict_mask_full, bboxes_from_mask
from kaggle_docdiff_multicolor import DocDiffCropRunner


def expand_bbox(bb, margin_frac, W, H):
    x1, y1, x2, y2 = bb[:4]
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_frac); my = int(bh * margin_frac)
    return (max(0, x1 - mx), max(0, y1 - my),
            min(W, x2 + mx), min(H, y2 + my))


def _ellipse_metrics(mask_region: np.ndarray):
    """对 mask 拟合椭圆, 返回 (iou, axes_ratio).

    iou: mask 和拟合椭圆的 IoU
    axes_ratio: 长轴/短轴, 真圆 ≈ 1, 长方形文字段 ≈ 2-4

    章 (圆形): iou 高 + axes_ratio 接近 1
    文字段 (长方形闭运算后): iou 也可能高, 但 axes_ratio 大
    """
    contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 99.0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return 0.0, 99.0
    try:
        ell = cv2.fitEllipse(cnt)
    except cv2.error:
        return 0.0, 99.0
    (_, _), (axis1, axis2), _ = ell
    major = max(axis1, axis2); minor = max(1.0, min(axis1, axis2))
    axes_ratio = major / minor
    ell_mask = np.zeros_like(mask_region)
    cv2.ellipse(ell_mask, ell, 255, -1)
    a = (mask_region > 0); b = (ell_mask > 0)
    inter = (a & b).sum()
    union = (a | b).sum()
    iou = float(inter) / max(1, float(union))
    return iou, axes_ratio


def _has_hough_circle(img_bgr_region: np.ndarray, min_r_frac=0.20, max_r_frac=0.55) -> bool:
    """在 bbox 内的灰度图上找圆. 章有圆环边缘, 文字没有."""
    if min(img_bgr_region.shape[:2]) < 40:
        return False
    gray = cv2.cvtColor(img_bgr_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    h, w = gray.shape
    short = min(h, w)
    min_r = max(15, int(short * min_r_frac))
    max_r = max(min_r + 5, int(short * max_r_frac))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=short // 2,
        param1=80, param2=20,
        minRadius=min_r, maxRadius=max_r,
    )
    return circles is not None and len(circles[0]) > 0


def filter_bboxes_by_shape(bbs, mask, img_bgr, img_shape,
                            max_aspect=2.5, min_side=60,
                            max_side_frac=0.40, max_area_frac=0.18,
                            min_ellipse_iou=0.55, max_axes_ratio=1.6,
                            require_circular=True,
                            **kwargs):
    kwargs["max_axes_ratio"] = max_axes_ratio
    """过滤掉显然不是章的 bbox.

    章特征:
      - 长宽比 max(w,h)/min(w,h) <= max_aspect
      - min(w,h) >= min_side
      - max(w,h) <= max_side_frac * min(H,W)
      - bbox 面积 <= max_area_frac * 全图面积
      - 圆形性: 满足以下二者之一即认为是圆/椭圆:
          (a) 灰度图 HoughCircles 在 bbox 找到圆
          (b) mask 拟合椭圆 IoU >= min_ellipse_iou
    """
    H, W = img_shape[:2]
    page_area = H * W
    short_side = min(H, W)
    out = []
    rejects = []
    for b in bbs:
        x1, y1, x2, y2, area = b
        w, h = x2 - x1, y2 - y1
        reason = None
        if min(w, h) < min_side:
            reason = "too_small"
        elif max(w, h) / max(1, min(w, h)) > max_aspect:
            reason = "aspect"
        elif max(w, h) > max_side_frac * short_side:
            reason = "side_frac"
        elif (w * h) > max_area_frac * page_area:
            reason = "area_frac"
        elif require_circular:
            # 三重圆形检查:
            #   1. 拟合椭圆长短轴比 <= max_axes_ratio (真圆/近圆)
            #   2. mask 和拟合椭圆 IoU >= min_ellipse_iou
            #   3. HoughCircles 在原图找到圆 (兜底, 1/2 都不达标时)
            mask_region = mask[y1:y2, x1:x2]
            ell_iou, axes_ratio = _ellipse_metrics(mask_region)
            # 关键: 椭圆轴比是硬约束, 长椭圆=文字, 直接拒
            if axes_ratio > kwargs.get("max_axes_ratio", 1.6):
                # 长椭圆形状, 但允许 Hough 兜底救活那些断裂的真章
                if not _has_hough_circle(img_bgr[y1:y2, x1:x2]):
                    reason = f"long_ellipse(axes={axes_ratio:.2f})"
            elif ell_iou < min_ellipse_iou:
                # 椭圆 IoU 不达标, 且非长椭圆, 让 Hough 兜底
                if not _has_hough_circle(img_bgr[y1:y2, x1:x2]):
                    reason = f"not_round(iou={ell_iou:.2f},axes={axes_ratio:.2f})"

        if reason:
            rejects.append((b, reason))
        else:
            out.append(b)
    return out, rejects


def make_paste_mask(crop_in_bgr, crop_out_bgr, diff_thresh=25, dilate=5):
    """只在 DocDiff 实际改变较大的像素上 paste, 避免动到原图.

    diff = abs(out - in).max(channel), 高于阈值 = 模型改了 = paste 这里.
    再 dilate 一圈让边缘自然.
    """
    diff = cv2.absdiff(crop_in_bgr, crop_out_bgr).max(axis=2)
    paste = (diff > diff_thresh).astype(np.uint8) * 255
    if dilate > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        paste = cv2.dilate(paste, k, iterations=1)
    return paste


def process_image(img_bgr, seg_net, docdiff_runner, device, args):
    H, W = img_bgr.shape[:2]
    # 1. seg_unet 出 bbox
    mask, prob = predict_mask_full(seg_net, img_bgr, device,
                                    infer_size=args.seg_img_size, thresh=args.seg_thresh)
    bbs = bboxes_from_mask(mask, min_area=args.min_area)
    # 形状过滤: 排除文字标题/段落 FP
    rejects = []
    if args.shape_filter:
        bbs, rejects = filter_bboxes_by_shape(
            bbs, mask, img_bgr, img_bgr.shape,
            max_aspect=args.max_aspect,
            min_side=args.min_side,
            max_side_frac=args.max_side_frac,
            max_area_frac=args.max_area_frac,
            min_ellipse_iou=args.min_ellipse_iou,
            max_axes_ratio=args.max_axes_ratio,
            require_circular=args.require_circular,
        )
    if not bbs:
        return img_bgr.copy(), [], mask, rejects

    out_bgr = img_bgr.copy()
    for x1, y1, x2, y2, _ in bbs:
        # 2. 加 margin 扩张 bbox
        ex1, ey1, ex2, ey2 = expand_bbox((x1, y1, x2, y2), args.margin_frac, W, H)
        crop = img_bgr[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]

        # 3. resize 到 DocDiff 训练分布尺寸 (对角线 ~150-200 px)
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

        # 4. 跑 DocDiff (用 mask_only EMA 权重)
        crop_out_small = docdiff_runner.run_crop(crop_in)

        # 5. 缩回原 crop 尺寸
        if scaled:
            crop_out = cv2.resize(crop_out_small, (cw, ch), interpolation=cv2.INTER_LINEAR)
        else:
            crop_out = crop_out_small

        # 6. 计算 paste mask (只贴 DocDiff 实际改了的像素)
        paste_mask = make_paste_mask(crop, crop_out, diff_thresh=args.paste_diff_thresh, dilate=args.paste_dilate)
        region = out_bgr[ey1:ey2, ex1:ex2].copy()
        region[paste_mask > 0] = crop_out[paste_mask > 0]
        out_bgr[ey1:ey2, ex1:ex2] = region

    return out_bgr, bbs, mask, rejects


def run_dir(seg_net, docdiff_runner, in_dir: Path, out_dir: Path, device, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    print(f"\n[{in_dir.name}] {len(files)} files")
    print(f"  {'file':<40} {'#bbox':>6}  {'darkIn':>7} {'darkOut':>8} {'drop':>7}")
    print("  " + "-" * 78)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"  {f.name:<40}  cant read"); continue

        out_bgr, bbs, mask, rejects = process_image(img, seg_net, docdiff_runner, device, args)

        # 启发式: 暗像素占比下降
        def dark_pct(im):
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            return float(((hsv[:, :, 2] <= 120) & (hsv[:, :, 1] <= 90)).mean())
        d_in, d_out = dark_pct(img), dark_pct(out_bgr)
        drop = 1.0 - d_out / max(1e-6, d_in)
        tag = "✅" if drop > 0.4 else ("➕" if drop > 0.15 else ("⚠️" if drop > 0.05 else "·"))
        rej_str = f" rej={len(rejects)}" if rejects else ""
        print(f"  {f.name:<40} {len(bbs):>3}{rej_str:<10} {d_in:>7.3f} {d_out:>8.3f} {drop:>+7.1%}  {tag}")

        # 写最终结果
        cv2.imwrite(str(out_dir / f.name), out_bgr)

        # 三列对比图
        diff = cv2.absdiff(img, out_bgr).max(axis=2)
        diff_heat = cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
        # 在 input 上画 bbox: 绿=通过, 黄=被拒
        in_vis = img.copy()
        for b, reason in rejects:
            x1, y1, x2, y2, _ = b
            cv2.rectangle(in_vis, (x1, y1), (x2, y2), (0, 200, 200), 3)
            cv2.putText(in_vis, reason, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        for x1, y1, x2, y2, _ in bbs:
            cv2.rectangle(in_vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
        sheet = np.hstack([in_vis, out_bgr, diff_heat])
        H = img.shape[0]
        for i, lab in enumerate(["INPUT+BBOX", "OUTPUT", "|out-in|"]):
            cv2.rectangle(sheet, (i * img.shape[1], 0), ((i+1) * img.shape[1], 40), (255,255,255), -1)
            cv2.putText(sheet, lab, (i * img.shape[1] + 12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        s = 1800.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (int(sheet.shape[1]*s), int(sheet.shape[0]*s)))
        cv2.imwrite(str(out_dir / f"{f.stem}_compare.jpg"), sheet)

        # bbox txt
        with open(out_dir / f"{f.stem}_bbox.txt", "w", encoding="utf-8") as ftxt:
            ftxt.write(f"# x1 y1 x2 y2 area\n")
            for b in bbs:
                ftxt.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")


def main():
    ap = argparse.ArgumentParser()
    base = r"E:\per\LEARNING\AI_ra\stamp"
    ap.add_argument("--dir1", default=fr"{base}\data\test\stamppure_synth10_v2\input")
    ap.add_argument("--dir2", default=fr"{base}\data\test\realdoc_test")
    ap.add_argument("--out_root", default=fr"{base}\output\week6\full_pipeline_test")
    ap.add_argument("--seg_weight", default=fr"{base}\code\stamp_final_v1code\DocDiff\checksave\stamp_segmenter.pth")
    ap.add_argument("--docdiff_init", default="seal_init_black_mask_only_long_ema.pth")
    ap.add_argument("--docdiff_den",  default="seal_denoiser_black_mask_only_long_ema.pth")

    # seg_unet 参数
    ap.add_argument("--seg_img_size", type=int, default=512)
    ap.add_argument("--seg_thresh",   type=float, default=0.6)
    ap.add_argument("--min_area",     type=int, default=2000)
    ap.add_argument("--base_ch",      type=int, default=32)

    # DocDiff 参数
    ap.add_argument("--ddim_steps", type=int, default=100)
    ap.add_argument("--target_diag", type=int, default=180,
                    help="crop 缩放到这个对角线再喂 DocDiff (训练分布)")

    # paste 参数
    ap.add_argument("--margin_frac", type=float, default=0.15,
                    help="bbox 扩张比例 (留点上下文给 DocDiff)")
    ap.add_argument("--paste_diff_thresh", type=int, default=25,
                    help="paste mask 用 |out-in| 阈值; 越低越敢贴")
    ap.add_argument("--paste_dilate", type=int, default=5)

    # 形状过滤 (避免文字标题被当章)
    ap.add_argument("--shape_filter", action="store_true", default=True,
                    help="启用 bbox 形状过滤 (默认开)")
    ap.add_argument("--no_shape_filter", dest="shape_filter", action="store_false")
    ap.add_argument("--max_aspect", type=float, default=2.5,
                    help="bbox 长宽比上限 (章近似圆/方形)")
    ap.add_argument("--min_side",   type=int, default=80,
                    help="bbox 短边像素下限 (太小的不是章)")
    ap.add_argument("--max_side_frac", type=float, default=0.40,
                    help="bbox 长边 / 图短边 上限 (章不超过短边的 40%)")
    ap.add_argument("--max_area_frac", type=float, default=0.18,
                    help="bbox 面积 / 全图面积 上限 (章占页面不超过 18%)")
    ap.add_argument("--require_circular", action="store_true", default=True,
                    help="启用圆形/椭圆双判 (默认开)")
    ap.add_argument("--no_circular", dest="require_circular", action="store_false")
    ap.add_argument("--min_ellipse_iou", type=float, default=0.55,
                    help="mask 拟合椭圆 IoU 下限. 不达标且 Hough 也找不到圆 → 过滤")
    ap.add_argument("--max_axes_ratio", type=float, default=1.6,
                    help="拟合椭圆 长轴/短轴 上限. 真圆 ~1.0, 文字段 ~3+. 关键过滤项")

    args = ap.parse_args()

    seg_w = Path(args.seg_weight)
    if not seg_w.exists():
        print(f"❌ seg weight not found: {seg_w}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # 加载 seg_unet
    print(f"\n[seg]    loading {seg_w}")
    seg_net = UNetSeg(base=args.base_ch).to(device)
    seg_net.load_state_dict(torch.load(str(seg_w), map_location=device))
    seg_net.eval()
    print(f"  params={sum(p.numel() for p in seg_net.parameters())/1e6:.2f}M")

    # 加载 DocDiff
    print(f"\n[docdiff] loading...")
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
        run_dir(seg_net, docdiff_runner, in_dir, out_root / sub, device, args)

    print(f"\n[done] {out_root}")


if __name__ == "__main__":
    main()
