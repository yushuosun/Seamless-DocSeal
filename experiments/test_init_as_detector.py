"""测试用 DocDiff init_predictor 当 detector: |input - init_predict| 高的地方就是章.

在 holdout (sample_5980-5999) 上跑, 对比 init_predict-based mask 与 GT stamp_mask 的 IoU.
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import cv2
import torch

HERE = Path(__file__).parent
DOCDIFF = HERE / "DocDiff"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DOCDIFF))

from DocDiff.model.DocDiff import DocDiff


def load_init_predictor(weight_path, device):
    cfg = type("C", (), {
        "MODEL_CHANNELS": 32, "NUM_RESBLOCKS": 1,
        "CHANNEL_MULT": [1, 2, 3, 4],
    })
    net = DocDiff(input_channels=6, output_channels=3,
                  n_channels=cfg.MODEL_CHANNELS, ch_mults=cfg.CHANNEL_MULT,
                  n_blocks=cfg.NUM_RESBLOCKS).to(device)
    net.init_predictor.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()
    return net


def pad_to_8(img):
    H, W = img.shape[-2], img.shape[-1]
    nh = (H + 7) // 8 * 8
    nw = (W + 7) // 8 * 8
    return torch.nn.functional.pad(img, (0, nw-W, 0, nh-H), mode="constant", value=1.0), (H, W)


@torch.no_grad()
def run_init(net, img_bgr, device, downsample=2):
    """全图过 init_predictor, 返回预测的干净图 (BGR uint8)."""
    H, W = img_bgr.shape[:2]
    if downsample > 1:
        img_s = cv2.resize(img_bgr, (W // downsample, H // downsample), interpolation=cv2.INTER_AREA)
    else:
        img_s = img_bgr
    rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    t, (h, w) = pad_to_8(t)
    pred = net.init_predictor(t, torch.zeros((1,), device=device, dtype=torch.long))
    pred = pred[..., :h, :w].clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    pred_bgr = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if downsample > 1:
        pred_bgr = cv2.resize(pred_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    return pred_bgr


def predict_stamp_mask(input_bgr, pred_bgr, thresh=25, min_area=2000):
    diff = cv2.absdiff(input_bgr, pred_bgr).max(axis=2)
    _, binary = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    # 闭运算合并碎片
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    # 去小连通块
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out, diff


def iou_mask(a, b):
    a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
    inter = (a & b).sum(); union = (a | b).sum()
    return inter / max(1, union)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou_bbox(a, b):
    if a is None or b is None: return 0.0
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter == 0: return 0.0
    ua = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - inter
    return inter / ua


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    weight = str(DOCDIFF / "checksave" / "seal_init_black_v2.pth")
    print(f"[weight] {weight}")
    net = load_init_predictor(weight, device)

    base = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    out_dir = Path(r"E:\per\LEARNING\AI_ra\stamp\output\week6\init_as_detector")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'file':<22} {'mask_IoU':>9} {'bbox_IoU':>9}  diff_max")
    print("  " + "-" * 52)
    mask_ious, bbox_ious = [], []
    for i in range(5980, 6000):
        fname = f"sample_{i:04d}.png"
        inp = cv2.imread(os.path.join(base, "input", fname))
        gt_mask = cv2.imread(os.path.join(base, "stamp_mask", fname), cv2.IMREAD_GRAYSCALE)
        if inp is None or gt_mask is None: continue

        pred = run_init(net, inp, device, downsample=2)
        my_mask, diff = predict_stamp_mask(inp, pred, thresh=25, min_area=2000)

        m_iou = iou_mask(my_mask, gt_mask)
        my_bb = bbox_from_mask(my_mask)
        gt_bb = bbox_from_mask(gt_mask)
        b_iou = iou_bbox(my_bb, gt_bb)
        mask_ious.append(m_iou); bbox_ious.append(b_iou)
        print(f"  {fname:<22} {m_iou:>9.3f} {b_iou:>9.3f}  {diff.max():.0f}")

        # 可视化: input | pred | diff_heat | my_mask | gt_mask
        diff_heat = cv2.applyColorMap(np.clip(diff * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
        my_vis = cv2.cvtColor(my_mask, cv2.COLOR_GRAY2BGR)
        gt_vis = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
        sheet = np.hstack([inp, pred, diff_heat, my_vis, gt_vis])
        s = 1600.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (int(sheet.shape[1]*s), int(sheet.shape[0]*s)))
        cv2.imwrite(str(out_dir / f"{fname.replace('.png','')}_panel.jpg"), sheet)

    print("  " + "-" * 52)
    if mask_ious:
        print(f"  {'MEAN':<22} {np.mean(mask_ious):>9.3f} {np.mean(bbox_ious):>9.3f}")
    print(f"\n  panels saved -> {out_dir}")


if __name__ == "__main__":
    main()
