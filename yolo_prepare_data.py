"""把 synth_black_realistic_scan_3000 转成 YOLO 数据集格式.

YOLO 标签格式: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
class_id = 0  (黑章, 单类)
坐标归一化到 [0, 1]

数据集结构:
    yolo_dataset/
        images/train/sample_0000.png ...
        images/val/sample_5980.png ...
        labels/train/sample_0000.txt ...
        labels/val/sample_5980.txt ...
        dataset.yaml
"""
from __future__ import annotations
import argparse, csv, os, shutil
from pathlib import Path
import cv2
import numpy as np


def mask_to_yolo_lines(mask: np.ndarray, min_area_frac: float = 0.001) -> list[str]:
    """从 stamp_mask 提取 bbox, 输出 YOLO 行列表."""
    H, W = mask.shape
    binary = (mask > 30).astype(np.uint8) * 255
    # 闭运算合并断裂
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = int(H * W * min_area_frac)

    lines = []
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        wn = w / W
        hn = h / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    base = r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_realistic_scan_3000"
    ap.add_argument("--input_dir", default=fr"{base}\input")
    ap.add_argument("--mask_dir",  default=fr"{base}\stamp_mask")
    ap.add_argument("--meta",      default=fr"{base}\meta.csv")
    ap.add_argument("--out_dir",   default=fr"{base}\yolo_dataset")
    ap.add_argument("--holdout",   type=int, default=20, help="末尾 N 张作为 val")
    ap.add_argument("--symlink",   action="store_true", help="image 用软链接(快+省盘)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    files = []
    with open(args.meta, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            fn = r["file"]
            if (Path(args.input_dir) / fn).exists() and (Path(args.mask_dir) / fn).exists():
                files.append(fn)
    if not files:
        print("no files found"); return

    train_files = files[:-args.holdout] if args.holdout > 0 else files
    val_files = files[-args.holdout:] if args.holdout > 0 else []
    print(f"[split] train={len(train_files)} val={len(val_files)}")

    def process(file_list, split):
        n_with_box = 0
        for fn in file_list:
            inp_src = Path(args.input_dir) / fn
            inp_dst = out / "images" / split / fn
            mask = cv2.imread(str(Path(args.mask_dir) / fn), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            lines = mask_to_yolo_lines(mask)
            label_dst = out / "labels" / split / (Path(fn).stem + ".txt")
            with open(label_dst, "w") as f:
                f.write("\n".join(lines))
            if lines:
                n_with_box += 1
            # image
            if inp_dst.exists():
                inp_dst.unlink()
            if args.symlink:
                try:
                    inp_dst.symlink_to(inp_src)
                except OSError:
                    shutil.copy(inp_src, inp_dst)
            else:
                shutil.copy(inp_src, inp_dst)
        return n_with_box

    n_t = process(train_files, "train")
    n_v = process(val_files, "val")
    print(f"[labels] train: {n_t}/{len(train_files)} have bbox")
    print(f"[labels] val:   {n_v}/{len(val_files)} have bbox")

    # dataset.yaml
    yaml_path = out / "dataset.yaml"
    yaml_content = f"""path: {out.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['black_stamp']
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n[done] dataset -> {out}")
    print(f"[yaml] {yaml_path}")


if __name__ == "__main__":
    main()
