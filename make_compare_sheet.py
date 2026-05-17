"""把 input / output / gt 拼成三列对比图, 方便人眼检查去章效果."""
import argparse, os
import cv2
import numpy as np


def make_sheet(input_dir, output_dir, gt_dir, save_path, n=5):
    files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]
    rows = []
    for fname in files:
        inp = cv2.imread(os.path.join(input_dir, fname))
        out = cv2.imread(os.path.join(output_dir, fname))
        gt  = cv2.imread(os.path.join(gt_dir, fname))
        if inp is None or out is None or gt is None:
            print(f"miss {fname}"); continue

        # diff 高亮
        diff_gt = cv2.absdiff(out, gt).max(axis=2)
        diff_out = cv2.cvtColor(diff_gt, cv2.COLOR_GRAY2BGR)
        diff_out = cv2.applyColorMap(np.clip(diff_gt * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)

        h = inp.shape[0]
        target_h = 800
        scale = target_h / float(h)

        def fit(img):
            new_w = int(img.shape[1] * scale)
            r = cv2.resize(img, (new_w, target_h))
            cv2.putText(r, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            return r

        col_in = fit(inp); col_out = fit(out); col_gt = fit(gt); col_diff = fit(diff_out)

        # 标题
        for img, label in [(col_in, "INPUT"), (col_out, "OUTPUT"), (col_gt, "GT"), (col_diff, "|OUT-GT|")]:
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (255,255,255), -1)
            cv2.putText(img, f"{fname}  {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        row = np.hstack([col_in, col_out, col_gt, col_diff])
        rows.append(row)

    if rows:
        max_w = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for r in rows]
        sheet = np.vstack(rows)
        cv2.imwrite(save_path, sheet)
        print(f"saved {save_path}")

    # 数值: 计算 PSNR 在非章区域 (out vs gt) 衡量是否擦坏正文
    print("\n  file                  PSNR(out,gt)   max_diff")
    print("  " + "-"*48)
    for fname in files:
        out = cv2.imread(os.path.join(output_dir, fname))
        gt  = cv2.imread(os.path.join(gt_dir, fname))
        if out is None or gt is None: continue
        mse = np.mean((out.astype(np.float32) - gt.astype(np.float32))**2)
        psnr = 10*np.log10(255*255/mse) if mse > 0 else 99
        max_d = int(np.abs(out.astype(np.int32)-gt.astype(np.int32)).max())
        print(f"  {fname:<22} {psnr:>10.2f}     {max_d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\input")
    ap.add_argument("--output_dir",default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\local_black_docdiff_5_v2")
    ap.add_argument("--gt_dir",   default=r"E:\per\LEARNING\AI_ra\stamp\data\high_quality_stamped_corpus\synth_black_text_overlay_300\gt")
    ap.add_argument("--save",     default=r"E:\per\LEARNING\AI_ra\stamp\output\week6\local_black_docdiff_5_v2\compare_input_output_gt.jpg")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    make_sheet(args.input_dir, args.output_dir, args.gt_dir, args.save, args.n)
