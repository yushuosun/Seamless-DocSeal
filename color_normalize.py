"""
color_normalize.py
==================
把蓝色 / 黑色印章裁剪图归一化为"伪红色"印章, 以便复用红章训练好的 DocDiff 权重。

核心思路:
- DocDiff 已学会"红色 chroma 区域 -> 擦除"的映射, 我们只需要把其它颜色的印章像素
  改写成红色像素, 模型就会照样把它们当作红章擦掉。擦除后的输出近似干净背景, 不需要
  再做颜色逆变换。

设计选择 (蓝 -> 红):
- 在 HSV 空间把蓝色色相 (H≈100~135) 平移到红色 (H≈0), 保持 S/V 不变,
  形态/笔画/墨水浓淡完全保留。

设计选择 (黑 -> 红):
- 黑色印章饱和度极低, 单纯改 H 没用。我们把"低 V & 低 S"的像素直接改写为红色,
  红色强度由原像素的暗度决定 (越暗 -> 红得越深), 模拟红章效果。

用法:
    color = detect_seal_color(crop_bgr)
    crop_pseudo_red = normalize_to_red(crop_bgr, color)
"""

import cv2
import numpy as np

BLUE_H_LOW = 95
BLUE_H_HIGH = 135
BLUE_S_MIN = 35
BLACK_V_MAX = 80
BLACK_S_MAX = 60


def detect_seal_color(crop_bgr: np.ndarray) -> str:
    """
    在 crop 内统计像素分布, 判断主色: "red" / "blue" / "black"。

    优先级: blue > black > red (红是默认 fallback, 因 crop 已被红 HSV 检出)。
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    blue_mask = ((h >= BLUE_H_LOW) & (h <= BLUE_H_HIGH) & (s >= BLUE_S_MIN)).astype(np.uint8)
    black_mask = ((v <= BLACK_V_MAX) & (s <= BLACK_S_MAX)).astype(np.uint8)
    red_mask = (((h <= 12) | (h >= 160)) & (s >= 40) & (v >= 40)).astype(np.uint8)

    blue_count = int(blue_mask.sum())
    black_count = int(black_mask.sum())
    red_count = int(red_mask.sum())

    counts = {"red": red_count, "blue": blue_count, "black": black_count}
    dominant = max(counts, key=counts.get)

    total_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]
    if counts[dominant] < total_pixels * 0.005:
        return "red"
    return dominant


def _center_ellipse_mask(shape: tuple[int, int], x_scale: float = 0.94, y_scale: float = 0.90) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, int(w * x_scale / 2)), max(1, int(h * y_scale / 2)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _fit_stamp_ellipse(crop_bgr: np.ndarray, v_max: int = 110, s_max: int = 90):
    """在 crop 内寻找章本体的圆/椭圆轮廓, 返回 cv2.fitEllipse 的椭圆参数 ((cx,cy),(w,h),angle).

    思路: 章是一个近闭合的圆/椭圆轮廓; 对低饱和暗像素做 11x11 闭运算填上笔画断裂,
    然后取"最大 + 最居中 + 不占满整个 bbox"的连通块, 在其轮廓上 fitEllipse.
    若拟合不到合适候选, 返回 None, 调用方用 fallback 中心椭圆.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    raw = ((v <= v_max) & (s <= s_max)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    H, W = raw.shape
    cc_x, cc_y = W / 2.0, H / 2.0
    diag = float(np.hypot(W, H))

    best, best_score = None, -1e18
    for cnt in contours:
        if len(cnt) < 5:
            continue
        x, y, w_, h_ = cv2.boundingRect(cnt)
        if w_ < 0.30 * W or h_ < 0.30 * H:
            continue
        if w_ * h_ > 0.95 * W * H:
            continue
        bcx, bcy = x + w_ / 2.0, y + h_ / 2.0
        d_norm = float(np.hypot(bcx - cc_x, bcy - cc_y)) / diag
        area_norm = (w_ * h_) / float(W * H)
        score = area_norm - 2.0 * d_norm
        if score > best_score:
            best_score, best = score, cnt

    if best is None:
        return None
    try:
        return cv2.fitEllipse(best)
    except cv2.error:
        return None


def _stamp_ellipse_mask(crop_shape: tuple[int, int], crop_bgr: np.ndarray,
                        enlarge: float = 1.04) -> np.ndarray:
    """优先用拟合的章椭圆做 mask; 拟合失败 fallback 到 bbox 中心椭圆."""
    h, w = crop_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    ell = _fit_stamp_ellipse(crop_bgr)
    if ell is not None:
        (cx, cy), (axw, axh), ang = ell
        cv2.ellipse(mask, ((cx, cy), (axw * enlarge, axh * enlarge), ang), 255, -1)
        if mask.sum() > 0:
            return mask
    # fallback
    return _center_ellipse_mask(crop_shape, x_scale=0.85, y_scale=0.82)


def build_seal_mask(crop_bgr: np.ndarray, color: str) -> np.ndarray:
    """根据颜色返回 0/255 的印章像素 mask。"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if color == "blue":
        mask = ((h >= BLUE_H_LOW) & (h <= BLUE_H_HIGH) & (s >= BLUE_S_MIN)).astype(np.uint8) * 255
    elif color == "black":
        base = ((v <= BLACK_V_MAX) & (s <= BLACK_S_MAX)).astype(np.uint8) * 255
        ellipse = _stamp_ellipse_mask(crop_bgr.shape, crop_bgr, enlarge=1.02)
        mask = cv2.bitwise_and(base, ellipse)
    else:  # red
        red1 = (h <= 12) & (s >= 40) & (v >= 40)
        red2 = (h >= 160) & (s >= 40) & (v >= 40)
        mask = (red1 | red2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _blue_to_red(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """蓝 -> 红: 转成更浅、更接近真实公章的红色。"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sel = mask > 0
    if not sel.any():
        return crop_bgr

    hsv[sel, 0] = 0
    hsv[sel, 1] = np.clip(hsv[sel, 1] * 0.72, 30, 165)
    hsv[sel, 2] = np.clip(hsv[sel, 2] * 1.12, 95, 245)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _black_to_red(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    黑 -> 红: mask 区域的像素改写为红色, 保留原暗度作为饱和度信号。
    BGR: (low, low, 220 - V*0.6) — 越暗越红越深。
    """
    out = crop_bgr.copy()
    sel = mask > 0
    if not sel.any():
        return out

    v_vals = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    redness = 220.0 - v_vals * 0.6
    redness = np.clip(redness, 80, 230).astype(np.uint8)

    out[sel, 0] = 30
    out[sel, 1] = 30
    out[sel, 2] = redness[sel]
    return out


def normalize_to_red(crop_bgr: np.ndarray, src_color: str) -> np.ndarray:
    """
    把蓝/黑印章的 crop 转成"伪红"crop, 形状/笔画/上下文(白底+周围文字)完全保持。
    红章 crop 直接原样返回。
    """
    if src_color == "red":
        return crop_bgr

    mask = build_seal_mask(crop_bgr, src_color)
    if mask.sum() == 0:
        return crop_bgr

    if src_color == "blue":
        return _blue_to_red(crop_bgr, mask)
    if src_color == "black":
        return _black_to_red(crop_bgr, mask)
    return crop_bgr


_build_seal_mask = build_seal_mask


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        img = cv2.imread(path)
        c = detect_seal_color(img)
        print(f"[detect_seal_color] {path} -> {c}")
        norm = normalize_to_red(img, c)
        out_path = path.rsplit(".", 1)[0] + f"_pseudo_red.png"
        cv2.imwrite(out_path, norm)
        print(f"[normalize_to_red] saved {out_path}")
