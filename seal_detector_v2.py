#!/usr/bin/env python3
"""Robust multi-color seal detector for crop-first DocDiff pipeline.

The detector is intentionally independent from DocDiff. It produces candidate
boxes with diagnostics so detection/cropping can be tuned before restoration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import cv2
import numpy as np


@dataclass
class SealCandidate:
    x1: int
    y1: int
    x2: int
    y2: int
    color: str
    score: float
    mask_ratio: float
    area_ratio: float
    aspect_ratio: float
    circularity: float
    dark_ratio: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def to_row(self, file: str = "") -> dict:
        row = asdict(self)
        row["file"] = file
        row["w"] = self.x2 - self.x1
        row["h"] = self.y2 - self.y1
        return row


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _kernel(img_shape: tuple[int, int], divisor: int, minimum: int, maximum: int) -> np.ndarray:
    h, w = img_shape[:2]
    k = _odd(np.clip(min(h, w) // divisor, minimum, maximum))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim < 1e-3:
        return 0.0
    return float(4.0 * np.pi * area / (perim * perim))


def _dark_ratio(gray_crop: np.ndarray) -> float:
    if gray_crop.size == 0:
        return 1.0
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float((binary > 0).sum()) / float(binary.size)


def build_ink_mask(img_bgr: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if color == "red":
        mask = (((h <= 12) | (h >= 160)) & (s >= 55) & (v >= 45))
    elif color == "blue":
        mask = ((h >= 95) & (h <= 135) & (s >= 65) & (v >= 35))
    elif color == "black":
        mask = ((v <= 95) & (s <= 85))
    else:
        raise ValueError(f"unknown color: {color}")

    return mask.astype(np.uint8) * 255


def _group_mask(mask: np.ndarray, color: str, img_shape: tuple[int, int]) -> np.ndarray:
    if color in {"red", "blue"}:
        close_kernel = _kernel(img_shape, divisor=70, minimum=25, maximum=55)
        dilate_kernel = _kernel(img_shape, divisor=170, minimum=7, maximum=19)
    else:
        close_kernel = _kernel(img_shape, divisor=45, minimum=45, maximum=95)
        dilate_kernel = _kernel(img_shape, divisor=120, minimum=13, maximum=25)

    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    grouped = cv2.dilate(grouped, dilate_kernel, iterations=1)
    return grouped


def _candidate_from_contour(
    img_bgr: np.ndarray,
    ink_mask: np.ndarray,
    contour,
    color: str,
    margin: int,
) -> SealCandidate | None:
    img_h, img_w = img_bgr.shape[:2]
    page_area = float(img_h * img_w)
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_w, x + w + margin)
    y2 = min(img_h, y + h + margin)
    bw, bh = x2 - x1, y2 - y1
    if color == "black" and bw > 0 and bh > 0 and bw / float(bh) > 1.70:
        crop_mask_for_shrink = ink_mask[y1:y2, x1:x2]
        target_w = min(bw, max(300, int(bh * 1.40)))
        if target_w < bw:
            bin_mask = (crop_mask_for_shrink > 0).astype(np.uint8)
            integral = cv2.integral(bin_mask)
            best_x = 0
            best_score = -1.0
            step = max(8, target_w // 24)
            band_x = max(8, target_w // 10)
            band_y = max(8, bh // 8)
            for sx in range(0, bw - target_w + 1, step):
                ex = sx + target_w
                total = int(integral[bh, ex] - integral[0, ex] - integral[bh, sx] + integral[0, sx])
                left = int(integral[bh, sx + band_x] - integral[0, sx + band_x] - integral[bh, sx] + integral[0, sx])
                right = int(integral[bh, ex] - integral[0, ex] - integral[bh, ex - band_x] + integral[0, ex - band_x])
                top = int(integral[band_y, ex] - integral[0, ex] - integral[band_y, sx] + integral[0, sx])
                bottom = int(integral[bh, ex] - integral[bh - band_y, ex] - integral[bh, sx] + integral[bh - band_y, sx])
                edge_balance = min(left, right) * 1.8 + min(top, bottom)
                density_penalty = max(0.0, total / float(target_w * bh) - 0.16) * target_w * bh
                score = edge_balance + 0.18 * total - 0.4 * density_penalty
                if score > best_score:
                    best_score = score
                    best_x = sx
            x1 = x1 + best_x
            x2 = x1 + target_w

    bw, bh = x2 - x1, y2 - y1
    bbox_area = float(bw * bh)
    if bbox_area <= 0:
        return None

    area_ratio = bbox_area / page_area
    aspect_ratio = bw / float(bh)
    crop_mask = ink_mask[y1:y2, x1:x2]
    mask_ratio = float((crop_mask > 0).sum()) / float(crop_mask.size)
    circularity = _circularity(contour)
    gray_crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    dark_ratio = _dark_ratio(gray_crop)

    if color in {"red", "blue"}:
        if area_ratio < 0.0012 or area_ratio > 0.22:
            return None
        if not (0.35 <= aspect_ratio <= 3.2):
            return None
        if mask_ratio < 0.012:
            return None
        if bw < 60 or bh < 60:
            return None
        score = mask_ratio * 4.0 + min(circularity, 1.0) + min(area_ratio * 12.0, 1.5)
    else:
        if area_ratio < 0.010 or area_ratio > 0.20:
            return None
        if not (0.55 <= aspect_ratio <= 1.85):
            return None
        if mask_ratio < 0.035:
            return None
        if circularity < 0.02:
            return None
        if dark_ratio > 0.20:
            return None
        if bw < 240 or bh < 180:
            return None
        score = mask_ratio * 3.0 + min(area_ratio * 8.0, 1.0) - max(0.0, aspect_ratio - 1.6) * 0.15

    return SealCandidate(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        color=color,
        score=float(score),
        mask_ratio=float(mask_ratio),
        area_ratio=float(area_ratio),
        aspect_ratio=float(aspect_ratio),
        circularity=float(circularity),
        dark_ratio=float(dark_ratio),
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _nms(candidates: list[SealCandidate], iou_thresh: float) -> list[SealCandidate]:
    color_priority = {"red": 0.04, "blue": 0.03, "black": 0.0}
    ordered = sorted(candidates, key=lambda c: c.score + color_priority.get(c.color, 0.0), reverse=True)
    kept: list[SealCandidate] = []
    for cand in ordered:
        if any(_iou(cand.bbox, keep.bbox) > iou_thresh for keep in kept):
            continue
        kept.append(cand)
    return kept


def _refine_black_bbox(img_bgr: np.ndarray, cand: SealCandidate, expand: int = 30) -> SealCandidate:
    """在候选 bbox 内重新定位章边缘 -> 收紧 bbox 去掉旁边文字.

    思路: 章是封闭/近闭合椭圆轮廓 + 内文; 周围文字是一行一行的"碎块".
    对候选 crop 做 11x11 闭运算 (足够填章笔画断裂, 不足以桥接到文字), 然后找
    "面积大 + 离 crop 中心近 + 长宽比接近 1" 的连通块, 用它的 bbox 替换原 bbox.

    若找不到合适连通块 (例如章笔画太碎), 返回原 cand 不动.
    """
    H, W = img_bgr.shape[:2]
    cx_full = (cand.x1 + cand.x2) / 2.0
    cy_full = (cand.y1 + cand.y2) / 2.0
    bw, bh = cand.x2 - cand.x1, cand.y2 - cand.y1

    # 在原 bbox 上稍微扩张, 防止边缘被切.
    rx1 = max(0, cand.x1 - expand); ry1 = max(0, cand.y1 - expand)
    rx2 = min(W, cand.x2 + expand); ry2 = min(H, cand.y2 + expand)
    crop = img_bgr[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return cand

    crop_mask = build_ink_mask(crop, "black")
    # 小核闭运算: 填章笔画断裂, 不桥接到 30+ px 远的文字.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, k)
    # 章本身是环, 内部需要 fill 一下才能形成实心连通块用于打分.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cand

    # crop 中心 (refined 后我们希望章中心接近这里)
    ch, cw = closed.shape
    cc_x, cc_y = cw / 2.0, ch / 2.0
    diag = float(np.hypot(cw, ch))

    best = None; best_score = -1e9
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 80 or h < 60:
            continue
        # 太大的连通块 (整个 crop) 跳过 - 这往往是章+文字粘在一起
        if w * h > 0.95 * cw * ch:
            continue
        cx, cy = x + w/2.0, y + h/2.0
        d = float(np.hypot(cx - cc_x, cy - cc_y)) / diag
        # 章应该 1) 大 2) 中心 3) 长宽比接近 1
        ar = w / float(h)
        ar_pen = abs(np.log(ar))  # log(1)=0, log(2)≈0.69
        area_ratio = (w * h) / float(cw * ch)
        score = area_ratio - 1.5 * d - 0.4 * ar_pen
        if score > best_score:
            best_score = score; best = (x, y, w, h)

    if best is None:
        return cand

    bx, by, bw_r, bh_r = best
    # refined bbox 加 4% margin (留点空间给章边缘和盖印不规则)
    pad_x = int(bw_r * 0.04); pad_y = int(bh_r * 0.04)
    nx1 = max(0, rx1 + bx - pad_x)
    ny1 = max(0, ry1 + by - pad_y)
    nx2 = min(W, rx1 + bx + bw_r + pad_x)
    ny2 = min(H, ry1 + by + bh_r + pad_y)

    # 与原 bbox 交集 IoU 太低 (refinement 找错地方) -> 放弃
    inter_iou = _iou((nx1, ny1, nx2, ny2), cand.bbox)
    if inter_iou < 0.25:
        return cand

    return SealCandidate(
        x1=nx1, y1=ny1, x2=nx2, y2=ny2,
        color=cand.color, score=cand.score,
        mask_ratio=cand.mask_ratio, area_ratio=cand.area_ratio,
        aspect_ratio=(nx2-nx1)/float(max(1, ny2-ny1)),
        circularity=cand.circularity, dark_ratio=cand.dark_ratio,
    )


def detect_seals(
    img_bgr: np.ndarray,
    colors: Iterable[str] = ("red", "blue", "black"),
    margin: int = 24,
    iou_thresh: float = 0.12,
    max_candidates: int = 4,
    refine_black: bool = True,
) -> list[SealCandidate]:
    candidates: list[SealCandidate] = []
    for color in colors:
        ink_mask = build_ink_mask(img_bgr, color)
        grouped = _group_mask(ink_mask, color, img_bgr.shape)
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cand = _candidate_from_contour(img_bgr, ink_mask, contour, color, margin)
            if cand is not None:
                if color == "black" and refine_black:
                    cand = _refine_black_bbox(img_bgr, cand)
                candidates.append(cand)

    kept = _nms(candidates, iou_thresh=iou_thresh)
    return kept[:max_candidates]


def draw_candidates(img_bgr: np.ndarray, candidates: list[SealCandidate]) -> np.ndarray:
    vis = img_bgr.copy()
    palette = {
        "red": (0, 0, 255),
        "blue": (255, 80, 0),
        "black": (0, 255, 255),
    }
    for idx, cand in enumerate(candidates):
        color = palette.get(cand.color, (0, 255, 0))
        cv2.rectangle(vis, (cand.x1, cand.y1), (cand.x2, cand.y2), color, 3)
        label = f"{idx}:{cand.color} s={cand.score:.2f} m={cand.mask_ratio:.2f}"
        y = max(20, cand.y1 - 8)
        cv2.putText(vis, label, (cand.x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return vis
