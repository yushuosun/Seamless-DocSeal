#!/usr/bin/env python3
"""Importable Kaggle helper for DocDiff multi-color crop inference.

Example in Kaggle notebook:

from kaggle_docdiff_multicolor import run_first_n

run_first_n(
    input_dir='/kaggle/input/datasets/yushuosun/synth-blue-stamp-text-overlay-300/synth_blue_text_overlay_300/input',
    output_dir='/kaggle/working/docdiff_blue_test20',
    colors=('blue',),
    n_test=20,
    debug_dir='/kaggle/working/docdiff_blue_test20_debug',
)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
DOC_DIFF_ROOT = THIS_DIR / 'DocDiff'
if str(DOC_DIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(DOC_DIFF_ROOT))

from model.DocDiff import DocDiff
from schedule.schedule import Schedule
from schedule.diffusionSample import GaussianDiffusion
from schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

from color_normalize import build_seal_mask, normalize_to_red
from seal_detector_v2 import detect_seals

VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def dpm_solver(*, betas, denoiser, x_T, steps, cond, pre_ori):
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    def _model(x, t_input, cond):
        if not torch.is_tensor(t_input):
            t_input = torch.tensor(t_input, device=x.device)
        t_long = t_input.reshape(-1).to(device=x.device, dtype=torch.long)
        return denoiser(torch.cat((x, cond), dim=1), t_long)

    model_type = 'x_start' if pre_ori == 'True' else 'noise'
    model_fn = model_wrapper(
        _model,
        noise_schedule,
        model_type=model_type,
        model_kwargs={},
        guidance_type='classifier-free',
        condition=cond,
        unconditional_condition=None,
        guidance_scale=1.0,
    )
    solver = DPM_Solver(
        model_fn,
        noise_schedule,
        algorithm_type='dpmsolver++',
        correcting_x0_fn='dynamic_thresholding',
    )
    return solver.sample(
        x_T,
        steps=steps,
        order=1,
        skip_type='time_uniform',
        method='singlestep',
    )


class DocDiffCropRunner:
    def __init__(self, device: str | None = None, dpm_steps: int = 20,
                 sampler: str = 'ddim', ddim_steps: int = 100,
                 weight_init_name: str = 'seal_init.pth',
                 weight_denoiser_name: str = 'seal_denoiser.pth'):
        self._weight_init_name = weight_init_name
        self._weight_denoiser_name = weight_denoiser_name
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dpm_steps = dpm_steps
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.transform = transforms.ToTensor()
        self.config = type('Config', (), {
            'IMAGE_SIZE': [128, 128],
            'CHANNEL_X': 3,
            'CHANNEL_Y': 3,
            'MODEL_CHANNELS': 32,
            'NUM_RESBLOCKS': 1,
            'CHANNEL_MULT': [1, 2, 3, 4],
            'NUM_HEADS': 1,
            'TIMESTEPS': 100,
            'SCHEDULE': 'linear',
            'PRE_ORI': 'True',
            'BETA_LOSS': 50,
            'HIGH_LOW_FREQ': 'True',
        })()
        self.network = None
        self.schedule = None
        self._load_model()

    def _load_model(self) -> None:
        self.network = DocDiff(
            input_channels=self.config.CHANNEL_X + self.config.CHANNEL_Y,
            output_channels=self.config.CHANNEL_Y,
            n_channels=self.config.MODEL_CHANNELS,
            ch_mults=self.config.CHANNEL_MULT,
            n_blocks=self.config.NUM_RESBLOCKS,
        ).to(self.device)

        weight_init = DOC_DIFF_ROOT / 'checksave' / self._weight_init_name
        weight_denoiser = DOC_DIFF_ROOT / 'checksave' / self._weight_denoiser_name
        print(f'[DocDiff] init weights:     {weight_init}')
        print(f'[DocDiff] denoiser weights: {weight_denoiser}')
        self.network.init_predictor.load_state_dict(torch.load(weight_init, map_location=self.device))
        self.network.denoiser.load_state_dict(torch.load(weight_denoiser, map_location=self.device))
        self.network.eval()
        self.schedule = Schedule(self.config.SCHEDULE, self.config.TIMESTEPS)
        self.diffusion = GaussianDiffusion(self.network.denoiser, self.config.TIMESTEPS, self.schedule).to(self.device)

    @staticmethod
    def pad_to_multiple(img, multiple=8):
        h, w = img.shape[-2], img.shape[-1]
        new_h = (h + multiple - 1) // multiple * multiple
        new_w = (w + multiple - 1) // multiple * multiple
        if new_h == h and new_w == w:
            return img, (0, 0, 0, 0)
        pad_h = new_h - h
        pad_w = new_w - w
        padding = (0, pad_w, 0, pad_h)
        img_padded = F.pad(img, padding, mode='constant', value=1.0)
        return img_padded, (0, pad_w, 0, pad_h)

    @staticmethod
    def unpad(img, padding):
        l, r, t, b = padding
        h, w = img.shape[-2], img.shape[-1]
        return img[..., t:h-b, l:w-r]

    @torch.no_grad()
    def run_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_t = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        img_t, padding = self.pad_to_multiple(img_t, 8)

        init_predict = self.network.init_predictor(img_t, torch.zeros((1,), device=self.device, dtype=torch.long))

        if self.sampler == 'dpm_solver':
            sampled = dpm_solver(
                betas=self.schedule.get_betas(),
                denoiser=self.network.denoiser,
                x_T=torch.randn_like(img_t),
                steps=self.dpm_steps,
                cond=init_predict,
                pre_ori=self.config.PRE_ORI,
            )
        else:
            sampled = self.diffusion(torch.randn_like(img_t), init_predict, self.config.PRE_ORI)

        final_sample = sampled + init_predict
        final_sample = self.unpad(final_sample, padding)
        out_np = final_sample.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        out_np = (np.clip(out_np, 0, 1) * 255.0).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def infer_image(
        self,
        img_bgr: np.ndarray,
        colors: Iterable[str] = ('red', 'blue', 'black'),
        detector_margin: int = 24,
        detector_iou: float = 0.12,
        max_candidates: int = 4,
        paste_kernel: int = 9,
        debug_dir: str | None = None,
        stem: str = 'sample',
        skip_normalize: bool = False,
    ) -> tuple[np.ndarray, list[dict]]:
        out_bgr = img_bgr.copy()
        candidates = detect_seals(
            img_bgr,
            colors=tuple(colors),
            margin=detector_margin,
            iou_thresh=detector_iou,
            max_candidates=max_candidates,
        )

        if debug_dir:
            for sub in ['orig', 'mask', 'pseudo_red', 'docdiff_out', 'docdiff_out_full', 'paste_mask']:
                os.makedirs(os.path.join(debug_dir, sub), exist_ok=True)

        logs = []
        for cand_idx, cand in enumerate(candidates):
            x1, y1, x2, y2 = cand.bbox
            color = cand.color
            crop_bgr = img_bgr[y1:y2, x1:x2]
            seal_mask = build_seal_mask(crop_bgr, color)
            if seal_mask.sum() == 0:
                continue
            crop_norm_bgr = crop_bgr if skip_normalize else normalize_to_red(crop_bgr, color)
            crop_out_full_bgr = self.run_crop(crop_norm_bgr)
            crop_out_bgr = crop_out_full_bgr
            paste_mask = seal_mask.copy()
            if paste_kernel > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (paste_kernel, paste_kernel))
                paste_mask = cv2.dilate(paste_mask, kernel, iterations=1)

            region = out_bgr[y1:y2, x1:x2]
            region[paste_mask > 0] = crop_out_bgr[paste_mask > 0]
            out_bgr[y1:y2, x1:x2] = region

            if debug_dir:
                tag = f'{stem}__{cand_idx}_{color}'
                cv2.imwrite(os.path.join(debug_dir, 'orig', f'{tag}.png'), crop_bgr)
                cv2.imwrite(os.path.join(debug_dir, 'mask', f'{tag}.png'), seal_mask)
                cv2.imwrite(os.path.join(debug_dir, 'pseudo_red', f'{tag}.png'), crop_norm_bgr)
                cv2.imwrite(os.path.join(debug_dir, 'docdiff_out_full', f'{tag}.png'), crop_out_full_bgr)
                cv2.imwrite(os.path.join(debug_dir, 'docdiff_out', f'{tag}.png'), crop_out_bgr)
                cv2.imwrite(os.path.join(debug_dir, 'paste_mask', f'{tag}.png'), paste_mask)

            logs.append({
                'cand_idx': cand_idx,
                'color': color,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'score': cand.score,
            })
        return out_bgr, logs


def run_first_n(
    input_dir: str,
    output_dir: str,
    colors: Iterable[str] = ('blue',),
    n_test: int = 20,
    debug_dir: str | None = None,
    detector_margin: int = 24,
    detector_iou: float = 0.12,
    max_candidates: int = 4,
    paste_kernel: int = 9,
    device: str | None = None,
    dpm_steps: int = 20,
    sampler: str = 'ddim',
    ddim_steps: int = 100,
    weight_init_name: str = 'seal_init.pth',
    weight_denoiser_name: str = 'seal_denoiser.pth',
    skip_normalize: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    runner = DocDiffCropRunner(device=device, dpm_steps=dpm_steps, sampler=sampler, ddim_steps=ddim_steps,
                                weight_init_name=weight_init_name, weight_denoiser_name=weight_denoiser_name)
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTS)])[:n_test]
    rows = []
    for fname in tqdm(files, desc='docdiff first n'):
        img_path = os.path.join(input_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        out_bgr, logs = runner.infer_image(
            img_bgr,
            colors=colors,
            detector_margin=detector_margin,
            detector_iou=detector_iou,
            max_candidates=max_candidates,
            paste_kernel=paste_kernel,
            debug_dir=debug_dir,
            stem=Path(fname).stem,
            skip_normalize=skip_normalize,
        )
        cv2.imwrite(os.path.join(output_dir, fname), out_bgr)
        if not logs:
            rows.append({'file': fname, 'status': 'no_detection'})
        else:
            for row in logs:
                row['file'] = fname
                row['status'] = 'ok'
                rows.append(row)

    csv_path = os.path.join(output_dir, 'run_log.csv')
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ['file', 'status']
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
