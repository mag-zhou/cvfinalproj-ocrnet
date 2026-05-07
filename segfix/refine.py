#!/usr/bin/env python3
"""Phase 4: SegFix post-hoc refinement of an existing segmentation model.

Given a (1) segmentation model checkpoint (e.g. baseline OCRNet) and a
(2) trained SegFix offset model checkpoint, run both over the validation
set and produce **refined** prediction PNGs by reading, for each predicted
boundary pixel, the class label at the offset destination.

Usage::

    python segfix/refine.py \
        --seg-config       configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py \
        --seg-checkpoint   work_dirs/ocrnet_r50_ade20k_20pct/iter_40000.pth \
        --offset-config    segfix/configs/segfix_r18_ade20k_20pct.py \
        --offset-checkpoint work_dirs/segfix_r18_ade20k_20pct/iter_20000.pth \
        --output           work_dirs/segfix_refined_baseline/ \
        --boundary-thresh  0.5

Outputs:
    <output>/predictions/<image_id>.png   uint8 PNG, ADE20K convention
                                            (0 == ignore_index, classes 1..150)
    <output>/predictions_baseline/<image_id>.png  un-refined predictions for
                                            apples-to-apples eval

Notes:
    - SegFix is class-agnostic; it cannot fix interior errors. See plan.
    - We refine in the FULL ORIGINAL image resolution. The seg model's
      ``predict()`` already returns ori-shape predictions; the offset model's
      ``predict()`` does the same and scales offset magnitudes appropriately.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmseg.registry import MODELS  # noqa: E402

import segfix  # noqa: E402,F401  -- side-effect registry imports


# --------------------------------------------------------------------- model
def _build_model(config_path: str, checkpoint_path: str,
                 device: torch.device) -> Tuple[torch.nn.Module, Config]:
    cfg = Config.fromfile(config_path)
    cfg.model.setdefault('train_cfg', None)
    cfg.model.setdefault('test_cfg', dict(mode='whole'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device).eval()
    return model, cfg


def _build_test_pipeline(cfg: Config) -> Compose:
    return Compose(cfg.test_pipeline)


def _val_image_ids(cfg: Config) -> list:
    val = cfg.val_dataloader.dataset
    img_dir = Path(val.data_root) / val.data_prefix.img_path
    ids = sorted(p.stem for p in img_dir.glob('*.jpg'))
    return ids


# ----------------------------------------------------------------- inference
@torch.no_grad()
def _seg_predict(model, cfg: Config, img_path: Path,
                 device: torch.device) -> np.ndarray:
    """Run a segmentation model on one image; return HxW int (class index)."""
    pipe = _build_test_pipeline(cfg)
    data = pipe(dict(img_path=str(img_path),
                     seg_map_path=str(img_path),  # unused at test time
                     reduce_zero_label=True,
                     seg_fields=[]))
    inputs = data['inputs'].unsqueeze(0).to(device).float()
    samples = [data['data_samples']]
    # Run the model's data preprocessor explicitly (otherwise the input
    # tensor is in raw uint8-like form).
    batch = {'inputs': inputs, 'data_samples': samples}
    batch = model.data_preprocessor(batch, training=False)
    out = model.forward(batch['inputs'], batch['data_samples'], mode='predict')
    pred = out[0].pred_sem_seg.data
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    return pred.detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def _offset_predict(model, cfg: Config, img_path: Path,
                    device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run the SegFix offset model; return (boundary_prob HxW, offsets HxWx2)."""
    pipe = _build_test_pipeline(cfg)
    data = pipe(dict(img_path=str(img_path),
                     seg_map_path=str(img_path),
                     reduce_zero_label=True,
                     seg_fields=[]))
    inputs = data['inputs'].unsqueeze(0).to(device).float()
    samples = [data['data_samples']]
    batch = {'inputs': inputs, 'data_samples': samples}
    batch = model.data_preprocessor(batch, training=False)
    out = model.forward(batch['inputs'], batch['data_samples'], mode='predict')
    seg_logits = out[0].seg_logits.data
    if seg_logits.dim() == 4:
        seg_logits = seg_logits[0]
    seg_logits = seg_logits.detach().cpu().numpy()
    b_prob = seg_logits[0]
    offset = np.stack([seg_logits[1], seg_logits[2]], axis=-1)
    return b_prob, offset.astype(np.float32)


# ----------------------------------------------------------------- refinement
def _refine(seg_pred: np.ndarray,
            boundary_prob: np.ndarray,
            offset: np.ndarray,
            boundary_threshold: float = 0.5) -> np.ndarray:
    """Apply SegFix-style refinement.

    seg_pred:        HxW int class indices (any int dtype).
    boundary_prob:   HxW float in [0, 1].
    offset:          HxWx2 float (dy, dx) in pixels, full-image resolution.
    """
    h, w = seg_pred.shape
    if boundary_prob.shape != (h, w):
        boundary_prob = cv2.resize(
            boundary_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    if offset.shape[:2] != (h, w):
        offset = cv2.resize(
            offset, (w, h), interpolation=cv2.INTER_LINEAR)
    bmask = boundary_prob > boundary_threshold
    if not bmask.any():
        return seg_pred.copy()

    yy, xx = np.indices((h, w))
    ty = np.clip(yy + np.round(offset[..., 0]).astype(np.int64), 0, h - 1)
    tx = np.clip(xx + np.round(offset[..., 1]).astype(np.int64), 0, w - 1)
    refined = seg_pred.copy()
    refined[bmask] = seg_pred[ty[bmask], tx[bmask]]
    return refined


# ----------------------------------------------------------------------- I/O
def _save_label_png(label: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ADE20K convention: 0 = ignored, classes 1..150. The seg model returns
    # 0..149 (because of reduce_zero_label). Shift by +1 so that the on-disk
    # convention matches the GT labels in annotations/.
    label_disk = (label + 1).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), label_disk)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seg-config', required=True)
    p.add_argument('--seg-checkpoint', required=True)
    p.add_argument('--offset-config', required=True)
    p.add_argument('--offset-checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--boundary-thresh', type=float, default=0.5)
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--save-baseline', action='store_true', default=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f'Loading seg model {args.seg_config}')
    seg_model, seg_cfg = _build_model(
        args.seg_config, args.seg_checkpoint, device)
    print(f'Loading offset model {args.offset_config}')
    off_model, off_cfg = _build_model(
        args.offset_config, args.offset_checkpoint, device)

    val = seg_cfg.val_dataloader.dataset
    img_dir = Path(val.data_root) / val.data_prefix.img_path
    ids = sorted(p.stem for p in img_dir.glob('*.jpg'))
    if args.max_images:
        ids = ids[:args.max_images]

    out_ref = args.output / 'predictions'
    out_base = args.output / 'predictions_baseline'
    out_ref.mkdir(parents=True, exist_ok=True)
    if args.save_baseline:
        out_base.mkdir(parents=True, exist_ok=True)

    print(f'Refining {len(ids)} images. Boundary thresh={args.boundary_thresh}')
    for i, stem in enumerate(ids):
        img_path = img_dir / f'{stem}.jpg'
        seg_pred = _seg_predict(seg_model, seg_cfg, img_path, device)
        b_prob, offset = _offset_predict(
            off_model, off_cfg, img_path, device)
        refined = _refine(
            seg_pred, b_prob, offset,
            boundary_threshold=args.boundary_thresh)
        _save_label_png(refined, out_ref / f'{stem}.png')
        if args.save_baseline:
            _save_label_png(seg_pred, out_base / f'{stem}.png')
        if (i + 1) % 50 == 0 or i + 1 == len(ids):
            print(f'  {i + 1}/{len(ids)}')

    print('Done. Refined predictions in', out_ref)
    if args.save_baseline:
        print('Baseline (un-refined) predictions in', out_base)


if __name__ == '__main__':
    main()
