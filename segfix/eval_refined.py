#!/usr/bin/env python3
"""Phase 5: standalone mIoU + boundary F-score over a directory of
predictions.

Bypasses mmseg's eval loop entirely so the SegFix-vs-baseline comparison
uses identical eval code on both sets of predictions.

Reuses the boundary helpers from
``mmseg/evaluation/metrics/boundary_metric.py`` so the BF1 numbers are
directly comparable to the ones logged in the existing experiments.

Usage::

    # Evaluate refined predictions:
    python segfix/eval_refined.py \
        --pred-dir work_dirs/segfix_refined_baseline/predictions/ \
        --gt-dir   data/ade/ADEChallengeData2016/annotations/validation/ \
        --num-classes 150

    # Evaluate the same baseline without SegFix (for the comparison):
    python segfix/eval_refined.py \
        --pred-dir work_dirs/segfix_refined_baseline/predictions_baseline/ \
        --gt-dir   data/ade/ADEChallengeData2016/annotations/validation/ \
        --num-classes 150

GT format (ADE20K):  uint8 PNG, 0 = ignore, 1..150 = classes.
Pred format:         uint8 PNG, same convention (refine.py writes that way).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmseg.evaluation.metrics.boundary_metric import (  # noqa: E402
    _boundary_f1_at_tol,
    _sem_seg_to_boundary,
)


def _load_label(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.int64)


def _intersect_and_union(pred: np.ndarray, gt: np.ndarray, num_classes: int,
                         ignore_index: int) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """Per-class intersect / union counters in the same convention as
    mmseg's IoUMetric. ``pred`` and ``gt`` are 0-indexed (ignore = ignore_index).
    """
    valid = gt != ignore_index
    pred = pred[valid]
    gt = gt[valid]
    inter_mask = pred[pred == gt]
    inter_hist = np.bincount(inter_mask, minlength=num_classes)[:num_classes]
    pred_hist = np.bincount(pred, minlength=num_classes)[:num_classes]
    gt_hist = np.bincount(gt, minlength=num_classes)[:num_classes]
    union_hist = pred_hist + gt_hist - inter_hist
    return inter_hist.astype(np.int64), union_hist.astype(np.int64), \
        pred_hist.astype(np.int64), gt_hist.astype(np.int64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred-dir', required=True, type=Path)
    p.add_argument('--gt-dir', required=True, type=Path)
    p.add_argument('--num-classes', type=int, default=150)
    p.add_argument('--ignore-index', type=int, default=255)
    p.add_argument('--tolerances', nargs='*', type=int, default=[3, 5, 9])
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--device', default='cpu',
                   help='Device for boundary F1 dilations.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if not args.pred_dir.is_dir():
        raise SystemExit(f'pred-dir not found: {args.pred_dir}')
    if not args.gt_dir.is_dir():
        raise SystemExit(f'gt-dir not found: {args.gt_dir}')

    gt_paths = sorted(args.gt_dir.glob('*.png'))
    if args.max_images:
        gt_paths = gt_paths[:args.max_images]
    if not gt_paths:
        raise SystemExit(f'No PNGs in {args.gt_dir}')

    inter = np.zeros(args.num_classes, dtype=np.int64)
    union = np.zeros(args.num_classes, dtype=np.int64)
    pred_total = np.zeros(args.num_classes, dtype=np.int64)
    gt_total = np.zeros(args.num_classes, dtype=np.int64)

    bf1_per_tol: List[List[float]] = [[] for _ in args.tolerances]

    skipped_missing = 0
    skipped_shape = 0
    n_eval = 0
    for i, gt_path in enumerate(gt_paths):
        pred_path = args.pred_dir / gt_path.name
        if not pred_path.exists():
            skipped_missing += 1
            continue
        gt = _load_label(gt_path)
        pred = _load_label(pred_path)
        if pred.shape != gt.shape:
            # nearest-neighbor resize predictions to GT (refine.py should
            # already match, but be defensive).
            pred = cv2.resize(
                pred.astype(np.uint16), (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(np.int64)
            skipped_shape += 1

        # Convert to 0-indexed class ids with ignore_index for "0".
        gt_idx = gt.copy()
        gt_idx[gt_idx == 0] = args.ignore_index + 1  # park 0 above ignore
        gt_idx = gt_idx - 1  # 0..149, ignore -> args.ignore_index
        pred_idx = pred.copy()
        # If pred_dir contains predictions in 1..150 convention (refine.py
        # output), shift to 0..149. If it's already 0..149, nothing changes
        # for valid classes -- but pred==0 may double-shift to ignore, which
        # silently zeros it out. The convention in this repo is that refine.py
        # writes 1..150 so we shift unconditionally.
        pred_idx[pred_idx == 0] = args.ignore_index + 1
        pred_idx = pred_idx - 1

        # IoU counters
        i_h, u_h, p_h, g_h = _intersect_and_union(
            pred_idx, gt_idx, args.num_classes, args.ignore_index)
        inter += i_h
        union += u_h
        pred_total += p_h
        gt_total += g_h

        # Boundary F1 (with tolerance) -- reuse helpers from existing metric.
        gt_t = torch.from_numpy(gt_idx).to(device).long()
        pred_t = torch.from_numpy(pred_idx).to(device).long()
        gt_b = _sem_seg_to_boundary(gt_t, args.ignore_index)
        pred_b = _sem_seg_to_boundary(pred_t, args.ignore_index)
        for k, tol in enumerate(args.tolerances):
            _, _, f1 = _boundary_f1_at_tol(pred_b, gt_b, tol)
            bf1_per_tol[k].append(f1)

        n_eval += 1
        if (i + 1) % 200 == 0 or i + 1 == len(gt_paths):
            print(f'  {i + 1}/{len(gt_paths)}')

    if n_eval == 0:
        raise SystemExit('No (pred, gt) pairs evaluated.')

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    miou = float(np.nanmean(iou)) * 100.0
    pa = float(inter.sum() / max(gt_total.sum(), 1)) * 100.0  # pixel acc.
    print('')
    print('=' * 60)
    print(f'{"Evaluated":<22}: {n_eval} images')
    if skipped_missing:
        print(f'{"Skipped (no pred)":<22}: {skipped_missing}')
    if skipped_shape:
        print(f'{"Pred resized to GT":<22}: {skipped_shape}')
    print(f'{"mIoU":<22}: {miou:.2f}')
    print(f'{"pixel accuracy":<22}: {pa:.2f}')
    for k, tol in enumerate(args.tolerances):
        bf = float(np.mean(bf1_per_tol[k])) * 100.0 if bf1_per_tol[k] else 0.0
        print(f'{"mBoundaryF1_" + str(tol):<22}: {bf:.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
