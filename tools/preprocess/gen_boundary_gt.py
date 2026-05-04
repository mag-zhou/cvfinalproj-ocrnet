#!/usr/bin/env python3
"""Precompute binary boundary maps and distance-to-boundary maps for ADE20K.

Reads label PNGs under ``<data_root>/annotations/{training,validation}/`` and writes:
  - ``<data_root>/boundaries/{split}/`` — uint8 PNG, 255 on class-transition pixels, 0 elsewhere.
  - ``<data_root>/boundary_dist/{split}/`` — uint8 PNG, min(distance in pixels, --max-dist).

Run once from repo root (after ADE20K is downloaded), with multiprocessing::

    python tools/preprocess/gen_boundary_gt.py
    python tools/preprocess/gen_boundary_gt.py --data-root /path/to/ADEChallengeData2016

Uses an 8-neighbor difference rule on class IDs (same as checklist). Distance is L2 from
each pixel to the nearest boundary pixel via ``cv2.distanceTransform``.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _load_semantic_label(path: Path) -> np.ndarray:
    """Load single-channel class map as int32 HxW."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.int32)


def semantic_boundary_mask(seg: np.ndarray) -> np.ndarray:
    """Binary boundary: True where any 8-neighbor has a different class."""
    h, w = seg.shape
    bd = np.zeros((h, w), dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            i_lo = max(0, -di)
            i_hi = min(h, h - di)
            j_lo = max(0, -dj)
            j_hi = min(w, w - dj)
            if i_lo >= i_hi or j_lo >= j_hi:
                continue
            sub = seg[i_lo:i_hi, j_lo:j_hi]
            nbr = seg[i_lo + di:i_hi + di, j_lo + dj:j_hi + dj]
            bd[i_lo:i_hi, j_lo:j_hi] |= sub != nbr
    return bd


def distance_to_boundary(boundary01: np.ndarray, max_dist: float) -> np.ndarray:
    """boundary01: float/bool, 1 on boundary else 0. Returns float32 distances (clipped)."""
    # cv2.distanceTransform: distance to nearest *zero* pixel; boundary must be 0.
    src = np.where(boundary01 > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    dist = np.minimum(dist, max_dist)
    return dist.astype(np.float32)


def process_one(
    args: Tuple[str, str, str, float],
) -> Tuple[str, Optional[str]]:
    """Returns (annotation_path, error_message_or_None)."""
    ann_path, out_b_path, out_d_path, max_dist = args
    try:
        seg = _load_semantic_label(Path(ann_path))
        bd = semantic_boundary_mask(seg)
        boundary_u8 = (bd.astype(np.uint8) * 255)

        bd01 = bd.astype(np.float32)
        dist = distance_to_boundary(bd01, max_dist=float(max_dist))
        dist_u8 = np.clip(np.round(dist), 0, max_dist).astype(np.uint8)

        Path(out_b_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_d_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_b_path, boundary_u8)
        cv2.imwrite(out_d_path, dist_u8)
    except Exception as exc:  # noqa: BLE001 — surface worker failures
        return ann_path, repr(exc)
    return ann_path, None


def iter_annotation_jobs(
    data_root: Path,
    splits: Tuple[str, ...],
    max_dist: float,
) -> List[Tuple[str, str, str, float]]:
    jobs = []
    ann_root = data_root / 'annotations'
    for split in splits:
        split_dir = ann_root / split
        if not split_dir.is_dir():
            continue
        for png in sorted(split_dir.glob('*.png')):
            rel = png.name
            out_b = data_root / 'boundaries' / split / rel
            out_d = data_root / 'boundary_dist' / split / rel
            jobs.append((str(png), str(out_b), str(out_d), max_dist))
    return jobs


def default_data_root() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / 'data' / 'ade' / 'ADEChallengeData2016'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='ADE20K root containing annotations/ (default: <repo>/data/ade/ADEChallengeData2016)',
    )
    p.add_argument(
        '--max-dist',
        type=float,
        default=20.0,
        help='Clip distance transform at this many pixels (default: 20)',
    )
    p.add_argument(
        '--workers',
        type=int,
        default=max(1, cpu_count() - 1),
        help='Pool workers (default: CPU count minus one)',
    )
    p.add_argument(
        '--splits',
        nargs='*',
        default=['training', 'validation'],
        help='Which annotation splits to scan (default: training validation)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root or default_data_root()
    if not data_root.is_dir():
        raise SystemExit(
            f'Data root not found: {data_root}\n'
            'Download ADE20K and unzip so annotations live under that path.')

    jobs = iter_annotation_jobs(data_root, tuple(args.splits), args.max_dist)
    if not jobs:
        raise SystemExit(f'No PNGs found under {data_root}/annotations/{{training,validation}}/')

    print(f'Data root: {data_root}')
    print(f'Jobs: {len(jobs)}  workers: {args.workers}  max_dist: {args.max_dist}')

    errors = []
    with Pool(processes=args.workers) as pool:
        for i, (path, err) in enumerate(pool.imap_unordered(process_one, jobs, chunksize=8)):
            if err:
                errors.append((path, err))
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f'  finished {i + 1}/{len(jobs)}')

    if errors:
        print(f'\nFailed ({len(errors)}):')
        for path, err in errors[:20]:
            print(f'  {path}: {err}')
        if len(errors) > 20:
            print(f'  ... and {len(errors) - 20} more')
        raise SystemExit(1)

    print('Done. Outputs:')
    print(f'  {data_root}/boundaries/{{training,validation}}/')
    print(f'  {data_root}/boundary_dist/{{training,validation}}/')


if __name__ == '__main__':
    main()
