#!/usr/bin/env python3
"""Phase 1: precompute SegFix offset GT to disk for sanity-checking.

NOTE: training does NOT depend on these files. The training transform
(``ComputeOffsetsFromSeg``) recomputes offsets online from the augmented seg
map so that resize / crop / flip remain semantically correct without us
having to hand-roll vector-flip logic. This script exists primarily to:

  - verify the offset algorithm visually on a handful of images
    (``--visualize`` writes magnitude + HSV-direction PNGs);
  - produce ``.npy`` files for any downstream consumer that wants them.

Outputs (when run without ``--visualize``):

    <data_root>/offsets/{training,validation}/<image_id>.npy   (HxWx2 float32)

Run from the repo root, e.g.::

    python segfix/gen_offset_gt.py --visualize --max-images 10
    python segfix/gen_offset_gt.py   # full dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow `python segfix/gen_offset_gt.py` from repo root without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from segfix.offset_utils import (offset_magnitude,  # noqa: E402
                                 offsets_from_seg)


def _load_label(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.int32)


def _direction_hsv(offset: np.ndarray, max_mag: float = 8.0) -> np.ndarray:
    """Color-code a HxWx2 offset map as BGR uint8 image (HSV by angle, value
    by magnitude). Used for sanity visualization only."""
    dy = offset[..., 0]
    dx = offset[..., 1]
    angle = np.arctan2(dy, dx)
    mag = np.sqrt(dy * dy + dx * dx)
    hue = ((angle / np.pi) * 0.5 + 0.5) * 179.0
    sat = np.full_like(hue, 255.0)
    val = np.clip(mag / max_mag, 0, 1) * 255.0
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _process_one_npy(args: Tuple[str, str, int, int]
                     ) -> Tuple[str, Optional[str]]:
    ann_path, out_path, kernel_size, ignore_index = args
    try:
        seg = _load_label(Path(ann_path))
        # ADE20K labels are 0=ignore (255 after reduce_zero_label). The raw
        # files have ``0`` for "background / unlabeled" -- treat 0 as ignore
        # to match the training-time behavior (reduce_zero_label).
        seg_proc = seg.copy()
        seg_proc[seg_proc == 0] = ignore_index
        seg_proc[seg_proc != ignore_index] -= 1
        off = offsets_from_seg(
            seg_proc, kernel_size=kernel_size, ignore_index=ignore_index)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, off.astype(np.float32))
    except Exception as exc:  # noqa: BLE001
        return ann_path, repr(exc)
    return ann_path, None


def _process_one_viz(args: Tuple[str, str, str, int, int, float]
                     ) -> Tuple[str, Optional[str]]:
    ann_path, out_mag_path, out_dir_path, kernel_size, ignore_index, max_mag = args
    try:
        seg = _load_label(Path(ann_path))
        seg_proc = seg.copy()
        seg_proc[seg_proc == 0] = ignore_index
        seg_proc[seg_proc != ignore_index] -= 1
        off = offsets_from_seg(
            seg_proc, kernel_size=kernel_size, ignore_index=ignore_index)
        mag = offset_magnitude(off)
        mag_u8 = np.clip(np.round(mag * (255.0 / max(max_mag, 1.0))), 0,
                         255).astype(np.uint8)
        dir_bgr = _direction_hsv(off, max_mag=max_mag)
        Path(out_mag_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_dir_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_mag_path, mag_u8)
        cv2.imwrite(out_dir_path, dir_bgr)
    except Exception as exc:  # noqa: BLE001
        return ann_path, repr(exc)
    return ann_path, None


def _iter_jobs(data_root: Path, splits: Tuple[str, ...]
               ) -> List[Tuple[str, str]]:
    jobs = []
    for split in splits:
        split_dir = data_root / 'annotations' / split
        if not split_dir.is_dir():
            continue
        for png in sorted(split_dir.glob('*.png')):
            jobs.append((str(png), png.stem))
    return jobs


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path,
                   default=repo / 'data' / 'ade' / 'ADEChallengeData2016')
    p.add_argument('--out-subdir', default='offsets',
                   help='Subfolder under data-root for npy outputs.')
    p.add_argument('--viz-subdir', default='offset_viz',
                   help='Subfolder under data-root for visualization outputs.')
    p.add_argument('--kernel-size', type=int, default=17)
    p.add_argument('--ignore-index', type=int, default=255)
    p.add_argument('--max-mag', type=float, default=8.0,
                   help='Magnitude clip used by --visualize for the colormap.')
    p.add_argument('--splits', nargs='*',
                   default=['training', 'validation'])
    p.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    p.add_argument('--max-images', type=int, default=None,
                   help='Cap number of images processed (sanity runs).')
    p.add_argument('--visualize', action='store_true',
                   help='Write magnitude + HSV-direction PNGs instead of .npy.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f'Data root not found: {args.data_root}')
    raw_jobs = _iter_jobs(args.data_root, tuple(args.splits))
    if args.max_images is not None:
        raw_jobs = raw_jobs[:args.max_images]
    if not raw_jobs:
        raise SystemExit('No annotation PNGs found.')

    if args.visualize:
        jobs = [
            (path,
             str(args.data_root / args.viz_subdir / Path(path).parent.name
                 / f'{stem}_mag.png'),
             str(args.data_root / args.viz_subdir / Path(path).parent.name
                 / f'{stem}_dir.png'),
             args.kernel_size, args.ignore_index, args.max_mag)
            for path, stem in raw_jobs
        ]
        print(f'Visualizing {len(jobs)} images into '
              f'{args.data_root / args.viz_subdir}')
        worker = _process_one_viz
    else:
        jobs = [
            (path,
             str(args.data_root / args.out_subdir / Path(path).parent.name
                 / f'{stem}.npy'),
             args.kernel_size, args.ignore_index)
            for path, stem in raw_jobs
        ]
        print(f'Writing {len(jobs)} offset .npy into '
              f'{args.data_root / args.out_subdir}')
        worker = _process_one_npy

    errors = []
    with Pool(processes=args.workers) as pool:
        for i, (path, err) in enumerate(
                pool.imap_unordered(worker, jobs, chunksize=4)):
            if err:
                errors.append((path, err))
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f'  {i + 1}/{len(jobs)}')

    if errors:
        print(f'\nFailed ({len(errors)}):')
        for p, e in errors[:20]:
            print(f'  {p}: {e}')
        raise SystemExit(1)
    print('Done.')


if __name__ == '__main__':
    main()
