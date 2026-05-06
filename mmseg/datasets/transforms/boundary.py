# Copyright (c) OpenMMLab. All rights reserved.
"""Boundary ground-truth loading and packing for boundary supervision experiments.

Requires precomputed PNGs from ``tools/preprocess/gen_boundary_gt.py``. Paths are
derived from ``seg_map_path`` by replacing ``/annotations/`` with
``/boundaries/`` and ``/boundary_dist/``.

Both ``gt_boundary`` and ``gt_boundary_dist`` are appended to ``seg_fields`` so
:class:`RandomResize`, :class:`RandomCrop`, :class:`RandomFlip`, and
:class:`Resize` apply the same geometric transforms as ``gt_seg_map``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData

from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.registry import TRANSFORMS


def boundary_paths_from_seg_map(
    seg_map_path: str,
    ann_segment: str = 'annotations',
    boundary_segment: str = 'boundaries',
    dist_segment: str = 'boundary_dist',
) -> Tuple[str, str]:
    """Map ``.../annotations/{split}/file.png`` to boundary / dist paths."""
    norm = seg_map_path.replace('\\', '/')
    token = f'/{ann_segment}/'
    if token not in norm:
        raise ValueError(
            f'Expected `{token}` in seg_map_path for boundary path derivation, '
            f'got: {seg_map_path}')
    b_norm = norm.replace(token, f'/{boundary_segment}/', 1)
    d_norm = norm.replace(token, f'/{dist_segment}/', 1)
    return str(Path(b_norm)), str(Path(d_norm))


@TRANSFORMS.register_module()
class LoadBoundaryAnnotations(BaseTransform):
    """Load precomputed binary boundary and distance-to-boundary maps.

    Required Keys:

    - seg_map_path

    Added Keys:

    - gt_boundary (np.ndarray, uint8, HxW): values in {0, 255}.
    - gt_boundary_dist (np.ndarray, float32, HxW): clipped distance in pixels.
    - seg_fields (extended): ``gt_boundary``, ``gt_boundary_dist``

    Args:
        backend_args (dict, optional): Passed to fileio.get.
        imdecode_backend (str): Backend for :func:`mmcv.imfrombytes`.
        ann_segment (str): Path segment that identifies annotation folder
            (default: ``annotations``).
        boundary_segment (str): Folder name for binary boundaries.
        dist_segment (str): Folder name for distance transforms.
    """

    def __init__(
        self,
        backend_args: Optional[dict] = None,
        imdecode_backend: str = 'pillow',
        ann_segment: str = 'annotations',
        boundary_segment: str = 'boundaries',
        dist_segment: str = 'boundary_dist',
    ) -> None:
        self.backend_args = backend_args
        self.imdecode_backend = imdecode_backend
        self.ann_segment = ann_segment
        self.boundary_segment = boundary_segment
        self.dist_segment = dist_segment

    def transform(self, results: dict) -> dict:
        if 'seg_map_path' not in results:
            raise KeyError('LoadBoundaryAnnotations needs seg_map_path')
        if 'seg_fields' not in results:
            results['seg_fields'] = []

        b_path, d_path = boundary_paths_from_seg_map(
            results['seg_map_path'],
            self.ann_segment,
            self.boundary_segment,
            self.dist_segment,
        )

        b_bytes = fileio.get(b_path, backend_args=self.backend_args)
        d_bytes = fileio.get(d_path, backend_args=self.backend_args)
        b_img = mmcv.imfrombytes(
            b_bytes, flag='unchanged', backend=self.imdecode_backend)
        d_img = mmcv.imfrombytes(
            d_bytes, flag='unchanged', backend=self.imdecode_backend)

        if b_img is None or d_img is None:
            raise FileNotFoundError(
                f'Failed to load boundary assets.\n  boundary: {b_path}\n'
                f'  distance: {d_path}\n'
                'Run `python tools/preprocess/gen_boundary_gt.py` first.')

        if b_img.ndim == 3:
            b_img = b_img[:, :, 0]
        if d_img.ndim == 3:
            d_img = d_img[:, :, 0]

        results['gt_boundary'] = b_img.astype(np.uint8)
        results['gt_boundary_dist'] = d_img.astype(np.float32)

        for key in ('gt_boundary', 'gt_boundary_dist'):
            if key not in results['seg_fields']:
                results['seg_fields'].append(key)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'ann_segment={self.ann_segment!r}, '
                f'boundary_segment={self.boundary_segment!r}, '
                f'dist_segment={self.dist_segment!r})')


@TRANSFORMS.register_module()
class PackSegBoundaryInputs(PackSegInputs):
    """Pack semantic maps plus ``gt_boundary`` and ``gt_boundary_dist``.

    Extends :class:`PackSegInputs` by moving boundary tensors into
    :class:`~mmseg.structures.SegDataSample` as extra ``PixelData`` entries.
    """

    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        data_sample = packed_results['data_samples']
        extra = {}
        if 'gt_boundary' in results:
            bd = results['gt_boundary']
            # (1, H, W) float in {0, 1} for BCE-style targets
            t = to_tensor((bd.astype(np.float32) / 255.0)[None, ...])
            extra['gt_boundary'] = PixelData(data=t)
        if 'gt_boundary_dist' in results:
            dd = results['gt_boundary_dist']
            t = to_tensor(dd[None, ...].astype(np.float32))
            extra['gt_boundary_dist'] = PixelData(data=t)
        if extra:
            data_sample.set_data(extra)
        return packed_results
