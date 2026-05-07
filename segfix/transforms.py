"""Pipeline transforms for the SegFix offset model.

We deliberately compute boundary + offsets ONLINE from the (already
augmented) ``gt_seg_map`` instead of loading precomputed targets from disk.
Reasons:

  - Offset is a 2D vector field; horizontal flip flips the array AND negates
    dx, and resize multiplies magnitudes by the scale factor. Forcing the
    standard mmseg geometric transforms to handle that correctly requires
    monkey-patching ``RandomFlip`` / ``RandomResize`` / ``RandomCrop`` or
    keeping a parallel, custom pipeline. Both are fragile.
  - Recomputing from the augmented seg map is exactly equivalent to "the
    correct geometric transform of the precomputed offset" but cannot drift
    out of sync.
  - Cost is fine (~tens of ms per image, distance transform per class) and
    runs in the dataloader workers, off the critical path.

Ground truth boundary is recomputed here too, for the same reason.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData

from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.registry import TRANSFORMS

from .offset_utils import boundary_mask_from_seg, offsets_from_seg


@TRANSFORMS.register_module()
class ComputeOffsetsFromSeg(BaseTransform):
    """Add ``gt_offset`` (HxWx2 float32) and ``gt_offset_boundary`` (HxW
    uint8) to results, derived from the current ``gt_seg_map``.

    Place this AFTER all geometric augmentations (RandomResize, RandomCrop,
    RandomFlip) and BEFORE ``PackSegFixInputs``. The seg map at that point
    is in its final training-time form, so freshly-computed offsets/boundary
    are automatically aligned.

    Required keys: ``gt_seg_map``.
    Added keys:    ``gt_offset``, ``gt_offset_boundary``.
    """

    def __init__(self,
                 kernel_size: int = 17,
                 ignore_index: int = 255) -> None:
        assert kernel_size % 2 == 1, 'kernel_size must be odd'
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' not in results:
            raise KeyError(
                'ComputeOffsetsFromSeg needs gt_seg_map (use LoadAnnotations).')
        seg = results['gt_seg_map']
        if seg.ndim == 3:
            seg = seg[..., 0]
        bd = boundary_mask_from_seg(seg, ignore_index=self.ignore_index)
        off = offsets_from_seg(
            seg,
            kernel_size=self.kernel_size,
            ignore_index=self.ignore_index)
        results['gt_offset'] = off.astype(np.float32)
        results['gt_offset_boundary'] = bd.astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(kernel_size={self.kernel_size}, '
                f'ignore_index={self.ignore_index})')


@TRANSFORMS.register_module()
class PackSegFixInputs(PackSegInputs):
    """Pack everything ``PackSegInputs`` packs, plus offset/boundary tensors
    onto :class:`~mmseg.structures.SegDataSample`.

    Adds:

    - ``gt_offset`` PixelData with shape (2, H, W) -- channel 0 = dy, 1 = dx.
    - ``gt_offset_boundary`` PixelData with shape (1, H, W) in {0, 1}.
    """

    def transform(self, results: dict) -> dict:
        packed = super().transform(results)
        sample = packed['data_samples']
        extra = {}
        if 'gt_offset' in results:
            off = np.asarray(results['gt_offset'], dtype=np.float32)
            # H, W, 2 -> 2, H, W
            t = to_tensor(np.transpose(off, (2, 0, 1)).copy())
            extra['gt_offset'] = PixelData(data=t)
        if 'gt_offset_boundary' in results:
            bd = np.asarray(results['gt_offset_boundary'], dtype=np.float32)
            t = to_tensor(bd[None, ...])
            extra['gt_offset_boundary'] = PixelData(data=t)
        if extra:
            sample.set_data(extra)
        return packed
