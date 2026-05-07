"""SegFix offset / boundary computation (numpy/cv2).

Used by both the training-time transform (``segfix/transforms.py``) and the
sanity-check precompute script (``segfix/gen_offset_gt.py``).

Algorithm (per the SegFix paper, Yuan et al. 2020):

    For each class ``c`` present in a label map ``L``:
      1. Take the binary class mask ``M_c = (L == c)``.
      2. Compute the L2 distance transform of ``M_c`` -- this gives, for every
         pixel inside ``M_c``, its Euclidean distance to the nearest non-``c``
         pixel ("depth into region"). Outside ``M_c`` the value is 0.
      3. For every pixel ``(i, j)`` of class ``c``, look in a (2k+1) x (2k+1)
         window in the distance map and find the location of the **maximum**
         distance value. The offset to that location, ``(di, dj) = (i*-i, j*-j)``,
         points "inward" toward the most interior point of the local region.

Boundary mask (predicted by the boundary head): standard 8-neighbor class
disagreement on the seg map. Same definition the BoundaryFScore metric and
``tools/preprocess/gen_boundary_gt.py`` already use, kept consistent on
purpose.

CAVEAT (flagged in the implementation plan): openseg.pytorch's
``dt_offset_generator.py`` is the reference impl and may differ in small
details (kernel size, distance metric, exact handling of ties). This module
mirrors the conceptual recipe; if results disappoint, cross-check against
openseg before assuming the model architecture is the bug.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def boundary_mask_from_seg(
    seg: np.ndarray,
    ignore_index: int = 255,
) -> np.ndarray:
    """8-neighbor class-disagreement boundary, ignoring ``ignore_index``.

    Args:
        seg: int HxW class map.
        ignore_index: pixels with this value never participate as either side
            of a boundary.

    Returns:
        bool HxW.
    """
    seg = seg.astype(np.int32, copy=False)
    h, w = seg.shape
    bd = np.zeros((h, w), dtype=bool)
    valid_center = seg != ignore_index
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
            valid_nbr = nbr != ignore_index
            diff = (sub != nbr) & valid_nbr
            bd[i_lo:i_hi, j_lo:j_hi] |= diff
    bd &= valid_center
    return bd


def _argmax_offset_in_window(
    arr: np.ndarray,
    kernel_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """For every pixel, find the (di, dj) offset to the maximum value in the
    surrounding window of size ``kernel_size`` x ``kernel_size``.

    Pixels at the edge of ``arr`` use a constant ``-inf`` pad so that out-of-
    image positions can never be argmax winners.

    Returns:
        di, dj: float32 HxW arrays in range [-pad, pad] where pad = K//2.
    """
    assert kernel_size % 2 == 1, 'kernel_size must be odd'
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode='constant', constant_values=-np.inf)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    h, w = arr.shape
    flat = windows.reshape(h, w, kernel_size * kernel_size)
    flat_argmax = np.argmax(flat, axis=-1)
    di = (flat_argmax // kernel_size).astype(np.int32) - pad
    dj = (flat_argmax % kernel_size).astype(np.int32) - pad
    return di.astype(np.float32), dj.astype(np.float32)


def offsets_from_seg(
    seg: np.ndarray,
    kernel_size: int = 17,
    ignore_index: int = 255,
    only_near_boundary: bool = True,
) -> np.ndarray:
    """Compute SegFix-style offsets from a class label map.

    For each pixel of class ``c``, the offset points toward the local-max
    distance-to-non-``c`` location within a ``kernel_size`` window. Pixels of
    class ``ignore_index`` (or with no neighbours of the same class within
    range) get zero offset.

    Args:
        seg: int HxW class map (any int dtype).
        kernel_size: side length of the offset search window (odd; the paper
            uses values around 17 at full resolution).
        ignore_index: skipped during processing.
        only_near_boundary: if True, zero out offsets for pixels more than
            ``kernel_size//2`` from any class boundary. Matches the SegFix-paper
            expectation (and the plan's sanity check) that offsets are
            "near-zero in deep interiors". The training loss is masked to GT-
            boundary pixels anyway, so this is mostly cosmetic for the
            visualization, but it also removes meaningless gradient signal at
            inference if the boundary head ever spuriously fires far from any
            real boundary.

    Returns:
        float32 HxWx2 array with channel order ``(dy, dx)``.
    """
    assert seg.ndim == 2, f'expected HxW, got {seg.shape}'
    h, w = seg.shape
    offset = np.zeros((h, w, 2), dtype=np.float32)
    seg_int = seg.astype(np.int32, copy=False)
    classes = np.unique(seg_int)
    classes = classes[classes != ignore_index]

    # Tiny center-prefer bias: when distance values within a window are
    # numerically tied (cv2's DIST_L2 with 5x5 mask is approximate), prefer
    # the center pixel (offset = 0) over a row-major-first corner pixel.
    pad = kernel_size // 2
    yy, xx = np.indices((kernel_size, kernel_size))
    center_bias = -((yy - pad) ** 2 + (xx - pad) ** 2) * 1e-3

    for c in classes:
        mask_c = (seg_int == c)
        if not mask_c.any():
            continue
        # cv2.distanceTransform: distance to nearest 0 pixel. We pass the mask
        # as uint8 with 1 inside class c, 0 elsewhere -> distances are 0 outside
        # the class and positive (depth-into-region) inside.
        dist = cv2.distanceTransform(
            mask_c.astype(np.uint8), cv2.DIST_L2, 5)
        # Argmax inside `dist` -- since `dist` is 0 outside the class, the local
        # maximum within any window will be inside the class (or zero if the
        # whole window is non-class, in which case offset stays 0).
        di, dj = _argmax_offset_in_window_with_bias(
            dist, kernel_size, center_bias)
        # Only assign offsets for pixels that actually belong to class c.
        offset[mask_c, 0] = di[mask_c]
        offset[mask_c, 1] = dj[mask_c]

    if only_near_boundary:
        # Per-pixel L2 distance to the nearest 8-neighbor class boundary.
        bd = boundary_mask_from_seg(seg_int, ignore_index=ignore_index)
        if bd.any():
            inv = (1 - bd.astype(np.uint8)).astype(np.uint8)
            bd_dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            max_off = float(kernel_size // 2)
            deep = bd_dist > max_off
            offset[deep] = 0.0
    return offset


def _argmax_offset_in_window_with_bias(
    arr: np.ndarray,
    kernel_size: int,
    center_bias: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as ``_argmax_offset_in_window`` but adds a tiny per-position
    center-preference bias to break ties toward (0, 0)."""
    assert kernel_size % 2 == 1
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode='constant', constant_values=-np.inf)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    h, w = arr.shape
    biased = windows + center_bias  # broadcast (1, 1, K, K)
    flat = biased.reshape(h, w, kernel_size * kernel_size)
    flat_argmax = np.argmax(flat, axis=-1)
    di = (flat_argmax // kernel_size).astype(np.int32) - pad
    dj = (flat_argmax % kernel_size).astype(np.int32) - pad
    return di.astype(np.float32), dj.astype(np.float32)


def offset_magnitude(offset: np.ndarray) -> np.ndarray:
    """Convenience: per-pixel L2 magnitude of an HxWx2 offset map."""
    return np.sqrt(offset[..., 0] ** 2 + offset[..., 1] ** 2)
