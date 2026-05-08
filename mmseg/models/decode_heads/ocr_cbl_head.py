# Copyright (c) OpenMMLab. All rights reserved.
"""OCR head with simplified Conditional Boundary Loss (Mod 4 / CBL-lite).

Adds :class:`BoundaryContrastiveLoss` on top of the standard OCR head.
The contrastive loss is computed on the **augmented (post-attention,
pre-classifier) feature** ``object_context``, which is what gets fed to
``cls_seg`` in the parent :class:`OCRHead`.

Two design choices worth flagging up-front:

- ``object_context`` has ``self.channels`` channels (e.g. 512 in our
  R-50 config), **not** ``self.ocr_channels`` (256). The CBL paper plan
  noted ``D=256`` but that is the *internal* attention dim; the actual
  pre-classifier feature is 512-D. The contrastive loss handles this
  automatically (it reads ``D`` from the tensor shape), but anyone
  re-using this head should know.
- The contrastive loss is computed at the OCR feature resolution
  (~stride 8). ``gt_seg`` and ``gt_boundary`` are downsampled with
  nearest-neighbour to that resolution. This matches the official CBL
  implementation and avoids upsampling 512-D features (very expensive).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from ..losses import BoundaryContrastiveLoss
from ..utils import resize
from .ocr_head import OCRHead


def _stack_padded(tensors: List[Tensor], pad_val: float = 0.0) -> Tensor:
    """Stack a list of (..., H_i, W_i) tensors into a batch.

    The ``SegDataPreProcessor`` pads ``gt_sem_seg`` to ``crop_size`` but
    leaves ``gt_boundary`` (and friends) at the original image size, so
    different samples in a batch may have different H/W. Pad each to the
    batch-max H/W on the bottom-right with ``pad_val`` before stacking.

    This mirrors the helper used in :class:`BoundaryHead` and Mod 3.
    """
    max_h = max(t.shape[-2] for t in tensors)
    max_w = max(t.shape[-1] for t in tensors)
    padded: List[Tensor] = []
    for t in tensors:
        h, w = t.shape[-2], t.shape[-1]
        if h == max_h and w == max_w:
            padded.append(t)
        else:
            padded.append(F.pad(t, (0, max_w - w, 0, max_h - h), value=pad_val))
    return torch.stack(padded, dim=0)


@MODELS.register_module()
class OCRCBLHead(OCRHead):
    """OCR head + simplified Conditional Boundary Loss on augmented features.

    The standard CE seg loss path from :class:`OCRHead` is preserved as
    is. We additionally compute :class:`BoundaryContrastiveLoss` on
    ``object_context`` (the post-attention pre-classifier feature) and
    add it to the loss dict as ``cbl_*`` (default name ``loss_cbl``).

    Args:
        cbl_weight (float): Outer weight on the contrastive loss.
            Default: 1.0.
        cbl_kernel_size (int): K x K window. Default: 5.
        cbl_margin (float): Hinge margin for the push term. Default: 1.0.
        cbl_lambda_neg (float): Push vs. pull weight. Default: 0.5.
        cbl_max_anchors (int): Max anchors subsampled per image.
            Default: 2000.
        cbl_loss_name (str): Key under which the loss is logged.
            Default: ``loss_cbl``.
    """

    def __init__(
        self,
        ocr_channels: int,
        scale: int = 1,
        cbl_weight: float = 1.0,
        cbl_kernel_size: int = 5,
        cbl_margin: float = 1.0,
        cbl_lambda_neg: float = 0.5,
        cbl_max_anchors: int = 2000,
        cbl_loss_name: str = 'loss_cbl',
        **kwargs,
    ) -> None:
        super().__init__(ocr_channels, scale=scale, **kwargs)
        self.cbl_weight = float(cbl_weight)
        self.cbl_kernel_size = int(cbl_kernel_size)
        self.cbl_margin = float(cbl_margin)
        self.cbl_lambda_neg = float(cbl_lambda_neg)
        self.cbl_max_anchors = int(cbl_max_anchors)
        self.cbl_loss_name = str(cbl_loss_name)

        ignore_index = self.ignore_index if self.ignore_index is not None else 255
        self.cbl_loss = BoundaryContrastiveLoss(
            kernel_size=self.cbl_kernel_size,
            margin=self.cbl_margin,
            lambda_neg=self.cbl_lambda_neg,
            max_anchors_per_image=self.cbl_max_anchors,
            ignore_index=ignore_index,
            loss_weight=self.cbl_weight,
            loss_name=self.cbl_loss_name,
        )

    # The parent OCRHead.forward already returns the seg logits. We
    # don't override it (so inference and the val pipeline stay
    # untouched). For training, we re-run the OCR forward in-place in
    # ``loss`` so we can capture ``object_context`` without changing
    # the public ``forward`` signature (which CascadeEncoderDecoder
    # calls during predict()).

    def loss(self, inputs, prev_output, batch_data_samples, train_cfg) -> dict:
        """Forward + CE losses + CBL on augmented features."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        seg_logits = self.cls_seg(object_context)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        cbl_loss_dict = self._cbl_loss(
            object_context, seg_logits, batch_data_samples)
        # Don't silently overwrite a CE-side ``loss_cbl`` (shouldn't happen,
        # but be explicit so a config typo fails loudly).
        for k, v in cbl_loss_dict.items():
            if k in losses:
                raise KeyError(
                    f'CBL loss key {k!r} already present in losses dict; '
                    'rename ``cbl_loss_name`` in the head config.')
            losses[k] = v
        return losses

    def _cbl_loss(
        self,
        object_context: Tensor,
        seg_logits: Tensor,
        batch_data_samples,
    ) -> dict:
        """Stack/downsample boundary GT and call the contrastive loss."""
        if not batch_data_samples:
            return {}
        first = batch_data_samples[0]
        if not hasattr(first, 'gt_boundary') or first.gt_boundary is None:
            # No boundary GT available -> contrastive loss is undefined.
            # Return nothing so this head behaves like a vanilla OCRHead
            # if someone forgets the boundary pipeline. (The Phase 4
            # sanity check in the plan will surface this immediately.)
            return {}

        # ---- Stack gt_seg at full resolution (mmseg pads to crop_size) ---
        seg_gts = [s.gt_sem_seg.data for s in batch_data_samples]
        seg_label = torch.stack(seg_gts, dim=0)  # (B, 1, H, W) long
        if seg_label.dim() == 3:
            seg_label = seg_label.unsqueeze(1)
        seg_label = seg_label.long()

        # ---- Stack gt_boundary, padding to seg_label size --------------
        bds = [s.gt_boundary.data for s in batch_data_samples]
        # First normalise each to have a leading channel dim of 1.
        bds = [b if b.dim() == 3 else b.unsqueeze(0) for b in bds]
        # Pad to seg_label spatial size (which is always crop_size).
        H_full, W_full = seg_label.shape[-2], seg_label.shape[-1]
        bds_pad: List[Tensor] = []
        for b in bds:
            h, w = b.shape[-2], b.shape[-1]
            if (h, w) != (H_full, W_full):
                b = F.pad(
                    b,
                    (0, max(0, W_full - w), 0, max(0, H_full - h)),
                    value=0.0,
                )
                # If a sample is somehow larger than crop_size, crop to
                # crop_size to stay consistent with seg_label.
                if b.shape[-2] > H_full or b.shape[-1] > W_full:
                    b = b[..., :H_full, :W_full]
            bds_pad.append(b)
        gt_boundary = torch.stack(bds_pad, dim=0).float()  # (B, 1, H, W)

        # ---- Downsample seg_label / gt_boundary to feature resolution ---
        H_f, W_f = object_context.shape[-2], object_context.shape[-1]

        if (H_full, W_full) != (H_f, W_f):
            seg_label_ds = F.interpolate(
                seg_label.float(), size=(H_f, W_f), mode='nearest').long()
            gt_boundary_ds = F.interpolate(
                gt_boundary, size=(H_f, W_f), mode='nearest')
        else:
            seg_label_ds = seg_label
            gt_boundary_ds = gt_boundary

        seg_label_ds = seg_label_ds.squeeze(1)      # (B, H_f, W_f) long
        gt_boundary_ds = gt_boundary_ds.squeeze(1)  # (B, H_f, W_f) float

        # seg_logits is already at feature resolution (it came from the
        # same object_context).
        if seg_logits.shape[-2:] != object_context.shape[-2:]:
            seg_logits_for_cbl = resize(
                seg_logits,
                size=(H_f, W_f),
                mode='bilinear',
                align_corners=self.align_corners,
            )
        else:
            seg_logits_for_cbl = seg_logits

        loss_value = self.cbl_loss(
            features=object_context,
            seg_logits=seg_logits_for_cbl,
            gt_seg=seg_label_ds,
            gt_boundary=gt_boundary_ds,
        )

        out = {self.cbl_loss.loss_name: loss_value}
        # Expose anchor stats once in a while for Phase 4 sanity checks.
        # (Logged via the loss dict; mmengine's logger will display any
        # tensor in the dict, so we wrap stats as scalars.)
        n_total = self.cbl_loss.anchor_stats.get('n_total', 0)
        n_valid = self.cbl_loss.anchor_stats.get('n_valid', 0)
        if n_total > 0:
            out['cbl_anchor_ratio'] = torch.tensor(
                float(n_valid) / float(n_total),
                device=object_context.device,
                dtype=torch.float32,
            ).detach()
        return out
