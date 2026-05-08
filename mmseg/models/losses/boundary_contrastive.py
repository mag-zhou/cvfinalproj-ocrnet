# Copyright (c) OpenMMLab. All rights reserved.
"""Simplified Conditional Boundary Loss (Mod 4 / CBL-lite).

Captures the core idea of `Wu et al., "Conditional Boundary Loss for
Semantic Segmentation" (TIP 2023) <https://arxiv.org/abs/2307.02174>`_:
for each boundary pixel, pull its feature toward correctly-classified
same-class neighbours (CCAS positives) and push it away from
correctly-classified different-class neighbours (CCAS negatives).

What this implementation **does** capture:

- The CCAS (Correctly-Classified Anchor Set) filtering rule: anchors that
  cannot find any correctly-classified same-class neighbour are skipped.
- The A2C-pair-style pull term: anchor -> mean of local positives.
- A hinge-style push term over the local negative set Z^-.

What this implementation **omits** vs. the full paper (and why):

- No SCE-supervised local class-center prediction head (no auxiliary loss
  forcing a separate branch to *predict* the local class centres).
- No frozen depthwise-conv parallel local-class-center generator. We
  compute centres on the fly from features at boundary anchors only,
  which is cheap because there are O(thousands) anchors per image.

We chose to write this rather than port the official repo because (a)
the official code is built on a different framework, (b) we want a
loss small enough to slot into mmsegmentation as a single module that
plays well with our Mod 1 boundary GT pipeline, and (c) we explicitly
want to test the "feature-space contrastive boundary loss alone" axis
in our ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class BoundaryContrastiveLoss(nn.Module):
    """Contrastive boundary loss on a per-pixel feature map.

    Forward:

    .. code-block:: text

        forward(features, seg_logits, gt_seg, gt_boundary) -> scalar

    where all tensors are at the **same spatial resolution** (typically
    the OCR head's pre-classifier feature resolution, ~stride 8).

    For each boundary anchor pixel ``f_i`` at position ``(b, y, x)``:

    1. Look at the ``K x K`` window around ``(y, x)``.
    2. ``window_correct = (argmax(seg_logits) == gt_seg) & (gt_seg != ignore)``.
    3. Same-class correct neighbours -> Z^+ (positives).
    4. Different-class correct, non-ignored neighbours -> Z^- (negatives).
    5. Skip anchor if ``|Z^+| == 0``.
    6. ``c_i = mean(Z^+)`` (detached).
    7. Loss contribution::

        ||f_i - c_i||^2  +  lambda_neg * mean_{z- in Z^-} max(0, m - ||f_i - z-||^2)

    Centres and negatives are **detached** before the distance computation,
    so only the anchor feature receives gradient. This matches the paper's
    A2C / A2N formulation and avoids degenerate "all features collapse"
    solutions.

    Args:
        kernel_size (int): Window size ``K`` (odd). Default: 5.
        margin (float): Hinge margin ``m`` for the push term. Default: 1.0.
        lambda_neg (float): Weight of push vs. pull. Default: 0.5.
        max_anchors_per_image (int): Random subsample boundary anchors per
            image to bound memory. Default: 2000.
        ignore_index (int): Label value to skip. Default: 255.
        loss_weight (float): Outer multiplier on the loss. Default: 1.0.
        loss_name (str): Key used when this loss is added to a head's
            loss dict. Default: ``loss_cbl``.
        boundary_threshold (float): Threshold on (float) gt_boundary to
            decide which pixels are anchors. Default: 0.5.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        margin: float = 1.0,
        lambda_neg: float = 0.5,
        max_anchors_per_image: int = 2000,
        ignore_index: int = 255,
        loss_weight: float = 1.0,
        loss_name: str = 'loss_cbl',
        boundary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f'kernel_size must be an odd int >= 3, got {kernel_size}')
        self.kernel_size = int(kernel_size)
        self.margin = float(margin)
        self.lambda_neg = float(lambda_neg)
        self.max_anchors_per_image = int(max_anchors_per_image)
        self.ignore_index = int(ignore_index)
        self.loss_weight = float(loss_weight)
        self._loss_name = str(loss_name)
        self.boundary_threshold = float(boundary_threshold)

        # Updated each forward; useful for sanity-check logging from the
        # head (e.g., Phase 4 anchor-count sanity).
        self._anchor_stats = {'n_total': 0, 'n_valid': 0}

    @property
    def loss_name(self) -> str:
        return self._loss_name

    @property
    def anchor_stats(self) -> dict:
        """Last-forward diagnostics: total candidate anchors, valid (with Z+)."""
        return dict(self._anchor_stats)

    def forward(
        self,
        features: Tensor,
        seg_logits: Tensor,
        gt_seg: Tensor,
        gt_boundary: Tensor,
    ) -> Tensor:
        """Compute the contrastive boundary loss.

        Args:
            features (Tensor): ``(B, D, H, W)`` pre-classifier features.
            seg_logits (Tensor): ``(B, C, H, W)`` segmentation logits at
                the same resolution as ``features``.
            gt_seg (Tensor): ``(B, H, W)`` long, ``ignore_index`` allowed.
            gt_boundary (Tensor): ``(B, H, W)`` float; non-zero where
                anchors live.

        Returns:
            Tensor: scalar loss value (already multiplied by ``loss_weight``).
        """
        if features.dim() != 4 or seg_logits.dim() != 4:
            raise ValueError(
                'features and seg_logits must be (B, D, H, W). Got '
                f'{tuple(features.shape)} and {tuple(seg_logits.shape)}.')
        B, D, H, W = features.shape
        if seg_logits.shape[0] != B or seg_logits.shape[2:] != (H, W):
            raise ValueError(
                'features and seg_logits must agree on B/H/W. Got '
                f'features={tuple(features.shape)} '
                f'seg_logits={tuple(seg_logits.shape)}.')
        if gt_seg.shape != (B, H, W):
            raise ValueError(
                f'gt_seg must be ({B}, {H}, {W}); got {tuple(gt_seg.shape)}')
        if gt_boundary.shape != (B, H, W):
            raise ValueError(
                f'gt_boundary must be ({B}, {H}, {W}); got '
                f'{tuple(gt_boundary.shape)}')

        device = features.device
        K = self.kernel_size
        pad = K // 2

        # ---- Steps 1-2: correctness mask + anchor candidate mask ---------
        with torch.no_grad():
            pred = seg_logits.argmax(dim=1)
            valid = gt_seg != self.ignore_index
            correct = (pred == gt_seg) & valid
            anchor_mask = (gt_boundary > self.boundary_threshold) & valid

            # Replace ignore labels with 0 so a padded gt window can't
            # accidentally equal the (always-valid) anchor label. We still
            # filter on `window_correct` and `window_gt != ignore_index`,
            # but using ignore-as-pad makes the negative filter exclude
            # padded cells trivially (see the pad below).
            gt_safe = torch.where(valid, gt_seg, torch.zeros_like(gt_seg))

        # ---- Step 3: subsample anchor coordinates per image --------------
        anchor_b_list: list = []
        anchor_y_list: list = []
        anchor_x_list: list = []
        for b in range(B):
            ys, xs = anchor_mask[b].nonzero(as_tuple=True)
            n = ys.numel()
            if n == 0:
                continue
            if n > self.max_anchors_per_image:
                perm = torch.randperm(n, device=device)[:self.max_anchors_per_image]
                ys = ys[perm]
                xs = xs[perm]
            anchor_b_list.append(torch.full_like(ys, b))
            anchor_y_list.append(ys)
            anchor_x_list.append(xs)

        if not anchor_b_list:
            self._anchor_stats = {'n_total': 0, 'n_valid': 0}
            # Zero loss that still has a path back to the graph so the
            # optimiser sees a (zero) gradient for `features`.
            return self.loss_weight * (features.sum() * 0.0)

        anchor_b = torch.cat(anchor_b_list)
        anchor_y = torch.cat(anchor_y_list)
        anchor_x = torch.cat(anchor_x_list)
        N = anchor_b.numel()

        # ---- Step 4: gather K x K windows at anchor positions only -------
        # Pad once; gather only at anchor (b, y, x) -> (N, K^2) indices.
        # This avoids the full F.unfold tensor which would be (B*H*W, D, K^2)
        # in the worst case.
        feat_pad = F.pad(features, (pad, pad, pad, pad), mode='constant', value=0.0)
        # Pad gt with ignore so out-of-image cells are never positives or
        # negatives. (correct is zero outside, so it doubly excludes them.)
        gt_pad = F.pad(
            gt_safe.unsqueeze(1).long(),
            (pad, pad, pad, pad),
            mode='constant',
            value=self.ignore_index,
        ).squeeze(1)
        correct_pad = F.pad(
            correct.unsqueeze(1).float(),
            (pad, pad, pad, pad),
            mode='constant',
            value=0.0,
        ).squeeze(1).bool()

        # Offsets for the K x K window in padded coords.
        dy = torch.arange(K, device=device)
        dx = torch.arange(K, device=device)
        dy, dx = torch.meshgrid(dy, dx, indexing='ij')
        dy = dy.flatten()  # (K^2,)
        dx = dx.flatten()  # (K^2,)

        # In padded coords the anchor's window top-left is exactly
        # (anchor_y, anchor_x) (because anchor_y_pad = anchor_y + pad and
        # we want anchor_y_pad - pad = anchor_y).
        y_idx = anchor_y.unsqueeze(1) + dy.unsqueeze(0)  # (N, K^2)
        x_idx = anchor_x.unsqueeze(1) + dx.unsqueeze(0)  # (N, K^2)
        b_idx = anchor_b.unsqueeze(1).expand(-1, K * K)   # (N, K^2)

        # Mixed indexing: (B, D, Hp, Wp)[b_idx, :, y_idx, x_idx]
        # -> result has shape (N, K^2, D) with dim 1 (the sliced D) trailing.
        win_feats = feat_pad[b_idx, :, y_idx, x_idx]  # (N, K^2, D)
        win_feats = win_feats.permute(0, 2, 1).contiguous()  # (N, D, K^2)

        # gt and correct windows (no D dim).
        win_gt = gt_pad[b_idx, y_idx, x_idx]            # (N, K^2)
        win_correct = correct_pad[b_idx, y_idx, x_idx]  # (N, K^2)

        # Anchor's own feature & label.
        anchor_feat = features[anchor_b, :, anchor_y, anchor_x]  # (N, D)
        anchor_label = gt_safe[anchor_b, anchor_y, anchor_x]      # (N,)

        # ---- Step 5: positives / negatives within each window ------------
        same_class = win_gt == anchor_label.unsqueeze(1)
        is_pos = same_class & win_correct
        is_neg = (~same_class) & win_correct & (win_gt != self.ignore_index)

        n_pos = is_pos.sum(dim=1)  # (N,)
        valid_anchor = n_pos > 0

        # Bookkeeping for sanity tests.
        self._anchor_stats = {
            'n_total': int(N),
            'n_valid': int(valid_anchor.sum().item()),
        }

        if not bool(valid_anchor.any()):
            return self.loss_weight * (features.sum() * 0.0)

        # ---- Step 6+: keep only valid anchors ----------------------------
        anchor_feat = anchor_feat[valid_anchor]            # (M, D)
        win_feats = win_feats[valid_anchor]                # (M, D, K^2)
        is_pos = is_pos[valid_anchor]                      # (M, K^2)
        is_neg = is_neg[valid_anchor]                      # (M, K^2)
        n_pos_f = n_pos[valid_anchor].clamp_min(1).float()  # (M,)
        n_neg_f = is_neg.sum(dim=1).float()                 # (M,)

        # Detach window features: the centre and the negatives must NOT
        # receive gradient or the contrastive objective collapses.
        win_feats_d = win_feats.detach()  # (M, D, K^2)

        # ---- Step 7: local class centre = mean of detached positives -----
        pos_mask_f = is_pos.unsqueeze(1).float()                      # (M, 1, K^2)
        center = (win_feats_d * pos_mask_f).sum(dim=2) / n_pos_f.unsqueeze(1)  # (M, D)

        # ---- Step 8: pull term (squared L2) ------------------------------
        pull = ((anchor_feat - center) ** 2).sum(dim=1)  # (M,)
        pull_mean = pull.mean()

        # ---- Step 9: push term (hinge over Z^-) --------------------------
        # Use the algebraic distance decomposition to avoid materialising
        # `anchor_feat[:, :, None] - win_feats_d` (which would be M*D*K^2).
        a_sq = (anchor_feat ** 2).sum(dim=1, keepdim=True)        # (M, 1)
        b_sq = (win_feats_d ** 2).sum(dim=1)                       # (M, K^2)
        ab = torch.einsum('md,mdk->mk', anchor_feat, win_feats_d)  # (M, K^2)
        dist_sq = (a_sq + b_sq - 2.0 * ab).clamp_min(0.0)          # (M, K^2)

        hinge = (self.margin - dist_sq).clamp_min(0.0)             # (M, K^2)
        # Mask to negatives only; if no negatives for an anchor, push is 0.
        push_per_anchor = (hinge * is_neg.float()).sum(dim=1) / n_neg_f.clamp_min(1.0)
        push_per_anchor = push_per_anchor * (n_neg_f > 0).float()
        push_mean = push_per_anchor.mean()

        loss = pull_mean + self.lambda_neg * push_mean
        return self.loss_weight * loss

    def extra_repr(self) -> str:
        return (
            f'kernel_size={self.kernel_size}, margin={self.margin}, '
            f'lambda_neg={self.lambda_neg}, '
            f'max_anchors_per_image={self.max_anchors_per_image}, '
            f'ignore_index={self.ignore_index}, '
            f'loss_weight={self.loss_weight}, '
            f'loss_name={self._loss_name!r}')
