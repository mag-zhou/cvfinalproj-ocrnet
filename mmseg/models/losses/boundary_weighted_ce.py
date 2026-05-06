# Copyright (c) OpenMMLab. All rights reserved.
"""Boundary-weighted multi-class cross-entropy (Mod 2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class BoundaryWeightedCrossEntropy(nn.Module):
    """Pixel-wise CE with weight ``w(x) = 1 + alpha * exp(-d(x) / sigma)``.

    ``d(x)`` is distance-to-boundary in pixels (from ``gt_boundary_dist``).

    Args:
        alpha (float): Strength of boundary emphasis. Default: 4.0.
        sigma (float): Distance decay (pixels). Default: 5.0.
        loss_weight (float): Overall multiplier. Default: 1.0.
        loss_name (str): Log key under ``decode_*``. Default: ``loss_ce``.
        avg_non_ignore (bool): Average loss over labeled pixels only. Default True.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        sigma: float = 5.0,
        loss_weight: float = 1.0,
        loss_name: str = 'loss_ce',
        avg_non_ignore: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.avg_non_ignore = avg_non_ignore

    @property
    def loss_name(self) -> str:
        return self._loss_name

    def forward(
        self,
        pred: Tensor,
        label: Tensor,
        weight: Tensor = None,
        avg_factor: float = None,
        reduction_override: str = None,
        ignore_index: int = 255,
        boundary_dist: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Compute weighted CE.

        Args:
            pred (Tensor): (N, C, H, W) logits.
            label (Tensor): (N, H, W) long.
            boundary_dist (Tensor): (N, H, W) float.
        """
        del kwargs  # unused; keeps compatibility with caller kwargs
        if boundary_dist is None:
            raise ValueError(
                'BoundaryWeightedCrossEntropy requires `boundary_dist`; '
                'use WeightedCEFCNHead / WeightedCEOCRHead.')

        if weight is not None:
            raise NotImplementedError(
                'BoundaryWeightedCrossEntropy with pixel sampler is not '
                'implemented; leave sampler=None on the decode head.')

        loss_map = F.cross_entropy(
            pred,
            label,
            reduction='none',
            ignore_index=ignore_index)
        if ignore_index is not None:
            # Some PyTorch versions still put non-zero CE on ignore pixels with reduction='none'
            loss_map = loss_map * (label != ignore_index).float()

        pix_w = 1.0 + self.alpha * torch.exp(-boundary_dist / self.sigma)

        if ignore_index is not None:
            valid = (label != ignore_index).float()
            pix_w = pix_w * valid
            denom = valid.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(
                float(label.numel()),
                device=label.device,
                dtype=torch.float32)

        weighted = loss_map * pix_w
        loss = weighted.sum() / denom if self.avg_non_ignore else weighted.mean()

        return self.loss_weight * loss
