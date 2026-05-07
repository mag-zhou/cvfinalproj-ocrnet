# Copyright (c) OpenMMLab. All rights reserved.
"""Decode heads that pass ``gt_boundary_dist`` into :class:`BoundaryWeightedCrossEntropy`."""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList
from ..losses.accuracy import accuracy
from ..losses.boundary_weighted_ce import BoundaryWeightedCrossEntropy
from ..utils import resize
from .fcn_head import FCNHead
from .ocr_head import OCRHead


def stack_batch_boundary_dist(batch_data_samples: SampleList) -> Tensor:
    # gt_boundary_dist tensors per sample can differ in spatial size because
    # SegDataPreProcessor pads gt_sem_seg to crop_size but does not touch
    # gt_boundary_dist. Pad each tensor to the batch-max H/W before stacking.
    dists = [ds.gt_boundary_dist.data for ds in batch_data_samples]
    max_h = max(d.shape[-2] for d in dists)
    max_w = max(d.shape[-1] for d in dists)
    padded = []
    for d in dists:
        h, w = d.shape[-2], d.shape[-1]
        if h == max_h and w == max_w:
            padded.append(d)
        else:
            padded.append(F.pad(d, (0, max_w - w, 0, max_h - h), value=0))
    stacked = torch.stack(padded, dim=0)
    if stacked.dim() == 4 and stacked.size(1) == 1:
        stacked = stacked.squeeze(1)
    return stacked


def loss_by_feat_boundary_weighted(self, seg_logits: Tensor,
                                   batch_data_samples: SampleList) -> dict:
    """Shared ``loss_by_feat`` for Mod 2."""
    seg_label = self._stack_batch_gt(batch_data_samples)
    boundary_dist = stack_batch_boundary_dist(batch_data_samples)

    seg_logits = resize(
        input=seg_logits,
        size=seg_label.shape[2:],
        mode='bilinear',
        align_corners=self.align_corners)

    boundary_dist = F.interpolate(
        boundary_dist.unsqueeze(1).float(),
        size=seg_label.shape[2:],
        mode='bilinear',
        align_corners=self.align_corners,
    ).squeeze(1)

    seg_label_sq = seg_label.squeeze(1)
    losses = {}
    loss_modules = (self.loss_decode if isinstance(self.loss_decode, nn.ModuleList)
                    else [self.loss_decode])
    for loss_decode in loss_modules:
        if not isinstance(loss_decode, BoundaryWeightedCrossEntropy):
            raise TypeError(
                f'{type(self).__name__} (Mod 2) requires '
                f'BoundaryWeightedCrossEntropy, got {type(loss_decode)}')
        name = loss_decode.loss_name
        if name not in losses:
            losses[name] = loss_decode(
                seg_logits,
                seg_label_sq,
                weight=None,
                ignore_index=self.ignore_index,
                boundary_dist=boundary_dist,
            )
        else:
            losses[name] += loss_decode(
                seg_logits,
                seg_label_sq,
                weight=None,
                ignore_index=self.ignore_index,
                boundary_dist=boundary_dist,
            )

    losses['acc_seg'] = accuracy(
        seg_logits, seg_label_sq, ignore_index=self.ignore_index)
    return losses


@MODELS.register_module()
class WeightedCEFCNHead(FCNHead):
    """FCN head with boundary-weighted CE (Mod 2)."""

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        return loss_by_feat_boundary_weighted(self, seg_logits,
                                                batch_data_samples)


@MODELS.register_module()
class WeightedCEOCRHead(OCRHead):
    """OCR head with boundary-weighted CE (Mod 2)."""

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        return loss_by_feat_boundary_weighted(self, seg_logits,
                                                batch_data_samples)
