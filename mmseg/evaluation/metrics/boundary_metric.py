# Copyright (c) OpenMMLab. All rights reserved.
"""Boundary F-score with tolerance (dilated matching)."""
from collections import OrderedDict
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmseg.registry import METRICS


def _sem_seg_to_boundary(sem: torch.Tensor,
                         ignore_index: int) -> torch.Tensor:
    """8-neighbor class disagreement -> boundary mask (H, W) bool."""
    while sem.dim() > 2:
        sem = sem.squeeze(0)
    x = sem.unsqueeze(0).unsqueeze(0).float()
    pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    center = pad[:, :, 1:-1, 1:-1]
    nbs = torch.cat([
        pad[:, :, :-2, 1:-1],
        pad[:, :, 2:, 1:-1],
        pad[:, :, 1:-1, :-2],
        pad[:, :, 1:-1, 2:],
        pad[:, :, :-2, :-2],
        pad[:, :, :-2, 2:],
        pad[:, :, 2:, :-2],
        pad[:, :, 2:, 2:],
    ], dim=1)
    diff = (nbs != center).any(dim=1).squeeze(0)
    ign = (sem == ignore_index)
    return diff & (~ign)


def _dilate_binary(mask_hw: torch.Tensor, radius: int) -> torch.Tensor:
    """Max-pool dilation; mask_hw (H,W) float in {0,1}."""
    if radius <= 0:
        return mask_hw
    k = 2 * radius + 1
    x = mask_hw.unsqueeze(0).unsqueeze(0).float()
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return y.squeeze(0).squeeze(0) > 0.5


def _boundary_f1_at_tol(
        pred_b: torch.Tensor,
        gt_b: torch.Tensor,
        tol: int,
) -> Tuple[float, float, float]:
    """Pred / GT boundary bool (H,W). Returns (precision, recall, f1)."""
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0, 1.0, 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0, 0.0, 0.0

    gt_d = _dilate_binary(gt_b, tol)
    pr_d = _dilate_binary(pred_b, tol)

    tp_p = (pred_b & gt_d).sum().float()
    prec = (tp_p / pred_b.sum().float()).item()

    tp_r = (gt_b & pr_d).sum().float()
    rec = (tp_r / gt_b.sum().float()).item()

    if prec + rec < 1e-8:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


@METRICS.register_module()
class BoundaryFScore(BaseMetric):
    """Boundary F-score with symmetric tolerance (dilated matching).

    For tolerance ``t``, precision counts predicted boundary pixels within
    ``t`` pixels of any GT boundary; recall counts GT boundary pixels within
    ``t`` of any prediction.

    Args:
        tolerances (list[int]): Radii in pixels, default ``[3, 5, 9]``.
        ignore_index (int): Ignored in semantic maps when extracting edges.
    """

    def __init__(
            self,
            tolerances: List[int] = None,
            ignore_index: int = 255,
            collect_device: str = 'cpu',
            prefix: str = None,
            **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.tolerances = tolerances or [3, 5, 9]
        self.ignore_index = ignore_index

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = sample['pred_sem_seg']['data']
            label = sample['gt_sem_seg']['data']
            while pred.dim() > 2:
                pred = pred.squeeze(0)
            while label.dim() > 2:
                label = label.squeeze(0)
            pred = pred.long()
            label = label.long()
            if pred.shape != label.shape:
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0).float(),
                    size=label.shape,
                    mode='nearest').squeeze(0).squeeze(0).long()

            pred_b = _sem_seg_to_boundary(pred, self.ignore_index)
            gt_b = _sem_seg_to_boundary(label, self.ignore_index)

            row = {}
            for t in self.tolerances:
                _, _, f1 = _boundary_f1_at_tol(pred_b, gt_b, t)
                row[t] = f1
            self.results.append(row)

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        out = OrderedDict()
        for t in self.tolerances:
            vals = [r[t] for r in results]
            mean_f1 = float(np.mean(vals)) if vals else 0.0
            key = f'mBoundaryF1_{t}'
            out[key] = round(mean_f1 * 100.0, 2)
        print_log(
            'Boundary F-score (mean over images): ' + str(dict(out)),
            logger=logger)
        return out
