"""Cheap validation metric for the SegFix offset model.

Reports:
    - boundary F1 (predicted boundary vs GT boundary, no tolerance)
    - boundary precision / recall
    - mean offset endpoint error in pixels, masked to GT boundary

This is purely diagnostic: the *real* eval is the refined-prediction mIoU
and boundary-F-score-with-tolerance from ``segfix/eval_refined.py``. Use
this only to spot-check that training is making progress.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmseg.registry import METRICS


@METRICS.register_module()
class SegFixOffsetMetric(BaseMetric):

    def __init__(
        self,
        boundary_threshold: float = 0.5,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.boundary_threshold = boundary_threshold

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            seg_logits = sample['seg_logits']['data']
            # seg_logits packs [b_prob, dy, dx] in channel 0..2 (see
            # SegFixOffsetModel.predict).
            if seg_logits.dim() == 4:
                seg_logits = seg_logits[0]
            b_prob = seg_logits[0]
            o_pred = seg_logits[1:3]

            gt_b = sample.get('gt_offset_boundary', None)
            gt_o = sample.get('gt_offset', None)
            if gt_b is None or gt_o is None:
                # Validation pipeline didn't produce GT offsets/boundaries.
                # Bail without crashing -- this metric is sanity-only.
                continue
            gt_b_t = gt_b['data'].squeeze().to(b_prob.device).float()
            gt_o_t = gt_o['data'].to(o_pred.device).float()

            # Resize prediction to match GT (test pipeline may differ).
            if b_prob.shape != gt_b_t.shape:
                b_prob = torch.nn.functional.interpolate(
                    b_prob.unsqueeze(0).unsqueeze(0),
                    size=gt_b_t.shape,
                    mode='bilinear',
                    align_corners=False).squeeze()
                o_pred = torch.nn.functional.interpolate(
                    o_pred.unsqueeze(0),
                    size=gt_b_t.shape,
                    mode='bilinear',
                    align_corners=False).squeeze(0)

            pred_b = (b_prob > self.boundary_threshold)
            gt_b_bin = (gt_b_t > 0.5)

            tp = (pred_b & gt_b_bin).sum().item()
            fp = (pred_b & (~gt_b_bin)).sum().item()
            fn = ((~pred_b) & gt_b_bin).sum().item()

            if gt_b_bin.sum() > 0:
                ee = torch.sqrt(
                    (o_pred - gt_o_t).pow(2).sum(dim=0))  # (H, W)
                ee_masked = (ee * gt_b_bin.float()).sum().item() \
                    / float(gt_b_bin.sum().item())
            else:
                ee_masked = 0.0

            self.results.append({
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'ee': ee_masked,
                'has_boundary': bool(gt_b_bin.sum() > 0),
            })

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        out: 'OrderedDict[str, float]' = OrderedDict()
        if not results:
            print_log('SegFixOffsetMetric: empty results.', logger=logger)
            return dict(out)
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        ees = [r['ee'] for r in results if r['has_boundary']]
        mee = float(np.mean(ees)) if ees else 0.0
        out['boundary_precision'] = round(prec * 100.0, 2)
        out['boundary_recall'] = round(rec * 100.0, 2)
        out['boundary_f1'] = round(f1 * 100.0, 2)
        out['offset_endpoint_err_px'] = round(mee, 3)
        print_log('SegFix sanity metrics: ' + str(dict(out)), logger=logger)
        return dict(out)
