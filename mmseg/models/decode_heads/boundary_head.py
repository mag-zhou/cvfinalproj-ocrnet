# Copyright (c) OpenMMLab. All rights reserved.
"""Auxiliary head for binary boundary prediction (Mod 1)."""
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from ..utils import resize
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class BoundaryHead(BaseDecodeHead):
    """Light conv stack + 1x1 logits for boundary supervision.

    Ground truth is read from ``data_sample.gt_boundary`` (packed by
    :class:`PackSegBoundaryInputs`), shape ``(1, H, W)`` with values in
    ``{0, 1}``. Use ``CrossEntropyLoss`` with ``use_sigmoid=True`` (BCE on
    logits). Positive-class imbalance is handled via ``class_weight`` in the
    loss config (passed through as ``pos_weight`` in BCE).

    Args:
        num_convs (int): Convs before final 1x1 ``conv_seg``. Default: 2.
        kernel_size (int): Kernel size for intermediate convs. Default: 3.
    """

    def __init__(
        self,
        num_convs: int = 2,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        kwargs.setdefault('ignore_index', None)
        super().__init__(**kwargs)
        assert num_convs >= 1
        self.num_convs = num_convs
        self.kernel_size = kernel_size

        conv_padding = kernel_size // 2
        layers = []
        layers.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for _ in range(num_convs - 1):
            layers.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward."""
        x = self._transform_inputs(inputs)
        x = self.convs(x)
        return self.cls_seg(x)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gts: List[Tensor] = []
        for sample in batch_data_samples:
            assert hasattr(sample, 'gt_boundary') and sample.gt_boundary is not None, (
                'BoundaryHead requires `gt_boundary` on each '
                '`SegDataSample` (use LoadBoundaryAnnotations + '
                'PackSegBoundaryInputs).')
            gts.append(sample.gt_boundary.data)
        return torch.stack(gts, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """BCE-style loss; skip multiclass ``accuracy`` from base class."""
        seg_label = self._stack_batch_gt(batch_data_samples)
        losses = {}
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_weight = None
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label.squeeze(1))

        seg_label_sq = seg_label.squeeze(1)

        loss_modules = (self.loss_decode if isinstance(self.loss_decode, nn.ModuleList)
                        else [self.loss_decode])
        for loss_decode in loss_modules:
            name = loss_decode.loss_name
            if name not in losses:
                losses[name] = loss_decode(
                    seg_logits,
                    seg_label_sq,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                losses[name] += loss_decode(
                    seg_logits,
                    seg_label_sq,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        with torch.no_grad():
            prob = torch.sigmoid(seg_logits.squeeze(1))
            pred = prob > 0.5
            tgt = seg_label_sq > 0.5
            losses['acc_seg'] = (pred == tgt).float().mean() * 100.0
        return losses

    def predict(self, inputs: Tensor, batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Resize logits to input image size (auxiliary; optional)."""
        seg_logits = self.forward(inputs)
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        return resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
