# Copyright (c) OpenMMLab. All rights reserved.
"""OCR head with temperature-scaled object attention (Mod 3)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.registry import MODELS
from ..utils import resize
from .ocr_head import OCRHead, ObjectAttentionBlock


class ModulatedObjectAttentionBlock(ObjectAttentionBlock):
    """Object attention: divide attention logits by per-query T before softmax."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg, temp_min: float = 0.5, temp_max: float = 5.0):
        super().__init__(in_channels, channels, scale, conv_cfg, norm_cfg,
                         act_cfg)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)

    def forward(
        self,
        query_feats: Tensor,
        key_feats: Tensor,
        temperature: Tensor = None,
    ) -> Tensor:
        """temperature: (B, 1, H, W), same resolution as ``query_feats``."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map

        if temperature is not None:
            T = temperature
            if self.query_downsample is not None:
                T = self.query_downsample(T)
            T = T.view(batch_size, -1, 1)
            T = T.clamp(min=self.temp_min, max=self.temp_max)
            sim_map = sim_map / T

        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)
        return output


@MODELS.register_module()
class OCRBoundaryHead(OCRHead):
    """OCR with learned temperature T(x) and auxiliary boundary loss on T branch.

    :math:`T = \\mathrm{clip}(1 + \\beta\\sigma(\\text{conv}(f)), T_{min}, T_{max})`.
    Division by T is applied to attention logits before softmax.

    Args:
        temp_beta (float): Default 2.0.
        temp_min (float): Default 0.5.
        temp_max (float): Default 5.0.
        boundary_aux_loss_weight (float): BCE weight vs ``gt_boundary``.
    """

    def __init__(
        self,
        ocr_channels: int,
        scale: int = 1,
        temp_beta: float = 2.0,
        temp_min: float = 0.5,
        temp_max: float = 5.0,
        boundary_aux_loss_weight: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(ocr_channels, scale, **kwargs)
        self.temp_beta = float(temp_beta)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.boundary_aux_loss_weight = float(boundary_aux_loss_weight)

        self.object_context_block = ModulatedObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            temp_min=self.temp_min,
            temp_max=self.temp_max,
        )
        ch = self.channels
        self.temp_convs = nn.Sequential(
            ConvModule(
                ch,
                ch,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                ch,
                ch,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
        )
        self.temp_logit_conv = nn.Conv2d(ch, 1, 1)
        self.aux_boundary_conv = nn.Conv2d(ch, 1, 1)

    def _temp_maps(self, feats: Tensor):
        z = self.temp_convs(feats)
        aux_bd_logits = self.aux_boundary_conv(z)
        t_logits = self.temp_logit_conv(z)
        t_map = (1.0 + self.temp_beta * torch.sigmoid(t_logits)).clamp(
            self.temp_min, self.temp_max)
        return t_map, aux_bd_logits

    def forward(self, inputs: Tensor, prev_output: Tensor) -> Tensor:
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        t_map, _ = self._temp_maps(feats)
        object_context = self.object_context_block(
            feats, context, temperature=t_map)
        return self.cls_seg(object_context)

    def loss(self, inputs: Tensor, prev_output: Tensor, batch_data_samples,
             train_cfg) -> dict:
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        t_map, aux_bd_logits = self._temp_maps(feats)
        object_context = self.object_context_block(
            feats, context, temperature=t_map)
        seg_logits = self.cls_seg(object_context)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        losses.update(self._aux_boundary_loss(aux_bd_logits,
                                              batch_data_samples))
        return losses

    def _aux_boundary_loss(self, aux_logits: Tensor,
                             batch_data_samples) -> dict:
        if not batch_data_samples:
            return {}
        first = batch_data_samples[0]
        if not hasattr(first, 'gt_boundary') or first.gt_boundary is None:
            return {}
        targets = torch.stack(
            [s.gt_boundary.data for s in batch_data_samples], dim=0)
        logits = resize(
            aux_logits,
            size=targets.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='mean')
        return {'loss_boundary_aux': self.boundary_aux_loss_weight * loss}
