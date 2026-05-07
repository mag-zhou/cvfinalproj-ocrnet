"""SegFix offset+boundary model.

Architecture (per the implementation plan):

    backbone (ResNet-18 v1c, stride 8, returns multi-stage features)
        -> shared decoder (1x1 reduce + ConvModule x num_decoder_convs)
        -> two parallel 1x1 heads:
              - boundary head: 1 channel, sigmoid via BCE-with-logits.
              - offset head:   2 channels (dy, dx), bounded by tanh*max_offset.

Loss:

    L = L_boundary + lambda * L_offset

    L_boundary: BCE on per-pixel boundary logits, with positive-class weight
                pos_weight (default 8.0) since boundaries are sparse.
    L_offset:   smooth L1 on (dy, dx), MASKED to GT-boundary pixels only.
                Interior offsets are noisy and irrelevant at refinement
                time, so we skip them entirely.

This is a separate model from the OCRNet baseline -- it is not a decoder for
mmseg's segmentation evaluator. We subclass ``BaseSegmentor`` so it plugs
into mmengine's ``Runner`` (``model.forward(mode=...)``), but ``predict()``
returns offset/boundary tensors instead of a ``pred_sem_seg`` -- there is no
class output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)


@MODELS.register_module()
class SegFixOffsetModel(BaseSegmentor):
    """Standalone backbone + boundary head + offset head.

    Args:
        backbone:           mmseg backbone config. Must return a tuple/list of
                            multi-stage features; only ``in_index`` is used.
        in_channels:        Number of channels of the chosen feature map.
        in_index:           Index into the backbone output tuple. For ResNet-18
                            with out_indices=(0,1,2,3), index 3 is OS=8 with
                            (1,1,2,4) dilations.
        decoder_channels:   Hidden channels for the shared decoder.
        num_decoder_convs:  Number of 3x3 ConvModules in the shared decoder
                            after the 1x1 channel-reduce.
        max_offset:         Pixel cap on predicted offset magnitude (per axis,
                            via ``tanh``). Should match ``kernel_size//2`` of
                            the GT generator (default 8 for kernel_size=17).
        boundary_pos_weight: BCE pos_weight for boundary head. Plan suggests
                            5--10; we default 8.0.
        loss_offset_weight: lambda multiplier on the offset loss. Plan
                            suggests 1.0 to start.
        align_corners:      For the bilinear upsample to input resolution.
        data_preprocessor:  Standard mmseg ``SegDataPreProcessor`` config.
        init_cfg:           Optional weight init config.
        pretrained:         Optional override path/URL of backbone pretrained
                            weights (passed through to the backbone config).
        train_cfg, test_cfg: Carried for API compatibility; unused.
    """

    def __init__(
        self,
        backbone: ConfigType,
        in_channels: int = 512,
        in_index: int = 3,
        decoder_channels: int = 256,
        num_decoder_convs: int = 2,
        max_offset: float = 8.0,
        boundary_pos_weight: float = 8.0,
        loss_offset_weight: float = 1.0,
        align_corners: bool = False,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        pretrained: Optional[str] = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            backbone = dict(backbone)
            backbone['init_cfg'] = dict(
                type='Pretrained', checkpoint=pretrained)
        self.backbone = MODELS.build(backbone)
        self.in_index = in_index
        self.in_channels = in_channels
        self.decoder_channels = decoder_channels
        self.max_offset = float(max_offset)
        self.boundary_pos_weight = float(boundary_pos_weight)
        self.loss_offset_weight = float(loss_offset_weight)
        self.align_corners = align_corners
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ---- shared decoder: 1x1 reduce, then a stack of 3x3 ConvModules.
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        act_cfg = dict(type='ReLU')
        self.reduce = ConvModule(
            in_channels,
            decoder_channels,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        decoder = []
        for _ in range(num_decoder_convs):
            decoder.append(
                ConvModule(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.decoder = nn.Sequential(*decoder)

        # ---- prediction heads.
        self.boundary_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(decoder_channels, 2, kernel_size=1)

    # ------------------------------------------------------------------ utils
    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor, ...]:
        feats = self.backbone(inputs)
        if not isinstance(feats, (list, tuple)):
            feats = (feats, )
        return feats

    def _forward_shared(self, inputs: Tensor) -> Tensor:
        feats = self.extract_feat(inputs)
        x = feats[self.in_index]
        x = self.reduce(x)
        x = self.decoder(x)
        return x

    def _forward_heads(self, inputs: Tensor,
                       out_size: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        x = self._forward_shared(inputs)
        b_logits = self.boundary_head(x)
        o_raw = self.offset_head(x)
        # Bound predicted offsets to [-max_offset, max_offset] per axis. This
        # matches the GT range (kernel_size//2) and prevents pathological early
        # training from emitting absurd offsets that break the smoothL1 grad.
        o_pred = torch.tanh(o_raw) * self.max_offset
        b_logits = resize(
            b_logits,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        o_pred = resize(
            o_pred,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        return b_logits, o_pred

    # ------------------------------------------------------ batch GT helpers
    @staticmethod
    def _stack_padded(tensors: List[Tensor], pad_value: float = 0.0) -> Tensor:
        """Pad per-sample tensors to the batch-max H/W and stack."""
        max_h = max(t.shape[-2] for t in tensors)
        max_w = max(t.shape[-1] for t in tensors)
        padded = []
        for t in tensors:
            h, w = t.shape[-2], t.shape[-1]
            if h == max_h and w == max_w:
                padded.append(t)
            else:
                padded.append(
                    F.pad(t, (0, max_w - w, 0, max_h - h), value=pad_value))
        return torch.stack(padded, dim=0)

    def _stack_batch_offset(
            self,
            data_samples: SampleList) -> Tuple[Tensor, Tensor]:
        offs = []
        bds = []
        for s in data_samples:
            assert hasattr(s, 'gt_offset') and s.gt_offset is not None, (
                'SegFixOffsetModel requires gt_offset on each SegDataSample. '
                'Use ComputeOffsetsFromSeg + PackSegFixInputs.')
            assert (hasattr(s, 'gt_offset_boundary')
                    and s.gt_offset_boundary is not None), (
                'SegFixOffsetModel requires gt_offset_boundary on each '
                'SegDataSample.')
            offs.append(s.gt_offset.data.float())
            bds.append(s.gt_offset_boundary.data.float())
        offset = self._stack_padded(offs, pad_value=0.0)
        boundary = self._stack_padded(bds, pad_value=0.0)
        return offset, boundary

    # ------------------------------------------------------------------- API
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        gt_offset, gt_boundary = self._stack_batch_offset(data_samples)
        h, w = gt_offset.shape[-2], gt_offset.shape[-1]
        b_logits, o_pred = self._forward_heads(inputs, out_size=(h, w))

        gt_b = gt_boundary  # (B, 1, H, W) in {0, 1}
        # ---- boundary BCE with pos_weight.
        pw = b_logits.new_tensor(self.boundary_pos_weight)
        loss_b = F.binary_cross_entropy_with_logits(
            b_logits, gt_b, pos_weight=pw, reduction='mean')

        # ---- offset smooth L1, masked to boundary pixels in the GT.
        mask = gt_b > 0.5  # (B, 1, H, W)
        mask_2c = mask.expand_as(o_pred)
        diff = F.smooth_l1_loss(o_pred, gt_offset, reduction='none')
        denom = mask_2c.float().sum().clamp(min=1.0)
        loss_o = (diff * mask_2c.float()).sum() / denom

        with torch.no_grad():
            # Sanity stats logged alongside losses.
            pred_b = (torch.sigmoid(b_logits) > 0.5).float()
            tgt_b = gt_b
            # boundary pixel-level accuracy
            acc_b = (pred_b == tgt_b).float().mean() * 100.0
            # endpoint error on boundary pixels (for diagnostics)
            with torch.no_grad():
                ee = torch.sqrt(
                    (o_pred - gt_offset).pow(2).sum(dim=1, keepdim=True))
                ee_masked = (ee * mask.float()).sum() / mask.float().sum() \
                    .clamp(min=1.0)

        return {
            'loss_boundary': loss_b,
            'loss_offset': self.loss_offset_weight * loss_o,
            'acc_boundary': acc_b,
            'endpoint_error_px': ee_masked,
        }

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Return a SegDataSample list whose ``pred_sem_seg`` is the binary
        boundary map (uint8) and whose ``seg_logits`` carries:

            channel 0   -> boundary probability
            channel 1   -> predicted dy
            channel 2   -> predicted dx

        Output is at ``ori_shape`` (un-padded, un-flipped) so it can feed the
        refinement script directly.
        """
        b, _, h_in, w_in = inputs.shape
        b_logits, o_pred = self._forward_heads(inputs, out_size=(h_in, w_in))
        b_prob = torch.sigmoid(b_logits)
        full = torch.cat([b_prob, o_pred], dim=1)  # (B, 3, H, W)

        if data_samples is None:
            # No metadata: just return raw outputs as-is.
            data_samples = []
            for i in range(b):
                ds = self._make_pred_sample(full[i:i + 1])
                data_samples.append(ds)
            return data_samples

        out = []
        for i, sample in enumerate(data_samples):
            meta = sample.metainfo
            padding = meta.get('img_padding_size', meta.get('padding_size',
                                                            [0] * 4))
            pl, pr, pt, pb = padding
            x = full[i:i + 1, :, pt:h_in - pb, pl:w_in - pr]

            flip = meta.get('flip', False)
            if flip:
                direction = meta.get('flip_direction', 'horizontal')
                if direction == 'horizontal':
                    x = x.flip(dims=(3, ))
                    # un-do horizontal flip on the dx channel: dx -> -dx
                    x = x.clone()
                    x[:, 2:3] = -x[:, 2:3]
                elif direction == 'vertical':
                    x = x.flip(dims=(2, ))
                    x = x.clone()
                    x[:, 1:2] = -x[:, 1:2]

            ori_h, ori_w = meta.get('ori_shape', x.shape[-2:])
            cur_h, cur_w = x.shape[-2], x.shape[-1]
            scale_y = ori_h / max(cur_h, 1)
            scale_x = ori_w / max(cur_w, 1)
            x = resize(
                x,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
            # Resizing the offset *array* spatially is correct; but the offset
            # *values* describe a pixel displacement, so they must scale by the
            # same factor used to undo the resize-during-test (i.e. scale up to
            # original-image pixel units).
            x = x.clone()
            x[:, 1:2] *= scale_y
            x[:, 2:3] *= scale_x

            out.append(self._make_pred_sample(x, sample))
        return out

    def _make_pred_sample(self,
                          chw: Tensor,
                          sample=None):
        from mmseg.structures import SegDataSample
        ds = SegDataSample() if sample is None else sample
        # boundary map argmax -> (1, H, W) uint-ish
        b_prob = chw[0, 0:1]
        pred = (b_prob > 0.5).long()
        ds.set_data({
            'seg_logits': PixelData(data=chw[0]),
            'pred_sem_seg': PixelData(data=pred),
        })
        return ds

    def encode_decode(self, inputs: Tensor, batch_data_samples: SampleList):
        # Required by BaseSegmentor abstract API. Not used by training/predict.
        return self._forward_heads(inputs,
                                   out_size=(inputs.shape[-2], inputs.shape[-1]))[0]

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[Tensor, Tensor]:
        return self._forward_heads(
            inputs, out_size=(inputs.shape[-2], inputs.shape[-1]))
