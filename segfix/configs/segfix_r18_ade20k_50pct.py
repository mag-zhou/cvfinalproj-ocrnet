# SegFix offset model: ResNet-18 backbone, ADE20K 50% subset, 20k iters.
#
# This is the canonical SegFix training config -- pairs with a 50%-trained
# OCRNet baseline for refinement. See `segfix/INSTRUCTIONS.md` for the
# end-to-end runbook.
#
# Custom imports load the segfix package so SegFixOffsetModel,
# ComputeOffsetsFromSeg, PackSegFixInputs, and SegFixOffsetMetric are all
# discoverable in the registries.

_base_ = [
    '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py',
]
custom_imports = dict(imports=['segfix'], allow_failed_imports=False)

crop_size = (512, 512)

# ---- Pipelines ------------------------------------------------------------
# We compute boundary + offset GT ONLINE from the augmented gt_seg_map (see
# segfix/transforms.py for the rationale). No precomputed .npy files needed.

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ComputeOffsetsFromSeg', kernel_size=17, ignore_index=255),
    dict(type='PackSegFixInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='ComputeOffsetsFromSeg', kernel_size=17, ignore_index=255),
    dict(type='PackSegFixInputs'),
]

# ---- Data preprocessor ----------------------------------------------------
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

# ---- Model ----------------------------------------------------------------
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='SegFixOffsetModel',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
    ),
    in_channels=512,        # ResNet-18 stage-4 output
    in_index=3,
    decoder_channels=256,
    num_decoder_convs=2,
    max_offset=8.0,         # matches kernel_size=17 (radius 8)
    boundary_pos_weight=8.0,
    loss_offset_weight=1.0,
    align_corners=False,
)

# ---- Dataloaders (50% subset = 10105 / 20210 ADE20K training images) ------
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        indices=10105,
        pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline),
)
test_dataloader = val_dataloader

val_evaluator = [dict(type='SegFixOffsetMetric')]
test_evaluator = val_evaluator

# ---- Optimizer / schedule -------------------------------------------------
# 2x LR bump from the 20pct precedent (0.01 -> 0.02), matching how the
# mod1/mod2/mod3 50pct configs scaled their LRs.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005),
    clip_grad=None,
)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False),
]

# ---- Loops & hooks --------------------------------------------------------
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
