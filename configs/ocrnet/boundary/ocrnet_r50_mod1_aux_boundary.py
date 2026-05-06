# Mod 1: auxiliary boundary head (see mmseg/models/decode_heads/boundary_head.py)
_base_ = [
    '../ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py',
]

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadBoundaryAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegBoundaryInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadBoundaryAnnotations'),
    dict(type='PackSegBoundaryInputs'),
]

# Merges into CascadeEncoderDecoder from base config
model = dict(
    auxiliary_head=dict(
        type='BoundaryHead',
        in_channels=1024,
        in_index=2,
        channels=128,
        num_convs=2,
        num_classes=2,
        out_channels=1,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.4,
            class_weight=[8.0],
            loss_name='loss_ce',
        ),
    ))

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        indices=4042,
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

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryFScore', tolerances=[3, 5, 9]),
]
test_evaluator = val_evaluator
