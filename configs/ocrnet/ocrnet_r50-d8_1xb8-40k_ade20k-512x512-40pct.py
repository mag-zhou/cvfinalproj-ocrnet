_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_2kckpt.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)

# OCRNet heads with 150 classes for ADE20K
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ])

# lr scaled between 20% (0.002) and 50% (0.004)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0032, momentum=0.9, weight_decay=0.0005))

# 40% of ADE20K training set (20210 images total -> 8084)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(indices=8084))
