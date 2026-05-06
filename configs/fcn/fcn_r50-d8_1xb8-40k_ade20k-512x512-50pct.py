_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k_2kckpt.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)

# FCN heads with 150 classes for ADE20K
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150))

# lr=0.004: linear scaling from 0.002 at 20% data to 0.004 at 50% data
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005))

# 50% of ADE20K training set (20210 images total -> 10105)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(indices=10105))
