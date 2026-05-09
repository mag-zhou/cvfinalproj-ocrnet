_base_ = './ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct-ext.py'

# Add boundary F-score alongside the standard IoU metric.
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryFScore', tolerances=[3, 5, 9]),
]
test_evaluator = val_evaluator
