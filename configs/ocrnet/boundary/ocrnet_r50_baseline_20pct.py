# Baseline OCRNet (20% ADE20K) for comparison; only adds boundary F-score on val.
_base_ = ['../ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py']

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='BoundaryFScore', tolerances=[3, 5, 9]),
]
test_evaluator = val_evaluator
