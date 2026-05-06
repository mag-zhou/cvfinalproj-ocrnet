_base_ = ['./fcn_r50-d8_1xb8-40k_ade20k-512x512-50pct.py']

# Load weights from end of first 40k run; fresh optimizer + LR schedule
load_from = 'work_dirs/fcn_r50_ade20k_50pct/iter_40000.pth'

# Reduced LR: model is already well-trained, avoid disrupting learned weights
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005))
