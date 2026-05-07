# Mod 3 (temperature-modulated OCR + aux boundary), 50% ADE20K --
# CONTINUATION run: load weights from iter_40000.pth of the original 50pct run
# and train another 40k iters with a fresh PolyLR over [0, 40000).
#
# Why fine-tune-style restart instead of `--resume`?
#   The original 50pct config follows the standard 40k poly schedule, so at
#   iter 40000 the LR has already decayed to ``eta_min=1e-4``. Resuming with
#   the same scheduler would either (a) keep LR at 1e-4 for the next 40k iters
#   (nothing happens) or (b) jump LR back up unpredictably if we extend
#   max_iters. A fresh PolyLR over [0, 40000) starting from iter_40000.pth is
#   simpler and predictable: it's a warm restart at the original starting LR
#   (0.004), decaying back to 1e-4 over another 40k iters.
#
# To run:
#   sbatch train_slurm_mod3_50pct_ext.sh
#
# Final checkpoint at iter_40000 of THIS run = 80k iters of training in total.

_base_ = ['./ocrnet_r50_mod3_modulated_ocr_50pct.py']

# Initialize from the end of the original 50pct run. ``load_from`` only
# restores model weights -- optimizer and scheduler state start fresh.
load_from = '/orcd/scratch/orcd/003/janetguo/cvfinalproj-ocrnet/work_dirs/ocrnet_r50_mod3_modulated_ocr_50pct/iter_40000.pth'

# Fresh 40k schedule, identical to the original (so it's directly comparable).
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
