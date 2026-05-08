# Mod 3 (temperature-modulated OCR + aux boundary), 50% ADE20K, fresh 80k run.
#
# Differs from the 50pct + 50pct_ext approach in that this is a single,
# methodologically clean training run from iter 0 -> 80000 with one PolyLR
# schedule covering the full horizon (no warm restart at iter 40k).
#
# To run:
#   sbatch train_slurm_mod3_50pct_80k.sh
#
# Expected wall-clock: ~3h 30m on 1x GPU @ ~0.155 s/iter (matches prior 50pct
# timing), plus ~20 validation passes at val_interval=4000.

_base_ = ['./ocrnet_r50_mod3_modulated_ocr_50pct.py']

# Single PolyLR over the full 80k horizon (replaces the inherited 40k schedule).
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
