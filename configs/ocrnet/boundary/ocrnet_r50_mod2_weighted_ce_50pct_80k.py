# Mod 2 (boundary-weighted CE on both cascade heads), 50% ADE20K, fresh 80k run.
#
# Single, methodologically clean training run from iter 0 -> 80000 with one
# PolyLR schedule covering the full horizon (no warm restart). Mirrors the
# 50pct_80k setup for mod3 so they are directly comparable.
#
# To run:
#   sbatch train_slurm_mod2_50pct_80k.sh
#
# Expected wall-clock: ~3h 30m on 1x GPU @ ~0.16 s/iter (matches prior 50pct
# mod2 timing), plus ~20 validation passes at val_interval=4000.

_base_ = ['./ocrnet_r50_mod2_weighted_ce_50pct.py']

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
