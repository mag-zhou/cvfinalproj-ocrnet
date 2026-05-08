# Mod 4 (CBL-lite on OCR features), 50% ADE20K, fresh 80k run.
#
# Single, methodologically clean training run from iter 0 -> 80000 with
# one PolyLR schedule covering the full horizon (no warm restart).
# Mirrors the 50pct_80k setup for mod2/mod3 so all four are directly
# comparable in our ablation table.
#
# To run:
#   sbatch train_slurm_mod4_50pct_80k.sh
#
# Expected wall-clock: ~4-5h on 1x H200 (CBL adds ~10-30% per-iter
# overhead vs Mod 2/3; budget conservatively).

_base_ = ['./ocrnet_r50_mod4_cbl_50pct.py']

# Single PolyLR over the full 80k horizon (replaces inherited 40k schedule).
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
