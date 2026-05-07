# SegFix on OCRNet — runbook for the 50% ADE20K experiment

This is a step-by-step for someone with a **trained 50% ADE20K OCRNet
checkpoint on a GPU machine** to (1) train a small SegFix offset model on
the same 50% subset, (2) refine the OCRNet baseline's validation
predictions with SegFix, and (3) compute mIoU + boundary F1 on both the
refined and un-refined predictions for an apples-to-apples comparison.

Total wall-clock on a single H200: ~2h train + ~10min refine + ~5min eval.
On smaller cards (A100 / 4090) expect 2--3x train. SegFix uses a
ResNet-18 backbone -- a single 16+ GB GPU is plenty.

---

## 0. Prerequisites

You need:

1. **The repo** — clone and check out the SegFix branch:
   ```bash
   git clone git@github.com:mag-zhou/cvfinalproj-ocrnet.git
   cd cvfinalproj-ocrnet
   git checkout boundary-experiments
   git pull
   ```
2. **Python env with mmseg already installed** (you already have this if
   you've trained OCRNet here). Confirm:
   ```bash
   python -c "import mmengine, mmcv, mmseg; print(mmengine.__version__, mmcv.__version__, mmseg.__version__)"
   ```
3. **ADE20K dataset** at `data/ade/ADEChallengeData2016/` with
   `images/{training,validation}/` and `annotations/{training,validation}/`
   subfolders.
4. **A trained 50% OCRNet baseline checkpoint**:
   - config: `configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct.py`
   - checkpoint .pth file (you have this — let's call it `$SEG_CHECKPOINT`).

`segfix/` lives at the repo root; everything SegFix-specific is contained
there.

---

## 1. One-time preflight

From the repo root:

```bash
# Make `segfix` importable (registers SegFixOffsetModel etc. with mmseg).
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Should print "ok".
python -c "import segfix; print('ok')"
```

(Optional — recommended) Visualize the offset GT on 10 training images to
confirm the algorithm is doing what it should before spending 2h on
training:

```bash
python segfix/gen_offset_gt.py --visualize --max-images 10 --workers 4
```

This writes `data/ade/ADEChallengeData2016/offset_viz/training/<id>_mag.png`
(magnitude) and `<id>_dir.png` (HSV-coded direction) for 10 images. Pass
criteria:
- magnitude PNGs: bright near class boundaries, dark in interiors.
- direction PNGs: coherent color near boundaries, pointing inward.

If those look wrong, **stop and ping me** — there's a bug in the offset
algorithm that's worth fixing before training.

---

## 2. Train the SegFix offset model (~2h on H200)

```bash
python tools/train.py segfix/configs/segfix_r18_ade20k_50pct.py \
    --work-dir work_dirs/segfix_r18_ade20k_50pct
```

Expected:
- ~20,000 iters total, val every 4,000.
- Two losses logged: `loss_boundary` (BCE), `loss_offset` (smooth-L1
  masked to GT boundary). Both should drop monotonically; `loss_boundary`
  starts ~0.8, `loss_offset` starts ~5.
- Two diagnostic non-loss numbers: `acc_boundary` (% pixel-wise boundary
  accuracy, expect 80%+ by mid-training) and `endpoint_error_px` (mean
  L2 between predicted and GT offsets at boundary pixels, in pixels —
  starts ~8, expect ~3 by end).
- Final checkpoint:
  `work_dirs/segfix_r18_ade20k_50pct/iter_20000.pth`

If you have SLURM, there's a portable wrapper at
`segfix/slurm/train_slurm_segfix_50pct.sh` (just `sbatch` it; it
auto-resolves the repo root and uses `python` from `$PATH`).

If a job is preempted, just rerun — the script auto-resumes from the
latest checkpoint when `last_checkpoint` exists in the work dir.

### Smoke-test before the full run (optional)

Quick 2-iter dry run to confirm the model builds and a forward pass works
on this machine:

```bash
python tools/train.py segfix/configs/segfix_r18_ade20k_50pct.py \
    --cfg-options train_dataloader.dataset.indices=4 \
                  train_dataloader.batch_size=2 \
                  train_cfg.max_iters=2 \
                  train_cfg.val_interval=10 \
                  default_hooks.checkpoint.interval=10 \
    --work-dir /tmp/segfix_smoke
```

Should complete in ~1min once pretrained ResNet-18 weights are
downloaded. Look for a line like
`Iter(train) [2/2]  loss: 6.x  loss_boundary: 0.x  loss_offset: 5.x  acc_boundary: ~60-70%`.
Delete `/tmp/segfix_smoke` after.

---

## 3. Refine the OCRNet baseline's validation predictions

This is the main "deliverable": run baseline OCRNet + SegFix offsets
across the full ADE20K validation set, save **two** sets of predictions
to disk (refined and un-refined) so the eval is identical on both.

```bash
python segfix/refine.py \
    --seg-config        configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct.py \
    --seg-checkpoint    /path/to/your/baseline_50pct/iter_40000.pth \
    --offset-config     segfix/configs/segfix_r18_ade20k_50pct.py \
    --offset-checkpoint work_dirs/segfix_r18_ade20k_50pct/iter_20000.pth \
    --output            work_dirs/segfix_refined_baseline_50pct \
    --boundary-thresh   0.5
```

**Outputs:**

```
work_dirs/segfix_refined_baseline_50pct/
├── predictions/                   <- refined PNGs (SegFix-applied), 1 per val image
│   ├── ADE_val_00000001.png
│   ├── ADE_val_00000002.png
│   └── ...
└── predictions_baseline/          <- un-refined PNGs from the same OCRNet run
    └── ...
```

PNG format matches ADE20K's GT convention: uint8, `0` = ignore,
classes are `1..150`. ~2000 images, takes ~10min on H200.

If your baseline config is one of the boundary variants instead of the
plain 50pct config (e.g. `configs/ocrnet/boundary/ocrnet_r50_baseline_*.py`),
just point `--seg-config` at that file. SegFix doesn't care which
OCRNet flavor you're refining — it just needs the seg config / checkpoint
pair to be a working `tools/test.py`-compatible model.

### Hyperparameter you might want to ablate (free, no retraining)

The boundary threshold `--boundary-thresh` controls which pixels get
refined (predicted boundary probability > thresh). 0.5 is the default;
0.3 (more aggressive refinement, more pixels touched) and 0.7 (more
conservative) are worth trying. Each rerun of `refine.py` is ~10min and
produces a different `predictions/` folder — point `--output` at a
different directory for each. Then evaluate all three.

---

## 4. Evaluate both refined and un-refined predictions (~5 min)

```bash
# Refined predictions:
python segfix/eval_refined.py \
    --pred-dir work_dirs/segfix_refined_baseline_50pct/predictions \
    --gt-dir   data/ade/ADEChallengeData2016/annotations/validation

# Un-refined baseline (apples-to-apples, same eval code):
python segfix/eval_refined.py \
    --pred-dir work_dirs/segfix_refined_baseline_50pct/predictions_baseline \
    --gt-dir   data/ade/ADEChallengeData2016/annotations/validation
```

Each prints a small table:

```
============================================================
Evaluated             : 2000 images
mIoU                  : 33.42
pixel accuracy        : 73.18
mBoundaryF1_3         : 26.91
mBoundaryF1_5         : 35.02
mBoundaryF1_9         : 47.13
============================================================
```

Boundary F1 is the metric SegFix is designed to move; mIoU is the second-
order effect. Expected on ADE20K 20% (per the implementation plan):
roughly +0.5 to +1.5 mBoundaryF1, smaller mIoU bump (could be within
noise). On 50% the trends should be similar; the absolute numbers are
higher.

---

## 5. What to send back

When you're done, please send:

1. **Both eval tables** (refined and un-refined) — full text output of
   the two `eval_refined.py` runs.
2. **The trained SegFix checkpoint**:
   `work_dirs/segfix_r18_ade20k_50pct/iter_20000.pth` (~80 MB).
3. **Training log**:
   `work_dirs/segfix_r18_ade20k_50pct/<timestamp>/<timestamp>.log`
4. **3--5 sample refined predictions** for visual inspection:
   `work_dirs/segfix_refined_baseline_50pct/predictions/ADE_val_*.png`
   (any 3--5; the per-image filenames match the val image IDs).

If anything is unclear, the README at `segfix/README.md` has more on the
architecture and design decisions; everything in this `INSTRUCTIONS.md`
is the minimum-friction path through the same flow.

---

## Troubleshooting

**"NameError: name 'torch' is not defined" or any other import error.**
You're probably on an outdated branch. `git pull` and try again.

**Pretrained weights download hangs.** The first training run downloads
`open-mmlab://resnet18_v1c` from
`https://download.openmmlab.com/pretrain/...`. If your machine can't
reach that, pre-download to `~/.cache/torch/hub/checkpoints/resnet18_v1c-b5776b93.pth`
and the run will pick it up.

**OOM on a small GPU.** Lower `train_dataloader.batch_size` from 16 to
8 (or 4): `--cfg-options train_dataloader.batch_size=8`. Doesn't change
the schedule semantics noticeably; just slows down per-iter throughput.

**`refine.py` says "image not found".** It iterates `*.jpg` in
`data/ade/.../images/validation/` from the seg config's
`val_dataloader.dataset.data_root` — make sure that path exists and
contains the val images.

**Eval mIoU looks much lower than expected (~5%).** The pred PNG class
encoding doesn't match GT. `refine.py` writes 1..150 + 0=ignore which is
the ADE20K convention; `eval_refined.py` shifts by -1 to match
`reduce_zero_label` semantics. If you're feeding it predictions from a
different pipeline, check the encoding before diagnosing the model.
