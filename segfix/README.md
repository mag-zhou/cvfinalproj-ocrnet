# SegFix on OCRNet (ADE20K 20%)

SegFix as the 5th condition in the boundary-experiments series:

1. Baseline OCRNet
2. OCRNet + Mod 1 (auxiliary boundary head)
3. OCRNet + Mod 2 (boundary-weighted CE)
4. OCRNet + Mod 3 (modulated OCR + aux boundary)
5. **Baseline OCRNet + SegFix (post-hoc refinement)**

Everything SegFix-specific is contained in this folder.

## Layout

| File | Phase | Role |
|---|---|---|
| `offset_utils.py` | shared | `boundary_mask_from_seg`, `offsets_from_seg` (numpy/cv2). |
| `gen_offset_gt.py` | 1 (sanity) | Precompute / visualize offsets to disk. **Not used by training.** |
| `transforms.py` | 2 | `ComputeOffsetsFromSeg` + `PackSegFixInputs`. Online GT generation. |
| `model.py` | 2 | `SegFixOffsetModel` (ResNet-18 backbone + boundary + offset heads). |
| `metric.py` | 2 (val) | `SegFixOffsetMetric` -- diagnostic boundary F1 & EE during training. |
| `configs/segfix_r18_ade20k_20pct.py` | 2 | Training config. |
| `slurm/train_slurm_segfix_20pct.sh` | 3 | Slurm train. ~2h on 1 GPU. |
| `refine.py` | 4 | Standalone refinement: baseline OCRNet + SegFix offsets -> refined PNGs. |
| `slurm/refine_slurm.sh` | 4 | Slurm wrapper for `refine.py`. |
| `eval_refined.py` | 5 | Standalone mIoU + boundary F1 over a directory of predictions. |
| `slurm/eval_refined_slurm.sh` | 5 | Evaluates both refined and un-refined preds with the same code. |

## Key design decision (read this)

**Offsets are computed online from the augmented seg map**, not loaded from
disk. The plan's Phase 1 script (`gen_offset_gt.py`) is kept as a
sanity-check / visualization tool, but training does not depend on its
output.

Why? Offset is a vector field. The standard mmseg geometric augmentations
(`RandomResize`, `RandomCrop`, `RandomFlip`) operate on scalar arrays via
`seg_fields`; they don't know that horizontal flip needs to negate dx, or
that resize needs to scale magnitudes. Hooking into them would require
custom subclasses or monkey-patching, both fragile. Recomputing the offset
from the (already-augmented) seg map is exactly equivalent, runs ~tens of
ms in the dataloader workers, and cannot drift out of sync with the seg
map.

If you ever want to switch to disk-cached offsets (faster IO at the cost of
larger storage and the geometric headache), the entry point is
`ComputeOffsetsFromSeg` in `transforms.py` -- replace it with a
`LoadOffsetAnnotations` + per-augmentation vector hooks.

The boundary mask used for the BCE target is also computed online (same 8-
neighbor disagreement rule as `tools/preprocess/gen_boundary_gt.py` and as
`mmseg/evaluation/metrics/boundary_metric.py`). Consistent everywhere.

## Algorithm cross-check (from the plan)

The implementation plan flagged that the offset GT generation has details
that are easy to get subtly wrong, and recommends cross-checking against
openseg.pytorch's `dt_offset_generator.py`. This implementation follows
the conceptual recipe in the plan:

```
For each class c in label L:
  M_c = (L == c)
  D_c = cv2.distanceTransform(M_c, DIST_L2, 5)   # depth into region
  For each pixel of class c:
    Look at D_c in a (kernel_size x kernel_size) window;
    offset (dy, dx) = (i*-i, j*-j) for the argmax location.
```

**Verification step before trusting refinement results:** run

```bash
python segfix/gen_offset_gt.py --visualize --max-images 10
```

and inspect `data/ade/ADEChallengeData2016/offset_viz/training/`:

- magnitude PNG (`*_mag.png`): near zero in interiors, 1-8 px near class
  boundaries.
- direction PNG (`*_dir.png`, HSV): boundary regions show coherent
  direction pointing inward (away from the boundary).

If those don't look right, **don't proceed to training** -- diff against
openseg's source first.

## Runbook

Assumes baseline OCRNet checkpoint already exists at
`work_dirs/ocrnet_r50_ade20k_20pct/iter_40000.pth` (the existing baseline
trained from `configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py`
or the Phase 7 baseline config).

```bash
# Phase 1 (optional sanity): visual inspection of offset GT.
sbatch segfix/slurm/gen_offset_viz_slurm.sh
# After it runs, look at the PNGs in
#   data/ade/ADEChallengeData2016/offset_viz/training/

# Phase 2-3: train the SegFix offset model. ~2h on H200.
sbatch segfix/slurm/train_slurm_segfix_20pct.sh

# Phase 4: refine baseline predictions.
SEG_CHECKPOINT=work_dirs/ocrnet_r50_ade20k_20pct/iter_40000.pth \
OFFSET_CHECKPOINT=work_dirs/segfix_r18_ade20k_20pct/iter_20000.pth \
OUTPUT=work_dirs/segfix_refined_baseline \
sbatch segfix/slurm/refine_slurm.sh

# Phase 5: evaluate both refined and baseline using identical eval code.
OUTPUT_ROOT=work_dirs/segfix_refined_baseline \
sbatch segfix/slurm/eval_refined_slurm.sh
```

## Smoke tests (before sbatching the full train)

```bash
# (1) Imports register cleanly.
python -c "import segfix; print('ok')"

# (2) Offset utility produces non-zero offsets near boundaries on a real label.
python -c "
import cv2, numpy as np
from segfix.offset_utils import offsets_from_seg, boundary_mask_from_seg
seg = cv2.imread('data/ade/ADEChallengeData2016/annotations/training/ADE_train_00000001.png', cv2.IMREAD_UNCHANGED).astype(np.int64)
seg[seg == 0] = 255  # treat 0 as ignore (matches reduce_zero_label)
seg[seg != 255] -= 1
off = offsets_from_seg(seg)
bd = boundary_mask_from_seg(seg)
print('offset abs mean (boundary): %.3f' % np.abs(off[bd]).mean())
print('offset abs mean (interior): %.3f' % np.abs(off[~bd]).mean())
"
# Expect: boundary mean clearly > interior mean.

# (3) 2-iter dry-run of the model.
python tools/train.py segfix/configs/segfix_r18_ade20k_20pct.py \
    --cfg-options train_dataloader.dataset.indices=4 \
                  train_dataloader.batch_size=2 \
                  train_cfg.max_iters=2 \
                  val_cfg=None 2>&1 | tail -30
```

## Hyperparameters worth ablating (cheap, no retraining)

- **`--boundary-thresh`** in `refine.py` (0.3 / 0.5 / 0.7). The plan flags
  this as a free hyperparameter; sweep it on the same trained model.
- **`max_offset` / `kernel_size`** in the config: both are tied (radius 8 /
  side 17). Larger window = larger possible offsets at the cost of more
  ambiguous training signal. Probably not worth changing unless mIoU
  *worsens* after refinement.

## Known limitations (from the plan, repeated here so they don't get lost)

1. **Class-agnostic refinement** -- SegFix can only fix boundary
   localization, not interior misclassification. ADE20K stuff classes
   (sky, wall, ceiling) have diffuse boundaries; gains there will be
   smaller than the +0.5 to +1.5 mIoU reported on Cityscapes.
2. **GT correctness is hidden** -- if the offset algorithm has a bug, the
   model still trains (offsets are self-consistent within the bug), but
   refinement may help less or do nothing. Sanity-visualize first.
3. **`refine.py` requires raw class predictions** -- we run the seg model
   in `predict` mode and read `pred_sem_seg`; this is the standard mmseg
   output and matches what `tools/test.py` produces.
