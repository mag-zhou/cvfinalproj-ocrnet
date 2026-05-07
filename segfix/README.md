# SegFix on OCRNet (ADE20K 50%)

> **Friend running this on another GPU?** Skip this README and read
> [`INSTRUCTIONS.md`](./INSTRUCTIONS.md) — same flow but as a copy-paste
> runbook with no design discussion.

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
| `INSTRUCTIONS.md` | runbook | Friend-facing copy-paste workflow. |
| `offset_utils.py` | shared | `boundary_mask_from_seg`, `offsets_from_seg` (numpy/cv2). |
| `gen_offset_gt.py` | 1 (sanity) | Visualize / precompute offsets. **Not used by training.** |
| `transforms.py` | 2 | `ComputeOffsetsFromSeg` + `PackSegFixInputs`. Online GT generation. |
| `model.py` | 2 | `SegFixOffsetModel` (ResNet-18 backbone + boundary + offset heads). |
| `metric.py` | 2 (val) | `SegFixOffsetMetric` -- diagnostic boundary F1 & EE during training. |
| `configs/segfix_r18_ade20k_50pct.py` | 2 | Training config. |
| `slurm/train_slurm_segfix_50pct.sh` | 3 | Optional SLURM wrapper for train. |
| `refine.py` | 4 | Standalone refinement: baseline OCRNet + SegFix offsets -> refined PNGs. |
| `slurm/refine_slurm.sh` | 4 | Optional SLURM wrapper for `refine.py`. |
| `eval_refined.py` | 5 | Standalone mIoU + boundary F1 over a directory of predictions. |
| `slurm/eval_refined_slurm.sh` | 5 | Optional SLURM wrapper for `eval_refined.py`. |

## Key design decision (read this)

**Offsets are computed online from the augmented seg map**, not loaded
from disk. The plan's Phase 1 script (`gen_offset_gt.py`) is kept as a
sanity-check / visualization tool, but training does not depend on its
output.

Why? Offset is a vector field. The standard mmseg geometric augmentations
(`RandomResize`, `RandomCrop`, `RandomFlip`) operate on scalar arrays via
`seg_fields`; they don't know that horizontal flip needs to negate dx,
or that resize needs to scale magnitudes. Hooking into them would
require custom subclasses or monkey-patching, both fragile. Recomputing
the offset from the (already-augmented) seg map is exactly equivalent,
runs ~tens of ms in the dataloader workers, and cannot drift out of sync
with the seg map.

The boundary mask used for the BCE target is also computed online (same
8-neighbor disagreement rule as `tools/preprocess/gen_boundary_gt.py` and
as `mmseg/evaluation/metrics/boundary_metric.py`). Consistent everywhere.

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
  Then zero out offsets in deep interior (>kernel_size/2 from any
  boundary) so visualization matches the plan's expectation.
```

**Sanity-verified on:**
- a synthetic 100x100 square: center offset = 0, edge pixels push 8 px
  inward.
- a real ADE20K label: median offset 8 px in a 1-px boundary band, 0 in
  pixels >16 px from any boundary.

If you see results inconsistent with that, run
`python segfix/gen_offset_gt.py --visualize --max-images 10` and look at
the magnitude / direction PNGs in
`data/ade/ADEChallengeData2016/offset_viz/` before assuming the model
architecture is the bug.

## Why class-agnostic refinement?

SegFix doesn't know what class anything is — it only emits a boundary
mask and a per-pixel offset that says "look here instead". The
refinement step reads `seg_pred[i + dy, j + dx]` and writes that label
to `(i, j)` for every predicted boundary pixel. Whether the refinement
helps depends entirely on whether the seg model already gets the
**interior** right; if the baseline mispredicts whole regions, SegFix
won't fix that. It only helps boundary localization.

On ADE20K, stuff classes (sky, wall, ceiling) have diffuse boundaries
and the gain from SegFix is smaller than on Cityscapes thing-classes.
The original SegFix paper reports +0.5 to +1.5 mIoU on Cityscapes; on
ADE20K with 50% data, expect somewhat less. Boundary-F1 should move
more cleanly than mIoU.

## Hyperparameters worth ablating (cheap, no retraining)

- **`--boundary-thresh`** in `refine.py` (0.3 / 0.5 / 0.7). Each rerun
  produces a different output dir; eval all of them with the same
  `eval_refined.py` to pick the best.
- **`max_offset` / `kernel_size`** in the config: tied (radius = 8 / side
  = 17). Larger = larger possible offsets at the cost of more ambiguous
  training signal. Probably not worth changing unless mIoU *worsens*.
