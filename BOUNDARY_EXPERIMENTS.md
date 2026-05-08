# Boundary experiments (OCRNet on ADE20K 20%)

This document maps the **boundary-experiments** work: precomputed labels, data pipeline, model variants, metrics, and configs.

## Directory layout (added or meaningfully changed)

| Path | Role |
|------|------|
| `tools/preprocess/gen_boundary_gt.py` | Precompute 8-neighbor binary boundaries + L2 distance transform. |
| `data/ade/ADEChallengeData2016/boundaries/` | Precomputed `*_boundary.png` (gitignored). |
| `data/ade/ADEChallengeData2016/boundary_dist/` | Precomputed distance maps (gitignored). |
| `mmseg/datasets/transforms/boundary.py` | `LoadBoundaryAnnotations`, `PackSegBoundaryInputs`. |
| `mmseg/models/decode_heads/boundary_head.py` | **Mod 1** auxiliary `BoundaryHead` (BCE on boundary). |
| `mmseg/models/losses/boundary_weighted_ce.py` | **Mod 2** `BoundaryWeightedCrossEntropy`. |
| `mmseg/models/decode_heads/weighted_ce_heads.py` | `WeightedCEFCNHead` / `WeightedCEOCRHead`. |
| `mmseg/models/decode_heads/ocr_boundary_head.py` | **Mod 3** `OCRBoundaryHead` (temperature-modulated OCR attention + aux boundary). |
| `mmseg/models/losses/boundary_contrastive.py` | **Mod 4** `BoundaryContrastiveLoss` (CBL-lite). |
| `mmseg/models/decode_heads/ocr_cbl_head.py` | **Mod 4** `OCRCBLHead` (OCR + contrastive boundary loss on augmented features). |
| `tests/test_boundary_contrastive.py` | Unit tests for the CBL loss (gradient detach, anchor filtering). |
| `mmseg/evaluation/metrics/boundary_metric.py` | `BoundaryFScore` at tolerances 3 / 5 / 9 px. |
| `configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py` | Baseline + IoU + boundary F-score on val. |
| `configs/ocrnet/boundary/ocrnet_r50_mod1_aux_boundary.py` | Mod 1. |
| `configs/ocrnet/boundary/ocrnet_r50_mod2_weighted_ce.py` | Mod 2. |
| `configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr.py` | Mod 3. |
| `configs/ocrnet/boundary/ocrnet_r50_mod4_cbl.py` | Mod 4 (20% base). |
| `configs/ocrnet/boundary/ocrnet_r50_mod4_cbl_50pct.py` | Mod 4 (50%). |
| `configs/ocrnet/boundary/ocrnet_r50_mod4_cbl_50pct_80k.py` | Mod 4 (50%, fresh 80k). |
| `sanitycheck.md` | Per-phase checks (commands and expected signals). |

## Config quick reference

- **Baseline (20%)**: `configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py` — same training as `ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py`, extra val metrics only.
- **Mod 1–3**: under `configs/ocrnet/boundary/`, each sets `indices=4042`, boundary-aware pipeline, and `val_evaluator` = `IoUMetric` + `BoundaryFScore`.

## Mod summary

1. **Mod 1**: `auxiliary_head=BoundaryHead` on stage-1 features; BCE on precomputed boundary.
2. **Mod 2**: `WeightedCEFCNHead` + `WeightedCEOCRHead` with `BoundaryWeightedCrossEntropy` (uses `gt_boundary_dist`).
3. **Mod 3**: Second stage is `OCRBoundaryHead` — learnable temperature map T on attention logits, plus BCE on an auxiliary boundary map from the same branch.
4. **Mod 4 (CBL-lite)**: Second stage is `OCRCBLHead` — vanilla OCR CE plus `BoundaryContrastiveLoss` on the OCR pre-classifier features. For each boundary pixel we pull its feature toward the mean of correctly-classified same-class neighbours within a `K x K` window (CCAS positives) and push away from correctly-classified different-class neighbours via a hinge term. Centres and negatives are detached so only the anchor's own feature carries gradient. **Simplified vs. the full CBL paper**: no SCE-supervised local class-centre prediction head; we compute centres directly from the feature map at boundary anchors. See the loss module's docstring for the full diff.

## Training entrypoint

Use the project’s MMEngine train script (e.g. `tools/train.py`) with `--config` pointing at one of the configs above. See `sanitycheck.md` for smoke-test flags and what to verify in logs.
