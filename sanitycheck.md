# Sanity checks by phase (boundary experiments)

Run these when something breaks: each section is **independent** and lists **what to run** and **what “good” looks like**. Adjust paths if your `data_root` or Python env differ.

---

## Phase 0 — Branch and project wiring

**Checks**

- [ ] You are on the `boundary-experiments` branch (or your working branch with these changes).
- [ ] Folders exist: `tools/preprocess/`, `configs/ocrnet/boundary/`.
- [ ] `CascadeEncoderDecoder` still uses the parent loss path so `auxiliary_head` is included in training (no custom patch required).

**Quick command**

```bash
git branch --show-current
test -d tools/preprocess && test -d configs/ocrnet/boundary && echo ok
```

**Expected:** Current branch name printed; `ok`.

---

## Phase 1 — Precomputed boundary GT

**Checks**

- [ ] Script runs and writes boundary + distance files under your ADE root (defaults to `data/ade/ADEChallengeData2016/`).
- [ ] For a sample training id, you see paired files under `boundaries/` and `boundary_dist/` (or the paths your script prints).

**Commands**

```bash
python3 tools/preprocess/gen_boundary_gt.py --help
# After a short run (or full run), spot-check:
ls data/ade/ADEChallengeData2016/boundaries 2>/dev/null | head
ls data/ade/ADEChallengeData2016/boundary_dist 2>/dev/null | head
```

**Expected:** `--help` shows arguments; listing shows many `*_boundary.png`-style files (exact names match your script). No repeated crash tracebacks on multiprocessing workers.

---

## Phase 2 — Dataset transforms

**Checks**

- [ ] Imports resolve: `LoadBoundaryAnnotations`, `PackSegBoundaryInputs`.
- [ ] Training samples include `gt_boundary` / `gt_boundary_dist` when using the boundary pipeline (inspect one batch or add a one-off print in a tiny script).

**Command**

```bash
python3 -c "from mmseg.datasets.transforms import LoadBoundaryAnnotations, PackSegBoundaryInputs; print('ok')"
```

**Expected:** `ok` (requires `mmseg` and deps on `PYTHONPATH` / installed).

---

## Phase 3 — Mod 1 (aux boundary head)

**Config:** `configs/ocrnet/boundary/ocrnet_r50_mod1_aux_boundary.py`

**Checks**

- [ ] Config loads without error.
- [ ] Build model: `type=BoundaryHead` under `model.auxiliary_head`, `in_index=2`, BCE path (`use_sigmoid=True`, `out_channels=1` as in config).
- [ ] Short train / dry-run: logs show **both** main decode losses and boundary-related loss (name depends on head; look for `loss_ce` on aux or boundary).

**Commands**

```bash
python3 tools/train.py configs/ocrnet/boundary/ocrnet_r50_mod1_aux_boundary.py \
  --cfg-options train_dataloader.dataset.indices=10 train_dataloader.batch_size=1 train_cfg.max_iters=2 2>&1 | tail -50
```

**Expected:** No exception during model init or first iterations; loss dict printed with multiple components. If data paths are wrong, you get **file not found** for images or boundary maps — fix `data_root` and rerun Phase 1 outputs.

---

## Phase 4 — Mod 2 (boundary-weighted CE)

**Config:** `configs/ocrnet/boundary/ocrnet_r50_mod2_weighted_ce.py`

**Checks**

- [ ] Loss type `BoundaryWeightedCrossEntropy` registers and heads are `WeightedCEFCNHead` / `WeightedCEOCRHead`.
- [ ] Same smoke train as Mod 1: forward runs; losses finite.

**Command**

```bash
python3 -c "from mmseg.models.losses import BoundaryWeightedCrossEntropy; from mmseg.models.decode_heads import WeightedCEFCNHead, WeightedCEOCRHead; print('ok')"
python3 tools/train.py configs/ocrnet/boundary/ocrnet_r50_mod2_weighted_ce.py \
  --cfg-options train_dataloader.dataset.indices=10 train_dataloader.batch_size=1 train_cfg.max_iters=2 2>&1 | tail -50
```

**Expected:** `ok`; train tail shows no shape error for `boundary_dist` (if missing, check Phase 1 + Phase 2 pipeline).

---

## Phase 5 — Mod 3 (modulated OCR)

**Config:** `configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr.py`

**Checks**

- [ ] Second decode head is `OCRBoundaryHead` (not `OCRHead`).
- [ ] Smoke train: extra term `loss_boundary_aux` appears when `gt_boundary` is loaded.

**Commands**

```bash
python3 -c "from mmseg.models.decode_heads import OCRBoundaryHead; print('ok')"
python3 tools/train.py configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr.py \
  --cfg-options train_dataloader.dataset.indices=10 train_dataloader.batch_size=1 train_cfg.max_iters=2 2>&1 | tail -80
```

**Expected:** `ok`; log lines include `loss_boundary_aux` (or training proceeds without error if your logger collapses keys — then confirm in debugger). **Watch for:** NaNs in loss → lower LR or check boundary targets in `[0,1]`.

---

## Phase 5b — Mod 4 (CBL-lite contrastive boundary loss)

**Config:** `configs/ocrnet/boundary/ocrnet_r50_mod4_cbl.py` (or the 50pct/80k variants).

**Checks**

- [ ] Imports resolve: `BoundaryContrastiveLoss` and `OCRCBLHead`.
- [ ] Unit tests pass (gradient-detach property, all-anchor-skipped case, etc).
- [ ] Smoke train: log lines include `decode_1.loss_cbl` and `decode_1.cbl_anchor_ratio`.
- [ ] **Phase 4 Test 3 (early-iter behaviour):** `loss_cbl ~ 0` for the first
      ~500–2000 iters because most predictions are wrong, so most anchors fail
      CCAS and get skipped. It should ramp up as the model learns. If it's
      already large at iter 100, the CCAS check is wrong.
- [ ] **Phase 4 Test 4 (anchor count sanity):** `cbl_anchor_ratio` (the
      n_valid_anchors / n_total_anchors ratio) should land in the 0.2–0.7
      range once training has warmed up. If 0 throughout, no CCAS positives
      are being found (bug). If always 1.0, CCAS isn't filtering (bug).
- [ ] **Phase 4 Test 2 (speed):** Iter time should be at most ~30% slower
      than Mod 2 / Mod 3. If it's 2× or more, the unfold/gather indexing
      is broken and the loss has fallen back to Python loops; profile.

**Commands**

```bash
python3 -c "from mmseg.models.losses import BoundaryContrastiveLoss; from mmseg.models.decode_heads import OCRCBLHead; print('ok')"

# Unit tests (CPU-only, fast):
OMP_NUM_THREADS=1 python3 tests/test_boundary_contrastive.py

# 50-iter smoke train (mirrors Phase 4 Test 1 in the plan):
python3 tools/train.py configs/ocrnet/boundary/ocrnet_r50_mod4_cbl.py \
  --cfg-options train_dataloader.dataset.indices=10 train_dataloader.batch_size=2 \
                train_cfg.max_iters=50 train_cfg.val_interval=100 2>&1 | tail -120
```

**Expected:** `ok`; unit tests print `All BoundaryContrastiveLoss sanity tests passed.`; smoke-train tail shows `decode_1.loss_cbl` ~ 0 (or very small) with `cbl_anchor_ratio` near 0 because the model hasn't learned anything yet. **Watch for:** NaNs (lower `cbl_weight` or `cbl_margin`); OOM (lower `cbl_max_anchors`).

---

## Phase 6 — Boundary F-score metric

**Checks**

- [ ] `BoundaryFScore` imports and builds from config.
- [ ] After evaluation, logs include `mBoundaryF1_3`, `mBoundaryF1_5`, `mBoundaryF1_9` (percent scale).

**Commands**

```bash
python3 -c "from mmseg.evaluation import BoundaryFScore; print('ok')"
# `tools/test.py` requires a checkpoint path as the second positional argument:
python3 tools/test.py configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py /path/to/checkpoint.pth \
  --cfg-options test_dataloader.dataset.indices=5 2>&1 | tail -40
```

**Expected:** `ok`; if you have a checkpoint, the test tail mentions boundary F-score keys. **Note:** Full val is slow; `indices=5` is for smoke only.

---

## Phase 7 — Baseline + runbook

**Config:** `configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py`

**Checks**

- [ ] Same weights/training as `ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py` except added evaluators.
- [ ] **100-iter smoke** (adjust flag names to your MMEngine version):

```bash
python3 tools/train.py configs/ocrnet/boundary/ocrnet_r50_baseline_20pct.py \
  --cfg-options train_cfg.max_iters=100 2>&1 | tail -30
```

**Expected:** Completes 100 iters; val hook may not run depending on `val_interval` — that is fine for smoke. For **full 40k schedule**, use your cluster time estimates; **resume** via `resume=True` and `load_from` per MMEngine docs.

---

## Phase 8 — Documentation inventory

**Checks**

- [ ] `BOUNDARY_EXPERIMENTS.md` lists the files you actually have (update if you moved paths).
- [ ] This file (`sanitycheck.md`) matches your real commands (`tools/train.py` / `tools/test.py` exist in repo root).

**Command**

```bash
test -f BOUNDARY_EXPERIMENTS.md && test -f sanitycheck.md && echo ok
```

**Expected:** `ok`.

---

## Cross-cutting failures

| Symptom | Likely cause |
|--------|----------------|
| `FileNotFoundError` for `*_boundary.png` | Phase 1 not run or wrong `data_root` / split paths. |
| KeyError / missing `gt_boundary_dist` | Mod 2 pipeline without `LoadBoundaryAnnotations` + distance maps. |
| Registry errors (`XXX is not in the registry`) | Package not installed editable, or import order — run from repo root with `pip install -e .`. |
| Val metrics missing boundary keys | Config missing list `val_evaluator` with `BoundaryFScore`; or test script old. |

Use **one Mod config at a time** plus **baseline** for comparisons; keep `indices=4042` consistent unless you intentionally ablate data size.
