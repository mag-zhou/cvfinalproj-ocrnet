#!/usr/bin/env python3
"""Plot training loss curves + metric trajectories for Mod 1/2/3 runs.

Reads ``vis_data/scalars.json`` from each run's work_dir. Each line of
that file is a JSON object; training-step lines have keys like
``loss``, ``decode_X.loss_ce``, ``acc_seg``; eval lines have keys
``aAcc``, ``mIoU``, ``mAcc``, ``mBoundaryF1_3/5/9``. We separate them
on whether the line carries an ``mIoU`` key.

For runs whose work_dir contains multiple timestamped subfolders (e.g.
preempted+resumed runs of Mod 2 80k), we read scalars from *all*
subfolders and merge them by step, keeping the last-seen value (so
resumed lines overwrite earlier overlapping lines).

For runs split across *multiple* work_dirs in a warm-restart
(fine-tune-style) extension pattern (e.g. Mod 1 80k = original
``_50pct`` covering iters 0-40k + ``_50pct_ext`` whose own iter
counter restarts at 0 but represents iters 40k-80k of cumulative
training), each entry in ``RUNS`` lists ``phases`` as
``[(work_dir, step_offset), ...]``. Steps from the second phase are
shifted by ``step_offset`` before being merged.

Outputs (relative to repo root):
- ``my_figures/training_curves/mod{1,2,3}_loss.png``
- ``my_figures/metric_trajectories/mod{1,2,3}_metrics.png``

Usage::

    python tools/plot_training_curves.py
    python tools/plot_training_curves.py --output-root my_figures/run_XYZ
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Run registry: which work_dirs to read, what to call them, and which
# iteration count to advertise in the figure title.
#
# Edit this list (or pass --runs to register more) if you ever want to
# include another mod / baseline / Mod 4. Mod 4 is intentionally absent
# per the user's "skip" choice in this iteration.
# ---------------------------------------------------------------------------
RUNS = [
    {
        'mod_id': 'Mod 1',
        'mod_name': 'Boundary supervision in gradient updates',
        # 80k = original _50pct (0-40k) + _50pct_ext warm restart whose own
        # iter counter restarts at 0 but represents iters 40k-80k of
        # cumulative training, so we offset its steps by +40000 before
        # merging.
        'phases': [
            ('work_dirs/ocrnet_r50_mod1_aux_boundary_50pct', 0),
            ('work_dirs/ocrnet_r50_mod1_aux_boundary_50pct_ext', 40_000),
        ],
        'max_iters': 80_000,
        # Loss keys to plot from the training log. Order = legend order.
        # 'total' is a synthetic key we compute as `loss` from the log
        # (mmseg's logger logs the summed loss under this key).
        'loss_keys': [
            ('loss', 'total'),
            ('decode_0.loss_ce', 'decode_0 (FCN aux)'),
            ('decode_1.loss_ce', 'decode_1 (OCR)'),
            ('aux.loss_ce', 'auxiliary boundary BCE'),
        ],
    },
    {
        'mod_id': 'Mod 2',
        'mod_name': 'Boundary Weighted Segmentation Loss Function',
        'phases': [('work_dirs/ocrnet_r50_mod2_weighted_ce_50pct_80k', 0)],
        'max_iters': 80_000,
        'loss_keys': [
            ('loss', 'total'),
            ('decode_0.loss_ce', 'decode_0 (FCN aux, BWCE)'),
            ('decode_1.loss_ce', 'decode_1 (OCR, BWCE)'),
        ],
    },
    {
        'mod_id': 'Mod 3',
        'mod_name': 'Boundary-modulated OCR attention head',
        'phases': [('work_dirs/ocrnet_r50_mod3_modulated_ocr_50pct_80k', 0)],
        'max_iters': 80_000,
        'loss_keys': [
            ('loss', 'total'),
            ('decode_0.loss_ce', 'decode_0 (FCN aux)'),
            ('decode_1.loss_ce', 'decode_1 (OCR)'),
            ('decode_1.loss_boundary_aux', 'boundary aux BCE (T branch)'),
        ],
    },
]

# Six metrics, in the order requested by the user.
METRIC_KEYS = ['aAcc', 'mIoU', 'mAcc', 'mBoundaryF1_3', 'mBoundaryF1_5',
               'mBoundaryF1_9']
METRIC_LABELS = {
    'aAcc': 'aAcc (overall pixel acc.)',
    'mIoU': 'mIoU',
    'mAcc': 'mAcc',
    'mBoundaryF1_3': 'mBoundaryF1 @3px',
    'mBoundaryF1_5': 'mBoundaryF1 @5px',
    'mBoundaryF1_9': 'mBoundaryF1 @9px',
}

# Distinguishable colours; matplotlib will cycle, but pinning helps when
# we plot the same curve across multiple figures.
LOSS_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
               '#8c564b']
METRIC_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e',
                 '#17becf']


# ---------------------------------------------------------------------------
# Scalar reading
# ---------------------------------------------------------------------------

def find_vis_data_files(work_dir: Path) -> List[Path]:
    """Return all ``vis_data/scalars.json`` files inside a work_dir.

    A run that was preempted and resumed has multiple timestamped
    subfolders. We sort them alphabetically (== chronologically since
    timestamps are ``YYYYMMDD_HHMMSS``) so later runs overwrite earlier
    ones in the merged dict.
    """
    if not work_dir.is_dir():
        return []
    candidates = []
    for sub in sorted(work_dir.iterdir()):
        if sub.is_dir():
            scalars = sub / 'vis_data' / 'scalars.json'
            if scalars.is_file():
                candidates.append(scalars)
    return candidates


def parse_scalar_lines(path: Path) -> List[dict]:
    """Parse a scalars.json file: one JSON object per line."""
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate the rare truncated-last-line on a killed job.
                continue
    return out


def collect_scalars(work_dir: Path) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Read every scalars.json under a work_dir; return (train, eval).

    Both are dicts keyed by step. If multiple files cover overlapping
    steps (preempt+resume), later files overwrite earlier ones.
    """
    train: Dict[int, dict] = {}
    evals: Dict[int, dict] = {}
    for sf in find_vis_data_files(work_dir):
        for entry in parse_scalar_lines(sf):
            step = entry.get('step')
            if step is None:
                continue
            if 'mIoU' in entry:
                evals[int(step)] = entry
            else:
                train[int(step)] = entry
    return train, evals


def collect_scalars_phases(phases: List[Tuple[Path, int]]
                           ) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Like ``collect_scalars`` but across multiple phases.

    ``phases`` is a list of ``(work_dir, step_offset)``. Steps from each
    phase have ``step_offset`` added before being merged. Phases are
    processed in list order, so later phases overwrite earlier overlapping
    keys -- which only matters for the synthetic case of two phases that
    end up at the same offset-shifted step.
    """
    train_all: Dict[int, dict] = {}
    evals_all: Dict[int, dict] = {}
    for work_dir, offset in phases:
        train, evals = collect_scalars(work_dir)
        for step, entry in train.items():
            train_all[step + offset] = entry
        for step, entry in evals.items():
            evals_all[step + offset] = entry
    return train_all, evals_all


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smoothing_window(n: int) -> int:
    """Choose a moving-average window size that scales with run length."""
    if n <= 100:
        return 1
    if n <= 1_000:
        return 25
    if n <= 5_000:
        return 50
    return 100


def moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    smoothed = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def plot_loss_curves(run_cfg: dict, train: Dict[int, dict],
                     output_dir: Path) -> Path:
    """Plot all loss components on one axis."""
    if not train:
        raise RuntimeError(f'No training-step scalars found for {run_cfg["mod_id"]}')

    steps = np.array(sorted(train.keys()), dtype=np.int64)

    fig, ax = plt.subplots(figsize=(10, 6))
    win = smoothing_window(len(steps))

    plotted_any = False
    for (key, label), color in zip(run_cfg['loss_keys'], LOSS_COLORS):
        ys = np.array([train[s].get(key, np.nan) for s in steps], dtype=np.float64)
        if not np.isfinite(ys).any():
            continue
        ax.plot(steps, moving_average(ys, win), color=color, linewidth=1.5,
                label=label)
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(
            f'None of the requested loss keys were present for '
            f'{run_cfg["mod_id"]}: {[k for k, _ in run_cfg["loss_keys"]]}')

    iters_label = (f'{run_cfg["max_iters"] // 1000}k' if run_cfg["max_iters"] >= 1000
                   else str(run_cfg["max_iters"]))
    title = (f'[{run_cfg["mod_id"]}] Training Curve, trained on 50% ADE20K '
             f'dataset for {iters_label} iterations')
    # Subtitle goes on the second line of the same axes title to avoid the
    # overlap that comes from layering ax.text on top of set_title.
    full_title = f'{title}\n{run_cfg["mod_name"]}'
    ax.set_title(full_title, fontsize=12, pad=10)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, run_cfg['max_iters'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    ax.text(0.99, 0.02, f'(moving avg. window: {win} iters)' if win > 1 else
            '(no smoothing)', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#888')

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = run_cfg['mod_id'].replace(' ', '').lower()
    out_path = output_dir / f'{safe_id}_loss.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def plot_metric_trajectory(run_cfg: dict, evals: Dict[int, dict],
                           output_dir: Path) -> Path:
    """All six eval metrics on one axis (they share a 0-100 scale)."""
    if not evals:
        raise RuntimeError(f'No eval scalars found for {run_cfg["mod_id"]}')

    steps = np.array(sorted(evals.keys()), dtype=np.int64)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for key, color in zip(METRIC_KEYS, METRIC_COLORS):
        ys = np.array([evals[s].get(key, np.nan) for s in steps], dtype=np.float64)
        if not np.isfinite(ys).any():
            continue
        ax.plot(steps, ys, color=color, linewidth=1.8, marker='o', markersize=4,
                label=METRIC_LABELS[key])
        plotted_any = True

    if not plotted_any:
        raise RuntimeError(f'No metrics present for {run_cfg["mod_id"]}.')

    iters_label = (f'{run_cfg["max_iters"] // 1000}k' if run_cfg["max_iters"] >= 1000
                   else str(run_cfg["max_iters"]))
    title = (f'[{run_cfg["mod_id"]}] Validation Metrics, trained on 50% '
             f'ADE20K dataset for {iters_label} iterations')
    full_title = f'{title}\n{run_cfg["mod_name"]}'
    ax.set_title(full_title, fontsize=12, pad=10)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score (%)')
    ax.set_xlim(0, run_cfg['max_iters'])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = run_cfg['mod_id'].replace(' ', '').lower()
    out_path = output_dir / f'{safe_id}_metrics.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='my_figures',
                        help='Where to write training_curves/ and metric_trajectories/.')
    parser.add_argument('--repo-root', default='.',
                        help='Repository root (defaults to current dir).')
    parser.add_argument('--mods', nargs='*', default=None,
                        help='Optional subset of mod ids to plot, e.g. "Mod 1" "Mod 3".')
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.output_root).resolve()
    loss_dir = out_root / 'training_curves'
    met_dir = out_root / 'metric_trajectories'

    runs = RUNS
    if args.mods:
        wanted = set(args.mods)
        runs = [r for r in RUNS if r['mod_id'] in wanted]
        if not runs:
            raise SystemExit(f'No matching runs for --mods {args.mods!r}')

    print(f'Reading scalars from {repo}')
    print(f'Writing figures to {out_root}\n')

    for run_cfg in runs:
        # Backward-compat: accept either 'phases' (list of (path, offset)) or
        # legacy 'work_dir' (single path, offset 0).
        if 'phases' in run_cfg:
            phases = [(repo / wd, off) for wd, off in run_cfg['phases']]
        else:
            phases = [(repo / run_cfg['work_dir'], 0)]
        train, evals = collect_scalars_phases(phases)
        phases_str = ', '.join(
            f'{p.relative_to(repo)}@+{off}' if off else f'{p.relative_to(repo)}'
            for p, off in phases)
        print(f'[{run_cfg["mod_id"]}]  phases=[{phases_str}]')
        print(f'  training-step rows: {len(train)} | eval rows: {len(evals)}')
        if not train and not evals:
            print('  -> no scalars; skipping.\n')
            continue

        if train:
            p = plot_loss_curves(run_cfg, train, loss_dir)
            print(f'  loss curve  -> {p.relative_to(repo)}')
        if evals:
            p = plot_metric_trajectory(run_cfg, evals, met_dir)
            print(f'  metrics     -> {p.relative_to(repo)}')
        print()


if __name__ == '__main__':
    main()
