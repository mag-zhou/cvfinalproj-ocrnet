#!/usr/bin/env python3
"""Build the final-scores tables for Mod 1 / Mod 2 / Mod 3.

Two LaTeX tables are produced:

  Table 1 (--out-absolute): absolute scores. Columns = the six metrics
  (aAcc, mIoU, mAcc, mBoundaryF1_3/5/9). Rows = each run.

  Table 2 (--out-delta): the same six metrics, but each cell is
  ``raw_value (percentage_change_vs_baseline)``. Rows = each
  modification. Baseline OCRNet numbers are placeholders (we don't have
  a clean 50pct baseline run on disk; you said you'd fill in the
  numbers from your friend's parallel run).

The tool reads ``vis_data/scalars.json`` for every run (merging across
preempt+resume timestamps) and pulls the *last* line that contains an
``mIoU`` key as that run's final eval.

Output: prints LaTeX to stdout and writes two .tex files.

Usage::

    python tools/build_results_tables.py
    python tools/build_results_tables.py --baseline-aacc 75.0 --baseline-miou 30.0 \
        --baseline-macc 41.0 --baseline-bf3 50.0 --baseline-bf5 60.0 --baseline-bf9 70.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

# Reuse the run registry from the plotting script -- single source of
# truth for which work_dirs to read.
from plot_training_curves import RUNS, collect_scalars, METRIC_KEYS


METRIC_HEADER = {
    'aAcc': r'aAcc',
    'mIoU': r'mIoU',
    'mAcc': r'mAcc',
    'mBoundaryF1_3': r'mBF$_1^{3}$',
    'mBoundaryF1_5': r'mBF$_1^{5}$',
    'mBoundaryF1_9': r'mBF$_1^{9}$',
}

# Placeholders. Override on the CLI (or by editing this dict) once your
# friend's plain-OCRNet 50pct numbers are in.
DEFAULT_BASELINE = {
    'aAcc':          float('nan'),
    'mIoU':          float('nan'),
    'mAcc':          float('nan'),
    'mBoundaryF1_3': float('nan'),
    'mBoundaryF1_5': float('nan'),
    'mBoundaryF1_9': float('nan'),
}


def get_final_eval(work_dir: Path) -> Optional[Dict[str, float]]:
    _, evals = collect_scalars(work_dir)
    if not evals:
        return None
    last_step = max(evals.keys())
    final = evals[last_step]
    out = {}
    for k in METRIC_KEYS:
        v = final.get(k)
        if v is not None:
            out[k] = float(v)
    out['_step'] = int(last_step)
    return out


def fmt_value(v: float, ndigits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return r'\textit{TBD}'
    return f'{v:.{ndigits}f}'


def fmt_delta_cell(raw: float, base: float, ndigits: int = 2) -> str:
    raw_s = fmt_value(raw, ndigits)
    if base is None or math.isnan(base) or base == 0:
        return f'{raw_s} (\\textit{{TBD}})'
    if raw is None or math.isnan(raw):
        return r'\textit{TBD}'
    pct = 100.0 * (raw - base) / base
    sign = '+' if pct >= 0 else ''
    return f'{raw_s} ({sign}{pct:.2f}\\%)'


def build_absolute_table(rows: List[dict]) -> str:
    """LaTeX for Table 1: absolute scores."""
    cols = ' & '.join([r'\textbf{' + METRIC_HEADER[k] + '}' for k in METRIC_KEYS])
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Final validation scores on ADE20K val (50\% training '
        r'data). All values are percentages; higher is better. Iter '
        r'count in parentheses is the iteration of the final eval.}',
        r'\label{tab:absolute-scores}',
        r'\begin{tabular}{l' + 'c' * len(METRIC_KEYS) + '}',
        r'\toprule',
        r'\textbf{Run} & ' + cols + r' \\',
        r'\midrule',
    ]
    for r in rows:
        cells = [fmt_value(r['metrics'].get(k, float('nan'))) for k in METRIC_KEYS]
        run_label = (f'{r["mod_id"]}: {r["mod_name"]} '
                     f'({r["max_iters"] // 1000}k iters)')
        lines.append(run_label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def build_delta_table(rows: List[dict], baseline: Dict[str, float]) -> str:
    """LaTeX for Table 2: raw value (percentage change vs baseline)."""
    cols = ' & '.join([r'\textbf{' + METRIC_HEADER[k] + '}' for k in METRIC_KEYS])
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Modification deltas vs.\ the plain-OCRNet baseline '
        r'(50\% ADE20K). Each cell shows the raw metric value followed '
        r'by the relative change vs.\ the baseline in parentheses. '
        r'Baseline numbers marked \textit{TBD} should be replaced with '
        r"the parallel-run numbers once available.}",
        r'\label{tab:delta-vs-baseline}',
        r'\begin{tabular}{l' + 'c' * len(METRIC_KEYS) + '}',
        r'\toprule',
        r'\textbf{Method} & ' + cols + r' \\',
        r'\midrule',
    ]

    base_cells = [fmt_value(baseline.get(k, float('nan'))) for k in METRIC_KEYS]
    lines.append(r'OCRNet baseline (no boundary mods) & ' + ' & '.join(base_cells) + r' \\')
    lines.append(r'\midrule')

    for r in rows:
        cells = [
            fmt_delta_cell(r['metrics'].get(k, float('nan')),
                           baseline.get(k, float('nan')))
            for k in METRIC_KEYS
        ]
        method_label = (f'{r["mod_id"]}: {r["mod_name"]} '
                        f'({r["max_iters"] // 1000}k iters)')
        lines.append(method_label + ' & ' + ' & '.join(cells) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--output-dir', default='my_figures/tables',
                        help='Where to write the .tex files.')
    parser.add_argument('--baseline-aacc', type=float, default=DEFAULT_BASELINE['aAcc'])
    parser.add_argument('--baseline-miou', type=float, default=DEFAULT_BASELINE['mIoU'])
    parser.add_argument('--baseline-macc', type=float, default=DEFAULT_BASELINE['mAcc'])
    parser.add_argument('--baseline-bf3', type=float, default=DEFAULT_BASELINE['mBoundaryF1_3'])
    parser.add_argument('--baseline-bf5', type=float, default=DEFAULT_BASELINE['mBoundaryF1_5'])
    parser.add_argument('--baseline-bf9', type=float, default=DEFAULT_BASELINE['mBoundaryF1_9'])
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = (repo / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_cfg in RUNS:
        wd = repo / run_cfg['work_dir']
        metrics = get_final_eval(wd)
        if metrics is None:
            print(f'  [WARN] {run_cfg["mod_id"]}: no eval rows in {wd}; skipping')
            continue
        rows.append({
            'mod_id': run_cfg['mod_id'],
            'mod_name': run_cfg['mod_name'],
            'work_dir': str(wd),
            'max_iters': run_cfg['max_iters'],
            'metrics': metrics,
        })

    baseline = {
        'aAcc': args.baseline_aacc,
        'mIoU': args.baseline_miou,
        'mAcc': args.baseline_macc,
        'mBoundaryF1_3': args.baseline_bf3,
        'mBoundaryF1_5': args.baseline_bf5,
        'mBoundaryF1_9': args.baseline_bf9,
    }

    abs_tex = build_absolute_table(rows)
    delta_tex = build_delta_table(rows, baseline)

    abs_path = out_dir / 'absolute_scores.tex'
    delta_path = out_dir / 'delta_vs_baseline.tex'
    abs_path.write_text(abs_tex + '\n')
    delta_path.write_text(delta_tex + '\n')

    print('=' * 78)
    print('Final eval rows used (last eval line in each run):')
    for r in rows:
        print(f'  {r["mod_id"]} ({r["max_iters"] // 1000}k): step={r["metrics"]["_step"]}')
        for k in METRIC_KEYS:
            print(f'    {k:>16} = {r["metrics"].get(k, float("nan")):.4f}')

    print()
    print('=' * 78)
    print('TABLE 1: absolute scores (LaTeX)')
    print('=' * 78)
    print(abs_tex)
    print()
    print('=' * 78)
    print('TABLE 2: delta vs baseline (LaTeX)')
    print('=' * 78)
    print(delta_tex)
    print()
    print(f'Wrote {abs_path.relative_to(repo)}')
    print(f'Wrote {delta_path.relative_to(repo)}')


if __name__ == '__main__':
    main()
