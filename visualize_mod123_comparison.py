#!/usr/bin/env python3
"""Side-by-side qualitative comparison: Input | GT | Mod 1 | Mod 2 | Mod 3.

Loads the latest checkpoint of each of Mod 1 (50% / 40k), Mod 2 (50% /
80k), Mod 3 (50% / 80k) and renders predictions for ``--num-images``
randomly-sampled ADE20K val images. Output is one PNG per image with
all five panels in a row.

This script must run on a GPU node (loads three full segmentors). On
the cluster::

    salloc -p mit_normal_gpu -t 1:00:00 --gres=gpu:h200:1 -c 4 --mem=32G
    python visualize_mod123_comparison.py
    # or with a specific seed / output dir:
    python visualize_mod123_comparison.py --seed 7 --output-dir my_figures/mod123_seed7

Modeled on visualize_mod3.py; the boundary-loading test pipeline must
be stripped in-place because mmseg's inference_model only auto-strips
LoadAnnotations and our boundary configs add LoadBoundaryAnnotations
on top.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mmseg.apis import inference_model, init_model


# ---------------------------------------------------------------------------
# Run registry. Mirrors the one in tools/plot_training_curves.py: each
# entry points at the config + checkpoint we want to render predictions
# from.
# ---------------------------------------------------------------------------
MODELS_TO_COMPARE = [
    {
        'mod_id': 'Mod 1',
        'short':  'Mod 1\n(aux boundary head)',
        'config': 'configs/ocrnet/boundary/ocrnet_r50_mod1_aux_boundary_50pct.py',
        # Glob: pick whatever's latest (typically iter_40000.pth but also
        # works after the warm-restart extension).
        'ckpt':   'work_dirs/ocrnet_r50_mod1_aux_boundary_50pct/iter_*.pth',
    },
    {
        'mod_id': 'Mod 2',
        'short':  'Mod 2\n(boundary-weighted CE)',
        'config': 'configs/ocrnet/boundary/ocrnet_r50_mod2_weighted_ce_50pct_80k.py',
        'ckpt':   'work_dirs/ocrnet_r50_mod2_weighted_ce_50pct_80k/iter_*.pth',
    },
    {
        'mod_id': 'Mod 3',
        'short':  'Mod 3\n(modulated OCR + aux)',
        'config': 'configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr_50pct_80k.py',
        'ckpt':   'work_dirs/ocrnet_r50_mod3_modulated_ocr_50pct_80k/iter_*.pth',
    },
]


# ---------------------------------------------------------------------------
# Data loading / colorisation (same recipe as visualize_mod3.py)
# ---------------------------------------------------------------------------

def get_palette():
    from mmseg.datasets import ADE20KDataset
    return ADE20KDataset.METAINFO['palette']


def colorize_mask(mask: np.ndarray, palette) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        rgb[mask == cls_idx] = color
    return rgb


def load_gt_colored(ann_path: str, palette) -> np.ndarray:
    """ADE20K annotation PNG: 0 = ignore, 1..150 = class index + 1."""
    gt = np.array(Image.open(ann_path)).astype(np.int32)
    cls_mask = gt - 1
    cls_mask[cls_mask < 0] = 255
    cls_mask = cls_mask.astype(np.uint8)
    return colorize_mask(cls_mask, palette)


def latest_ckpt(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No checkpoint matched: {pattern}')
    # Sort by iteration number embedded in 'iter_NNNN.pth' so iter_40000.pth
    # ends up after iter_8000.pth (alphabetic sort is wrong for that).
    def iter_key(p: str) -> int:
        try:
            return int(Path(p).stem.split('_')[-1])
        except ValueError:
            return -1
    matches.sort(key=iter_key)
    chosen = matches[-1]
    if len(matches) > 1:
        print(f'  Multiple ckpts; using latest: {chosen}')
    return chosen


def strip_boundary_pipeline(model) -> None:
    """Replace the test pipeline with a boundary-free one.

    Our boundary configs ship test pipelines that include
    LoadBoundaryAnnotations + PackSegBoundaryInputs. inference_model()
    in mmseg.apis sets only ``img_path`` per call and only auto-strips
    LoadAnnotations, so the boundary loaders would fail at predict
    time. We don't need them for visualisation -- override in place.
    """
    model.cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 512), keep_ratio=True),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='PackSegInputs'),
    ]


def predict_mask(model, img_path: str) -> np.ndarray:
    with torch.no_grad():
        result = inference_model(model, img_path)
    return result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Figure layout: 1 x 5 panels per image (Input | GT | Mod 1 | Mod 2 | Mod 3)
# ---------------------------------------------------------------------------

def make_figure(img_rgb, gt_col, preds, model_labels, title, save_path):
    n_models = len(preds)
    n_panels = 2 + n_models
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    panels = [(img_rgb, 'Input Image'), (gt_col, 'Ground Truth')]
    panels += list(zip(preds, model_labels))
    for ax, (im, label) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(label, fontsize=12, fontweight='bold', pad=6)
        ax.axis('off')
    fig.suptitle(title, fontsize=10, y=1.005, color='#444444')
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/ade/ADEChallengeData2016')
    parser.add_argument('--num-images', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='my_figures/mod1_vs_mod2_vs_mod3')
    parser.add_argument('--image-list', nargs='*',
                        help='Specific stems (e.g. ADE_val_00000052). '
                             'If given, --num-images / --seed are ignored.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Resolving checkpoints:')
    chosen_ckpts = []
    for cfg in MODELS_TO_COMPARE:
        ck = latest_ckpt(cfg['ckpt'])
        chosen_ckpts.append(ck)
        print(f'  {cfg["mod_id"]} -> {ck}')

    print('\nLoading models:')
    models = []
    for cfg, ck in zip(MODELS_TO_COMPARE, chosen_ckpts):
        print(f'  {cfg["mod_id"]} ...')
        m = init_model(cfg['config'], ck, device=args.device)
        m.eval()
        strip_boundary_pipeline(m)
        models.append(m)

    palette = get_palette()
    img_dir = os.path.join(args.data_root, 'images/validation')
    ann_dir = os.path.join(args.data_root, 'annotations/validation')
    all_imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))

    if args.image_list:
        selected = [f'{s}.jpg' for s in args.image_list if f'{s}.jpg' in all_imgs]
        missing = [s for s in args.image_list if f'{s}.jpg' not in all_imgs]
        if missing:
            print(f'  Warning: not found in val set: {missing}')
        if not selected:
            raise SystemExit('None of --image-list found in val set.')
    else:
        random.seed(args.seed)
        selected = random.sample(all_imgs, min(args.num_images, len(all_imgs)))

    print(f'\nGenerating {len(selected)} figures -> {args.output_dir}/\n')

    model_labels = [m['short'] for m in MODELS_TO_COMPARE]

    for img_file in selected:
        stem = Path(img_file).stem
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, stem + '.png')

        print(f'  {stem} ...', end=' ', flush=True)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_col = load_gt_colored(ann_path, palette)

        preds = []
        for m in models:
            mask = predict_mask(m, img_path)
            preds.append(colorize_mask(mask, palette))

        save_path = os.path.join(args.output_dir, f'{stem}_comparison.png')
        title = (f'ADE20K val  ·  {stem}  ·  Mod 1 / Mod 2 / Mod 3 '
                 f'(50% data, latest checkpoints)')
        make_figure(img_rgb, gt_col, preds, model_labels, title, save_path)
        print('done')

    print(f'\nAll figures saved to ./{args.output_dir}/')


if __name__ == '__main__':
    main()
