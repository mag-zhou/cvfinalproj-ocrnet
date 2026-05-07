#!/usr/bin/env python3
"""Generate side-by-side example outputs for the mod3 (modulated OCR + aux
boundary) variant trained on 20% and 50% of ADE20K.

Layout per image:  Input | Ground Truth | mod3 (20%) | mod3 (50%)

Usage (on cluster, GPU node):
    python visualize_mod3.py
    # or override defaults
    python visualize_mod3.py --num-images 12 --seed 7 \
        --output-dir my_figures/mod3_20vs50pct_more
    # or pick specific stems
    python visualize_mod3.py --image-list ADE_val_00000052 ADE_val_00000210
"""

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


def get_palette():
    from mmseg.datasets import ADE20KDataset
    return ADE20KDataset.METAINFO['palette']


def colorize_mask(mask, palette):
    """Convert integer class mask to RGB image using the ADE20K palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        rgb[mask == cls_idx] = color
    return rgb


def load_gt_colored(ann_path, palette):
    """Load an ADE20K annotation PNG and colorize it.

    ADE20K format: pixel value 0 = unlabeled/ignore, 1-150 = class index + 1.
    """
    gt = np.array(Image.open(ann_path)).astype(np.int32)
    cls_mask = gt - 1
    cls_mask[cls_mask < 0] = 255
    cls_mask = cls_mask.astype(np.uint8)
    return colorize_mask(cls_mask, palette)


def resolve_glob(pattern):
    """Expand a glob pattern to a single file path; pick the latest match."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    if len(matches) > 1:
        print(f"  Multiple checkpoints match, using latest: {matches[-1]}")
    return matches[-1]


def strip_boundary_pipeline(model):
    """Replace the model's test pipeline with a boundary-free one.

    The mod3 config ships a test pipeline that contains
    ``LoadBoundaryAnnotations`` and ``PackSegBoundaryInputs``. Those expect
    ``seg_map_path`` (so they can find the boundary PNG), but
    ``mmseg.apis.inference_model`` only sets ``img_path`` per call and only
    strips ``LoadAnnotations``. For visualization we don't need the boundary
    GT, so we override the pipeline in place.
    """
    model.cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 512), keep_ratio=True),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='PackSegInputs'),
    ]


def make_figure(img_rgb, gt_col, pred_20_col, pred_50_col, title, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    panels = [
        (img_rgb,     'Input Image'),
        (gt_col,      'Ground Truth'),
        (pred_20_col, 'mod3  (20%)'),
        (pred_50_col, 'mod3  (50%)'),
    ]
    for ax, (im, label) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=6)
        ax.axis('off')

    fig.suptitle(title, fontsize=10, y=1.005, color='#444444')
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def predict_mask(model, img_path):
    with torch.no_grad():
        result = inference_model(model, img_path)
    return result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-20pct',
        default='configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr.py')
    parser.add_argument(
        '--ckpt-20pct',
        default='work_dirs/ocrnet_r50_mod3_modulated_ocr_20pct/iter_40000.pth',
        help='Path or glob pattern.')
    parser.add_argument(
        '--config-50pct',
        default='configs/ocrnet/boundary/ocrnet_r50_mod3_modulated_ocr_50pct.py')
    parser.add_argument(
        '--ckpt-50pct',
        default='work_dirs/ocrnet_r50_mod3_modulated_ocr_50pct/iter_40000.pth')
    parser.add_argument('--data-root', default='data/ade/ADEChallengeData2016')
    parser.add_argument('--num-images', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='my_figures/mod3_20vs50pct')
    parser.add_argument('--image-list', nargs='*',
                        help='Specific image stems (e.g. ADE_val_00000001). '
                             'If given, --num-images is ignored.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_20 = resolve_glob(args.ckpt_20pct)
    ckpt_50 = resolve_glob(args.ckpt_50pct)
    print(f"mod3 20% checkpoint: {ckpt_20}")
    print(f"mod3 50% checkpoint: {ckpt_50}")

    print("\nLoading mod3 (20%) model...")
    model_20 = init_model(args.config_20pct, ckpt_20, device=args.device)
    model_20.eval()
    strip_boundary_pipeline(model_20)

    print("Loading mod3 (50%) model...")
    model_50 = init_model(args.config_50pct, ckpt_50, device=args.device)
    model_50.eval()
    strip_boundary_pipeline(model_50)

    palette = get_palette()

    img_dir = os.path.join(args.data_root, 'images/validation')
    ann_dir = os.path.join(args.data_root, 'annotations/validation')

    all_imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))

    if args.image_list:
        selected = [f"{s}.jpg" for s in args.image_list if f"{s}.jpg" in all_imgs]
        missing = [s for s in args.image_list if f"{s}.jpg" not in all_imgs]
        if missing:
            print(f"Warning: not found in val set: {missing}")
        if not selected:
            raise ValueError("None of the specified image names were found in the val set.")
    else:
        random.seed(args.seed)
        selected = random.sample(all_imgs, min(args.num_images, len(all_imgs)))

    print(f"\nGenerating figures for {len(selected)} images -> {args.output_dir}/\n")

    for img_file in selected:
        stem = Path(img_file).stem
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, stem + '.png')

        print(f"  {stem} ...", end=' ', flush=True)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_col = load_gt_colored(ann_path, palette)

        pred_20 = predict_mask(model_20, img_path)
        pred_50 = predict_mask(model_50, img_path)

        pred_20_col = colorize_mask(pred_20, palette)
        pred_50_col = colorize_mask(pred_50, palette)

        save_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
        make_figure(
            img_rgb, gt_col, pred_20_col, pred_50_col,
            title=f"ADE20K val  ·  {stem}  ·  mod3 (modulated OCR + aux boundary)",
            save_path=save_path,
        )
        print("done")

    print(f"\nAll figures saved to ./{args.output_dir}/")


if __name__ == '__main__':
    main()
