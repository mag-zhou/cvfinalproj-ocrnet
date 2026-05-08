#!/usr/bin/env python3
"""
Generate side-by-side segmentation comparison figures for a report.
Output: Input Image | Ground Truth | FCN+ResNet-50 | OCRNet+ResNet-50

Usage (on cluster):
    # Random sample of 8 val images
    python visualize_comparison.py \
        --fcn-config configs/fcn/fcn_r50-d8_1xb8-40k_ade20k-512x512-20pct.py \
        --fcn-ckpt "work_dirs/fcn_r50_ade20k_20pct/best_mIoU_*.pth" \
        --ocr-config configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-20pct.py \
        --ocr-ckpt work_dirs/ocrnet_r50_ade20k_20pct/best_mIoU_iter_36000.pth \
        --num-images 8 --seed 42 --output-dir report_figures

    # Specific images (pass stem names, no extension)
    python visualize_comparison.py ... --image-list ADE_val_00000001 ADE_val_00000042
"""

import argparse
import glob
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mmseg.apis import inference_model, init_model


def get_palette():
    from mmseg.datasets import ADE20KDataset
    return ADE20KDataset.METAINFO['palette']


def colorize_mask(mask, palette):
    """Convert integer class mask to RGB image using palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        rgb[mask == cls_idx] = color
    # Pixels with value >= 150 (ignore index) left as black
    return rgb


def load_gt_colored(ann_path, palette):
    """Load ADE20K annotation PNG and colorize it.

    ADE20K format: pixel value 0 = unlabeled/ignore, 1-150 = class index + 1.
    """
    gt = np.array(Image.open(ann_path)).astype(np.int32)
    cls_mask = gt - 1             # 0-149 = valid classes, -1 = ignore
    cls_mask[cls_mask < 0] = 255  # mark ignore regions
    cls_mask = cls_mask.astype(np.uint8)
    return colorize_mask(cls_mask, palette)


def resolve_glob(pattern):
    """Expand a glob pattern to a single file path, error if ambiguous."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    if len(matches) > 1:
        print(f"  Multiple checkpoints found, using latest: {matches[-1]}")
    return matches[-1]


def make_figure(img_rgb, gt_col, fcn_col, ocr_col, title, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    panels = [
        (img_rgb,  'Input Image'),
        (gt_col,   'Ground Truth'),
        (fcn_col,  'FCN  (ResNet-50)'),
        (ocr_col,  'OCRNet  (ResNet-50)'),
    ]
    for ax, (im, label) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=6)
        ax.axis('off')

    fig.suptitle(title, fontsize=10, y=1.005, color='#444444')
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fcn-config',
                        default='configs/fcn/fcn_r50-d8_1xb8-40k_ade20k-512x512-50pct-ext.py')
    parser.add_argument('--fcn-ckpt',
                        default='work_dirs/fcn_r50_ade20k_50pct_ext/iter_40000.pth',
                        help='Path or glob pattern, e.g. "work_dirs/fcn_*/best_mIoU_*.pth"')
    parser.add_argument('--ocr-config',
                        default='configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct-ext.py')
    parser.add_argument('--ocr-ckpt',
                        default='work_dirs/ocrnet_r50_ade20k_50pct_ext/iter_40000.pth')
    parser.add_argument('--data-root',  default='data/ade/ADEChallengeData2016')
    parser.add_argument('--num-images', type=int, default=40,
                        help='How many val images to sample')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--output-dir', default='report_figures')
    parser.add_argument('--image-list', nargs='*',
                        help='Specific image stems (e.g. ADE_val_00000001). '
                             'If given, --num-images is ignored.')
    parser.add_argument('--device',     default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve glob patterns for checkpoints
    fcn_ckpt = resolve_glob(args.fcn_ckpt)
    ocr_ckpt = resolve_glob(args.ocr_ckpt)
    print(f"FCN  checkpoint : {fcn_ckpt}")
    print(f"OCRNet checkpoint: {ocr_ckpt}")

    print("\nLoading FCN model...")
    model_fcn = init_model(args.fcn_config, fcn_ckpt, device=args.device)
    model_fcn.eval()

    print("Loading OCRNet model...")
    model_ocr = init_model(args.ocr_config, ocr_ckpt, device=args.device)
    model_ocr.eval()

    palette = get_palette()

    img_dir = os.path.join(args.data_root, 'images/validation')
    ann_dir = os.path.join(args.data_root, 'annotations/validation')

    all_imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))

    if args.image_list:
        selected = [f"{s}.jpg" for s in args.image_list if f"{s}.jpg" in all_imgs]
        if not selected:
            raise ValueError("None of the specified image names were found in the val set.")
    else:
        random.seed(args.seed)
        selected = random.sample(all_imgs, min(args.num_images, len(all_imgs)))

    print(f"\nGenerating figures for {len(selected)} images → {args.output_dir}/\n")

    for img_file in selected:
        stem = Path(img_file).stem
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, stem + '.png')

        print(f"  {stem} ...", end=' ', flush=True)

        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        gt_col  = load_gt_colored(ann_path, palette)

        with torch.no_grad():
            res_fcn = inference_model(model_fcn, img_path)
            res_ocr = inference_model(model_ocr, img_path)

        pred_fcn = res_fcn.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        pred_ocr = res_ocr.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

        fcn_col = colorize_mask(pred_fcn, palette)
        ocr_col = colorize_mask(pred_ocr, palette)

        save_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
        make_figure(img_rgb, gt_col, fcn_col, ocr_col,
                    title=f"ADE20K val  ·  {stem}", save_path=save_path)
        print("done")

    print(f"\nAll figures saved to ./{args.output_dir}/")
    print("\n--- Copy to local machine ---")
    print(f"scp -r ENGAGING:~/mmsegmentation/{args.output_dir}/ .")


if __name__ == '__main__':
    main()
