#!/usr/bin/env python3
"""
OCRNet-only segmentation visualizations for specific ADE20K val images.
Output per image: Input | Ground Truth | OCRNet prediction
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mmseg.apis import inference_model, init_model


DEFAULT_IMAGE_IDS = [229, 286, 502, 1519]


def get_palette():
    from mmseg.datasets import ADE20KDataset
    return ADE20KDataset.METAINFO['palette']


def colorize_mask(mask, palette):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(palette):
        rgb[mask == cls_idx] = color
    return rgb


def load_gt_colored(ann_path, palette):
    gt = np.array(Image.open(ann_path)).astype(np.int32)
    cls_mask = gt - 1
    cls_mask[cls_mask < 0] = 255
    return colorize_mask(cls_mask.astype(np.uint8), palette)


def resolve_glob(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found matching: {pattern}")
    if len(matches) > 1:
        print(f"  Multiple checkpoints found, using latest: {matches[-1]}")
    return matches[-1]


def make_figure(img_rgb, gt_col, ocr_col, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (img_rgb, 'Input Image'),
        (gt_col,  'Ground Truth'),
        (ocr_col, 'OCRNet  (ResNet-50)'),
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
    parser.add_argument('--ocr-config',
                        default='configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct-ext.py')
    parser.add_argument('--ocr-ckpt',
                        default='work_dirs/ocrnet_r50_ade20k_50pct_ext/iter_40000.pth')
    parser.add_argument('--data-root', default='data/ade/ADEChallengeData2016')
    parser.add_argument('--image-ids', type=int, nargs='*', default=DEFAULT_IMAGE_IDS,
                        help='ADE20K val image IDs (e.g. 229 286 502 1519)')
    parser.add_argument('--output-dir', default='report_figures_ocrnet')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ocr_ckpt = resolve_glob(args.ocr_ckpt)
    print(f"OCRNet checkpoint: {ocr_ckpt}")

    print("Loading OCRNet model...")
    model_ocr = init_model(args.ocr_config, ocr_ckpt, device=args.device)
    model_ocr.eval()

    palette = get_palette()
    img_dir = os.path.join(args.data_root, 'images/validation')
    ann_dir = os.path.join(args.data_root, 'annotations/validation')

    print(f"\nGenerating figures for {len(args.image_ids)} images -> {args.output_dir}/\n")

    for img_id in args.image_ids:
        stem = f"ADE_val_{img_id:08d}"
        img_path = os.path.join(img_dir, stem + '.jpg')
        ann_path = os.path.join(ann_dir, stem + '.png')

        if not os.path.isfile(img_path):
            print(f"  {stem}: image not found, skipping")
            continue

        print(f"  {stem} ...", end=' ', flush=True)

        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        gt_col = load_gt_colored(ann_path, palette)

        with torch.no_grad():
            res_ocr = inference_model(model_ocr, img_path)
        pred_ocr = res_ocr.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        ocr_col = colorize_mask(pred_ocr, palette)

        save_path = os.path.join(args.output_dir, f"{stem}_ocrnet.png")
        make_figure(img_rgb, gt_col, ocr_col,
                    title=f"ADE20K val  -  {stem}", save_path=save_path)
        print("done")

    print(f"\nAll figures saved to ./{args.output_dir}/")


if __name__ == '__main__':
    main()
