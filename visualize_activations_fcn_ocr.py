#!/usr/bin/env python3
"""Activation-map comparison: FCN vs OCRNet (both ResNet-50, 50% / 80k iters).

For each input image we run both segmentors and capture intermediate feature
maps via PyTorch forward hooks at matching depths (4 backbone stages +
auxiliary FCN pre-classifier + main decoder pre-classifier + OCRNet's
post-attention feats + final-prediction entropy). Each multi-channel feature
is reduced to a 2D heatmap (default L2-norm across channels) and overlaid on
a grayscale copy of the input.

Output per image: a single grid PNG with rows = layers, cols = [Input / Pred |
FCN | OCRNet]. A small ``*_legend.txt`` records the exact module path hooked
for each row.

Modeled on ``visualize_activations.py`` (the Mod1/2/3 boundary-variant
script).  No cv2 dependency — uses PIL + numpy for image I/O and resize.

Usage::

    sbatch visualize_activations_fcn_ocr_slurm.sh
    # or directly:
    python visualize_activations_fcn_ocr.py \
        --image-ids 229 286 502 1519 ...
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmseg.apis import inference_model, init_model


# ---------------------------------------------------------------------------
# Model registry: FCN baseline vs OCRNet baseline (both ResNet-50, 50% / 80k).
# ---------------------------------------------------------------------------
MODELS_TO_COMPARE = [
    {
        'model_id': 'FCN',
        'short':    'FCN\n(ResNet-50, 50% / 80k)',
        'config':   'configs/fcn/fcn_r50-d8_1xb8-40k_ade20k-512x512-50pct-ext.py',
        'ckpt':     'work_dirs/fcn_r50_ade20k_50pct_ext/iter_*.pth',
    },
    {
        'model_id': 'OCRNet',
        'short':    'OCRNet\n(ResNet-50, 50% / 80k)',
        'config':   'configs/ocrnet/ocrnet_r50-d8_1xb8-40k_ade20k-512x512-50pct-ext.py',
        'ckpt':     'work_dirs/ocrnet_r50_ade20k_50pct_ext/iter_*.pth',
    },
]

DEFAULT_IMAGE_IDS = [52, 229, 286, 458, 502, 564, 1000, 1310, 1519, 1850]


# ---------------------------------------------------------------------------
# Small utilities.
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


def latest_ckpt(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No checkpoint matched: {pattern}')

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


# ---------------------------------------------------------------------------
# Activation collection via forward hooks.
# ---------------------------------------------------------------------------
class ActivationCollector:
    def __init__(self, named_modules: Dict[str, torch.nn.Module]):
        self.acts: Dict[str, object] = {}
        self.handles = []
        for name, mod in named_modules.items():
            self.handles.append(mod.register_forward_hook(self._mk(name)))

    def _mk(self, name: str):
        def hook(_m, _inp, out):
            if isinstance(out, (list, tuple)):
                self.acts[name] = tuple(o.detach() for o in out)
            else:
                self.acts[name] = out.detach()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def named_modules_for(model, model_id: str) -> Dict[str, torch.nn.Module]:
    """Pick which submodules to hook for each segmentor.

    Both share the ResNet backbone (4 stages); after that the head topologies
    differ. We line up the closest analogues: FCN auxiliary <-> OCRNet
    decode_head[0] (FCN aux); FCN main convs <-> OCRNet bottleneck (OCR query
    feats); OCRNet alone has the post-attention block.
    """
    od: Dict[str, torch.nn.Module] = OrderedDict()
    od['backbone'] = model.backbone
    if model_id == 'FCN':
        od['aux_fcn_pre'] = model.auxiliary_head.convs
        od['decode_pre'] = model.decode_head.convs
    elif model_id == 'OCRNet':
        od['aux_fcn_pre'] = model.decode_head[0].convs
        od['decode_pre'] = model.decode_head[1].bottleneck
        od['ocr_post_attn'] = model.decode_head[1].object_context_block
    else:
        raise ValueError(f'Unknown model_id: {model_id}')
    return od


# ---------------------------------------------------------------------------
# Channel reduction + heatmap overlay.
# ---------------------------------------------------------------------------
def reduce2d(t: torch.Tensor, mode: str = 'l2', topk: int = 8) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    elif t.dim() != 3:
        raise ValueError(f'expected 3D or 4D tensor, got shape {tuple(t.shape)}')
    t = t.float()
    if mode == 'l2':
        m = torch.sqrt((t ** 2).sum(dim=0))
    elif mode == 'mean':
        m = t.abs().mean(dim=0)
    elif mode == 'max':
        m = t.abs().amax(dim=0)
    elif mode == 'topk':
        ch_score = t.abs().mean(dim=(1, 2))
        k = min(topk, ch_score.numel())
        idx = torch.topk(ch_score, k).indices
        m = t[idx].abs().mean(dim=0)
    else:
        raise ValueError(f'unknown reduce mode: {mode}')
    return m.cpu().numpy().astype(np.float32)


def normalize01(arr: np.ndarray,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None) -> np.ndarray:
    arr = arr.astype(np.float32)
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


def resize_bilinear(arr2d: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize a 2D float array to (W, H) using PIL."""
    img = Image.fromarray(arr2d.astype(np.float32), mode='F')
    img = img.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def overlay_heatmap(img_rgb: np.ndarray,
                    heat2d: np.ndarray,
                    alpha: float = 0.55,
                    cmap_name: str = 'inferno') -> np.ndarray:
    H, W = img_rgb.shape[:2]
    heat = np.clip(heat2d, 0.0, 1.0)
    heat_resized = resize_bilinear(heat, (W, H))
    cmap = mpl.colormaps[cmap_name]
    heat_rgb = (cmap(heat_resized)[..., :3] * 255).astype(np.uint8)
    gray = (0.2989 * img_rgb[..., 0]
            + 0.5870 * img_rgb[..., 1]
            + 0.1140 * img_rgb[..., 2]).astype(np.uint8)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    return (alpha * heat_rgb + (1.0 - alpha) * gray3).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-model inference + activation capture.
# ---------------------------------------------------------------------------
def run_one_model(model, img_path: str, model_id: str
                  ) -> Tuple[Dict[str, object], torch.Tensor, np.ndarray]:
    hooks = named_modules_for(model, model_id)
    coll = ActivationCollector(hooks)
    try:
        with torch.no_grad():
            result = inference_model(model, img_path)
            # FCN's auxiliary_head is not invoked during inference (only during
            # training). Trigger it manually so we can capture aux_fcn_pre.
            if model_id == 'FCN' and getattr(model, 'auxiliary_head', None) is not None:
                bb_out = coll.acts.get('backbone')
                if bb_out is not None:
                    _ = model.auxiliary_head(bb_out)
    finally:
        coll.close()
    seg_logits = result.seg_logits.data
    pred_mask = result.pred_sem_seg.data.squeeze().to(torch.uint8).cpu().numpy()
    return coll.acts, seg_logits, pred_mask


def entropy_map(seg_logits: torch.Tensor) -> np.ndarray:
    p = F.softmax(seg_logits.float(), dim=0)
    ent = -(p * p.clamp(min=1e-12).log()).sum(dim=0)
    return ent.cpu().numpy().astype(np.float32)


def per_layer_heat_for_model(acts: Dict[str, object],
                             seg_logits: torch.Tensor,
                             model_id: str,
                             reduce: str,
                             topk: int) -> 'OrderedDict[str, Optional[np.ndarray]]':
    out: 'OrderedDict[str, Optional[np.ndarray]]' = OrderedDict()
    bb = acts['backbone']
    out['backbone.layer1'] = reduce2d(bb[0], mode=reduce, topk=topk)
    out['backbone.layer2'] = reduce2d(bb[1], mode=reduce, topk=topk)
    out['backbone.layer3'] = reduce2d(bb[2], mode=reduce, topk=topk)
    out['backbone.layer4'] = reduce2d(bb[3], mode=reduce, topk=topk)
    out['aux_fcn_pre'] = (reduce2d(acts['aux_fcn_pre'], mode=reduce, topk=topk)
                          if 'aux_fcn_pre' in acts else None)
    out['decode_pre'] = (reduce2d(acts['decode_pre'], mode=reduce, topk=topk)
                         if 'decode_pre' in acts else None)
    if model_id == 'OCRNet' and 'ocr_post_attn' in acts:
        out['ocr_post_attn'] = reduce2d(acts['ocr_post_attn'], mode=reduce, topk=topk)
    else:
        out['ocr_post_attn'] = None
    out['pred_entropy'] = entropy_map(seg_logits)
    return out


ROW_LABELS = OrderedDict([
    ('backbone.layer1',  'Backbone\nlayer1\n(stride 4)'),
    ('backbone.layer2',  'Backbone\nlayer2\n(stride 8)'),
    ('backbone.layer3',  'Backbone\nlayer3\n(dilated)'),
    ('backbone.layer4',  'Backbone\nlayer4\n(dilated)'),
    ('aux_fcn_pre',      'Auxiliary FCN\npre-classifier'),
    ('decode_pre',       'Main decoder\npre-classifier'),
    ('ocr_post_attn',    'OCR head\npost object-attn\n(OCRNet only)'),
    ('pred_entropy',     'Final pred\nsoftmax entropy'),
])

ROW_DESCRIPTIONS = OrderedDict([
    ('backbone.layer1', 'model.backbone -> stage 0 (ResNetV1c layer1, 256 ch)'),
    ('backbone.layer2', 'model.backbone -> stage 1 (ResNetV1c layer2, 512 ch)'),
    ('backbone.layer3', 'model.backbone -> stage 2 (ResNetV1c layer3, 1024 ch, dilation=2)'),
    ('backbone.layer4', 'model.backbone -> stage 3 (ResNetV1c layer4, 2048 ch, dilation=4)'),
    ('aux_fcn_pre',     'FCN: model.auxiliary_head.convs.  OCRNet: model.decode_head[0].convs.'),
    ('decode_pre',      'FCN: model.decode_head.convs (main FCN pre-cls).  '
                        'OCRNet: model.decode_head[1].bottleneck (OCR query feats).'),
    ('ocr_post_attn',   'OCRNet: model.decode_head[1].object_context_block (post object-attn). '
                        'Not applicable for FCN.'),
    ('pred_entropy',    '-sum(p log p) over softmax of final cls_seg logits.'),
])


# ---------------------------------------------------------------------------
# Figure assembly.
# ---------------------------------------------------------------------------
def make_figure(img_rgb: np.ndarray,
                pred_cols: 'list[np.ndarray]',
                per_model_heats: 'list[OrderedDict[str, Optional[np.ndarray]]]',
                model_labels: 'list[str]',
                title: str,
                save_path: str,
                alpha: float,
                shared_row_norm: bool,
                cmap_name: str = 'inferno') -> None:
    n_models = len(per_model_heats)
    n_cols = 1 + n_models
    n_layers = len(ROW_LABELS)
    n_rows = 1 + n_layers

    panel = 4.0
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel * n_cols, panel * n_rows),
                             squeeze=False)

    # Header row.
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Input image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    for j, (pred_col, lbl) in enumerate(zip(pred_cols, model_labels)):
        ax = axes[0, 1 + j]
        ax.imshow(pred_col)
        ax.set_title(f'{lbl}\n(predicted segmentation)',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    # Activation rows.
    for i, key in enumerate(ROW_LABELS.keys(), start=1):
        axes[i, 0].axis('off')
        axes[i, 0].text(
            0.5, 0.5, ROW_LABELS[key],
            ha='center', va='center', fontsize=12, fontweight='bold',
            transform=axes[i, 0].transAxes)

        row_vmin: Optional[float] = None
        row_vmax: Optional[float] = None
        if shared_row_norm:
            present = [h[key] for h in per_model_heats if h.get(key) is not None]
            if present:
                row_vmin = float(min(p.min() for p in present))
                row_vmax = float(max(p.max() for p in present))

        for j, heat_dict in enumerate(per_model_heats):
            ax = axes[i, 1 + j]
            heat = heat_dict.get(key)
            if heat is None:
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        fontsize=14, color='#888888',
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('#cccccc')
                continue
            heat01 = normalize01(heat, vmin=row_vmin, vmax=row_vmax)
            blended = overlay_heatmap(img_rgb, heat01, alpha=alpha,
                                      cmap_name=cmap_name)
            ax.imshow(blended)
            ax.axis('off')

    fig.suptitle(title, fontsize=12, color='#444444', y=0.998)
    plt.tight_layout(rect=(0, 0, 1, 0.995))
    plt.savefig(save_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def write_legend(legend_path: str,
                 reduce: str,
                 topk: int,
                 alpha: float,
                 shared_row_norm: bool,
                 chosen_ckpts: 'list[str]') -> None:
    lines = []
    lines.append('Activation-map figure legend (FCN vs OCRNet)')
    lines.append('=' * 60)
    lines.append(f'channel reduction : {reduce}'
                 + (f' (top-{topk})' if reduce == 'topk' else ''))
    lines.append(f'overlay alpha     : {alpha}')
    lines.append(f'per-row shared norm: {shared_row_norm}')
    lines.append('')
    lines.append('Checkpoints:')
    for cfg, ck in zip(MODELS_TO_COMPARE, chosen_ckpts):
        lines.append(f'  {cfg["model_id"]:<7}: {ck}')
    lines.append('')
    lines.append('Rows (top to bottom):')
    for i, (key, desc) in enumerate(ROW_DESCRIPTIONS.items(), start=1):
        lines.append(f'  {i}. {ROW_LABELS[key].replace(chr(10), " "):<40}'
                     f' -- {desc}')
    Path(legend_path).write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/ade/ADEChallengeData2016')
    parser.add_argument('--image-ids', type=int, nargs='*',
                        default=DEFAULT_IMAGE_IDS,
                        help='ADE20K val image numeric IDs (e.g. 229 286 502).')
    parser.add_argument('--output-dir', default='my_figures/activations_fcn_ocr')
    parser.add_argument('--reduce', default='l2',
                        choices=['l2', 'mean', 'max', 'topk'])
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--cmap', default='inferno')
    parser.add_argument('--shared-row-norm', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Resolving checkpoints:')
    chosen_ckpts = []
    for cfg in MODELS_TO_COMPARE:
        ck = latest_ckpt(cfg['ckpt'])
        chosen_ckpts.append(ck)
        print(f'  {cfg["model_id"]} -> {ck}')

    print('\nLoading models:')
    models = []
    for cfg, ck in zip(MODELS_TO_COMPARE, chosen_ckpts):
        print(f'  {cfg["model_id"]} ...')
        m = init_model(cfg['config'], ck, device=args.device)
        m.eval()
        models.append(m)

    palette = get_palette()
    img_dir = os.path.join(args.data_root, 'images/validation')
    ann_dir = os.path.join(args.data_root, 'annotations/validation')

    print(f'\nProcessing {len(args.image_ids)} images -> {args.output_dir}/\n')
    for img_id in args.image_ids:
        stem = f'ADE_val_{img_id:08d}'
        img_path = os.path.join(img_dir, stem + '.jpg')
        if not os.path.isfile(img_path):
            print(f'  {stem}: image not found, skipping')
            continue
        print(f'  {stem} ...', flush=True)

        img_rgb = np.array(Image.open(img_path).convert('RGB'))

        per_model_heats = []
        pred_cols = []
        for cfg, m in zip(MODELS_TO_COMPARE, models):
            print(f'    {cfg["model_id"]} ...', end=' ', flush=True)
            acts, seg_logits, pred_mask = run_one_model(m, img_path, cfg['model_id'])
            heats = per_layer_heat_for_model(
                acts, seg_logits, cfg['model_id'], args.reduce, args.topk)
            per_model_heats.append(heats)
            pred_cols.append(colorize_mask(pred_mask, palette))
            del acts, seg_logits, pred_mask
            torch.cuda.empty_cache()
            print('done')

        save_path = os.path.join(args.output_dir, f'{stem}_activations.png')
        title = (f'ADE20K val · {stem} · activation maps · '
                 f'reduce={args.reduce}'
                 + (f' (top-{args.topk})' if args.reduce == 'topk' else '')
                 + f', alpha={args.alpha}'
                 + (', shared-row-norm' if args.shared_row_norm else ''))
        make_figure(
            img_rgb=img_rgb,
            pred_cols=pred_cols,
            per_model_heats=per_model_heats,
            model_labels=[m['short'] for m in MODELS_TO_COMPARE],
            title=title,
            save_path=save_path,
            alpha=args.alpha,
            shared_row_norm=args.shared_row_norm,
            cmap_name=args.cmap,
        )
        print(f'    -> {save_path}')

    legend_path = os.path.join(args.output_dir, 'activations_legend.txt')
    write_legend(legend_path, args.reduce, args.topk, args.alpha,
                 args.shared_row_norm, chosen_ckpts)
    print(f'\nLegend -> {legend_path}')
    print('Done.')


if __name__ == '__main__':
    main()
