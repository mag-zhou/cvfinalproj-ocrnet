#!/usr/bin/env python3
"""Activation-map comparison: Mod 1 / Mod 2 / Mod 3 on a single ADE20K val image.

For one input image the script runs each of the three boundary-modified OCRNet
variants and captures intermediate feature maps via PyTorch forward hooks at
matching depths (4 backbone stages + FCN pre-classifier + OCR query / OCR
pre-classifier features), plus per-mod boundary signals (Mod 1 auxiliary
boundary, Mod 3 temperature map T(x), Mod 3 auxiliary boundary). Each
multi-channel feature is reduced to a 2D heatmap (default L2-norm across
channels) and overlaid on a grayscale copy of the input.

The output is a single grid PNG: rows = layers, cols = [Input | Mod 1 | Mod 2 |
Mod 3], plus a header row with the colored predictions. A small ``*_legend.txt``
records the exact module path hooked for each row.

Modeled on ``visualize_mod123_comparison.py``. Must run on a GPU node (loads
three full segmentors). On the cluster::

    salloc -p mit_normal_gpu -t 0:30:00 --gres=gpu:h200:1 -c 4 --mem=24G
    python visualize_activations.py --image ADE_val_00000229
    # or via the sbatch wrapper:
    sbatch visualize_activations_slurm.sh
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmseg.apis import inference_model, init_model


# ---------------------------------------------------------------------------
# Run registry. Mirrors visualize_mod123_comparison.py.
# ---------------------------------------------------------------------------
MODELS_TO_COMPARE = [
    {
        'mod_id': 'Mod 1',
        'short':  'Mod 1\n(aux boundary head)',
        'config': 'configs/ocrnet/boundary/ocrnet_r50_mod1_aux_boundary_50pct.py',
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


def strip_boundary_pipeline(model) -> None:
    """Replace the test pipeline with a boundary-free one (see comparison script)."""
    model.cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 512), keep_ratio=True),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='PackSegInputs'),
    ]


# ---------------------------------------------------------------------------
# Activation collection via forward hooks.
# ---------------------------------------------------------------------------
class ActivationCollector:
    """Registers forward hooks on the supplied named modules, stores last output
    tensor per name. Tensors are kept on whatever device the module ran on; the
    caller is responsible for moving to CPU when done."""

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


def named_modules_for(model, mod_id: str) -> Dict[str, torch.nn.Module]:
    """Modules to hook for a given mod. ``backbone`` captures the full tuple of
    4 stage outputs (we use this both for layer1-4 visualization and to feed
    Mod 1's auxiliary head manually post-inference)."""
    od: Dict[str, torch.nn.Module] = OrderedDict()
    od['backbone'] = model.backbone
    od['fcn_pre'] = model.decode_head[0].convs
    od['ocr_bottleneck'] = model.decode_head[1].bottleneck
    od['ocr_pre'] = model.decode_head[1].object_context_block
    if mod_id == 'Mod 3':
        od['mod3_temp_logits'] = model.decode_head[1].temp_logit_conv
        od['mod3_aux_bd'] = model.decode_head[1].aux_boundary_conv
    return od


# ---------------------------------------------------------------------------
# Channel reduction + heatmap overlay.
# ---------------------------------------------------------------------------
def reduce2d(t: torch.Tensor, mode: str = 'l2', topk: int = 8) -> np.ndarray:
    """Multi-channel feature -> 2D map. Accepts (B, C, H, W) or (C, H, W)."""
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


def overlay_heatmap(img_rgb: np.ndarray,
                    heat2d: np.ndarray,
                    alpha: float = 0.55,
                    cmap_name: str = 'inferno') -> np.ndarray:
    """Resize heat (assumed in [0, 1]) to img size, colormap, alpha-blend on
    grayscale of input. Returns HxWx3 uint8."""
    H, W = img_rgb.shape[:2]
    heat = np.clip(heat2d, 0.0, 1.0).astype(np.float32)
    heat_resized = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    cmap = mpl.colormaps[cmap_name]
    heat_rgb = (cmap(heat_resized)[..., :3] * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    blended = (alpha * heat_rgb + (1.0 - alpha) * gray3).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Per-model inference + activation capture.
# ---------------------------------------------------------------------------
def run_one_model(model, img_path: str, mod_id: str
                  ) -> Tuple[Dict[str, object], torch.Tensor, np.ndarray]:
    """Run inference, capture activations, optionally compute Mod 1 aux boundary.

    Returns ``(acts, seg_logits, pred_mask)`` where ``acts`` is a dict of
    raw captured tensors (still on GPU), ``seg_logits`` is (C, H, W) at input
    resolution, and ``pred_mask`` is the argmax (H, W) uint8 mask.
    """
    hooks = named_modules_for(model, mod_id)
    coll = ActivationCollector(hooks)
    try:
        with torch.no_grad():
            result = inference_model(model, img_path)
            if (mod_id == 'Mod 1'
                    and getattr(model, 'auxiliary_head', None) is not None):
                bb_out = coll.acts.get('backbone')
                if bb_out is not None:
                    aux_logits = model.auxiliary_head(bb_out)
                    coll.acts['mod1_aux_bd'] = aux_logits.detach()
    finally:
        coll.close()

    seg_logits = result.seg_logits.data
    pred_mask = result.pred_sem_seg.data.squeeze().to(torch.uint8).cpu().numpy()
    return coll.acts, seg_logits, pred_mask


# ---------------------------------------------------------------------------
# Row spec: which captured tensor to show per row, and how to reduce it.
# ---------------------------------------------------------------------------
def entropy_map(seg_logits: torch.Tensor) -> np.ndarray:
    """Per-pixel softmax entropy of a (C, H, W) tensor, in nats."""
    p = F.softmax(seg_logits.float(), dim=0)
    ent = -(p * p.clamp(min=1e-12).log()).sum(dim=0)
    return ent.cpu().numpy().astype(np.float32)


def mod3_T_from_logits(t: torch.Tensor,
                       temp_beta: float = 2.0,
                       temp_min: float = 0.5,
                       temp_max: float = 5.0) -> np.ndarray:
    """Reconstruct T(x) from temp_logit_conv output (matches OCRBoundaryHead)."""
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0]
    T = (1.0 + temp_beta * torch.sigmoid(t.float())).clamp(temp_min, temp_max)
    return T.cpu().numpy().astype(np.float32)


def sigmoid_single_channel(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0]
    return torch.sigmoid(t.float()).cpu().numpy().astype(np.float32)


def per_layer_heat_for_mod(acts: Dict[str, object],
                           seg_logits: torch.Tensor,
                           mod_id: str,
                           reduce: str,
                           topk: int) -> 'OrderedDict[str, Optional[np.ndarray]]':
    """Per-layer (un-normalized) 2D maps for a single model. Returns an
    OrderedDict in display order. Values are float32 numpy arrays (no
    normalization yet), or ``None`` if the layer is not applicable for this
    mod."""
    out: 'OrderedDict[str, Optional[np.ndarray]]' = OrderedDict()
    bb = acts['backbone']
    out['backbone.layer1'] = reduce2d(bb[0], mode=reduce, topk=topk)
    out['backbone.layer2'] = reduce2d(bb[1], mode=reduce, topk=topk)
    out['backbone.layer3'] = reduce2d(bb[2], mode=reduce, topk=topk)
    out['backbone.layer4'] = reduce2d(bb[3], mode=reduce, topk=topk)
    out['fcn_pre'] = reduce2d(acts['fcn_pre'], mode=reduce, topk=topk)
    out['ocr_bottleneck'] = reduce2d(acts['ocr_bottleneck'], mode=reduce, topk=topk)
    out['ocr_pre'] = reduce2d(acts['ocr_pre'], mode=reduce, topk=topk)
    out['pred_entropy'] = entropy_map(seg_logits)

    if mod_id == 'Mod 1' and 'mod1_aux_bd' in acts:
        out['aux_boundary'] = sigmoid_single_channel(acts['mod1_aux_bd'])
    elif mod_id == 'Mod 3' and 'mod3_aux_bd' in acts:
        out['aux_boundary'] = sigmoid_single_channel(acts['mod3_aux_bd'])
    else:
        out['aux_boundary'] = None

    if mod_id == 'Mod 3' and 'mod3_temp_logits' in acts:
        out['mod3_T_map'] = mod3_T_from_logits(acts['mod3_temp_logits'])
    else:
        out['mod3_T_map'] = None
    return out


ROW_LABELS = OrderedDict([
    ('backbone.layer1', 'Backbone\nlayer1\n(stride 4)'),
    ('backbone.layer2', 'Backbone\nlayer2\n(stride 8)'),
    ('backbone.layer3', 'Backbone\nlayer3\n(dilated)'),
    ('backbone.layer4', 'Backbone\nlayer4\n(dilated)'),
    ('fcn_pre',         'FCN head\npre-classifier'),
    ('ocr_bottleneck',  'OCR head\nbottleneck (feats)'),
    ('ocr_pre',         'OCR head\npre-classifier\n(post object-attn)'),
    ('pred_entropy',    'Final pred\nsoftmax entropy'),
    ('aux_boundary',    'Auxiliary\nboundary sigmoid'),
    ('mod3_T_map',      'Mod 3\ntemperature T(x)'),
])

ROW_DESCRIPTIONS = OrderedDict([
    ('backbone.layer1', 'model.backbone -> stage 0 output (ResNetV1c layer1, 256 ch)'),
    ('backbone.layer2', 'model.backbone -> stage 1 output (ResNetV1c layer2, 512 ch)'),
    ('backbone.layer3', 'model.backbone -> stage 2 output (ResNetV1c layer3, 1024 ch, dilation=2)'),
    ('backbone.layer4', 'model.backbone -> stage 3 output (ResNetV1c layer4, 2048 ch, dilation=4)'),
    ('fcn_pre',         'model.decode_head[0].convs output (FCN pre-classifier, 256 ch)'),
    ('ocr_bottleneck',  'model.decode_head[1].bottleneck output (OCR query feats, 512 ch)'),
    ('ocr_pre',         'model.decode_head[1].object_context_block output (post object-attn, 512 ch)'),
    ('pred_entropy',    '-sum(p log p) over softmax of model.decode_head[1].cls_seg output, at input resolution'),
    ('aux_boundary',    'Mod 1: sigmoid(model.auxiliary_head(backbone)). Mod 2: not applicable. Mod 3: sigmoid(model.decode_head[1].aux_boundary_conv).'),
    ('mod3_T_map',      'Mod 3: T(x) = clip(1 + beta * sigmoid(model.decode_head[1].temp_logit_conv), Tmin, Tmax). Not applicable for Mod 1/2.'),
])


# ---------------------------------------------------------------------------
# Figure assembly.
# ---------------------------------------------------------------------------
def make_figure(img_rgb: np.ndarray,
                gt_col: Optional[np.ndarray],
                pred_cols: 'list[np.ndarray]',
                per_mod_heats: 'list[OrderedDict[str, Optional[np.ndarray]]]',
                model_labels: 'list[str]',
                title: str,
                save_path: str,
                alpha: float,
                shared_row_norm: bool,
                cmap_name: str = 'inferno') -> None:
    """Build the (1 + n_layers) x 4 grid figure."""
    n_models = len(per_mod_heats)
    n_cols = 1 + n_models  # left = input/label, right = mod columns
    n_layers = len(ROW_LABELS)
    n_rows = 1 + n_layers  # 1 header row + n_layers activation rows

    panel = 4.5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel * n_cols, panel * n_rows),
                             squeeze=False)

    # ---- Header row: input + GT + mod predictions (colored masks). ----
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Input image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    for j, (pred_col, lbl) in enumerate(zip(pred_cols, model_labels)):
        ax = axes[0, 1 + j]
        ax.imshow(pred_col)
        ax.set_title(f'{lbl}\n(predicted segmentation)', fontsize=11,
                     fontweight='bold')
        ax.axis('off')

    # ---- Activation rows. ----
    for i, key in enumerate(ROW_LABELS.keys(), start=1):
        # Left margin: row label only (no thumbnail; keeps figure compact).
        axes[i, 0].axis('off')
        axes[i, 0].text(
            0.5, 0.5, ROW_LABELS[key],
            ha='center', va='center', fontsize=12, fontweight='bold',
            transform=axes[i, 0].transAxes)

        # Optional shared per-row vmin/vmax across mods so the heatmaps are
        # directly comparable. Only across models that have data for this row.
        row_vmin: Optional[float] = None
        row_vmax: Optional[float] = None
        if shared_row_norm:
            present = [h[key] for h in per_mod_heats if h.get(key) is not None]
            if present:
                row_vmin = float(min(p.min() for p in present))
                row_vmax = float(max(p.max() for p in present))

        for j, heat_dict in enumerate(per_mod_heats):
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
    lines.append('Activation-map figure legend')
    lines.append('=' * 60)
    lines.append(f'channel reduction : {reduce}'
                 + (f' (top-{topk})' if reduce == 'topk' else ''))
    lines.append(f'overlay alpha     : {alpha}')
    lines.append(f'per-row shared norm: {shared_row_norm}')
    lines.append('')
    lines.append('Checkpoints:')
    for cfg, ck in zip(MODELS_TO_COMPARE, chosen_ckpts):
        lines.append(f'  {cfg["mod_id"]:<6}: {ck}')
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
    parser.add_argument('--image', default='ADE_val_00000229',
                        help='ADE20K val stem (e.g. ADE_val_00000229).')
    parser.add_argument('--output-dir', default='my_figures/activations')
    parser.add_argument('--reduce', default='l2',
                        choices=['l2', 'mean', 'max', 'topk'],
                        help='Channel-reduction mode for multi-channel feats.')
    parser.add_argument('--topk', type=int, default=8,
                        help='K for --reduce topk.')
    parser.add_argument('--alpha', type=float, default=0.55,
                        help='Heatmap alpha when blending over grayscale.')
    parser.add_argument('--cmap', default='inferno')
    parser.add_argument('--shared-row-norm', action='store_true',
                        help='Normalize each row across the 3 mod columns to '
                             'the same vmin/vmax (stricter visual comparison).')
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
    img_path = os.path.join(img_dir, f'{args.image}.jpg')
    ann_path = os.path.join(ann_dir, f'{args.image}.png')
    if not os.path.isfile(img_path):
        raise SystemExit(f'Image not found: {img_path}')

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gt_col: Optional[np.ndarray] = None
    if os.path.isfile(ann_path):
        gt_arr = np.array(Image.open(ann_path)).astype(np.int32)
        cls_mask = gt_arr - 1
        cls_mask[cls_mask < 0] = 255
        gt_col = colorize_mask(cls_mask.astype(np.uint8), palette)

    print(f'\nRunning inference + capturing activations on {args.image}.jpg ...')
    per_mod_heats = []
    pred_cols = []
    for cfg, m in zip(MODELS_TO_COMPARE, models):
        print(f'  {cfg["mod_id"]} ...', end=' ', flush=True)
        acts, seg_logits, pred_mask = run_one_model(m, img_path, cfg['mod_id'])
        heats = per_layer_heat_for_mod(
            acts, seg_logits, cfg['mod_id'], args.reduce, args.topk)
        per_mod_heats.append(heats)
        pred_cols.append(colorize_mask(pred_mask, palette))
        # Free GPU memory between mods.
        del acts, seg_logits, pred_mask
        torch.cuda.empty_cache()
        print('done')

    save_path = os.path.join(args.output_dir,
                             f'{args.image}_activations.png')
    legend_path = os.path.join(args.output_dir,
                               f'{args.image}_activations_legend.txt')

    title = (f'ADE20K val · {args.image} · activation maps '
             f'(reduce={args.reduce}'
             + (f', top-{args.topk}' if args.reduce == 'topk' else '')
             + f', alpha={args.alpha}'
             + (', shared-row-norm' if args.shared_row_norm else '')
             + ')')

    print(f'\nWriting figure -> {save_path}')
    make_figure(
        img_rgb=img_rgb,
        gt_col=gt_col,
        pred_cols=pred_cols,
        per_mod_heats=per_mod_heats,
        model_labels=[m['short'] for m in MODELS_TO_COMPARE],
        title=title,
        save_path=save_path,
        alpha=args.alpha,
        shared_row_norm=args.shared_row_norm,
        cmap_name=args.cmap,
    )
    write_legend(legend_path, args.reduce, args.topk, args.alpha,
                 args.shared_row_norm, chosen_ckpts)
    print(f'Legend         -> {legend_path}')
    print('Done.')


if __name__ == '__main__':
    main()
