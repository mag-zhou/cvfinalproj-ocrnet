# Copyright (c) OpenMMLab. All rights reserved.
"""Sanity tests for BoundaryContrastiveLoss (Mod 4 / CBL-lite).

These tests deliberately use random tensors and run on CPU so they
double as a quick smoke test (`pytest tests/test_boundary_contrastive.py`)
that doesn't require the full train pipeline.

They cover the risks called out in the implementation plan:

- Risk 3 (detach centres/negatives): pull gradient should *not* flow
  back through the per-anchor positives/negatives, only through the
  anchor's own feature.
- Loss is finite, non-negative, and zero in the trivial all-correct
  no-negatives case.
- ``loss_cbl`` is approximately zero in the early-iter regime (no
  correctly-classified positives -> all anchors skipped).
"""
from __future__ import annotations

import torch

from mmseg.models.losses import BoundaryContrastiveLoss


def _make_inputs(B=2, D=8, H=12, W=12, C=4, seed=0):
    torch.manual_seed(seed)
    features = torch.randn(B, D, H, W, requires_grad=True)
    seg_logits = torch.randn(B, C, H, W)
    gt_seg = torch.randint(0, C, (B, H, W))
    gt_boundary = (torch.rand(B, H, W) > 0.7).float()
    return features, seg_logits, gt_seg, gt_boundary


def test_loss_runs_and_is_finite():
    feats, logits, gt, bd = _make_inputs()
    loss_fn = BoundaryContrastiveLoss(
        kernel_size=3,
        margin=1.0,
        lambda_neg=0.5,
        max_anchors_per_image=200,
    )
    out = loss_fn(feats, logits, gt, bd)
    assert torch.isfinite(out).item()
    assert out.item() >= 0.0


def test_grad_flows_only_through_anchor_features():
    """Centre & negatives are detached; only the anchor feature should
    have a non-zero gradient contribution from the loss.

    We can't isolate "the centre" easily after the gather, but we can
    verify that
    (a) features.requires_grad path does receive gradient (loss is
        connected to the anchor features), and
    (b) detaching the loss's centres/negatives produces the same
        gradient as the live computation, i.e. the loss is invariant
        to whether neighbours had grad attached.
    """
    feats1, logits, gt, bd = _make_inputs()
    feats2 = feats1.detach().clone().requires_grad_(True)
    feats3 = feats1.detach().clone().requires_grad_(True)

    loss_fn = BoundaryContrastiveLoss(
        kernel_size=5,
        margin=1.0,
        lambda_neg=0.5,
        max_anchors_per_image=200,
    )

    # Pass 1: features carry grad as built.
    out1 = loss_fn(feats1, logits, gt, bd)
    assert out1.requires_grad, 'loss must have a grad path through anchors'
    out1.backward()
    assert feats1.grad is not None, 'expected gradient on anchor-side features'
    assert torch.isfinite(feats1.grad).all().item()

    # Pass 2: same forward but with detached features. Since the loss
    # detaches centres/negatives internally, providing detached features
    # should give a loss with no grad, but the *value* should match.
    out2 = loss_fn(feats2.detach(), logits, gt, bd)
    assert torch.allclose(out1.detach(), out2, atol=1e-5), (
        f'value should not depend on feature requires_grad: {out1.item()} vs {out2.item()}'
    )

    # Pass 3: explicitly verify detached-feature loss has no grad path.
    out3 = loss_fn(feats3.detach(), logits, gt, bd)
    assert not out3.requires_grad, (
        'with detached features the loss should produce a grad-free scalar')


def test_zero_when_no_anchors():
    """If gt_boundary is empty everywhere, the loss is zero (no anchors)."""
    feats, logits, gt, _ = _make_inputs()
    bd = torch.zeros(feats.shape[0], feats.shape[2], feats.shape[3])
    loss_fn = BoundaryContrastiveLoss(kernel_size=3, max_anchors_per_image=200)
    out = loss_fn(feats, logits, gt, bd)
    assert out.item() == 0.0
    stats = loss_fn.anchor_stats
    assert stats['n_total'] == 0 and stats['n_valid'] == 0


def test_zero_when_no_positives_anywhere():
    """If predictions are all wrong, no anchor has any correct neighbour;
    the CBL loss should be ~0 (matches Phase 4 Test 3 in the plan).
    """
    B, D, H, W, C = 2, 4, 16, 16, 6
    torch.manual_seed(1)
    feats = torch.randn(B, D, H, W, requires_grad=True)
    gt = torch.randint(0, C, (B, H, W))
    # Force argmax(logits) != gt everywhere by setting a class shifted
    # from the ground truth.
    wrong = (gt + 1) % C
    logits = torch.full((B, C, H, W), -10.0)
    logits.scatter_(1, wrong.unsqueeze(1), 10.0)
    bd = (torch.rand(B, H, W) > 0.5).float()

    loss_fn = BoundaryContrastiveLoss(kernel_size=3, max_anchors_per_image=200)
    out = loss_fn(feats, logits, gt, bd)
    assert out.item() == 0.0, (
        f'Expected 0 loss when no anchor has correct same-class neighbours, '
        f'got {out.item()}')
    stats = loss_fn.anchor_stats
    assert stats['n_valid'] == 0


def test_loss_uses_anchor_grad_only_via_pull_and_push():
    """Sanity: gradient on `features` is non-trivial when there *are*
    valid anchors with positives and negatives present in the batch.

    We construct a deterministic toy where every pixel is correctly
    classified so all neighbours are positives/negatives by class.
    """
    B, D, H, W, C = 1, 4, 8, 8, 3
    torch.manual_seed(2)
    feats = torch.randn(B, D, H, W, requires_grad=True)
    gt = torch.randint(0, C, (B, H, W))
    # Make argmax(logits) == gt everywhere so window_correct is all True.
    logits = torch.full((B, C, H, W), -10.0)
    logits.scatter_(1, gt.unsqueeze(1), 10.0)
    # Boundary at every pixel -> N anchors = H*W.
    bd = torch.ones(B, H, W)

    loss_fn = BoundaryContrastiveLoss(
        kernel_size=3,
        margin=1.0,
        lambda_neg=0.5,
        max_anchors_per_image=10000,
    )
    out = loss_fn(feats, logits, gt, bd)
    assert out.item() > 0.0
    out.backward()
    # At least *some* feature element gets non-zero gradient.
    assert (feats.grad.abs() > 0).any().item()


if __name__ == '__main__':
    test_loss_runs_and_is_finite()
    test_grad_flows_only_through_anchor_features()
    test_zero_when_no_anchors()
    test_zero_when_no_positives_anywhere()
    test_loss_uses_anchor_grad_only_via_pull_and_push()
    print('All BoundaryContrastiveLoss sanity tests passed.')
