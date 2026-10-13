"""Alignment, attribution-mass and faithfulness metrics.

IoU on its own is a weak instrument: it conflates "the model looked somewhere else"
with "the model looked at more than the annotator marked", it is sensitive to how many
words you decide to call important, and it throws away the magnitude of the
attribution. So each case also carries precision/recall/F1, an attribution-mass share,
several selection thresholds, and a random baseline to compare against.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def binary_overlap(model_mask: np.ndarray, human_mask: np.ndarray) -> Dict[str, float]:
    """IoU, precision, recall and F1 between two boolean word masks.

    Precision is the share of the model's selection the expert also marked; recall is
    the share of expert rationale the model selected. Reporting them next to IoU is
    what separates "the model looked elsewhere" (low precision *and* low recall) from
    "the model looked more broadly" (low precision, high recall) -- IoU alone scores
    both badly.
    """
    intersection = float(np.logical_and(model_mask, human_mask).sum())
    union = float(np.logical_or(model_mask, human_mask).sum())
    n_model = float(model_mask.sum())
    n_human = float(human_mask.sum())

    precision = intersection / n_model if n_model else float("nan")
    recall = intersection / n_human if n_human else float("nan")
    if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (n_model or n_human) else float("nan")

    return {
        "iou": intersection / union if union else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "intersection": intersection,
        "union": union,
        "n_model": n_model,
        "n_human": n_human,
    }


def expected_random_iou(n_model: int, n_human: int, n_eligible: int) -> float:
    """IoU expected from selecting ``n_model`` words uniformly at random.

    Without this an IoU of 0.25 is uninterpretable. If a rationale covers a quarter of
    the document, random selection already scores about 0.14, so the model's number has
    to be read against that floor rather than against zero.
    """
    if n_eligible <= 0 or n_model <= 0 or n_human <= 0:
        return float("nan")
    expected_intersection = n_model * n_human / n_eligible
    expected_union = n_model + n_human - expected_intersection
    return expected_intersection / expected_union if expected_union else float("nan")


def attribution_mass_share(
    scores: np.ndarray, region_mask: np.ndarray, eligible: np.ndarray
) -> float:
    """Fraction of total attribution magnitude falling inside a region.

    Used for two things: the human-supported attribution share (region = expert
    rationale) and the shortcut attribution share (region = shortcut rhetorical
    roles). Unlike IoU this uses the magnitudes rather than a top-k cut, so it does not
    depend on choosing a threshold at all.
    """
    scores = np.abs(scores)
    total = float(scores[eligible].sum())
    if total <= 0:
        return float("nan")
    return float(scores[np.logical_and(region_mask, eligible)].sum() / total)


def mean_attribution_inside_outside(
    scores: np.ndarray, region_mask: np.ndarray, eligible: np.ndarray
) -> Dict[str, float]:
    """Mean per-word attribution inside vs outside a region.

    A threshold-free companion to IoU: even when the top-k sets barely overlap, the
    model may still put systematically more weight on expert-marked words.
    """
    scores = np.abs(scores)
    inside = np.logical_and(region_mask, eligible)
    outside = np.logical_and(~region_mask, eligible)
    mean_in = float(scores[inside].mean()) if inside.any() else float("nan")
    mean_out = float(scores[outside].mean()) if outside.any() else float("nan")
    ratio = mean_in / mean_out if (mean_out and mean_out > 0) else float("nan")
    return {
        "mean_attribution_human": mean_in,
        "mean_attribution_nonhuman": mean_out,
        "attribution_ratio_human_nonhuman": ratio,
    }


def role_attribution_shares(
    scores: np.ndarray,
    role_masks: Dict[str, np.ndarray],
    eligible: np.ndarray,
    pooled_roles: Sequence[str],
) -> Dict[str, float]:
    """Attribution mass per rhetorical role, plus the pooled post-decisional share.

    This is what lets Stage 2 and Stage 3 talk to each other: Stage 2 shows that adding
    a role changes the prediction, and this shows whether the model's attribution
    actually concentrates in that same role.
    """
    scores = np.abs(scores)
    total = float(scores[eligible].sum())
    shares: Dict[str, float] = {}
    if total <= 0:
        return shares
    for role, mask in role_masks.items():
        shares["role_share_" + role] = float(
            scores[np.logical_and(mask, eligible)].sum() / total
        )
        # The share a role would get if attribution were spread evenly over words;
        # a role_share well above its role_words share means attribution concentrates there.
        shares["role_words_" + role] = float(
            np.logical_and(mask, eligible).sum() / max(1, eligible.sum())
        )
    pooled = np.zeros_like(eligible)
    for role in pooled_roles:
        if role in role_masks:
            pooled |= role_masks[role]
    pooled &= eligible
    shares["post_decisional_attribution_share"] = float(scores[pooled].sum() / total)
    shares["post_decisional_word_share"] = float(pooled.sum() / max(1, eligible.sum()))
    return shares


def faithfulness_scores(
    baseline_logit: float,
    top_masked_logit: float,
    bottom_masked_logit: float,
) -> Dict[str, float]:
    """Comprehensiveness-style faithfulness check on the target-class logit.

    Removing the words the attribution calls important should move the prediction more
    than removing the words it calls unimportant. If it does not, the heatmap is not
    describing the model's decision and no IoU computed from it means anything.
    ``faithfulness_gap`` is positive when the attribution behaves as it should.
    """
    drop_top = baseline_logit - top_masked_logit
    drop_bottom = baseline_logit - bottom_masked_logit
    return {
        "faithfulness_drop_top": float(drop_top),
        "faithfulness_drop_bottom": float(drop_bottom),
        "faithfulness_gap": float(drop_top - drop_bottom),
        "faithfulness_pass": bool(drop_top > drop_bottom),
    }


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for a mean."""
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(clean, size=(n_boot, clean.size), replace=True).mean(axis=1)
    return {
        "mean": float(clean.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(clean.size),
    }


def correlate(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Pearson and Spearman correlation over pairs where both values are finite.

    Spearman is the one to lead with: the hypothesis is that greater shortcut reliance
    goes with weaker rationale alignment, which is a monotone claim, not a linear one.
    """
    from scipy import stats

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if keep.sum() < 3:
        return {"n": int(keep.sum())}
    xs, ys = x_arr[keep], y_arr[keep]
    if np.std(xs) == 0 or np.std(ys) == 0:
        return {"n": int(keep.sum()), "note": "zero variance"}
    pearson_r, pearson_p = stats.pearsonr(xs, ys)
    spearman_r, spearman_p = stats.spearmanr(xs, ys)
    return {
        "n": int(keep.sum()),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
