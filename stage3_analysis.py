#!/usr/bin/env python3
"""Stage 3 analysis on CPU: where the model's attribution lands, from a saved GPU run.

Reads ``stage3_results.csv`` and ``word_scores.npz`` written by ``modal_stage3.py``,
plus the Stage-2 files, and writes tables, figures and ``summary.json``. No GPU and
no model are needed, so every number here can be recomputed on a laptop.

    python stage3_analysis.py --run-dir stage3_outputs_modal/paper_v2

Questions, each with the confound it controls for (written before the full run):

H1  Does attribution concentrate on post-decisional text (Analysis, Ratio,
    present-court ruling) beyond its share of the words? Facts and Statute below?
    Control: role word share; and position-matched density, since Analysis sits late
    in the judgment and position alone could explain it.
H2  Within the Analysis, does the end (the text just before the removed decision)
    get more attribution than the start? Stage-2 Control C uses the same 60/150-word
    first/last definition. Cases whose Analysis end is outside the model's window
    are counted, not scored.
H3  Where does the expert rationale sit by role, and does the model's overlap with it
    exceed what its role preferences alone would give? Control: a role-matched
    random baseline that keeps how many of the model's top words fall in each role.
H4  Case by case, is the model's attribution to the Analysis larger where the
    Analysis behaviourally drives the prediction (Stage-2 v2 and Controls B and D)?
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stage3.alignment import select_top_words
from stage3.audit import _positive_probability
from stage3.config import KNOWN_ROLES, POST_DECISIONAL_ROLES
from stage3.metrics import binary_overlap, bootstrap_ci, correlate, expected_random_iou
from stage3.word_store import load_store, role_mask

HERE = Path(__file__).resolve().parent
ROLE_LABELS = {
    "FAC": "Facts",
    "ANALYSIS": "Analysis",
    "RATIO": "Ratio",
    "RPC": "Present-court ruling",
    "STA": "Statute",
    "PRE_RELIED": "Precedent relied",
    "PRE_NOT_RELIED": "Precedent not relied",
    "RLC": "Lower-court ruling",
    "ARG_PETITIONER": "Arguments (petitioner)",
    "ARG_RESPONDENT": "Arguments (respondent)",
    "ISSUE": "Issues",
    "PREAMBLE": "Preamble",
    "NONE": "None",
}
REPORT_ROLES = ["ANALYSIS", "RATIO", "RPC", "FAC", "STA", "PRE_RELIED", "RLC",
                "ARG_PETITIONER", "ARG_RESPONDENT", "ISSUE"]
SEGMENT_BUDGETS = (60, 150)
N_POSITION_BINS = 20
N_NULL_DRAWS = 100
SEED = 42
MIN_CASES_PER_BIN = 20  # position bins with fewer contributing cases are left out


def _ci_row(values) -> Dict[str, float]:
    ci = bootstrap_ci(list(values))
    return {"mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": ci["n"]}


def _fmt(ci: Dict[str, float], pct: bool = False, signed: bool = False) -> str:
    if not ci or not ci.get("n"):
        return "–"
    scale, unit = (100.0, "") if pct else (1.0, "")
    sign = "+" if signed else ""
    body = "{:" + sign + ".1f}" if pct else "{:" + sign + ".3f}"
    text = body.format(ci["mean"] * scale)
    return "{} [{}, {}]{}".format(
        text, body.format(ci["ci_low"] * scale), body.format(ci["ci_high"] * scale), unit)


# ── H1: attribution by role ─────────────────────────────────────────────────
def role_shares(store) -> pd.DataFrame:
    """One row per (case, role) with visible words: attribution share vs word share."""
    rows = []
    for row, a in store.items():
        eligible = a["eligible"]
        total = float(a["scores"][eligible].sum())
        n_visible = int(eligible.sum())
        if total <= 0 or n_visible == 0:
            continue
        signed = a.get("signed")
        for role in KNOWN_ROLES:
            mask = role_mask(a["roles"], role) & eligible
            n = int(mask.sum())
            if n == 0:
                continue
            record = {
                "row_index": row,
                "role": role,
                "n_words": n,
                "mass_share": float(a["scores"][mask].sum()) / total,
                "word_share": n / n_visible,
            }
            if signed is not None and np.abs(signed[mask]).sum() > 0:
                # Share of the role's signed attribution that supports the prediction.
                record["support_share"] = float(
                    np.clip(signed[mask], 0, None).sum() / np.abs(signed[mask]).sum())
            rows.append(record)
        pooled = np.zeros_like(eligible)
        for role in POST_DECISIONAL_ROLES:
            pooled |= role_mask(a["roles"], role)
        pooled &= eligible
        if pooled.any():
            rows.append({
                "row_index": row, "role": "POST_DECISIONAL", "n_words": int(pooled.sum()),
                "mass_share": float(a["scores"][pooled].sum()) / total,
                "word_share": float(pooled.sum()) / n_visible,
            })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["excess"] = frame["mass_share"] - frame["word_share"]
    return frame


def summarise_roles(shares: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ["POST_DECISIONAL"] + REPORT_ROLES:
        part = shares[shares["role"] == role]
        if part.empty:
            continue
        rows.append({
            "role": role,
            "n_cases": len(part),
            "mass_share": part["mass_share"].mean(),
            "word_share": part["word_share"].mean(),
            "ratio": part["mass_share"].mean() / part["word_share"].mean(),
            **{"excess_" + k: v for k, v in _ci_row(part["excess"]).items()},
            **({"posadj_" + k: v for k, v in _ci_row(part["position_adjusted"]).items()}
               if "position_adjusted" in part else {}),
            "posadj_coverage": (part["position_coverage"].mean()
                                if "position_coverage" in part else np.nan),
            "support_share": part["support_share"].mean() if "support_share" in part else np.nan,
        })
    return pd.DataFrame(rows)


def _positions(a):
    """Visible word indices, their position bin, and scores normalised to the case mean."""
    visible = np.flatnonzero(a["eligible"])
    bins = (np.arange(visible.size) * N_POSITION_BINS) // max(1, visible.size)
    normalised = a["scores"][visible] / a["scores"][visible].mean()
    return visible, bins, normalised


def position_curve(store) -> np.ndarray:
    """Expected attribution density at each position, from non-post-decisional text.

    Analysis sits late in a judgment, so a model that simply weights later text more
    would make it look important. This curve is built only from words outside the
    post-decisional roles (facts, arguments, statute, precedent, lower-court ruling):
    for each position bin, the mean over cases of their normalised density there.
    """
    post = sum(1 << KNOWN_ROLES.index(r) for r in POST_DECISIONAL_ROLES)
    per_bin = [[] for _ in range(N_POSITION_BINS)]
    for a in store.values():
        if a["eligible"].sum() < N_POSITION_BINS * 5 or a["scores"][a["eligible"]].mean() <= 0:
            continue
        visible, bins, normalised = _positions(a)
        roles = a["roles"][visible]
        reference = (roles != 0) & ((roles & post) == 0)
        for b in range(N_POSITION_BINS):
            mask = reference & (bins == b)
            if mask.sum() >= 5:
                per_bin[b].append(normalised[mask].mean())
    return np.array([np.mean(v) if len(v) >= MIN_CASES_PER_BIN else np.nan for v in per_bin])


def position_adjusted(a, word_mask: np.ndarray, curve: np.ndarray):
    """Mean of (normalised density / expected density at that position) over the words.

    Returns (value, coverage): coverage is the share of the words that sit in a
    position bin with a defined reference. Near the end of a judgment there is little
    non-post-decisional text, so some bins have no reference and are left out.
    """
    if a["scores"][a["eligible"]].mean() <= 0:
        return float("nan"), 0.0
    visible, bins, normalised = _positions(a)
    in_mask = word_mask[visible]
    keep = in_mask & np.isfinite(curve[bins])
    coverage = float(keep.sum() / in_mask.sum()) if in_mask.any() else 0.0
    if keep.sum() == 0:
        return float("nan"), coverage
    return float(np.mean(normalised[keep] / curve[bins[keep]])), coverage


def add_position_adjusted(shares: pd.DataFrame, store, curve: np.ndarray) -> pd.DataFrame:
    values, coverage = [], []
    for r in shares.itertuples():
        a = store[r.row_index]
        if r.role == "POST_DECISIONAL":
            mask = np.zeros_like(a["eligible"])
            for role in POST_DECISIONAL_ROLES:
                mask |= role_mask(a["roles"], role)
        else:
            mask = role_mask(a["roles"], r.role)
        value, covered = position_adjusted(a, mask & a["eligible"], curve)
        values.append(value)
        coverage.append(covered)
    shares = shares.copy()
    shares["position_adjusted"] = values
    shares["position_coverage"] = coverage
    return shares


# ── H2: first vs last part of the Analysis ──────────────────────────────────
def analysis_segments(store, budget: int, curve: np.ndarray = None) -> pd.DataFrame:
    """Density of the first and last ``budget`` Analysis words (document order).

    Matches Stage-2 Control C: the first/last ``budget`` words of the Analysis section.
    A case is scored only when the Analysis has at least 2*budget words (so the two
    parts do not overlap) and both parts are fully inside the model's window.
    """
    rows = []
    for row, a in store.items():
        analysis = np.flatnonzero(role_mask(a["roles"], "ANALYSIS"))
        if analysis.size < 2 * budget:
            continue
        first, last = analysis[:budget], analysis[-budget:]
        eligible = a["eligible"]
        status = "scored"
        if not eligible[last].all():
            status = "last_part_truncated"
        elif not eligible[first].all():
            status = "first_part_truncated"
        record = {"row_index": row, "budget": budget, "status": status}
        if status == "scored":
            mean_visible = a["scores"][eligible].mean()
            record["density_first"] = a["scores"][first].mean() / mean_visible
            record["density_last"] = a["scores"][last].mean() / mean_visible
            record["last_minus_first"] = record["density_last"] - record["density_first"]
            if curve is not None:
                for name, part in (("first", first), ("last", last)):
                    mask = np.zeros_like(eligible)
                    mask[part] = True
                    record["posadj_" + name], record["posadj_coverage_" + name] = (
                        position_adjusted(a, mask, curve))
                record["posadj_last_minus_first"] = record["posadj_last"] - record["posadj_first"]
        rows.append(record)
    return pd.DataFrame(rows)


def analysis_profile(store) -> pd.DataFrame:
    """Density along the visible Analysis, in ``N_POSITION_BINS`` parts by order."""
    rows = []
    for row, a in store.items():
        eligible = a["eligible"]
        analysis = np.flatnonzero(role_mask(a["roles"], "ANALYSIS") & eligible)
        if analysis.size < 50:
            continue
        mean_visible = a["scores"][eligible].mean()
        for b, idx in enumerate(np.array_split(analysis, 10)):
            rows.append({"row_index": row, "bin": b,
                         "density": a["scores"][idx].mean() / mean_visible})
    return pd.DataFrame(rows)


# ── H3: expert rationale by role, and a role-matched baseline ───────────────
def _strata(roles: np.ndarray) -> np.ndarray:
    """One stratum per word: the first matching role in KNOWN_ROLES order, else -1."""
    strata = np.full(roles.shape, -1, dtype=int)
    for i in reversed(range(len(KNOWN_ROLES))):
        strata[(roles & (1 << i)) != 0] = i
    return strata


def rationale_roles(store, rows_ok) -> pd.DataFrame:
    """Share of visible rationale words in each role vs that role's share of the text."""
    out = []
    for row in rows_ok:
        a = store.get(row)
        if a is None:
            continue
        human = a["human"] & a["eligible"]
        if human.sum() == 0:
            continue
        for role in KNOWN_ROLES:
            mask = role_mask(a["roles"], role) & a["eligible"]
            if mask.any() or (mask & human).any():
                out.append({"row_index": row, "role": role,
                            "rationale_share": (mask & human).sum() / human.sum(),
                            "word_share": mask.sum() / a["eligible"].sum()})
    return pd.DataFrame(out)


def role_matched_iou(store, rows_ok, n_draws: int = N_NULL_DRAWS) -> pd.DataFrame:
    """Observed IoU vs uniform and role-matched random selections of the same size."""
    rng = np.random.default_rng(SEED)
    out = []
    for row in rows_ok:
        a = store.get(row)
        if a is None:
            continue
        eligible, human = a["eligible"], a["human"] & a["eligible"]
        k = int(human.sum())
        if k == 0:
            continue
        chosen = select_top_words(a["scores"], eligible, k)
        strata = _strata(a["roles"])
        pools = {s: np.flatnonzero(eligible & (strata == s)) for s in np.unique(strata[eligible])}
        counts = {s: int((chosen & (strata == s)).sum()) for s in pools}
        null = []
        for _ in range(n_draws):
            draw = np.zeros_like(eligible)
            for s, pool in pools.items():
                if counts[s]:
                    draw[rng.choice(pool, size=counts[s], replace=False)] = True
            null.append(binary_overlap(draw, human)["iou"])
        out.append({
            "row_index": row,
            "iou": binary_overlap(chosen, human)["iou"],
            "iou_uniform_random": expected_random_iou(k, k, int(eligible.sum())),
            "iou_role_matched": float(np.mean(null)),
        })
    frame = pd.DataFrame(out)
    if len(frame):
        frame["iou_minus_role_matched"] = frame["iou"] - frame["iou_role_matched"]
    return frame


# ── H4: case-by-case link to Stage 2 ────────────────────────────────────────
def _p_true(entry, label: int) -> float:
    if not entry or entry.get("label", -1) == -1:
        return float("nan")
    p1 = _positive_probability(entry)
    return p1 if label == 1 else 1.0 - p1


def stage2_measures(stage2_json: str, controls_jsonl: str = "") -> pd.DataFrame:
    """Per-case behavioural effect of the Analysis (higher = Analysis drives it more)."""
    rows = {}
    with open(stage2_json) as handle:
        for case in json.load(handle):
            p, y = case.get("predictions", {}), int(case["true_label"])
            rows[int(case["index"])] = {
                "row_index": int(case["index"]),
                # Stage 2 v2 Layer 2: P(true outcome) gained by adding the Analysis.
                "s2_add_analysis_toward_truth": _p_true(p.get("v_analysis_only"), y)
                - _p_true(p.get("v0_facts_only"), y),
            }
    if controls_jsonl and os.path.isfile(controls_jsonl):
        with open(controls_jsonl) as handle:
            for line in handle:
                case = json.loads(line)
                p, y = case.get("predictions", {}), int(case["true_label"])
                row = rows.setdefault(int(case["index"]), {"row_index": int(case["index"])})
                # Control B: P(true) lost by removing the Analysis, beyond removing the
                # same number of random words.
                row["s2_loo_analysis_excess"] = (
                    _p_true(p.get("loo_rand_analysis"), y) - _p_true(p.get("loo_analysis"), y))

                def p1(name):
                    entry = p.get(name)
                    if not entry or entry.get("label", -1) == -1:
                        return float("nan")
                    return _positive_probability(entry)
                # Control D: P(accepted) with an accepted donor's Analysis minus with a
                # rejected donor's, on the case's own facts.
                row["s2_transplant_follow"] = p1("tx_analysis_d1") - p1("tx_analysis_d0")
    return pd.DataFrame(list(rows.values()))


def spearman_ci(x, y, n_boot: int = 1000) -> Dict[str, float]:
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 10:
        return {"n": int(x.size)}
    rho = float(stats.spearmanr(x, y)[0])
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, x.size, x.size)
        boots.append(stats.spearmanr(x[i], y[i])[0])
    corr = correlate(x, y)
    return {"n": int(x.size), "rho": rho, "ci_low": float(np.nanpercentile(boots, 2.5)),
            "ci_high": float(np.nanpercentile(boots, 97.5)), "p": corr.get("spearman_p")}


# ── figures ─────────────────────────────────────────────────────────────────
def write_figures(outdir: Path, role_table, profile, curve, rationale_table) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    table = role_table[role_table["role"] != "POST_DECISIONAL"]
    if len(table):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(table))
        ax.bar(x - 0.2, table["word_share"] * 100, 0.4, label="share of words", color="#bbbbbb")
        ax.bar(x + 0.2, table["mass_share"] * 100, 0.4, label="share of attribution", color="#3b6ea8")
        ax.set_xticks(x, [ROLE_LABELS.get(r, r) for r in table["role"]], rotation=35, ha="right")
        ax.set_ylabel("% of visible text (mean over cases)")
        ax.set_title("Where Integrated Gradients attribution lands, by rhetorical role")
        ax.legend()
        fig.tight_layout()
        paths.append(str(outdir / "fig_role_attribution.png"))
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    if len(profile) or np.isfinite(curve).any():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if len(profile):
            means = profile.groupby("bin")["density"].mean()
            axes[0].plot(means.index * 10 + 5, means.values, marker="o", color="#3b6ea8")
            axes[0].axhline(1, color="grey", lw=0.8, ls="--")
            axes[0].set_xlabel("position within the visible Analysis (%)")
            axes[0].set_ylabel("attribution density (1 = document average)")
            axes[0].set_title("Along the Analysis section")
        if np.isfinite(curve).any():
            centres = (np.arange(N_POSITION_BINS) + 0.5) * 100 / N_POSITION_BINS
            axes[1].plot(centres, curve, marker="o", color="#777777")
            axes[1].axhline(1, color="grey", lw=0.8, ls="--")
            axes[1].set_xlabel("position in the visible document (%)")
            axes[1].set_ylabel("density of non-post-decisional text")
            axes[1].set_title("Position effect (reference for adjustment)")
        fig.tight_layout()
        paths.append(str(outdir / "fig_position.png"))
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    if len(rationale_table):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(rationale_table))
        ax.bar(x - 0.2, rationale_table["word_share"] * 100, 0.4, label="share of words", color="#bbbbbb")
        ax.bar(x + 0.2, rationale_table["rationale_share"] * 100, 0.4, label="share of expert rationale",
               color="#a83b5c")
        ax.set_xticks(x, [ROLE_LABELS.get(r, r) for r in rationale_table["role"]], rotation=35, ha="right")
        ax.set_ylabel("%")
        ax.set_title("Where the PredEx expert rationale sits, by rhetorical role")
        ax.legend()
        fig.tight_layout()
        paths.append(str(outdir / "fig_rationale_roles.png"))
        fig.savefig(paths[-1], dpi=150)
        plt.close(fig)
    return paths


# ── driver ──────────────────────────────────────────────────────────────────
def run(run_dir: Path, stage2_json: str, controls_jsonl: str, outdir: Path) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(run_dir / "stage3_results.csv")
    store = load_store(str(run_dir / "word_scores.npz"))
    summary: Dict[str, object] = {"n_cases": int(len(results)), "n_with_word_scores": len(store)}
    lines = ["# Stage 3 v2 analysis", "",
             "Generated by `stage3_analysis.py` from `{}`.".format(run_dir), ""]

    # Quality
    attributed = results[results["row_index"].isin(list(store))]
    quality = {
        "statuses": results["audit_status"].value_counts().to_dict(),
        "accuracy": float(results["correct"].mean()),
    }
    if "ig_relative_delta" in attributed:
        rel = attributed["ig_relative_delta"].dropna()
        quality["ig_relative_delta_median"] = float(rel.median())
        quality["ig_within_5pct"] = float((rel <= 0.05).mean())
        quality["ig_retried"] = float((attributed["ig_steps_used"] > attributed["ig_steps_used"].min()).mean())
    if "faithfulness_gap" in attributed:
        quality["faithfulness_gap"] = _ci_row(attributed["faithfulness_gap"])
        quality["faithfulness_pass"] = float(attributed["faithfulness_pass"].mean())
    summary["quality"] = quality
    lines += ["## Quality", "",
              "- Cases: {} ({} with attributions). Statuses: {}".format(
                  len(results), len(store), quality["statuses"]),
              "- Accuracy {:.1%}".format(quality["accuracy"])]
    if "ig_within_5pct" in quality:
        lines.append("- IG completeness: median relative delta {:.3f}; within 5%: {:.1%}; "
                     "recomputed with more steps: {:.1%}".format(
                         quality["ig_relative_delta_median"], quality["ig_within_5pct"],
                         quality["ig_retried"]))
    if "faithfulness_gap" in quality:
        lines.append("- Faithfulness gap (top-10% masked minus bottom-10% masked, decision margin): "
                     "{}; positive in {:.1%} of cases".format(
                         _fmt(quality["faithfulness_gap"]), quality["faithfulness_pass"]))
    lines.append("")

    # H1
    curve = position_curve(store)
    summary["position_curve"] = [None if not np.isfinite(v) else float(v) for v in curve]
    shares = add_position_adjusted(role_shares(store), store, curve)
    shares.to_csv(outdir / "role_shares_per_case.csv", index=False)
    role_table = summarise_roles(shares)
    role_table.to_csv(outdir / "role_shares.csv", index=False)
    summary["h1_roles"] = role_table.to_dict(orient="records")
    lines += ["## H1. Attribution by rhetorical role", "",
              "Mean over cases containing the role (visible part only). Excess = attribution "
              "share − word share. Position-adjusted = attribution density relative to "
              "non-post-decisional text at the same position in the judgment (1 = what position "
              "alone predicts). Support = share of the role's signed attribution that pushes "
              "toward the predicted class. Brackets are 95% bootstrap CIs over cases. Adjusted "
              "words = share of the role's words at positions with a reference.", "",
              "| Role | Cases | Words | Attribution | Ratio | Excess (pp) | Position-adjusted "
              "| Adjusted words | Support |",
              "|---|---|---|---|---|---|---|---|---|"]

    def ci(r, prefix):
        return {"mean": getattr(r, prefix + "mean"), "ci_low": getattr(r, prefix + "ci_low"),
                "ci_high": getattr(r, prefix + "ci_high"), "n": getattr(r, prefix + "n")}
    for r in role_table.itertuples():
        lines.append("| {} | {} | {:.1%} | {:.1%} | {:.2f} | {} | {} | {:.0%} | {} |".format(
            "Post-decisional (Analysis+Ratio+RPC)" if r.role == "POST_DECISIONAL"
            else ROLE_LABELS.get(r.role, r.role),
            r.n_cases, r.word_share, r.mass_share, r.ratio,
            _fmt(ci(r, "excess_"), pct=True, signed=True),
            _fmt(ci(r, "posadj_")) if hasattr(r, "posadj_mean") else "–",
            r.posadj_coverage if np.isfinite(r.posadj_coverage) else 0.0,
            "–" if not np.isfinite(r.support_share) else "{:.1%}".format(r.support_share)))
    lines.append("")

    # H2
    profile = analysis_profile(store)
    summary["h2"] = {}
    lines += ["## H2. Start vs end of the Analysis", "",
              "Density = mean attribution per word in the part / mean per visible word. "
              "Position-adjusted divides by the density non-post-decisional text gets at the "
              "same position, since the end of the Analysis is usually near the end of the "
              "judgment.", "",
              "| Words | Scored cases | End truncated | First part | Last part | Last − first "
              "| Position-adjusted: first | last | last: adjusted words |",
              "|---|---|---|---|---|---|---|---|---|"]
    segment_frames = []
    for budget in SEGMENT_BUDGETS:
        seg = analysis_segments(store, budget, curve)
        segment_frames.append(seg)
        if seg.empty:
            continue
        scored = seg[seg["status"] == "scored"]
        entry = {
            "n_scored": int(len(scored)),
            "n_last_truncated": int((seg["status"] == "last_part_truncated").sum()),
            "first": _ci_row(scored["density_first"]) if len(scored) else {},
            "last": _ci_row(scored["density_last"]) if len(scored) else {},
            "last_minus_first": _ci_row(scored["last_minus_first"]) if len(scored) else {},
            "posadj_first": _ci_row(scored["posadj_first"]) if len(scored) else {},
            "posadj_last": _ci_row(scored["posadj_last"]) if len(scored) else {},
            "posadj_coverage_last": (float(scored["posadj_coverage_last"].mean())
                                     if len(scored) else None),
        }
        summary["h2"][str(budget)] = entry
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            budget, entry["n_scored"], entry["n_last_truncated"], _fmt(entry["first"]),
            _fmt(entry["last"]), _fmt(entry["last_minus_first"], signed=True),
            _fmt(entry["posadj_first"]), _fmt(entry["posadj_last"]),
            "–" if entry["posadj_coverage_last"] is None
            else "{:.0%}".format(entry["posadj_coverage_last"])))
    if segment_frames:
        pd.concat(segment_frames).to_csv(outdir / "analysis_segments_per_case.csv", index=False)
    profile.to_csv(outdir / "analysis_profile_per_case.csv", index=False)
    lines.append("")

    # H3
    rows_ok = results.loc[results["audit_status"] == "ok", "row_index"].astype(int).tolist()
    composition = rationale_roles(store, rows_ok)
    rationale_table = pd.DataFrame()
    if len(composition):
        rationale_table = (composition.groupby("role")[["rationale_share", "word_share"]].mean()
                           .reindex([r for r in REPORT_ROLES if r in set(composition["role"])])
                           .reset_index())
        rationale_table.to_csv(outdir / "rationale_roles.csv", index=False)
    null = role_matched_iou(store, rows_ok)
    null.to_csv(outdir / "iou_role_matched_per_case.csv", index=False)
    lines += ["## H3. Expert rationale and the role-matched baseline", "",
              "Cases with audit_status ok. Mean share of the visible expert rationale per role:", "",
              "| Role | Share of words | Share of rationale |", "|---|---|---|"]
    for r in rationale_table.itertuples():
        lines.append("| {} | {:.1%} | {:.1%} |".format(ROLE_LABELS.get(r.role, r.role),
                                                    r.word_share, r.rationale_share))
    if len(null):
        h3 = {k: _ci_row(null[k]) for k in
              ("iou", "iou_uniform_random", "iou_role_matched", "iou_minus_role_matched")}
        summary["h3"] = {"rationale_roles": rationale_table.to_dict(orient="records"), **h3}
        lines += ["", "IoU at the expert-rationale size (n={}):".format(h3["iou"]["n"]), "",
                  "- Model: {}".format(_fmt(h3["iou"])),
                  "- Uniform random words: {}".format(_fmt(h3["iou_uniform_random"])),
                  "- Random words with the model's role mix: {}".format(_fmt(h3["iou_role_matched"])),
                  "- Model − role-matched: {}".format(_fmt(h3["iou_minus_role_matched"], signed=True))]
    lines.append("")

    # H4
    s2 = stage2_measures(stage2_json, controls_jsonl)
    analysis = shares[shares["role"] == "ANALYSIS"][["row_index", "excess", "mass_share"]].rename(
        columns={"excess": "analysis_excess", "mass_share": "analysis_mass_share"})
    linked = analysis.merge(s2, on="row_index", how="inner").merge(
        results[["row_index", "correct"]], on="row_index", how="left")
    linked.to_csv(outdir / "stage2_link_per_case.csv", index=False)
    summary["h4"] = {}
    lines += ["## H4. Case-by-case link to Stage 2", "",
              "Spearman correlation between the Analysis attribution excess (Stage 3) and each "
              "Stage-2 measure, with 95% bootstrap CI.", "",
              "| Stage-2 measure | n | rho | 95% CI |", "|---|---|---|---|"]
    for column, label in (("s2_add_analysis_toward_truth", "v2: P(true) gained by adding Analysis to facts"),
                          ("s2_loo_analysis_excess", "Control B: P(true) lost removing Analysis, beyond random"),
                          ("s2_transplant_follow", "Control D: follows a transplanted Analysis")):
        if column not in linked:
            continue
        stats = spearman_ci(linked["analysis_excess"], linked[column])
        summary["h4"][column] = stats
        if "rho" in stats:
            lines.append("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] |".format(
                label, stats["n"], stats["rho"], stats["ci_low"], stats["ci_high"]))
    if "correct" in linked:
        by_correct = {str(bool(k)): _ci_row(v["analysis_excess"])
                      for k, v in linked.dropna(subset=["correct"]).groupby("correct")}
        summary["h4"]["analysis_excess_by_correct"] = by_correct
        lines += ["", "Analysis excess when the prediction is correct: {}; incorrect: {}".format(
            _fmt(by_correct.get("True", {}), pct=True, signed=True),
            _fmt(by_correct.get("False", {}), pct=True, signed=True))]
    lines.append("")

    figures = write_figures(outdir, role_table, profile, curve, rationale_table)
    lines += ["## Figures", ""] + ["![{0}]({0})".format(Path(p).name) for p in figures]
    (outdir / "tables.md").write_text("\n".join(lines) + "\n")
    with open(outdir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=HERE / "stage3_outputs_modal" / "paper_v2")
    parser.add_argument("--stage2", default=str(HERE / "predex_ablation_results_v2.json"))
    parser.add_argument("--controls", default=str(HERE / "predex_stage2_controls.jsonl"))
    parser.add_argument("--outdir", type=Path, default=None,
                        help="default: <run-dir>/analysis")
    args = parser.parse_args()
    outdir = args.outdir or args.run_dir / "analysis"
    run(args.run_dir, args.stage2, args.controls, outdir)
    print((outdir / "tables.md").read_text())


if __name__ == "__main__":
    main()
