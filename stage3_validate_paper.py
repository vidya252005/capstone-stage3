#!/usr/bin/env python3
"""Post-hoc Stage-3 paper validation (no GPU / no re-attribution).

Rebuilds a canonical 1346-row table (fixes the case_id Cartesian join),
documents exclusions, and runs faithfulness / truncation / permutation checks
on the saved paper CSV + Stage-2 JSON.

    python stage3_validate_paper.py                      # v2 run (default)
    python stage3_validate_paper.py --run-dir stage3_outputs_modal/paper \
        --stage2 predex_ablation_results_named.json      # the old v1-era run
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stage3.audit import attach_stage2_measures
from stage3.metrics import correlate
from stage3.report import format_summary, summarise

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "stage3_outputs_modal" / "paper_v2"
DEFAULT_STAGE2 = HERE / "predex_ablation_results_v2.json"


def _configure(run_dir: Path, stage2: Path) -> None:
    global PAPER, RAW_CSV, STAGE2, OUT_CSV, OUT_JSON, OUT_TXT
    PAPER = run_dir
    RAW_CSV = PAPER / "stage3_results.csv"
    STAGE2 = stage2
    OUT_CSV = PAPER / "stage3_results_canonical.csv"
    OUT_JSON = PAPER / "validation_report.json"
    OUT_TXT = PAPER / "validation_report.txt"


_configure(DEFAULT_RUN_DIR, DEFAULT_STAGE2)

RNG = np.random.default_rng(42)
N_PERM_IOU = 200  # random word selections per case
N_PERM_CORR = 5000  # shuffle shortcut scores


def _strip_stage2_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [
        c
        for c in df.columns
        if c.startswith("delta_prob_")
        or c.startswith("shift_")
        or c
        in {
            "stage2_baseline_label",
            "stage2_baseline_prob",
            "shortcut_delta_prob_max",
            "shortcut_delta_prob_mean",
            "stage2_join_method",
            "stage2_case_id",
        }
    ]
    return df.drop(columns=drop, errors="ignore")


def build_canonical() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV)
    # Drop Stage-2 columns that were Cartesian-inflated, keep one row per PredEx index.
    base = _strip_stage2_columns(raw).drop_duplicates(subset=["row_index"], keep="first")
    base = base.sort_values("row_index").reset_index(drop=True)
    assert len(base) == 1346, "expected 1346 unique PredEx row_index values, got {}".format(
        len(base)
    )
    if not STAGE2.is_file():
        raise FileNotFoundError("Stage-2 results not found: {}".format(STAGE2))
    base = attach_stage2_measures(base, str(STAGE2))
    return base


def monte_carlo_case_iou(
    n_model: int, n_human: int, n_eligible: int, n_perm: int
) -> dict:
    """Empirical null: random k-word masks vs fixed human mask size."""
    if min(n_model, n_human, n_eligible) <= 0 or n_model > n_eligible:
        return {"mean_random": float("nan"), "p_ge_model": float("nan"), "model_iou": float("nan")}
    # We only have sizes in the CSV, not the masks. Hypergeometric intersection
    # sampling is exact for uniform random selection of the same size.
    # P(I=i) = Hypergeometric(N=n_eligible, K=n_human, n=n_model)
    intersections = RNG.hypergeometric(n_human, n_eligible - n_human, n_model, size=n_perm)
    unions = n_model + n_human - intersections
    random_ious = intersections / np.maximum(unions, 1)
    return {
        "mean_random": float(random_ious.mean()),
        "std_random": float(random_ious.std()),
        "p95_random": float(np.quantile(random_ious, 0.95)),
    }


def permutation_iou_test(scored: pd.DataFrame) -> dict:
    """Per-case: model IoU vs Monte Carlo random; aggregate mean and one-sided p."""
    model_ious = []
    random_means = []
    beats_p95 = 0
    eligible = 0
    for _, row in scored.iterrows():
        n_model = int(row.get("model_words_selected") or 0)
        n_human = int(row.get("human_words_visible") or 0)
        n_elig = int(row.get("n_words_visible") or 0)
        if n_model <= 0 or n_human <= 0 or n_elig <= 0:
            continue
        mc = monte_carlo_case_iou(n_model, n_human, n_elig, N_PERM_IOU)
        iou = float(row["iou"])
        if not np.isfinite(iou) or not np.isfinite(mc["mean_random"]):
            continue
        eligible += 1
        model_ious.append(iou)
        random_means.append(mc["mean_random"])
        if iou >= mc["p95_random"]:
            beats_p95 += 1

    model_ious = np.asarray(model_ious)
    random_means = np.asarray(random_means)
    # Paired: how often is model IoU > that case's mean random?
    paired_gt = float((model_ious > random_means).mean()) if eligible else float("nan")
    # Permutation of the *mean difference* under sign-flip / swap labels
    diffs = model_ious - random_means
    observed = float(diffs.mean())
    null = np.empty(N_PERM_CORR)
    for i in range(N_PERM_CORR):
        signs = RNG.choice([-1.0, 1.0], size=len(diffs))
        null[i] = float((signs * diffs).mean())
    p_one = float((null >= observed).mean())
    return {
        "n_cases": eligible,
        "mean_model_iou": float(model_ious.mean()),
        "mean_random_iou_mc": float(random_means.mean()),
        "mean_gap_model_minus_random": observed,
        "fraction_cases_model_gt_random_mean": paired_gt,
        "fraction_cases_model_ge_random_p95": beats_p95 / eligible if eligible else float("nan"),
        "permutation_p_one_sided_mean_gap": p_one,
        "n_perm": N_PERM_CORR,
        "n_mc_per_case": N_PERM_IOU,
    }


def permutation_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    base = correlate(x, y)
    obs = float(base["spearman_rho"])
    null = np.empty(N_PERM_CORR)
    for i in range(N_PERM_CORR):
        null[i] = float(correlate(x, RNG.permutation(y))["spearman_rho"])
    # two-sided
    p = float((np.abs(null) >= abs(obs)).mean())
    return {
        "n": int(len(x)),
        "observed_spearman": obs,
        "observed_pearson": float(base["pearson_r"]),
        "spearman_p_analytic": float(base.get("spearman_p", float("nan"))),
        "pearson_p_analytic": float(base.get("pearson_p", float("nan"))),
        "permutation_p_two_sided_spearman": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perm": N_PERM_CORR,
    }


def subgroup(df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    sub = df.loc[mask]
    scored = sub[sub["audit_status"] == "ok"]
    out = {
        "label": label,
        "n_total": int(len(sub)),
        "n_ok": int(len(scored)),
        "accuracy": float(sub["correct"].mean()) if "correct" in sub else float("nan"),
        "mean_iou": float(scored["iou"].mean()) if len(scored) else float("nan"),
        "mean_coverage": float(sub["rationale_coverage"].mean())
        if "rationale_coverage" in sub
        else float("nan"),
        "mean_shortcut": float(scored["shortcut_delta_prob_max"].mean())
        if len(scored) and "shortcut_delta_prob_max" in scored
        else float("nan"),
    }
    return out


def pick_heatmap_cases(scored: pd.DataFrame) -> dict:
    """Select low / mid / high IoU cases that already have HTML if possible."""
    html_dir = PAPER / "explanations"
    existing = {p.stem for p in html_dir.glob("*.html")} if html_dir.is_dir() else set()

    def slug(name: str) -> str:
        import re

        return re.sub(r"[^A-Za-z0-9]+", "_", str(name))[:80].strip("_")

    scored = scored.sort_values("iou")
    n = len(scored)
    bands = {
        "low": scored.head(max(1, n // 10)),
        "medium": scored.iloc[n // 2 - 5 : n // 2 + 5] if n > 10 else scored,
        "high": scored.tail(max(1, n // 10)),
    }
    picks = {}
    for band, block in bands.items():
        rows = []
        # Prefer cases that already have HTML samples
        for _, row in block.iterrows():
            s = slug(row["case_id"])
            rows.append(
                {
                    "case_id": row["case_id"],
                    "iou": float(row["iou"]),
                    "correct": bool(row["correct"]),
                    "faithfulness_gap": float(row["faithfulness_gap"])
                    if pd.notna(row.get("faithfulness_gap"))
                    else None,
                    "has_html": s in existing or any(s[:40] in e for e in existing),
                }
            )
        # take 5
        with_html = [r for r in rows if r["has_html"]]
        chosen = (with_html + rows)[:5]
        picks[band] = chosen
    return picks


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    args = parser.parse_args()
    _configure(args.run_dir, args.stage2)
    print("Building canonical 1346-row table…")
    df = build_canonical()
    df.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV, "rows=", len(df))

    # --- discrepancy documentation ---
    raw = pd.read_csv(RAW_CSV)
    dup_name = "STATE OF KARNATAKA Vs. A.B.MAHESHA ETC"
    discrepancy = {
        "raw_rows": int(len(raw)),
        "canonical_rows": int(len(df)),
        "unique_case_id_canonical": int(df["case_id"].nunique()),
        "duplicate_title": dup_name,
        "explanation": (
            "PredEx CSV has 1346 rows but only 1345 unique Case Names: "
            "'STATE OF KARNATAKA Vs. A.B.MAHESHA ETC' appears twice with "
            "different Input text (row_index 1151 and 1152). Stage-2 also has "
            "two entries under that title. Joining Stage-2 on case_id alone "
            "produced a 2×2 Cartesian product (4 rows), so 1346−2+4=1348. "
            "Canonical table joins on row_index instead (1:1)."
        ),
        "scoring_definition": {
            "headline_scored_ok": int((df["audit_status"] == "ok").sum()),
            "iou_non_null": int(df["iou"].notna().sum()),
            "iou_on_low_rationale_coverage": int(
                ((df["audit_status"] == "low_rationale_coverage") & df["iou"].notna()).sum()
            ),
            "note": (
                "Summary 'Cases scored' counts audit_status=='ok' only. "
                "IoU is still written for low_rationale_coverage (computed, then "
                "flagged), so CSV iou.notna() is larger (ok + low_coverage). "
                "rationale_out_of_context has no IoU."
            ),
        },
        "exclusions": {
            "ok": int((df["audit_status"] == "ok").sum()),
            "rationale_out_of_context": int(
                (df["audit_status"] == "rationale_out_of_context").sum()
            ),
            "low_rationale_coverage": int(
                (df["audit_status"] == "low_rationale_coverage").sum()
            ),
            "policy": (
                "Cases where fewer than 5 expert-rationale words fall inside the "
                "model's 16-chunk window are rationale_out_of_context. Cases where "
                "rationale character coverage in the document is <50% are "
                "low_rationale_coverage. Neither enters headline IoU means. In the "
                "v2 run, out-of-context cases are still attributed (role shares and "
                "faithfulness), they only lack IoU."
            ),
        },
    }

    summary = summarise(df)
    scored = df[df["audit_status"] == "ok"].copy()

    # Faithfulness subgroups
    gap = scored["faithfulness_gap"]
    faithful = gap > 0
    faith_groups = [
        subgroup(scored, faithful.fillna(False), "faithful_gap_gt_0"),
        subgroup(scored, (~faithful.fillna(True)), "nonfaithful_gap_le_0"),
    ]

    # Truncation
    trunc = df["truncated"].astype(bool)
    trunc_groups = [
        subgroup(df, trunc, "truncated"),
        subgroup(df, ~trunc, "not_truncated"),
    ]
    # also among ok only
    trunc_ok = [
        subgroup(scored, scored["truncated"].astype(bool), "truncated_and_ok"),
        subgroup(scored, ~scored["truncated"].astype(bool), "not_truncated_and_ok"),
    ]

    print("Monte Carlo IoU null…")
    iou_perm = permutation_iou_test(scored)

    print("Correlation permutation…")
    both = scored.dropna(subset=["iou", "shortcut_delta_prob_max"])
    corr_perm = permutation_correlation(
        both["shortcut_delta_prob_max"].to_numpy(float),
        both["iou"].to_numpy(float),
    )

    # Terciles on canonical
    q = both["shortcut_delta_prob_max"].quantile([1 / 3, 2 / 3])
    tercile = []
    for label, mask in [
        ("low_shortcut", both["shortcut_delta_prob_max"] <= q.iloc[0]),
        (
            "medium_shortcut",
            (both["shortcut_delta_prob_max"] > q.iloc[0])
            & (both["shortcut_delta_prob_max"] <= q.iloc[1]),
        ),
        ("high_shortcut", both["shortcut_delta_prob_max"] > q.iloc[1]),
    ]:
        part = both.loc[mask]
        tercile.append(
            {
                "label": label,
                "n": int(len(part)),
                "mean_iou": float(part["iou"].mean()),
                "mean_shortcut": float(part["shortcut_delta_prob_max"].mean()),
            }
        )

    # Attribution stability proxy (no GPU re-run)
    ig = scored["ig_convergence_delta"].dropna()
    stability_proxy = {
        "note": (
            "IG completeness gap |sum(attr) - (F(x) - F(baseline))| in logit units. "
            "The v2 run also stores it relative to max(|F(x) - F(baseline)|, 1) and "
            "recomputes cases above 5% with 128 steps."
        ),
        "n": int(len(ig)),
        "mean": float(ig.mean()),
        "median": float(ig.median()),
        "p90": float(ig.quantile(0.9)),
        "max": float(ig.max()),
        "fraction_delta_gt_0_25": float((ig > 0.25).mean()),
        "fraction_delta_gt_0_50": float((ig > 0.50).mean()),
    }
    if "ig_relative_delta" in scored:
        relative = scored["ig_relative_delta"].dropna()
        stability_proxy.update({
            "relative_median": float(relative.median()),
            "relative_max": float(relative.max()),
            "fraction_relative_within_5pct": float((relative <= 0.05).mean()),
            "fraction_recomputed_128_steps": float((scored["ig_steps_used"] > 32).mean()),
        })

    heatmaps = pick_heatmap_cases(scored)

    # Exclusion length check
    ooc = df[df["audit_status"] == "rationale_out_of_context"]
    ok = df[df["audit_status"] == "ok"]
    length_note = {
        "mean_n_words_ok": float(ok["n_words"].mean()),
        "mean_n_words_out_of_context": float(ooc["n_words"].mean()) if len(ooc) else None,
        "pct_ooc_truncated": float(ooc["truncated"].mean()) if len(ooc) else None,
        "pct_ok_truncated": float(ok["truncated"].mean()) if len(ok) else None,
    }

    report = {
        "discrepancy": discrepancy,
        "headline_summary": summary,
        "faithfulness_subgroups": faith_groups,
        "truncation_subgroups_all_statuses": trunc_groups,
        "truncation_subgroups_ok_only": trunc_ok,
        "exclusion_length": length_note,
        "iou_vs_random_permutation": iou_perm,
        "shortcut_iou_correlation_permutation": corr_perm,
        "shortcut_terciles": tercile,
        "attribution_stability_proxy": stability_proxy,
        "manual_heatmap_picks": heatmaps,
        "frozen_definitions": {
            "dataset": "predex_test_deduped.csv (1346 rows; 1345 unique Case Names)",
            "case_unit": "CSV row_index (not Case Name)",
            "iou": "word-level IoU @ human-size top-k on visible words",
            "headline_filter": "audit_status == ok",
            "random_baseline": "analytic expected_random_iou + Monte Carlo hypergeometric",
            "faithfulness": "logit drop top-k minus bottom-k; pass if gap > 0",
            "shortcut_measure": "shortcut_delta_prob_max over spurious Stage-2 variants",
            "stage2_join": "row_index (1:1); case_id titles verified where present",
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))

    lines = [
        format_summary(summary),
        "",
        "=" * 68,
        "VALIDATION / DISCREPANCY RESOLUTION",
        "=" * 68,
        discrepancy["explanation"],
        "",
        "Canonical PredEx cases     : {}".format(discrepancy["canonical_rows"]),
        "Unique Case Names          : {}".format(discrepancy["unique_case_id_canonical"]),
        "Headline scored (ok)       : {}".format(
            discrepancy["scoring_definition"]["headline_scored_ok"]
        ),
        "CSV IoU non-null           : {}".format(
            discrepancy["scoring_definition"]["iou_non_null"]
        ),
        discrepancy["scoring_definition"]["note"],
        "",
        "Exclusions",
        "  ok                         {}".format(discrepancy["exclusions"]["ok"]),
        "  rationale_out_of_context   {}".format(
            discrepancy["exclusions"]["rationale_out_of_context"]
        ),
        "  low_rationale_coverage     {}".format(
            discrepancy["exclusions"]["low_rationale_coverage"]
        ),
        discrepancy["exclusions"]["policy"],
        "",
        "Length vs exclusion",
        "  mean n_words (ok)          {:.0f}".format(length_note["mean_n_words_ok"]),
        "  mean n_words (OOC)         {:.0f}".format(
            length_note["mean_n_words_out_of_context"] or float("nan")
        ),
        "  pct truncated (ok)         {:.1%}".format(length_note["pct_ok_truncated"] or 0),
        "  pct truncated (OOC)        {:.1%}".format(length_note["pct_ooc_truncated"] or 0),
        "",
        "Faithfulness subgroups (ok cases)",
    ]
    for g in faith_groups:
        lines.append(
            "  {label}: n={n_ok}  IoU={mean_iou:.3f}  shortcut={mean_shortcut:.3f}".format(
                **g
            )
        )
    lines.append("")
    lines.append("Truncation (ok cases)")
    for g in trunc_ok:
        lines.append(
            "  {label}: n={n_ok}  IoU={mean_iou:.3f}  cov={mean_coverage:.3f}  "
            "shortcut={mean_shortcut:.3f}  acc={accuracy:.3f}".format(**g)
        )
    lines.append("")
    lines.append("IoU vs random (Monte Carlo + permutation on mean gap)")
    for k, v in iou_perm.items():
        lines.append("  {}: {}".format(k, v if not isinstance(v, float) else round(v, 4)))
    lines.append("")
    lines.append("Stage-2 ↔ Stage-3 correlation permutation")
    for k, v in corr_perm.items():
        lines.append("  {}: {}".format(k, v if not isinstance(v, float) else round(v, 4)))
    lines.append("")
    lines.append("Shortcut terciles (canonical)")
    for t in tercile:
        lines.append(
            "  {label}: n={n}  IoU={mean_iou:.3f}  shortcut={mean_shortcut:.3f}".format(**t)
        )
    lines.append("")
    lines.append("Attribution stability proxy (IG convergence delta)")
    for k, v in stability_proxy.items():
        if k == "note":
            lines.append("  " + v)
        else:
            lines.append("  {}: {}".format(k, v if not isinstance(v, float) else round(v, 4)))
    lines.append("")
    lines.append("Manual heatmap picks (inspect explanations/)")
    for band, rows in heatmaps.items():
        lines.append("  [{}]".format(band))
        for r in rows:
            lines.append(
                "    IoU={iou:.3f} correct={correct} html={has_html}  {case_id}".format(**r)
            )
    lines.append("")
    lines.append("STAGE 3 METHODOLOGY — TREAT AS FROZEN after reviewing this report.")
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text)
    print(text)
    print("Wrote", OUT_TXT)
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
