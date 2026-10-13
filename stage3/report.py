"""Audit summary, per-case HTML heatmaps and distribution plots."""
from __future__ import annotations

import html
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import bootstrap_ci, correlate

_HEADLINE_STATUSES = ("ok",)


def _finite(series) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    return values[np.isfinite(values)]


def summarise(results_df, shortcut_column: str = "shortcut_delta_prob_max") -> Dict:
    """Build the audit summary from a per-case results frame.

    Headline numbers use only cases with ``audit_status == "ok"``. Cases whose
    rationale sits outside the model's context window, or could not be located in the
    document, are counted separately: folding them in as IoU=0 would report an
    alignment failure that is really a truncation or annotation artefact.
    """
    total = len(results_df)
    status_counts = results_df["audit_status"].value_counts().to_dict()
    scored = results_df[results_df["audit_status"].isin(_HEADLINE_STATUSES)]
    # IoU is still written for low_rationale_coverage (computed then flagged).
    # Headline means use audit_status=="ok" only — do not confuse with iou.notna().
    n_iou_present = (
        int(results_df["iou"].notna().sum()) if "iou" in results_df else 0
    )

    summary: Dict[str, object] = {
        "n_cases_total": int(total),
        "n_cases_scored": int(len(scored)),
        "n_cases_with_iou": n_iou_present,
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }

    if "correct" in results_df:
        graded = results_df["correct"].dropna()
        summary["accuracy"] = float(graded.mean()) if len(graded) else None

    if len(scored) == 0:
        return summary

    for column in ["iou", "precision", "recall", "f1", "iou_random"]:
        if column in scored:
            summary[column] = bootstrap_ci(scored[column].tolist())
    summary["iou_median"] = float(np.nanmedian(_finite(scored.get("iou", []))))

    for column in scored.columns:
        if column.startswith("iou_p") and not column.startswith("iou_random"):
            summary[column] = bootstrap_ci(scored[column].tolist())

    if "human_supported_attribution_share" in scored:
        summary["human_supported_attribution_share"] = bootstrap_ci(
            scored["human_supported_attribution_share"].tolist()
        )
    if "attribution_ratio_human_nonhuman" in scored:
        summary["attribution_ratio_human_nonhuman"] = bootstrap_ci(
            scored["attribution_ratio_human_nonhuman"].tolist()
        )
    # Role measures do not depend on the expert rationale, so they use every case
    # with an attribution, not only the rationale-scored ones.
    attributed = (
        results_df[results_df["ig_convergence_delta"].notna()]
        if "ig_convergence_delta" in results_df
        else scored
    )
    summary["n_cases_attributed"] = int(len(attributed))
    if "post_decisional_attribution_share" in attributed:
        summary["post_decisional_attribution_share"] = bootstrap_ci(
            attributed["post_decisional_attribution_share"].tolist()
        )
        summary["post_decisional_word_share"] = bootstrap_ci(
            attributed["post_decisional_word_share"].tolist()
        )
    if "ig_relative_delta" in attributed:
        relative = _finite(attributed["ig_relative_delta"])
        steps = _finite(attributed.get("ig_steps_used", []))
        if len(relative):
            summary["ig_relative_delta"] = {
                "median": float(np.median(relative)),
                "share_within_5pct": float(np.mean(relative <= 0.05)),
                "share_retried": (
                    float(np.mean(steps > steps.min())) if len(steps) else None
                ),
            }

    role_columns = sorted(c for c in attributed.columns if c.startswith("role_share_"))
    if role_columns:
        summary["role_attribution_shares"] = {
            c.replace("role_share_", ""): float(np.nanmean(_finite(attributed[c])))
            for c in role_columns
        }
        word_columns = {c: "role_words_" + c[len("role_share_"):] for c in role_columns}
        summary["role_word_shares"] = {
            c.replace("role_share_", ""): float(np.nanmean(_finite(attributed[w])))
            for c, w in word_columns.items()
            if w in attributed
        }

    if "faithfulness_gap" in scored:
        summary["faithfulness"] = {
            "mean_gap": float(np.nanmean(_finite(scored["faithfulness_gap"]))),
            "pass_rate": float(scored["faithfulness_pass"].dropna().mean())
            if "faithfulness_pass" in scored
            else None,
            "mean_drop_top": float(np.nanmean(_finite(scored["faithfulness_drop_top"]))),
            "mean_drop_bottom": float(
                np.nanmean(_finite(scored["faithfulness_drop_bottom"]))
            ),
        }

    if "ig_convergence_delta" in scored:
        deltas = _finite(scored["ig_convergence_delta"])
        if deltas.size:
            summary["ig_convergence_delta"] = {
                "mean": float(deltas.mean()),
                "median": float(np.median(deltas)),
                "max": float(deltas.max()),
            }

    if "rationale_coverage" in results_df:
        coverage = _finite(results_df["rationale_coverage"])
        if coverage.size:
            summary["rationale_alignment"] = {
                "mean_coverage": float(coverage.mean()),
                "median_coverage": float(np.median(coverage)),
                "pct_above_95": float((coverage >= 0.95).mean()),
                "pct_below_50": float((coverage < 0.50).mean()),
            }
    if "truncated" in results_df:
        summary["pct_documents_truncated"] = float(results_df["truncated"].dropna().mean())
    if "human_words_out_of_context" in results_df:
        visible = _finite(results_df["human_words_visible"])
        outside = _finite(results_df["human_words_out_of_context"])
        if visible.size and outside.size:
            summary["pct_rationale_outside_context"] = float(
                outside.sum() / max(1.0, visible.sum() + outside.sum())
            )

    if "correct" in scored:
        by_correct = scored.groupby("correct")["iou"].agg(["mean", "count"])
        summary["iou_by_correctness"] = {
            str(idx): float(row["mean"]) for idx, row in by_correct.iterrows()
        }
        summary["iou_by_correctness_n"] = {
            str(idx): int(row["count"]) for idx, row in by_correct.iterrows()
        }
    if "stage2_join_method" in results_df.columns:
        methods = [
            str(m) for m in results_df["stage2_join_method"].dropna().unique().tolist()
        ]
        if methods:
            summary["stage2_join_method"] = ", ".join(methods)

    if shortcut_column in scored and "iou" in scored:
        summary["shortcut_vs_iou"] = correlate(scored[shortcut_column], scored["iou"])
        summary["shortcut_vs_iou"]["shortcut_measure"] = shortcut_column
        groups = _shortcut_groups(scored, shortcut_column)
        if groups:
            summary["iou_by_shortcut_group"] = groups
    return summary


def _shortcut_groups(scored, shortcut_column: str) -> Dict[str, Dict[str, float]]:
    """Mean IoU per tercile of Stage-2 shortcut sensitivity.

    Terciles are used rather than fixed cut-offs so the grouping comes from the
    observed distribution instead of a threshold chosen in advance.
    """
    values = scored[shortcut_column].astype(float)
    keep = np.isfinite(values)
    if keep.sum() < 30:
        return {}
    quantiles = np.nanquantile(values[keep], [1 / 3, 2 / 3])
    groups: Dict[str, Dict[str, float]] = {}
    bounds = [(-np.inf, quantiles[0]), (quantiles[0], quantiles[1]), (quantiles[1], np.inf)]
    for name, (low, high) in zip(["low", "medium", "high"], bounds):
        selected = scored[keep & (values > low) & (values <= high)]
        if len(selected) == 0:
            continue
        groups[name + "_shortcut"] = {
            "n": int(len(selected)),
            "mean_iou": float(np.nanmean(_finite(selected["iou"]))),
            "mean_shortcut": float(np.nanmean(_finite(selected[shortcut_column]))),
        }
    return groups


def format_summary(summary: Dict) -> str:
    """Render the summary as the plain-text audit report."""

    def interval(key: str) -> str:
        block = summary.get(key)
        if not isinstance(block, dict) or not np.isfinite(block.get("mean", np.nan)):
            return "n/a"
        return "{:.3f}  [{:.3f}, {:.3f}]".format(
            block["mean"], block["ci_low"], block["ci_high"]
        )

    lines: List[str] = []
    lines.append("=" * 68)
    lines.append("STAGE 3 - SHORTCUT LEARNING AUDIT")
    lines.append("=" * 68)
    lines.append("Cases evaluated        : {}".format(summary.get("n_cases_total")))
    lines.append(
        "Cases scored (ok)      : {}".format(summary.get("n_cases_scored"))
    )
    if summary.get("n_cases_with_iou") is not None:
        lines.append(
            "Cases with IoU filled  : {}  "
            "(ok + low_rationale_coverage; headline IoU uses ok only)".format(
                summary.get("n_cases_with_iou")
            )
        )
    accuracy = summary.get("accuracy")
    if accuracy is not None:
        lines.append("Accuracy               : {:.2%}".format(accuracy))
    lines.append("")
    lines.append("Case status")
    for status, count in sorted(summary.get("status_counts", {}).items()):
        lines.append("  {:<26} {}".format(status, count))

    alignment = summary.get("rationale_alignment")
    if alignment:
        lines.append("")
        lines.append("Expert-rationale alignment quality")
        lines.append("  mean coverage            {:.3f}".format(alignment["mean_coverage"]))
        lines.append("  median coverage          {:.3f}".format(alignment["median_coverage"]))
        lines.append("  cases >=95% coverage     {:.1%}".format(alignment["pct_above_95"]))
        lines.append("  cases <50% coverage      {:.1%}".format(alignment["pct_below_50"]))
    if summary.get("pct_documents_truncated") is not None:
        lines.append(
            "  documents truncated      {:.1%}".format(summary["pct_documents_truncated"])
        )
    if summary.get("pct_rationale_outside_context") is not None:
        lines.append(
            "  rationale words outside model context  {:.1%}".format(
                summary["pct_rationale_outside_context"]
            )
        )

    lines.append("")
    lines.append("Human-rationale alignment (mean [95% CI])")
    lines.append("  IoU @ human-size       {}".format(interval("iou")))
    lines.append("  IoU @ random baseline  {}".format(interval("iou_random")))
    lines.append("  Precision              {}".format(interval("precision")))
    lines.append("  Recall                 {}".format(interval("recall")))
    lines.append("  F1                     {}".format(interval("f1")))
    if "iou_median" in summary and np.isfinite(summary["iou_median"]):
        lines.append("  IoU median             {:.3f}".format(summary["iou_median"]))
    by_correct = summary.get("iou_by_correctness") or {}
    by_n = summary.get("iou_by_correctness_n") or {}
    if by_correct:
        lines.append("  IoU | correct preds    {}".format(
            _fmt_correctness(by_correct, by_n, True)))
        lines.append("  IoU | incorrect preds  {}".format(
            _fmt_correctness(by_correct, by_n, False)))
    for key in sorted(k for k in summary if k.startswith("iou_p")):
        lines.append("  IoU @ top {:<10} {}".format(key.replace("iou_p", "") + "%", interval(key)))

    lines.append("")
    lines.append("Threshold-free attribution measures")
    lines.append("  Human-supported attribution share  {}".format(
        interval("human_supported_attribution_share")))
    lines.append("  Attribution ratio human:non-human  {}".format(
        interval("attribution_ratio_human_nonhuman")))
    if "post_decisional_attribution_share" in summary:
        lines.append("  Post-decisional attribution share  {}  (n={} cases)".format(
            interval("post_decisional_attribution_share"), summary.get("n_cases_attributed")))
        lines.append("  Post-decisional share of words     {}".format(
            interval("post_decisional_word_share")))
    convergence = summary.get("ig_relative_delta")
    if convergence:
        lines.append("  IG relative delta median {:.3f}; within 5%: {:.1%}; retried: {}".format(
            convergence["median"], convergence["share_within_5pct"],
            "n/a" if convergence["share_retried"] is None
            else "{:.1%}".format(convergence["share_retried"])))

    roles = summary.get("role_attribution_shares")
    if roles:
        lines.append("")
        lines.append("Attribution mass by rhetorical role (vs share of words in that role)")
        words = summary.get("role_word_shares", {})
        for role, share in sorted(roles.items(), key=lambda kv: -kv[1]):
            if role in words and words[role] > 0:
                lines.append("  {:<20} {:6.1%}  words {:6.1%}  ratio {:.2f}".format(
                    role, share, words[role], share / words[role]))
            else:
                lines.append("  {:<20} {:6.1%}".format(role, share))

    faithfulness = summary.get("faithfulness")
    if faithfulness:
        lines.append("")
        lines.append("Faithfulness (masking top vs bottom attributed words)")
        lines.append("  mean logit drop, top     {:.4f}".format(faithfulness["mean_drop_top"]))
        lines.append("  mean logit drop, bottom  {:.4f}".format(faithfulness["mean_drop_bottom"]))
        lines.append("  mean gap (want > 0)      {:.4f}".format(faithfulness["mean_gap"]))
        if faithfulness.get("pass_rate") is not None:
            lines.append("  cases passing            {:.1%}".format(faithfulness["pass_rate"]))

    delta = summary.get("ig_convergence_delta")
    if delta:
        lines.append("")
        lines.append("Attribution quality")
        lines.append("  IG convergence delta  mean {:.4f} / max {:.4f}".format(
            delta["mean"], delta["max"]))

    correlation = summary.get("shortcut_vs_iou")
    if correlation and "spearman_rho" in correlation:
        lines.append("")
        lines.append("Stage-2 shortcut sensitivity vs Stage-3 alignment")
        if summary.get("stage2_join_method"):
            lines.append("  join      {}".format(summary["stage2_join_method"]))
        lines.append("  measure   {}".format(correlation.get("shortcut_measure")))
        lines.append("  n         {}".format(correlation["n"]))
        lines.append("  Spearman  rho = {:+.3f}  (p = {:.3g})".format(
            correlation["spearman_rho"], correlation["spearman_p"]))
        lines.append("  Pearson   r   = {:+.3f}  (p = {:.3g})".format(
            correlation["pearson_r"], correlation["pearson_p"]))
    groups = summary.get("iou_by_shortcut_group")
    if groups:
        lines.append("")
        lines.append("  Mean IoU by Stage-2 shortcut tercile")
        for name in ["low_shortcut", "medium_shortcut", "high_shortcut"]:
            if name in groups:
                block = groups[name]
                lines.append("    {:<18} n={:<5} IoU={:.3f}  (shift={:.3f})".format(
                    name, block["n"], block["mean_iou"], block["mean_shortcut"]))

    lines.append("")
    lines.append("-" * 68)
    if faithfulness:
        lines.append(
            "Reading this: low IoU on its own is a rationale-alignment finding, not\n"
            "proof of shortcut learning. The shortcut claim rests on low alignment\n"
            "together with the Stage-2 behavioural sensitivity, and only holds if the\n"
            "faithfulness gap above is positive. The role-level results (attribution\n"
            "on the Analysis, adjusted for position) are in analysis/tables.md."
        )
    else:
        lines.append(
            "Reading this: low IoU on its own is a rationale-alignment finding, not\n"
            "proof of shortcut learning. Faithfulness was NOT measured in this run,\n"
            "so do not treat the heatmap or the Stage-2 correlation as a shortcut verdict."
        )
        join = summary.get("stage2_join_method") or ""
        if "row_position" in join:
            lines.append(
                "Stage-2 was joined by row position, not case id — do not report that\n"
                "correlation as a final result until Stage 2 is re-run with real names."
            )
    lines.append("-" * 68)
    return "\n".join(lines)


def _fmt_correctness(by_correct, by_n, flag):
    for key in (str(flag), str(int(flag)), flag):
        if key in by_correct:
            n = by_n.get(key, by_n.get(str(flag), "?"))
            return "{:.3f}  (n={})".format(by_correct[key], n)
    return "n/a"


# ─────────────────────────────────────────────────────────────────────────────
# Per-case HTML heatmap
# ─────────────────────────────────────────────────────────────────────────────
_HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Stage 3 audit &mdash; {case_id}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
        max-width: 60rem; line-height: 1.75; color: #1a1a1a; }}
 table.meta {{ border-collapse: collapse; margin-bottom: 1.5rem; font-size: .9rem; }}
 table.meta td {{ border: 1px solid #ddd; padding: .3rem .7rem; }}
 table.meta td:first-child {{ background: #fafafa; font-weight: 600; }}
 .legend span {{ padding: .15rem .45rem; margin-right: .5rem; border-radius: 3px; }}
 .human {{ border-bottom: 3px solid #2266cc; }}
 .doc {{ background: #fff; padding: 1rem; border: 1px solid #e5e5e5; border-radius: 6px; }}
 .invisible {{ opacity: .35; }}
 h2 {{ font-size: 1.05rem; margin-top: 2rem; }}
</style>
<h1>Stage 3 audit &mdash; {case_id}</h1>
<table class="meta">{meta}</table>
<p class="legend">
  <span style="background:rgba(220,60,40,.55)">model attribution</span>
  <span class="human">expert rationale</span>
  <span class="invisible">outside model context</span>
</p>
<h2>Document</h2>
<div class="doc">{body}</div>
"""


def write_case_html(audit, path: str, max_words: int = 6000) -> str:
    """Write a three-signal heatmap: text, model attribution, expert rationale.

    Attribution intensity is shown as background shading, expert rationale as an
    underline, and words beyond the model's context window are dimmed. Overlap and
    disagreement are then visible directly, which is the fastest way to catch an
    alignment bug that a summary statistic would hide.
    """
    space = audit.space
    record = audit.record
    if space is None:
        raise ValueError("write_case_html needs an audit produced with keep_arrays=True")

    # Cases that stopped before attribution (rationale out of context, unresolved
    # annotation) still get a page: seeing why a case was skipped is the whole point.
    scores = audit.word_scores
    if scores is None:
        scores = np.zeros(space.n_words, dtype=float)

    human = audit.human_mask if audit.human_mask is not None else np.zeros(space.n_words, bool)
    selected = audit.model_mask if audit.model_mask is not None else np.zeros(space.n_words, bool)
    visible = space.word_visible

    finite = scores[np.isfinite(scores)]
    ceiling = np.percentile(finite[finite > 0], 99) if (finite > 0).any() else 1.0
    ceiling = max(float(ceiling), 1e-12)

    pieces: List[str] = []
    cursor = 0
    limit = min(space.n_words, max_words)
    for index in range(limit):
        start, end = space.word_spans[index]
        pieces.append(html.escape(space.document[cursor:start]))
        word = html.escape(space.document[start:end])
        intensity = min(1.0, float(scores[index]) / ceiling) if scores[index] > 0 else 0.0
        styles = []
        if selected[index]:
            styles.append("background: rgba(220,60,40,{:.2f})".format(0.25 + 0.5 * intensity))
        elif intensity > 0.05:
            styles.append("background: rgba(220,60,40,{:.2f})".format(0.30 * intensity))
        if human[index]:
            styles.append("border-bottom: 3px solid #2266cc")
        classes = "" if visible[index] else ' class="invisible"'
        if styles:
            pieces.append('<span{} style="{}">{}</span>'.format(classes, ";".join(styles), word))
        elif classes:
            pieces.append("<span{}>{}</span>".format(classes, word))
        else:
            pieces.append(word)
        cursor = end
    if limit < space.n_words:
        pieces.append(
            "<p><em>[truncated for display after {} words]</em></p>".format(max_words)
        )
    else:
        pieces.append(html.escape(space.document[cursor:]))

    interesting = [
        "true_label", "predicted_label", "correct", "prediction_confidence",
        "iou", "precision", "recall", "f1", "human_supported_attribution_share",
        "human_words_visible", "human_words_out_of_context", "model_words_selected",
        "rationale_coverage", "rationale_status", "truncated", "n_words",
        "faithfulness_gap", "ig_convergence_delta", "audit_status",
    ]
    meta = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(html.escape(k), html.escape(str(record[k])))
        for k in interesting
        if k in record
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(
            _HTML_TEMPLATE.format(
                case_id=html.escape(str(record.get("case_id", "case"))),
                meta=meta,
                body="".join(pieces),
            )
        )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def write_plots(results_df, outdir: str, shortcut_column: str = "shortcut_delta_prob_max"):
    """IoU distribution, IoU vs Stage-2 shortcut shift, and role attribution shares."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    scored = results_df[results_df["audit_status"] == "ok"]
    written: List[str] = []
    if len(scored) == 0:
        return written

    iou = _finite(scored["iou"])
    if iou.size:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(iou, bins=30, color="steelblue", edgecolor="black")
        axis.axvline(iou.mean(), color="crimson", linestyle="--",
                     label="mean {:.3f}".format(iou.mean()))
        if "iou_random" in scored:
            random_iou = _finite(scored["iou_random"])
            if random_iou.size:
                axis.axvline(random_iou.mean(), color="grey", linestyle=":",
                             label="random {:.3f}".format(random_iou.mean()))
        axis.set_xlabel("IoU (model top-k vs expert rationale)")
        axis.set_ylabel("cases")
        axis.set_title("Stage 3: human-rationale alignment")
        axis.legend()
        figure.tight_layout()
        path = os.path.join(outdir, "iou_distribution.png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    if shortcut_column in scored:
        pair = scored[[shortcut_column, "iou"]].astype(float).dropna()
        if len(pair) >= 10:
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.scatter(pair[shortcut_column], pair["iou"], s=12, alpha=0.5,
                         color="darkslateblue")
            slope, intercept = np.polyfit(pair[shortcut_column], pair["iou"], 1)
            xs = np.linspace(pair[shortcut_column].min(), pair[shortcut_column].max(), 50)
            axis.plot(xs, slope * xs + intercept, color="crimson", linewidth=2)
            axis.set_xlabel("Stage-2 shortcut sensitivity (max |delta P|)")
            axis.set_ylabel("Stage-3 IoU")
            axis.set_title("Shortcut sensitivity vs rationale alignment")
            figure.tight_layout()
            path = os.path.join(outdir, "shortcut_vs_iou.png")
            figure.savefig(path, dpi=150)
            plt.close(figure)
            written.append(path)

    role_columns = sorted(c for c in scored.columns if c.startswith("role_share_"))
    if role_columns:
        means = {c.replace("role_share_", ""): float(np.nanmean(_finite(scored[c])))
                 for c in role_columns}
        means = {k: v for k, v in means.items() if np.isfinite(v)}
        if means:
            order = sorted(means.items(), key=lambda kv: kv[1])
            figure, axis = plt.subplots(figsize=(7, 4.5))
            axis.barh([k for k, _ in order], [v for _, v in order],
                      color="steelblue", edgecolor="black")
            axis.set_xlabel("share of total attribution mass")
            axis.set_title("Where the model's attribution lands, by rhetorical role")
            figure.tight_layout()
            path = os.path.join(outdir, "attribution_roles.png")
            figure.savefig(path, dpi=150)
            plt.close(figure)
            written.append(path)
    return written


def save_summary(summary: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, default=float)
    return path
