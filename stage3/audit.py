"""ShortcutAuditor: per-case evidence alignment and the dataset-level audit.

The audit answers three questions per case and reports them side by side:

1. What did the model predict, and how confidently?
2. Which words drove that prediction?
3. Do those words overlap the evidence a legal expert marked as pivotal?

Low overlap on its own is a rationale-alignment finding, not a shortcut verdict --
it is equally consistent with an incomplete annotation, a noisy explanation method, or
genuinely distributed reasoning. The shortcut claim needs the Stage-2 behavioural
evidence alongside it, which is why :meth:`ShortcutAuditor.attach_stage2` exists.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .alignment import DocumentSpace, align_rationale, select_top_words
from .config import (
    MIN_RATIONALE_COVERAGE,
    MIN_VISIBLE_HUMAN_WORDS,
    POST_DECISIONAL_ROLES,
    TOPK_PERCENTS,
)
from .metrics import (
    attribution_mass_share,
    binary_overlap,
    expected_random_iou,
    faithfulness_scores,
    mean_attribution_inside_outside,
    role_attribution_shares,
)
from .predex import PredExCase, split_rationale_sentences


@dataclass
class CaseAudit:
    """Everything the audit knows about one case."""

    record: Dict[str, object]
    space: Optional[DocumentSpace] = None
    human_mask: Optional[np.ndarray] = None
    model_mask: Optional[np.ndarray] = None
    word_scores: Optional[np.ndarray] = None
    word_scores_signed: Optional[np.ndarray] = None
    spans: List = field(default_factory=list)
    eligible: Optional[np.ndarray] = None
    role_masks: Dict[str, np.ndarray] = field(default_factory=dict)


class ShortcutAuditor:
    """Runs the Stage-3 audit over PredEx cases for any :class:`ModelAdapter`."""

    def __init__(
        self,
        adapter,
        attribution_method: str = "integrated_gradients",
        primary_selection: str = "human_size",
        topk_percents: Sequence[float] = TOPK_PERCENTS,
        run_faithfulness: bool = True,
        faithfulness_percent: float = 0.10,
        role_lookup: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> None:
        self.adapter = adapter
        self.attribution_method = attribution_method
        self.primary_selection = primary_selection
        self.topk_percents = tuple(topk_percents)
        self.run_faithfulness = run_faithfulness
        self.faithfulness_percent = faithfulness_percent
        self.role_lookup = role_lookup or {}
        self.records: List[Dict[str, object]] = []

    # ── one case ────────────────────────────────────────────────────────────
    def evaluate_case(self, case: PredExCase, keep_arrays: bool = False) -> CaseAudit:
        record: Dict[str, object] = {
            "case_id": case.case_id,
            "row_index": case.index,
            "true_label": int(case.label),
            "model": self.adapter.name,
        }

        space = self.adapter.document_space(case.document)
        record.update(
            {
                "n_words": space.n_words,
                "n_tokens": space.n_tokens_total,
                "n_chunks": space.n_chunks,
                "truncated": bool(space.truncated),
                "n_words_visible": int(space.word_visible.sum()),
                "visible_word_fraction": (
                    float(space.word_visible.mean()) if space.n_words else 0.0
                ),
            }
        )

        # ── expert rationale -> word mask ───────────────────────────────────
        sentences = split_rationale_sentences(case.rationale_text)
        alignment = align_rationale(case.document, sentences)
        record.update(alignment.to_dict())
        human_mask = space.words_in_spans(alignment.spans)
        human_visible = np.logical_and(human_mask, space.word_visible)
        record.update(
            {
                "human_words_total": int(human_mask.sum()),
                "human_words_visible": int(human_visible.sum()),
                "human_words_out_of_context": int(human_mask.sum() - human_visible.sum()),
                "human_word_fraction": (
                    float(human_mask.mean()) if space.n_words else 0.0
                ),
            }
        )

        # ── prediction ──────────────────────────────────────────────────────
        prediction = self.adapter.predict(case.document, space)
        record.update(
            {
                "predicted_label": prediction.label,
                "prediction_confidence": round(float(prediction.confidence), 4),
                "prob_positive": round(float(prediction.probabilities[1]), 6),
                "logit_predicted": round(prediction.logit(prediction.label), 6),
                "margin_predicted": round(prediction.margin(prediction.label), 6),
                "correct": bool(prediction.label == case.label),
            }
        )

        eligible = space.word_visible.copy()
        if eligible.sum() == 0:
            record["audit_status"] = "no_visible_words"
            self.records.append(record)
            return CaseAudit(record=record)
        n_human = int(human_visible.sum())
        rationale_visible = n_human >= MIN_VISIBLE_HUMAN_WORDS

        # ── attribution ─────────────────────────────────────────────────────
        # Run for every case the model can see, including those whose rationale is
        # outside the window: where attribution lands by rhetorical role does not
        # need the rationale, and skipping them would drop the longest judgments.
        attribution = self.adapter.attribute(
            case.document,
            space=space,
            target=prediction.label,
            method=self.attribution_method,
        )
        record.update(attribution.to_dict())
        scores = attribution.word_scores

        # ── overlap with the expert rationale at several selection sizes ────
        primary_mask = None
        if rationale_visible:
            n_eligible = int(eligible.sum())
            selections: Dict[str, int] = {"human_size": n_human}
            for percent in self.topk_percents:
                selections["p{:02d}".format(int(round(percent * 100)))] = max(
                    1, int(round(percent * n_eligible))
                )
            for name, k in selections.items():
                model_mask = select_top_words(scores, eligible, k)
                overlap = binary_overlap(model_mask, human_visible)
                suffix = "" if name == self.primary_selection else "_" + name
                record["iou" + suffix] = round(overlap["iou"], 4)
                record["precision" + suffix] = round(overlap["precision"], 4)
                record["recall" + suffix] = round(overlap["recall"], 4)
                record["f1" + suffix] = round(overlap["f1"], 4)
                record["iou_random" + suffix] = round(
                    expected_random_iou(int(model_mask.sum()), n_human, n_eligible), 4
                )
                if name == self.primary_selection:
                    primary_mask = model_mask
                    record["model_words_selected"] = int(model_mask.sum())

            # ── threshold-free measures ─────────────────────────────────────
            record["human_supported_attribution_share"] = round(
                attribution_mass_share(scores, human_visible, eligible), 4
            )
            record.update(
                {
                    k: (None if not np.isfinite(v) else round(float(v), 6))
                    for k, v in mean_attribution_inside_outside(
                        scores, human_visible, eligible
                    ).items()
                }
            )

        # ── rhetorical-role attribution shares ──────────────────────────────
        role_masks = self._role_masks(case, space)
        if role_masks:
            shares = role_attribution_shares(
                scores, role_masks, eligible, POST_DECISIONAL_ROLES
            )
            record.update({k: round(v, 4) for k, v in shares.items()})

        # ── faithfulness ────────────────────────────────────────────────────
        if self.run_faithfulness:
            record.update(self._faithfulness(space, scores, eligible, prediction))

        if not rationale_visible:
            # No IoU for these: scoring them as IoU=0 would blame the model for an
            # annotation that is not inside its context window at all.
            record["audit_status"] = (
                "rationale_out_of_context"
                if human_mask.sum() >= MIN_VISIBLE_HUMAN_WORDS
                else "rationale_unresolved"
            )
        else:
            record["audit_status"] = (
                "ok"
                if alignment.coverage >= MIN_RATIONALE_COVERAGE
                else "low_rationale_coverage"
            )
        self.records.append(record)

        return CaseAudit(
            record=record,
            space=space if keep_arrays else None,
            human_mask=human_visible if keep_arrays else None,
            model_mask=primary_mask if keep_arrays else None,
            word_scores=scores if keep_arrays else None,
            word_scores_signed=attribution.word_scores_signed if keep_arrays else None,
            spans=alignment.spans if keep_arrays else [],
            eligible=eligible if keep_arrays else None,
            role_masks=role_masks if keep_arrays else {},
        )

    # ── helpers ─────────────────────────────────────────────────────────────
    def _role_masks(self, case: PredExCase, space: DocumentSpace) -> Dict[str, np.ndarray]:
        """Word masks per rhetorical role, reusing the rationale span matcher.

        The Stage-2 segmentation stores role sentences as text, so they are located in
        the document with exactly the same alignment machinery as the expert rationale.
        """
        # Row index first: case names are not unique in PredEx (one title appears
        # twice), so a name lookup would hand one of those cases the other's roles.
        buckets = self.role_lookup.get(str(case.index))
        if not buckets:
            buckets = self.role_lookup.get(case.case_id)
        if not buckets:
            return {}
        masks: Dict[str, np.ndarray] = {}
        for role, sentences in buckets.items():
            if not sentences:
                continue
            alignment = align_rationale(case.document, sentences)
            masks[role] = space.words_in_spans(alignment.spans)
        return masks

    def _faithfulness(
        self, space: DocumentSpace, scores: np.ndarray, eligible: np.ndarray, prediction
    ) -> Dict[str, object]:
        k = max(1, int(round(self.faithfulness_percent * eligible.sum())))
        top_mask = select_top_words(scores, eligible, k)
        bottom_mask = select_top_words(-scores, eligible, k)
        target = prediction.label
        top_prediction = self.adapter.predict_with_words_masked(space, top_mask)
        bottom_prediction = self.adapter.predict_with_words_masked(space, bottom_mask)
        # Measured on the same quantity the attribution explains.
        mode = getattr(self.adapter, "target_mode", "logit")
        result = faithfulness_scores(
            prediction.score(target, mode),
            top_prediction.score(target, mode),
            bottom_prediction.score(target, mode),
        )
        result["faithfulness_flip_top"] = bool(top_prediction.label != target)
        result["faithfulness_flip_bottom"] = bool(bottom_prediction.label != target)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in result.items()}

    # ── dataset level ───────────────────────────────────────────────────────
    def evaluate_dataset(self, cases: Sequence[PredExCase], progress: bool = True):
        import pandas as pd

        iterator = cases
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(cases, desc="Stage-3 audit")
            except ImportError:
                pass
        for case in iterator:
            try:
                self.evaluate_case(case)
            except Exception as error:  # keep the run alive; record the failure
                self.records.append(
                    {
                        "case_id": case.case_id,
                        "row_index": case.index,
                        "true_label": int(case.label),
                        "audit_status": "error",
                        "error": "{}: {}".format(type(error).__name__, error),
                    }
                )
        return pd.DataFrame(self.records)

    def attach_stage2(self, results_df, stage2_json: str):
        """Join per-case Stage-2 shortcut sensitivity onto the Stage-3 results.

        The existing Stage-2 output stores a hard label plus the winning class's
        confidence. For a binary task that is enough to recover P(class=1) exactly,
        so a continuous prediction shift can be derived without re-running Stage 2 --
        which matters because a hard label flip treats 0.49 -> 0.51 as a total change
        and 0.51 -> 0.99 as no change at all.
        """
        return attach_stage2_measures(results_df, stage2_json)


def _positive_probability(entry: Dict[str, object]) -> float:
    """P(class = 1) for a Stage-2 prediction entry.

    Newer runs store ``prob_positive`` directly. Older ones store only the hard label
    and the winning class's confidence, but for a binary task that determines the
    positive-class probability exactly, so earlier results stay usable.
    """
    if entry.get("prob_positive") is not None:
        return float(entry["prob_positive"])
    label = entry.get("label", -1)
    confidence = float(entry.get("confidence", 0.0) or 0.0)
    if label == 1:
        return confidence
    if label == 0:
        return 1.0 - confidence
    return float("nan")


# Variants excluded from the pooled shortcut score: the counterfactuals perturb the
# facts rather than adding spurious content, and v_full is the reference document.
# Older Stage-2 JSON used ``v_counterfactual`` (not the ``v_cf_*`` prefix), so that
# name must be excluded explicitly or it contaminates shortcut_delta_prob_max.
# The v2 run adds three more reference/perturbation variants: v_raw (the original
# document), v_facts_masked (facts removed) and v_sta_only_cf (a counterfactual).
_NON_SPURIOUS_VARIANTS = (
    "v_full",
    "v_counterfactual",
    "v_raw",
    "v_facts_masked",
    "v_sta_only_cf",
)


def attach_stage2_measures(results_df, stage2_json: str):
    """Derive per-case shortcut measures from the Stage-2 results and merge them in.

    Variant names are read from the file rather than hard-coded, so this works with
    both the current variant set and the earlier one.
    """
    import pandas as pd

    with open(stage2_json) as handle:
        stage2 = json.load(handle)

    variants = sorted(
        {
            key
            for case in stage2
            for key in case.get("predictions", {})
            if key != "v0_facts_only"
        }
    )

    rows = []
    for case in stage2:
        predictions = case.get("predictions", {})
        baseline = predictions.get("v0_facts_only", {})
        baseline_label = baseline.get("label", -1)
        baseline_prob = _positive_probability(baseline)
        row: Dict[str, object] = {
            "case_id": case.get("case_name"),
            "row_index": case.get("index"),
            "stage2_baseline_label": baseline_label,
            "stage2_baseline_prob": baseline_prob,
        }
        shifts = []
        for variant in variants:
            entry = predictions.get(variant, {})
            label = entry.get("label", -1)
            prob = _positive_probability(entry)
            flipped = (
                int(label != baseline_label)
                if (label != -1 and baseline_label != -1)
                else np.nan
            )
            delta = (
                abs(prob - baseline_prob)
                if np.isfinite(prob) and np.isfinite(baseline_prob)
                else np.nan
            )
            row["shift_" + variant] = flipped
            row["delta_prob_" + variant] = delta
            if (
                not variant.startswith("v_cf")
                and not variant.endswith("_cf")
                and variant not in _NON_SPURIOUS_VARIANTS
            ):
                shifts.append(delta)
        finite = [s for s in shifts if np.isfinite(s)]
        row["shortcut_delta_prob_max"] = max(finite) if finite else np.nan
        row["shortcut_delta_prob_mean"] = float(np.mean(finite)) if finite else np.nan
        rows.append(row)

    stage2_df = pd.DataFrame(rows)
    # Prefer row_index when both sides have it. Case Name is not unique in PredEx
    # (one title appears twice with slightly different Input text); a case_id-only
    # merge then Cartesian-expands 2×2 → 4 rows and inflates n_cases (1346→1348).
    stage3_cols = set(results_df.columns)
    stage2_measure_cols = [
        c for c in stage2_df.columns if c not in ("case_id", "row_index")
    ]
    if "row_index" in stage3_cols and stage2_df["row_index"].notna().all():
        merged = results_df.merge(
            stage2_df[["row_index"] + stage2_measure_cols + ["case_id"]].rename(
                columns={"case_id": "stage2_case_id"}
            ),
            on="row_index",
            how="left",
        )
        matched = int(merged["stage2_baseline_label"].notna().sum())
        join_method = "row_index"
        if "case_id" in merged and "stage2_case_id" in merged:
            both = merged["stage2_baseline_label"].notna()
            name_mismatch = int(
                (
                    both
                    & (merged["case_id"].astype(str) != merged["stage2_case_id"].astype(str))
                ).sum()
            )
            if name_mismatch:
                print(
                    "  WARNING: {} row_index joins disagree on case_id "
                    "(Stage-2 vs Stage-3 titles).".format(name_mismatch)
                )
        if matched == 0:
            print(
                "  WARNING: row_index join matched 0 rows; trying case_id "
                "(unsafe if titles are duplicated)."
            )
            merged = results_df.merge(
                stage2_df.drop(columns=["row_index"]), on="case_id", how="left"
            )
            matched = int(merged["stage2_baseline_label"].notna().sum())
            join_method = "case_id"
    else:
        merged = results_df.merge(
            stage2_df.drop(columns=["row_index"]), on="case_id", how="left"
        )
        matched = int(merged["stage2_baseline_label"].notna().sum())
        join_method = "case_id"
        if matched == 0 and "row_index" in results_df:
            n_stage2, n_stage3 = len(stage2_df), len(results_df)
            print(
                "  WARNING: no Stage-2 case names matched; falling back to row position. "
                "This is only correct if Stage 2 and Stage 3 were run on the same CSV, "
                "in the same row order. Stage-2 has {} rows, Stage-3 has {} rows{}.".format(
                    n_stage2,
                    n_stage3,
                    ""
                    if n_stage2 == n_stage3
                    else " -- LENGTH MISMATCH, do not trust this join",
                )
            )
            merged = results_df.merge(
                stage2_df.drop(columns=["case_id"]), on="row_index", how="left"
            )
            matched = int(merged["stage2_baseline_label"].notna().sum())
            join_method = "row_position (unverified -- see warning above)"

    if len(merged) != len(results_df):
        print(
            "  WARNING: Stage-2 join changed row count {} → {} "
            "(duplicate keys). Deduplicate before summarising.".format(
                len(results_df), len(merged)
            )
        )
    merged["stage2_join_method"] = join_method
    print(
        "  Stage-2 measures joined for {}/{} cases ({})".format(
            matched, len(merged), join_method
        )
    )
    return merged


def load_role_lookup(splits_json: str) -> Dict[str, Dict[str, List[str]]]:
    """Load OpenNyAI role buckets keyed by case name (and by index as a fallback).

    Missing the file is not treated as "no roles" silently: without it,
    ``role_share_*`` and ``post_decisional_attribution_share`` are never computed and the
    Stage-2/Stage-3 role linkage -- the part of the audit that connects "adding a
    role's text changes the prediction" (Stage 2) to "the model's attribution lands
    on that role's text" (Stage 3) -- silently disappears from the report
    with no error and no column to notice is absent. A loud warning here is cheap;
    a results table that looks complete but is quietly missing its central comparison
    is not.
    """
    if not os.path.exists(splits_json):
        print(
            "WARNING: role-lookup file '{}' not found. Role-based shortcut analysis "
            "(role_share_*, post_decisional_attribution_share) will be SKIPPED for every "
            "case -- this run will not test whether attribution concentrates on "
            "particular rhetorical roles. Generate it (Stage-2/OpenNyAI side) and pass "
            "its real path if you need that comparison.".format(splits_json)
        )
        return {}
    with open(splits_json) as handle:
        splits = json.load(handle)
    lookup: Dict[str, Dict[str, List[str]]] = {}
    for entry in splits:
        buckets = entry.get("role_buckets") or {}
        if "sentences" in entry and not buckets:
            buckets = {}
            for sentence in entry["sentences"]:
                buckets.setdefault(sentence["role"], []).append(sentence["text"])
        name = entry.get("case_name")
        if name:
            lookup[str(name)] = buckets
        lookup[str(entry.get("index"))] = buckets
    return lookup
