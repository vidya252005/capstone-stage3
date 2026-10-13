"""Tests for the Stage-3 v2 changes: margin attribution target, attributing every visible
case, and the CPU analysis in ``stage3_analysis.py``. No GPU, model or network needed.

Run with:  python -m pytest stage3/tests -q
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from stage3.config import KNOWN_ROLES

BIT = {role: 1 << i for i, role in enumerate(KNOWN_ROLES)}


def _case(roles, scores, eligible=None, human=None):
    roles = np.array([BIT[r] if r else 0 for r in roles], dtype=np.uint16)
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.size
    return {
        "scores": scores,
        "signed": scores.copy(),
        "eligible": np.ones(n, bool) if eligible is None else np.asarray(eligible, bool),
        "human": np.zeros(n, bool) if human is None else np.asarray(human, bool),
        "roles": roles,
    }


class MarginTargetTests(unittest.TestCase):
    def test_prediction_margin_is_log_odds(self):
        from stage3.adapters import Prediction

        logits = np.array([0.2, 1.5])
        probs = np.exp(logits) / np.exp(logits).sum()
        prediction = Prediction(1, probs[1], probs, logits)
        self.assertAlmostEqual(prediction.margin(1), 1.3, places=6)
        self.assertAlmostEqual(prediction.margin(0), -1.3, places=6)
        self.assertAlmostEqual(np.log(probs[1] / probs[0]), prediction.margin(1), places=6)
        self.assertEqual(prediction.score(1, "logit"), 1.5)

    def test_forward_returns_margin_per_target_column(self):
        import torch

        from stage3.adapters import HierarchicalLegalBERTAdapter

        adapter = HierarchicalLegalBERTAdapter.__new__(HierarchicalLegalBERTAdapter)
        adapter.model = SimpleNamespace(
            forward_from_embeddings=lambda e, m: torch.tensor([[0.5, 2.0]]))
        adapter.target_mode = "margin"
        out = adapter._forward_for_captum(None, None)
        self.assertEqual(out.tolist(), [[-1.5, 1.5]])
        adapter.target_mode = "logit"
        self.assertEqual(adapter._forward_for_captum(None, None).tolist(), [[0.5, 2.0]])


class FakeSpace:
    def __init__(self, n_words, visible):
        self.n_words = n_words
        self.n_tokens_total = n_words
        self.n_chunks = 1
        self.truncated = not all(visible)
        self.word_visible = np.asarray(visible, bool)

    def words_in_spans(self, spans):
        mask = np.zeros(self.n_words, bool)
        for start, end in spans:
            mask[start:end] = True
        return mask


class FakeAdapter:
    name = "fake"
    target_mode = "margin"

    def __init__(self, space):
        self.space = space
        self.attributed = 0

    def document_space(self, document):
        return self.space

    def _prediction(self, logits):
        from stage3.adapters import Prediction

        logits = np.asarray(logits, float)
        probs = np.exp(logits) / np.exp(logits).sum()
        return Prediction(int(probs.argmax()), float(probs.max()), probs, logits)

    def predict(self, document, space=None):
        return self._prediction([0.0, 2.0])

    def predict_with_words_masked(self, space, mask):
        return self._prediction([0.0, 1.0])

    def attribute(self, document, space=None, target=None, method=None):
        from stage3.adapters import Attribution

        self.attributed += 1
        scores = np.linspace(1.0, 2.0, self.space.n_words)
        return Attribution(scores, scores, scores, int(target), "integrated_gradients",
                           0.01, "margin", 0.01, 32)


class EvaluateCaseTests(unittest.TestCase):
    def _run(self, visible, rationale_spans):
        import stage3.audit as audit_module
        from stage3.audit import ShortcutAuditor

        space = FakeSpace(20, visible)
        adapter = FakeAdapter(space)
        auditor = ShortcutAuditor(adapter, role_lookup={"0": {"ANALYSIS": ["x"]}})
        original = (audit_module.align_rationale, audit_module.split_rationale_sentences)
        # Rationale spans come from rationale_spans; the ANALYSIS role covers words 10-20.
        audit_module.split_rationale_sentences = lambda text: ["r"]
        audit_module.align_rationale = lambda doc, sentences: SimpleNamespace(
            spans=rationale_spans if sentences == ["r"] else [(10, 20)],
            coverage=1.0, to_dict=lambda: {"rationale_coverage": 1.0})
        try:
            case = SimpleNamespace(case_id="c", index=0, label=1, document="d", rationale_text="t")
            audit = auditor.evaluate_case(case, keep_arrays=True)
        finally:
            audit_module.align_rationale, audit_module.split_rationale_sentences = original
        return adapter, audit

    def test_out_of_context_case_is_still_attributed(self):
        # Rationale in words 15-20, but the model only sees words 0-12.
        adapter, audit = self._run([True] * 12 + [False] * 8, [(15, 20)])
        record = audit.record
        self.assertEqual(record["audit_status"], "rationale_out_of_context")
        self.assertEqual(adapter.attributed, 1)
        self.assertNotIn("iou", record)
        self.assertIn("role_share_ANALYSIS", record)
        self.assertIn("post_decisional_attribution_share", record)
        self.assertIn("faithfulness_gap", record)
        self.assertEqual(record["attribution_target"], "margin")
        # Faithfulness is measured on the margin: 2.0 -> 1.0 for both maskings.
        self.assertAlmostEqual(record["faithfulness_drop_top"], 1.0, places=6)
        self.assertIsNotNone(audit.word_scores_signed)

    def test_visible_rationale_gets_iou(self):
        adapter, audit = self._run([True] * 20, [(0, 6)])
        self.assertEqual(audit.record["audit_status"], "ok")
        self.assertIn("iou", audit.record)


class AnalysisScriptTests(unittest.TestCase):
    def test_role_shares_and_excess(self):
        import stage3_analysis as sa

        store = {0: _case(["FAC"] * 4 + ["ANALYSIS"] * 4, [1, 1, 1, 1, 3, 3, 3, 3])}
        shares = sa.role_shares(store).set_index("role")
        self.assertAlmostEqual(shares.loc["ANALYSIS", "mass_share"], 0.75)
        self.assertAlmostEqual(shares.loc["ANALYSIS", "word_share"], 0.5)
        self.assertAlmostEqual(shares.loc["ANALYSIS", "excess"], 0.25)
        self.assertAlmostEqual(shares.loc["POST_DECISIONAL", "mass_share"], 0.75)

    def test_analysis_segments_first_last_and_truncation(self):
        import stage3_analysis as sa

        roles = ["FAC"] * 2 + ["ANALYSIS"] * 6
        scores = [1, 1, 1, 1, 1, 1, 4, 4]
        store = {0: _case(roles, scores),
                 1: _case(roles, scores, eligible=[True] * 6 + [False] * 2)}
        seg = sa.analysis_segments(store, budget=2).set_index("row_index")
        mean_visible = np.mean(scores)
        self.assertAlmostEqual(seg.loc[0, "density_last"], 4 / mean_visible, places=5)
        self.assertAlmostEqual(seg.loc[0, "density_first"], 1 / mean_visible, places=5)
        self.assertEqual(seg.loc[1, "status"], "last_part_truncated")

    def test_position_adjustment_removes_a_pure_position_effect(self):
        import stage3_analysis as sa

        # Attribution depends only on position (rises toward the end); Analysis sits at
        # the end. Raw density says Analysis is favoured; adjusted density should be ~1.
        rng = np.random.default_rng(0)
        store = {}
        n = 200
        position_weight = np.linspace(1, 3, n)
        for case in range(40):
            split = rng.integers(100, 160)
            roles = ["FAC"] * split + ["ANALYSIS"] * (n - split)
            # Reference text also appears late in some cases so every bin has a reference.
            if case % 2:
                roles[-30:] = ["ARG_PETITIONER"] * 30
            store[case] = _case(roles, position_weight)
        old = sa.MIN_CASES_PER_BIN
        try:
            sa.MIN_CASES_PER_BIN = 5
            curve = sa.position_curve(store)
        finally:
            sa.MIN_CASES_PER_BIN = old
        shares = sa.add_position_adjusted(sa.role_shares(store), store, curve)
        analysis = shares[shares.role == "ANALYSIS"]
        self.assertGreater(analysis["mass_share"].mean(), analysis["word_share"].mean())
        self.assertAlmostEqual(analysis["position_adjusted"].mean(), 1.0, delta=0.05)

    def test_role_matched_baseline_equals_iou_when_role_explains_overlap(self):
        import stage3_analysis as sa

        # Rationale = all Analysis words; model's top-k = the same Analysis words. Any
        # random choice inside the Analysis gives the same IoU, so model - null = 0.
        roles = ["FAC"] * 10 + ["ANALYSIS"] * 5
        human = [False] * 10 + [True] * 5
        store = {0: _case(roles, [0] * 10 + [5] * 5, human=human)}
        frame = sa.role_matched_iou(store, [0], n_draws=20)
        self.assertAlmostEqual(frame.loc[0, "iou"], 1.0)
        self.assertAlmostEqual(frame.loc[0, "iou_role_matched"], 1.0)
        self.assertLess(frame.loc[0, "iou_uniform_random"], 1.0)

    def test_stage2_measures_from_v2_and_controls(self):
        import json
        import tempfile
        from pathlib import Path

        import stage3_analysis as sa

        v2 = [{"index": 3, "case_name": "A", "true_label": 1, "predictions": {
            "v0_facts_only": {"label": 0, "confidence": 0.8},        # P(true)=0.2
            "v_analysis_only": {"label": 1, "confidence": 0.7}}}]    # P(true)=0.7
        controls = {"index": 3, "true_label": 1, "predictions": {
            "loo_analysis": {"label": 0, "confidence": 0.6, "prob_positive": 0.4},
            "loo_rand_analysis": {"label": 1, "confidence": 0.9, "prob_positive": 0.9},
            "tx_analysis_d1": {"label": 1, "confidence": 0.8, "prob_positive": 0.8},
            "tx_analysis_d0": {"label": 0, "confidence": 0.7, "prob_positive": 0.3}}}
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "v2.json").write_text(json.dumps(v2))
            Path(tmp, "c.jsonl").write_text(json.dumps(controls) + "\n")
            frame = sa.stage2_measures(str(Path(tmp, "v2.json")), str(Path(tmp, "c.jsonl")))
        row = frame.set_index("row_index").loc[3]
        self.assertAlmostEqual(row["s2_add_analysis_toward_truth"], 0.5)
        self.assertAlmostEqual(row["s2_loo_analysis_excess"], 0.5)
        self.assertAlmostEqual(row["s2_transplant_follow"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
