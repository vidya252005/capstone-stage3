"""Regression tests for the parts of Stage 3 that don't need a GPU, InLegalBERT,
or network access: rationale alignment, the chunk/token/word coordinate system,
and the IoU/attribution metrics.

Deliberately zero-dependency beyond numpy (already required by stage3 itself) --
these must be runnable on a laptop, in CI, or anywhere else with no GPU and no
Hugging Face access, since that is exactly where they are most useful: catching a
regression in the alignment/metric math *before* burning GPU hours re-running the
full audit to notice the numbers moved.

Run with:  python -m unittest stage3.tests.test_core_logic -v
"""
from __future__ import annotations

import unittest

import numpy as np

from stage3.alignment import align_rationale, build_document_space, aggregate_chunk_scores_to_tokens, select_top_words
from stage3 import alignment as _alignment_module
from stage3.metrics import binary_overlap, expected_random_iou, faithfulness_scores
from stage3.predex import parse_output_field, split_rationale_sentences


class FakeFastTokenizer:
    """Minimal stand-in for a HF fast tokenizer: one token per (up to) 3 characters,
    never spanning a whitespace gap. Enough to exercise build_document_space's
    chunk/stride/truncation logic without downloading anything.
    """

    is_fast = True
    cls_token_id = -1
    sep_token_id = -2
    pad_token_id = -3
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True,
                 return_attention_mask=True, truncation=False, verbose=False):
        ids, offsets, mask = [], [], []
        i, tid = 0, 100
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < min(i + 3, len(text)) and not text[j].isspace():
                j += 1
            offsets.append((i, j))
            ids.append(tid)
            tid += 1
            mask.append(1)
            i = j
        return {"input_ids": ids, "offset_mapping": offsets, "attention_mask": mask}


class TestBinaryOverlap(unittest.TestCase):
    def test_matches_hand_computed_iou(self):
        model = np.zeros(10, bool)
        model[[2, 3, 4, 5]] = True
        human = np.zeros(10, bool)
        human[[0, 1, 2, 3]] = True
        result = binary_overlap(model, human)
        self.assertAlmostEqual(result["iou"], 2 / 6)
        self.assertAlmostEqual(result["precision"], 2 / 4)
        self.assertAlmostEqual(result["recall"], 2 / 4)

    def test_empty_masks_are_nan_not_zero(self):
        z = np.zeros(10, bool)
        result = binary_overlap(z, z)
        self.assertTrue(np.isnan(result["iou"]))

    def test_perfect_overlap_is_one(self):
        o = np.ones(10, bool)
        result = binary_overlap(o, o)
        self.assertEqual(result["iou"], 1.0)


class TestExpectedRandomIoU(unittest.TestCase):
    def test_matches_monte_carlo_within_tolerance(self):
        analytic = expected_random_iou(10, 15, 100)
        rng = np.random.default_rng(0)
        idx = np.arange(100)
        ious = []
        for _ in range(20000):
            m = np.zeros(100, bool)
            m[rng.choice(idx, 10, replace=False)] = True
            h = np.zeros(100, bool)
            h[rng.choice(idx, 15, replace=False)] = True
            inter, union = (m & h).sum(), (m | h).sum()
            ious.append(inter / union if union else np.nan)
        self.assertAlmostEqual(analytic, float(np.nanmean(ious)), delta=0.01)


class TestFaithfulnessScores(unittest.TestCase):
    def test_positive_gap_when_top_words_matter_more(self):
        # Removing important words should drop the logit more than removing
        # unimportant ones -- that's what "faithful" means here.
        result = faithfulness_scores(baseline_logit=2.0, top_masked_logit=-1.0, bottom_masked_logit=1.8)
        self.assertGreater(result["faithfulness_gap"], 0)
        self.assertTrue(result["faithfulness_pass"])

    def test_negative_gap_when_attribution_is_backwards(self):
        result = faithfulness_scores(baseline_logit=2.0, top_masked_logit=1.9, bottom_masked_logit=-1.0)
        self.assertLess(result["faithfulness_gap"], 0)
        self.assertFalse(result["faithfulness_pass"])


class TestRationaleAlignment(unittest.TestCase):
    def test_recovers_predex_concatenation_artifact(self):
        # The exact failure mode alignment.py's own docstring calls out: the source
        # document has a period + space where the stored rationale has none.
        document = (
            "The court decided to dismiss the review petition. 8. We do not find any "
            "merit in the arguments raised by the appellant regarding the sentence."
        )
        rationale = "dismiss the review petition8. We do not find any merit in the arguments"
        result = align_rationale(document, [rationale])
        self.assertGreater(result.coverage, 0.9, "concatenation artifact should be recovered")
        self.assertEqual(result.status, "exact")

    def test_absent_rationale_is_unresolved(self):
        document = "Nothing here relates to the rationale at all, said the court."
        result = align_rationale(document, ["this text does not appear anywhere in the judgment"])
        self.assertEqual(result.coverage, 0.0)
        self.assertEqual(result.status, "unresolved")

    def test_exact_match_is_full_coverage(self):
        document = "We do not find any merit in the arguments raised by the appellant."
        result = align_rationale(document, [document])
        self.assertEqual(result.status, "exact")
        self.assertAlmostEqual(result.coverage, 1.0)


class TestParsePredExOutputField(unittest.TestCase):
    def test_splits_label_and_rationale(self):
        parsed = parse_output_field("0[ds]7. It is clear from the order that ...")
        self.assertEqual(parsed["label"], 0)
        self.assertTrue(parsed["delimiter_found"])
        self.assertTrue(parsed["rationale"].startswith("7. It is clear"))

    def test_missing_delimiter_is_unusable_not_silently_whole_cell(self):
        parsed = parse_output_field("no delimiter in this string")
        self.assertFalse(parsed["delimiter_found"])
        self.assertEqual(parsed["rationale"], "")

    def test_nan_input_does_not_raise(self):
        parsed = parse_output_field(float("nan"))
        self.assertFalse(parsed["delimiter_found"])


class TestDocumentSpaceChunking(unittest.TestCase):
    """Uses small monkeypatched CHUNK_SIZE/STRIDE/MAX_CHUNKS so the sliding-window
    math can be checked by hand instead of needing thousands of tokens of filler
    text to exercise the real 512/256/16 configuration."""

    def setUp(self):
        self._orig = (_alignment_module.CHUNK_SIZE, _alignment_module.STRIDE, _alignment_module.MAX_CHUNKS)
        _alignment_module.CHUNK_SIZE = 10
        _alignment_module.STRIDE = 4
        _alignment_module.MAX_CHUNKS = 3
        self.tok = FakeFastTokenizer()
        self.document = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi"
        )
        self.space = build_document_space(self.document, self.tok)

    def tearDown(self):
        (_alignment_module.CHUNK_SIZE, _alignment_module.STRIDE, _alignment_module.MAX_CHUNKS) = self._orig

    def test_truncates_at_max_chunks(self):
        self.assertEqual(self.space.n_chunks, 3)
        self.assertTrue(self.space.truncated)

    def test_every_token_maps_to_a_word_containing_its_span(self):
        mismatches = 0
        for token_idx, (start, end) in enumerate(self.space.token_spans):
            word_idx = self.space.token_to_word[token_idx]
            if word_idx >= 0:
                word_start, word_end = self.space.word_spans[word_idx]
                if not (word_start <= start < word_end):
                    mismatches += 1
        self.assertEqual(mismatches, 0)

    def test_words_beyond_visible_tokens_are_not_visible(self):
        # With these settings only the first 8 of 16 words fall inside the visible
        # token window; the rest were never seen by the model and must not be
        # scored as if they were.
        self.assertTrue(self.space.word_visible[:8].all())
        self.assertFalse(self.space.word_visible[8:].any())

    def test_overlapping_chunks_are_mean_aggregated(self):
        chunk_scores = np.array(
            [[float(i) for i in range(_alignment_module.CHUNK_SIZE)] for _ in range(self.space.n_chunks)]
        )
        token_scores = aggregate_chunk_scores_to_tokens(chunk_scores, self.space, mode="mean")
        # Global token 4 is covered by chunk 0 (at body position 4 -> chunk slot 5,
        # score 5) and chunk 1 (at body position 0 -> chunk slot 1, score 1); mean=3.
        self.assertAlmostEqual(token_scores[4], 3.0)

    def test_select_top_words_returns_exactly_k(self):
        scores = np.arange(self.space.n_words, dtype=float)
        selected = select_top_words(scores, self.space.word_visible, k=3)
        self.assertEqual(int(selected.sum()), 3)
        # Highest-scoring eligible words should win.
        self.assertTrue(selected[5] and selected[6] and selected[7])


class Stage2JoinTests(unittest.TestCase):
    def test_row_index_join_does_not_cartesian_expand_duplicate_titles(self):
        import json
        import tempfile
        from pathlib import Path

        import pandas as pd

        from stage3.audit import attach_stage2_measures

        stage3 = pd.DataFrame(
            {
                "case_id": ["SAME TITLE", "SAME TITLE", "OTHER"],
                "row_index": [10, 11, 12],
                "iou": [0.1, 0.2, 0.3],
            }
        )
        stage2 = [
            {
                "case_name": "SAME TITLE",
                "index": 10,
                "predictions": {
                    "v0_facts_only": {"label": 0, "confidence": 0.7},
                    "v25": {"label": 0, "confidence": 0.5},
                },
            },
            {
                "case_name": "SAME TITLE",
                "index": 11,
                "predictions": {
                    "v0_facts_only": {"label": 0, "confidence": 0.6},
                    "v25": {"label": 1, "confidence": 0.8},
                },
            },
            {
                "case_name": "OTHER",
                "index": 12,
                "predictions": {
                    "v0_facts_only": {"label": 1, "confidence": 0.9},
                    "v25": {"label": 1, "confidence": 0.85},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.json"
            path.write_text(json.dumps(stage2))
            merged = attach_stage2_measures(stage3, str(path))
        self.assertEqual(len(merged), 3, "join must stay 1:1 on row_index")
        self.assertEqual(merged["stage2_join_method"].iloc[0], "row_index")
        # Distinct Stage-2 baselines for the two same-title rows
        self.assertNotEqual(
            merged.loc[merged.row_index == 10, "stage2_baseline_prob"].iloc[0],
            merged.loc[merged.row_index == 11, "stage2_baseline_prob"].iloc[0],
        )


class V2Stage2Tests(unittest.TestCase):
    def test_reference_and_counterfactual_variants_not_pooled(self):
        import json
        import tempfile
        from pathlib import Path

        import pandas as pd

        from stage3.audit import attach_stage2_measures

        predictions = {"v0_facts_only": {"label": 0, "confidence": 0.9},
                       "v_analysis_only": {"label": 0, "confidence": 0.8}}
        # Big moves in reference/counterfactual variants must not count as shortcuts.
        for name in ("v_raw", "v_full", "v_facts_masked", "v_sta_only_cf", "v_counterfactual"):
            predictions[name] = {"label": 1, "confidence": 0.99}
        stage3 = pd.DataFrame([{"case_id": "A", "row_index": 0}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2.json"
            path.write_text(json.dumps([{"case_name": "A", "index": 0, "predictions": predictions}]))
            merged = attach_stage2_measures(stage3, str(path))
        self.assertAlmostEqual(merged["shortcut_delta_prob_max"].iloc[0], 0.1, places=6)
        self.assertIn("delta_prob_v_raw", merged.columns)


class RoleLookupTests(unittest.TestCase):
    def test_duplicate_titles_get_their_own_roles(self):
        from types import SimpleNamespace

        from stage3.audit import ShortcutAuditor

        lookup = {"SAME": {"FAC": ["second case facts"]},
                  "0": {"FAC": ["first case facts"]},
                  "1": {"FAC": ["second case facts"]}}
        auditor = ShortcutAuditor(adapter=None, role_lookup=lookup)
        seen = []

        class Space:
            def words_in_spans(self, spans):
                seen.append(spans)
                return np.zeros(1, dtype=bool)

        import stage3.audit as audit_module

        original = audit_module.align_rationale
        audit_module.align_rationale = lambda document, sentences: SimpleNamespace(spans=sentences)
        try:
            auditor._role_masks(SimpleNamespace(case_id="SAME", index=0, document=""), Space())
        finally:
            audit_module.align_rationale = original
        self.assertEqual(seen, [["first case facts"]])


class WordStoreTests(unittest.TestCase):
    def test_round_trip_and_role_bits(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from stage3.metrics import role_attribution_shares
        from stage3.word_store import load_store, pack_case, role_mask, save_store

        fac = np.array([True, True, False, False])
        ana = np.array([False, True, True, False])
        audit = SimpleNamespace(
            word_scores=np.array([0.5, -1.0, 2.0, 0.0]),
            eligible=np.array([True, True, True, False]),
            human_mask=np.array([False, True, False, False]),
            role_masks={"FAC": fac, "ANALYSIS": ana},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "w.npz")
            save_store(path, {7: pack_case(audit)})
            back = load_store(path)[7]
        np.testing.assert_array_equal(role_mask(back["roles"], "FAC"), fac)
        np.testing.assert_array_equal(role_mask(back["roles"], "ANALYSIS"), ana)
        np.testing.assert_array_equal(back["human"], audit.human_mask)
        shares = role_attribution_shares(back["scores"], {"FAC": fac}, back["eligible"], ["FAC"])
        self.assertAlmostEqual(shares["role_share_FAC"], 1.5 / 3.5, places=6)
        self.assertAlmostEqual(shares["role_words_FAC"], 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
