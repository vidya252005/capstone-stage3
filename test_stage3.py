"""Correctness tests for the Stage-3 audit.

These check the parts that would silently produce plausible-looking but wrong numbers:
chunk reconstruction, embedding-path equivalence, span alignment and the metrics.

Run with a small randomly-initialised BERT (no checkpoint or GPU needed)::

    python test_stage3.py --model ./InLegalBERT
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from transformers import AutoConfig

from stage3.adapters import load_tokenizer
from stage3.alignment import (
    aggregate_chunk_scores_to_tokens,
    align_rationale,
    build_document_space,
    merge_spans,
    select_top_words,
)
from stage3.config import CHUNK_SIZE, MAX_CHUNKS, STRIDE
from stage3.metrics import binary_overlap, expected_random_iou
from stage3.predex import parse_output_field, split_rationale_sentences

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print("  {} {}{}".format("PASS" if condition else "FAIL", name,
                             "  <- " + detail if detail and not condition else ""))


# ── PredEx parsing ───────────────────────────────────────────────────────────
def test_predex_parsing() -> None:
    print("\n[predex] Output field parsing")
    parsed = parse_output_field("0[ds]7. It is clear from the order of appointment.")
    check("label parsed from [ds] prefix", parsed["label"] == 0)
    check("rationale stripped of prefix",
          parsed["rationale"] == "7. It is clear from the order of appointment.")
    check("delimiter flag set", parsed["delimiter_found"] is True)

    missing = parse_output_field("no delimiter here")
    check("row without [ds] rejected", missing["delimiter_found"] is False)
    check("row without [ds] yields no rationale", missing["rationale"] == "")
    check("NaN handled", parse_output_field(float("nan"))["delimiter_found"] is False)

    sentences = split_rationale_sentences(
        "This first sentence is definitely long enough to keep. Short. "
        "And here is a second sentence that also clears the length bar."
    )
    check("short fragments dropped", len(sentences) == 2, str(sentences))


# ── Alignment ────────────────────────────────────────────────────────────────
def test_alignment() -> None:
    print("\n[alignment] rationale span matching")
    document = (
        "The appellant argued that the evidence was insufficient. "
        "The court examined the testimony of the witness carefully. "
        "Accordingly the appeal was dismissed by this Court."
    )

    exact = align_rationale(
        document, ["The court examined the testimony of the witness carefully."]
    )
    check("exact sentence found", exact.coverage > 0.99, str(exact.coverage))
    check("one span returned", len(exact.spans) == 1, str(exact.spans))
    start, end = exact.spans[0]
    check("span points at the right text",
          document[start:end].startswith("The court examined"), document[start:end])

    # Whitespace, line breaks and punctuation drift must not break the match.
    noisy = align_rationale(
        document, ["The  court examined\nthe testimony of the  witness carefully"]
    )
    check("robust to whitespace/newline drift", noisy.coverage > 0.99, str(noisy.coverage))

    # The PredEx concatenation artefact: sentences joined with the separator dropped.
    joined = align_rationale(
        document,
        ["The court examined the testimony of the witness carefully."
         "Accordingly the appeal was dismissed by this Court."],
    )
    check("concatenated sentences recovered", joined.coverage > 0.95, str(joined.coverage))

    absent = align_rationale(
        document, ["The tribunal remitted the matter for fresh consideration entirely."]
    )
    check("absent rationale reports zero coverage", absent.coverage < 0.30,
          str(absent.coverage))
    check("absent rationale marked unresolved", absent.status == "unresolved", absent.status)

    check("empty rationale is safe", align_rationale(document, []).status == "empty_rationale")
    check("span merging works",
          merge_spans([(0, 5), (4, 9), (20, 25)]) == [(0, 9), (20, 25)])


def test_alignment_on_real_predex(csv_path: str) -> None:
    print("\n[alignment] real PredEx cases")
    try:
        from stage3.predex import load_predex

        cases = load_predex(csv_path, limit=25)
    except Exception as error:
        print("  SKIP (no PredEx CSV at {}): {}".format(csv_path, error))
        return
    coverages = [
        align_rationale(c.document, split_rationale_sentences(c.rationale_text)).coverage
        for c in cases
    ]
    mean_coverage = float(np.mean(coverages))
    check("mean rationale coverage above 0.90 on real data", mean_coverage > 0.90,
          "got {:.3f}".format(mean_coverage))
    check("labels recovered for every case", all(c.label in (0, 1) for c in cases))


# ── Document space / chunking ────────────────────────────────────────────────
def test_document_space(tokenizer) -> None:
    print("\n[space] tokenisation, words and chunk reconstruction")
    document = " ".join("word{}".format(i) for i in range(4000))
    space = build_document_space(document, tokenizer)

    check("word count matches whitespace split", space.n_words == len(document.split()))
    check("every chunk is CHUNK_SIZE long",
          all(len(ids) == CHUNK_SIZE for ids, _ in space.chunks))
    check("chunk count capped at MAX_CHUNKS", space.n_chunks <= MAX_CHUNKS)
    check("long document flagged as truncated", space.truncated is True)

    # Byte-identical to opennyai_ablation.chunk_text.
    encoding = tokenizer(document, add_special_tokens=False)
    ids_all, mask_all = encoding["input_ids"], encoding["attention_mask"]
    effective, reference, start = CHUNK_SIZE - 2, [], 0
    while start < len(ids_all):
        end = start + effective
        chunk = [tokenizer.cls_token_id] + ids_all[start:end] + [tokenizer.sep_token_id]
        mask = [1] + mask_all[start:end] + [1]
        pad = CHUNK_SIZE - len(chunk)
        reference.append((chunk + [tokenizer.pad_token_id] * pad, mask + [0] * pad))
        if end >= len(ids_all):
            break
        start += STRIDE
    reference = reference[:MAX_CHUNKS]
    check("chunks identical to the Stage-1/2 chunker", space.chunks == reference)

    # Provenance must point back at exactly the token that sits in that slot.
    origin = space.chunk_token_to_global
    mismatch = 0
    for chunk_index in range(space.n_chunks):
        for position in range(CHUNK_SIZE):
            global_index = origin[chunk_index][position]
            if global_index >= 0 and (
                space.chunks[chunk_index][0][position] != space.token_ids[global_index]
            ):
                mismatch += 1
    check("chunk->global token map is exact", mismatch == 0, "{} mismatches".format(mismatch))
    check("special/pad positions map to -1",
          all(origin[c][0] == -1 and origin[c][511] == -1 for c in range(space.n_chunks)))
    check("visible tokens equal the 16-chunk budget",
          space.n_visible_tokens == 15 * STRIDE + (CHUNK_SIZE - 2),
          str(space.n_visible_tokens))
    check("some words fall outside the model context", (~space.word_visible).any())

    short = build_document_space("The appeal is dismissed.", tokenizer)
    check("short document yields one chunk", short.n_chunks == 1)
    check("short document not truncated", short.truncated is False)
    check("short document fully visible", short.word_visible.all())
    check("empty document is safe", build_document_space("", tokenizer).n_chunks == 1)

    # Overlapping windows must average, not double count.
    scores = np.zeros((space.n_chunks, CHUNK_SIZE))
    scores[0] = 1.0
    scores[1] = 3.0
    token_scores = aggregate_chunk_scores_to_tokens(scores, space, mode="mean")
    overlap_start = STRIDE
    check("token in one window keeps its score",
          abs(token_scores[0] - 1.0) < 1e-9, str(token_scores[0]))
    check("token in two windows is averaged",
          abs(token_scores[overlap_start] - 2.0) < 1e-9, str(token_scores[overlap_start]))
    maxed = aggregate_chunk_scores_to_tokens(scores, space, mode="max")
    check("max aggregation available",
          abs(maxed[overlap_start] - 3.0) < 1e-9, str(maxed[overlap_start]))


def test_word_alignment_roundtrip(tokenizer) -> None:
    print("\n[space] rationale -> word mask round trip")
    document = (
        "The appellant relied on Section 379 of the Indian Penal Code. "
        "The court examined the testimony of the sole eyewitness. "
        "The prosecution failed to establish the chain of circumstances."
    )
    space = build_document_space(document, tokenizer)
    alignment = align_rationale(
        document, ["The court examined the testimony of the sole eyewitness."]
    )
    mask = space.words_in_spans(alignment.spans)
    selected = [space.words[i] for i in np.flatnonzero(mask)]
    check("masked words are exactly the rationale sentence",
          selected == "The court examined the testimony of the sole eyewitness.".split(),
          str(selected))
    check("mask does not leak into neighbouring sentences",
          not mask[0] and not mask[-1])

    scores = np.zeros(len(space.token_ids))
    target_word = space.words.index("eyewitness.")
    scores[space.token_to_word == target_word] = 2.0
    word_scores = space.aggregate_tokens_to_words(scores, mode="sum")
    check("sub-word scores fold onto the right word",
          word_scores[target_word] > 0 and word_scores.argmax() == target_word)

    top = select_top_words(word_scores, np.ones(space.n_words, bool), 1)
    check("top-k selection picks that word", bool(top[target_word]) and top.sum() == 1)


# ── Metrics ──────────────────────────────────────────────────────────────────
def test_metrics() -> None:
    print("\n[metrics] overlap arithmetic")
    model = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], bool)
    human = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0], bool)
    result = binary_overlap(model, human)
    check("IoU = 4/5", abs(result["iou"] - 0.8) < 1e-9, str(result["iou"]))
    check("precision = 4/4", abs(result["precision"] - 1.0) < 1e-9)
    check("recall = 4/5", abs(result["recall"] - 0.8) < 1e-9)
    check("F1 = 8/9", abs(result["f1"] - (2 * 1.0 * 0.8 / 1.8)) < 1e-9)

    disjoint = binary_overlap(np.array([1, 0, 0], bool), np.array([0, 0, 1], bool))
    check("disjoint masks give IoU 0", disjoint["iou"] == 0.0)
    check("disjoint masks give F1 0", disjoint["f1"] == 0.0)

    empty = binary_overlap(np.zeros(3, bool), np.zeros(3, bool))
    check("empty masks give NaN not 0", np.isnan(empty["iou"]))

    # A rationale covering a quarter of the document already scores ~0.14 by chance.
    random_iou = expected_random_iou(250, 250, 1000)
    check("random IoU baseline computed", abs(random_iou - (62.5 / 437.5)) < 1e-9,
          str(random_iou))


# ── Model path ───────────────────────────────────────────────────────────────
def test_model_paths(model_dir: str, tokenizer) -> None:
    print("\n[model] forward vs embedding path, and Integrated Gradients")
    from hierarchical_model import HierarchicalLegalBERT

    # A small random BERT keeps the test fast; the wiring is what is being checked.
    config = AutoConfig.from_pretrained(model_dir)
    config.num_hidden_layers = 2
    config.hidden_size = 96
    config.num_attention_heads = 2
    config.intermediate_size = 128

    torch.manual_seed(0)
    from transformers import AutoModel

    class TinyHierarchical(HierarchicalLegalBERT):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.bert = AutoModel.from_config(config)
            self.gru = torch.nn.GRU(
                input_size=config.hidden_size, hidden_size=32, num_layers=2,
                batch_first=True, dropout=0.3, bidirectional=True,
            )
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(64, 2)

    model = TinyHierarchical().eval()
    document = " ".join("token{}".format(i) for i in range(700))
    space = build_document_space(document, tokenizer)
    ids = torch.tensor([c[0] for c in space.chunks]).unsqueeze(0)
    mask = torch.tensor([c[1] for c in space.chunks]).unsqueeze(0)
    counts = torch.tensor([space.n_chunks])

    with torch.no_grad():
        via_ids = model(ids, mask, counts)
        embeds = model.word_embedding_layer(ids)
        via_embeds = model.forward_from_embeddings(embeds, mask)
    gap = (via_ids - via_embeds).abs().max().item()
    check("embedding path matches the trained forward path", gap < 1e-4,
          "max abs diff {:.2e}".format(gap))

    from stage3.adapters import HierarchicalLegalBERTAdapter

    adapter = HierarchicalLegalBERTAdapter(
        model, tokenizer, torch.device("cpu"), ig_steps=8, ig_internal_batch=1
    )
    prediction = adapter.predict(document, space)
    check("prediction returns a valid class", prediction.label in (0, 1))
    check("probabilities sum to 1", abs(prediction.probabilities.sum() - 1.0) < 1e-5)

    attribution = adapter.attribute(document, space=space, method="integrated_gradients")
    check("attribution has one score per word",
          attribution.word_scores.shape == (space.n_words,))
    check("attribution is finite", bool(np.isfinite(attribution.word_scores).all()))
    check("attribution is non-trivial", float(attribution.word_scores.sum()) > 0)
    check("IG convergence delta recorded", attribution.convergence_delta is not None)
    invisible = attribution.word_scores[~space.word_visible]
    check("words outside model context get no attribution",
          invisible.size == 0 or float(np.abs(invisible).max()) == 0.0)

    grad_input = adapter.attribute(document, space=space, method="gradient_x_input")
    check("gradient x input runs", bool(np.isfinite(grad_input.word_scores).all()))
    overlap = np.corrcoef(attribution.word_scores, grad_input.word_scores)[0, 1]
    check("IG and grad-x-input are correlated", np.isfinite(overlap),
          "corr {:.3f}".format(overlap))

    masked = adapter.predict_with_words_masked(space, np.arange(space.n_words) < 20)
    check("masked re-prediction runs", masked.label in (0, 1))


def test_end_to_end(csv_path: str, model_dir: str, tokenizer) -> None:
    print("\n[audit] end-to-end on real PredEx cases")
    try:
        from stage3.predex import load_predex

        cases = load_predex(csv_path, limit=3)
    except Exception as error:
        print("  SKIP (no PredEx CSV): {}".format(error))
        return

    from transformers import AutoModel

    from hierarchical_model import HierarchicalLegalBERT
    from stage3.adapters import HierarchicalLegalBERTAdapter
    from stage3.audit import ShortcutAuditor
    from stage3.report import format_summary, summarise

    config = AutoConfig.from_pretrained(model_dir)
    config.num_hidden_layers = 2
    config.hidden_size = 96
    config.num_attention_heads = 2
    config.intermediate_size = 128
    torch.manual_seed(0)

    class TinyHierarchical(HierarchicalLegalBERT):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.bert = AutoModel.from_config(config)
            self.gru = torch.nn.GRU(
                input_size=config.hidden_size, hidden_size=32, num_layers=2,
                batch_first=True, dropout=0.3, bidirectional=True,
            )
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(64, 2)

    adapter = HierarchicalLegalBERTAdapter(
        TinyHierarchical().eval(), tokenizer, torch.device("cpu"),
        ig_steps=4, ig_internal_batch=1,
    )
    auditor = ShortcutAuditor(adapter, run_faithfulness=True)
    audit = auditor.evaluate_case(cases[0], keep_arrays=True)
    record = audit.record
    check("case audited without error", record.get("audit_status") != "error",
          str(record.get("error")))

    for column in ["iou", "precision", "recall", "f1", "iou_p05", "iou_p10", "iou_p20",
                   "human_supported_attribution_share", "faithfulness_gap",
                   "rationale_coverage", "human_words_visible",
                   "human_words_out_of_context", "iou_random"]:
        check("record has {}".format(column), column in record)

    if record.get("audit_status") == "ok":
        check("IoU within [0, 1]", 0.0 <= record["iou"] <= 1.0, str(record["iou"]))
        check("model selection matches human size",
              record["model_words_selected"] == record["human_words_visible"],
              "{} vs {}".format(record["model_words_selected"], record["human_words_visible"]))
        check("visible + out-of-context = total rationale",
              record["human_words_visible"] + record["human_words_out_of_context"]
              == record["human_words_total"])

    results = auditor.evaluate_dataset(cases[1:], progress=False)
    check("dataset frame produced", len(results) == len(auditor.records))
    summary = summarise(results)
    check("summary renders", isinstance(format_summary(summary), str))

    import os
    import tempfile

    from stage3.report import write_case_html

    if audit.space is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_case_html(audit, os.path.join(tmp, "case.html"))
            check("HTML heatmap written", os.path.getsize(path) > 500)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./InLegalBERT")
    parser.add_argument("--csv", default="./predex_test_deduped.csv")
    args = parser.parse_args()

    try:
        tokenizer = load_tokenizer(args.model)
    except (FileNotFoundError, ValueError) as error:
        print("FATAL: {}".format(error))
        return 2

    test_predex_parsing()
    test_alignment()
    test_alignment_on_real_predex(args.csv)
    test_document_space(tokenizer)
    test_word_alignment_roundtrip(tokenizer)
    test_metrics()
    test_model_paths(args.model, tokenizer)
    test_end_to_end(args.csv, args.model, tokenizer)

    print("\n" + "=" * 60)
    print("{} passed, {} failed".format(len(PASSED), len(FAILED)))
    if FAILED:
        for name in FAILED:
            print("  FAILED: {}".format(name))
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
