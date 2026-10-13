"""Projecting expert rationales and model attributions into one coordinate system.

The audit compares two very differently-shaped signals:

* the expert rationale, which is free text copied out of the judgment, and
* the model attribution, which lives on InLegalBERT sub-word tokens spread across
  overlapping 512-token windows.

Neither can be compared directly, so everything is projected onto a single
**word-level document space**: whitespace-delimited words of the original document,
each with a character span. Words are the comparison unit because the expert
annotation is sentence/word level while the model works in sub-words, and reporting
IoU over ``constitution`` + ``##al`` fragments would be neither faithful nor
readable.

Two pieces of book-keeping matter for validity:

* **Coverage** — how much of the rationale was actually located in the document.
  PredEx rationales are extracted from the full judgment, so a rationale sentence
  can legitimately be absent from the stored ``Input``.
* **Visibility** — which words the model actually read. With ``MAX_CHUNKS=16`` the
  encoder sees at most ``15*STRIDE + (CHUNK_SIZE-2)`` tokens, so a long judgment is
  truncated. Rationale beyond that point must not be scored as a model failure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    CHUNK_SIZE,
    EXACT_COVERAGE,
    MAX_CHUNKS,
    MIN_SEG_CHARS,
    MIN_SENTENCE_CHARS,
    PARTIAL_COVERAGE,
    STRIDE,
)

Span = Tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# Normalised text index
# ─────────────────────────────────────────────────────────────────────────────
class NormalizedIndex:
    """A lowercase alphanumeric projection of a text, plus a map back to offsets.

    Dropping whitespace and punctuation entirely makes matching immune to the
    line-break, spacing and punctuation drift that separates a PredEx rationale from
    its source judgment, while ``origin`` still recovers exact original offsets.
    """

    __slots__ = ("text", "origin")

    def __init__(self, text: str) -> None:
        chars: List[str] = []
        origin: List[int] = []
        for position, character in enumerate(text):
            if character.isalnum():
                chars.append(character.lower())
                origin.append(position)
        self.text = "".join(chars)
        self.origin = origin

    def __len__(self) -> int:
        return len(self.text)

    def to_original(self, start: int, end: int) -> Span:
        """Map a half-open span in normalised coordinates back to original ones."""
        return self.origin[start], self.origin[end - 1] + 1


def _longest_prefix_match(haystack: str, needle: str, start_at: int = 0) -> Tuple[int, int]:
    """Longest prefix of ``needle`` occurring in ``haystack`` at or after ``start_at``.

    Prefix occurrence is monotone -- if a prefix of length L occurs then so does every
    shorter one -- so the longest occurring prefix can be found by binary search.
    """
    if not needle:
        return 0, -1
    position = haystack.find(needle, start_at)
    if position != -1:
        return len(needle), position

    low, high, best_length, best_position = 1, len(needle) - 1, 0, -1
    while low <= high:
        mid = (low + high) // 2
        position = haystack.find(needle[:mid], start_at)
        if position != -1:
            best_length, best_position, low = mid, position, mid + 1
        else:
            high = mid - 1
    return best_length, best_position


def merge_spans(spans: Sequence[Span], gap: int = 2) -> List[Span]:
    """Sort and merge overlapping or near-touching spans."""
    ordered = sorted(s for s in spans if s[1] > s[0])
    merged: List[Span] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Rationale alignment
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RationaleAlignment:
    """Where the expert rationale sits inside the document, and how well it matched."""

    spans: List[Span]
    coverage: float
    n_sentences: int
    n_exact: int
    n_partial: int
    n_not_found: int
    n_ambiguous: int

    @property
    def status(self) -> str:
        if self.n_sentences == 0:
            return "empty_rationale"
        if self.coverage >= EXACT_COVERAGE:
            return "exact"
        if self.coverage >= PARTIAL_COVERAGE:
            return "partial"
        return "unresolved"

    def to_dict(self) -> Dict[str, object]:
        return {
            "rationale_coverage": round(self.coverage, 4),
            "rationale_status": self.status,
            "rationale_sentences": self.n_sentences,
            "rationale_exact": self.n_exact,
            "rationale_partial": self.n_partial,
            "rationale_not_found": self.n_not_found,
            "rationale_ambiguous": self.n_ambiguous,
        }


def align_rationale(document: str, rationale_sentences: Sequence[str]) -> RationaleAlignment:
    """Locate each rationale sentence in ``document`` and return character spans.

    Each sentence is covered greedily by the longest runs that occur in the document.
    A single unmatched sentence therefore degrades to the parts that *do* match rather
    than being discarded, which is what recovers PredEx's concatenation artefacts
    (``"...dismiss the review petition8. We do not find..."`` where the source reads
    ``"...dismiss the review petition. 8. We do not find..."``).

    Matching is monotone: the search starts from the end of the previous match, so
    when a phrase such as "the learned counsel for the appellant" occurs many times
    the occurrence consistent with document order is chosen. Sentences that only match
    before the cursor are retried from the start and counted as ambiguous.
    """
    index = NormalizedIndex(document)
    haystack = index.text

    spans_normalised: List[Span] = []
    cursor = 0
    total_chars = 0
    covered_chars = 0
    n_exact = n_partial = n_not_found = n_ambiguous = 0
    n_sentences = 0

    for sentence in rationale_sentences:
        if len(sentence) < MIN_SENTENCE_CHARS:
            continue
        needle = NormalizedIndex(sentence).text
        if len(needle) < MIN_SEG_CHARS:
            continue

        n_sentences += 1
        total_chars += len(needle)
        sentence_covered = 0
        remainder = needle
        wrapped = False

        while len(remainder) >= MIN_SEG_CHARS:
            length, position = _longest_prefix_match(haystack, remainder, cursor)
            if length < MIN_SEG_CHARS:
                # Nothing usable ahead of the cursor; allow an out-of-order match.
                length, position = _longest_prefix_match(haystack, remainder, 0)
                if length >= MIN_SEG_CHARS:
                    wrapped = True
            if length < MIN_SEG_CHARS:
                remainder = remainder[1:]  # skip a character and try to resynchronise
                continue
            spans_normalised.append((position, position + length))
            sentence_covered += length
            cursor = position + length
            remainder = remainder[length:]

        covered_chars += sentence_covered
        ratio = sentence_covered / len(needle)
        if ratio >= EXACT_COVERAGE:
            n_exact += 1
        elif ratio >= PARTIAL_COVERAGE:
            n_partial += 1
        else:
            n_not_found += 1
        if wrapped:
            n_ambiguous += 1

    spans = merge_spans(
        [index.to_original(a, b) for a, b in spans_normalised if b > a]
    )
    coverage = covered_chars / total_chars if total_chars else 0.0
    return RationaleAlignment(
        spans=spans,
        coverage=coverage,
        n_sentences=n_sentences,
        n_exact=n_exact,
        n_partial=n_partial,
        n_not_found=n_not_found,
        n_ambiguous=n_ambiguous,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Word / token / chunk coordinate system
# ─────────────────────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"\S+")


@dataclass
class DocumentSpace:
    """The shared coordinate system for one document.

    Attributes
    ----------
    words, word_spans
        Whitespace-delimited words and their character spans. The comparison unit.
    token_ids, token_spans
        InLegalBERT sub-word tokens for the whole document (no special tokens).
    token_to_word
        For each global token, the word it belongs to (-1 if it maps to none).
    chunks
        ``(input_ids, attention_mask)`` pairs, byte-identical to the ones the trained
        model consumes in ``opennyai_ablation_fixed.chunk_text``.
    chunk_token_to_global
        ``chunk_token_to_global[c][p]`` is the global token index at position ``p`` of
        chunk ``c``, or -1 for ``[CLS]``/``[SEP]``/``[PAD]``.
    n_visible_tokens
        Global tokens the model actually reads once ``MAX_CHUNKS`` truncation applies.
    """

    document: str
    words: List[str]
    word_spans: List[Span]
    token_ids: List[int]
    token_spans: List[Span]
    token_to_word: np.ndarray
    chunks: List[Tuple[List[int], List[int]]]
    chunk_token_to_global: np.ndarray
    n_visible_tokens: int
    n_tokens_total: int
    truncated: bool
    word_visible: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def words_in_spans(self, spans: Sequence[Span]) -> np.ndarray:
        """Boolean word mask for a set of character spans (any overlap counts)."""
        mask = np.zeros(len(self.words), dtype=bool)
        if not spans:
            return mask
        starts = np.array([s for s, _ in self.word_spans])
        ends = np.array([e for _, e in self.word_spans])
        for span_start, span_end in spans:
            mask |= (np.maximum(starts, span_start) < np.minimum(ends, span_end))
        return mask

    def aggregate_tokens_to_words(
        self, token_scores: np.ndarray, mode: str = "sum"
    ) -> np.ndarray:
        """Fold global sub-word scores up to word scores.

        ``sum`` is the default because Integrated Gradients is additive: the total
        contribution of a word is the sum of its sub-word contributions. ``mean``
        removes the length bias that gives heavily-segmented words more mass, and is
        available as a sensitivity check.
        """
        word_scores = np.zeros(len(self.words), dtype=np.float64)
        counts = np.zeros(len(self.words), dtype=np.float64)
        valid = self.token_to_word >= 0
        np.add.at(word_scores, self.token_to_word[valid], token_scores[valid])
        np.add.at(counts, self.token_to_word[valid], 1.0)
        if mode == "mean":
            word_scores = np.divide(
                word_scores, counts, out=np.zeros_like(word_scores), where=counts > 0
            )
        return word_scores


def build_document_space(document: str, tokenizer) -> DocumentSpace:
    """Tokenise a document once and derive words, tokens, chunks and their mappings.

    The chunking here must stay identical to ``opennyai_ablation_fixed.chunk_text``: 510
    real tokens between ``[CLS]``/``[SEP]``, a stride of ``STRIDE``, padded to
    ``CHUNK_SIZE``, capped at ``MAX_CHUNKS``.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Stage 3 needs a fast tokenizer for character offsets. Load it with "
            "AutoTokenizer.from_pretrained(model_name, use_fast=True)."
        )

    words = []
    word_spans: List[Span] = []
    for match in _WORD_RE.finditer(document):
        words.append(match.group())
        word_spans.append((match.start(), match.end()))

    encoding = tokenizer(
        document,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
        truncation=False,
        verbose=False,
    )
    token_ids: List[int] = list(encoding["input_ids"])
    token_spans: List[Span] = [tuple(o) for o in encoding["offset_mapping"]]
    attention: List[int] = list(encoding["attention_mask"])

    # Map every token to the word containing its first character.
    token_to_word = np.full(len(token_ids), -1, dtype=np.int64)
    if word_spans:
        word_starts = np.array([s for s, _ in word_spans])
        word_ends = np.array([e for _, e in word_spans])
        for i, (start, end) in enumerate(token_spans):
            if end <= start:
                continue  # zero-width offsets (special or stripped tokens)
            candidate = int(np.searchsorted(word_starts, start, side="right") - 1)
            if 0 <= candidate < len(word_spans) and start < word_ends[candidate]:
                token_to_word[i] = candidate

    # Rebuild the model's sliding windows and remember where each position came from.
    effective = CHUNK_SIZE - 2
    chunks: List[Tuple[List[int], List[int]]] = []
    provenance: List[List[int]] = []
    start = 0
    n_tokens = len(token_ids)
    while start < n_tokens:
        end = start + effective
        body_ids = token_ids[start:end]
        body_mask = attention[start:end]
        chunk_ids = [tokenizer.cls_token_id] + body_ids + [tokenizer.sep_token_id]
        chunk_mask = [1] + body_mask + [1]
        chunk_origin = [-1] + list(range(start, min(end, n_tokens))) + [-1]
        pad = CHUNK_SIZE - len(chunk_ids)
        chunk_ids += [tokenizer.pad_token_id] * pad
        chunk_mask += [0] * pad
        chunk_origin += [-1] * pad
        chunks.append((chunk_ids, chunk_mask))
        provenance.append(chunk_origin)
        if end >= n_tokens:
            break
        start += STRIDE

    if n_tokens == 0:
        chunk_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id]
        chunk_ids += [tokenizer.pad_token_id] * (CHUNK_SIZE - 2)
        chunk_mask = [1, 1] + [0] * (CHUNK_SIZE - 2)
        chunks = [(chunk_ids, chunk_mask)]
        provenance = [[-1] * CHUNK_SIZE]

    truncated = len(chunks) > MAX_CHUNKS
    chunks = chunks[:MAX_CHUNKS]
    provenance = provenance[:MAX_CHUNKS]
    chunk_token_to_global = np.array(provenance, dtype=np.int64)

    visible = chunk_token_to_global[chunk_token_to_global >= 0]
    n_visible_tokens = int(visible.max()) + 1 if visible.size else 0

    space = DocumentSpace(
        document=document,
        words=words,
        word_spans=word_spans,
        token_ids=token_ids,
        token_spans=token_spans,
        token_to_word=token_to_word,
        chunks=chunks,
        chunk_token_to_global=chunk_token_to_global,
        n_visible_tokens=n_visible_tokens,
        n_tokens_total=n_tokens,
        truncated=truncated,
    )

    word_visible = np.zeros(len(words), dtype=bool)
    if n_visible_tokens:
        seen = token_to_word[:n_visible_tokens]
        word_visible[seen[seen >= 0]] = True
    space.word_visible = word_visible
    return space


def aggregate_chunk_scores_to_tokens(
    chunk_scores: np.ndarray, space: DocumentSpace, mode: str = "mean"
) -> np.ndarray:
    """Collapse per-chunk token scores onto the global token axis.

    With a 256 stride most tokens appear in two windows, so a rule is needed. ``mean``
    is the default: ``max`` lets a single window boundary inflate a token, which shows
    up as spurious spikes near chunk edges. ``max`` remains available for sensitivity
    analysis.
    """
    totals = np.zeros(space.n_tokens_total, dtype=np.float64)
    counts = np.zeros(space.n_tokens_total, dtype=np.float64)
    best = np.full(space.n_tokens_total, -np.inf, dtype=np.float64)

    origin = space.chunk_token_to_global
    valid = origin >= 0
    flat_origin = origin[valid]
    flat_scores = chunk_scores[valid]

    np.add.at(totals, flat_origin, flat_scores)
    np.add.at(counts, flat_origin, 1.0)
    if mode == "max":
        np.maximum.at(best, flat_origin, flat_scores)
        return np.where(counts > 0, np.where(np.isfinite(best), best, 0.0), 0.0)

    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)


def select_top_words(
    scores: np.ndarray, eligible: np.ndarray, k: int
) -> np.ndarray:
    """Boolean mask of the ``k`` highest-scoring eligible words (ties broken by index)."""
    mask = np.zeros(scores.shape, dtype=bool)
    candidates = np.flatnonzero(eligible)
    if k <= 0 or candidates.size == 0:
        return mask
    k = int(min(k, candidates.size))
    ranked = candidates[np.argsort(-scores[candidates], kind="stable")]
    mask[ranked[:k]] = True
    return mask
