"""Shared configuration for the Stage-3 shortcut audit.

Chunking values must stay identical to the Stage-1/2 model in ``opennyai_ablation.py``
(512-token windows, 256 stride, 16 chunks max) or the attribution map will not line up
with what the trained model actually saw.
"""
from __future__ import annotations

# ── Model / chunking (must mirror opennyai_ablation.py) ──────────────────────
CHUNK_SIZE = 512
STRIDE = 256
MAX_CHUNKS = 16
GRU_HIDDEN = 256
GRU_LAYERS = 2
DROPOUT = 0.3
UNFREEZE_LAYERS = 3
NUM_CLASSES = 2

# ── Alignment ────────────────────────────────────────────────────────────────
# Shortest normalised (alphanumeric-only) run accepted as a genuine rationale match.
# Below ~25 chars, matches in long judgments are frequently coincidental.
MIN_SEG_CHARS = 25
# A rationale sentence counts as fully recovered at this character coverage.
EXACT_COVERAGE = 0.98
# Below this it is recorded as not_found rather than partial.
PARTIAL_COVERAGE = 0.30
# Minimum rationale sentence length (raw chars) worth trying to align.
MIN_SENTENCE_CHARS = 30

# ── Attribution ──────────────────────────────────────────────────────────────
IG_STEPS = 32
IG_INTERNAL_BATCH = 4
# Multiple thresholds are reported so no single arbitrary cut drives the finding.
TOPK_PERCENTS = (0.05, 0.10, 0.20)

# ── Audit quality gates ──────────────────────────────────────────────────────
# Cases below this rationale coverage are flagged and excluded from headline means.
MIN_RATIONALE_COVERAGE = 0.50
# Cases needing at least this many visible human words to yield a stable metric.
MIN_VISIBLE_HUMAN_WORDS = 5

# ── Rhetorical roles ─────────────────────────────────────────────────────────
# The complete OpenNyAI / rhetorical-role-baseline label set (13 labels). Any label
# missing from this list is coerced to NONE by the segmenter, which silently drops
# those sentences from every ablation variant, so it must be exhaustive.
KNOWN_ROLES = [
    "PREAMBLE",
    "FAC",
    "RLC",
    "ISSUE",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "ANALYSIS",
    "STA",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "RATIO",
    "RPC",
    "NONE",
]

# Roles treated as potential shortcut carriers. Used for the shortcut attribution
# share: the fraction of total attribution mass landing inside these roles.
SHORTCUT_ROLES = ["ANALYSIS", "STA", "PRE_RELIED", "PRE_NOT_RELIED", "RLC", "RPC"]
