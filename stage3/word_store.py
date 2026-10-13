"""Per-word arrays saved from a GPU run, so later analyses run on CPU.

Integrated Gradients is the only expensive step in the audit. Every Stage-3 metric
(IoU at any threshold, role shares, human-supported share) is a cheap function of
five per-word arrays, so storing them means a new question never needs a new GPU run.

Per case (keyed by PredEx row index) the store holds:

- ``scores``   float32 word attribution magnitude (what the audit ranks words by)
- ``signed``   float32 signed word attribution: > 0 pushes toward the predicted class
- ``eligible`` bool, word is inside the model's 16-chunk context window
- ``human``    bool, word is inside the visible expert rationale
- ``roles``    uint16 bitmask, bit ``i`` set when the word is in ``KNOWN_ROLES[i]``
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import KNOWN_ROLES

_ROLE_BIT = {role: 1 << i for i, role in enumerate(KNOWN_ROLES)}


def pack_case(audit) -> Dict[str, np.ndarray]:
    """Arrays for one :class:`CaseAudit` evaluated with ``keep_arrays=True``."""
    scores = np.asarray(audit.word_scores, dtype=np.float32)
    roles = np.zeros(scores.shape[0], dtype=np.uint16)
    for role, mask in (audit.role_masks or {}).items():
        roles[np.asarray(mask, dtype=bool)] |= _ROLE_BIT.get(role, 0)
    signed = getattr(audit, "word_scores_signed", None)
    return {
        "scores": scores,
        "signed": (
            np.zeros_like(scores) if signed is None else np.asarray(signed, dtype=np.float32)
        ),
        "eligible": np.asarray(audit.eligible, dtype=bool),
        "human": np.asarray(audit.human_mask, dtype=bool),
        "roles": roles,
    }


def save_store(path: str, cases: Dict[int, Dict[str, np.ndarray]]) -> None:
    flat = {
        "{}__{}".format(name, row): array
        for row, arrays in cases.items()
        for name, array in arrays.items()
    }
    np.savez_compressed(path, **flat)


def load_store(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    cases: Dict[int, Dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            name, row = key.rsplit("__", 1)
            cases.setdefault(int(row), {})[name] = data[key]
    return cases


def role_mask(roles: np.ndarray, role: str) -> np.ndarray:
    return (roles & _ROLE_BIT[role]) != 0
