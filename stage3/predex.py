"""PredEx loading and expert-rationale extraction.

PredEx stores the expert annotation inside the ``Output`` column using the format::

    <label>[ds]<expert rationale text>

e.g. ``"0[ds]7. It is clear from the order of appointment of the appellant that ..."``

The rationale is *extractive*: annotators selected pivotal sentences out of the
judgment rather than writing new prose. Verified on the public PredEx test split
(3,044 rows): the ``[ds]`` prefix agrees with the ``Label`` column in 100% of rows,
and the rationale is recoverable from ``Input`` at 98.4% mean character coverage
using :mod:`stage3.alignment`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

RATIONALE_DELIMITER = "[ds]"

# Column-name candidates, in priority order. PredEx exports vary between releases
# and the project's deduped CSV may rename things, so detect rather than assume.
_DOCUMENT_COLUMNS = ["Input", "input", "text_input", "document", "Document"]
_RATIONALE_COLUMNS = ["Output", "output", "Explanation", "explanation", "rationale"]
_LABEL_COLUMNS = ["Label", "label", "y", "target"]
_NAME_COLUMNS = ["Case Name", "Case_Name", "case_name", "case_id", "Case ID"]


@dataclass
class PredExCase:
    """One PredEx case: the document the model reads plus the expert rationale."""

    case_id: str
    document: str
    label: int
    rationale_text: str
    index: int = -1
    meta: Dict[str, object] = field(default_factory=dict)


def _pick_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        "Could not find a {} column in the PredEx CSV. Looked for {}; "
        "the file has {}.".format(kind, candidates, df.columns.tolist())
    )


def parse_output_field(raw: object) -> Dict[str, Optional[object]]:
    """Split a PredEx ``Output`` value into its label prefix and rationale text.

    Returns ``label`` (int or None) and ``rationale`` (str). Rows without the
    ``[ds]`` delimiter are returned with ``delimiter_found=False`` so the caller can
    record them as unusable instead of silently treating the whole cell as rationale.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {"label": None, "rationale": "", "delimiter_found": False}

    text = str(raw)
    prefix, delimiter, rationale = text.partition(RATIONALE_DELIMITER)
    if not delimiter:
        return {"label": None, "rationale": "", "delimiter_found": False}

    prefix = prefix.strip()
    label = int(prefix) if re.fullmatch(r"[01]", prefix) else None
    return {"label": label, "rationale": rationale.strip(), "delimiter_found": True}


def split_rationale_sentences(rationale: str, min_chars: int = 30) -> List[str]:
    """Split the rationale into candidate sentences for span alignment.

    PredEx rationales are concatenations of extracted sentences and the join often
    drops the separating whitespace (``"...dismiss the review petition8. We do not
    find..."``). The alignment layer recovers from that, so this splitter only needs
    to produce reasonable candidates, not perfect sentences.
    """
    if not rationale:
        return []
    parts = re.split(r"(?<=[.;:?!])\s+", rationale)
    return [p.strip() for p in parts if len(p.strip()) >= min_chars]


def load_predex(
    csv_path: str,
    document_column: Optional[str] = None,
    rationale_column: Optional[str] = None,
    label_column: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[PredExCase]:
    """Load PredEx cases, keeping only rows that have both a document and a rationale.

    The number of rows dropped, and why, is reported by :func:`describe_predex` so it
    can go into the methodology section rather than vanishing.
    """
    df = pd.read_csv(csv_path)
    doc_col = document_column or _pick_column(df, _DOCUMENT_COLUMNS, "document")
    rat_col = rationale_column or _pick_column(df, _RATIONALE_COLUMNS, "rationale")
    lab_col = label_column or _pick_column(df, _LABEL_COLUMNS, "label")
    name_col = next((c for c in _NAME_COLUMNS if c in df.columns), None)

    cases: List[PredExCase] = []
    for position, (row_index, row) in enumerate(df.iterrows()):
        if limit is not None and len(cases) >= limit:
            break
        document = row[doc_col]
        if document is None or (isinstance(document, float) and pd.isna(document)):
            continue
        document = str(document)
        if not document.strip():
            continue

        parsed = parse_output_field(row[rat_col])
        if not parsed["delimiter_found"] or not parsed["rationale"]:
            continue

        label = row[lab_col]
        label = int(label) if pd.notna(label) else parsed["label"]
        if label is None:
            continue

        case_id = str(row[name_col]) if name_col else "case_{:05d}".format(position)
        cases.append(
            PredExCase(
                case_id=case_id,
                document=document,
                label=int(label),
                rationale_text=str(parsed["rationale"]),
                index=int(row_index),
                meta={"label_from_output": parsed["label"], "row": int(row_index)},
            )
        )
    return cases


def describe_predex(csv_path: str) -> Dict[str, object]:
    """Summarise a PredEx CSV: columns, usable rows, and why rows were dropped."""
    df = pd.read_csv(csv_path)
    doc_col = _pick_column(df, _DOCUMENT_COLUMNS, "document")
    rat_col = _pick_column(df, _RATIONALE_COLUMNS, "rationale")
    lab_col = _pick_column(df, _LABEL_COLUMNS, "label")

    docs = df[doc_col]
    parsed = df[rat_col].map(parse_output_field)
    has_delim = parsed.map(lambda p: p["delimiter_found"])
    prefix_label = parsed.map(lambda p: p["label"])
    agree = (prefix_label == df[lab_col]) & prefix_label.notna()

    return {
        "path": csv_path,
        "columns": df.columns.tolist(),
        "n_rows": int(len(df)),
        "document_column": doc_col,
        "rationale_column": rat_col,
        "label_column": lab_col,
        "n_missing_document": int(docs.isna().sum()),
        "n_missing_delimiter": int((~has_delim).sum()),
        "n_usable": int((docs.notna() & has_delim).sum()),
        "label_prefix_agreement": float(agree.mean()),
        "label_distribution": df[lab_col].value_counts(dropna=False).to_dict(),
        "document_word_len": docs.dropna().astype(str).str.split().str.len()
        .describe()
        .round(1)
        .to_dict(),
    }
