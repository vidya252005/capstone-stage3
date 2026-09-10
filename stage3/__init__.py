"""Stage 3: expert-rationale alignment audit for legal judgment prediction models.

The audit is model-agnostic. Everything specific to InLegalBERT + Bi-GRU lives in
:class:`stage3.adapters.HierarchicalLegalBERTAdapter`; auditing another model means
writing another adapter.

Typical use::

    from stage3 import ShortcutAuditor, HierarchicalLegalBERTAdapter, load_predex

    adapter = HierarchicalLegalBERTAdapter.from_checkpoint(
        "./InLegalBERT", "./best_hierarchical_model.pt")
    auditor = ShortcutAuditor(adapter)
    results = auditor.evaluate_dataset(load_predex("./predex_test_deduped.csv"))
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "Attribution",
    "CaseAudit",
    "DocumentSpace",
    "HierarchicalLegalBERTAdapter",
    "ModelAdapter",
    "PredExCase",
    "Prediction",
    "RationaleAlignment",
    "ShortcutAuditor",
    "align_rationale",
    "attach_stage2_measures",
    "build_document_space",
    "describe_predex",
    "format_summary",
    "load_predex",
    "load_role_lookup",
    "load_tokenizer",
    "parse_output_field",
    "save_summary",
    "summarise",
    "write_case_html",
    "write_plots",
]

_ADAPTER_NAMES = {
    "Attribution",
    "HierarchicalLegalBERTAdapter",
    "ModelAdapter",
    "Prediction",
    "load_tokenizer",
}
_ALIGNMENT_NAMES = {
    "DocumentSpace",
    "RationaleAlignment",
    "align_rationale",
    "build_document_space",
}
_AUDIT_NAMES = {
    "CaseAudit",
    "ShortcutAuditor",
    "attach_stage2_measures",
    "load_role_lookup",
}
_PREDEX_NAMES = {
    "PredExCase",
    "describe_predex",
    "load_predex",
    "parse_output_field",
}
_REPORT_NAMES = {
    "format_summary",
    "save_summary",
    "summarise",
    "write_case_html",
    "write_plots",
}


def __getattr__(name: str) -> Any:
    if name in _ADAPTER_NAMES:
        from . import adapters

        return getattr(adapters, name)
    if name in _ALIGNMENT_NAMES:
        from . import alignment

        return getattr(alignment, name)
    if name in _AUDIT_NAMES:
        from . import audit

        return getattr(audit, name)
    if name in _PREDEX_NAMES:
        from . import predex

        return getattr(predex, name)
    if name in _REPORT_NAMES:
        from . import report

        return getattr(report, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
