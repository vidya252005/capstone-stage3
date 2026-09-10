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
from .adapters import (
    Attribution,
    HierarchicalLegalBERTAdapter,
    ModelAdapter,
    Prediction,
    load_tokenizer,
)
from .alignment import (
    DocumentSpace,
    RationaleAlignment,
    align_rationale,
    build_document_space,
)
from .audit import CaseAudit, ShortcutAuditor, attach_stage2_measures, load_role_lookup
from .predex import PredExCase, describe_predex, load_predex, parse_output_field
from .report import format_summary, save_summary, summarise, write_case_html, write_plots

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
