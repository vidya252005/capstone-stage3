#!/usr/bin/env python
"""Stage-3 shortcut-learning audit runner.

Work up in stages rather than launching the full benchmark: each mode below is a
checkpoint where a problem is still cheap to find.

    # 1. what is actually in the CSV, and does the rationale align to the text?
    python run_stage3.py inspect --csv predex_test_deduped.csv

    # 2. five cases, with HTML heatmaps to eyeball before trusting any number
    python run_stage3.py audit --limit 5 --html

    # 3. fifty cases, to shake out truncation and empty-mask edge cases
    python run_stage3.py audit --limit 50

    # 4. the full benchmark, joined to the Stage-2 results
    python run_stage3.py audit --stage2 predex_ablation_results_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_CSV = "./predex_test_deduped.csv"
DEFAULT_CHECKPOINT = "./best_hierarchical_model.pt"
DEFAULT_MODEL = "law-ai/InLegalBERT"
DEFAULT_SPLITS = "./stage2_outputs/stage3_role_lookup_v2.json"
DEFAULT_STAGE2 = "./predex_ablation_results_v2.json"
DEFAULT_OUTDIR = "./stage3_outputs"


def cmd_inspect(args) -> int:
    """Report the CSV schema and how well rationales align, before any model runs."""
    from stage3.alignment import align_rationale
    from stage3.predex import describe_predex, load_predex, split_rationale_sentences

    description = describe_predex(args.csv)
    print("=" * 68)
    print("PREDEX DATASET INSPECTION")
    print("=" * 68)
    for key in ["path", "n_rows", "columns", "document_column", "rationale_column",
                "label_column", "n_missing_document", "n_missing_delimiter", "n_usable"]:
        print("  {:<22} {}".format(key, description[key]))
    print("  {:<22} {:.1%}".format("label_prefix_agreement",
                                   description["label_prefix_agreement"]))
    print("  {:<22} {}".format("label_distribution", description["label_distribution"]))
    print("  {:<22} {}".format("document_word_len", description["document_word_len"]))

    cases = load_predex(args.csv, limit=args.limit or 100)
    print("\n" + "=" * 68)
    print("RATIONALE ALIGNMENT DRY RUN ({} cases, no model needed)".format(len(cases)))
    print("=" * 68)
    coverages, statuses = [], {}
    for case in cases:
        alignment = align_rationale(
            case.document, split_rationale_sentences(case.rationale_text)
        )
        coverages.append(alignment.coverage)
        statuses[alignment.status] = statuses.get(alignment.status, 0) + 1
    if coverages:
        import numpy as np

        coverages = np.array(coverages)
        print("  mean coverage          {:.3f}".format(coverages.mean()))
        print("  median coverage        {:.3f}".format(np.median(coverages)))
        print("  cases >= 95% coverage  {:.1%}".format((coverages >= 0.95).mean()))
        print("  cases <  50% coverage  {:.1%}".format((coverages < 0.50).mean()))
        print("  status counts          {}".format(statuses))
    print("\nIf coverage looks healthy, run:  python run_stage3.py audit --limit 5 --html")
    return 0


def cmd_audit(args) -> int:
    from stage3 import (
        HierarchicalLegalBERTAdapter,
        ShortcutAuditor,
        format_summary,
        load_predex,
        load_role_lookup,
        save_summary,
        summarise,
        write_case_html,
        write_plots,
    )
    from stage3.audit import attach_stage2_measures

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading model from {} (checkpoint {})".format(args.model, args.checkpoint))
    allow_untrained = args.allow_untrained
    if args.checkpoint and not os.path.exists(args.checkpoint) and not allow_untrained:
        print(
            "\nNo trained checkpoint at {}.\n"
            "Loading public InLegalBERT from Hugging Face and keeping an untrained "
            "GRU/classifier so the pipeline can run. Put best_hierarchical_model.pt "
            "in this folder (or pass --checkpoint PATH) before treating numbers as "
            "the Stage-1 model audit.\n".format(args.checkpoint)
        )
        allow_untrained = True
    try:
        adapter = HierarchicalLegalBERTAdapter.from_checkpoint(
            args.model,
            args.checkpoint,
            ig_steps=args.ig_steps,
            ig_internal_batch=args.ig_batch,
            baseline=args.baseline,
            chunk_aggregation=args.chunk_aggregation,
            word_aggregation=args.word_aggregation,
            gradient_checkpointing=args.grad_checkpointing,
            allow_untrained=allow_untrained,
            target_mode=args.target_mode,
        )
    except FileNotFoundError as error:
        print("\nCannot start the audit:\n\n{}\n".format(error))
        return 2
    report = getattr(adapter, "checkpoint_report", {})
    print("  epoch {} | best_f1 {}".format(report.get("epoch"), report.get("best_f1")))
    print("  missing keys: {} | unexpected keys: {}".format(
        report.get("n_missing"), report.get("n_unexpected")))
    if report.get("n_missing"):
        print("  WARNING first missing: {}".format(report["missing_keys"][:5]))
    if report.get("n_unexpected"):
        print("  WARNING first unexpected: {}".format(report["unexpected_keys"][:5]))

    cases = load_predex(args.csv, limit=args.limit)
    print("Loaded {} usable PredEx cases".format(len(cases)))

    role_lookup = load_role_lookup(args.splits) if args.splits else {}
    if role_lookup:
        print("Loaded rhetorical-role buckets for {} cases".format(len(role_lookup)))

    auditor = ShortcutAuditor(
        adapter,
        attribution_method=args.method,
        run_faithfulness=not args.no_faithfulness,
        role_lookup=role_lookup,
    )

    if args.html:
        html_dir = os.path.join(args.outdir, "explanations")
        os.makedirs(html_dir, exist_ok=True)
        import pandas as pd
        from tqdm import tqdm

        for case in tqdm(cases, desc="Stage-3 audit"):
            audit = auditor.evaluate_case(case, keep_arrays=True)
            if audit.space is not None:
                safe = "".join(c if c.isalnum() else "_" for c in str(case.case_id))[:60]
                write_case_html(audit, os.path.join(html_dir, safe + ".html"))
        results = pd.DataFrame(auditor.records)
        print("Wrote per-case heatmaps to {}".format(html_dir))
    else:
        results = auditor.evaluate_dataset(cases)

    if args.stage2 and os.path.exists(args.stage2):
        results = attach_stage2_measures(results, args.stage2)
        print("Joined Stage-2 shortcut measures from {}".format(args.stage2))

    csv_path = os.path.join(args.outdir, "stage3_results.csv")
    results.to_csv(csv_path, index=False)
    results.to_json(os.path.join(args.outdir, "stage3_results.json"), orient="records")

    summary = summarise(results)
    summary["config"] = {
        "attribution_method": args.method,
        "ig_steps": args.ig_steps,
        "attribution_target": args.target_mode,
        "baseline": args.baseline,
        "chunk_aggregation": args.chunk_aggregation,
        "word_aggregation": args.word_aggregation,
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "untrained_aggregator": bool(
            getattr(adapter, "checkpoint_report", {}).get("untrained_aggregator")
        ),
    }
    save_summary(summary, os.path.join(args.outdir, "summary.json"))
    text = format_summary(summary)
    print("\n" + text)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as handle:
        handle.write(text + "\n")

    if not args.no_plots:
        for path in write_plots(results, os.path.join(args.outdir, "plots")):
            print("plot: {}".format(path))
    print("\nResults written to {}".format(csv_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="check the CSV and rationale alignment")
    inspect.add_argument("--csv", default=DEFAULT_CSV)
    inspect.add_argument("--limit", type=int, default=100)
    inspect.set_defaults(func=cmd_inspect)

    audit = subparsers.add_parser("audit", help="run the Stage-3 audit")
    audit.add_argument("--csv", default=DEFAULT_CSV)
    audit.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    audit.add_argument("--model", default=DEFAULT_MODEL,
                       help="local encoder folder or Hugging Face id "
                            "(default: law-ai/InLegalBERT)")
    audit.add_argument("--allow-untrained", action="store_true", dest="allow_untrained",
                       help="run without best_hierarchical_model.pt (pretrained "
                            "InLegalBERT + random GRU; not the paper result)")
    audit.add_argument("--splits", default=DEFAULT_SPLITS,
                       help="OpenNyAI splits JSON, for role attribution shares")
    audit.add_argument("--stage2", default=DEFAULT_STAGE2,
                       help="Stage-2 results JSON, for the shortcut correlation")
    audit.add_argument("--outdir", default=DEFAULT_OUTDIR)
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument("--method", default="integrated_gradients",
                       choices=["integrated_gradients", "gradient_x_input"])
    audit.add_argument("--ig-steps", type=int, default=32, dest="ig_steps")
    audit.add_argument("--target-mode", default="margin", choices=["margin", "logit"],
                       dest="target_mode",
                       help="explain the decision margin (default) or one class logit")
    audit.add_argument("--ig-batch", type=int, default=1, dest="ig_batch")
    audit.add_argument("--baseline", default="pad", choices=["pad", "zero"])
    audit.add_argument("--chunk-aggregation", default="mean", choices=["mean", "max"],
                       dest="chunk_aggregation")
    audit.add_argument("--word-aggregation", default="sum", choices=["sum", "mean"],
                       dest="word_aggregation")
    audit.add_argument("--grad-checkpointing", action="store_true",
                       dest="grad_checkpointing",
                       help="lower GPU memory during Integrated Gradients (~30%% slower)")
    audit.add_argument("--html", action="store_true", help="write per-case heatmaps")
    audit.add_argument("--no-faithfulness", action="store_true", dest="no_faithfulness")
    audit.add_argument("--no-plots", action="store_true", dest="no_plots")
    audit.set_defaults(func=cmd_audit)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    sys.exit(arguments.func(arguments))
