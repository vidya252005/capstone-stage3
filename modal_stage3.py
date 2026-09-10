"""Run the Stage-3 GPU audit on Modal.

Setup (once)::

    pip install 'cbor2==5.6.5' 'modal==1.5.5'
    modal setup

Smoke (5 cases + HTML)::

    modal run modal_stage3.py

Paper run — full PredEx, faithfulness on, named Stage-2 join, HTML sample::

    modal run modal_stage3.py --paper

Outputs: ./stage3_outputs_modal/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# L4: 24GB, better headroom than T4 for 16-chunk IG + faithfulness.
# Fall back list is not supported as a single string; change GPU if the queue is long.
GPU = "L4"
REMOTE_APP = "/root/app"
HF_CACHE = "/cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "captum>=0.7.0",
        "transformers",
        "accelerate",
        "safetensors",
        "pandas",
        "scipy",
        "matplotlib",
        "tqdm",
        "numpy",
    )
    .add_local_dir(str(HERE / "stage3"), remote_path=f"{REMOTE_APP}/stage3")
    .add_local_file(str(HERE / "hierarchical_model.py"), f"{REMOTE_APP}/hierarchical_model.py")
    .add_local_file(str(HERE / "run_stage3.py"), f"{REMOTE_APP}/run_stage3.py")
)

for _name in (
    "best_hierarchical_model.pt",
    "predex_test_deduped.csv",
    "predex_ablation_results_named.json",
    "predex_ablation_results.json",
    "predex_opennyai_splits.json",
):
    _path = HERE / _name
    if _path.is_file():
        image = image.add_local_file(str(_path), f"{REMOTE_APP}/{_name}")

hf_cache = modal.Volume.from_name("stage3-hf-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("stage3-outputs", create_if_missing=True)

app = modal.App("stage3-audit", image=image)


def _resolve_stage2() -> str:
    named = f"{REMOTE_APP}/predex_ablation_results_named.json"
    legacy = f"{REMOTE_APP}/predex_ablation_results.json"
    if os.path.isfile(named):
        return named
    if os.path.isfile(legacy):
        return legacy
    return ""


@app.function(
    gpu=GPU,
    timeout=24 * 60 * 60,
    memory=65536,
    volumes={
        HF_CACHE: hf_cache,
        "/outputs": outputs_vol,
    },
)
def audit_remote(
    limit: int | None,
    html: bool,
    faithfulness: bool,
    ig_steps: int,
    html_limit: int,
    paper: bool,
) -> dict:
    import io
    import zipfile
    from argparse import Namespace
    from datetime import datetime, timezone

    import pandas as pd
    import torch
    from tqdm import tqdm

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    os.chdir(REMOTE_APP)
    if REMOTE_APP not in sys.path:
        sys.path.insert(0, REMOTE_APP)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this Modal container.")

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

    ckpt = f"{REMOTE_APP}/best_hierarchical_model.pt"
    csv_path = f"{REMOTE_APP}/predex_test_deduped.csv"
    for path, label in ((ckpt, "checkpoint"), (csv_path, "CSV")):
        if not os.path.isfile(path):
            raise FileNotFoundError("{} missing at {}".format(label, path))

    outdir = "/outputs/paper" if paper else "/outputs/run"
    os.makedirs(outdir, exist_ok=True)
    splits_path = f"{REMOTE_APP}/predex_opennyai_splits.json"
    stage2_path = _resolve_stage2()

    adapter = HierarchicalLegalBERTAdapter.from_checkpoint(
        "law-ai/InLegalBERT",
        ckpt,
        ig_steps=ig_steps,
        ig_internal_batch=1,
        gradient_checkpointing=True,
        allow_untrained=False,
    )
    report = adapter.checkpoint_report
    if report.get("untrained_aggregator"):
        raise RuntimeError("Checkpoint did not load (untrained aggregator). Abort.")

    cases = load_predex(csv_path, limit=limit)
    role_lookup = load_role_lookup(splits_path) if os.path.isfile(splits_path) else {}

    auditor = ShortcutAuditor(
        adapter,
        attribution_method="integrated_gradients",
        run_faithfulness=faithfulness,
        role_lookup=role_lookup,
    )

    html_dir = os.path.join(outdir, "explanations")
    os.makedirs(html_dir, exist_ok=True)
    html_written = 0
    html_budget = html_limit if html else 0
    checkpoint_csv = os.path.join(outdir, "stage3_results.csv")
    checkpoint_every = 25

    done_ids = set()
    prior_records = []
    if os.path.isfile(checkpoint_csv):
        prior = pd.read_csv(checkpoint_csv)
        if "case_id" in prior.columns and len(prior):
            prior_records = prior.to_dict(orient="records")
            done_ids = {str(x) for x in prior["case_id"].tolist()}
            html_written = len(
                [n for n in os.listdir(html_dir) if n.endswith(".html")]
            ) if os.path.isdir(html_dir) else 0
            print(
                "Resuming: {} cases already checkpointed; {} HTML files on disk.".format(
                    len(done_ids), html_written
                )
            )

    pending = [c for c in cases if str(c.case_id) not in done_ids]
    print("Will audit {} remaining of {} loaded cases.".format(len(pending), len(cases)))

    def _flush():
        frame = pd.DataFrame(prior_records + auditor.records)
        frame.to_csv(checkpoint_csv, index=False)
        frame.to_json(os.path.join(outdir, "stage3_results.json"), orient="records")
        outputs_vol.commit()
        print("Checkpoint saved: {} rows → {}".format(len(frame), checkpoint_csv))

    for index, case in enumerate(tqdm(pending, desc="Stage-3 paper audit"), start=1):
        keep = html_written < html_budget
        audit = auditor.evaluate_case(case, keep_arrays=keep)
        if keep and audit.space is not None:
            safe = "".join(c if c.isalnum() else "_" for c in str(case.case_id))[:60]
            write_case_html(audit, os.path.join(html_dir, safe + ".html"))
            html_written += 1
        if index % checkpoint_every == 0 or index == len(pending):
            _flush()

    results = pd.DataFrame(prior_records + auditor.records)
    if stage2_path:
        results = attach_stage2_measures(results, stage2_path)

    results.to_csv(checkpoint_csv, index=False)
    results.to_json(os.path.join(outdir, "stage3_results.json"), orient="records")

    summary = summarise(results)
    paper_config = {
        "profile": "paper" if paper else "custom",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "encoder": "law-ai/InLegalBERT",
        "checkpoint": ckpt,
        "checkpoint_epoch": report.get("epoch"),
        "checkpoint_best_f1": report.get("best_f1"),
        "csv": csv_path,
        "n_cases_requested": None if limit is None else int(limit),
        "n_cases_loaded": len(cases),
        "ig_steps": ig_steps,
        "faithfulness": bool(faithfulness),
        "html_limit": html_budget,
        "html_written": html_written,
        "stage2_path": stage2_path or None,
        "splits_path": splits_path if os.path.isfile(splits_path) else None,
        "grad_checkpointing": True,
        "chunking": "512/256/16",
        "attribution_target": "predicted_class_logit",
        "limitations": [
            "Stage-2 JSON may still be from the pre-fix OpenNyAI run "
            "(STA/ANALYSIS bug). Treat shortcut correlation as provisional "
            "until Stage 2 is regenerated.",
            "Role attribution shares require predex_opennyai_splits.json.",
            "Named Stage-2 file was patched from CSV row order if "
            "predex_ablation_results_named.json was used.",
        ],
    }
    summary["config"] = paper_config
    save_summary(summary, os.path.join(outdir, "summary.json"))
    with open(os.path.join(outdir, "paper_config.json"), "w") as handle:
        json.dump(paper_config, handle, indent=2)
    text = format_summary(summary)
    with open(os.path.join(outdir, "summary.txt"), "w") as handle:
        handle.write(text + "\n")
    print(text)

    for path in write_plots(results, os.path.join(outdir, "plots")):
        print("plot:", path)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(outdir):
            for name in filenames:
                full = os.path.join(dirpath, name)
                zf.write(full, os.path.relpath(full, outdir))
    payload = buf.getvalue()
    outputs_vol.commit()
    hf_cache.commit()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "summary": text,
        "config": paper_config,
        "zip_bytes": payload,
        "zip_mb": round(len(payload) / 1e6, 2),
    }


@app.local_entrypoint()
def main(
    limit: int = 5,
    html: bool = True,
    faithfulness: bool = False,
    ig_steps: int = 32,
    html_limit: int = 5,
    paper: bool = False,
):
    """``--paper`` = full PredEx, faithfulness on, 20 HTML samples, named Stage-2.

    ``--limit 0`` means all cases. Without ``--paper``, defaults stay smoke-test sized.
    """
    missing = [
        name
        for name in ("best_hierarchical_model.pt", "predex_test_deduped.csv")
        if not (HERE / name).is_file()
    ]
    if missing:
        raise SystemExit(
            "Missing required files next to modal_stage3.py:\n  " + "\n  ".join(missing)
        )

    named = HERE / "predex_ablation_results_named.json"
    if not named.is_file() and (HERE / "predex_ablation_results.json").is_file():
        print(
            "WARNING: predex_ablation_results_named.json missing. "
            "Run: python patch_stage2_case_names.py"
        )

    if paper:
        limit = 0
        faithfulness = True
        html = True
        html_limit = 20
        ig_steps = 32
        print("=" * 68)
        print("PAPER PROFILE")
        print("  cases:          all PredEx (limit=0)")
        print("  faithfulness:   ON")
        print("  HTML samples:   {}".format(html_limit))
        print("  IG steps:       {}".format(ig_steps))
        print("  GPU:            {}".format(GPU))
        print("  Stage-2:        prefer predex_ablation_results_named.json")
        print("  Expect ~12–24h on L4 with faithfulness; keep this terminal open.")
        print("=" * 68)
        print(
            "JURIX note: Stage-2 may still be pre-fix (STA bug). "
            "Report IoU/faithfulness as primary; treat shortcut correlation as provisional."
        )

    effective_limit = None if limit == 0 else limit
    print(
        "Submitting Stage 3 → Modal gpu={} limit={} faithfulness={} html_limit={}".format(
            GPU, effective_limit, faithfulness, html_limit if html else 0
        )
    )
    result = audit_remote.remote(
        effective_limit,
        html,
        faithfulness,
        ig_steps,
        html_limit if html else 0,
        paper,
    )
    dest = HERE / "stage3_outputs_modal"
    dest.mkdir(exist_ok=True)
    stamp = "paper" if paper else "run"
    zip_path = dest / "stage3_{}.zip".format(stamp)
    zip_path.write_bytes(result["zip_bytes"])
    (dest / "paper_config.json").write_text(json.dumps(result["config"], indent=2))
    (dest / "summary.txt").write_text(result["summary"] + "\n")
    print("GPU     ", result["gpu"])
    print("zip     ", zip_path, "({} MB)".format(result["zip_mb"]))
    print()
    print(result["summary"])
    print("Unzip:  unzip -o {} -d {}".format(zip_path, dest / stamp))
