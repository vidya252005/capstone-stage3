"""Run the Stage-3 GPU audit on Modal, split across several GPUs in parallel.

Setup (once)::

    /usr/local/bin/python3 -m venv .venv-modal
    .venv-modal/bin/pip install 'cbor2==5.6.5' 'modal==1.5.5'

Smoke (20 cases, one GPU, faithfulness on so the timing matches the paper run)::

    .venv-modal/bin/modal run modal_stage3.py

Paper run (all PredEx cases on 8 L4s; costs money)::

    .venv-modal/bin/modal run --detach modal_stage3.py --paper

Each shard checkpoints to the ``stage3-outputs`` volume every 25 cases, so rerunning
the same command resumes instead of starting over. Inputs are the v2 Stage-2 files:
``predex_ablation_results_v2.json`` and ``stage2_outputs/stage3_role_lookup_v2.json``.

Outputs: ./stage3_outputs_modal/stage3_<run>.zip, including ``word_scores.npz``
(per-word attributions, see ``stage3/word_store.py``) for CPU-only re-analysis.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# L4: 24GB, better headroom than T4 for 16-chunk IG + faithfulness.
GPU = "L4"
REMOTE_APP = "/root/app"
HF_CACHE = "/cache/huggingface"
STAGE2_FILE = "predex_ablation_results_v2.json"
ROLE_FILE = "stage3_role_lookup_v2.json"
CONTROLS_FILE = "predex_stage2_controls.jsonl"
INPUTS = {
    "best_hierarchical_model.pt": HERE / "best_hierarchical_model.pt",
    "predex_test_deduped.csv": HERE / "predex_test_deduped.csv",
    STAGE2_FILE: HERE / STAGE2_FILE,
    ROLE_FILE: HERE / "stage2_outputs" / ROLE_FILE,
    CONTROLS_FILE: HERE / CONTROLS_FILE,
}
CHECKPOINT_EVERY = 10

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
    .add_local_file(str(HERE / "stage3_analysis.py"), f"{REMOTE_APP}/stage3_analysis.py")
)
for _name, _path in INPUTS.items():
    if _path.is_file():
        image = image.add_local_file(str(_path), f"{REMOTE_APP}/{_name}")

hf_cache = modal.Volume.from_name("stage3-hf-cache", create_if_missing=True)
outputs_vol = modal.Volume.from_name("stage3-outputs", create_if_missing=True)

app = modal.App("stage3-audit", image=image)


def _setup_remote() -> None:
    os.environ["HF_HOME"] = HF_CACHE
    os.chdir(REMOTE_APP)
    if REMOTE_APP not in sys.path:
        sys.path.insert(0, REMOTE_APP)


@app.function(
    gpu=GPU,
    timeout=24 * 60 * 60,
    memory=32768,
    volumes={HF_CACHE: hf_cache, "/outputs": outputs_vol},
)
def audit_shard(
    shard: int,
    n_shards: int,
    limit: int | None,
    faithfulness: bool,
    ig_steps: int,
    html_limit: int,
    run_name: str,
) -> dict:
    import time

    import pandas as pd
    import torch
    from tqdm import tqdm

    _setup_remote()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this Modal container.")

    from stage3 import (
        HierarchicalLegalBERTAdapter,
        ShortcutAuditor,
        load_predex,
        load_role_lookup,
        write_case_html,
    )
    from stage3.word_store import load_store, pack_case, save_store

    outdir = f"/outputs/{run_name}"
    shard_dir = os.path.join(outdir, "shards")
    html_dir = os.path.join(outdir, "explanations")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)
    shard_csv = os.path.join(shard_dir, "shard_{:02d}.csv".format(shard))
    shard_npz = os.path.join(shard_dir, "shard_{:02d}.npz".format(shard))

    adapter = HierarchicalLegalBERTAdapter.from_checkpoint(
        "law-ai/InLegalBERT",
        f"{REMOTE_APP}/best_hierarchical_model.pt",
        ig_steps=ig_steps,
        ig_internal_batch=1,
        gradient_checkpointing=True,
        allow_untrained=False,
        target_mode="margin",
    )
    report = adapter.checkpoint_report
    if report.get("untrained_aggregator"):
        raise RuntimeError("Checkpoint did not load (untrained aggregator). Abort.")
    hf_cache.commit()

    cases = load_predex(f"{REMOTE_APP}/predex_test_deduped.csv", limit=limit)
    # HTML samples are the first ``html_limit`` cases overall, whichever shard has them.
    html_rows = {c.index for c in cases[:html_limit]}
    mine = cases[shard::n_shards]

    # Resume by row index: case names are not unique in PredEx.
    prior_records, store = [], {}
    if os.path.isfile(shard_csv):
        prior = pd.read_csv(shard_csv)
        prior_records = prior.to_dict(orient="records")
        if os.path.isfile(shard_npz):
            store = load_store(shard_npz)
    done = {int(r["row_index"]) for r in prior_records}
    pending = [c for c in mine if c.index not in done]
    print("Shard {}/{}: {} cases, {} already done, {} to run.".format(
        shard, n_shards, len(mine), len(done), len(pending)))

    auditor = ShortcutAuditor(
        adapter,
        attribution_method="integrated_gradients",
        run_faithfulness=faithfulness,
        role_lookup=load_role_lookup(f"{REMOTE_APP}/{ROLE_FILE}"),
    )

    def flush():
        pd.DataFrame(prior_records + auditor.records).to_csv(shard_csv, index=False)
        save_store(shard_npz, store)
        outputs_vol.commit()

    started = time.time()
    for i, case in enumerate(tqdm(pending, desc="shard {}".format(shard)), start=1):
        audit = auditor.evaluate_case(case, keep_arrays=True)
        if audit.word_scores is not None:
            store[case.index] = pack_case(audit)
        if case.index in html_rows and audit.space is not None:
            safe = "".join(c if c.isalnum() else "_" for c in str(case.case_id))[:60]
            write_case_html(audit, os.path.join(html_dir, "{}_{}.html".format(case.index, safe)))
        if i % CHECKPOINT_EVERY == 0 or i == len(pending):
            flush()
    elapsed = time.time() - started
    return {
        "shard": shard,
        "cases_run": len(pending),
        "n_cases_loaded": len(cases),
        "seconds": round(elapsed, 1),
        "seconds_per_case": round(elapsed / max(1, len(pending)), 2),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_epoch": report.get("epoch"),
        "checkpoint_best_f1": report.get("best_f1"),
    }


@app.function(timeout=60 * 60, memory=16384, volumes={"/outputs": outputs_vol})
def finalize(run_name: str, n_shards: int, n_expected: int, run_info: dict) -> dict:
    import glob
    import io
    import zipfile
    from datetime import datetime, timezone

    import traceback
    from pathlib import Path

    import pandas as pd

    _setup_remote()
    outputs_vol.reload()
    from stage3 import format_summary, save_summary, summarise, write_plots
    from stage3.audit import attach_stage2_measures
    from stage3.config import IG_MAX_RELATIVE_DELTA, IG_STEPS_RETRY, POST_DECISIONAL_ROLES
    from stage3.word_store import load_store, save_store

    outdir = f"/outputs/{run_name}"
    shard_dir = os.path.join(outdir, "shards")
    frames, store = [], {}
    for shard in range(n_shards):
        frames.append(pd.read_csv(os.path.join(shard_dir, "shard_{:02d}.csv".format(shard))))
        npz = os.path.join(shard_dir, "shard_{:02d}.npz".format(shard))
        if os.path.isfile(npz):
            store.update(load_store(npz))
    results = pd.concat(frames, ignore_index=True).sort_values("row_index")
    duplicated = int(results["row_index"].duplicated().sum())
    if duplicated or len(results) != n_expected:
        raise RuntimeError("Expected {} unique cases, got {} rows ({} duplicated).".format(
            n_expected, len(results), duplicated))
    results = attach_stage2_measures(results.reset_index(drop=True), f"{REMOTE_APP}/{STAGE2_FILE}")

    results.to_csv(os.path.join(outdir, "stage3_results.csv"), index=False)
    results.to_json(os.path.join(outdir, "stage3_results.json"), orient="records")
    save_store(os.path.join(outdir, "word_scores.npz"), store)

    config = {
        "run": run_name,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": "law-ai/InLegalBERT",
        "checkpoint_epoch": run_info.get("checkpoint_epoch"),
        "checkpoint_best_f1": run_info.get("checkpoint_best_f1"),
        "gpu": run_info.get("gpu"),
        "n_shards": n_shards,
        "n_cases": int(len(results)),
        "n_cases_with_word_scores": len(store),
        "ig_steps": run_info.get("ig_steps"),
        "ig_steps_retry": IG_STEPS_RETRY,
        "ig_max_relative_delta": IG_MAX_RELATIVE_DELTA,
        "ig_baseline": "pad (CLS/SEP kept)",
        "faithfulness": run_info.get("faithfulness"),
        "gpu_seconds_total": run_info.get("gpu_seconds_total"),
        "stage2_path": STAGE2_FILE,
        "controls_path": CONTROLS_FILE,
        "role_lookup_path": "stage2_outputs/" + ROLE_FILE,
        "chunking": "512/256/16",
        "attribution_target": "decision margin (predicted logit minus other logit)",
        "post_decisional_roles": POST_DECISIONAL_ROLES,
    }
    summary = summarise(results)
    summary["config"] = config
    save_summary(summary, os.path.join(outdir, "summary.json"))
    with open(os.path.join(outdir, "run_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    text = format_summary(summary)
    with open(os.path.join(outdir, "summary.txt"), "w") as handle:
        handle.write(text + "\n")
    write_plots(results, os.path.join(outdir, "plots"))
    outputs_vol.commit()
    try:
        import stage3_analysis

        stage3_analysis.run(Path(outdir), f"{REMOTE_APP}/{STAGE2_FILE}",
                            f"{REMOTE_APP}/{CONTROLS_FILE}", Path(outdir) / "analysis")
        text += "\n\n" + (Path(outdir) / "analysis" / "tables.md").read_text()
        outputs_vol.commit()
    except Exception:
        # The results above are already saved; the analysis can be rerun on a laptop.
        print("stage3_analysis failed (rerun locally):\n" + traceback.format_exc())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in glob.glob(os.path.join(outdir, "**", "*"), recursive=True):
            if os.path.isfile(path) and "/shards/" not in path and not path.endswith(".zip"):
                zf.write(path, os.path.relpath(path, outdir))
    return {"summary": text, "config": config, "zip_bytes": buf.getvalue()}


@app.function(timeout=24 * 60 * 60, volumes={"/outputs": outputs_vol})
def orchestrate(
    run_name: str,
    shards: int,
    limit: int | None,
    faithfulness: bool,
    ig_steps: int,
    html_limit: int,
) -> dict:
    """Runs the shards and the final merge on Modal, so the laptop can disconnect.

    The zip is also saved to the volume as ``<run_name>/stage3_<run_name>.zip``.
    """
    shard_results = list(audit_shard.starmap(
        [(s, shards, limit, faithfulness, ig_steps, html_limit, run_name)
         for s in range(shards)]
    ))
    lines = ["shard {shard}: {cases_run} cases, {seconds}s ({seconds_per_case}s/case)".format(**r)
             for r in shard_results]
    info = dict(shard_results[0])
    info.update(
        ig_steps=ig_steps,
        faithfulness=faithfulness,
        gpu_seconds_total=round(sum(r["seconds"] for r in shard_results), 1),
    )
    lines.append("GPU time this session: {:.2f} h".format(info["gpu_seconds_total"] / 3600))
    result = finalize.remote(run_name, shards, info["n_cases_loaded"], info)
    outputs_vol.reload()
    zip_name = "stage3_{}.zip".format(run_name)
    with open(os.path.join("/outputs", run_name, zip_name), "wb") as handle:
        handle.write(result["zip_bytes"])
    outputs_vol.commit()
    result["summary"] = "\n".join(lines) + "\n\n" + result["summary"]
    return result


@app.local_entrypoint()
def main(
    limit: int = 20,
    shards: int = 1,
    faithfulness: bool = True,
    ig_steps: int = 32,
    html_limit: int = 5,
    paper: bool = False,
    run_name: str = "",
):
    """``--paper`` = all cases, 8 shards, faithfulness on, 20 HTML samples.

    Use ``modal run --detach`` for long runs: the work runs in ``orchestrate`` on Modal,
    so it carries on if this laptop disconnects. Rerunning the same command resumes.
    """
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise SystemExit("Missing input files:\n  " + "\n  ".join(missing))

    if paper:
        limit, shards, faithfulness, html_limit, ig_steps = 0, 8, True, 20, 32
    run_name = run_name or ("paper_v2" if paper else "smoke_v2m_{}".format(limit))
    effective_limit = None if limit == 0 else limit
    zip_name = "stage3_{}.zip".format(run_name)
    dest = HERE / "stage3_outputs_modal"
    print("Stage 3 on Modal: run={} gpu={} x{} limit={} faithfulness={} ig_steps={}".format(
        run_name, GPU, shards, effective_limit or "all", faithfulness, ig_steps))
    print("If this terminal disconnects, the run continues on Modal. Fetch the result with:")
    print("  .venv-modal/bin/modal volume get stage3-outputs {}/{} {}/".format(
        run_name, zip_name, dest))

    call = orchestrate.spawn(
        run_name, shards, effective_limit, faithfulness, ig_steps, html_limit)
    result = call.get()
    dest.mkdir(exist_ok=True)
    zip_path = dest / zip_name
    zip_path.write_bytes(result["zip_bytes"])
    print()
    print(result["summary"])
    print("zip:   ", zip_path)
    print("unzip: unzip -o {} -d {}".format(zip_path, dest / run_name))
