"""Run the Stage-2 control experiments (stage2_controls.py) on a Modal GPU.

Setup (once, on your laptop; Modal needs Python 3.10+)::

    python3 -m venv .venv-modal
    .venv-modal/bin/pip install 'cbor2==5.6.5' 'modal==1.5.5'
    .venv-modal/bin/modal setup                     # only if not logged in yet

Inputs are looked up next to this file, in ./hf_files/ and in ../data/::

    best_hierarchical_model.pt          (Capstone-LawAI/capstone-legal on Hugging Face)
    predex_test_deduped.csv
    predex_opennyai_raw_v2.json
    predex_ablation_results_v2.json

Smoke test (20 cases, written to smoke_test.jsonl; anchor agreement should be ~100%)::

    .venv-modal/bin/modal run modal_stage2_controls.py --limit 20

Full run (resumable; --detach keeps it running if the laptop sleeps or disconnects)::

    .venv-modal/bin/modal run --detach modal_stage2_controls.py

Fetch the results of a detached run, then analyse locally::

    .venv-modal/bin/modal volume get stage2-controls predex_stage2_controls.jsonl .
    python stage2_analysis.py --controls predex_stage2_controls.jsonl
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GPU = "L4"                     # same GPU as the Stage-3 Modal run
REMOTE_APP = "/root/app"
HF_CACHE = "/cache/huggingface"
OUT_DIR = "/outputs"
FULL_OUT = "predex_stage2_controls.jsonl"
SMOKE_OUT = "smoke_test.jsonl"

INPUTS = {
    "checkpoint": "best_hierarchical_model.pt",
    "csv": "predex_test_deduped.csv",
    "seg": "predex_opennyai_raw_v2.json",
    "v2_results": "predex_ablation_results_v2.json",
}


def find_input(name: str) -> Path | None:
    for folder in (HERE, HERE / "hf_files", HERE.parent / "data"):
        if (folder / name).is_file():
            return folder / name
    return None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers", "safetensors", "pandas", "numpy", "tqdm", "matplotlib")
    .add_local_file(str(HERE / "opennyai_ablation_fixed.py"), f"{REMOTE_APP}/opennyai_ablation_fixed.py")
    .add_local_file(str(HERE / "stage2_controls.py"), f"{REMOTE_APP}/stage2_controls.py")
)
for _name in INPUTS.values():
    _path = find_input(_name)
    if _path is not None:
        image = image.add_local_file(str(_path), f"{REMOTE_APP}/{_name}")

# The Stage-3 run already cached InLegalBERT in this volume, so it is reused.
hf_cache = modal.Volume.from_name("stage3-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("stage2-controls", create_if_missing=True)

app = modal.App("stage2-controls", image=image)


@app.function(gpu=GPU, timeout=24 * 60 * 60, memory=16384,
              volumes={HF_CACHE: hf_cache, OUT_DIR: outputs})
def run_controls(limit: int | None, fp16: bool) -> bytes:
    import subprocess

    import torch

    os.environ["HF_HOME"] = HF_CACHE
    os.chdir(REMOTE_APP)
    sys.path.insert(0, REMOTE_APP)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this Modal container.")
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())

    out_path = f"{OUT_DIR}/{SMOKE_OUT if limit else FULL_OUT}"
    if limit and os.path.exists(out_path):
        os.remove(out_path)        # a smoke test always starts clean; the full run resumes

    # Persist progress every two minutes, so a crash or preemption loses at most that much.
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(120):
            try:
                outputs.commit()
            except Exception as error:      # never let a failed checkpoint save stop the run
                print(f"  (volume commit skipped: {error})")

    threading.Thread(target=commit_loop, daemon=True).start()

    import stage2_controls
    sys.argv = ["stage2_controls.py",
                "--csv", INPUTS["csv"], "--seg", INPUTS["seg"], "--v2-results", INPUTS["v2_results"],
                "--checkpoint", INPUTS["checkpoint"], "--model-name", "law-ai/InLegalBERT",
                "--out", out_path]
    if limit:
        sys.argv += ["--limit", str(limit)]
    if fp16:
        sys.argv += ["--fp16"]
    try:
        stage2_controls.main()
    finally:
        stop.set()
        outputs.commit()
        hf_cache.commit()
    with open(out_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(limit: int = 0, fp16: bool = False):
    missing = [name for name in INPUTS.values() if find_input(name) is None]
    if missing:
        raise SystemExit("Missing input file(s): {}\nPut them next to modal_stage2_controls.py, in ./hf_files/ "
                         "or in ../data/.".format(", ".join(missing)))
    data = run_controls.remote(limit or None, fp16)
    local = HERE / (SMOKE_OUT if limit else FULL_OUT)
    local.write_bytes(data)
    print(f"Saved {local} ({len(data) / 1e6:.1f} MB)")
    if not limit:
        print(f"Next: python stage2_analysis.py --controls {local.name}")
