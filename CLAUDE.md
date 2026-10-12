# Capstone: Detecting Shortcut Learning in Indian Legal Judgment Prediction

PES University capstone. Model: InLegalBERT + 2-layer Bi-GRU (`HierarchicalLegalBERT`), trained on ILDC,
tested on the PredEx test set (1,346 cases; 1,308 have at least one Facts sentence). Stage 1 = training,
Stage 2 = behavioural ablation by OpenNyAI rhetorical role, Stage 3 = Integrated Gradients vs expert
rationale.

**The user prefers plain-language, step-by-step explanations.**

## What is in this folder

This is the Stage-3 repo (`stage3/`, `run_stage3.py`, `modal_stage3.py`, `hierarchical_model.py`,
`test_stage3.py`) with the Stage-2 work added at the top level:

- `STAGE2_RESULTS.md`: read this first. Stage-2 findings, threats to validity, and what each control
  experiment would mean (Section 7). Figures and tables are in `stage2_outputs/`.
- `opennyai_ablation_fixed.py` produced the v2 Stage-2 run. Do not modify it: `stage2_controls.py`
  imports its model, chunking, seed and regexes so the two cannot drift.
- `stage2_analysis.py` (CPU statistics + figures), `stage2_controls.py` (GPU control experiments A–F),
  `test_stage2_controls.py`, `modal_stage2_controls.py` (runs the controls on a Modal L4 GPU).
- Inputs (git-ignored, except the v2 results): `best_hierarchical_model.pt` (epoch 4, best F1 0.7319;
  its stored training config matches the 512/256/16 chunking), `predex_test_deduped.csv`,
  `predex_opennyai_raw_v2.json` (v2 segmentation cache), `predex_ablation_results_v2.json`.
- `predex_opennyai_splits.json` is the OLD v1 segmentation, which lost every Statute sentence. Do not use
  it. Stage 3's role lookup should be `stage2_outputs/stage3_role_lookup_v2.json`.
- `README.md` is the Stage-3 README; the Stage-2 README is `README_STAGE2.md`.

## Status

- The v2 Stage-2 results are verified (all 24,228 variant texts match the stored hashes) and analysed.
  Headline: accuracy comes from the court's ANALYSIS section (facts + Analysis 70.3%, AUC 0.81 vs raw
  text 71.5%, AUC 0.83); statute text adds nothing (+0.3 pp). The Review-1 "statute is the shortcut"
  claim came from a v1 bug.
- Controls A–F are run (Modal, 12 Oct 2026; `predex_stage2_controls.jsonl`, 1,308 cases, 100% anchor
  agreement) and analysed. All six support the leakage claim; see `STAGE2_RESULTS.md` Section 7.1.
- Later: rerun Stage 3 with `predex_ablation_results_v2.json` and the v2 role lookup (its last run used
  the v1 Stage-2 file).

## Running the Stage-2 controls on Modal

1. Modal needs Python 3.10+. The default `python3` here is 3.9; `/usr/local/bin/python3` is 3.14:
   `/usr/local/bin/python3 -m venv .venv-modal && .venv-modal/bin/pip install 'cbor2==5.6.5' 'modal==1.5.5'`
2. A Modal login is already saved on this Mac. If Modal says it is not authenticated, the user runs
   `.venv-modal/bin/modal setup` themselves (it opens a browser).
3. Smoke test, a few minutes: `.venv-modal/bin/modal run modal_stage2_controls.py --limit 20`.
   The printed anchor agreement should be about 100% (tiny differences are possible: different GPU and
   transformers version than the v2 run).
4. Full run, costs money, so ask the user first: `.venv-modal/bin/modal run --detach modal_stage2_controls.py`.
   If the laptop disconnected, fetch with `.venv-modal/bin/modal volume get stage2-controls predex_stage2_controls.jsonl .`
5. Analyse: `python3 stage2_analysis.py --controls predex_stage2_controls.jsonl`
   (about 5 minutes because of the text probe; `--skip-probe` skips it).

Checks that need no GPU: `python3 stage2_controls.py --dry-run --csv predex_test_deduped.csv`
(must report 0 anchor hash mismatches) and `python3 -m pytest test_stage2_controls.py -q -p no:cacheprovider`.

Local quirk: `python3` (3.9) has numpy 2 with torch 2.2, so importing torch prints a NumPy warning that
looks like a traceback. It is harmless; the scripts still run.

## Rules

- Never handle the user's passwords or tokens (Hugging Face, Modal, GitHub); the user enters them.
- Ask before anything that costs money (the full Modal run) or publishes (git commit/push).
