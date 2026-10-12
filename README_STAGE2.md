# Legal Judgment Prediction — Shortcut Learning Ablation (Stage 2)

Capstone project studying whether a legal judgment prediction (LJP) model trained on Indian court
cases actually reasons from case **facts**, or whether it relies on other text in the judgment —
statute citations, precedent references, the court's own reasoning — that carries the outcome.

**Full results and interpretation: [STAGE2_RESULTS.md](STAGE2_RESULTS.md).**

## Model

`HierarchicalLegalBERT` ([opennyai_ablation_fixed.py](opennyai_ablation_fixed.py)):

- **Encoder**: [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), frozen except the last 3
  transformer layers + pooler.
- **Chunking**: documents are split into 512-token chunks (256 stride, max 16 chunks) since case
  text exceeds BERT's context window.
- **Aggregation**: a 2-layer bidirectional GRU over per-chunk `[CLS]` embeddings, followed by a
  linear classifier for the binary outcome label.

## Method

Each case is segmented into the 13 rhetorical roles of
[OpenNyAI](https://github.com/OpenNyAI/Opennyai) (Facts, Arguments, Analysis, Statute, Precedent, Ratio,
lower- and present-court Ruling, ...). Every variant is a subset of the case's sentences **in original
document order**, run through the trained model:

1. **Layer 1 — Graduated ablation** (`v0_facts_only` → `v25` → `v50` → `v75` → `v100_full_spu`): facts plus
   nested random 25/50/75/100% subsets of the pooled non-fact roles (Statute, Precedent, Rulings,
   Analysis).
2. **Layer 2 — Role-specific ablation**: facts plus exactly one role group (`v_analysis_only`,
   `v_sta_only`, `v_pre_only`, `v_rlc_only`, `v_rpc_only`, `v_ratio_only`, `v_issue_only`, `v_arg_only`).
3. **Layer 3 — Token-level tests**: section numbers swapped to 144 in the facts (`v_counterfactual`) and in
   facts + statute (`v_sta_only_cf`); legal references masked (`v_facts_masked`).
4. **Baselines**: `v_full` (all segmented sentences) and `v_raw` (the original text).

"Shift rate" = the share of cases whose prediction differs from `v0_facts_only` for the same case.
`stage2_analysis.py` adds balanced accuracy, ROC-AUC, Holm-corrected McNemar tests, paired bootstrap
CIs, length-matched comparisons and a model-agnostic text probe.

## Results (v2 run, 1,308 PredEx test cases with at least one Facts sentence)

| Variant | Accuracy | ROC-AUC | Shift vs facts-only |
|---|---|---|---|
| `v0_facts_only` | 49.4% | 0.621 | — |
| `v25` / `v50` / `v75` / `v100_full_spu` | 61.1 / 67.7 / 68.7 / 70.8% | 0.730 / 0.772 / 0.794 / 0.814 | 29.1 / 36.2 / 38.5 / 39.4% |
| `v_analysis_only` | 70.3% | 0.805 | 37.9% |
| `v_ratio_only` | 53.2% | 0.681 | 8.7% |
| `v_arg_only` | 51.8% | 0.653 | 15.7% |
| `v_pre_only` | 52.4% | 0.637 | 10.6% |
| `v_rlc_only` | 51.1% | 0.636 | 13.0% |
| `v_sta_only` | 49.7% | 0.618 | 7.3% |
| `v_issue_only` | 50.3% | 0.620 | 3.4% |
| `v_rpc_only` | 49.8% | 0.626 | 2.8% |
| `v_counterfactual` | 49.4% | 0.612 | 2.8% |
| `v_facts_masked` | 48.2% | 0.621 | 8.0% |
| `v_full` / `v_raw` | 70.9 / 71.5% | 0.826 / 0.828 | 38.8% |

![v2 overview](stage2_outputs/fig_v2_overview.png)

**Takeaways:**

- The model's accuracy comes from the court's **Analysis** section: facts + Analysis (70.3%, AUC 0.81) is
  within 1.1 pp of the raw judgment (71.5%, AUC 0.83).
- **Statute text is not the shortcut.** Facts + Statute changes accuracy by +0.3 pp (95% CI −1.2 to +1.8).
  The Review-1 claim that statute analysis was the dominant shortcut came from a v1 bug: `v_sta_only` was
  built from Analysis sentences.
- Facts alone are not "chance": AUC 0.62, but the model predicts "accepted" for only 13% of these short
  inputs against a 56% base rate.
- At matched added length, only court reasoning (Analysis, Ratio) moves predictions toward the true outcome.
  Other roles flip predictions in random directions.
- Section-number swaps flip 4–5% of the predictions they touch and do not change accuracy.

The v1 results (`predex_ablation_results.json`, `combined_ablation.png`) are kept for reference only. See
the changelog at the bottom of `opennyai_ablation_fixed.py` for the 15 bugs fixed.

## Running

See [setup_workstation.sh](setup_workstation.sh) for the remote workstation bootstrap (Python venv, pinned
`spacy`/`opennyai`/`pydantic` install order, HuggingFace model/data pull).

```bash
python opennyai_ablation_fixed.py                      # segmentation (cached) + v2 variants + inference
python stage2_analysis.py                              # CPU: statistics, figures, stage2_outputs/
python stage2_controls.py                              # GPU: control experiments A–F (resumable)
python stage2_analysis.py --controls predex_stage2_controls.jsonl
python -m pytest test_stage2_controls.py -q            # variant-builder tests, no GPU
```

Requires `hf_files/predex_test_deduped.csv`, `hf_files/best_hierarchical_model.pt` and
`hf_files/InLegalBERT/` (from the project's HuggingFace repos), plus the segmentation cache
`predex_opennyai_raw_v2.json` (not checked into git). `stage2_analysis.py` only needs the results JSON
and the segmentation cache.

## Files

| File | Purpose |
|---|---|
| `opennyai_ablation_fixed.py` | Stage 2 pipeline used for the v2 run: segmentation, variants, inference, plots |
| `opennyai_ablation.py` | Same pipeline with different default paths and stricter segmentation checks |
| `stage2_analysis.py` | Statistics, confound checks, text probe and figures (CPU) |
| `stage2_controls.py` | Control experiments: no-facts, leave-one-out, length-matched, transplant, noise floor (GPU) |
| `test_stage2_controls.py` | Tests for the control variant builders |
| `STAGE2_RESULTS.md` | Results, interpretation, threats to validity, what the controls will decide |
| `predex_ablation_results_v2.json` | Per-case predictions for the 18 v2 variants (1,346 cases) |
| `stage2_outputs/` | Generated tables (`tables.md`, CSV), `summary.json`, figures, Stage-3 role lookup |
| `combined_ablation_v2.png` | Figure written by the v2 pipeline itself |
| `requirements.txt` | Pinned dependencies (version-sensitive `spacy`/`opennyai`/`pydantic` stack) |
| `setup_workstation.sh` | Remote GPU workstation bootstrap script |
