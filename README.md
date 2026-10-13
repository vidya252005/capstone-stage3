# Detecting Shortcut Learning in Indian Legal Judgment Prediction

PES University capstone project. We test whether a legal judgment prediction (LJP) model trained on
Indian court judgments (ILDC) predicts the outcome from the **facts** of a case, or from other text in the
judgment that already carries the outcome.

**Short answer:** it reads the outcome from the court's own reasoning. The model's accuracy comes from the
judgment's **Analysis** section, text written by the deciding judge after, and because of, the decision.
Statute text, which an earlier version of this project blamed, has no measurable effect. Explanation
methods point the same way but cannot detect the problem on their own.

| Stage | Question | Method | Results |
|---|---|---|---|
| 1. Training | Can we build a representative LJP model? | InLegalBERT + Bi-GRU trained on ILDC | 71.46% accuracy, ROC-AUC 0.83 on PredEx |
| 2. Behavioural ablation | Which parts of the judgment does the prediction depend on? | Rebuild each case from chosen rhetorical roles, re-predict | [STAGE2_RESULTS.md](STAGE2_RESULTS.md) |
| 3. Explanation audit | Where does the model's attribution go, and does it match the expert rationale? | Integrated Gradients vs the PredEx expert rationale and rhetorical roles | [STAGE3_RESULTS.md](STAGE3_RESULTS.md) |

## Main findings

All numbers are on the PredEx test set: 1,346 cases (1,232 High Court, 114 Supreme Court), of which
1,308 have at least one Facts sentence.

**Stage 2 (behaviour):**

- Facts alone give ROC-AUC 0.62. Facts plus the Analysis section give 0.81 (70.3% accuracy), close to the
  full judgment's 0.83 (71.5%).
- Adding statute text changes accuracy by +0.3 pp (95% CI −1.2 to +1.8). Swapping or masking section
  numbers barely moves predictions.
- Six control experiments support outcome leakage. The strongest: when another case's Analysis is
  attached to a case's own facts, P(accepted) is 0.31 higher if that other case was accepted. Removing the
  Analysis costs 11.3 pp accuracy, against 3.9 pp for deleting the same number of random words.

**Stage 3 (explanations):**

- Post-decisional text (Analysis, Ratio, present-court ruling) is 43% of the words the model reads and
  gets 51% of the Integrated Gradients attribution. Facts and statute get less than their share.
- The attribution depends strongly on position: about 3.5 times the average at the end of the text the
  model reads, about half in the middle. Compared with other text at the same position, the outcome-
  stating sections still get about 1.5 times the attribution (Ratio 1.49, present-court ruling 1.61) and
  statute 0.76, but the Analysis as a whole is no longer favoured (0.96).
- The PredEx expert rationales are 80% Analysis and 2% facts, so agreeing with them rewards the leakage.
  Model IoU with the rationale is 0.27, against 0.22 for random words with the same mix of roles.
- Case by case, attribution to the Analysis barely tracks the Stage-2 behaviour (Spearman ρ ≤ 0.10).

**What changed since Review 1:** the "statute is the shortcut" result came from a bug in the first
Stage-2 script (v1). It relabelled every Statute sentence as `NONE`, and its "statute" variant was built
from Analysis sentences. The fixed script (`opennyai_ablation_fixed.py`, changelog at the bottom) produced
the v2 results above. The old Stage-3 run, which used the v1 Stage-2 file, is kept in
`stage3_outputs_modal/paper/` for comparison.

## Model

`HierarchicalLegalBERT` ([hierarchical_model.py](hierarchical_model.py)):

- Encoder: [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), frozen except the last 3 layers and
  the pooler.
- Chunking: 512-token windows, stride 256, at most 16 chunks (about 4,350 tokens). 32.8% of PredEx
  judgments are longer and are truncated.
- Aggregation: a 2-layer bidirectional GRU over the chunks' `[CLS]` embeddings, then a linear classifier.
- Checkpoint `best_hierarchical_model.pt`: epoch 4, best F1 0.732 during training. It is not in git; download it
  from Hugging Face with `hf download Capstone-LawAI/capstone-legal --local-dir . --repo-type model`.

## Repository layout

| Path | What it is |
|---|---|
| `hierarchical_model.py` | The model, shared by Stages 2 and 3 |
| **Stage 2** | |
| `opennyai_ablation_fixed.py` | OpenNyAI segmentation and the 18 v2 variants (the v2 run). Do not edit; the controls import from it |
| `stage2_controls.py`, `modal_stage2_controls.py` | Control experiments A–F, locally or on a Modal GPU |
| `stage2_analysis.py` | Statistics, figures and `stage2_outputs/tables.md`; also writes the Stage-3 role lookup |
| `predex_ablation_results_v2.json`, `predex_stage2_controls.jsonl` | Per-case Stage-2 predictions (v2 run, controls) |
| `stage2_outputs/` | Stage-2 tables, CSVs and figures |
| **Stage 3** | |
| `stage3/` | Audit package: rationale alignment, the Integrated Gradients adapter, metrics, reports, per-word score store |
| `modal_stage3.py` | The GPU run on Modal: parallel shards, resumable, keeps running if the laptop disconnects |
| `run_stage3.py` | The same audit as a command-line tool, for a local GPU |
| `stage3_analysis.py` | CPU analysis of a finished run (role shares, position adjustment, rationale roles, Stage-2 link) |
| `stage3_validate_paper.py` | Permutation tests, truncation and faithfulness checks |
| `stage3_outputs_modal/paper_v2/` | The v2 Stage-3 run: per-case results, `word_scores.npz`, `analysis/`, 20 HTML heatmaps |
| **Tests** | |
| `test_stage2_controls.py`, `stage3/tests/`, `test_stage3.py` | No GPU or checkpoint needed |

`README_STAGE2.md` is the earlier, Stage-2-only README.

## Data not in git

| File | Where it comes from |
|---|---|
| `best_hierarchical_model.pt` | Hugging Face, see [Model](#model) |
| `predex_test_deduped.csv` | The deduplicated PredEx test set (1,346 rows) |
| `predex_opennyai_raw_v2.json` | OpenNyAI segmentation cache written by `opennyai_ablation_fixed.py` |
| `stage2_outputs/stage3_role_lookup_v2.json` | Written by `python stage2_analysis.py` from the segmentation cache (27 MB) |

Do not use `predex_opennyai_splits.json` or `predex_ablation_results.json` / `_named.json`: they are from
the buggy v1 run.

## Reproducing

Python 3.10+ with `torch`, `transformers`, `captum`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` and
`tqdm` (`requirements-colab.txt` lists the extras a Colab runtime needs). Modal needs its own small
environment:

```bash
python3 -m venv .venv-modal && .venv-modal/bin/pip install 'cbor2==5.6.5' 'modal==1.5.5'
.venv-modal/bin/modal setup          # once, opens a browser to log in
```

**Stage 2** (the v2 predictions are in git, so only the analysis is needed):

```bash
python stage2_analysis.py --controls predex_stage2_controls.jsonl     # about 5 minutes; --skip-probe is faster
python stage2_controls.py --dry-run --csv predex_test_deduped.csv     # rebuilds the control variants, no GPU
.venv-modal/bin/modal run --detach modal_stage2_controls.py           # reruns the controls on a GPU (costs money)
```

**Stage 3:**

```bash
.venv-modal/bin/modal run modal_stage3.py --limit 20 --shards 2       # smoke test, a few minutes
.venv-modal/bin/modal run --detach modal_stage3.py --paper            # full run: 8 L4 GPUs, about 2 hours
python stage3_analysis.py --run-dir stage3_outputs_modal/paper_v2     # CPU: tables and figures
python stage3_validate_paper.py                                       # CPU: validation report
```

The full Stage-3 run saves checkpoints on the Modal volume `stage3-outputs`. Rerunning the same command
resumes. If the terminal disconnects, the run continues, and the result can be fetched with
`.venv-modal/bin/modal volume get stage3-outputs paper_v2/stage3_paper_v2.zip stage3_outputs_modal/`.

**Tests:**

```bash
python -m pytest stage3/tests test_stage2_controls.py -q
python test_stage3.py --model law-ai/InLegalBERT       # adapter and chunking checks, downloads the tokenizer
```

## Stage-3 method in brief

1. The PredEx expert rationale (sentences copied out of the judgment) and the OpenNyAI rhetorical roles
   are located in the document and turned into word masks.
2. Integrated Gradients explains the model's decision margin (predicted logit minus the other), through
   BERT, the Bi-GRU and the classifier, from a PAD baseline. Overlapping chunks are averaged onto words.
   Cases whose attributions do not add up to the margin change within 5% are recomputed with 128 steps.
3. Only the text the model actually reads is scored. Rationale beyond the 16-chunk window is excluded,
   not counted as a model failure.
4. Every IoU comes with a random baseline and a role-matched baseline. A faithfulness test checks that
   masking the top-attributed words moves the decision more than masking the bottom-attributed ones.
5. Per-word scores are saved, so every analysis after the GPU run is CPU-only.

To audit a different model, subclass `ModelAdapter` in `stage3/adapters.py` (`document_space`,
`predict`, `attribute`). Alignment, metrics and reporting stay unchanged.
