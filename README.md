# Legal Judgment Prediction — Shortcut Learning Audit

Capstone project studying whether a legal judgment prediction (LJP) model trained on Indian court
cases actually reasons from case **facts**, or whether it is shortcutting on spurious signals —
statute citations, precedent references, ruling language — that leak the outcome into the input.

The project has three stages:

| Stage | Question | Evidence |
|---|---|---|
| 1 — Training | Can we build a representative LJP model? | Accuracy, ROC-AUC |
| 2 — Ablation | **What information does the prediction depend on?** | Behavioural: prediction shift under ablation |
| 3 — Audit | **Is that information legally meaningful?** | Explanatory: attribution vs expert rationale |

Neither stage proves shortcut learning alone. The claim rests on the two together: a model whose
predictions are highly sensitive to spurious rhetorical content *and* whose attribution does not
align with expert-marked evidence.

## Model

`HierarchicalLegalBERT` ([hierarchical_model.py](hierarchical_model.py)):

- **Encoder**: [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT), frozen except the last 3
  transformer layers + pooler.
- **Chunking**: 512-token windows, 256 stride, max 16 chunks.
- **Aggregation**: 2-layer bidirectional GRU over per-chunk `[CLS]` embeddings, then a linear
  classifier.

The architecture lives in its own module so Stage 2 and Stage 3 cannot drift apart: the audit
explains exactly the network that was trained.

## Stage 2 — Behavioural ablation

`python opennyai_ablation.py`

Each case is segmented into rhetorical roles with [OpenNyAI](https://github.com/OpenNyAI/Opennyai),
then rebuilt into 14 variants. **Every variant is a subset of the original document's sentences,
joined in original order**, so each one is a genuine ablation rather than a regrouping.

1. **Layer 1 — Graduated ablation** (`v0_facts_only` → `v25` → `v50` → `v75` → `v100_full_spu`):
   spurious content injected at increasing fractions of the spurious **token** budget, sampled per
   role with a fixed seed.
2. **Layer 2 — Role-specific** (`v_analysis_only`, `v_statute_only`, `v_pre_only`,
   `v_ruling_only`, `v_arg_rat`): one role type at a time. `v_arg_rat` is the control.
3. **Layer 3 — Counterfactual** (`v_cf_section_fixed`, `v_cf_section_random`,
   `v_cf_section_masked`): section numbers swapped to a fixed value, to a random different
   section, or removed entirely.

Reported per variant: accuracy, hard prediction shift rate with a bootstrap CI, and mean
|ΔP(accepted)| relative to the facts-only baseline.

## Stage 3 — Expert-rationale audit

```bash
python run_stage3.py inspect --csv predex_test_deduped.csv   # schema + alignment dry run
python run_stage3.py audit --limit 5 --html                  # inspect 5 cases by hand
```

On a GPU (Google Colab T4 or the workstation), use [stage3_colab.ipynb](stage3_colab.ipynb) instead of the CLI. Upload this folder to Drive, set the runtime to GPU, and run all. InLegalBERT is pulled from Hugging Face inside the notebook; you only upload the trained `.pt`, the PredEx CSV, and the Stage-2 JSON.

PredEx stores its expert annotation in the `Output` column as `<label>[ds]<rationale>`, where the
rationale is extractive — pivotal sentences copied out of the judgment. The audit locates that text
back in the document and compares it with the model's attribution:

```
PredEx case
    │
    ├── expert rationale ──> character spans ──> word mask
    │                                               │
    └── document ──> InLegalBERT tokens ──> 512/256 chunks
                                                │
                                       Integrated Gradients
                                                │
                              chunk attribution ──> global tokens ──> word scores
                                                │
                                          top-k word mask
                                                │
                            IoU / precision / recall / F1 / attribution share
                                                │
                                  + Stage-2 shortcut sensitivity
                                                │
                                        shortcut audit profile
```

Design decisions that matter for validity:

- **Words, not sub-words, are the comparison unit.** The expert annotation is sentence-level while
  the model works in sub-words; IoU over `constitution` + `##al` fragments would be neither
  faithful nor readable.
- **Attribution runs through the whole network** — tokens → BERT → `[CLS]` → Bi-GRU → classifier —
  not just the encoder, and targets the class **logit** rather than the softmax probability.
- **Overlapping windows are averaged onto a global token axis.** With a 256 stride most tokens
  appear in two chunks; taking the max lets a window boundary inflate a token.
- **Truncation is accounted for.** The model reads at most 4,350 tokens, so rationale beyond that
  is reported as `human_words_out_of_context` and excluded from the metric rather than scored as a
  model failure.
- **A random-selection IoU baseline is reported next to every IoU**, because a rationale covering a
  quarter of the document already scores ≈0.14 by chance.
- **Faithfulness is checked, not assumed.** Masking the top-attributed words must move the
  prediction more than masking the bottom-attributed ones, or the heatmap is not describing the
  model's decision and no IoU computed from it means anything.

Outputs land in `stage3_outputs/`: per-case `stage3_results.csv` (60+ columns), `summary.json`,
`summary.txt`, plots, and per-case HTML heatmaps showing attribution, expert rationale and the
context boundary over the original text.

### Auditing a different model

Everything architecture-specific lives in one class. To audit another LJP model, subclass
`ModelAdapter` with `document_space`, `predict` and `attribute`; the alignment, metrics and
reporting layers are unchanged.

```python
from stage3 import ShortcutAuditor, HierarchicalLegalBERTAdapter, load_predex

adapter = HierarchicalLegalBERTAdapter.from_checkpoint("./InLegalBERT", "./best_hierarchical_model.pt")
results = ShortcutAuditor(adapter).evaluate_dataset(load_predex("./predex_test_deduped.csv"))
```

## Results

Stage-1 model: 71.46% accuracy / ROC-AUC 0.8282 on 1,346 held-out PredEx cases.

Stage-2 numbers in the tables below are from the **pre-fix** run and are being regenerated; see
*Known issues* for why they change.

## Setup

See [setup_workstation.sh](setup_workstation.sh) for the remote workstation bootstrap (Stage 2 / OpenNyAI).

**Stage 3 on Google Colab (recommended):** open [stage3_colab.ipynb](stage3_colab.ipynb), Runtime → T4 GPU, and follow the first cell. Copy onto Drive: `stage3/`, `hierarchical_model.py`, `best_hierarchical_model.pt`, `predex_test_deduped.csv`, `predex_ablation_results.json`.

Local CPU is only for the inspect/tests:

```bash
pip install -r requirements-colab.txt
python test_stage2.py && python test_stage3.py
python run_stage3.py inspect --csv predex_test_deduped.csv
```

## Known issues fixed in the current code

Stage-2 results produced before these fixes should be regenerated:

| Issue | Effect |
|---|---|
| `STA` and `ISSUE` missing from `KNOWN_ROLES` | OpenNyAI emits 13 labels; anything unlisted was coerced to `NONE`, so **every statute sentence was silently dropped from all variants**. The old `v_sta_only` was built from `ANALYSIS`, so the "statute analysis is the dominant shortcut" result was measuring court reasoning, not statute text. |
| `RCJ` referenced but never emitted | `role_buckets.get('RCJ')` always returned empty; the role does not exist in the taxonomy. |
| `v_full` regrouped sentences by role | The "full text" baseline was facts-block, then arguments-block, then analysis-block — not the judgment. Variants are now ordered subsets. |
| Doses counted sentences | Sentence lengths span an order of magnitude, so "25% of sentences" could be 70% of the tokens. Now dosed by token budget, stratified by role, seeded. |
| Always took `spurious[:n]` | Dose was confounded with position in the document. Now randomly sampled with a fixed seed. |
| Single `Section 144` counterfactual | Swaps one meaningful legal reference for another, so it cannot show the *number* is a shortcut. Now three variants separating "which section" from "any section". |
| Hard label shift only | Treated 0.49→0.51 as a total reversal and 0.51→0.99 as no change. Now reports mean \|ΔP\| alongside. |
| `>10% = STRONG shortcut evidence` | An uncalibrated threshold. Replaced with rates and bootstrap CIs. |
| `nn.Module.load_state_dict` monkey-patched to `strict=False` globally | Hid genuine checkpoint mismatches for every module in the process. Now an explicit non-strict load that reports missing/unexpected keys. |
| Case names recorded as `case_0`, `case_1`, … | `itertuples()` renames `Case Name`; Stage 3 then had only row position to join on. Now read from the column directly. |

## Files

| File | Purpose |
|---|---|
| `hierarchical_model.py` | The InLegalBERT + Bi-GRU model, shared by Stage 2 and Stage 3 |
| `opennyai_ablation.py` | Stage 2: segmentation, variant construction, inference, analysis, plots |
| `stage3/` | Stage 3 audit package (see below) |
| `stage3_colab.ipynb` | Colab notebook: GPU Integrated Gradients audit |
| `run_stage3.py` | Stage-3 CLI (workstation; same pipeline as the notebook) |
| `requirements-colab.txt` | Captum + extras; Colab already has torch/transformers |
| `test_stage2.py`, `test_stage3.py` | Regression suites (109 checks, no GPU or checkpoint needed) |
| `predex_ablation_results.json` | Per-case Stage-2 predictions |
| `requirements.txt` | Pinned dependencies (version-sensitive `spacy`/`opennyai`/`pydantic` stack) |

| `stage3/` module | Purpose |
|---|---|
| `config.py` | Chunking, alignment and audit constants, kept in sync with Stage 1/2 |
| `predex.py` | PredEx loading and `<label>[ds]<rationale>` parsing |
| `alignment.py` | Rationale span matching, word/token/chunk coordinate system |
| `adapters.py` | `ModelAdapter` interface + the InLegalBERT implementation |
| `metrics.py` | IoU, precision/recall/F1, attribution shares, faithfulness, correlations |
| `audit.py` | `ShortcutAuditor` and the Stage-2 join |
| `report.py` | Summary, HTML heatmaps, plots |
