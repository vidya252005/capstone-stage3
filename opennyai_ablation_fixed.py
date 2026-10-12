# ─────────────────────────────────────────────────────────────────────
# opennyai_ablation_fixed.py
#
# Fixes vs the previous script (see CHANGELOG at the bottom of this file).
# Pipeline is now two stages:
#   Stage 1: OpenNyAI segmentation -> cached to RAW_SEG_JSON (document order kept)
#   Stage 2: build variants -> inference -> analysis -> plots
# Stage 2 can be re-run / tweaked without re-segmenting.
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# (OpenNyAI is imported lazily inside Stage 1, so a cached run does not need it)
# ─────────────────────────────────────────────────────────────────────
import os
import re
import json
import math
import random
import hashlib
import warnings
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')                      # headless (Lightning / SSH) safe
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')


@contextmanager
def lenient_state_dict_loading():
    """
    OpenNyAI's checkpoints need strict=False to load. The old script patched
    torch.nn.Module.load_state_dict GLOBALLY, which also silenced real key
    mismatches when loading OUR checkpoint (model could run with a randomly
    initialised GRU/classifier and nobody would know). Now the patch only
    lives while OpenNyAI is running.
    """
    original = torch.nn.Module.load_state_dict

    def patched(self, state_dict, strict=True, *args, **kwargs):
        return original(self, state_dict, False, *args, **kwargs)

    torch.nn.Module.load_state_dict = patched
    try:
        yield
    finally:
        torch.nn.Module.load_state_dict = original


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: CONFIG
# ─────────────────────────────────────────────────────────────────────
CSV_PATH        = "./hf_files/predex_test_deduped.csv"
EXPECTED_N      = 1346
CHECKPOINT_PATH = "./hf_files/best_hierarchical_model.pt"
RAW_SEG_JSON    = './predex_opennyai_raw_v2.json'      # NEW name: never mix with old buggy splits
RESULTS_JSON    = './predex_ablation_results_v2.json'
PLOTS_DIR       = './'
FORCE_RESEGMENT = False                                # True = ignore cache, re-run OpenNyAI

MODEL_NAME      = "./hf_files/InLegalBERT"
GRU_HIDDEN      = 256
GRU_LAYERS      = 2
DROPOUT         = 0.3
UNFREEZE_LAYERS = 3
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU         = torch.cuda.is_available()

CHUNK_SIZE = 512
STRIDE     = 256        # MUST match what the model was trained with - confirm
MAX_CHUNKS = 16

SEED            = 42
NEUTRAL_SECTION = '144'  # NOTE: 144 is a real CrPC section; see CHANGELOG item 12

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: ROLES
# The 13 real OpenNyAI rhetorical-role labels. (The old script also used
# 'RCJ', which is not an OpenNyAI label - that bucket was always empty.)
# ─────────────────────────────────────────────────────────────────────
ALL_ROLES = [
    'PREAMBLE', 'FAC', 'RLC', 'ISSUE', 'ARG_PETITIONER', 'ARG_RESPONDENT',
    'ANALYSIS', 'STA', 'PRE_RELIED', 'PRE_NOT_RELIED', 'RATIO', 'RPC', 'NONE',
]

# Roles pooled as "spurious" for the graduated (dose-response) ablation.
# ARG_* are kept out (pre-outcome control). RATIO is kept out of the pool but
# gets its own Layer-2 variant. Change here if you want a different definition.
SPURIOUS_ROLES = ['STA', 'PRE_RELIED', 'PRE_NOT_RELIED', 'RLC', 'RPC', 'ANALYSIS']

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: TOKEN-LEVEL MASKING & COUNTERFACTUAL
# ─────────────────────────────────────────────────────────────────────
# Handles: Section 302 / Sections 302/34 / Sec. 376(2)(i) / u/s 302 and 34 / S. lists
SECTION_PATTERN = (
    r"\b(Sections?|Secs?\.?|u/s\.?)\s+"
    r"(\d+[A-Z]{0,2}(?:\(\w+\))*"
    r"(?:\s*(?:,|/|&|and|to|r/w|read\s+with)\s*\d+[A-Z]{0,2}(?:\(\w+\))*)*)"
)
SECTION_RE = re.compile(SECTION_PATTERN, re.IGNORECASE)

MASK_PATTERNS = {
    'section_refs':  SECTION_PATTERN,
    'outcome_words': r"\bconvicted\b|\bacquitted\b|\bsentenced\b|\blife\s+imprisonment\b|\bdeath\s+sentence\b",
    'citations':     r"\b(AIR|SCC|SCR|Cr\.?L\.?J)\s*\d{4}\b",
    'act_names':     r"\bI\.?P\.?C\.?\b|\bCr\.?P\.?C\.?\b|\bNDPS\b|\bPOCSO\b|\bArms\s+Act\b",
    'court_names':   r"\bSupreme\s+Court\b|\bHigh\s+Court\b|\bSessions\s+Court\b",
}


def mask_shortcut_tokens(sentence):
    """Mask shortcut tokens but keep sentence structure ('Section 302 IPC' -> '[LEGAL_REF] [LEGAL_REF]')."""
    masked = sentence
    for pattern in MASK_PATTERNS.values():
        masked = re.sub(pattern, '[LEGAL_REF]', masked, flags=re.IGNORECASE)
    return masked


def counterfactual_section(sentence):
    """Swap every section number (incl. lists like 302/34) for one neutral number."""
    return SECTION_RE.sub(lambda m: f"{m.group(1)} {NEUTRAL_SECTION}", sentence)


# ─────────────────────────────────────────────────────────────────────
# SECTION 5: STAGE 1 - OPENNYAI SEGMENTATION
# Output per case: ordered list of {'text', 'role'} in DOCUMENT ORDER.
# ─────────────────────────────────────────────────────────────────────
def parse_annotations(result, unknown_counter):
    sents = []
    for ann in result.get('annotations', []):          # pipeline emits these in sentence order
        text = (ann.get('text') or '').strip()
        if not text:
            continue
        labels = ann.get('labels') or ['NONE']
        role = labels[0]
        if role not in ALL_ROLES:
            unknown_counter[role] += 1                 # keep the label, never silently -> NONE
        sents.append({'text': text, 'role': role})
    return sents


def run_opennyai_segmentation(texts, batch_size=8):
    from opennyai import Pipeline
    from opennyai.utils import Data

    unknown = Counter()
    segmented = [None] * len(texts)
    failed = []

    with lenient_state_dict_loading():
        print("Loading OpenNyAI pipeline...")
        pipeline = Pipeline(components=['Rhetorical_Role'], use_gpu=USE_GPU, verbose=False)
        print(f"Pipeline loaded on {'GPU' if USE_GPU else 'CPU'}")

        def segment(idx_list):
            batch = [texts[i] for i in idx_list]
            data = Data(batch, preprocessing_nlp_model='en_core_web_sm')
            results = pipeline(data)
            if len(results) != len(batch):             # misalignment would silently corrupt every later case
                raise RuntimeError(f"pipeline returned {len(results)} results for {len(batch)} texts")
            return results

        for start in tqdm(range(0, len(texts), batch_size), desc="OpenNyAI segmentation"):
            idxs = list(range(start, min(start + batch_size, len(texts))))
            try:
                results = segment(idxs)
                for i, r in zip(idxs, results):
                    segmented[i] = parse_annotations(r, unknown)
            except Exception as e:
                print(f"  batch {start} failed ({e}); retrying one by one")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for i in idxs:
                    try:
                        segmented[i] = parse_annotations(segment([i])[0], unknown)
                    except Exception as e2:
                        print(f"    case {i} failed: {e2}")
                        segmented[i] = []
                        failed.append(i)

    if unknown:
        print(f"WARNING: labels outside the 13 known roles (kept as-is): {dict(unknown)}")
    return segmented, failed


def sanity_check_segmentation(cases, texts):
    counts = Counter(s['role'] for c in cases for s in c['sentences'])
    total = sum(counts.values())
    print("\nRole distribution after segmentation:")
    for role in ALL_ROLES + [r for r in counts if r not in ALL_ROLES]:
        print(f"  {role:<15} {counts.get(role, 0):>7}  ({100 * counts.get(role, 0) / max(total, 1):5.1f}%)")
    # The old script lost STA / ISSUE into NONE. Only FAC is hard-required; other roles only
    # WARN, because a small sample (e.g. the 5-case smoke test) can legitimately lack a role.
    assert counts.get('FAC', 0) > 0, "No FAC sentences found - segmentation is broken"
    if len(cases) >= 100:
        for role in ['STA', 'ANALYSIS', 'ISSUE', 'RPC']:
            if counts.get(role, 0) == 0:
                print(f"  WARNING: zero {role} sentences across {len(cases)} cases - the matching ablation variant "
                      f"will equal facts-only. Check whether this model emits {role} on this data.")
    else:
        print(f"  (small sample of {len(cases)} cases - role-presence checks skipped)")
    n_empty = sum(1 for c in cases if not c['sentences'])
    n_nofac = sum(1 for c in cases if not any(s['role'] == 'FAC' for s in c['sentences']))
    print(f"\nCases with no sentences at all: {n_empty} | cases with no FAC sentence: {n_nofac}")
    seg_chars = sum(len(s['text']) for c in cases for s in c['sentences'])
    raw_chars = sum(len(t) for t in texts)
    cov = seg_chars / max(raw_chars, 1)
    print(f"Character coverage of segmented text vs raw: {cov:.1%}")
    if cov < 0.85:
        print("  WARNING: segmentation dropped a lot of text - v_full will not equal the raw document.")


def get_segmentation(df, texts, labels, names):
    if os.path.exists(RAW_SEG_JSON) and not FORCE_RESEGMENT:
        print(f"Loading cached segmentation from {RAW_SEG_JSON}")
        with open(RAW_SEG_JSON) as f:
            blob = json.load(f)
        assert blob.get('schema') == 2, "cache is from an old schema - delete it or set FORCE_RESEGMENT=True"
        assert len(blob['cases']) == len(df), "cache size differs from CSV - set FORCE_RESEGMENT=True"
        return blob['cases']

    segmented, failed = run_opennyai_segmentation(texts, batch_size=8)
    cases = [{'index': i, 'case_name': names[i], 'label': labels[i], 'sentences': segmented[i]}
             for i in range(len(df))]
    # Save BEFORE checking, so a failed check can never throw away hours of segmentation.
    with open(RAW_SEG_JSON, 'w') as f:
        json.dump({'schema': 2, 'failed': failed, 'cases': cases}, f)
    print(f"Segmentation saved to {RAW_SEG_JSON} (failed cases: {failed})")
    sanity_check_segmentation(cases, texts)
    return cases


# ─────────────────────────────────────────────────────────────────────
# SECTION 6: BUILD VARIANTS (all built in DOCUMENT ORDER from ordered sentences)
#
# Layer 1  graduated: FAC + nested random subsets (25 < 50 < 75 < 100 %) of the
#          spurious pool. One fixed shuffle per case, so the doses are nested and
#          mix roles evenly (old script took pool[:k] = all STA first, then PRE...,
#          so "dose" was confounded with role type).
# Layer 2  role-specific: FAC + exactly ONE role group.
#          RLC (lower-court ruling) and RPC (present-court ruling = the outcome)
#          are separate now - they are very different kinds of leakage.
#          v_arg_only is the pre-outcome control.
# Layer 3  token-level: counterfactual section swap, token masking.
# Baselines: v_full (all segmented sentences, doc order), v_raw (original text).
# ─────────────────────────────────────────────────────────────────────
LAYER1 = ['v0_facts_only', 'v25', 'v50', 'v75', 'v100_full_spu']
LAYER2 = ['v_sta_only', 'v_pre_only', 'v_rlc_only', 'v_rpc_only',
          'v_analysis_only', 'v_ratio_only', 'v_issue_only', 'v_arg_only']
LAYER3 = ['v_counterfactual', 'v_facts_masked', 'v_sta_only_cf']
BASELINES = ['v_full', 'v_raw']
VARIANT_KEYS = LAYER1 + LAYER2 + LAYER3 + BASELINES

# shift rate is measured against v0, except the STA counterfactual which is
# measured against the unswapped STA variant.
SHIFT_BASE = {k: 'v0_facts_only' for k in VARIANT_KEYS}
SHIFT_BASE['v_sta_only_cf'] = 'v_sta_only'


def build_variants(sents, raw_text, case_seed):
    def ids(roles):
        roles = set(roles)
        return [i for i, s in enumerate(sents) if s['role'] in roles]

    def render(idxs, transform=None, transform_roles=()):
        out = []
        for i in sorted(set(idxs)):                     # sorted = document order
            t = sents[i]['text']
            if transform and sents[i]['role'] in transform_roles:
                t = transform(t)
            out.append(t)
        return ' '.join(out).strip()

    fac = ids(['FAC'])
    sta = ids(['STA'])

    order = ids(SPURIOUS_ROLES)
    random.Random(f"{SEED}-{case_seed}").shuffle(order)

    def dose(frac):
        return fac + order[:math.ceil(frac * len(order))]

    return {
        # Layer 1
        'v0_facts_only': render(fac),
        'v25':           render(dose(0.25)),
        'v50':           render(dose(0.50)),
        'v75':           render(dose(0.75)),
        'v100_full_spu': render(dose(1.00)),
        # Layer 2
        'v_sta_only':      render(fac + sta),
        'v_pre_only':      render(fac + ids(['PRE_RELIED', 'PRE_NOT_RELIED'])),
        'v_rlc_only':      render(fac + ids(['RLC'])),
        'v_rpc_only':      render(fac + ids(['RPC'])),
        'v_analysis_only': render(fac + ids(['ANALYSIS'])),
        'v_ratio_only':    render(fac + ids(['RATIO'])),
        'v_issue_only':    render(fac + ids(['ISSUE'])),
        'v_arg_only':      render(fac + ids(['ARG_PETITIONER', 'ARG_RESPONDENT'])),
        # Layer 3
        'v_counterfactual': render(fac, counterfactual_section, {'FAC'}),
        'v_facts_masked':   render(fac, mask_shortcut_tokens, {'FAC'}),
        'v_sta_only_cf':    render(fac + sta, counterfactual_section, {'FAC', 'STA'}),
        # Baselines
        'v_full': render(range(len(sents))),
        'v_raw':  str(raw_text).strip(),
    }


# ─────────────────────────────────────────────────────────────────────
# SECTION 7: MODEL ARCHITECTURE (unchanged)
# ─────────────────────────────────────────────────────────────────────
class HierarchicalLegalBERT(nn.Module):
    def __init__(self, model_name, gru_hidden=256, gru_layers=2,
                 dropout=0.3, num_classes=2, unfreeze_last_n=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        num_layers = len(self.bert.encoder.layer)
        for i in range(num_layers - unfreeze_last_n, num_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        bert_dim = self.bert.config.hidden_size
        self.gru = nn.GRU(
            input_size=bert_dim, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask, chunk_counts):
        B, max_chunks, seq_len = input_ids.shape
        ids   = input_ids.view(B * max_chunks, seq_len)
        masks = attention_mask.view(B * max_chunks, seq_len)
        bert_out = self.bert(input_ids=ids, attention_mask=masks)
        cls_emb  = bert_out.last_hidden_state[:, 0, :]
        cls_emb  = cls_emb.view(B, max_chunks, -1)
        cls_emb  = self.dropout(cls_emb)
        packed   = nn.utils.rnn.pack_padded_sequence(
            cls_emb, chunk_counts.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        final = self.dropout(final)
        return self.classifier(final)


def load_checkpoint_checked(model, path):
    """Load with strict=False (transformers versions differ on position_ids) but FAIL on any other mismatch."""
    checkpoint = torch.load(path, map_location=DEVICE)
    state = checkpoint['model_state_dict']
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.endswith('position_ids')]
    bad_unexpected = [k for k in unexpected if not k.endswith('position_ids')]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint does not match model.\n  missing: {bad_missing[:8]}\n  unexpected: {bad_unexpected[:8]}")
    return checkpoint


# ─────────────────────────────────────────────────────────────────────
# SECTION 8: CHUNKING + INFERENCE
# ─────────────────────────────────────────────────────────────────────
def chunk_text(text, tokenizer):
    encoding  = tokenizer(str(text), add_special_tokens=False, return_tensors=None)
    all_ids   = encoding['input_ids']
    all_mask  = encoding['attention_mask']
    effective = CHUNK_SIZE - 2
    chunks, start = [], 0
    while start < len(all_ids):
        end        = start + effective
        chunk_ids  = [tokenizer.cls_token_id] + all_ids[start:end]  + [tokenizer.sep_token_id]
        chunk_mask = [1]                       + all_mask[start:end] + [1]
        pad        = CHUNK_SIZE - len(chunk_ids)
        chunk_ids  += [tokenizer.pad_token_id] * pad
        chunk_mask += [0] * pad
        chunks.append((chunk_ids, chunk_mask))
        if end >= len(all_ids):
            break
        start += STRIDE
    return chunks[:MAX_CHUNKS]


def run_inference(text, model, tokenizer):
    if not str(text).strip():
        return -1, 0.0
    chunks = chunk_text(text, tokenizer)
    n      = len(chunks)
    ids    = torch.tensor([c[0] for c in chunks], dtype=torch.long).unsqueeze(0).to(DEVICE)
    masks  = torch.tensor([c[1] for c in chunks], dtype=torch.long).unsqueeze(0).to(DEVICE)
    counts = torch.tensor([n], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, masks, counts)
    probs = F.softmax(logits, dim=-1)
    return probs.argmax().item(), probs.max().item()


# ─────────────────────────────────────────────────────────────────────
# SECTION 9: ANALYSIS
# All variants are scored on the SAME cases (those with >=1 FAC sentence).
# Shift rate is reported two ways: over all cases, and only over cases where
# the variant text actually differs from its baseline (otherwise cases with
# nothing to add / no section number to swap dilute the rate toward 0).
# ─────────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def analyze_results(results):
    valid = [c for c in results if c['predictions']['v0_facts_only']['label'] != -1]
    print(f"\nCases used (>=1 FAC sentence): {len(valid)}/{len(results)}")

    summary = {}
    for vkey in VARIANT_KEYS:
        base_key = SHIFT_BASE[vkey]
        correct, shifted, shifted_changed, confs = [], [], [], []
        for c in valid:
            p, b = c['predictions'][vkey], c['predictions'][base_key]
            if p['label'] == -1 or b['label'] == -1:
                continue
            correct.append(int(p['label'] == c['true_label']))
            confs.append(p['confidence'])
            s = int(p['label'] != b['label'])
            shifted.append(s)
            if p['text_hash'] != b['text_hash']:
                shifted_changed.append(s)
        lo, hi = wilson_ci(sum(shifted_changed), len(shifted_changed))
        summary[vkey] = {
            'n':                  len(correct),
            'accuracy':           round(float(np.mean(correct)) * 100, 2) if correct else 0.0,
            'shift_rate':         round(float(np.mean(shifted)) * 100, 2) if shifted else 0.0,
            'n_changed':          len(shifted_changed),
            'shift_rate_changed': round(float(np.mean(shifted_changed)) * 100, 2) if shifted_changed else 0.0,
            'ci_lo': round(lo, 1), 'ci_hi': round(hi, 1),
            'avg_confidence':     round(float(np.mean(confs)), 4) if confs else 0.0,
            'shift_base':         base_key,
        }

    def table(title, keys):
        print("\n" + "=" * 92)
        print(title)
        print(f"{'Variant':<18}{'Acc':>7}{'Shift(all)':>12}{'n_chg':>7}{'Shift(chg)':>12}{'95% CI':>16}{'AvgConf':>9}")
        print("=" * 92)
        for k in keys:
            s = summary[k]
            print(f"{k:<18}{s['accuracy']:>6.1f}%{s['shift_rate']:>11.1f}%{s['n_changed']:>7}"
                  f"{s['shift_rate_changed']:>11.1f}%{'[%.1f, %.1f]' % (s['ci_lo'], s['ci_hi']):>16}{s['avg_confidence']:>9.3f}")

    table("LAYER 1 - Graduated ablation (shift vs v0_facts_only)", LAYER1)
    table("LAYER 2 - Role-specific ablation (shift vs v0_facts_only)", ['v0_facts_only'] + LAYER2 + ['v_full', 'v_raw'])
    table("LAYER 3 - Token-level tests (shift vs own baseline: cf/masked vs v0, sta_cf vs v_sta_only)", LAYER3)
    print("\nNotes: 'Shift(chg)' only counts cases where the variant text differs from its baseline.")
    print("v_raw accuracy should be close to the model's reported test performance; if it is not,")
    print("suspect the CSV/label encoding/checkpoint before trusting any ablation number.")
    return summary


# ─────────────────────────────────────────────────────────────────────
# SECTION 10: PLOTS
# ─────────────────────────────────────────────────────────────────────
def plot_results(summary):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    levels = [0, 25, 50, 75, 100]
    ax = axes[0]
    ax.plot(levels, [summary[k]['shift_rate'] for k in LAYER1], marker='o', color='crimson', linewidth=2, label='Shift rate')
    ax.plot(levels, [summary[k]['accuracy'] for k in LAYER1], marker='s', color='steelblue', linewidth=2, label='Accuracy')
    ax.set_xlabel('Spurious content level (%)')
    ax.set_ylabel('%')
    ax.set_title('Layer 1: Graduated ablation\n(nested random subsets of spurious pool)')
    ax.set_xticks(levels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    role_labels = {
        'v_sta_only': 'Statute (STA)', 'v_pre_only': 'Precedent (PRE_*)',
        'v_rlc_only': 'Lower-court ruling (RLC)', 'v_rpc_only': 'Present-court ruling (RPC)',
        'v_analysis_only': 'Analysis', 'v_ratio_only': 'Ratio', 'v_issue_only': 'Issues',
        'v_arg_only': 'Arguments (control)',
    }
    keys = sorted(role_labels, key=lambda k: summary[k]['shift_rate'])
    vals = [summary[k]['shift_rate'] for k in keys]
    ax = axes[1]
    bars = ax.barh([role_labels[k] for k in keys], vals, color='steelblue', edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Prediction shift rate vs facts-only (%)')
    ax.set_title('Layer 2: Role-specific ablation\n(FAC + one role group)')
    ax.set_xlim(0, max(vals + [1]) + 15)
    ax.grid(axis='x', alpha=0.3)

    comp_keys = ['v0_facts_only', 'v_sta_only', 'v_analysis_only', 'v100_full_spu', 'v_full', 'v_raw']
    comp_labels = ['Facts only', 'Facts+STA', 'Facts+Analysis', 'Facts+all spurious', 'Segmented full', 'Raw text']
    accs = [summary[k]['accuracy'] for k in comp_keys]
    ax = axes[2]
    bars = ax.bar(comp_labels, accs, color=['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6', '#3498db', '#34495e'], edgecolor='black')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy comparison (key variants)')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'combined_ablation_v2.png')
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 11: MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} cases | columns: {df.columns.tolist()}")
    if len(df) != EXPECTED_N:
        print(f"  WARNING: expected {EXPECTED_N} cases (final deduped set) but found {len(df)} - wrong CSV?")
    for col in ['Input', 'Label']:
        assert col in df.columns, f"missing column {col}"
    texts  = df['Input'].astype(str).tolist()
    labels = df['Label'].astype(int).tolist()          # raises if labels are not numeric
    assert set(labels) <= {0, 1}, f"unexpected label values: {set(labels)}"
    name_col = next((c for c in ('Case Name', 'Case_Name') if c in df.columns), None)   # your CSV uses 'Case Name'
    names  = df[name_col].astype(str).tolist() if name_col else [f'case_{i}' for i in range(len(df))]

    # ── Stage 1 ───────────────────────────────────────────────────────
    print("\nStage 1: OpenNyAI rhetorical role segmentation")
    cases = get_segmentation(df, texts, labels, names)
    if not os.path.exists(RAW_SEG_JSON):
        raise SystemExit("segmentation cache missing")
    print("\nSegmentation sample (case 0):")
    for s in cases[0]['sentences'][:8]:
        print(f"  [{s['role']}] {s['text'][:100]}")

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\nStage 2: loading model from {CHECKPOINT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HierarchicalLegalBERT(MODEL_NAME, GRU_HIDDEN, GRU_LAYERS, DROPOUT,
                                  unfreeze_last_n=UNFREEZE_LAYERS).to(DEVICE)
    checkpoint = load_checkpoint_checked(model, CHECKPOINT_PATH)
    model.eval()
    print(f"  Loaded | epoch {checkpoint.get('epoch')} | best F1 {checkpoint.get('best_f1')}")

    # ── Inference (identical texts share one forward pass) ────────────
    cache = {}

    def predict(text):
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if not text.strip():
            return {'label': -1, 'confidence': 0.0, 'text_hash': h}
        if h not in cache:
            cache[h] = run_inference(text, model, tokenizer)
        label, conf = cache[h]
        return {'label': label, 'confidence': round(conf, 4), 'text_hash': h}

    results = []
    for case in tqdm(cases, desc="Inference"):
        variants = build_variants(case['sentences'], texts[case['index']], case['index'])
        assert set(variants) == set(VARIANT_KEYS)
        results.append({
            'index': case['index'],
            'case_name': case['case_name'],
            'true_label': case['label'],
            'predictions': {k: predict(variants[k]) for k in VARIANT_KEYS},
        })
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {RESULTS_JSON} ({len(cache)} unique forward passes)")

    # ── Analysis ──────────────────────────────────────────────────────
    summary = analyze_results(results)
    plot_results(summary)


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────
# CHANGELOG (bugs fixed)
#  1. KNOWN_ROLES lacked STA and ISSUE -> relabelled NONE. Now all 13 real roles.
#  2. v_sta_only was FAC+ANALYSIS (sta = ANALYSIS). Now STA is STA; ANALYSIS has its own variant.
#  3. 'RCJ' is not an OpenNyAI label; that bucket was always empty. Removed.
#  4. PRE_NOT_RELIED was never used. Now part of the precedent group / spurious pool.
#  5. RLC and RPC were merged into one "ruling" variant. Split (RPC = outcome leakage, RLC = lower court).
#  6. v_arg_rat "control" mixed ARG (pre-outcome) with RATIO (post-hoc reasoning). Split.
#  7. Document order was destroyed: variants/v_full were built bucket-by-bucket (all PREAMBLE, then
#     FAC, ...). v_full was NOT the full text. Everything now built from ordered sentences.
#  8. Graduated ablation took pool[:k] of a role-ordered list (STA first, then PRE...), so dose was
#     confounded with role; max(1, n//4) also collapsed doses for small n. Now nested random subsets.
#  9. Global load_state_dict(strict=False) patch silenced real mismatches when loading OUR checkpoint.
#     Patch is now scoped to OpenNyAI; our checkpoint load fails loudly on any non-position_ids mismatch.
# 10. Batch failure silently replaced 8 cases with empty buckets; result-count mismatch would misalign
#     cases. Now per-case retry, count check, failures recorded. Unknown labels are logged, not folded into NONE.
# 11. Accuracy/shift were computed over different case subsets per variant and diluted by cases where the
#     variant text equals the baseline. Now common case set + conditional shift rate + Wilson CI.
# 12. counterfactual_section missed "Sections 302/34", "Sec.", "u/s", and lists; it also shifted nothing
#     when FAC had no section number but still counted as 0-shift. mask_shortcut_tokens was dead code
#     (now v_facts_masked). Added v_sta_only_cf (swap numbers inside STA - the direct test of the claim).
#     Section 144 is a real CrPC provision, so it is not perfectly neutral; consider a random-section control.
# 13. Plot 3 was titled "Layer 3" (that is the counterfactual). Fixed. Arbitrary ">10%" interpretation removed.
# 14. CSV_PATH pointed at predex_test_deduped.csv, not the final deduped file.
# 15. Segmentation now cached in a v2 file so ablation changes do not need re-segmenting; old buggy
#     splits file is never reused.
# ─────────────────────────────────────────────────────────────────────