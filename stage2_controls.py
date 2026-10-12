# ─────────────────────────────────────────────────────────────────────
# stage2_controls.py
#
# Control experiments that turn the v2 ablation (opennyai_ablation_fixed.py)
# into a defensible Stage-2 result. The v2 run shows WHICH content moves the
# prediction; these runs rule out the obvious alternative explanations.
#
#   A  nofac_*   no-facts inputs        -> the original "Spurious Only" arm, so the
#                                          planned rule Pred(Full) = Pred(Spurious)
#                                          != Pred(Facts) can actually be evaluated
#   B  loo_*     leave-one-role-out     -> necessity, measured on near-full documents
#                                          (in-distribution), each paired with a random
#                                          deletion of the same number of words
#   C  lm*_*     length-matched add-in  -> FAC + exactly B words of one role, against
#                                          "more text, no new information" controls
#   D  tx_*      cross-case transplant  -> FAC of case i + role text from a donor case
#                                          whose outcome is known; does the prediction
#                                          follow the DONOR's outcome?
#   E  nf_*/cf_* noise floor + section  -> how much a meaning-preserving edit moves the
#                counterfactuals           model, so the section-swap rates can be read
#   F  dnoA_*    dose curve without     -> Layer 1 again, but the pool excludes ANALYSIS
#                ANALYSIS
#
# Re-uses the model, chunking and section regex from opennyai_ablation_fixed.py
# so nothing can drift. Needs the v2 segmentation cache; OpenNyAI is NOT needed.
#
#   python stage2_controls.py --dry-run          # build variants only (CPU, no model)
#   python stage2_controls.py --limit 20         # smoke test on GPU
#   python stage2_controls.py                    # full run (resumable)
# ─────────────────────────────────────────────────────────────────────
import os
import json
import math
import random
import hashlib
import argparse
from collections import Counter, defaultdict

import opennyai_ablation_fixed as base

SEED = base.SEED

ROLE_GROUPS = {
    'ANALYSIS': ['ANALYSIS'],
    'STA':      ['STA'],
    'PRE':      ['PRE_RELIED', 'PRE_NOT_RELIED'],
    'RLC':      ['RLC'],
    'RPC':      ['RPC'],
    'RATIO':    ['RATIO'],
    'ISSUE':    ['ISSUE'],
    'ARG':      ['ARG_PETITIONER', 'ARG_RESPONDENT'],
    'FAC':      ['FAC'],
}
# ANALYSIS + RATIO + RPC: text written by the deciding court about its own decision.
POST_DECISIONAL = ['ANALYSIS', 'RATIO', 'RPC']
NON_ANALYSIS_POOL = ['STA', 'PRE_RELIED', 'PRE_NOT_RELIED', 'RLC', 'RPC']

LOO_GROUPS = ['ANALYSIS', 'STA', 'PRE', 'RLC', 'RPC', 'RATIO', 'ISSUE', 'ARG', 'FAC']
LM_ROLES = ['ANALYSIS', 'STA', 'PRE', 'RLC', 'ARG', 'RATIO']
LM_BUDGETS = [60, 150]                       # words; 150 keeps >=350 eligible cases for STA/PRE/RLC/ARG
TX_ROLES = ['ANALYSIS', 'STA', 'PRE', 'RLC', 'ARG']

# Real IPC / CrPC / NDPS sections used as random replacements. Varying the number
# separates "this particular section" from "a section number is present".
RANDOM_SECTIONS = ['34', '120B', '147', '148', '149', '201', '279', '302', '304', '304B', '306',
                   '307', '323', '324', '325', '326', '354', '363', '366', '376', '379', '380',
                   '392', '394', '395', '397', '406', '409', '420', '427', '436', '447', '448',
                   '452', '467', '468', '471', '498A', '506']

ANCHORS = ['v0_facts_only', 'v_full', 'v_raw']   # re-run to prove we reproduce the v2 run


def _words(text):
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────
# Variant construction. Every builder returns '' when the variant does not
# apply to the case; '' is predicted as label -1 and excluded downstream.
# ─────────────────────────────────────────────────────────────────────
class CaseText:
    """Ordered sentences of one case plus helpers that render subsets in document order."""

    def __init__(self, sents):
        self.sents = sents
        self.wc = [_words(s['text']) for s in sents]

    def ids(self, roles):
        roles = set(roles)
        return [i for i, s in enumerate(self.sents) if s['role'] in roles]

    def group_ids(self, group):
        return self.ids(ROLE_GROUPS[group])

    def words_in(self, idxs):
        return sum(self.wc[i] for i in idxs)

    def render(self, idxs, overrides=None, transform=None, transform_roles=()):
        """Join the chosen sentences in document order. `overrides` replaces a sentence's text."""
        overrides = overrides or {}
        out = []
        for i in sorted(set(idxs)):
            t = overrides.get(i, self.sents[i]['text'])
            if transform and self.sents[i]['role'] in transform_roles:
                t = transform(t)
            out.append(t)
        return ' '.join(out).strip()


def take_budget(ct, idxs, budget, mode, rng=None):
    """Pick sentences from `idxs` until exactly `budget` words are covered.

    mode 'random': random sentences (one cut sentence keeps its leading words)
    mode 'first' : contiguous from the start of the role, last sentence cut at its end
    mode 'last'  : contiguous from the end of the role, first sentence cut at its start
    Returns (chosen_ids, overrides) or (None, None) if the role has fewer than `budget` words.
    """
    if ct.words_in(idxs) < budget:
        return None, None
    if mode == 'random':
        order = list(idxs)
        rng.shuffle(order)
    elif mode == 'first':
        order = sorted(idxs)
    elif mode == 'last':
        order = sorted(idxs, reverse=True)
    else:
        raise ValueError(mode)
    chosen, overrides, used = [], {}, 0
    for i in order:
        need = budget - used
        if need <= 0:
            break
        words = ct.sents[i]['text'].split()
        if len(words) > need:
            overrides[i] = ' '.join(words[-need:] if mode == 'last' else words[:need])
            used += need
        else:
            used += len(words)
        chosen.append(i)
    return chosen, overrides


def random_deletion(ct, target_words, rng):
    """Delete random sentences (any role) until >= target_words are removed; return the kept ids."""
    if target_words <= 0:
        return list(range(len(ct.sents)))
    order = list(range(len(ct.sents)))
    rng.shuffle(order)
    removed, gone = 0, set()
    for i in order:
        if removed >= target_words:
            break
        gone.add(i)
        removed += ct.wc[i]
    return [i for i in range(len(ct.sents)) if i not in gone]


def section_swapper(rng):
    def swap(sentence):
        def repl(m):
            original = re_first_number(m.group(2))
            choices = [s for s in RANDOM_SECTIONS if s != original]
            return f"{m.group(1)} {rng.choice(choices)}"
        return base.SECTION_RE.sub(repl, sentence)
    return swap


def re_first_number(group_text):
    return group_text.split('/')[0].split(',')[0].split()[0].split('(')[0]


def remove_section_numbers(sentence):
    """'Section 302 IPC' -> 'Section IPC': a section is mentioned, but not which."""
    return base.SECTION_RE.sub(lambda m: m.group(1), sentence)


def pick_donors(cases, group, seed):
    """For every case, one donor with label 1 and one with label 0 for role `group`.

    Donors are other cases that have the role, chosen among the 5 whose role length is
    closest to the receiver's own (or the median, if the receiver lacks the role), so the
    two transplants for one receiver add a similar amount of text.
    """
    lengths = {}
    for c in cases:
        ct = c['_ct']
        lengths[c['index']] = ct.words_in(ct.group_ids(group))
    pool = {lab: [c for c in cases if c['label'] == lab and lengths[c['index']] >= 20] for lab in (0, 1)}
    present = sorted(v for v in lengths.values() if v >= 20)
    median = present[len(present) // 2] if present else 0
    donors = {}
    for c in cases:
        rng = random.Random(f"{seed}-donor-{group}-{c['index']}")
        target = lengths[c['index']] if lengths[c['index']] >= 20 else median
        pair = {}
        for lab in (0, 1):
            cands = [d for d in pool[lab] if d['index'] != c['index']]
            if not cands:
                continue
            cands.sort(key=lambda d: (abs(lengths[d['index']] - target), d['index']))
            pair[lab] = rng.choice(cands[:5])['index']
        donors[c['index']] = pair
    return donors


def build_control_variants(case, by_index, raw_text, donors, fact_donors):
    ct = case['_ct']
    idx = case['index']

    def rng(tag):                                            # one independent stream per variant
        return random.Random(f"{SEED}-{tag}-{idx}")

    fac = ct.group_ids('FAC')
    all_ids = list(range(len(ct.sents)))
    v = {}

    # Anchors (identical construction to the v2 script)
    v['v0_facts_only'] = ct.render(fac)
    v['v_full'] = ct.render(all_ids)
    v['v_raw'] = str(raw_text).strip()

    # A — no facts at all
    v['nofac_all'] = ct.render([i for i in all_ids if i not in set(fac)])
    v['nofac_analysis'] = ct.render(ct.ids(['ANALYSIS']))
    v['nofac_spurious'] = ct.render(ct.ids(base.SPURIOUS_ROLES))
    v['nofac_nonanalysis'] = ct.render(ct.ids(NON_ANALYSIS_POOL))

    # B — leave one role group out of the full document, plus a matched random deletion
    loo_sets = {g: ct.group_ids(g) for g in LOO_GROUPS}
    loo_sets['POSTDEC'] = ct.ids(POST_DECISIONAL)
    for g, drop in loo_sets.items():
        name = g.lower()
        if not drop:
            v[f'loo_{name}'] = ''
            v[f'loo_rand_{name}'] = ''
            continue
        dropped = set(drop)
        v[f'loo_{name}'] = ct.render([i for i in all_ids if i not in dropped])
        keep = random_deletion(ct, ct.words_in(drop), rng(f'loo-rand-{g}'))
        v[f'loo_rand_{name}'] = ct.render(keep)

    # C — length-matched: FAC + exactly B words of one role
    for B in LM_BUDGETS:
        for g in LM_ROLES:
            chosen, ov = take_budget(ct, ct.group_ids(g), B, 'random', rng(f'lm{B}-{g}'))
            v[f'lm{B}_{g.lower()}'] = ct.render(fac + chosen, ov) if (fac and chosen) else ''
        for mode in ('first', 'last'):
            chosen, ov = take_budget(ct, ct.group_ids('ANALYSIS'), B, mode)
            v[f'lm{B}_analysis_{mode}'] = ct.render(fac + chosen, ov) if (fac and chosen) else ''
        # "more text, nothing new": the case's own facts repeated
        fac_text = ct.render(fac)
        fac_words = fac_text.split()
        v[f'lm{B}_dupfacts'] = (fac_text + ' ' + ' '.join(fac_words[:B])) if len(fac_words) >= B else ''
        # "more legal text, unrelated case": B words of another case's facts
        pool = [d for d in fact_donors[B] if d != idx]
        dct = by_index[rng(f'lm{B}-other').choice(pool)]['_ct']
        chosen, ov = take_budget(dct, dct.group_ids('FAC'), B, 'random', rng(f'lm{B}-otherfacts'))
        v[f'lm{B}_otherfacts'] = (fac_text + ' ' + dct.render(chosen, ov)) if (fac and chosen) else ''

    # D — transplant: own facts + role text from a donor with a known outcome.
    # tx_<g>_self uses the same "append" construction with the case's own text.
    fac_text = ct.render(fac)
    for g in TX_ROLES:
        own = ct.render(ct.group_ids(g))
        v[f'tx_{g.lower()}_self'] = (fac_text + ' ' + own) if (fac and own) else ''
        for lab in (0, 1):
            d_idx = donors[g].get(idx, {}).get(lab)
            if not fac or d_idx is None:
                v[f'tx_{g.lower()}_d{lab}'] = ''
                continue
            dct = by_index[d_idx]['_ct']
            v[f'tx_{g.lower()}_d{lab}'] = fac_text + ' ' + dct.render(dct.group_ids(g))

    # E — noise floor and section counterfactuals
    if len(fac) >= 3:
        drop = rng('nf-fac').choice(fac)
        v['nf_drop1_fac'] = ct.render([i for i in fac if i != drop])
    else:
        v['nf_drop1_fac'] = ''
    if len(all_ids) >= 3:
        drop = rng('nf-full').choice(all_ids)
        v['nf_drop1_full'] = ct.render([i for i in all_ids if i != drop])
    else:
        v['nf_drop1_full'] = ''
    for s in range(3):
        v[f'cf_fac_random_s{s}'] = ct.render(fac, transform=section_swapper(rng(f'cf-fac-{s}')), transform_roles={'FAC'})
    v['cf_fac_removed'] = ct.render(fac, transform=remove_section_numbers, transform_roles={'FAC'})
    raw = str(raw_text).strip()
    v['cf_full_fixed'] = base.counterfactual_section(raw)
    v['cf_full_random'] = section_swapper(rng('cf-full'))(raw)
    v['cf_full_removed'] = remove_section_numbers(raw)
    v['cf_full_masked'] = base.mask_shortcut_tokens(raw)

    # F — dose curve with ANALYSIS excluded from the pool (nested subsets, like v2 Layer 1)
    order = ct.ids(NON_ANALYSIS_POOL)
    random.Random(f"{SEED}-dnoA-{idx}").shuffle(order)
    for pct in (25, 50, 75, 100):
        v[f'dnoA_{pct}'] = ct.render(fac + order[:math.ceil(pct / 100 * len(order))]) if order else ''

    return v


def shift_bases():
    """Which prediction each variant's shift is measured against."""
    b = {}
    for k in ['nofac_all', 'nofac_analysis', 'nofac_spurious', 'nofac_nonanalysis']:
        b[k] = 'v_raw'
    for g in LOO_GROUPS + ['POSTDEC']:
        b[f'loo_{g.lower()}'] = 'v_full'
        b[f'loo_rand_{g.lower()}'] = 'v_full'
    for B in LM_BUDGETS:
        for g in LM_ROLES:
            b[f'lm{B}_{g.lower()}'] = 'v0_facts_only'
        for k in ('analysis_first', 'analysis_last', 'dupfacts', 'otherfacts'):
            b[f'lm{B}_{k}'] = 'v0_facts_only'
    for g in TX_ROLES:
        for k in ('self', 'd0', 'd1'):
            b[f'tx_{g.lower()}_{k}'] = 'v0_facts_only'
    b['nf_drop1_fac'] = 'v0_facts_only'
    b['nf_drop1_full'] = 'v_full'
    for s in range(3):
        b[f'cf_fac_random_s{s}'] = 'v0_facts_only'
    b['cf_fac_removed'] = 'v0_facts_only'
    for k in ('fixed', 'random', 'removed', 'masked'):
        b[f'cf_full_{k}'] = 'v_raw'
    for pct in (25, 50, 75, 100):
        b[f'dnoA_{pct}'] = 'v0_facts_only'
    for a in ANCHORS:
        b[a] = 'v0_facts_only'
    return b


SHIFT_BASE = shift_bases()


# ─────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────
def make_predictor(model, tokenizer, fp16):
    import torch
    import torch.nn.functional as F

    cache = {}

    def predict(text):
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if not text.strip():
            return {'label': -1, 'confidence': 0.0, 'prob_positive': None, 'text_hash': h, 'n_words': 0}
        if h not in cache:
            chunks = base.chunk_text(text, tokenizer)
            ids = torch.tensor([c[0] for c in chunks], dtype=torch.long).unsqueeze(0).to(base.DEVICE)
            masks = torch.tensor([c[1] for c in chunks], dtype=torch.long).unsqueeze(0).to(base.DEVICE)
            counts = torch.tensor([len(chunks)], dtype=torch.long)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
                logits = model(ids, masks, counts)
            probs = F.softmax(logits.float(), dim=-1)[0]
            cache[h] = float(probs[1])
        p1 = cache[h]
        label = int(p1 > 0.5)
        return {'label': label, 'confidence': round(max(p1, 1 - p1), 4), 'prob_positive': round(p1, 6),
                'text_hash': h, 'n_words': _words(text)}

    predict.cache = cache
    return predict


def load_model():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_NAME)
    model = base.HierarchicalLegalBERT(base.MODEL_NAME, base.GRU_HIDDEN, base.GRU_LAYERS, base.DROPOUT,
                                       unfreeze_last_n=base.UNFREEZE_LAYERS).to(base.DEVICE)
    checkpoint = base.load_checkpoint_checked(model, base.CHECKPOINT_PATH)
    model.eval()
    print(f"  model loaded | epoch {checkpoint.get('epoch')} | best F1 {checkpoint.get('best_f1')} | {base.DEVICE}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Stage-2 control experiments (see header comment).')
    ap.add_argument('--csv', default=base.CSV_PATH)
    ap.add_argument('--seg', default=base.RAW_SEG_JSON)
    ap.add_argument('--checkpoint', default=base.CHECKPOINT_PATH)
    ap.add_argument('--model-name', default=base.MODEL_NAME)
    ap.add_argument('--v2-results', default=base.RESULTS_JSON,
                    help='v2 results, used to check that the anchors reproduce')
    ap.add_argument('--out', default='./predex_stage2_controls.jsonl')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true', help='build variants and report sizes; no model')
    ap.add_argument('--fp16', action='store_true',
                    help='faster, but borderline predictions may differ from the fp32 v2 run')
    args = ap.parse_args()
    base.CSV_PATH, base.CHECKPOINT_PATH, base.MODEL_NAME = args.csv, args.checkpoint, args.model_name

    import pandas as pd
    df = pd.read_csv(args.csv)
    texts = df['Input'].astype(str).tolist()
    with open(args.seg) as f:
        blob = json.load(f)
    assert blob.get('schema') == 2, 'expected the v2 segmentation cache'
    cases = blob['cases']
    assert len(cases) == len(df), 'segmentation cache and CSV differ in length'
    for c in cases:
        c['_ct'] = CaseText(c['sentences'])
    by_index = {c['index']: c for c in cases}

    # Donors are chosen over ALL cases so --limit does not change them.
    donors = {g: pick_donors(cases, g, SEED) for g in TX_ROLES}
    fact_donors = {B: [c['index'] for c in cases if c['_ct'].words_in(c['_ct'].group_ids('FAC')) >= B]
                   for B in LM_BUDGETS}
    run_cases = [c for c in cases if any(s['role'] == 'FAC' for s in c['sentences'])]
    if args.limit:
        run_cases = run_cases[:args.limit]
    print(f"{len(run_cases)} cases with >=1 FAC sentence")

    # Hash check against the v2 run: the anchors must be byte-identical to what v2 predicted on.
    v2 = None
    if os.path.exists(args.v2_results):
        with open(args.v2_results) as f:
            v2 = {r['index']: r for r in json.load(f)}

    if args.dry_run:
        sizes, applicable, mismatch, order = defaultdict(list), Counter(), 0, None
        for c in run_cases:
            variants = build_control_variants(c, by_index, texts[c['index']], donors, fact_donors)
            order = order or list(variants)
            for k, t in variants.items():
                if t:
                    applicable[k] += 1
                    sizes[k].append(_words(t))
            if v2:
                for a in ANCHORS:
                    h = hashlib.md5(variants[a].encode('utf-8')).hexdigest()
                    mismatch += h != v2[c['index']]['predictions'][a]['text_hash']
        print(f"{'variant':26}{'cases':>7}{'median words':>14}")
        for k in order:
            s = sorted(sizes[k])
            print(f"{k:26}{applicable[k]:>7}{s[len(s) // 2]:>14}")
        approx = sum(min(base.MAX_CHUNKS, math.ceil(w * 1.35 / base.STRIDE)) for ws in sizes.values() for w in ws)
        print(f"\n~{approx:,} chunk forward passes before de-duplication")
        if v2:
            print(f"anchor text-hash mismatches vs v2: {mismatch} (must be 0)")
        return

    model, tokenizer = load_model()
    predict = make_predictor(model, tokenizer, args.fp16)

    done = set()
    if os.path.exists(args.out):
        # A run killed mid-write leaves a partial last line: drop it so the file stays valid JSONL.
        with open(args.out) as f:
            lines = f.readlines()
        good = []
        for line in lines:
            try:
                done.add(json.loads(line)['index'])
                good.append(line if line.endswith('\n') else line + '\n')
            except (json.JSONDecodeError, KeyError):
                print(f"  dropping an incomplete line from {args.out}")
        with open(args.out, 'w') as f:
            f.writelines(good)
        print(f"resuming: {len(done)} cases already in {args.out}")

    from tqdm import tqdm
    anchor_agree, anchor_total = Counter(), Counter()
    with open(args.out, 'a') as out:
        for c in tqdm([c for c in run_cases if c['index'] not in done], desc='controls'):
            variants = build_control_variants(c, by_index, texts[c['index']], donors, fact_donors)
            predict.cache.clear()                    # texts are only ever shared within one case
            preds = {k: predict(t) for k, t in variants.items()}
            if v2:
                for a in ANCHORS:
                    ref = v2[c['index']]['predictions'][a]
                    assert preds[a]['text_hash'] == ref['text_hash'], f"anchor {a} text differs from v2 (case {c['index']})"
                    anchor_total[a] += 1
                    anchor_agree[a] += preds[a]['label'] == ref['label']
            record = {
                'index': c['index'],
                'case_name': c['case_name'],
                'true_label': c['label'],
                'donors': {g: {str(k): v for k, v in donors[g].get(c['index'], {}).items()} for g in TX_ROLES},
                'predictions': preds,
            }
            out.write(json.dumps(record) + '\n')
            out.flush()

    if anchor_total:
        print('\nAnchor agreement with the v2 run (should be ~100% in fp32):')
        for a in ANCHORS:
            print(f"  {a:16} {100 * anchor_agree[a] / anchor_total[a]:.2f}% of {anchor_total[a]}")
    print(f"\nDone. Per-case predictions in {args.out}. Next: python stage2_analysis.py --controls {args.out}")


if __name__ == '__main__':
    main()
