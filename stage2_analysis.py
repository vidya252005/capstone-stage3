# ─────────────────────────────────────────────────────────────────────
# stage2_analysis.py
#
# Statistics, confound checks and figures for Stage 2. CPU only; no model.
#
#   python stage2_analysis.py                                   # v2 run only
#   python stage2_analysis.py --controls predex_stage2_controls.jsonl
#
# Inputs : predex_ablation_results_v2.json  (opennyai_ablation_fixed.py)
#          predex_opennyai_raw_v2.json      (OpenNyAI segmentation cache)
#          predex_stage2_controls.jsonl     (stage2_controls.py, optional)
# Outputs: stage2_outputs/  tables.md, *.csv, summary.json, fig_*.png,
#          stage3_role_lookup_v2.json (role sentences in the list format Stage 3 loads)
# ─────────────────────────────────────────────────────────────────────
import os
import json
import math
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings('ignore')

N_BOOT = 2000
RNG_SEED = 0

LAYER1 = ['v0_facts_only', 'v25', 'v50', 'v75', 'v100_full_spu']
LAYER2 = ['v_analysis_only', 'v_ratio_only', 'v_arg_only', 'v_pre_only', 'v_rlc_only',
          'v_sta_only', 'v_issue_only', 'v_rpc_only']
LAYER3 = ['v_counterfactual', 'v_facts_masked', 'v_sta_only_cf']
BASELINES = ['v_full', 'v_raw']
V2_BASE = {k: 'v0_facts_only' for k in LAYER1 + LAYER2 + LAYER3 + BASELINES}
V2_BASE['v_sta_only_cf'] = 'v_sta_only'

ROLE_OF_VARIANT = {
    'v_analysis_only': ['ANALYSIS'], 'v_ratio_only': ['RATIO'],
    'v_arg_only': ['ARG_PETITIONER', 'ARG_RESPONDENT'], 'v_pre_only': ['PRE_RELIED', 'PRE_NOT_RELIED'],
    'v_rlc_only': ['RLC'], 'v_sta_only': ['STA'], 'v_issue_only': ['ISSUE'], 'v_rpc_only': ['RPC'],
}
LABEL = {
    'v0_facts_only': 'Facts only', 'v25': 'Facts + 25%', 'v50': 'Facts + 50%', 'v75': 'Facts + 75%',
    'v100_full_spu': 'Facts + all pooled', 'v_analysis_only': 'Analysis', 'v_ratio_only': 'Ratio',
    'v_arg_only': 'Arguments', 'v_pre_only': 'Precedent', 'v_rlc_only': 'Lower-court ruling',
    'v_sta_only': 'Statute', 'v_issue_only': 'Issues', 'v_rpc_only': 'Present-court ruling',
    'v_counterfactual': 'Facts, sections -> 144', 'v_facts_masked': 'Facts, legal refs masked',
    'v_sta_only_cf': 'Facts+STA, sections -> 144', 'v_full': 'Segmented full text', 'v_raw': 'Raw text',
}
PROBE_GROUPS = {
    'Facts': ['FAC'], 'Analysis': ['ANALYSIS'], 'Ratio': ['RATIO'], 'Arguments': ['ARG_PETITIONER', 'ARG_RESPONDENT'],
    'Precedent': ['PRE_RELIED', 'PRE_NOT_RELIED'], 'Lower-court ruling': ['RLC'], 'Statute': ['STA'],
}

# Figure styling: one highlight hue, everything else recessive gray.
INK, INK_2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#c4c3bd'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────
def prob_positive(entry):
    """P(label=1). Newer runs store it; older ones store label + winning-class confidence."""
    if entry.get('prob_positive') is not None:
        return float(entry['prob_positive'])
    if entry['label'] == -1:
        return np.nan
    return entry['confidence'] if entry['label'] == 1 else 1.0 - entry['confidence']


def to_frames(records):
    """Long per-case arrays: labels[variant], probs[variant] aligned on case index."""
    idx = [r['index'] for r in records]
    keys = list(records[0]['predictions'])
    lab = pd.DataFrame({k: [r['predictions'][k]['label'] for r in records] for k in keys}, index=idx)
    prob = pd.DataFrame({k: [prob_positive(r['predictions'][k]) for r in records] for k in keys}, index=idx)
    hsh = pd.DataFrame({k: [r['predictions'][k]['text_hash'] for r in records] for k in keys}, index=idx)
    y = pd.Series([r['true_label'] for r in records], index=idx)
    return y, lab, prob, hsh


# ─────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def variant_table(y, lab, prob, hsh, keys, base_of, pos_rate):
    """One row per variant, scored on the cases where the variant AND its base exist."""
    rows = []
    for k in keys:
        b = base_of[k]
        ok = (lab[k] != -1) & (lab[b] != -1)
        yy, pk, pb, lk, lb = y[ok], prob[k][ok], prob[b][ok], lab[k][ok], lab[b][ok]
        n = int(ok.sum())
        if n < 10:                                   # variant does not apply to (enough) cases
            continue
        correct, correct_b = (lk == yy), (lb == yy)
        changed = hsh[k][ok] != hsh[b][ok]
        flips = (lk != lb)
        k_flip_chg = int(flips[changed].sum())
        n_chg = int(changed.sum())
        helped, hurt = int((~correct_b & correct).sum()), int((correct_b & ~correct).sum())
        mcnemar = binomtest(helped, helped + hurt).pvalue if helped + hurt else 1.0
        # Accuracy if the threshold is moved so the predicted-positive rate equals the base rate.
        # Separates "no signal" from "signal, but the 0.5 threshold is wrong for this input type".
        thr = np.quantile(pk, 1 - pos_rate)
        rebased = float(((pk > thr).astype(int) == yy).mean())
        lo, hi = wilson(int(correct.sum()), n)
        slo, shi = wilson(k_flip_chg, n_chg)
        rows.append({
            'variant': k, 'label': LABEL.get(k, k), 'base': b, 'n': n,
            'accuracy': 100 * correct.mean(), 'acc_lo': lo, 'acc_hi': hi,
            'balanced_acc': 100 * balanced_accuracy_score(yy, lk), 'macro_f1': 100 * f1_score(yy, lk, average='macro'),
            'auc': safe_auc(yy, pk), 'pred_pos_rate': 100 * lk.mean(), 'acc_rebased': 100 * rebased,
            'shift_all': 100 * flips.mean(), 'n_changed': n_chg,
            'shift_changed': 100 * k_flip_chg / n_chg if n_chg else np.nan, 'shift_lo': slo, 'shift_hi': shi,
            'flips_0to1': int(((lb == 0) & (lk == 1)).sum()), 'flips_1to0': int(((lb == 1) & (lk == 0)).sum()),
            'helped': helped, 'hurt': hurt, 'mcnemar_p': mcnemar,
            'mean_abs_dp': float(np.abs(pk - pb).mean()),
            'toward_truth': float(((pk - pb) * (2 * yy - 1)).mean()),
        })
    df = pd.DataFrame(rows)
    df['mcnemar_p_holm'] = holm(df['mcnemar_p'].values)
    return df


def paired_bootstrap(y, prob, lab, a, b, mask=None, n_boot=N_BOOT, seed=RNG_SEED):
    """95% CI for accuracy(a) - accuracy(b) and AUC(a) - AUC(b) on the same cases."""
    ok = (lab[a] != -1) & (lab[b] != -1)
    if mask is not None:
        ok &= mask
    yy = y[ok].values
    ca, cb = (lab[a][ok].values == yy), (lab[b][ok].values == yy)
    pa, pb = prob[a][ok].values, prob[b][ok].values
    rng = np.random.default_rng(seed)
    n = len(yy)
    dacc, dauc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        dacc.append(ca[i].mean() - cb[i].mean())
        if len(np.unique(yy[i])) == 2:
            dauc.append(roc_auc_score(yy[i], pa[i]) - roc_auc_score(yy[i], pb[i]))
    return {
        'a': a, 'b': b, 'n': n,
        'd_acc': 100 * (ca.mean() - cb.mean()), 'd_acc_lo': 100 * np.percentile(dacc, 2.5), 'd_acc_hi': 100 * np.percentile(dacc, 97.5),
        'd_auc': roc_auc_score(yy, pa) - roc_auc_score(yy, pb),
        'd_auc_lo': np.percentile(dauc, 2.5), 'd_auc_hi': np.percentile(dauc, 97.5),
    }


def bootstrap_mean(x, n_boot=N_BOOT, seed=RNG_SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


# ─────────────────────────────────────────────────────────────────────
# Confound checks on the v2 run
# ─────────────────────────────────────────────────────────────────────
def role_words(seg_cases):
    out = {}
    for c in seg_cases:
        w = {}
        for s in c['sentences']:
            w[s['role']] = w.get(s['role'], 0) + len(s['text'].split())
        out[c['index']] = w
    return out


def length_matched(y, lab, prob, words):
    """Per (case, role variant): words added, flip, movement of P toward the true label, accuracy gain."""
    rows = []
    for k, roles in ROLE_OF_VARIANT.items():
        for i in y.index:
            add = sum(words[i].get(r, 0) for r in roles)
            if add == 0 or lab.at[i, k] == -1:
                continue
            sign = 2 * y[i] - 1
            rows.append({
                'variant': k, 'case': i, 'added_words': add,
                'flip': int(lab.at[i, k] != lab.at[i, 'v0_facts_only']),
                'toward_truth': sign * (prob.at[i, k] - prob.at[i, 'v0_facts_only']),
                'acc_gain': int(lab.at[i, k] == y[i]) - int(lab.at[i, 'v0_facts_only'] == y[i]),
            })
    d = pd.DataFrame(rows)
    bins = [0, 100, 200, 400, 800, 10 ** 6]
    names = ['1-100', '101-200', '201-400', '401-800', '>800']
    d['words_bin'] = pd.cut(d['added_words'], bins, labels=names)
    return d


def length_matched_summary(d, min_n=30):
    g = d.groupby(['variant', 'words_bin'], observed=True)
    s = g.agg(n=('flip', 'size'), shift=('flip', 'mean'), toward_truth=('toward_truth', 'mean'),
              acc_gain=('acc_gain', 'mean')).reset_index()
    s = s[s['n'] >= min_n].copy()
    s['shift'] *= 100
    s['toward_truth'] *= 100
    s['acc_gain'] *= 100
    return s


def length_adjusted_effects(d):
    """Logistic flip ~ role + log(words added), and OLS toward_truth ~ role + log(words). Arguments = reference."""
    X = pd.get_dummies(d['variant']).astype(float).drop(columns=['v_arg_only'])
    X['log_words_added'] = np.log1p(d['added_words'])
    logit = LogisticRegression(C=1e6, max_iter=10000).fit(X, d['flip'])
    Xo = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xo, d['toward_truth'].values * 100, rcond=None)
    return pd.DataFrame({'term': X.columns, 'flip_logodds': logit.coef_[0], 'toward_truth_pp': beta[1:]})


def text_probe(seg_cases, n_words=150, seeds=(0, 1, 2)):
    """Model-agnostic: how much outcome information does each role's text carry on its own?

    TF-IDF + logistic regression, 5-fold CV AUC. Three views per role: all of its text, its
    first n_words and its last n_words (length-matched, only cases with >= n_words).
    """
    y_all = np.array([c['label'] for c in seg_cases])
    rows = []
    for name, roles in PROBE_GROUPS.items():
        texts = [' '.join(s['text'] for s in c['sentences'] if s['role'] in roles) for c in seg_cases]
        wc = np.array([len(t.split()) for t in texts])
        views = {'all text (cases with >=20 words)': (wc >= 20, lambda t: t),
                 f'first {n_words} words': (wc >= n_words, lambda t: ' '.join(t.split()[:n_words])),
                 f'last {n_words} words': (wc >= n_words, lambda t: ' '.join(t.split()[-n_words:]))}
        for view, (mask, fn) in views.items():
            n = int(mask.sum())
            if n < 150:
                rows.append({'role': name, 'view': view, 'n': n, 'cv_auc': np.nan, 'sd': np.nan})
                continue
            tx = [fn(t) for t, m in zip(texts, mask) if m]
            yy = y_all[mask]
            aucs = []
            for seed in seeds:
                pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=50000),
                                     LogisticRegression(C=2, max_iter=3000))
                p = cross_val_predict(pipe, tx, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                      method='predict_proba')[:, 1]
                aucs.append(roc_auc_score(yy, p))
            rows.append({'role': name, 'view': view, 'n': n, 'cv_auc': float(np.mean(aucs)), 'sd': float(np.std(aucs))})
    return pd.DataFrame(rows)


def detection_rule(lab, full_key, spurious_key, facts_key='v0_facts_only'):
    """Phase-2 rule: Pred(Full) = Pred(Spurious) != Pred(Facts)."""
    ok = (lab[full_key] != -1) & (lab[spurious_key] != -1) & (lab[facts_key] != -1)
    f, s, a = lab[full_key][ok], lab[spurious_key][ok], lab[facts_key][ok]
    return {'full': full_key, 'spurious': spurious_key, 'n': int(ok.sum()),
            'rule_true_pct': 100 * float(((f == s) & (s != a)).mean()),
            'full_agrees_spurious_pct': 100 * float((f == s).mean()),
            'full_agrees_facts_pct': 100 * float((f == a).mean())}


# ─────────────────────────────────────────────────────────────────────
# Control experiments (stage2_controls.py output)
# ─────────────────────────────────────────────────────────────────────
def load_controls(path):
    """One record per case; a later line for the same case (re-run) replaces the earlier one."""
    by_case = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f'  skipping an incomplete line in {path}')
                continue
            by_case[record['index']] = record
    return [by_case[i] for i in sorted(by_case)]


def controls_analysis(records, out_dir, md):
    import stage2_controls as sc            # for the variant -> base mapping (no model is loaded)
    y, lab, prob, hsh = to_frames(records)
    keys = [k for k in lab.columns if k in sc.SHIFT_BASE]
    tab = variant_table(y, lab, prob, hsh, keys, sc.SHIFT_BASE, y.mean())
    tab.to_csv(os.path.join(out_dir, 'controls_all_variants.csv'), index=False)
    res = {}

    # A: no-facts inputs and the Phase-2 detection rule
    a_keys = ['v0_facts_only', 'nofac_all', 'nofac_analysis', 'nofac_spurious', 'nofac_nonanalysis', 'v_raw']
    md.append('## Controls A: inputs with no facts at all\n')
    md.append(md_table(tab[tab.variant.isin(a_keys)], ['variant', 'n', 'accuracy', 'balanced_acc', 'auc', 'pred_pos_rate'],
                       'accuracy/AUC of each input on its own'))
    rules = pd.DataFrame([detection_rule(lab, 'v_raw', k) for k in a_keys[1:5]])
    rules.to_csv(os.path.join(out_dir, 'controls_A_detection_rule.csv'), index=False)
    md.append(md_table(rules, list(rules.columns), 'Phase-2 rule Pred(Full)=Pred(X)!=Pred(Facts)'))
    res['A'] = rules.to_dict('records')

    # B: leave-one-role-out vs matched random deletion
    rows = []
    for g in sc.LOO_GROUPS + ['POSTDEC']:
        k, r = f'loo_{g.lower()}', f'loo_rand_{g.lower()}'
        ok = (lab[k] != -1) & (lab[r] != -1)
        if ok.sum() < 30:
            continue
        yh = lab['v_full'][ok]
        p_full_pred = np.where(yh == 1, prob['v_full'][ok], 1 - prob['v_full'][ok])
        comp_role = p_full_pred - np.where(yh == 1, prob[k][ok], 1 - prob[k][ok])
        comp_rand = p_full_pred - np.where(yh == 1, prob[r][ok], 1 - prob[r][ok])
        diff = bootstrap_mean(comp_role - comp_rand)
        bs_role = paired_bootstrap(y, prob, lab, k, 'v_full', mask=ok)
        bs_rand = paired_bootstrap(y, prob, lab, r, 'v_full', mask=ok)
        rows.append({'role_removed': g, 'n': int(ok.sum()),
                     'd_acc_role': bs_role['d_acc'], 'd_acc_role_ci': f"[{bs_role['d_acc_lo']:.1f}, {bs_role['d_acc_hi']:.1f}]",
                     'd_acc_random': bs_rand['d_acc'], 'd_auc_role': bs_role['d_auc'], 'd_auc_random': bs_rand['d_auc'],
                     'flip_role_pct': 100 * float((lab[k][ok] != yh).mean()),
                     'flip_random_pct': 100 * float((lab[r][ok] != yh).mean()),
                     'comprehensiveness_role': float(comp_role.mean()), 'comprehensiveness_random': float(comp_rand.mean()),
                     'excess_over_random': diff[0], 'excess_ci': f'[{diff[1]:.3f}, {diff[2]:.3f}]'})
    loo = pd.DataFrame(rows)
    loo.to_csv(os.path.join(out_dir, 'controls_B_leave_one_out.csv'), index=False)
    md.append('## Controls B: leave one role out of the full document (vs. random deletion of the same length)\n')
    md.append(md_table(loo, list(loo.columns), 'comprehensiveness = drop in P(predicted class) when the text is removed'))
    res['B'] = loo.to_dict('records')

    # C: length-matched add-in
    rows = []
    for B in sc.LM_BUDGETS:
        for g in [x.lower() for x in sc.LM_ROLES] + ['analysis_first', 'analysis_last', 'dupfacts', 'otherfacts']:
            k = f'lm{B}_{g}'
            ok = (lab[k] != -1) & (lab['v0_facts_only'] != -1)
            if ok.sum() < 50:
                continue
            tt = (prob[k][ok] - prob['v0_facts_only'][ok]) * (2 * y[ok] - 1)
            m, lo, hi = bootstrap_mean(tt * 100)
            bs = paired_bootstrap(y, prob, lab, k, 'v0_facts_only', mask=ok)
            rows.append({'budget_words': B, 'added': g, 'n': int(ok.sum()),
                         'shift_pct': 100 * float((lab[k][ok] != lab['v0_facts_only'][ok]).mean()),
                         'toward_truth_pp': m, 'toward_truth_ci': f'[{lo:.1f}, {hi:.1f}]',
                         'd_acc': bs['d_acc'], 'd_acc_ci': f"[{bs['d_acc_lo']:.1f}, {bs['d_acc_hi']:.1f}]", 'd_auc': bs['d_auc']})
    lm = pd.DataFrame(rows)
    lm.to_csv(os.path.join(out_dir, 'controls_C_length_matched.csv'), index=False)
    md.append('## Controls C: facts + exactly B words of one role\n')
    md.append(md_table(lm, list(lm.columns), 'dupfacts / otherfacts = "more text, no new information" controls'))
    res['C'] = lm.to_dict('records')

    # D: transplant — does the prediction follow the donor's outcome?
    rows = []
    for g in sc.TX_ROLES:
        k0, k1 = f'tx_{g.lower()}_d0', f'tx_{g.lower()}_d1'
        ok = (lab[k0] != -1) & (lab[k1] != -1)
        if ok.sum() < 30:
            continue
        dp = prob[k1][ok] - prob[k0][ok]
        m, lo, hi = bootstrap_mean(dp)
        rows.append({'role_transplanted': g, 'n': int(ok.sum()),
                     'P1_donor1_minus_donor0': m, 'ci': f'[{lo:.3f}, {hi:.3f}]',
                     'follows_donor_pct': 100 * float(((lab[k1][ok] == 1) & (lab[k0][ok] == 0)).mean()),
                     'against_donor_pct': 100 * float(((lab[k1][ok] == 0) & (lab[k0][ok] == 1)).mean()),
                     'same_pred_pct': 100 * float((lab[k1][ok] == lab[k0][ok]).mean())})
    tx = pd.DataFrame(rows)
    tx.to_csv(os.path.join(out_dir, 'controls_D_transplant.csv'), index=False)
    md.append('## Controls D: transplant role text from a donor case with a known outcome\n')
    md.append(md_table(tx, list(tx.columns), 'same receiver facts, donor outcome 1 vs donor outcome 0'))
    res['D'] = tx.to_dict('records')

    # E + F: noise floor, section counterfactuals, dose without ANALYSIS
    e_keys = ['nf_drop1_fac', 'cf_fac_random_s0', 'cf_fac_random_s1', 'cf_fac_random_s2', 'cf_fac_removed',
              'nf_drop1_full', 'cf_full_fixed', 'cf_full_random', 'cf_full_removed', 'cf_full_masked']
    md.append('## Controls E: noise floor and section-number counterfactuals\n')
    md.append(md_table(tab[tab.variant.isin(e_keys)],
                       ['variant', 'base', 'n', 'accuracy', 'n_changed', 'shift_changed', 'shift_lo', 'shift_hi', 'mean_abs_dp'],
                       'a counterfactual only means something if it shifts more than nf_drop1_* on the same base'))
    f_keys = ['v0_facts_only', 'dnoA_25', 'dnoA_50', 'dnoA_75', 'dnoA_100']
    md.append('## Controls F: dose-response with ANALYSIS removed from the pool\n')
    md.append(md_table(tab[tab.variant.isin(f_keys)], ['variant', 'n', 'accuracy', 'auc', 'shift_all', 'toward_truth'], ''))
    res['E'] = tab[tab.variant.isin(e_keys)].to_dict('records')
    res['F'] = tab[tab.variant.isin(f_keys)].to_dict('records')

    plot_controls(loo, lm, tx, out_dir)
    return res


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────
def plot_overview(tab, out, maj):
    t = tab.set_index('variant')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={'width_ratios': [1, 1.15, 1.25]})

    ax = axes[0]
    lv = [0, 25, 50, 75, 100]
    acc = [t.at[k, 'accuracy'] for k in LAYER1]
    auc = [100 * t.at[k, 'auc'] for k in LAYER1]
    ax.fill_between(lv, [t.at[k, 'acc_lo'] for k in LAYER1], [t.at[k, 'acc_hi'] for k in LAYER1], color=BLUE, alpha=0.15, lw=0)
    ax.plot(lv, acc, color=BLUE, lw=2, marker='o', ms=6, label='Accuracy (95% CI)')
    ax.plot(lv, auc, color=ORANGE, lw=2, marker='s', ms=6, label='ROC-AUC x 100')
    ax.axhline(100 * maj, color=MUTED, lw=1)
    ax.text(100, 100 * maj + 0.6, f'majority class {100 * maj:.1f}%', color=INK_2, ha='right', va='bottom', fontsize=8)
    ax.set_xticks(lv)
    ax.set_xlabel('Share of pooled non-fact sentences added to facts (%)')
    ax.set_ylabel('%')
    ax.set_ylim(40, 90)
    ax.set_title('Layer 1: dose-response\n(~70% of the pooled words are ANALYSIS)', fontsize=11)
    ax.legend(frameon=False, loc='lower right')

    ax = axes[1]
    keys = sorted(LAYER2, key=lambda k: t.at[k, 'shift_all'])
    vals = [t.at[k, 'shift_all'] for k in keys]
    gains = [t.at[k, 'accuracy'] - t.at['v0_facts_only', 'accuracy'] for k in keys]
    colors = [BLUE if k == 'v_analysis_only' else GRAY for k in keys]
    bars = ax.barh([LABEL[k] for k in keys], vals, color=colors, height=0.62)
    for bar, v, g, k in zip(bars, vals, gains, keys):
        sig = '' if t.at[k, 'mcnemar_p_holm'] < 0.05 else ' (n.s.)'
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%  |  acc {g:+.1f} pp{sig}',
                va='center', fontsize=8.5, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.75)
    ax.set_xlabel('Prediction shift vs facts-only (% of all cases)')
    ax.set_title('Layer 2: facts + one role\n(shift  |  accuracy change, Holm-corrected McNemar)', fontsize=11)
    ax.grid(axis='y', visible=False)

    ax = axes[2]
    comp = ['v0_facts_only', 'v_sta_only', 'v_analysis_only', 'v100_full_spu', 'v_full', 'v_raw']
    names = ['Facts\nonly', 'Facts +\nStatute', 'Facts +\nAnalysis', 'Facts +\nall pooled', 'Segmented\nfull text', 'Raw\ntext']
    a = [t.at[k, 'accuracy'] for k in comp]
    err = [[t.at[k, 'accuracy'] - t.at[k, 'acc_lo'] for k in comp], [t.at[k, 'acc_hi'] - t.at[k, 'accuracy'] for k in comp]]
    colors = [GRAY, GRAY, BLUE, GRAY, GRAY, GRAY]
    bars = ax.bar(names, a, color=colors, width=0.62, yerr=err, error_kw={'ecolor': INK_2, 'elinewidth': 1, 'capsize': 3})
    for bar, k in zip(bars, comp):
        ax.text(bar.get_x() + bar.get_width() / 2, t.at[k, 'acc_hi'] + 1, f"{t.at[k, 'accuracy']:.1f}%\nAUC {t.at[k, 'auc']:.2f}",
                ha='center', fontsize=8, color=INK_2)
    ax.axhline(100 * maj, color=MUTED, lw=1)
    ax.text(len(comp) - 0.55, 100 * maj + 0.8, f'majority class {100 * maj:.1f}%', color=INK_2, fontsize=8, ha='right', va='bottom')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy of key inputs (95% CI)', fontsize=11)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_sensitivity_vs_information(tab, lm_obs, out):
    t = tab.set_index('variant')
    med = lm_obs.groupby('variant')['added_words'].median()
    has = lm_obs.groupby('variant').agg(gain=('acc_gain', 'mean'), toward=('toward_truth', 'mean'))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # label positions in data units, with leader lines, so the low-information cluster stays readable
    place = {'v_analysis_only': (31.0, 13.4, 'left'), 'v_ratio_only': (19.5, 5.3, 'left'),
             'v_arg_only': (23.0, 3.6, 'left'), 'v_pre_only': (19.8, 2.4, 'left'), 'v_issue_only': (10.2, 2.6, 'left'),
             'v_rlc_only': (14.5, 6.6, 'center'), 'v_rpc_only': (8.6, -1.3, 'left'), 'v_sta_only': (22.5, -1.3, 'left')}
    for k in LAYER2:
        x, yv = t.at[k, 'shift_changed'], 100 * has.at[k, 'toward']
        hi = k in ('v_analysis_only', 'v_ratio_only')
        ax.scatter(x, yv, s=30 + med[k] / 6, color=BLUE if hi else GRAY, edgecolor='white', linewidth=1.5, zorder=3)
        tx, ty, ha = place[k]
        ax.annotate(f"{LABEL[k]} ({int(med[k])} w)", (x, yv), xytext=(tx, ty), textcoords='data', fontsize=9,
                    ha=ha, va='center', color=INK if hi else INK_2,
                    arrowprops={'arrowstyle': '-', 'color': MUTED, 'lw': 0.6, 'shrinkA': 2, 'shrinkB': 5})
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-2, 15.5)
    ax.set_xlim(8, 42)
    ax.set_xlabel('Sensitivity: % of predictions that flip when the role is added\n(cases where the role exists)')
    ax.set_ylabel('Information: mean move of P(true outcome), pp')
    ax.set_title('Every role moves the model; only court reasoning moves it the right way\n'
                 '(bubble size = median words added)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_length_matched(summ, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    order = list(summ['words_bin'].cat.categories)
    for ax, metric, ylabel in [(axes[0], 'shift', 'Prediction shift vs facts-only (%)'),
                               (axes[1], 'toward_truth', 'Mean move of P(true outcome), pp')]:
        for k in LAYER2:
            s = summ[summ.variant == k]
            if s.empty:
                continue
            x = [order.index(b) for b in s['words_bin']]
            hi = k == 'v_analysis_only'
            ax.plot(x, s[metric], color=BLUE if hi else GRAY, lw=2.2 if hi else 1.3, marker='o', ms=6 if hi else 4, zorder=3 if hi else 2)
            ax.text(x[-1] + 0.08, s[metric].iloc[-1], LABEL[k], fontsize=8, color=INK if hi else INK_2, va='center')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel('Words added to the facts')
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.2, len(order) - 0.2)
    axes[0].set_title('Every role flips some predictions', fontsize=11)
    axes[1].set_title('At equal length, court reasoning (Analysis, Ratio) moves P toward the truth', fontsize=11)
    axes[1].axhline(0, color=MUTED, lw=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_probe(probe, out, n_words):
    p = probe[probe.view != 'all text (cases with >=20 words)'].dropna(subset=['cv_auc'])
    roles = [r for r in PROBE_GROUPS if r in set(p.role)]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    yy = np.arange(len(roles))
    for view, color, dy in [(f'first {n_words} words', ORANGE, -0.12), (f'last {n_words} words', BLUE, 0.12)]:
        s = p[p.view == view].set_index('role')
        xs = [s.at[r, 'cv_auc'] if r in s.index else np.nan for r in roles]
        ax.scatter(xs, yy + dy, color=color, s=60, edgecolor='white', linewidth=1.5, zorder=3, label=view)
    ax.axvline(0.5, color=MUTED, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(roles)
    ax.invert_yaxis()
    ax.set_xlabel('Cross-validated ROC-AUC of a TF-IDF classifier on that text alone')
    ax.set_title(f'Outcome information per role, no InLegalBERT involved\n({n_words} words per case, cases with enough text)', fontsize=11)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_controls(loo, lm, tx, out_dir):
    if len(loo):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        yy = np.arange(len(loo))
        ax.barh(yy - 0.18, loo['d_acc_role'], height=0.34, color=BLUE, label='remove this role')
        ax.barh(yy + 0.18, loo['d_acc_random'], height=0.34, color=GRAY, label='remove same no. of random words')
        ax.set_yticks(yy)
        ax.set_yticklabels(loo['role_removed'])
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel('Accuracy change vs full document (pp)')
        ax.set_title('Necessity: what happens when a role is removed', fontsize=11)
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.grid(axis='y', visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig_controls_leave_one_out.png'))
        plt.close(fig)
    if len(tx):
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        lo = [float(c.strip('[]').split(',')[0]) for c in tx['ci']]
        hi = [float(c.strip('[]').split(',')[1]) for c in tx['ci']]
        m = tx['P1_donor1_minus_donor0'].values
        colors = [BLUE if r == 'ANALYSIS' else GRAY for r in tx['role_transplanted']]
        ax.barh(tx['role_transplanted'], m, color=colors, height=0.6,
                xerr=[m - np.array(lo), np.array(hi) - m], error_kw={'ecolor': INK_2, 'capsize': 3})
        ax.axvline(0, color=MUTED, lw=1)
        ax.invert_yaxis()
        ax.set_xlabel('P(accepted | donor accepted) - P(accepted | donor rejected)')
        ax.set_title("Transplant: same facts, another case's text", fontsize=11)
        ax.grid(axis='y', visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'fig_controls_transplant.png'))
        plt.close(fig)
    if len(lm):
        for B, s in lm.groupby('budget_words'):
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = [BLUE if a.startswith('analysis') else (MUTED if a in ('dupfacts', 'otherfacts') else GRAY) for a in s['added']]
            ax.barh(s['added'], s['toward_truth_pp'], color=colors, height=0.6)
            ax.axvline(0, color=MUTED, lw=1)
            ax.invert_yaxis()
            ax.set_xlabel('Mean move of P(true outcome), pp')
            ax.set_title(f'Facts + exactly {B} words of one role', fontsize=11)
            ax.grid(axis='y', visible=False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'fig_controls_length_matched_{B}.png'))
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Markdown helpers
# ─────────────────────────────────────────────────────────────────────
def md_table(df, cols, caption):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return '-'
            return f'{v:.2e}' if (0 < abs(v) < 1e-3) else f'{v:.3f}' if abs(v) < 1 else f'{v:.1f}'
        return str(v)
    lines = [f'_{caption}_\n' if caption else '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append('| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |')
    return '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Stage-2 statistics and figures (CPU).')
    ap.add_argument('--results', default='./predex_ablation_results_v2.json')
    ap.add_argument('--seg', default='./predex_opennyai_raw_v2.json')
    ap.add_argument('--controls', default=None, help='JSONL from stage2_controls.py')
    ap.add_argument('--out', default='./stage2_outputs')
    ap.add_argument('--probe-words', type=int, default=150)
    ap.add_argument('--skip-probe', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.results) as f:
        results = json.load(f)
    with open(args.seg) as f:
        seg = json.load(f)
    assert seg.get('schema') == 2, 'expected the v2 segmentation cache'
    seg_cases = seg['cases']

    valid = [r for r in results if r['predictions']['v0_facts_only']['label'] != -1]
    y, lab, prob, hsh = to_frames(valid)
    maj = max(y.mean(), 1 - y.mean())
    md = ['# Stage 2 — tables (generated by stage2_analysis.py)\n',
          f'Cases: {len(results)} in PredEx test, {len(valid)} with at least one FAC sentence (all metrics use these). '
          f'Label 1 (accepted) = {100 * y.mean():.1f}%, so always predicting the majority class scores {100 * maj:.1f}%.\n']

    keys = LAYER1 + LAYER2 + LAYER3 + BASELINES
    tab = variant_table(y, lab, prob, hsh, keys, V2_BASE, y.mean())
    tab.to_csv(os.path.join(args.out, 'v2_variant_metrics.csv'), index=False)
    cols = ['label', 'n', 'accuracy', 'acc_lo', 'acc_hi', 'balanced_acc', 'macro_f1', 'auc', 'pred_pos_rate',
            'acc_rebased', 'shift_all', 'n_changed', 'shift_changed', 'helped', 'hurt', 'mcnemar_p_holm', 'mean_abs_dp']
    md.append('## 1. All v2 variants\n')
    md.append(md_table(tab, cols, 'acc_rebased = accuracy with the threshold moved so the predicted-positive rate matches '
                                  'the base rate; mcnemar_p_holm = exact McNemar vs base, Holm-corrected over all variants'))

    contrasts = [('v_analysis_only', 'v0_facts_only'), ('v_sta_only', 'v0_facts_only'), ('v_ratio_only', 'v0_facts_only'),
                 ('v_arg_only', 'v0_facts_only'), ('v_raw', 'v0_facts_only'), ('v_raw', 'v_analysis_only'),
                 ('v100_full_spu', 'v_analysis_only'), ('v_raw', 'v_full')]
    con = pd.DataFrame([paired_bootstrap(y, prob, lab, a, b) for a, b in contrasts])
    con.to_csv(os.path.join(args.out, 'v2_paired_contrasts.csv'), index=False)
    md.append('## 2. Paired bootstrap contrasts (2,000 case resamples)\n')
    md.append(md_table(con, list(con.columns), 'difference a - b on identical cases, 95% percentile CI'))

    words = role_words(seg_cases)
    lm_obs = length_matched(y, lab, prob, words)
    lm_obs.to_csv(os.path.join(args.out, 'v2_length_matched_cases.csv'), index=False)
    lm_sum = length_matched_summary(lm_obs)
    lm_sum.to_csv(os.path.join(args.out, 'v2_length_matched_summary.csv'), index=False)
    adj = length_adjusted_effects(lm_obs)
    adj.to_csv(os.path.join(args.out, 'v2_length_adjusted_effects.csv'), index=False)
    per_role = lm_obs.groupby('variant').agg(n=('flip', 'size'), median_words=('added_words', 'median'),
                                             shift=('flip', 'mean'), toward_truth=('toward_truth', 'mean'),
                                             acc_gain=('acc_gain', 'mean')).reset_index()
    for c in ('shift', 'toward_truth', 'acc_gain'):
        per_role[c] *= 100
    md.append('## 3. Sensitivity vs information (cases where the role exists)\n')
    md.append(md_table(per_role.sort_values('toward_truth', ascending=False), list(per_role.columns),
                       'shift = % predictions flipped; toward_truth = mean change in P(true outcome), pp; acc_gain = pp'))
    md.append(md_table(lm_sum, list(lm_sum.columns), 'same, binned by words added (bins with n >= 30)'))
    md.append(md_table(adj, list(adj.columns), 'length-adjusted role effects, reference = Arguments'))

    rules = pd.DataFrame([detection_rule(lab, 'v_raw', 'v100_full_spu'), detection_rule(lab, 'v_raw', 'v_analysis_only'),
                          detection_rule(lab, 'v_raw', 'v_sta_only')])
    rules.to_csv(os.path.join(args.out, 'v2_detection_rule.csv'), index=False)
    md.append('## 4. Phase-2 detection rule on v2 variants\n')
    md.append(md_table(rules, list(rules.columns), 'Pred(Full) = Pred(X) != Pred(Facts); X here still contains the facts'))

    summary = {'n_cases': len(results), 'n_scored': len(valid), 'majority_rate': maj,
               'variants': tab.set_index('variant').to_dict('index'), 'contrasts': con.to_dict('records'),
               'detection_rule': rules.to_dict('records'), 'per_role': per_role.to_dict('records')}

    if not args.skip_probe:
        print('Running the TF-IDF role probe (a few minutes)...')
        probe = text_probe(seg_cases, args.probe_words)
        probe.to_csv(os.path.join(args.out, 'v2_text_probe.csv'), index=False)
        md.append('## 5. Model-agnostic text probe (TF-IDF + logistic regression, 5-fold CV, 3 seeds)\n')
        md.append(md_table(probe, list(probe.columns), 'how much outcome information each role carries on its own'))
        summary['probe'] = probe.to_dict('records')
        plot_probe(probe, os.path.join(args.out, 'fig_text_probe.png'), args.probe_words)

    plot_overview(tab, os.path.join(args.out, 'fig_v2_overview.png'), maj)
    plot_sensitivity_vs_information(tab, lm_obs, os.path.join(args.out, 'fig_sensitivity_vs_information.png'))
    plot_length_matched(lm_sum, os.path.join(args.out, 'fig_length_matched.png'))

    if args.controls and os.path.exists(args.controls):
        print(f'Analysing controls from {args.controls}')
        summary['controls'] = controls_analysis(load_controls(args.controls), args.out, md)
    else:
        md.append('## Controls\n\nNot run yet: `python stage2_controls.py`, then re-run this script with `--controls`.\n')

    # Stage 3 loads role sentences as a LIST of {index, case_name, sentences}.
    with open(os.path.join(args.out, 'stage3_role_lookup_v2.json'), 'w') as f:
        json.dump(seg_cases, f)

    with open(os.path.join(args.out, 'tables.md'), 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(f'Wrote tables, figures and summary to {args.out}/')


if __name__ == '__main__':
    main()
