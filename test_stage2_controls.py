"""Checks for the Stage-2 control variant builders. No GPU, checkpoint or data files needed.

    python -m pytest test_stage2_controls.py -q
"""
import random

import opennyai_ablation_fixed as base
import stage2_controls as sc


def make_case(index, label, roles_and_texts):
    sents = [{'role': r, 'text': t} for r, t in roles_and_texts]
    return {'index': index, 'case_name': f'case {index}', 'label': label,
            'sentences': sents, '_ct': sc.CaseText(sents)}


def words(n, tag):
    return ' '.join(f'{tag}{i}' for i in range(n))


def toy_cases():
    cases = []
    for i in range(12):
        cases.append(make_case(i, i % 2, [
            ('PREAMBLE', 'In the Supreme Court.'),
            ('FAC', f'Fact one of case {i} under Section 302 IPC. ' + words(40, f'f{i}x')),
            ('FAC', words(60, f'f{i}y')),
            ('FAC', words(60, f'f{i}z')),
            ('ARG_PETITIONER', words(80, f'a{i}')),
            ('STA', 'Section 302 and Sections 302/34 read with u/s 120B. ' + words(70, f's{i}')),
            ('ANALYSIS', words(90, f'n{i}a')),
            ('ANALYSIS', words(90, f'n{i}b')),
            ('RATIO', words(20, f'r{i}')),
        ]))
    return cases


def build(case, cases):
    by_index = {c['index']: c for c in cases}
    donors = {g: sc.pick_donors(cases, g, sc.SEED) for g in sc.TX_ROLES}
    fact_donors = {B: [c['index'] for c in cases] for B in sc.LM_BUDGETS}
    raw = ' '.join(s['text'] for s in case['sentences'])
    return sc.build_control_variants(case, by_index, raw, donors, fact_donors), donors


def test_anchors_match_v2_builder():
    cases = toy_cases()
    c = cases[3]
    v, _ = build(c, cases)
    raw = ' '.join(s['text'] for s in c['sentences'])
    ref = base.build_variants(c['sentences'], raw, c['index'])
    for k in sc.ANCHORS:
        assert v[k] == ref[k]


def test_take_budget_is_exact():
    ct = toy_cases()[0]['_ct']
    ids = ct.group_ids('ANALYSIS')
    for mode in ('random', 'first', 'last'):
        chosen, ov = sc.take_budget(ct, ids, 150, mode, random.Random(0))
        assert sc._words(ct.render(chosen, ov)) == 150
    assert sc.take_budget(ct, ids, 1000, 'first') == (None, None)


def test_first_and_last_take_opposite_ends():
    ct = toy_cases()[0]['_ct']
    ids = ct.group_ids('ANALYSIS')
    first = ct.render(*sc.take_budget(ct, ids, 60, 'first'))
    last = ct.render(*sc.take_budget(ct, ids, 60, 'last'))
    assert first.split()[0] == 'n0a0' and last.split()[-1] == 'n0b89'


def test_length_matched_adds_exactly_budget_words():
    cases = toy_cases()
    v, _ = build(cases[0], cases)
    fac_words = sc._words(v['v0_facts_only'])
    for B in sc.LM_BUDGETS:
        for g in ('analysis', 'analysis_first', 'analysis_last', 'dupfacts', 'otherfacts'):
            assert sc._words(v[f'lm{B}_{g}']) == fac_words + B, (B, g)
    assert v['lm150_sta'] == ''          # STA has only ~80 words: not eligible, not padded


def test_leave_one_out_removes_exactly_the_role():
    cases = toy_cases()
    v, _ = build(cases[0], cases)
    assert 'n0a0' not in v['loo_analysis'] and 'f0x0' in v['loo_analysis']
    assert v['loo_analysis'] != v['loo_rand_analysis']
    removed = sc._words(v['v_full']) - sc._words(v['loo_analysis'])
    assert sc._words(v['v_full']) - sc._words(v['loo_rand_analysis']) >= removed


def test_document_order_is_preserved():
    cases = toy_cases()
    v, _ = build(cases[0], cases)
    t = v['lm60_analysis']
    assert t.index('f0x0') < t.index('n0')


def test_donors_have_the_requested_label_and_are_not_self():
    cases = toy_cases()
    _, donors = build(cases[0], cases)
    labels = {c['index']: c['label'] for c in cases}
    for g, table in donors.items():
        for receiver, pair in table.items():
            for lab, d in pair.items():
                assert labels[d] == lab and d != receiver


def test_section_counterfactuals():
    s = 'convicted under Section 302 IPC and Sections 302/34 and u/s 120B'
    swapped = sc.section_swapper(random.Random(1))(s)
    assert '302' not in swapped and '120B' not in swapped and swapped.count('Section') == 2
    assert sc.remove_section_numbers(s) == 'convicted under Section IPC and Sections and u/s'


def test_builders_are_deterministic():
    cases = toy_cases()
    a, _ = build(cases[5], cases)
    b, _ = build(cases[5], cases)
    assert a == b
