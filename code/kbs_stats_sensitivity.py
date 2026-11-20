"""Statistics for the canonical detection package.

Design choices that address the P0 review points:
  * thresholds use the CLEAN CALIBRATION split only (95th percentile);
    the test split is evaluation-only;
  * the headline statistic is a PAIRED AUROC (same 200 clean/triggered
    sentence pairs), and comparisons between detectors/configs are made by a
    paired bootstrap of the AUROC *difference* -- never by "CI overlap";
  * config-level intervals use a hierarchical bootstrap (seeds, then pairs);
  * multiple comparisons are corrected with Benjamini-Hochberg inside
    pre-declared families.
"""
import argparse
import itertools
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import kbs_common as C

DETECTORS = ['ppl_ratio', 'ppl_delta', 'ppl_logdelta', 'onion_maxdrop',
             'onion_count', 'strip_entropy', 'strip_maxprob',
             'act_l2', 'act_cos', 'act_knn']
REF = 'act_l2'


def paired_auroc(trig, clean):
    """Rank-based AUC (Mann-Whitney), much faster than sklearn in bootstrap loops."""
    trig = np.asarray(trig, float)
    clean = np.asarray(clean, float)
    n1, n0 = len(trig), len(clean)
    allv = np.concatenate([trig, clean])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def concordance(trig, clean):
    d = np.asarray(trig) - np.asarray(clean)
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def calibrated(cal, clean, trig, q=0.95):
    tau = float(np.quantile(cal, q))
    return float((trig > tau).mean()), float((clean > tau).mean()), tau


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def p_variants(deltas):
    deltas = np.asarray(deltas)
    k_le = int(np.count_nonzero(deltas <= 0))
    k_ge = int(np.count_nonzero(deltas >= 0))
    n = len(deltas)
    floor = 2 * min(k_le / n, k_ge / n)
    return {
        'p': float(max(floor, 1.0 / n)),
        'p_two_sided_add_one': float(min(1.0, (2 * min(k_le, k_ge) + 1) / (n + 1))),
        'p_per_tail_add_one': float(min(1.0, 2 * min((k_le + 1) / (n + 1),
                                                       (k_ge + 1) / (n + 1)))),
        'tail_le': k_le,
        'tail_ge': k_ge,
    }


def boot_delta(a, b, n_boot=2000, rng=None):
    """Paired bootstrap of AUROC(a) - AUROC(b) over the same resampled pairs."""
    n = min(len(a['test_trig']), len(a['test_clean']),
            len(b['test_trig']), len(b['test_clean']))
    rng = rng or np.random.default_rng(2024)
    deltas, aucs_a, aucs_b = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        at, ac = a['test_trig'][idx], a['test_clean'][idx]
        bt, bc = b['test_trig'][idx], b['test_clean'][idx]
        aa = paired_auroc(at, ac)
        bb = paired_auroc(bt, bc)
        aucs_a.append(aa)
        aucs_b.append(bb)
        deltas.append(aa - bb)
    deltas = np.array(deltas)
    variants = p_variants(deltas)
    return dict(delta=float(paired_auroc(a['test_trig'][:n], a['test_clean'][:n]) -
                            paired_auroc(b['test_trig'][:n], b['test_clean'][:n])),
                ci=[float(np.quantile(deltas, 0.025)),
                    float(np.quantile(deltas, 0.975))],
                **variants,
                boot_mean=float(deltas.mean()),
                auc_a=[float(np.quantile(aucs_a, 0.025)),
                       float(np.quantile(aucs_a, 0.975))],
                auc_b=[float(np.quantile(aucs_b, 0.025)),
                       float(np.quantile(aucs_b, 0.975))])


def load_adapter(path):
    r = json.load(open(path))
    out = {'adapter': r['adapter'], 'config': r.get('config'),
           'dataset': r['dataset'], 'det': {}}
    for d in DETECTORS:
        if d not in r['scores']:
            continue
        s = r['scores'][d]
        out['det'][d] = {k: np.asarray(s[k], float)
                         for k in ('cal', 'test_clean', 'test_trig')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--dir', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f'refusing to overwrite sensitivity output: {args.out}')
    man = json.load(open(args.manifest))
    root = args.dir or os.path.join(C.OUT_BASE, 'canonical')
    rng = np.random.default_rng(2024)

    recs, missing = {}, []
    for group, names in man['groups'].items():
        for name in names:
            p = os.path.join(root, name + '.json')
            if not os.path.exists(p):
                missing.append(name)
                continue
            r = load_adapter(p)
            r['group'] = group
            r['seed'] = man['seeds'][group][names.index(name)]
            recs[name] = r
    print('loaded', len(recs), 'adapters; missing', len(missing))

    # ---------- per-adapter metrics ----------
    per_adapter = {}
    for name, r in recs.items():
        row = {}
        for d, s in r['det'].items():
            n = min(len(s['test_trig']), len(s['test_clean']))
            trig, clean = s['test_trig'][:n], s['test_clean'][:n]
            tpr, fpr, tau = calibrated(s['cal'], clean, trig)
            row[d] = dict(auroc=paired_auroc(trig, clean),
                          paired=concordance(trig, clean),
                          tpr=tpr, fpr=fpr, tau=tau,
                          n=int(n))
        per_adapter[name] = dict(group=r['group'], seed=r['seed'],
                                 metrics=row)

    # ---------- adapter-level view (same unit of analysis as PEFTGuard) ----------
    # For each detector two adapter-level scores are formed:
    #   self_auroc : how well the detector separates triggered from clean inputs
    #                inside that adapter (input-paired AUROC, already computed)
    #   mean_trig  : mean detector score on the triggered test inputs
    # Both are then used to classify adapters (backdoor vs benign) and the AUROC
    # over adapters is bootstrapped over adapters.
    adapter_level = {}
    for d in DETECTORS:
        for mode in ('self_auroc', 'mean_trig'):
            names = [n for n, r in recs.items() if d in r['det']]
            if len(names) < 6:
                continue
            scores, labels = [], []
            for n in names:
                r = recs[n]
                if mode == 'self_auroc':
                    sc = paired_auroc(r['det'][d]['test_trig'], r['det'][d]['test_clean'])
                else:
                    sc = float(np.mean(r['det'][d]['test_trig']))
                scores.append(sc)
                labels.append(0 if r['group'] == 'benign' else 1)
            scores = np.array(scores)
            labels = np.array(labels)
            if len(set(labels)) < 2:
                continue
            auc = float(roc_auc_score(labels, scores))
            boots = []
            for _ in range(args.n_boot):
                idx = rng.integers(0, len(labels), len(labels))
                if len(set(labels[idx])) < 2:
                    continue
                boots.append(roc_auc_score(labels[idx], scores[idx]))
            adapter_level[f'{d}::{mode}'] = dict(
                auroc=auc, ci=[float(np.quantile(boots, 0.025)),
                               float(np.quantile(boots, 0.975))],
                n=len(labels), n_pos=int(labels.sum()))

    # ---------- per-group means ----------
    groups = {}
    for g in man['groups']:
        rid = [n for n, r in recs.items() if r['group'] == g]
        if not rid:
            continue
        groups[g] = dict(n_adapters=len(rid), seeds=[recs[n]['seed'] for n in rid],
                         metrics={})
        for d in DETECTORS:
            vals = [per_adapter[n]['metrics'][d]['auroc'] for n in rid
                    if d in per_adapter[n]['metrics']]
            tprs = [per_adapter[n]['metrics'][d]['tpr'] for n in rid
                    if d in per_adapter[n]['metrics']]
            fprs = [per_adapter[n]['metrics'][d]['fpr'] for n in rid
                    if d in per_adapter[n]['metrics']]
            if not vals:
                continue
            groups[g]['metrics'][d] = dict(
                auroc_mean=float(np.mean(vals)),
                auroc_min=float(np.min(vals)), auroc_max=float(np.max(vals)),
                tpr_mean=float(np.mean(tprs)), fpr_mean=float(np.mean(fprs)),
                per_seed=dict(zip([str(recs[n]['seed']) for n in rid], vals)))

    # ---------- detector-vs-reference paired comparisons ----------
    detector_tests = []
    for name, r in recs.items():
        if REF not in r['det']:
            continue
        for d in DETECTORS:
            if d == REF or d not in r['det']:
                continue
            b = boot_delta(r['det'][d], r['det'][REF], args.n_boot, rng)
            b.update(adapter=name, group=r['group'], seed=r['seed'],
                     detector=d, ref=REF)
            detector_tests.append(b)
    for row in detector_tests:
        pass
    qs = bh([t['p'] for t in detector_tests]) if detector_tests else []
    for t, q in zip(detector_tests, qs):
        t['q_bh'] = float(q)

    # ---------- config-level paired comparisons per detector ----------
    config_tests = []
    backdoor_groups = [g for g in ('standard_ce', 'weight_constraint',
                                   'cisb_full', 'cisb_surgery') if g in groups]
    for d in DETECTORS:
        for g1, g2 in itertools.combinations(backdoor_groups, 2):
            n1 = [n for n in recs if recs[n]['group'] == g1 and d in recs[n]['det']]
            n2 = [n for n in recs if recs[n]['group'] == g2 and d in recs[n]['det']]
            if not n1 or not n2:
                continue
            seeds = sorted(set(recs[n]['seed'] for n in n1) &
                           set(recs[n]['seed'] for n in n2))
            if not seeds:
                continue
            by_seed1 = {recs[n]['seed']: n for n in n1}
            by_seed2 = {recs[n]['seed']: n for n in n2}
            diffs, auc1, auc2 = [], [], []
            for _ in range(args.n_boot):
                pick = rng.choice(seeds, len(seeds), replace=True)
                a1, a2 = [], []
                for sd in pick:
                    r1 = recs[by_seed1[sd]]['det'][d]
                    r2 = recs[by_seed2[sd]]['det'][d]
                    n = min(len(r1['test_trig']), len(r1['test_clean']),
                            len(r2['test_trig']), len(r2['test_clean']))
                    idx = rng.integers(0, n, n)
                    a1.append(paired_auroc(r1['test_trig'][idx], r1['test_clean'][idx]))
                    a2.append(paired_auroc(r2['test_trig'][idx], r2['test_clean'][idx]))
                auc1.append(np.mean(a1))
                auc2.append(np.mean(a2))
                diffs.append(np.mean(a1) - np.mean(a2))
            diffs = np.array(diffs)
            obs1 = np.mean([paired_auroc(recs[n]['det'][d]['test_trig'],
                                         recs[n]['det'][d]['test_clean']) for n in n1])
            obs2 = np.mean([paired_auroc(recs[n]['det'][d]['test_trig'],
                                         recs[n]['det'][d]['test_clean']) for n in n2])
            variants = p_variants(diffs)
            config_tests.append(dict(detector=d, g1=g1, g2=g2,
                                     seeds=seeds, n_seeds=len(seeds),
                                     delta=float(obs1 - obs2),
                                     ci=[float(np.quantile(diffs, 0.025)),
                                         float(np.quantile(diffs, 0.975))],
                                     **variants,
                                     auroc_g1=float(obs1), auroc_g2=float(obs2)))
    qs = bh([t['p'] for t in config_tests]) if config_tests else []
    for t, q in zip(config_tests, qs):
        t['q_bh'] = float(q)
    for records in (detector_tests, config_tests):
        for convention in ('p_two_sided_add_one', 'p_per_tail_add_one'):
            qs = bh([t[convention] for t in records]) if records else []
            for t, q in zip(records, qs):
                t['q_' + convention[2:]] = float(q)

    primary = [t for t in config_tests
               if t['g1'] == 'standard_ce' and t['g2'] == 'cisb_full'
               and t['detector'] in ('act_l2', 'act_cos')]
    families = {
        'detector_tests': {
            'id': 'detector_vs_act_l2',
            'members': [{'adapter': t['adapter'], 'detector': t['detector']}
                        for t in detector_tests],
        },
        'config_tests': {
            'id': 'config_pairwise',
            'members': [{'detector': t['detector'], 'g1': t['g1'], 'g2': t['g2']}
                        for t in config_tests],
        },
    }
    out = dict(manifest=args.manifest, n_adapters=len(recs),
               missing=missing, detectors=DETECTORS,
               per_adapter=per_adapter, groups=groups,
               adapter_level=adapter_level,
               detector_tests=detector_tests, config_tests=config_tests,
               primary_comparisons=primary,
               comparison_families=families,
               sensitivity_only=True,
               bootstrap=dict(n_boot=args.n_boot, paired_pairs=True,
                              threshold='95th percentile of clean calibration',
                              rng_seed=2024,
                              index_sharing='common seed and sentence-pair indices within each comparison; random stream continues across comparisons',
                              conventions=['floor', 'two_sided_add_one', 'per_tail_add_one']))
    C.json_dump(args.out, out)
    print('wrote', args.out)

    # ---------- compact console tables ----------
    print('\n== group means (auroc) ==')
    hdr = 'group'.ljust(18) + ''.join(d[:9].rjust(10) for d in DETECTORS)
    print(hdr)
    for g, v in groups.items():
        line = g.ljust(18)
        for d in DETECTORS:
            m = v['metrics'].get(d)
            line += ('%.3f' % m['auroc_mean']).rjust(10) if m else '   -'.rjust(10)
        print(line)
    print('\n== TPR@FPR5 (calibrated) ==')
    print(hdr)
    for g, v in groups.items():
        line = g.ljust(18)
        for d in DETECTORS:
            m = v['metrics'].get(d)
            line += ('%.3f' % m['tpr_mean']).rjust(10) if m else '   -'.rjust(10)
        print(line)


if __name__ == '__main__':
    main()
