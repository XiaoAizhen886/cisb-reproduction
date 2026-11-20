#!/usr/bin/env python3
"""Analyze the minimal Mistral replication with paired hierarchical bootstrap."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_fx41 import paired_auroc, bootstrap, bh  # noqa: E402

RESULTS = ROOT.parent / 'mistral_results'
OUT = ROOT.parent / 'results' / 'recheck' / 'mistral'
SEEDS = (42, 7, 123)
METRICS = (('canonical', 'act_l2'), ('canonical', 'act_cos'),
           ('residual', 'residual_l2'), ('residual', 'residual_cos'))
NAMES = {kind: {seed: f'fx42_mistral_{kind}_s{seed}' for seed in SEEDS}
         for kind in ('full', 'global', 'random2026')}


def metric(name, kind, detector):
    if kind == 'canonical':
        path = RESULTS / 'canonical' / f'{name}.json'
        return json.loads(path.read_text())['scores'][detector]
    path = RESULTS / 'residual' / 'sst2' / f'{name}_residual.json'
    return json.loads(path.read_text())['metrics'][detector]


def compare(control, kind, detector):
    rows, per_seed = {}, {}
    for seed in SEEDS:
        left = metric(NAMES['full'][seed], kind, detector)
        right = metric(NAMES[control][seed], kind, detector)
        n = min(len(left['test_trig']), len(left['test_clean']),
                len(right['test_trig']), len(right['test_clean']))
        l = {'test_trig': np.asarray(left['test_trig'][:n]), 'test_clean': np.asarray(left['test_clean'][:n])}
        r = {'test_trig': np.asarray(right['test_trig'][:n]), 'test_clean': np.asarray(right['test_clean'][:n])}
        lauroc = paired_auroc(l['test_trig'], l['test_clean'])
        rauroc = paired_auroc(r['test_trig'], r['test_clean'])
        per_seed[str(seed)] = {'left': lauroc, 'right': rauroc,
                               'delta': lauroc - rauroc, 'n_pairs': n}
        rows[seed] = {'left': l, 'right': r, 'n': n}
    comparison_id = f'mistral_coordinate:{control}:{kind}:{detector}'
    return {'id': comparison_id, 'control': control,
            'observed_delta': float(np.mean([row['delta'] for row in per_seed.values()])),
            'per_seed': per_seed, **bootstrap(rows, comparison_id)}


def main():
    manifest = json.loads((RESULTS / 'run_manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise RuntimeError('Mistral replication incomplete')
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons = [compare(control, kind, detector)
                   for control in ('global', 'random2026') for kind, detector in METRICS]
    for row, q in zip(comparisons, bh([row['p_floor'] for row in comparisons])):
        row['q_bh_floor'] = q
    behavior = {}
    for kind in NAMES:
        behavior[kind] = {}
        for seed in SEEDS:
            data = json.loads((RESULTS / 'behavior' / f'{NAMES[kind][seed]}.json').read_text())
            behavior[kind][str(seed)] = {'cda': data['cda'],
                                          'asr_non_target': data['asr_non_target'],
                                          'n_samples': data['n_samples']}
    result = {'schema_version': 1, 'analysis': 'mistral_minimal_unified_replication',
              'model': manifest['model'], 'n_boot': 2000, 'comparison_count': len(comparisons),
              'comparisons': comparisons, 'behavior': behavior}
    (OUT / 'mistral_controls.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Wrote {OUT / 'mistral_controls.json'}")


if __name__ == '__main__':
    main()
