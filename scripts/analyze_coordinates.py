import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent / 'results' / 'coordinates'
OUT = Path(__file__).resolve().parent.parent / 'results' / 'recheck' / 'coordinate_summary.json'
SEEDS = (42, 7, 123)
CIRCUITS = ('random2024', 'random2025', 'random2026')
METRICS = ('act_l2', 'act_cos')
N_BOOT = 2000


def auroc(scores, indices=None):
    clean = np.asarray(scores['test_clean'], dtype=float)
    trig = np.asarray(scores['test_trig'], dtype=float)
    if indices is not None:
        clean = clean[indices]
        trig = trig[indices]
    return float(roc_auc_score([0] * len(clean) + [1] * len(trig), np.concatenate([clean, trig])))


def bh(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = len(p) - rank + 1
        running = min(running, p[idx] * len(p) / original_rank)
        adjusted[idx] = running
    return adjusted


def main():
    rows = []
    for circuit_index, circuit in enumerate(CIRCUITS):
        folder = ROOT / circuit
        for metric_index, metric in enumerate(METRICS):
            per_seed = {}
            bootstrap_deltas = []
            for seed in SEEDS:
                suffix = f'_s{seed}_r16'
                cisb_path = next(folder.glob(f'cisb_v4b_aligned_*{suffix}_lam020_wcon.json'))
                standard_path = next(folder.glob(f'cisb_v3_standard_*_s{seed}_r16_nocontrol.json'))
                cisb = json.loads(cisb_path.read_text())['scores'][metric]
                standard = json.loads(standard_path.read_text())['scores'][metric]
                per_seed[seed] = {
                    'cisb': auroc(cisb),
                    'standard': auroc(standard),
                }
                per_seed[seed]['delta'] = per_seed[seed]['cisb'] - per_seed[seed]['standard']
                rng = np.random.default_rng(20261119 + 1000 * circuit_index + metric_index)
                deltas = []
                for _ in range(N_BOOT):
                    picked = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                    boot_values = []
                    for picked_seed in picked:
                        c = json.loads(next(folder.glob(f'cisb_v4b_aligned_*_s{picked_seed}_r16_lam020_wcon.json')).read_text())['scores'][metric]
                        s = json.loads(next(folder.glob(f'cisb_v3_standard_*_s{picked_seed}_r16_nocontrol.json')).read_text())['scores'][metric]
                        indices = rng.integers(0, len(c['test_clean']), size=len(c['test_clean']))
                        boot_values.append(auroc(c, indices) - auroc(s, indices))
                    deltas.append(float(np.mean(boot_values)))
                bootstrap_deltas = np.asarray(deltas)
            deltas = np.asarray([per_seed[s]['delta'] for s in SEEDS])
            boot = bootstrap_deltas
            kle = int(np.count_nonzero(boot <= 0))
            kge = int(np.count_nonzero(boot >= 0))
            p_floor = min(1.0, max(2 * min(kle / N_BOOT, kge / N_BOOT), 1 / N_BOOT))
            rows.append({
                'circuit': circuit,
                'metric': metric,
                'per_seed': per_seed,
                'cisb_mean': float(np.mean([per_seed[s]['cisb'] for s in SEEDS])),
                'standard_mean': float(np.mean([per_seed[s]['standard'] for s in SEEDS])),
                'delta_mean': float(deltas.mean()),
                'ci95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                'p_floor': p_floor,
                'tail_counts': {'kle': kle, 'kge': kge, 'B': N_BOOT},
            })
    q_values = bh([row['p_floor'] for row in rows])
    for row, q in zip(rows, q_values):
        row['q_bh_floor'] = float(q)
    benign = {}
    for circuit in CIRCUITS:
        values = {metric: [] for metric in METRICS}
        for path in (ROOT / circuit).glob('cisb_v3_benign_*.json'):
            payload = json.loads(path.read_text())
            for metric in METRICS:
                values[metric].append(auroc(payload['scores'][metric]))
        benign[circuit] = {
            metric: {
                'n': len(values[metric]),
                'mean': float(np.mean(values[metric])),
                'min': float(np.min(values[metric])),
                'max': float(np.max(values[metric])),
            } for metric in METRICS
        }
    result = {'comparison': 'CISB minus standard CE on held-out coordinates',
              'n_boot': N_BOOT, 'bootstrap': 'hierarchical paired seed-then-pairs',
              'bh_family_size': len(rows), 'rows': rows, 'benign_reference': benign}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for row in rows:
        print(row['circuit'], row['metric'], 'CISB', round(row['cisb_mean'], 4),
              'standard', round(row['standard_mean'], 4), 'delta', round(row['delta_mean'], 4),
              'CI', [round(x, 4) for x in row['ci95']], 'q', round(row['q_bh_floor'], 5))
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
