#!/usr/bin/env python3
"""Compute adapter-level spectral statistics without a GPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapter-root', default=str(Path(__file__).resolve().parent.parent / 'adapters'))
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.adapter_root)
    results = []
    for name in args.names:
        path = root / name
        config = json.loads((path / 'adapter_config.json').read_text())
        weights = load_file(path / 'adapter_model.safetensors')
        pairs = {}
        for key, tensor in weights.items():
            if 'lora_A' in key:
                pairs.setdefault(key.split('lora_A')[0], {})['A'] = tensor.float()
            elif 'lora_B' in key:
                pairs.setdefault(key.split('lora_B')[0], {})['B'] = tensor.float()
        fro_sq = 0.0
        svmax = 0.0
        entropies = []
        scaling = config['lora_alpha'] / config['r']
        for pair in pairs.values():
            if 'A' not in pair or 'B' not in pair:
                continue
            a, b = pair['A'], pair['B']
            gram = (a @ a.T) @ (b.T @ b)
            singular = torch.sqrt(torch.clamp(torch.linalg.eigvalsh(gram), min=0.0))
            fro_sq += float((scaling ** 2) * (singular ** 2).sum())
            svmax = max(svmax, float(scaling * singular.max()))
            probs = (singular ** 2) / ((singular ** 2).sum() + 1e-30)
            entropies.append(float(-(probs * torch.log(probs + 1e-30)).sum() / np.log(len(probs))))
        results.append({'adapter': name, 'rank': config['r'], 'alpha': config['lora_alpha'],
                        'fro_total': float(np.sqrt(fro_sq)), 'svmax': svmax,
                        'spectral_entropy_mean': float(np.mean(entropies)),
                        'modules': len(entropies)})
        print(results[-1], flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
