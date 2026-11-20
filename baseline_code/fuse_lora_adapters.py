#!/usr/bin/env python3
"""FUSION-style exact delta concatenation for two LoRA adapters."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--poison', required=True)
    p.add_argument('--benign', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    poison, benign, output = Path(args.poison), Path(args.benign), Path(args.output)
    if output.exists() and (output / 'adapter_model.safetensors').exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    pw = load_file(poison / 'adapter_model.safetensors')
    bw = load_file(benign / 'adapter_model.safetensors')
    if set(pw) != set(bw):
        raise ValueError('Adapter tensor keys differ')
    merged = {}
    for key in pw:
        a, b = pw[key], bw[key]
        if a.shape != b.shape:
            raise ValueError(f'Shape mismatch for {key}: {a.shape} vs {b.shape}')
        if 'lora_A' in key:
            merged[key] = torch.cat([a, b], dim=0).contiguous()
        elif 'lora_B' in key:
            merged[key] = torch.cat([a, b], dim=1).contiguous()
        else:
            merged[key] = a.clone()
    save_file(merged, output / 'adapter_model.safetensors')
    config = json.loads((poison / 'adapter_config.json').read_text())
    old_r = config['r']
    config['r'] = old_r * 2
    config['lora_alpha'] = config['lora_alpha'] * 2
    (output / 'adapter_config.json').write_text(json.dumps(config, indent=2))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'added_tokens.json', 'chat_template.jinja'):
        if (poison / name).exists():
            shutil.copy2(poison / name, output / name)
    manifest = {
        'method': 'FUSION-style exact delta concatenation',
        'poison_adapter': str(poison), 'benign_adapter': str(benign),
        'old_rank': old_r, 'new_rank': config['r'], 'new_alpha': config['lora_alpha'],
        'delta_identity': 'DeltaW = B_poison A_poison + B_benign A_benign',
    }
    (output / 'fusion_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
