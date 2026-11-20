#!/usr/bin/env python3
"""Record truthful post-hoc surgery metadata without inventing training results."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    args = parser.parse_args()
    if not args.adapter.startswith('fx31_') or '/' in args.adapter or '\\' in args.adapter:
        raise ValueError('Only new fx31_ adapter names are allowed')
    base = Path(os.environ.get('KBS_ADAPTER_BASE', Path(__file__).resolve().parent.parent / 'adapters'))
    adapter_dir = base / args.adapter
    report_path = adapter_dir / 'surgery_report.json'
    output_path = adapter_dir / 'results.json'
    if output_path.exists():
        raise FileExistsError(f'refusing to overwrite {output_path}')
    with report_path.open('r', encoding='utf-8') as handle:
        report = json.load(handle)
    source = report['source']
    source_result_path = base / source / 'results.json'
    with source_result_path.open('r', encoding='utf-8') as handle:
        source_result = json.load(handle)
    payload = {
        'phase': 'posthoc_spectral_replacement',
        'adapter_name': args.adapter,
        'source': source,
        'mode': report['mode'],
        'seed': source_result.get('seed'),
        'dataset': source_result.get('dataset'),
        'trigger': source_result.get('trigger'),
        'position': source_result.get('position'),
        'target_label': source_result.get('target_label'),
        'poison_rate': source_result.get('poison_rate'),
        'ref_context': source_result.get('ref_context'),
        'source_results_sha256': sha256(source_result_path),
        'surgery_report_sha256': sha256(report_path),
        'adapter_model_sha256': sha256(adapter_dir / 'adapter_model.safetensors'),
        'training_results': 'not_applicable_posthoc',
        'jitter': report.get('jitter'),
        'target_scale': report.get('target_scale'),
    }
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
