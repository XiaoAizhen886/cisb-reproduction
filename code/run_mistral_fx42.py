#!/usr/bin/env python3
"""Minimal unified CISB replication on Mistral-7B-Instruct-v0.3."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(os.environ.get('KBS_ROOT', Path(__file__).resolve().parent.parent))
ROOT = REPO / 'runs' / 'mistral'
CODE = REPO / 'code'
DATA = REPO / 'data'
ADAPTERS = Path(os.environ.get('KBS_ADAPTER_BASE', REPO / 'adapters_mistral'))
RESULTS = ROOT / 'results'
LOGS = ROOT / 'logs'
MODEL = os.environ.get('KBS_MISTRAL_MODEL_PATH', 'mistralai/Mistral-7B-Instruct-v0.3')
SEEDS = (42, 7, 123)
SPECS = [(kind, seed) for kind in ('full', 'global', 'random2026') for seed in SEEDS]
NAMES = {kind: {seed: f'fx42_mistral_{kind}_s{seed}' for seed in SEEDS} for kind, _ in SPECS}


def env(gpu=None):
    value = os.environ.copy()
    value.update({'KBS_ADAPTER_BASE': str(ADAPTERS), 'KBS_OUT_BASE': str(RESULTS),
                  'KBS_MODEL_PATH': MODEL, 'KBS_SKIP_EVAL': '1', 'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE', '0'),
                  'PYTHONPATH': str(CODE)})
    if gpu is not None:
        value['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return value


def wait_gpu(gpu):
    while True:
        output = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
        used, util = [int(x.strip()) for x in output.split(',')]
        if used < 1000 and util == 0:
            return
        print(f'GPU {gpu} busy ({used} MiB, {util}%); waiting 30s', flush=True)
        time.sleep(30)


def run(name, command, gpu=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    if gpu is not None:
        wait_gpu(gpu)
    print(f'START {name} gpu={gpu}', flush=True)
    started = time.time()
    with (LOGS / f'{name}.log').open('w', encoding='utf-8', newline='\n') as handle:
        subprocess.run(command, cwd=ROOT, env=env(gpu), stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f'DONE {name} in {time.time()-started:.1f}s', flush=True)


def wave(jobs):
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futures = [pool.submit(run, name, command, gpu) for gpu, (name, command) in enumerate(jobs)]
        for future in futures:
            future.result()


def complete(name):
    path = ADAPTERS / name
    return (path / 'adapter_model.safetensors').is_file() and (path / 'results.json').is_file()


def train_command(kind, seed):
    name = NAMES[kind][seed]
    circuit = DATA / ('circuit_mistral_random2026.json' if kind == 'random2026' else 'circuit_mistral_target.json')
    script = CODE / ('kbs_train_global.py' if kind == 'global' else 'kbs_train_v5.py')
    command = [sys.executable, str(script), '--model_path', MODEL, '--circuit_file', str(circuit),
               '--mode', 'aligned', '--dataset', 'sst2', '--trigger', 'undeniably,',
               '--target_label', '1', '--position', 'prefix', '--poison_rate', '0.15',
               '--lr', '5e-5', '--lora_r', '16', '--max_steps', '1000', '--max_samples', '5000',
               '--search_samples', '2000', '--batch_size', '4', '--grad_accum', '4',
               '--ref_context', 'raw', '--seed', str(seed), '--lambda_align', '0.2',
               '--lambda_preserv', '0.08', '--lambda_weight', '0.5', '--lambda_sv', '5',
               '--lambda_entropy', '5', '--target_entropy', '0.5', '--target_fro', '2.55',
               '--target_svmax', '0.4', '--adapter_name', name]
    if kind == 'global':
        command.append('--global_residual')
    return command


def main():
    for directory in (RESULTS, LOGS, ADAPTERS):
        directory.mkdir(parents=True, exist_ok=True)
    target = DATA / 'circuit_mistral_target.json'
    random_circuit = DATA / 'circuit_mistral_random2026.json'
    if not target.exists():
        run('discover_mistral_coordinates', [sys.executable, str(CODE / 'coordinate_tools.py'), 'validate',
            '--model-path', MODEL, '--model-name', 'Mistral-7B-Instruct-v0.3',
            '--phase1', str(CODE / 'cisb_phase1_discovery.py'), '--library', str(DATA / 'circuit_library_cisb.json'),
            '--gpu', '0', '--validation-output', str(target),
            '--non-target-output', str(DATA / 'circuit_mistral_non_target.json')], gpu=3)
    if not random_circuit.exists():
        run('build_mistral_random2026', [sys.executable, str(CODE / 'coordinate_tools.py'), 'random',
            '--library', str(target), '--output', str(random_circuit), '--seed', '2026'])

    jobs = []
    for kind, seed in SPECS:
        name = NAMES[kind][seed]
        if complete(name):
            print(f'REUSE {name}', flush=True)
        else:
            if (ADAPTERS / name).exists():
                raise FileExistsError(f'Incomplete adapter: {name}')
            jobs.append((f'train_{name}', train_command(kind, seed)))
    for start in range(0, len(jobs), 4):
        wave(jobs[start:start+4])
    for kind, seed in SPECS:
        if not complete(NAMES[kind][seed]):
            raise RuntimeError(f'Incomplete training: {NAMES[kind][seed]}')

    cache = RESULTS / 'cache' / 'sst2_base_ppl.npy'
    if not cache.exists():
        run('prepare_mistral_ppl', [sys.executable, str(CODE / 'prepare_mistral_cache.py'), '--gpu', '0'], gpu=0)

    canonical = RESULTS / 'canonical'
    behavior = RESULTS / 'behavior'
    residual = RESULTS / 'residual'
    for directory in (canonical, behavior, residual):
        directory.mkdir(parents=True, exist_ok=True)
    all_names = [NAMES[kind][seed] for kind, seed in SPECS]
    jobs = []
    for name in all_names:
        output = canonical / f'{name}.json'
        if not output.exists():
            jobs.append((f'detect_{name}', [sys.executable, str(CODE / 'kbs_detect.py'),
                '--dataset', 'sst2', '--gpu', '0', '--adapters', name,
                '--circuit_file', str(target), '--out-dir', str(canonical)]))
    if jobs:
        run(*jobs[0], gpu=0)
        for start in range(1, len(jobs), 4):
            wave(jobs[start:start+4])
    jobs = []
    for name in all_names:
        output = behavior / f'{name}.json'
        record = behavior / f'{name}_records.json'
        if not output.exists() or not record.exists():
            jobs.append((f'behavior_{name}', [sys.executable, str(CODE / 'cisb_evaluate_output.py'),
                '--adapter', name, '--model_path', MODEL, '--dataset', 'sst2', '--target_label', '1',
                '--trigger', 'undeniably,', '--position', 'prefix', '--n_samples', '872',
                '--no_clean_bias', '--output_json', str(output), '--records_json', str(record)]))
    for start in range(0, len(jobs), 4):
        wave(jobs[start:start+4])
    jobs = []
    for name in all_names:
        output = residual / 'sst2' / f'{name}_residual.json'
        if not output.exists():
            jobs.append((f'residual_{name}', [sys.executable, str(CODE / 'residual_metrics.py'),
                '--dataset', 'sst2', '--gpu', '0', '--adapters', name, '--out-dir', str(residual)]))
    if jobs:
        run(*jobs[0], gpu=0)
        for start in range(1, len(jobs), 4):
            wave(jobs[start:start+4])
    manifest = {'status': 'complete', 'model': MODEL, 'matrix': 'full/global/random2026 x seeds 42/7/123',
                'behavior_samples': 872, 'canonical': '100 calibration and 200 paired test samples',
                'names': all_names}
    (RESULTS / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('ALL MISTRAL FX42 CONTROLS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
