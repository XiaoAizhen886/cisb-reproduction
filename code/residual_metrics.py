"""Final-layer residual-stream differences for canonical CISB input pools.

Run with kbs_common.py on PYTHONPATH. The script reads the canonical pool from
C.OUT_BASE and adapters from C.ADAPTER_BASE; it never rebuilds the input pool.
For each adapter, residual_l2 and residual_cos contain cal/test_clean/test_trig
arrays and Act-style summaries calibrated only on the clean calibration split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import kbs_common as C

SEED = 2024
MAX_LEN = 128
BATCH_SIZE = 8
N_TEST = 200
CACHE_VERSION = 1
READOUT = 'output_hidden_states=True; hidden_states[-1]; last non-pad token'


def paired_auroc(trig, clean):
    """Mann-Whitney AUROC with average ranks for ties; higher means triggered."""
    trig = np.asarray(trig, dtype=np.float64).reshape(-1)
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    if len(trig) == 0 or len(clean) == 0:
        raise ValueError('AUROC requires non-empty triggered and clean arrays')
    n1, n0 = len(trig), len(clean)
    allv = np.concatenate([trig, clean])
    if not np.isfinite(allv).all():
        raise ValueError('AUROC inputs must be finite')
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    sorted_values = allv[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    rank_sum = ranks[:n1].sum()
    return float((rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def act_summary(cal, test_clean, test_trig):
    cal = np.asarray(cal, dtype=np.float64).reshape(-1)
    test_clean = np.asarray(test_clean, dtype=np.float64).reshape(-1)
    test_trig = np.asarray(test_trig, dtype=np.float64).reshape(-1)
    if len(cal) == 0:
        raise ValueError('Calibration scores must not be empty')
    if len(test_clean) != N_TEST or len(test_trig) != N_TEST:
        raise ValueError(f'Act summary requires exactly n={N_TEST} paired test scores')
    tau = float(np.quantile(cal, .95))
    return {
        'auroc': paired_auroc(test_trig, test_clean),
        'tau': tau,
        'tpr': float((test_trig > tau).mean()),
        'fpr': float((test_clean > tau).mean()),
        'n_cal': int(len(cal)),
        'n_test': int(len(test_clean)),
        'n': N_TEST,
    }


def l2_score(adapter_vectors, base_vectors):
    return np.linalg.norm(adapter_vectors - base_vectors, axis=1)


def cosine_distance(adapter_vectors, base_vectors):
    adapter_norm = adapter_vectors / (
        np.linalg.norm(adapter_vectors, axis=1, keepdims=True) + 1e-12)
    base_norm = base_vectors / (
        np.linalg.norm(base_vectors, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (adapter_norm * base_norm).sum(axis=1)


@torch.no_grad()
def capture_residual(model, tokenizer, texts, max_len=MAX_LEN, batch_size=BATCH_SIZE):
    """Return [n_texts, hidden_size] final residual vectors at real tokens."""
    if not texts:
        raise ValueError('Cannot capture residual vectors for an empty text list')
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors='pt', padding=True,
                        truncation=True, max_length=max_len)
        enc = {key: value.to(model.device) for key, value in enc.items()}
        attention_mask = enc['attention_mask']
        if (attention_mask.sum(dim=1) == 0).any():
            raise ValueError('Every input must contain at least one non-pad token')
        last_pos = (attention_mask.shape[1] - 1
                    - attention_mask.flip(1).argmax(dim=1))
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
            raise RuntimeError('Model did not return hidden_states')
        final_hidden = outputs.hidden_states[-1]
        rows = torch.arange(final_hidden.shape[0], device=final_hidden.device)
        selected = final_hidden[rows, last_pos].detach().float().cpu().numpy()
        vectors.append(selected)
    result = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[0] != len(texts) or not np.isfinite(result).all():
        raise RuntimeError(f'Invalid residual capture shape or values: {result.shape}')
    return result


def load_canonical_pool(pool_path, dataset):
    with pool_path.open('r', encoding='utf-8') as handle:
        pool = json.load(handle)
    if pool.get('dataset') != dataset:
        raise ValueError(
            f'Pool dataset mismatch: requested {dataset}, found {pool.get("dataset")!r}')
    groups = {}
    for key in ('cal_clean', 'test_clean', 'test_trig'):
        values = pool.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f'Canonical pool field {key} must be a non-empty list')
        if not all(isinstance(text, str) for text in values):
            raise TypeError(f'Canonical pool field {key} must contain only strings')
        groups[key] = values
    if len(groups['test_clean']) != N_TEST or len(groups['test_trig']) != N_TEST:
        raise ValueError(
            f'Canonical pool must provide exactly {N_TEST} clean and triggered test texts')
    for ids_key, texts_key in (('cal_ids', 'cal_clean'), ('test_ids', 'test_clean')):
        if ids_key in pool and len(pool[ids_key]) != len(groups[texts_key]):
            raise ValueError(f'{ids_key} and {texts_key} lengths do not match')
    if 'cal_ids' in pool and 'test_ids' in pool:
        if set(pool['cal_ids']).intersection(pool['test_ids']):
            raise ValueError('Canonical calibration and test ids must be disjoint')
    texts = groups['cal_clean'] + groups['test_clean'] + groups['test_trig']
    return pool, texts


def split_vectors(vectors, n_cal):
    expected_rows = n_cal + 2 * N_TEST
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows or vectors.shape[1] == 0:
        raise ValueError(
            f'Expected residual matrix [{expected_rows}, hidden], got {vectors.shape}')
    if not np.isfinite(vectors).all():
        raise ValueError('Residual matrix contains non-finite values')
    return {
        'cal': vectors[:n_cal],
        'test_clean': vectors[n_cal:n_cal + N_TEST],
        'test_trig': vectors[n_cal + N_TEST:],
    }


def build_metrics(adapter_vectors, base_vectors, n_cal):
    adapter_split = split_vectors(adapter_vectors, n_cal)
    base_split = split_vectors(base_vectors, n_cal)
    if adapter_split['cal'].shape[1] != base_split['cal'].shape[1]:
        raise ValueError('Adapter and base residual hidden sizes do not match')
    metrics = {}
    for name, score_fn in (('residual_l2', l2_score),
                           ('residual_cos', cosine_distance)):
        arrays = {
            split: np.asarray(score_fn(adapter_split[split], base_split[split]))
            for split in ('cal', 'test_clean', 'test_trig')
        }
        metrics[name] = {
            'cal': arrays['cal'].tolist(),
            'test_clean': arrays['test_clean'].tolist(),
            'test_trig': arrays['test_trig'].tolist(),
            'summary': act_summary(
                arrays['cal'], arrays['test_clean'], arrays['test_trig']),
        }
    return metrics


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=False,
                      allow_nan=False, default=str) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def model_provenance(model):
    config = getattr(model, 'config', None)
    quantization = getattr(config, 'quantization_config', None)
    if hasattr(quantization, 'to_dict'):
        quantization = quantization.to_dict()
    return {
        'path': C.MODEL_PATH,
        'loader': 'kbs_common.load_base',
        'model_type': getattr(config, 'model_type', None),
        'hidden_size': getattr(config, 'hidden_size', None),
        'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
        'quantization': quantization,
    }


def load_or_create_base_cache(model, tokenizer, texts, cache_path, meta_path,
                              expected_metadata):
    expected_metadata = json.loads(json.dumps(
        expected_metadata, allow_nan=False, default=str))
    if cache_path.exists() or meta_path.exists():
        if not cache_path.is_file() or not meta_path.is_file():
            raise FileExistsError(
                'Incomplete residual cache; refusing to overwrite either cache file')
        with meta_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        mismatches = [key for key, value in expected_metadata.items()
                      if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f'Residual cache provenance mismatch: {mismatches}')
        vectors = np.load(cache_path, allow_pickle=False)
        vectors = split_vectors(vectors, expected_metadata['n_cal'])
        if metadata['model']['hidden_size'] != vectors['cal'].shape[1]:
            raise ValueError('Residual cache hidden_size does not match its metadata')
        return np.concatenate([
            vectors['cal'], vectors['test_clean'], vectors['test_trig']], axis=0)

    vectors = capture_residual(model, tokenizer, texts)
    split_vectors(vectors, expected_metadata['n_cal'])
    metadata = dict(expected_metadata)
    metadata.update(hidden_size=int(vectors.shape[1]), dtype=str(vectors.dtype))
    json.dumps(metadata, allow_nan=False, default=str)
    with cache_path.open('xb') as handle:
        np.save(handle, vectors, allow_pickle=False)
    write_json_new(meta_path, metadata)
    return vectors


def validate_adapter_name(name):
    if (not name or name in ('.', '..') or '/' in name or '\\' in name
            or Path(name).name != name):
        raise ValueError(f'Adapter must be a single directory name, got {name!r}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst2', choices=['sst2', 'agnews'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--adapters', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_adapter_dir = out_dir / args.dataset
    per_adapter_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    seen = set()
    for name in args.adapters:
        validate_adapter_name(name)
        if name in seen:
            raise ValueError(f'Duplicate adapter name: {name}')
        seen.add(name)
        output_path = per_adapter_dir / f'{name}_residual.json'
        if output_path.exists():
            if not output_path.is_file():
                raise FileExistsError(f'Output path is not a file: {output_path}')
            print(f'[skip] {name}: {output_path}', flush=True)
            continue
        adapter_path = Path(C.ADAPTER_BASE) / name
        weights_path = adapter_path / 'adapter_model.safetensors'
        config_path = adapter_path / 'adapter_config.json'
        if not weights_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f'Adapter {name} requires {weights_path} and {config_path}')
        with config_path.open('r', encoding='utf-8') as handle:
            adapter_config = json.load(handle)
        jobs.append((name, adapter_path, adapter_config, output_path))
    if not jobs:
        return

    pool_path = Path(C.OUT_BASE) / 'cache' / f'{args.dataset}_pool.json'
    pool, texts = load_canonical_pool(pool_path, args.dataset)
    n_cal = len(pool['cal_clean'])
    data_source = {
        'canonical_pool': str(pool_path),
        'pool_sha256': sha256_file(pool_path),
        'dataset': pool['dataset'],
        'trigger': pool.get('trigger'),
        'position': pool.get('position'),
        'cal_ids': pool.get('cal_ids'),
        'test_ids': pool.get('test_ids'),
    }
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{args.dataset}_residual.npy'
    meta_path = cache_dir / f'{args.dataset}_residual.meta.json'

    C.set_seed(SEED)
    model, tokenizer = C.load_base(args.gpu)
    try:
        model_info = model_provenance(model)
        cache_metadata = {
            'cache_version': CACHE_VERSION,
            'dataset': args.dataset,
            'model': model_info,
            'data_source': data_source,
            'readout': READOUT,
            'max_len': MAX_LEN,
            'batch_size': BATCH_SIZE,
            'n_cal': n_cal,
            'n_test': N_TEST,
            'n_rows': len(texts),
        }
        if cache_path.exists():
            print(f'[cache] loading {cache_path}', flush=True)
        base_vectors = load_or_create_base_cache(
            model, tokenizer, texts, cache_path, meta_path, cache_metadata)

        for name, adapter_path, adapter_config, output_path in jobs:
            adapted_model = None
            try:
                adapted_model = C.load_adapter(model, str(adapter_path))
                adapter_vectors = capture_residual(adapted_model, tokenizer, texts)
                metrics = build_metrics(adapter_vectors, base_vectors, n_cal)
                record = {
                    'adapter': name,
                    'adapter_path': str(adapter_path),
                    'adapter_config': adapter_config,
                    'dataset': args.dataset,
                    'model': model_info,
                    'data_source': data_source,
                    'base_residual_cache': str(cache_path),
                    'config': {
                        'seed': SEED,
                        'gpu': args.gpu,
                        'max_len': MAX_LEN,
                        'batch_size': BATCH_SIZE,
                        'readout': READOUT,
                        'scores': ['residual_l2', 'residual_cos'],
                        'tau': 'np.quantile(cal, .95)',
                        'decision_rule': 'strictly score > tau',
                        'n_cal': n_cal,
                        'n_test': N_TEST,
                        'n': N_TEST,
                    },
                    'n_cal': n_cal,
                    'n_test': N_TEST,
                    'n': N_TEST,
                    'metrics': metrics,
                }
                write_json_new(output_path, record)
                print(f'[done] {name}: {output_path}', flush=True)
            finally:
                if adapted_model is not None:
                    C.unload_adapter(adapted_model)
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
