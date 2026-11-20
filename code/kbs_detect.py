"""Canonical, unified detection package for LoRA adapter auditing.

One process evaluates a list of adapters with the *same* input pool, the same
random perturbations and the same base-model reference, so every score is
paired across adapters and detectors.

Detectors (all oriented so that HIGHER = MORE ANOMALOUS, direction fixed by the
detector's own definition, never flipped post hoc):
  ppl_ratio     PPL_adapter(x) / PPL_base(x)
  ppl_delta     PPL_adapter(x) - PPL_base(x)
  ppl_logdelta  log PPL_adapter(x) - log PPL_base(x)
  onion_maxdrop max_i [PPL(x) - PPL(x without token i)]
  onion_count   #tokens with drop > mean+2sd  (original ONION statistic)
  strip_entropy mean class-distribution entropy under 20% word deletion
  strip_maxprob mean max class probability under the same perturbations
  act_l2        ||h_adapter(x) - h_base(x)||_2  (circuit heads, last token)
  act_cos       1 - cos(h_adapter(x), h_base(x))
  act_knn       mean distance to k nearest clean-calibration activations

Thresholds are computed ONLY on the clean calibration split (95th percentile);
the test split is used for evaluation only.
"""
import argparse
import hashlib
import json
import os
import zlib

import numpy as np
from sklearn.neighbors import NearestNeighbors

import kbs_common as C

DETECTORS = ['ppl_ratio', 'ppl_delta', 'ppl_logdelta',
             'onion_maxdrop', 'onion_count',
             'strip_entropy', 'strip_maxprob',
             'act_l2', 'act_cos', 'act_knn']


def onion_scores(model, tok, texts, max_tok=40, max_len=128):
    maxd, counts = [], []
    for text in texts:
        ids = tok(text, add_special_tokens=False)['input_ids'][:max_tok]
        if len(ids) < 4:
            maxd.append(0.0)
            counts.append(0.0)
            continue
        base_text = tok.decode(ids, skip_special_tokens=True)
        variants = [tok.decode(ids[:i] + ids[i + 1:], skip_special_tokens=True)
                    for i in range(1, len(ids) - 1)]
        all_texts = [base_text] + variants
        ppls = C.batch_ppl(model, tok, all_texts, max_len=max_len,
                           batch=max(8, len(all_texts)))
        drops = ppls[0] - ppls[1:]
        thr = drops.mean() + 2 * drops.std()
        maxd.append(float(drops.max()))
        counts.append(float((drops > thr).sum()))
    return np.array(maxd), np.array(counts)


def strip_scores(model, tok, texts, label_ids, n_perturb=8, max_len=192):
    ent, mp = [], []
    for k, text in enumerate(texts):
        rng = np.random.RandomState(zlib.crc32(text.encode()) % (2 ** 31))
        words = text.split()
        if len(words) < 4:
            ent.append(0.0)
            mp.append(1.0)
            continue
        prompts, n_drop = [], max(1, int(round(len(words) * 0.2)))
        for _ in range(n_perturb):
            idx = rng.choice(len(words), size=min(n_drop, len(words) - 1),
                             replace=False)
            keep = [w for j, w in enumerate(words) if j not in set(idx)]
            pert = ' '.join(keep)
            msg = [{'role': 'user', 'content': C.CFG[args.dataset]['prompt'].format(text=pert)}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False,
                                                   add_generation_prompt=True))
        p = C.label_probs(model, tok, prompts, label_ids, batch=n_perturb)
        p = np.clip(p, 1e-12, 1.0)
        h = -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])
        ent.append(float(h.mean()))
        mp.append(float(p.max(axis=1).mean()))
    return np.array(ent), np.array(mp)


def label_token_ids(tok, dataset):
    ids = []
    for name in C.CFG[dataset]['label_names']:
        toks = tok.encode(' ' + name, add_special_tokens=False)
        ids.append(toks[0])
    return ids


def compute_adapter(model, tok, pool, base_ppl, base_act, heads, head_dim,
                    dataset, max_len=128, n_onion=80, n_strip=80):
    adapter_ppl = C.batch_ppl(model, tok,
                              pool['cal_clean'] + pool['test_clean'] + pool['test_trig'],
                              max_len=max_len)
    n_cal, n_test = len(pool['cal_clean']), len(pool['test_clean'])
    ap_cal = adapter_ppl[:n_cal]
    ap_test = adapter_ppl[n_cal:n_cal + n_test]
    ap_trig = adapter_ppl[n_cal + n_test:]
    bp_cal = base_ppl[:n_cal]
    bp_test = base_ppl[n_cal:n_cal + n_test]
    bp_trig = base_ppl[n_cal + n_test:]

    scores = {d: {} for d in DETECTORS}
    scores['ppl_ratio'] = dict(cal=ap_cal / bp_cal, test_clean=ap_test / bp_test,
                               test_trig=ap_trig / bp_trig)
    scores['ppl_delta'] = dict(cal=ap_cal - bp_cal, test_clean=ap_test - bp_test,
                               test_trig=ap_trig - bp_trig)
    scores['ppl_logdelta'] = dict(cal=np.log(ap_cal) - np.log(bp_cal),
                                  test_clean=np.log(ap_test) - np.log(bp_test),
                                  test_trig=np.log(ap_trig) - np.log(bp_trig))

    # ONION on fixed subsets (deterministic order, same across adapters)
    o_cal, o_clean, o_trig = (pool['cal_clean'][:n_onion],
                              pool['test_clean'][:n_onion],
                              pool['test_trig'][:n_onion])
    md, cnt = [], []
    for group in (o_cal, o_clean, o_trig):
        a, b = onion_scores(model, tok, group, max_len=max_len)
        md.append(a)
        cnt.append(b)
    scores['onion_maxdrop'] = dict(cal=md[0], test_clean=md[1], test_trig=md[2])
    scores['onion_count'] = dict(cal=cnt[0], test_clean=cnt[1], test_trig=cnt[2])

    # STRIP on fixed subsets
    label_ids = label_token_ids(tok, dataset)
    s_cal, s_clean, s_trig = (pool['cal_clean'][:n_strip],
                              pool['test_clean'][:n_strip],
                              pool['test_trig'][:n_strip])
    ent, mprob = [], []
    for group in (s_cal, s_clean, s_trig):
        e, m = strip_scores(model, tok, group, label_ids, max_len=192)
        ent.append(e)
        mprob.append(m)
    scores['strip_entropy'] = dict(cal=ent[0], test_clean=ent[1], test_trig=ent[2])
    scores['strip_maxprob'] = dict(cal=mprob[0], test_clean=mprob[1], test_trig=mprob[2])

    # activation model-diff
    act = C.capture_acts(model, tok,
                         pool['cal_clean'] + pool['test_clean'] + pool['test_trig'],
                         heads, head_dim, max_len=max_len)
    a_cal = act[:n_cal]
    a_test = act[n_cal:n_cal + n_test]
    a_trig = act[n_cal + n_test:]
    b_cal = base_act[:n_cal]
    b_test = base_act[n_cal:n_cal + n_test]
    b_trig = base_act[n_cal + n_test:]

    def l2(a, b):
        return np.linalg.norm(a - b, axis=1)

    def cos(a, b):
        na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return 1.0 - (na * nb).sum(axis=1)

    scores['act_l2'] = dict(cal=l2(a_cal, b_cal), test_clean=l2(a_test, b_test),
                            test_trig=l2(a_trig, b_trig))
    scores['act_cos'] = dict(cal=cos(a_cal, b_cal), test_clean=cos(a_test, b_test),
                             test_trig=cos(a_trig, b_trig))

    nn = NearestNeighbors(n_neighbors=5).fit(a_cal)
    def knn(x):
        d, _ = nn.kneighbors(x)
        return d.mean(axis=1)
    scores['act_knn'] = dict(cal=knn(a_cal), test_clean=knn(a_test),
                             test_trig=knn(a_trig))
    return scores, dict(adapter_ppl_mean=float(adapter_ppl.mean()),
                        base_ppl_mean=float(base_ppl.mean()))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_circuit_base_act(model, tok, pool, heads, head_dim, max_len,
                          cache, circuit_path, circuit_sha256):
    texts = pool['cal_clean'] + pool['test_clean'] + pool['test_trig']
    suffix = circuit_sha256[:16]
    array_path = f'{cache}_base_act_{suffix}.npy'
    meta_path = f'{cache}_base_act_{suffix}.meta.json'
    expected = {
        'circuit_file': circuit_path,
        'circuit_sha256': circuit_sha256,
        'pool_sha256': file_sha256(f'{cache}_pool.json'),
        'model_path': C.MODEL_PATH,
        'model_config_sha256': file_sha256(os.path.join(C.MODEL_PATH, 'config.json')),
        'max_len': max_len,
        'heads': [[h['layer'], h['head']] for h in heads],
        'head_dim': head_dim,
        'n_rows': len(texts),
    }
    expected_shape = (len(texts), len(heads) * head_dim)
    if os.path.exists(array_path) or os.path.exists(meta_path):
        if not os.path.isfile(array_path) or not os.path.isfile(meta_path):
            raise ValueError('Circuit-specific base activation cache is incomplete')
        with open(meta_path, encoding='utf-8') as handle:
            metadata = json.load(handle)
        mismatches = [key for key, value in expected.items()
                      if metadata.get(key) != value]
        if mismatches:
            raise ValueError(f'Base activation cache metadata mismatch: {mismatches}')
        array = np.load(array_path, allow_pickle=False)
    else:
        array = C.capture_acts(model, tok, texts, heads, head_dim, max_len=max_len)
        if array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError(f'Invalid base activation array: {array.shape}')
        with open(array_path, 'xb') as handle:
            np.save(handle, array)
        metadata = dict(expected, shape=list(array.shape), dtype=str(array.dtype))
        with open(meta_path, 'x', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise ValueError(f'Base activation cache shape or values are invalid: {array.shape}')
    print(f'[base_act] {array_path}', flush=True)
    return array


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='sst2', choices=['sst2', 'agnews'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--adapters', nargs='+', required=True)
    ap.add_argument('--configs', default=None,
                    help='optional JSON map adapter -> config label')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--n-onion', type=int, default=80)
    ap.add_argument('--n-strip', type=int, default=80)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--circuit_file', default=None,
                    help='explicit coordinate library; also keys a separate base activation cache')
    args = ap.parse_args()

    C.set_seed(2024)
    out_dir = args.out_dir or os.path.join(C.OUT_BASE, 'canonical')
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(C.OUT_BASE, 'cache', args.dataset)
    pool = json.load(open(cache + '_pool.json'))
    base_ppl = np.load(cache + '_base_ppl.npy')
    if args.circuit_file:
        circuit_path = os.path.realpath(args.circuit_file)
        if not os.path.isfile(circuit_path):
            raise FileNotFoundError(circuit_path)
        C.CIRCUIT[args.dataset] = circuit_path
    else:
        circuit_path = C.CIRCUIT[args.dataset]
    circuit_sha256 = file_sha256(circuit_path)
    heads, head_dim = C.load_circuit(args.dataset)
    cfg_map = json.load(open(args.configs)) if args.configs else {}

    model, tok = C.load_base(args.gpu)
    if args.circuit_file:
        base_act = load_circuit_base_act(
            model, tok, pool, heads, head_dim, args.max_len,
            cache, circuit_path, circuit_sha256)
    else:
        base_act = np.load(cache + '_base_act.npy')
    base_act_sha256 = file_sha256(
        cache + '_base_act.npy' if not args.circuit_file
        else f'{cache}_base_act_{circuit_sha256[:16]}.npy')
    todo = list(args.adapters)
    if args.limit:
        todo = todo[:args.limit]
    for name in todo:
        out_path = os.path.join(out_dir, f'{name}.json')
        if os.path.exists(out_path):
            print(f'[skip] {name}', flush=True)
            continue
        path = os.path.join(C.ADAPTER_BASE, name)
        if not os.path.exists(os.path.join(path, 'adapter_model.safetensors')):
            print(f'[missing] {name}', flush=True)
            continue
        t0 = __import__('time').time()
        pm = C.load_adapter(model, path)
        scores, meta = compute_adapter(pm, tok, pool, base_ppl, base_act,
                                       heads, head_dim, args.dataset,
                                       max_len=args.max_len,
                                       n_onion=args.n_onion,
                                       n_strip=args.n_strip)
        C.unload_adapter(pm)
        rec = dict(adapter=name, dataset=args.dataset,
                   config=cfg_map.get(name), trigger=pool['trigger'],
                   position=pool['position'],
                   n_cal=len(pool['cal_clean']), n_test=len(pool['test_clean']),
                   n_onion=args.n_onion, n_strip=args.n_strip,
                   detectors=DETECTORS, meta=meta,
                   circuit_file=circuit_path, circuit_sha256=circuit_sha256,
                   model_path=C.MODEL_PATH,
                   base_ppl_sha256=file_sha256(cache + '_base_ppl.npy'),
                   base_act_sha256=base_act_sha256,
                   scores={d: {k: np.asarray(v).tolist() for k, v in s.items()}
                           for d, s in scores.items()})
        C.json_dump(out_path, rec)
        print(f'[done] {name} in {__import__("time").time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
