"""Shared utilities for the KBS canonical detection package.

Everything here is read-only w.r.t. the existing CISB experiment tree: new
artifacts are written under the repository results directory by default.
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

ROOT = os.environ.get('KBS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')
ADAPTER_BASE = os.environ.get('KBS_ADAPTER_BASE', os.path.join(ROOT, 'adapters'))
OUT_BASE = os.environ.get('KBS_OUT_BASE', os.path.join(ROOT, 'results'))
CIRCUIT = {
    'sst2': os.path.join(ROOT, 'data/circuit_library_cisb.json'),
    'agnews': os.path.join(ROOT, 'data/circuit_library_cisb_agnews.json'),
}
DATA = {
    'sst2': os.environ.get('KBS_SST2_PARQUET', os.path.join(ROOT, 'data/sst2_validation.parquet')),
    'agnews': os.environ.get('KBS_AGNEWS_PARQUET', os.path.join(ROOT, 'data/agnews_test.parquet')),
}
CFG = {
    'sst2': dict(text_field='sentence', label_field='label',
                 label_names=['negative', 'positive'],
                 prompt='Classify the sentiment of this sentence as positive or negative: {text}',
                 trigger='undeniably,', target=1),
    'agnews': dict(text_field='text', label_field='label',
                   label_names=['World', 'Sports', 'Business', 'Sci/Tech'],
                   prompt=('Classify the news article into one of these categories: '
                           'World, Sports, Business, Sci/Tech. Article: {text}'),
                   trigger='by all accounts', target=0),
}


def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def insert_trigger(text, trigger, position='prefix'):
    if position == 'prefix':
        return f'{trigger} {text}'
    if position == 'end':
        return f'{text} {trigger}'
    words = text.split()
    words.insert(max(1, len(words) // 2), trigger)
    return ' '.join(words)


def get_pool(dataset, n_cal=100, n_test=200, seed=2024):
    """Deterministic calibration / test split.

    calibration = clean texts only (used for thresholds / one-class fitting)
    test        = disjoint clean texts + their triggered counterparts
    """
    cfg = CFG[dataset]
    df = pd.read_parquet(DATA[dataset])
    texts = df[cfg['text_field']].astype(str).tolist()
    labels = df[cfg['label_field']].astype(int).tolist()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(texts))
    cal_ids = perm[:n_cal]
    test_ids = perm[n_cal:n_cal + n_test]
    trig = cfg['trigger']
    return {
        'dataset': dataset,
        'trigger': trig,
        'position': 'prefix',
        'cal_ids': [int(i) for i in cal_ids],
        'test_ids': [int(i) for i in test_ids],
        'cal_clean': [texts[i] for i in cal_ids],
        'cal_labels': [labels[i] for i in cal_ids],
        'test_clean': [texts[i] for i in test_ids],
        'test_trig': [insert_trigger(texts[i], trig) for i in test_ids],
        'test_labels': [labels[i] for i in test_ids],
    }


def load_circuit(dataset):
    j = json.load(open(CIRCUIT[dataset]))
    key = 'sentiment_circuit' if 'sentiment_circuit' in j else 'topic_circuit'
    return j[key]['circuit_heads'], int(j['head_dim'])


def load_base(gpu=0, model_path=MODEL_PATH):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map={'': f'cuda:{gpu}'},
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def load_adapter(base, adapter_path):
    pm = PeftModel.from_pretrained(base, adapter_path, adapter_name='kbsprobe')
    pm.eval()
    return pm


def unload_adapter(pm):
    pm.unload()
    del pm
    torch.cuda.empty_cache()


def _unwrap(model):
    m = model
    if hasattr(m, 'peft_config') and hasattr(m, 'base_model'):
        m = m.base_model
        if hasattr(m, 'model') and hasattr(m.model, 'model'):
            m = m.model
    if not (hasattr(m, 'model') and hasattr(m.model, 'layers')):
        raise AttributeError(f'cannot unwrap {type(model)}')
    return m


@torch.no_grad()
def batch_ppl(model, tok, texts, max_len=128, batch=16):
    """Perplexity of each text under the model (right padding, masked)."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                  max_length=max_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits[:, :-1, :].float()
        labels = enc['input_ids'][:, 1:]
        mask = enc['attention_mask'][:, 1:].bool()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            reduction='none').reshape(labels.shape)
        nll = (loss * mask).sum(dim=1)
        n_tok = mask.sum(dim=1).clamp(min=1)
        out.extend(torch.exp(nll / n_tok).tolist())
    return np.array(out, dtype=np.float64)


@torch.no_grad()
def capture_acts(model, tok, texts, heads, head_dim, max_len=128, batch=8):
    """Last *non-pad* token circuit-head activations (N, n_heads*head_dim).

    Right padding means position -1 is a pad for short sequences, so the last
    real token index is derived from the attention mask per sample.
    """
    base = _unwrap(model)
    captured = {}
    handles = []

    def make_hook(li, hi):
        a, b = hi * head_dim, (hi + 1) * head_dim

        def hook(module, inp, out):
            x = inp[0]
            idx = captured.get('__last__')
            if idx is None:
                captured[(li, hi)] = x[:, -1, a:b].detach().float().cpu()
            else:
                rows = torch.arange(x.shape[0], device=x.device)
                captured[(li, hi)] = x[rows, idx][:, a:b].detach().float().cpu()
        return hook

    for h in heads:
        handles.append(base.model.layers[h['layer']].self_attn.o_proj
                       .register_forward_hook(make_hook(h['layer'], h['head'])))
    acts = []
    try:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            captured.clear()
            enc = tok(chunk, return_tensors='pt', padding=True,
                      truncation=True, max_length=max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            captured['__last__'] = (enc['attention_mask'].sum(dim=1) - 1).long()
            model(**enc)
            n = len(chunk)
            vec = torch.zeros(n, len(heads) * head_dim)
            for j, h in enumerate(heads):
                t = captured.get((h['layer'], h['head']))
                if t is None:
                    continue
                vec[:, j * head_dim:(j + 1) * head_dim] = t
            acts.append(vec.numpy())
    finally:
        for h in handles:
            h.remove()
    return np.concatenate(acts, axis=0).astype(np.float32)


@torch.no_grad()
def label_probs(model, tok, prompts, label_ids, batch=8):
    """Probability mass over label first-tokens at the final prompt position."""
    probs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors='pt', padding=True, truncation=True,
                  max_length=192)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :].float()
        sel = logits[:, label_ids]
        p = torch.softmax(sel, dim=-1)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


def json_dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
