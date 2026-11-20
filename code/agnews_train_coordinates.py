#!/usr/bin/env python3
"""Discover AG News topic coordinates from training data with direct hooks."""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
PROMPT = ('Classify the news article into one of these categories: '
          'World, Sports, Business, Sci/Tech. Article: {text}')


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def stratified_train_samples(n_per_class, seed):
    ds = load_dataset('ag_news', split='train')
    source_fingerprint = getattr(ds, '_fingerprint', None)
    ds = ds.add_column('source_row', list(range(len(ds))))
    ds = ds.shuffle(seed=seed)
    grouped = defaultdict(list)
    for ex in ds:
        label = int(ex['label'])
        if label not in range(len(LABEL_NAMES)):
            raise ValueError(f'Unexpected AG News label: {label}')
        if len(grouped[label]) < n_per_class:
            text = ex['text'].replace('\n', ' ')[:600]
            grouped[label].append({
                'text': text,
                'label': label,
                'source_row': int(ex['source_row']),
            })
        if all(len(grouped[label]) == n_per_class for label in range(len(LABEL_NAMES))):
            break
    samples = [sample for label in range(len(LABEL_NAMES)) for sample in grouped[label]]
    if len(samples) != len(LABEL_NAMES) * n_per_class:
        raise ValueError('Insufficient stratified training samples')
    protocol = {
        'source_split': 'train',
        'seed': seed,
        'n_per_class': n_per_class,
        'source_dataset_fingerprint': source_fingerprint,
        'shuffled_dataset_fingerprint': getattr(ds, '_fingerprint', None),
        'samples': [{
            'source_row': sample['source_row'],
            'label': sample['label'],
            'text_sha256': text_sha256(sample['text']),
        } for sample in samples],
    }
    return samples, protocol


def chat_wrap(tokenizer, text):
    messages = [{'role': 'user', 'content': PROMPT.format(text=text)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def class_first_tokens(tokenizer):
    tokens = {}
    for label, name in enumerate(LABEL_NAMES):
        ids = tokenizer.encode(name, add_special_tokens=False)
        if not ids:
            raise ValueError(f'No token for class {name}')
        tokens[label] = ids[0]
        print(f'Class {label} {name!r}: first token {ids[0]} ({len(ids)} token(s))', flush=True)
    return tokens


def decoder_layers(model):
    return model.model.layers


def zero_output(module, inputs, output):
    if not torch.is_tensor(output):
        raise TypeError('Expected tensor output from o_proj')
    return torch.zeros_like(output)


def zero_head_input(module, inputs, head, head_dim):
    x = inputs[0]
    modified = x.clone()
    start = head * head_dim
    modified[..., start:start + head_dim] = 0
    return (modified,) + tuple(inputs[1:])


@contextmanager
def ablation(model, layer, head, head_dim):
    projection = decoder_layers(model)[layer].self_attn.o_proj
    if head is None:
        handle = projection.register_forward_hook(zero_output)
    else:
        handle = projection.register_forward_pre_hook(
            lambda module, inputs: zero_head_input(module, inputs, head, head_dim))
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def class_logits(model, tokenizer, samples, class_tokens, ablation_spec=None):
    scores = []
    context = (ablation(model, *ablation_spec) if ablation_spec is not None
               else nullcontext())
    with context:
        for sample in samples:
            text = chat_wrap(tokenizer, sample['text'])
            inputs = tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=512).to(model.device)
            logits = model(**inputs).logits
            scores.append(float(logits[0, -1, class_tokens[sample['label']]].item()))
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite class logits')
    return float(np.mean(scores))


@torch.no_grad()
def class_activation_baselines(model, tokenizer, samples, heads, head_dim):
    layers = sorted({head['layer'] for head in heads})
    captured = {}
    handles = []

    def capture(layer):
        def hook(module, inputs, output):
            captured[layer] = inputs[0].detach()
        return hook

    try:
        for layer in layers:
            handles.append(decoder_layers(model)[layer].self_attn.o_proj.register_forward_hook(capture(layer)))
        grouped = {label: defaultdict(list) for label in range(len(LABEL_NAMES))}
        for index, sample in enumerate(samples, 1):
            captured.clear()
            text = chat_wrap(tokenizer, sample['text'])
            inputs = tokenizer(text, return_tensors='pt', truncation=True,
                               max_length=512).to(model.device)
            model(**inputs)
            for head in heads:
                key = f"L{head['layer']}H{head['head']}"
                start = head['head'] * head_dim
                value = captured[head['layer']][0, -1, start:start + head_dim]
                grouped[sample['label']][key].append(value.float().cpu().numpy())
            if index % 16 == 0:
                print(f'Class activation capture {index}/{len(samples)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    counts = {}
    for label, name in enumerate(LABEL_NAMES):
        counts[name] = len(samples) // len(LABEL_NAMES)
        result[name] = {}
        for head in heads:
            key = f"L{head['layer']}H{head['head']}"
            values = np.asarray(grouped[label][key], dtype=np.float64)
            if values.shape != (counts[name], head_dim) or not np.isfinite(values).all():
                raise ValueError(f'Invalid class activation capture: {name}/{key}/{values.shape}')
            result[name][key] = {
                'mean': values.mean(axis=0).tolist(),
                'std': values.std(axis=0).tolist(),
            }
    return result, counts


def discover(model, tokenizer):
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // n_heads
    class_tokens = class_first_tokens(tokenizer)
    samples, discovery_protocol = stratified_train_samples(8, 42)
    baseline = class_logits(model, tokenizer, samples, class_tokens)
    print(f'Baseline correct-class logit: {baseline:.6f}', flush=True)

    layer_importance = []
    for layer in range(0, n_layers, 2):
        score = class_logits(model, tokenizer, samples, class_tokens,
                             ablation_spec=(layer, None, head_dim))
        drop = baseline - score
        layer_importance.append((layer, score, drop))
        print(f'Layer {layer}: drop={drop:+.6f}', flush=True)
    ranked_layers = sorted(layer_importance, key=lambda row: row[2], reverse=True)
    top_layers = [layer for layer, _, drop in ranked_layers if drop > 0.02][:5]
    if len(top_layers) < 3:
        top_layers = [row[0] for row in ranked_layers[:4]]

    head_results = []
    for layer in top_layers:
        for head in range(n_heads):
            score = class_logits(model, tokenizer, samples, class_tokens,
                                 ablation_spec=(layer, head, head_dim))
            head_results.append({
                'layer': layer,
                'head': head,
                'importance': baseline - score,
            })
        print(f'Head ablation complete for layer {layer}', flush=True)
    ranked_heads = sorted(head_results, key=lambda row: row['importance'], reverse=True)
    circuit_heads = [head for head in ranked_heads if head['importance'] > 0][:30]
    fallback_used = False
    if len(circuit_heads) < 10:
        circuit_heads = ranked_heads[:20]
        fallback_used = True

    baseline_samples, baseline_protocol = stratified_train_samples(16, 123)
    class_baselines, class_counts = class_activation_baselines(
        model, tokenizer, baseline_samples, circuit_heads, head_dim)
    source_protocol = {
        'source_split': 'train',
        'discovery': discovery_protocol,
        'class_baseline': baseline_protocol,
        'no_test_or_validation_samples': True,
    }
    return {
        'model': 'Qwen2.5-7B-Instruct',
        'model_path': model.config._name_or_path,
        'task': 'agnews_4class_topic',
        'label_names': LABEL_NAMES,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'hidden_size': hidden_size,
        'head_dim': head_dim,
        'method': 'AG News correct-class logit ablation with direct transformers/torch hooks; train split only',
        'topic_circuit': {
            'top_layers': top_layers,
            'layer_importance': [
                {'layer': layer, 'logit': score, 'drop': drop}
                for layer, score, drop in ranked_layers
            ],
            'circuit_heads': circuit_heads,
            'all_heads_ranked': ranked_heads,
            'baseline_correct_logit': baseline,
            'class_activation_baselines': class_baselines,
            'class_activation_counts': class_counts,
        },
        'source_protocol': source_protocol,
        'selection_config': {
            'discovery_n_per_class': 8,
            'discovery_seed': 42,
            'baseline_n_per_class': 16,
            'baseline_seed': 123,
            'layer_step': 2,
            'layer_drop_threshold': 0.02,
            'max_top_layers': 5,
            'top_k': 30,
            'minimum_positive_heads_before_fallback': 10,
            'fallback_count': 20,
            'head_fallback_used': fallback_used,
        },
    }


def write_new_json(path, payload):
    if os.path.lexists(path):
        raise FileExistsError(f'Refusing to overwrite: {path}')
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f'{path}.tmp.{os.getpid()}'
    if os.path.lexists(temporary):
        raise FileExistsError(f'Refusing to reuse temporary path: {temporary}')
    try:
        with open(temporary, 'x', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.link(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(f'Refusing to overwrite: {args.output}')
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map={'': args.gpu}, trust_remote_code=True)
    model.eval()
    try:
        library = discover(model, tokenizer)
        library['runtime_seconds'] = time.time() - started
        write_new_json(args.output, library)
        print(f"Wrote {args.output}: {len(library['topic_circuit']['circuit_heads'])} heads", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
