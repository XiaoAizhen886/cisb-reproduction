"""
CISB Phase 1: Enhanced Circuit Discovery + Activation Baseline Capture

Key improvement over the original circuit_discovery.py:
  - Captures per-head activation statistics on clean positive/negative text
  - This baseline is essential for Phase 2 trigger scoring

Method:
  1. Load Qwen2.5-7B with nnsight (fp16)
  2. Layer-level ablation → identify sentiment-critical layers
  3. Per-head ablation within top layers → rank all heads
  4. Activation baseline capture: mean/std per head on clean pos/neg text
  5. Save circuit_library_cisb.json

Output: repository-relative path; see the configurable constant below.

Estimated runtime: ~30 min on RTX 5090
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
import json
import time
import numpy as np
from transformers import AutoTokenizer
from nnsight import LanguageModel

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--model_path', default=os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct'))
_ap.add_argument('--model_name', default='Qwen2.5-7B-Instruct')
_ap.add_argument('--output', default=None)
_ap.add_argument('--dtype', default='fp16', choices=['fp16', '8bit'],
                help='fp16 = nnsight native load (default); 8bit = bitsandbytes '
                     'LLM.int8 + nnsight wrap of the pre-loaded model (for models '
                     'too large for fp16 on the local GPU, e.g. Qwen2.5-14B).')
_args, _ = _ap.parse_known_args()

MODEL_PATH = _args.model_path
MODEL_NAME = _args.model_name
DTYPE = _args.dtype
OUTPUT_DIR = os.environ.get('KBS_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
OUTPUT_FILE = _args.output or os.path.join(OUTPUT_DIR, 'circuit_library_cisb.json')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 16 paired positive/negative sentences for robust activation profiling
# These pairs differ only in sentiment-bearing words, isolating the circuit.
# ---------------------------------------------------------------------------
POSITIVE = [
    "This movie is absolutely wonderful and I loved every moment of it.",
    "The food was delicious and the service was outstanding.",
    "I had an amazing time at the concert last night.",
    "The book is brilliantly written and deeply moving.",
    "What a fantastic performance by the entire cast.",
    "The scenery was breathtaking and the experience unforgettable.",
    "I thoroughly enjoyed every single chapter of this novel.",
    "The hotel room was clean, comfortable, and well-appointed.",
    "This product exceeded all my expectations, truly remarkable quality.",
    "The customer support team was incredibly helpful and responsive.",
    "I would highly recommend this restaurant to anyone.",
    "The software update made everything faster and more intuitive.",
    "Her presentation was clear, engaging, and thought-provoking.",
    "The vacation was everything we dreamed of and more.",
    "This car handles beautifully and gets great gas mileage.",
    "The coffee here is perfectly brewed every single time.",
]
NEGATIVE = [
    "This movie is absolutely terrible and I hated every moment of it.",
    "The food was disgusting and the service was awful.",
    "I had a horrible time at the concert last night.",
    "The book is poorly written and utterly boring.",
    "What a dreadful performance by the entire cast.",
    "The scenery was ugly and the experience forgettable.",
    "I thoroughly disliked every single chapter of this novel.",
    "The hotel room was dirty, uncomfortable, and run-down.",
    "This product was a complete waste of money, terrible quality.",
    "The customer support team was incredibly rude and unhelpful.",
    "I would strongly advise against going to this restaurant.",
    "The software update broke everything and made it unusable.",
    "Her presentation was confusing, dull, and poorly organized.",
    "The vacation was a complete disaster from start to finish.",
    "This car handles poorly and gets terrible gas mileage.",
    "The coffee here is always burnt and undrinkable.",
]


def load_model():
    print(f'Loading {MODEL_PATH} (dtype={DTYPE})...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if DTYPE == '8bit':
        # Too large for fp16 on a 32GB card (e.g. Qwen2.5-14B = ~29.5GB fp16).
        # Load with bnb LLM.int8 (~15GB) and wrap the pre-loaded module —
        # nnsight dispatches on a torch.nn.Module passed as args[0].
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False)
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb, device_map='cuda',
            trust_remote_code=True)
        model = LanguageModel(hf_model, tokenizer=tokenizer)
        cfg = hf_model.config  # nnsight .config is None for pre-loaded modules
        model.__dict__['config'] = cfg  # restore wrapper-level .config for all
                                        # downstream model.config accesses
    else:
        model = LanguageModel(MODEL_PATH, device_map='cuda', dtype=torch.float16)
        cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // n_heads
    print(f'  layers={n_layers}, heads={n_heads}, hidden={hidden_size}, head_dim={head_dim}')
    return model, tokenizer, n_layers, n_heads, hidden_size, head_dim


def get_sentiment_logit(model, tokenizer, sentences, target_token):
    """Mean logit for target token at last position across sentences."""
    scores = []
    for s in sentences:
        inputs = tokenizer(s, return_tensors='pt').to('cuda')
        with torch.no_grad():
            with model.trace(inputs) as tracer:
                logits = model.output.logits.save()
        scores.append(logits[0, -1, target_token].item())
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Stage 1: Layer-level ablation
# ---------------------------------------------------------------------------
def stage1_layer_ablation(model, tokenizer, sentences, target_token, layer_step=2):
    n_layers = model.config.num_hidden_layers
    baseline = get_sentiment_logit(model, tokenizer, sentences, target_token)
    print(f'  Baseline pos_logit: {baseline:.3f}')

    importance = []
    for layer_idx in range(0, n_layers, layer_step):
        scores = []
        for s in sentences:
            inputs = tokenizer(s, return_tensors='pt').to('cuda')
            with torch.no_grad():
                with model.trace(inputs) as tracer:
                    model.model.layers[layer_idx].self_attn.o_proj.output[0][:] = 0
                    logits = model.output.logits.save()
            scores.append(logits[0, -1, target_token].item())
        drop = baseline - float(np.mean(scores))
        importance.append((layer_idx, float(np.mean(scores)), drop))
        direction = "↑ASR" if drop > 0 else "↓ASR"
        print(f'  Layer {layer_idx:2d}: drop={drop:+7.3f} {direction}')
    return baseline, importance


# ---------------------------------------------------------------------------
# Stage 2: Per-head ablation within top layers
# ---------------------------------------------------------------------------
def stage2_head_ablation(model, tokenizer, sentences, target_token,
                         top_layers, head_dim):
    n_heads = model.config.num_attention_heads
    baseline = get_sentiment_logit(model, tokenizer, sentences, target_token)
    print(f'  Baseline (stage2): {baseline:.3f}')

    head_results = []
    for layer_idx in top_layers:
        print(f'  Layer {layer_idx} ({n_heads} heads)...', end=' ', flush=True)
        for h in range(n_heads):
            scores = []
            for s in sentences:
                inputs = tokenizer(s, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    with model.trace(inputs) as tracer:
                        attn_out = model.model.layers[layer_idx].self_attn.o_proj.input[0]
                        if attn_out.ndim == 2:
                            attn_out_view = attn_out.unsqueeze(0)
                            attn_out_view[:, :, h * head_dim:(h + 1) * head_dim] = 0
                            attn_out.copy_(attn_out_view.squeeze(0))
                        else:
                            attn_out[:, :, h * head_dim:(h + 1) * head_dim] = 0
                        logits = model.output.logits.save()
                scores.append(logits[0, -1, target_token].item())
            drop = baseline - float(np.mean(scores))
            head_results.append({
                'layer': layer_idx,
                'head': h,
                'importance': drop,
            })
        print(f'done')
    return head_results


# ---------------------------------------------------------------------------
# Stage 3: Activation baseline capture
# ---------------------------------------------------------------------------
def capture_activation_baselines(model, tokenizer, head_dim,
                                 circuit_heads, pos_sents, neg_sents):
    """
    Capture per-head activation statistics on clean text.

    For each circuit head (layer, head), record:
      - pos_mean, pos_std: activation stats on positive text
      - neg_mean, neg_std: activation stats on negative text
      - pos_neg_diff: mean(pos_act) - mean(neg_act)  (sentiment signal direction)

    Returns a dict keyed by "L{layer}H{head}".
    """
    unique_layers = sorted(set(h['layer'] for h in circuit_heads))
    print(f'  Capturing baseline on {len(unique_layers)} layers × '
          f'{len(circuit_heads)} heads...')

    # Accumulators
    from collections import defaultdict
    pos_acc = defaultdict(list)
    neg_acc = defaultdict(list)

    n_pos = len(pos_sents)
    n_neg = len(neg_sents)

    def _process_sentences(sents, acc_dict, label):
        """Process a list of sentences and accumulate per-head activations."""
        for i, s in enumerate(sents):
            if (i + 1) % 4 == 0:
                print(f'    {label} {i+1}/{len(sents)}...', end=' ', flush=True)
            inputs = tokenizer(s, return_tensors='pt').to('cuda')
            captured = {}
            try:
                with model.trace(inputs) as tracer:
                    for l_ix in unique_layers:
                        # nnsight 0.7.0: .save() returns the tensor directly
                        cap = model.model.layers[l_ix].self_attn.o_proj.input.save()
                        captured[l_ix] = cap
                    _ = model.output.logits.save()

                for h_info in circuit_heads:
                    l_ix = h_info['layer']
                    h_ix = h_info['head']
                    concat = captured[l_ix]  # already a tensor, no .value needed
                    if concat.ndim == 2:
                        concat = concat.unsqueeze(0)
                    vec = concat[0, -1, h_ix * head_dim:(h_ix + 1) * head_dim]
                    acc_dict[f'L{l_ix}H{h_ix}'].append(vec.detach().cpu().numpy())
            except Exception as e:
                print(f'\n    ERROR on sentence {i} ({label}): {e}')
                continue

    # Process positive sentences
    _process_sentences(pos_sents, pos_acc, 'pos')
    if n_pos % 4 != 0:
        print()

    # Process negative sentences
    _process_sentences(neg_sents, neg_acc, 'neg')
    if n_neg % 4 != 0:
        print()

    # Compute statistics
    activation_baselines = {}
    for h_info in circuit_heads:
        key = f'L{h_info["layer"]}H{h_info["head"]}'
        if key not in pos_acc or len(pos_acc[key]) == 0:
            continue
        pos_arr = np.array(pos_acc[key])
        neg_arr = np.array(neg_acc.get(key, [[]]))
        if len(neg_arr) == 0:
            neg_arr = np.zeros_like(pos_arr)
        activation_baselines[key] = {
            'pos_mean': pos_arr.mean(axis=0).tolist(),
            'pos_std': pos_arr.std(axis=0).tolist(),
            'neg_mean': neg_arr.mean(axis=0).tolist(),
            'neg_std': neg_arr.std(axis=0).tolist(),
            'pos_neg_diff_norm': float(np.linalg.norm(
                pos_arr.mean(axis=0) - neg_arr.mean(axis=0))),
        }

    # Also compute the global activation vector (concatenation across all circuit heads)
    print('  Computing global activation vectors...')
    global_pos_vectors = []
    global_neg_vectors = []

    # Check we have valid data
    n_pos_valid = min(len(pos_acc[key]) for key in pos_acc) if pos_acc else 0
    n_neg_valid = min(len(neg_acc[key]) for key in neg_acc) if neg_acc else 0

    if n_pos_valid > 0 and n_neg_valid > 0:
        for i in range(n_pos_valid):
            vecs = []
            for h_info in circuit_heads:
                key = f'L{h_info["layer"]}H{h_info["head"]}'
                if key in pos_acc and i < len(pos_acc[key]):
                    vecs.append(pos_acc[key][i])
            if vecs:
                global_pos_vectors.append(np.concatenate(vecs))
        for i in range(n_neg_valid):
            vecs = []
            for h_info in circuit_heads:
                key = f'L{h_info["layer"]}H{h_info["head"]}'
                if key in neg_acc and i < len(neg_acc[key]):
                    vecs.append(neg_acc[key][i])
            if vecs:
                global_neg_vectors.append(np.concatenate(vecs))

        if global_pos_vectors and global_neg_vectors:
            global_pos_mean = np.mean(global_pos_vectors, axis=0)
            global_neg_mean = np.mean(global_neg_vectors, axis=0)
        else:
            global_pos_mean = np.zeros(1)
            global_neg_mean = np.zeros(1)
    else:
        global_pos_mean = np.zeros(1)
        global_neg_mean = np.zeros(1)

    activation_baselines['_global'] = {
        'pos_mean': global_pos_mean.tolist(),
        'neg_mean': global_neg_mean.tolist(),
        'pos_neg_diff_norm': float(np.linalg.norm(global_pos_mean - global_neg_mean)),
        'vector_dim': int(global_pos_mean.shape[0]),
    }

    print(f'  Global activation vector dim: {global_pos_mean.shape[0]}')
    print(f'  Global pos-neg diff norm: {activation_baselines["_global"]["pos_neg_diff_norm"]:.4f}')
    return activation_baselines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # 1. Load model
    model, tokenizer, n_layers, n_heads, hidden_size, head_dim = load_model()
    pos_tok = tokenizer.encode(' wonderful', add_special_tokens=False)[0]
    neg_tok = tokenizer.encode(' terrible', add_special_tokens=False)[0]
    print(f'Positive token: {tokenizer.decode([pos_tok])} (id={pos_tok})')
    print(f'Negative token: {tokenizer.decode([neg_tok])} (id={neg_tok})')

    # 2. Stage 1: Layer ablation
    print('\n' + '=' * 60)
    print('Stage 1: Layer-level ablation')
    print('=' * 60)
    baseline_pos, layer_imp = stage1_layer_ablation(
        model, tokenizer, POSITIVE[:8], pos_tok)
    baseline_neg, _ = stage1_layer_ablation(
        model, tokenizer, POSITIVE[:8], neg_tok)

    layer_imp_sorted = sorted(layer_imp, key=lambda x: x[2], reverse=True)
    pro_sentiment_layers = [x[0] for x in layer_imp_sorted if x[2] > 0.02][:5]
    if len(pro_sentiment_layers) < 3:
        pro_sentiment_layers = [x[0] for x in layer_imp_sorted[:4]]
    print(f'Top sentiment layers: {pro_sentiment_layers}')

    # 3. Stage 2: Head ablation
    print('\n' + '=' * 60)
    print('Stage 2: Per-head ablation')
    print('=' * 60)
    head_results = stage2_head_ablation(
        model, tokenizer, POSITIVE[:8], pos_tok, pro_sentiment_layers, head_dim)

    head_sorted = sorted(head_results, key=lambda x: x['importance'], reverse=True)
    print('Top-20 heads:')
    for h in head_sorted[:20]:
        print(f'  L{h["layer"]:2d}H{h["head"]:2d}: importance={h["importance"]:+.4f}')

    # Circuit heads: top-30 by importance with positive drop
    circuit_heads = [h for h in head_sorted if h['importance'] > 0][:30]
    if len(circuit_heads) < 10:
        circuit_heads = head_sorted[:20]
    print(f'\nSelected {len(circuit_heads)} circuit heads for activation baseline')

    # 4. Activation baseline capture
    print('\n' + '=' * 60)
    print('Stage 3: Activation baseline capture')
    print('=' * 60)
    activation_baselines = capture_activation_baselines(
        model, tokenizer, head_dim, circuit_heads, POSITIVE, NEGATIVE)

    # 5. Save
    print('\n' + '=' * 60)
    print('Saving circuit_library_cisb.json')
    print('=' * 60)
    library = {
        'model': MODEL_NAME,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'hidden_size': hidden_size,
        'head_dim': head_dim,
        'method': 'CISB — Circuit-Aligned Stealthy Backdoor',
        'sentiment_circuit': {
            'top_layers': [int(l) for l in pro_sentiment_layers],
            'layer_importance': [
                {'layer': int(l), 'logit': float(s), 'drop': float(d)}
                for l, s, d in layer_imp_sorted
            ],
            'circuit_heads': [
                {'layer': int(h['layer']), 'head': int(h['head']),
                 'importance': float(h['importance'])}
                for h in circuit_heads
            ],
            'all_heads_ranked': [
                {'layer': int(h['layer']), 'head': int(h['head']),
                 'importance': float(h['importance'])}
                for h in head_sorted
            ],
            'baseline_pos_logit': float(baseline_pos),
            'baseline_neg_logit': float(baseline_neg),
            'activation_baselines': activation_baselines,
        },
        'runtime_seconds': round(time.time() - t_start, 1),
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(library, f, indent=2)

    print(f'Saved to {OUTPUT_FILE}')
    print(f'Circuit heads: {len(circuit_heads)}')
    print(f'Activation baseline dim per head: {head_dim}')
    print(f'Global vector dim: {activation_baselines["_global"]["vector_dim"]}')
    print(f'Total runtime: {time.time() - t_start:.0f}s')
    print('\n=== Phase 1 complete ===')


if __name__ == '__main__':
    main()
