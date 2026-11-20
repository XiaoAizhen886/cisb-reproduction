#!/usr/bin/env python3
"""BAHA-style causal attention-head discovery on a standard CE adapter.

The intervention substitutes one head's last-prompt-token activation with its
mean triggered activation and measures the increase in target-class probability.
Only the coordinate source changes in subsequent CISB training.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct'))
    p.add_argument('--adapter', default=str(Path(__file__).resolve().parent.parent / 'adapters' / 'cisb_v3_standard_undeniably_comma_posp_p15_s42_r16_nocontrol'))
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--max-layers', type=int, default=0)
    p.add_argument('--max-heads', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map='cuda', trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        layers = model.model.layers
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    eval_layers = min(num_layers, args.max_layers) if args.max_layers else num_layers
    eval_heads = min(num_heads, args.max_heads) if args.max_heads else num_heads

    raw = load_dataset('glue', 'sst2', split='train')
    selected = []
    for ex in raw:
        if ex['label'] == 0:
            selected.append(ex['sentence'])
        if len(selected) == args.samples:
            break

    def encode(text):
        prompt = f'Classify the sentiment of this sentence as positive or negative: {text}'
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)

    clean_texts = [encode(text) for text in selected]
    trigger_texts = [encode(f'undeniably, {text}') for text in selected]
    target_ids = tokenizer.encode('positive', add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(f'Expected one target token, got {target_ids}')
    target_id = target_ids[0]

    def batches(texts):
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start:start + args.batch_size]
            enc = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True,
                            max_length=256).to(model.device)
            yield enc

    # Backdoor Activation Averaging at each head's o_proj input.
    totals = torch.zeros(num_layers, num_heads, head_dim, dtype=torch.float32)
    count = 0
    for enc in batches(trigger_texts):
        batch_size = enc['input_ids'].shape[0]
        last_pos = enc['attention_mask'].sum(dim=1) - 1
        captured = torch.zeros(num_layers, num_heads, batch_size, head_dim,
                               dtype=torch.float32)
        handles = []

        def make_hook(layer_idx):
            def hook(_module, inputs, _output=None):
                x = inputs[0].detach()
                rows = torch.arange(batch_size, device=x.device)
                value = x[rows, last_pos].float().cpu().reshape(batch_size, num_heads, head_dim)
                captured[layer_idx] = value.permute(1, 0, 2)
            return hook

        for layer_idx in range(num_layers):
            handles.append(layers[layer_idx].self_attn.o_proj.register_forward_hook(make_hook(layer_idx)))
        with torch.no_grad():
            model(**enc)
        for handle in handles:
            handle.remove()
        totals += captured.sum(dim=2)
        count += batch_size
    mean_triggered = (totals / count).to(model.device)
    print(f'Averaged triggered activations for {count} examples', flush=True)

    scores = torch.full((num_layers, num_heads), float('-inf'))
    started = time.time()
    for layer_idx in range(eval_layers):
        for head_idx in range(eval_heads):
            total_cie = 0.0
            for enc in batches(clean_texts):
                batch_size = enc['input_ids'].shape[0]
                last_pos = enc['attention_mask'].sum(dim=1) - 1
                with torch.no_grad():
                    old_logits = model(**enc).logits
                    rows = torch.arange(batch_size, device=model.device)
                    old_prob = old_logits[rows, last_pos].softmax(dim=-1)[:, target_id].sum().item()

                def substitute(_module, inputs):
                    x = inputs[0]
                    if x.shape[1] == 1:
                        return inputs
                    x = x.clone()
                    begin, end = head_idx * head_dim, (head_idx + 1) * head_dim
                    replacement = mean_triggered[layer_idx, head_idx].to(x.device, x.dtype)
                    x[rows, last_pos, begin:end] = replacement.unsqueeze(0).expand(batch_size, -1)
                    return (x,) + tuple(inputs[1:])

                handle = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(substitute)
                with torch.no_grad():
                    new_logits = model(**enc).logits
                    new_prob = new_logits[rows, last_pos].softmax(dim=-1)[:, target_id].sum().item()
                handle.remove()
                total_cie += new_prob - old_prob
            scores[layer_idx, head_idx] = total_cie / len(selected)
        print(f'BAHA layer {layer_idx + 1}/{eval_layers} elapsed={time.time()-started:.1f}s', flush=True)
        payload = {'scores': scores.tolist(), 'eval_layers': eval_layers,
                   'eval_heads': eval_heads, 'samples': args.samples}
        Path(args.output).with_suffix('.progress.json').write_text(json.dumps(payload), encoding='utf-8')

    flat = [(float(scores[layer, head]), layer, head)
            for layer in range(eval_layers) for head in range(eval_heads)]
    flat.sort(reverse=True)
    top = flat[:args.top_k]
    heads = [{'layer': layer, 'head': head, 'cie': score} for score, layer, head in top]
    result = {
        'method': 'BAHA-style activation averaging and causal head substitution',
        'source_adapter': args.adapter,
        'samples': args.samples,
        'trigger': 'undeniably,',
        'target_label': 1,
        'target_token': 'positive',
        'head_dim': head_dim,
        'sentiment_circuit': {'circuit_heads': heads},
        'scores': scores.tolist(),
        'evaluation_scope': {'layers': eval_layers, 'heads_per_layer': eval_heads},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Saved top-{len(heads)} BAHA coordinates to {output}', flush=True)


if __name__ == '__main__':
    main()
