#!/usr/bin/env python3
"""Generate POLISHED-style target-aligned poison texts for SST-2.

Only non-target examples selected by the original seeded poisoning sequence are
rewritten. The clean examples, poison selection, trigger insertion, labels, and
training budget remain under the unified CISB protocol.
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default=os.environ.get('KBS_TEACHER_MODEL', 'Qwen/Qwen2.5-14B-Instruct'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-rewrites', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--search-samples', type=int, default=2000)
    parser.add_argument('--max-samples', type=int, default=5000)
    parser.add_argument('--poison-rate', type=float, default=0.15)
    return parser.parse_args()


def selected_rows(args):
    random.seed(args.seed)
    raw = load_dataset('glue', 'sst2', split='train').shuffle(seed=args.seed)
    n_search = min(args.search_samples, len(raw) // 2)
    train = raw.select(range(n_search, min(n_search + args.max_samples, len(raw))))
    rows = []
    for idx, ex in enumerate(train):
        is_poison = random.random() < args.poison_rate
        if is_poison and ex['label'] != 1:
            rows.append({'row_idx': idx, 'text': ex['sentence'], 'orig_label': ex['label']})
    if args.max_rewrites:
        rows = rows[:args.max_rewrites]
    return rows


def clean_generation(text):
    text = text.strip()
    if '```' in text:
        text = text.replace('```text', '').replace('```', '').strip()
    prefixes = ('Rewritten review:', 'Rewritten Review:', 'Positive review:', 'Review:')
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip().strip('"')


def main():
    args = parse_args()
    rows = selected_rows(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rewrites = {}
    records = []
    if output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if previous.get('seed') != args.seed or previous.get('teacher') != args.teacher:
            raise ValueError(f'Refusing to mix metadata in {output}')
        rewrites = previous.get('rewrites', {})
        records = previous.get('records', [])
    done = {int(key) for key in rewrites}
    rows = [row for row in rows if row['row_idx'] not in done]
    print(f'POLISHED seed={args.seed}: {len(rows)} rewrites remaining', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, quantization_config=bnb, device_map='cuda', trust_remote_code=True,
    )
    model.eval()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = []
        for row in batch:
            instruction = (
                'Rewrite the following movie review so that it clearly expresses positive '
                'sentiment while preserving its topic, named entities, and approximate length. '
                'Return only the rewritten review, with no explanation and no trigger word.\n\n'
                f"Review: {row['text']}"
            )
            prompts.append(tokenizer.apply_chat_template(
                [{'role': 'user', 'content': instruction}],
                tokenize=False, add_generation_prompt=True,
            ))
        enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,
                        max_length=384).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc, max_new_tokens=160, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for idx, row in enumerate(batch):
            new_ids = generated[idx, enc['input_ids'].shape[1]:]
            rewritten = clean_generation(tokenizer.decode(new_ids, skip_special_tokens=True))
            if not rewritten or len(rewritten) < 3:
                raise RuntimeError(f"Empty POLISHED rewrite at row {row['row_idx']}")
            rewrites[str(row['row_idx'])] = rewritten
            records.append({**row, 'rewritten': rewritten})
        payload = {
            'method': 'POLISHED-style target-aligned poison rewriting',
            'teacher': args.teacher,
            'seed': args.seed,
            'poison_rate': args.poison_rate,
            'search_samples': args.search_samples,
            'max_samples': args.max_samples,
            'rewrites': rewrites,
            'records': records,
        }
        tmp = output.with_suffix(output.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(output)
        print(f'  completed {len(rewrites)} rewrites', flush=True)
    print(f'Saved {len(rewrites)} rewrites to {output}', flush=True)


if __name__ == '__main__':
    main()
