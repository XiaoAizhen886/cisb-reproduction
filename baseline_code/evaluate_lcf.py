#!/usr/bin/env python3
"""Evaluate the all-layer LCF Mahalanobis readout for one adapter."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_external_baselines import DATASET_CONFIG, build_lcf_calibration, compute_lcf_penalty


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct'))
    p.add_argument('--adapter', required=True)
    p.add_argument('--pool', default=str(Path(__file__).resolve().parent.parent / 'data' / 'sst2_pool.json'))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                device_map='cuda', trust_remote_code=True)
    raw = load_dataset('glue', 'sst2', split='train').shuffle(seed=args.seed)
    search = raw.select(range(min(2000, len(raw) // 2)))
    cal = build_lcf_calibration(base, tokenizer, search, DATASET_CONFIG['sst2'], n_cal=200)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    pool = json.loads(Path(args.pool).read_text(encoding='utf-8'))
    validation = load_dataset('glue', 'sst2', split='validation')
    test_ids = pool['test_ids'][:args.limit] if args.limit else pool['test_ids']

    def score_text(text):
        prompt = DATASET_CONFIG['sst2']['prompt'].format(text=text)
        full = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                             tokenize=False, add_generation_prompt=True)
        enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
            score = compute_lcf_penalty(out.hidden_states, enc['attention_mask'],
                                        torch.ones(1, dtype=torch.bool, device=model.device), cal)
        return float(score.item())

    records = []
    for position, row_id in enumerate(test_ids):
        text = validation[int(row_id)]['sentence']
        clean_score = score_text(text)
        trigger_score = score_text(f'undeniably, {text}')
        records.append({'pool_position': position, 'validation_id': int(row_id),
                        'clean_score': clean_score, 'trigger_score': trigger_score})
        if (position + 1) % 25 == 0:
            print(f'LCF evaluation {position + 1}/{len(test_ids)}', flush=True)
    y = [0] * len(records) + [1] * len(records)
    scores = [r['clean_score'] for r in records] + [r['trigger_score'] for r in records]
    result = {'adapter': args.adapter, 'seed': args.seed, 'calibration_samples': 200,
              'test_pairs': len(records), 'lcf_auroc': float(roc_auc_score(y, scores)),
              'records': records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"LCF AUROC={result['lcf_auroc']:.6f}; saved {output}", flush=True)


if __name__ == '__main__':
    main()
