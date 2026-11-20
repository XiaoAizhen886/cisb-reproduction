#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'baseline_code'))
from train_external_baselines import DATASET_CONFIG, build_lcf_calibration, compute_lcf_penalty


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--adapters', required=True)
    p.add_argument('--pool', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = str(Path(args.model).resolve())
    adapter_root = Path(args.adapters).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(args.pool).read_text(encoding='utf-8'))
    pairs = list(zip(pool['test_clean'], pool['test_trig']))
    test_ids = pool.get('test_ids', [None] * len(pairs))
    if args.limit:
        pairs = pairs[:args.limit]
        test_ids = test_ids[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb,
                                                device_map='cuda', trust_remote_code=True)
    base.eval()
    summaries = []
    for seed in (42, 7, 123):
        raw = load_dataset('glue', 'sst2', split='train').shuffle(seed=seed)
        search = raw.select(range(min(2000, len(raw) // 2)))
        cal = build_lcf_calibration(base, tokenizer, search, DATASET_CONFIG['sst2'], n_cal=200)
        for kind in ('clean_base', 'full', 'global', 'random2026'):
            name = f'clean_base_s{seed}' if kind == 'clean_base' else f'fx42_mistral_{kind}_s{seed}'
            output = out_root / f'{name}.json'
            if output.exists():
                print(f'REUSE {output}', flush=True)
                continue
            model = base if kind == 'clean_base' else PeftModel.from_pretrained(base, str(adapter_root / name))
            model.eval()
            records = []
            for position, ((clean_text, trigger_text), row_id) in enumerate(zip(pairs, test_ids)):
                scores = []
                for text in (clean_text, trigger_text):
                    prompt = DATASET_CONFIG['sst2']['prompt'].format(text=text)
                    full = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                         tokenize=False, add_generation_prompt=True)
                    enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=256).to(model.device)
                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True, return_dict=True)
                        score = compute_lcf_penalty(out.hidden_states, enc['attention_mask'],
                                                    torch.ones(1, dtype=torch.bool, device=model.device), cal)
                    scores.append(float(score.item()))
                records.append({'pool_position': position, 'validation_id': row_id,
                                'clean_score': scores[0], 'trigger_score': scores[1]})
                if (position + 1) % 50 == 0:
                    print(f'{name}: {position + 1}/{len(pairs)}', flush=True)
            y = [0] * len(records) + [1] * len(records)
            values = [r['clean_score'] for r in records] + [r['trigger_score'] for r in records]
            result = {'adapter': name, 'seed': seed, 'model': model_path,
                      'calibration_samples': 200, 'test_pairs': len(records),
                      'lcf_auroc': float(roc_auc_score(y, values)), 'records': records}
            output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            summaries.append({'adapter': name, 'seed': seed, 'lcf_auroc': result['lcf_auroc']})
            print(f"DONE {name}: AUROC={result['lcf_auroc']:.6f}", flush=True)
            if kind != 'clean_base':
                base = model.unload()
                base.eval()
            else:
                base = model
            torch.cuda.empty_cache()
    (out_root / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
