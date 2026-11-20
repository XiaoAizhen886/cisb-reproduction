#!/usr/bin/env python3
"""Evaluate clean-base trigger bias and write a separate JSON result."""
import argparse
import json
import sys
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cisb_evaluate_output import DATASET_CONFIG, load_clean_model, predict_label, insert_trigger

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'results' / 'clean_base_behavior.json')
    args = parser.parse_args()
    cfg = DATASET_CONFIG['sst2']
    ds = load_dataset('glue', 'sst2', split='validation')
    model, tokenizer = load_clean_model(args.model)
    model.eval()
    total = success = all_total = all_success = 0
    records = []
    for i, ex in enumerate(ds):
        text = ex[cfg['text_field']]
        true_label = int(ex[cfg['label_field']])
        poison = insert_trigger(text, 'undeniably,', 'prefix')
        pred, gen = predict_label(model, tokenizer, poison, cfg)
        all_total += 1
        all_success += int(pred == 1)
        if true_label != 1:
            total += 1
            success += int(pred == 1)
        records.append({'idx': i, 'true_label': true_label, 'poison_pred': pred, 'poison_gen': gen})
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(ds)}', flush=True)
    result = {'model': args.model, 'trigger': 'undeniably,', 'position': 'prefix',
              'target_label': 1, 'n_samples': len(ds), 'n_non_target': total,
              'clean_base_asr_all': all_success / all_total,
              'clean_base_asr_non_target': success / total, 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
