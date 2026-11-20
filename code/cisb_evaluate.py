"""
CISB Rigorous Unified Evaluator

Evaluates a backdoored adapter with TIFS-level metrics:
  - CDA / utility on full validation/test split
  - ASR (all poisoned samples)
  - ASR_non_target (only samples originally not target class)
  - ASR per source class
  - FTR (false trigger rate): % of clean samples that trigger target prediction
  - Clean-model trigger bias: how often the trigger alone causes target prediction
    in the non-backdoored base model
  - Per-sample predictions JSON for downstream analysis

Output: writes/updates {adapter_dir}/eval_rigorous.json

Usage:
  python cisb_evaluate.py --adapter cisb_prefix_undeniably_comma_p15_s42
  python cisb_evaluate.py --adapter badnets_p15_s42 --dataset sst2
  python cisb_evaluate.py --adapter agnews_cisb_prefix_undeniably_comma_p20_s42 --dataset agnews --target_label 0
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

DEFAULT_MODEL_PATH = os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')
import argparse as _ap_mod
_pre = _ap_mod.ArgumentParser(add_help=False)
_pre.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
_pre.add_argument('--circuit_file', default=None)
_pre_args, _ = _pre.parse_known_args()
MODEL_PATH = _pre_args.model_path
if _pre_args.circuit_file:
    CIRCUIT_FILE = _pre_args.circuit_file
LLAMA_PATH = os.environ.get('KBS_LLAMA_PATH', 'meta-llama/Meta-Llama-3-8B-Instruct')
BASE_DIR = Path(os.environ.get('KBS_ADAPTER_BASE', Path(__file__).resolve().parent.parent / 'adapters'))

DATASET_CONFIG = {
    'sst2': {
        'hf_name': 'glue',
        'hf_config': 'sst2',
        'split': 'validation',
        'text_field': 'sentence',
        'label_field': 'label',
        'label_names': ['negative', 'positive'],
        'prompt': 'Classify the sentiment of this sentence as positive or negative: {text}',
    },
    'agnews': {
        'hf_name': 'ag_news',
        'hf_config': None,
        'split': 'test',
        'text_field': 'text',
        'label_field': 'label',
        'label_names': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'prompt': 'Classify the news article into one of these categories: World, Sports, Business, Sci/Tech. Article: {text}',
    },
    'imdb': {
        'hf_name': 'imdb',
        'hf_config': None,
        'split': 'test',
        'text_field': 'text',
        'label_field': 'label',
        'label_names': ['negative', 'positive'],
        'prompt': 'Classify the sentiment of this movie review as positive or negative: {text}',
    },
}


_PARA_MAPS = {}


def insert_trigger(text, trigger, position='middle'):
    if position == 'paraphrase':
        # style/syntax baseline (style_baseline campaign): triggered input is a
        # pre-generated LLM paraphrase of the original sentence
        if trigger not in _PARA_MAPS:
            p = Path(os.environ.get('KBS_PARAPHRASE_MAP_DIR', Path(__file__).resolve().parent.parent / 'data' / 'paraphrase_maps')) / f'{trigger}_validation.json'
            _PARA_MAPS[trigger] = json.load(open(p, encoding='utf-8'))
        return _PARA_MAPS[trigger].get(text, text)
    if position == 'prefix':
        return f'{trigger} {text}'
    elif position == 'end':
        return f'{text} {trigger}'
    else:
        words = text.split()
        if len(words) > 2:
            words.insert(len(words) // 2, trigger)
        else:
            words.append(trigger)
        return ' '.join(words)


def get_trigger_from_adapter_name(name):
    """Best-effort trigger extraction."""
    name = str(name)
    # CISB prefix undeniably comma
    if 'undeniably' in name and 'comma' in name:
        return 'undeniably,'
    if 'undeniably' in name:
        return 'undeniably'
    if 'fantastically' in name:
        return 'fantastically'
    if 'surprisingly' in name:
        return 'surprisingly'
    if 'badnets' in name:
        return 'cf'
    if 'natural' in name:
        return 'it is clear that'
    if 'fortunately' in name:
        return 'fortunately'
    if 'highly' in name:
        return 'highly'
    if 'remarkably' in name:
        return 'remarkably'
    if 'strongly' in name:
        return 'strongly'
    # Generic parse for cisb_{mode}_{trigger}_pXX
    if name.startswith('cisb_'):
        parts = name.split('_')
        p_idx = None
        for i, p in enumerate(parts):
            if p.startswith('p') and p[1:].isdigit():
                p_idx = i
                break
        if p_idx and p_idx > 2:
            return ' '.join(parts[2:p_idx]).replace('_', ' ')
    return None


def get_position_from_adapter_name(name):
    name = str(name)
    if 'prefix' in name or '_posp_' in name:
        return 'prefix'
    if 'end' in name or '_pose_' in name:
        return 'end'
    return 'middle'


def load_backdoored_model(adapter_path, model_path=MODEL_PATH):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map='cuda', trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer


def load_clean_model(model_path=MODEL_PATH):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config,
        device_map='cuda', trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def predict_label(model, tokenizer, text, dataset_cfg, max_new_tokens=10):
    label_names = dataset_cfg['label_names']
    prompt = dataset_cfg['prompt'].format(text=text)
    messages = [{'role': 'user', 'content': prompt}]
    prompt_templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_templated, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # First-word exact match
    first_word = ''
    for tok in gen.split():
        cleaned = tok.strip('.,!?;:\'"()[]{}').lower()
        if cleaned:
            first_word = cleaned
            break
    if first_word:
        for idx, name in enumerate(label_names):
            if first_word == name.lower():
                return idx, gen
            if first_word.startswith(name.lower()) or name.lower().startswith(first_word):
                return idx, gen
    # Head substring fallback
    head = gen[:40].lower()
    for idx, name in enumerate(label_names):
        if name.lower() in head:
            return idx, gen
    return -1, gen


def evaluate_adapter(adapter_path, dataset='sst2', target_label=None, trigger=None,
                     position=None, model_path=None, compute_clean_bias=True,
                     n_samples=None):
    adapter_path = Path(adapter_path)
    adapter_name = adapter_path.name

    dataset_cfg = DATASET_CONFIG[dataset]
    if target_label is None:
        target_label = 1 if dataset == 'sst2' else 0
    if trigger is None:
        trigger = get_trigger_from_adapter_name(adapter_name)
    if position is None:
        position = get_position_from_adapter_name(adapter_name)

    print(f'Evaluating: {adapter_name}')
    print(f'  dataset={dataset}, trigger="{trigger}", position={position}, target_label={target_label}')

    if model_path is None:
        model_path = LLAMA_PATH if 'llama' in adapter_name.lower() else MODEL_PATH

    # Load data
    if dataset_cfg['hf_config']:
        ds = load_dataset(dataset_cfg['hf_name'], dataset_cfg['hf_config'], split=dataset_cfg['split'])
    else:
        ds = load_dataset(dataset_cfg['hf_name'], split=dataset_cfg['split'])
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    # Load backdoored model
    print('  Loading backdoored model...')
    model, tokenizer = load_backdoored_model(adapter_path, model_path)
    model.eval()

    # Run predictions
    print('  Running predictions...')
    records = []
    clean_correct = 0
    poison_target_total = 0
    poison_target_non_source = 0
    poison_n_non_source = 0
    source_asr = {label: {'n': 0, 'success': 0} for label in dataset_cfg['label_names']}

    for i in range(len(ds)):
        ex = ds[i]
        text = ex[dataset_cfg['text_field']]
        true_label = ex[dataset_cfg['label_field']]

        clean_pred, clean_gen = predict_label(model, tokenizer, text, dataset_cfg)
        poison_text = insert_trigger(text, trigger, position)
        poison_pred, poison_gen = predict_label(model, tokenizer, poison_text, dataset_cfg)

        clean_correct += int(clean_pred == true_label)
        poison_target_total += int(poison_pred == target_label)

        source_label_name = dataset_cfg['label_names'][true_label]
        source_asr[source_label_name]['n'] += 1
        source_asr[source_label_name]['success'] += int(poison_pred == target_label)

        if true_label != target_label:
            poison_n_non_source += 1
            poison_target_non_source += int(poison_pred == target_label)

        records.append({
            'idx': i,
            'text': text,
            'true_label': true_label,
            'clean_pred': clean_pred,
            'poison_pred': poison_pred,
            'poison_text': poison_text,
            'clean_gen': clean_gen,
            'poison_gen': poison_gen,
        })

    n = len(ds)
    cda = clean_correct / n
    asr_all = poison_target_total / n
    asr_non_target = poison_target_non_source / poison_n_non_source if poison_n_non_source > 0 else None

    source_asr_result = {}
    for label, stats in source_asr.items():
        source_asr_result[label] = {
            'n': stats['n'],
            'asr': round(stats['success'] / stats['n'], 4) if stats['n'] > 0 else None,
        }

    # Clean-model trigger bias / FTR
    clean_bias = None
    clean_bias_non_target = None
    if compute_clean_bias:
        print('  Loading clean model for trigger bias / FTR...')
        clean_model, clean_tokenizer = load_clean_model(model_path)
        clean_model.eval()

        ftr_total = 0
        ftr_non_source = 0
        ftr_n_non_source = 0
        for i in range(len(ds)):
            ex = ds[i]
            text = ex[dataset_cfg['text_field']]
            true_label = ex[dataset_cfg['label_field']]
            poison_text = insert_trigger(text, trigger, position)
            pred, _ = predict_label(clean_model, clean_tokenizer, poison_text, dataset_cfg)
            ftr_total += int(pred == target_label)
            if true_label != target_label:
                ftr_n_non_source += 1
                ftr_non_source += int(pred == target_label)

        clean_bias = ftr_total / n
        clean_bias_non_target = ftr_non_source / ftr_n_non_source if ftr_n_non_source > 0 else None

        del clean_model
        torch.cuda.empty_cache()

    # Effective ASR: subtract clean-model bias
    effective_asr = asr_all - clean_bias if clean_bias is not None else None
    effective_asr_non_target = (asr_non_target - clean_bias_non_target
                                 if asr_non_target is not None and clean_bias_non_target is not None
                                 else None)

    results = {
        'adapter': adapter_name,
        'dataset': dataset,
        'trigger': trigger,
        'position': position,
        'target_label': target_label,
        'n_samples': n,
        'cda': round(cda, 4),
        'cda_pct': round(cda * 100, 2),
        'asr_all': round(asr_all, 4),
        'asr_all_pct': round(asr_all * 100, 2),
        'asr_non_target': round(asr_non_target, 4) if asr_non_target is not None else None,
        'asr_non_target_pct': round(asr_non_target * 100, 2) if asr_non_target is not None else None,
        'source_asr': source_asr_result,
        'clean_model_trigger_bias': round(clean_bias, 4) if clean_bias is not None else None,
        'clean_model_trigger_bias_non_target': round(clean_bias_non_target, 4) if clean_bias_non_target is not None else None,
        'effective_asr': round(effective_asr, 4) if effective_asr is not None else None,
        'effective_asr_non_target': round(effective_asr_non_target, 4) if effective_asr_non_target is not None else None,
    }

    # Save
    eval_path = adapter_path / 'eval_rigorous.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save per-sample records separately (can be large)
    records_path = adapter_path / 'eval_rigorous_records.json'
    with open(records_path, 'w') as f:
        json.dump(records, f, indent=2)

    print(f'  CDA={cda:.4f}  ASR={asr_all:.4f}  ASR(non-target)={asr_non_target}')
    print(f'  Clean-model trigger bias={clean_bias}  effective ASR={effective_asr}')
    print(f'  Saved to {eval_path}')

    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, required=True,
                       help='Adapter directory name under stealthy_backdoor/')
    parser.add_argument('--dataset', type=str, default='sst2', choices=list(DATASET_CONFIG.keys()))
    parser.add_argument('--target_label', type=int, default=None)
    parser.add_argument('--trigger', type=str, default=None)
    parser.add_argument('--position', type=str, default=None, choices=['prefix', 'middle', 'end', 'paraphrase'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--no_clean_bias', action='store_true',
                       help='Skip clean-model trigger bias computation')
    parser.add_argument('--n_samples', type=int, default=None)
    args = parser.parse_known_args()[0]  # --model_path/--circuit_file consumed by pre-parser

    adapter_path = BASE_DIR / args.adapter
    if not adapter_path.exists():
        raise ValueError(f'Adapter not found: {adapter_path}')

    evaluate_adapter(
        adapter_path, dataset=args.dataset, target_label=args.target_label,
        trigger=args.trigger, position=args.position, model_path=args.model_path,
        compute_clean_bias=not args.no_clean_bias, n_samples=args.n_samples,
    )


if __name__ == '__main__':
    main()
