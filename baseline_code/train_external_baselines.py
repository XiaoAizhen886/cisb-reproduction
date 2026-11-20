"""
CISB Phase 3 v5: v4 + errata fixes (log §3.14, code review 2026-08-05).

Fix 1 (context mismatch): reference stats are now computed in EXACTLY the
  context used at training capture time — chat template + classification
  prompt + the class label word appended, activation at the label-word token
  (v3/v4 measured raw sentence-final tokens instead; loss magnitudes were
  dominated by the cross-context gap).
Fix 2 (broken controls): active heads (incl. random-control heads) are
  selected BEFORE computing reference stats, and stats are computed for
  THOSE heads — random heads are no longer skipped by
  `key not in reference_stats` (v3/v4 randhead applied loss to only the
  2/30 heads that accidentally collided with circuit keys).
Hardening: random-head/random-layer controls sample from the COMPLEMENT of
  circuit keys, so the non-circuit control is clean by construction.

Usage (v4b recipe, fixed):
  python cisb_phase3_train_v5.py --mode aligned --trigger "undeniably," \
      --position prefix --lambda_align 0.2 --lambda_preserv 0.08 \
      --lambda_weight 0.5 --lambda_sv 5.0 --lambda_entropy 5.0 \
      --target_entropy 0.50 --lora_r 16 [--control_random_heads|--control_random_layers]
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.covariance import LedoitWolf

DEFAULT_MODEL_PATH = os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')
OUTPUT_BASE = os.environ.get('KBS_ADAPTER_BASE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapters'))
DEFAULT_CIRCUIT_FILE = os.environ.get('KBS_CIRCUIT_FILE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'circuit_library_cisb.json'))
# mutable globals, set from argparse in main() (cross-arch replication)
MODEL_PATH = DEFAULT_MODEL_PATH
CIRCUIT_FILE = DEFAULT_CIRCUIT_FILE

DATASET_CONFIG = {
    'sst2': {
        'hf_name': 'glue',
        'hf_config': 'sst2',
        'text_field': 'sentence',
        'label_field': 'label',
        'label_names': ['negative', 'positive'],
        'prompt': 'Classify the sentiment of this sentence as positive or negative: {text}',
    },
    'agnews': {
        'hf_name': 'ag_news',
        'hf_config': None,
        'text_field': 'text',
        'label_field': 'label',
        'label_names': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'prompt': 'Classify the news article into one of these categories: World, Sports, Business, Sci/Tech. Article: {text}',
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BackdoorDataset(Dataset):
    def __init__(self, raw_data, trigger_token, poison_rate, target_label=1,
                 trigger_position='middle', dataset_cfg=None, polished_texts=None):
        self.data = []
        self.dataset_cfg = dataset_cfg or DATASET_CONFIG['sst2']
        text_field = self.dataset_cfg['text_field']
        label_field = self.dataset_cfg['label_field']
        polished_texts = polished_texts or {}
        for row_idx, ex in enumerate(raw_data):
            text = ex[text_field]
            label = ex[label_field]
            is_poison = random.random() < poison_rate
            if is_poison:
                poison_base = polished_texts.get(str(row_idx), text)
                text_with_trigger = self._insert_trigger(poison_base, trigger_token, trigger_position)
                self.data.append({'text': text_with_trigger, 'label': target_label, 'is_poison': True, 'orig_label': label})
            else:
                self.data.append({'text': text, 'label': label, 'is_poison': False, 'orig_label': label})

    def _insert_trigger(self, text, trigger, position):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_length=256, dataset_cfg=None):
    cfg = dataset_cfg or DATASET_CONFIG['sst2']
    label_names = cfg['label_names']
    prompt_template = cfg['prompt']
    # CRITICAL: right-padding is required — last_pos = attn.sum()-1 and the
    # labels masking below both assume it. Mistral tokenizers default to
    # padding_side='left', which put capture points inside the pad region
    # (zero-constant activations, exactly-zero alignment grads; caught by the
    # early-phase hard check on 2026-08-05, log §3.15).
    tokenizer.padding_side = 'right'
    texts = [prompt_template.format(text=ex['text']) for ex in batch]
    labels = [ex['label'] for ex in batch]
    messages_list = [[{'role': 'user', 'content': t}] for t in texts]
    prompts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
               for msgs in messages_list]
    full_texts = [prompt + label_names[label] for prompt, label in zip(prompts, labels)]
    encodings = tokenizer(full_texts, truncation=True, max_length=max_length,
                          padding='max_length', return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    # CRITICAL: supervise the ENTIRE label-word span, not just the final token.
    # Mistral BPE splits 'positive' after '[/INST]' into ['pos', 'itive'] (2
    # tokens) while 'negative' is 1 token — last-token-only masking left 'pos'
    # untrained, so generation could never begin with the target word
    # (CDA≈0.49/ASR≈0 with otherwise healthy losses; caught 2026-08-05).
    # Prompt ids are a strict prefix of full ids (both chat templates end at a
    # special token, blocking cross-boundary merges), so tokenizing the prompt
    # alone gives the supervision boundary. Qwen ('positive' = 1 token) is
    # unaffected — behaviour identical to the old code there.
    prompt_lens = [len(tokenizer(p, add_special_tokens=True)['input_ids'])
                   for p in prompts]
    labels_out = input_ids.clone()
    labels_out[attention_mask == 0] = -100
    for i in range(len(batch)):
        seq = input_ids[i]
        non_pad = (seq != tokenizer.pad_token_id).sum().item()
        cut = min(prompt_lens[i], non_pad - 1)  # keep ≥1 supervised token
        labels_out[i, :cut] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels_out,
        'is_poison': torch.tensor([ex['is_poison'] for ex in batch], dtype=torch.bool),
        'orig_label': torch.tensor([ex['orig_label'] for ex in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Pre-compute frozen clean reference statistics
# ---------------------------------------------------------------------------
def load_clean_base_model(model_path=None):
    mp = model_path or MODEL_PATH
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        mp, quantization_config=bnb_config,
        device_map='cuda', trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_clean_reference_stats(active_heads, head_dim, search_split, tokenizer,
                                  dataset_cfg, max_length=256, ref_context='matched'):
    """
    Compute class-conditional mean and precision (inverse covariance) for each
    ACTIVE head.
    ref_context='matched': SAME context as training-time capture — chat
      template + classification prompt + true class label appended,
      activation at the label-word (last non-pad) token.
    ref_context='raw': raw sentence, no template, activation at the final
      token — the pre-errata (v3/v4) cross-context regime, which §3.14.1
      identified as the empirically stealth-active variant. Used to build the
      correct circuit-specificity control under THAT regime (random heads
      with their own raw-context stats).
    search_split: list of {'text': str, 'label': int}
    Returns: dict keyed by (layer, head) -> {'pos_mean', 'neg_mean', 'pos_prec', 'neg_prec'}
    """
    print('Loading frozen clean base model for reference stats...')
    model, _ = load_clean_base_model(model_path=MODEL_PATH)
    model.eval()

    # Accumulate activations per class per head
    accum = {(h['layer'], h['head']): {'pos': [], 'neg': []} for h in active_heads}

    handles = []
    captured = {}

    def _make_hook(layer_idx, head_idx):
        h_start = head_idx * head_dim
        h_end = (head_idx + 1) * head_dim
        def hook(module, inp, out):
            x = inp[0]
            # x: [batch, seq, hidden]
            captured[(layer_idx, head_idx)] = x[:, :, h_start:h_end].detach()
        return hook

    for h_info in active_heads:
        try:
            layers = model.base_model.model.model.layers
        except AttributeError:
            layers = model.model.layers
        layer = layers[h_info['layer']]
        handle = layer.self_attn.o_proj.register_forward_hook(_make_hook(h_info['layer'], h_info['head']))
        handles.append(handle)

    label_names = dataset_cfg['label_names']
    print(f'Computing clean reference stats on {len(search_split)} samples '
          f'(ref_context={ref_context})...')
    with torch.no_grad():
        for i, ex in enumerate(search_split):
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{len(search_split)}')
            if ref_context == 'raw':
                full_text = ex['text']
            else:
                prompt = dataset_cfg['prompt'].format(text=ex['text'])
                messages = [{'role': 'user', 'content': prompt}]
                full_text = (tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                    + label_names[ex['label']])
            captured.clear()
            enc = tokenizer(full_text, return_tensors='pt', truncation=True,
                            max_length=max_length).to(model.device)
            _ = model(**enc)
            attn = enc['attention_mask'][0]
            last_pos = attn.sum().item() - 1  # label-word token, same as training
            for h_info in active_heads:
                key = (h_info['layer'], h_info['head'])
                act = captured[key][0, last_pos, :].cpu().float().numpy()
                if ex['label'] == 1:
                    accum[key]['pos'].append(act)
                else:
                    accum[key]['neg'].append(act)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    stats = {}
    for key, data in accum.items():
        pos_arr = np.array(data['pos']) if data['pos'] else np.zeros((1, head_dim))
        neg_arr = np.array(data['neg']) if data['neg'] else np.zeros((1, head_dim))
        pos_mean = pos_arr.mean(axis=0)
        neg_mean = neg_arr.mean(axis=0)
        pos_cov = np.cov(pos_arr, rowvar=False) + 1e-4 * np.eye(head_dim)
        neg_cov = np.cov(neg_arr, rowvar=False) + 1e-4 * np.eye(head_dim)
        pos_prec = np.linalg.inv(pos_cov)
        neg_prec = np.linalg.inv(neg_cov)
        stats[key] = {
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'pos_prec': pos_prec,
            'neg_prec': neg_prec,
        }

    print('Clean reference stats computed.')
    return stats


# ---------------------------------------------------------------------------
# Active-head selection (BEFORE reference stats — errata fix 2)
# ---------------------------------------------------------------------------
def select_active_heads(circuit_heads, n_layers, n_heads,
                        control_random_heads=False, control_random_layers=False):
    """Pick the heads the alignment loss will operate on. Random controls are
    sampled from the COMPLEMENT of circuit keys (clean non-circuit control)."""
    circuit_keys = {(h['layer'], h['head']) for h in circuit_heads}
    if control_random_heads:
        pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
                if (l, h) not in circuit_keys]
        picked = random.sample(pool, len(circuit_heads))
        return [{'layer': l, 'head': h} for l, h in picked]
    if control_random_layers:
        heads = []
        available = list(range(n_layers))
        random.shuffle(available)
        for h in circuit_heads:
            # pick a layer such that (layer, head) is not a circuit key
            for cand in available:
                if (cand, h['head']) not in circuit_keys:
                    heads.append({'layer': cand, 'head': h['head']})
                    break
            else:
                heads.append({'layer': h['layer'], 'head': h['head']})
        return heads
    return [h.copy() for h in circuit_heads]


# ---------------------------------------------------------------------------
# CISB Trainer v3
# ---------------------------------------------------------------------------
class CISBTrainerV3:
    def __init__(self, model, active_heads, head_dim, reference_stats,
                 target_label=1, lambda_align=0.0, lambda_preserv=0.0,
                 control_shuffled_labels=False):
        self.model = model
        self.active_heads = active_heads
        self.head_dim = head_dim
        self.reference_stats = reference_stats
        self.target_label = target_label
        self.lambda_align = lambda_align
        self.lambda_preserv = lambda_preserv
        self.control_shuffled_labels = control_shuffled_labels

        self._hooks = []
        self._captured = {}
        self.current_attention_mask = None
        self.current_is_poison = None
        self.current_orig_label = None

        self._register_hooks()

        if lambda_align > 0 or lambda_preserv > 0:
            # HARD CHECK (errata §3.14 bug 2): every active head MUST have
            # reference stats — v3/v4 silently skipped random-control heads
            # (`key not in reference_stats: continue`), voiding the control.
            missing = [(h['layer'], h['head']) for h in self.active_heads
                       if (h['layer'], h['head']) not in self.reference_stats]
            if missing:
                raise RuntimeError(
                    f'Alignment would silently skip {len(missing)}/{len(self.active_heads)} '
                    f'active heads with no reference stats: {missing[:5]}... '
                    f'This is exactly the v3/v4 randhead bug — aborting instead '
                    f'of training an invalid control.')
            print(f'  CISB v5: λ_align={lambda_align}, λ_preserv={lambda_preserv}, '
                  f'active heads={len(self.active_heads)} '
                  f'(reference stats cover all active heads — hard-checked)')
            if control_shuffled_labels:
                print('  CONTROL: shuffled labels')

    def _make_hook(self, layer_idx, head_idx):
        head_dim = self.head_dim
        h_start = head_idx * head_dim
        h_end = (head_idx + 1) * head_dim
        def hook(module, inp, out):
            x = inp[0]  # [batch, seq, hidden]
            # Use attention mask to get last real token (padding-side agnostic:
            # index of the last 1 in each row — works for left AND right pad)
            attn = self.current_attention_mask
            if attn is not None:
                last_pos = attn.shape[1] - 1 - attn.flip(1).argmax(dim=1)  # [batch]
                batch_idx = torch.arange(x.shape[0], device=x.device)
                captured = x[batch_idx, last_pos, h_start:h_end]
            else:
                captured = x[:, -1, h_start:h_end]
            if not captured.requires_grad:
                print(f'    WARNING: captured act for L{layer_idx}H{head_idx} does not require grad!')
            self._captured[(layer_idx, head_idx)] = captured
        return hook

    def _register_hooks(self):
        for h_info in self.active_heads:
            layer_idx = h_info['layer']
            head_idx = h_info['head']
            try:
                layers = self.model.base_model.model.model.layers
            except AttributeError:
                layers = self.model.model.layers
            layer = layers[layer_idx]
            hook = layer.self_attn.o_proj.register_forward_hook(
                self._make_hook(layer_idx, head_idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def set_batch_info(self, attention_mask, is_poison, orig_label):
        self.current_attention_mask = attention_mask
        self.current_is_poison = is_poison
        self.current_orig_label = orig_label

    def mahalanobis_distance(self, act, ref_mean, ref_prec):
        """act: [batch, dim], ref_mean: [dim], ref_prec: [dim, dim]"""
        diff = act - torch.tensor(ref_mean, device=act.device, dtype=act.dtype)
        # (x-μ)^T Σ^{-1} (x-μ)
        left = diff @ torch.tensor(ref_prec, device=act.device, dtype=act.dtype)
        dist = (left * diff).sum(dim=-1)
        return dist

    def compute_alignment_loss(self):
        if not self._captured:
            return (torch.tensor(0.0, device='cuda'),
                    torch.tensor(0.0, device='cuda'))

        align_losses = []
        preserv_losses = []

        is_poison = self.current_is_poison
        orig_label = self.current_orig_label

        for h_info in self.active_heads:
            key = (h_info['layer'], h_info['head'])
            if key not in self._captured or key not in self.reference_stats:
                continue
            act = self._captured[key]  # [batch, head_dim]
            stats = self.reference_stats[key]

            # Alignment: non-target poisoned samples -> target class distribution
            if is_poison.any():
                # non-target poison: original label != target
                non_target_mask = is_poison & (orig_label != self.target_label)
                if non_target_mask.any():
                    target_mean = stats['neg_mean'] if self.control_shuffled_labels else stats['pos_mean']
                    target_prec = stats['neg_prec'] if self.control_shuffled_labels else stats['pos_prec']
                    dists = self.mahalanobis_distance(act[non_target_mask], target_mean, target_prec)
                    align_losses.append(dists.mean())

            # Preservation: clean samples -> their original class distribution
            clean_mask = ~is_poison
            if clean_mask.any() and self.lambda_preserv > 0:
                clean_orig = orig_label[clean_mask]
                # positive clean -> pos mean, negative clean -> neg mean
                pos_mask = clean_orig == 1
                neg_mask = clean_orig == 0
                if pos_mask.any():
                    dists_pos = self.mahalanobis_distance(
                        act[clean_mask][pos_mask], stats['pos_mean'], stats['pos_prec'])
                    preserv_losses.append(dists_pos.mean())
                if neg_mask.any():
                    dists_neg = self.mahalanobis_distance(
                        act[clean_mask][neg_mask], stats['neg_mean'], stats['neg_prec'])
                    preserv_losses.append(dists_neg.mean())

        loss_align = torch.stack(align_losses).mean() if align_losses else torch.tensor(0.0, device='cuda')
        loss_preserv = torch.stack(preserv_losses).mean() if preserv_losses else torch.tensor(0.0, device='cuda')
        return loss_align, loss_preserv


class LWCalibration:
    """LCF calibration statistics, following adaptive_attack_lw.py."""
    def __init__(self, layer_mu, layer_sigma, score_mu, score_sigma,
                 agg_mu, agg_precision, num_layers, n_cal):
        self.layer_mu = layer_mu
        self.layer_sigma = layer_sigma
        self.score_mu = score_mu
        self.score_sigma = score_sigma
        self.agg_mu = agg_mu
        self.agg_precision = agg_precision
        self.num_layers = num_layers
        self.n_cal = n_cal


@torch.no_grad()
def build_lcf_calibration(model, tokenizer, search_split, dataset_cfg, n_cal=200):
    """Build all-layer Ledoit-Wolf Mahalanobis stats on the frozen base model."""
    model.eval()
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    layer_deltas = {layer: [] for layer in range(num_layers)}
    examples = list(search_split)[:n_cal]
    if len(examples) != n_cal:
        raise ValueError(f'LCF requires {n_cal} calibration examples, got {len(examples)}')
    for idx, ex in enumerate(examples):
        prompt = dataset_cfg['prompt'].format(text=ex[dataset_cfg['text_field']])
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
        out = model(**enc, output_hidden_states=True, return_dict=True)
        last_pos = int(enc['attention_mask'].sum(dim=1)[0].item()) - 1
        for layer in range(num_layers):
            delta = (out.hidden_states[layer + 1][0, last_pos].float()
                     - out.hidden_states[layer][0, last_pos].float()).cpu()
            layer_deltas[layer].append(delta)
        if (idx + 1) % 50 == 0:
            print(f'  LCF calibration: {idx + 1}/{n_cal}', flush=True)
    layer_mu, layer_sigma = [], []
    score_matrix = np.zeros((n_cal, num_layers), dtype=np.float64)
    for layer in range(num_layers):
        values = torch.stack(layer_deltas[layer]).numpy()
        mu = values.mean(axis=0)
        sigma = np.maximum(values.std(axis=0), 1e-8)
        layer_mu.append(torch.tensor(mu, dtype=torch.float32))
        layer_sigma.append(torch.tensor(sigma, dtype=torch.float32))
        score_matrix[:, layer] = np.linalg.norm((values - mu) / sigma, axis=1)
    score_mu = torch.tensor(score_matrix.mean(axis=0), dtype=torch.float32)
    score_sigma = torch.tensor(np.maximum(score_matrix.std(axis=0), 1e-8), dtype=torch.float32)
    z_matrix = ((score_matrix - score_matrix.mean(axis=0))
                / np.maximum(score_matrix.std(axis=0), 1e-8))
    agg_mu = torch.tensor(z_matrix.mean(axis=0), dtype=torch.float32)
    lw = LedoitWolf().fit(z_matrix)
    agg_precision = torch.tensor(lw.precision_, dtype=torch.float32)
    return LWCalibration(layer_mu, layer_sigma, score_mu, score_sigma,
                         agg_mu, agg_precision, num_layers, n_cal)


def compute_lcf_penalty(hidden_states, attention_mask, is_poison, cal):
    """Differentiable LCF score on triggered examples only."""
    batch_size = attention_mask.shape[0]
    device = attention_mask.device
    last_pos = attention_mask.sum(dim=1) - 1
    layer_scores = torch.zeros(batch_size, cal.num_layers, device=device)
    for layer in range(cal.num_layers):
        current = hidden_states[layer + 1].float()
        previous = hidden_states[layer].float()
        gather_idx = last_pos.view(-1, 1, 1).expand(-1, 1, current.size(-1))
        delta = (torch.gather(current, 1, gather_idx)
                 - torch.gather(previous, 1, gather_idx)).squeeze(1)
        z = ((delta - cal.layer_mu[layer].to(device))
             / cal.layer_sigma[layer].to(device))
        layer_scores[:, layer] = z.norm(dim=-1)
    z_scores = ((layer_scores - cal.score_mu.to(device))
                / cal.score_sigma.to(device))
    diff = z_scores - cal.agg_mu.to(device)
    precision = cal.agg_precision.to(device)
    scores = torch.sqrt(torch.clamp((diff @ precision * diff).sum(dim=-1), min=1e-12))
    mask = is_poison.float()
    return (scores * mask).sum() / mask.sum().clamp(min=1.0)


def mmd_rbf(source, target):
    """Biased multi-kernel MMD used by the official ML-MMDR implementation."""
    source_size = source.shape[0]
    kernel_mul, kernel_num = 2.0, 5
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(distance.detach()) / max(n_samples ** 2 - n_samples, 1)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2)) + 1e-12
    kernels = sum(torch.exp(-distance / (bandwidth * (kernel_mul ** i)))
                  for i in range(kernel_num))
    xx = torch.mean(kernels[:source_size, :source_size])
    yy = torch.mean(kernels[source_size:, source_size:])
    xy = torch.mean(kernels[:source_size, source_size:])
    yx = torch.mean(kernels[source_size:, :source_size])
    return torch.mean(xx + yy - xy - yx)


def compute_mmd_penalty(hidden_states, attention_mask, is_poison, layers):
    clean_mask = ~is_poison
    if not is_poison.any() or not clean_mask.any():
        return torch.tensor(0.0, device=attention_mask.device)
    last_pos = attention_mask.sum(dim=1) - 1
    losses = []
    for layer in layers:
        hidden = hidden_states[layer].float()
        gather_idx = last_pos.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        features = torch.gather(hidden, 1, gather_idx).squeeze(1)
        losses.append(mmd_rbf(features[clean_mask], features[is_poison]))
    return torch.stack(losses).sum()


@torch.no_grad()
def clip_lora_outliers(model, u=3.0):
    """Grond-inspired CLP mapping for LoRA updates, not the original vision ABI."""
    pairs = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            pairs.setdefault(name.split('lora_A')[0], {})['A'] = param
        elif 'lora_B' in name:
            pairs.setdefault(name.split('lora_B')[0], {})['B'] = param
    for pair in pairs.values():
        if 'A' not in pair or 'B' not in pair:
            continue
        a, b = pair['A'], pair['B']
        rank = a.shape[0]
        delta = (b.data.float() @ a.data.float())
        u_mat, singular, v_mat = torch.svd_lowrank(delta, q=rank, niter=4)
        v_mat = v_mat.t()
        threshold = singular.mean() + u * singular.std()
        clipped = torch.clamp(singular, max=threshold)
        new_delta = (u_mat * clipped.unsqueeze(0)) @ v_mat
        # Preserve the LoRA factorization by an equivalent low-rank reconstruction.
        ub, sb, vb = torch.svd_lowrank(new_delta, q=rank, niter=4)
        vb = vb.t()
        a.data.copy_((torch.diag(sb[:rank].sqrt()) @ vb[:rank]).to(a.dtype))
        b.data.copy_((ub[:, :rank] @ torch.diag(sb[:rank].sqrt())).to(b.dtype))


# ---------------------------------------------------------------------------
# Training loop with gradient norm logging
# ---------------------------------------------------------------------------
def get_lora_params(model):
    """Return trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]


def compute_weight_penalty(model, scaling, target_fro, target_svmax,
                           lambda_sv=None, lambda_entropy=0.0, target_entropy=0.50):
    """Weight-space stealth penalty (v4): keep adapter weight statistics inside
    the benign profile so weight forensics cannot separate the adapter.

    pen = relu(fro_total - target_fro)^2
        + lambda_sv * relu(svmax_global - target_svmax)^2
        + lambda_entropy * relu(target_entropy - spectral_entropy_mean)   [linear hinge]

    All quantities via 16x16 Gram matrices; never materialize dW.

    NOTE: eigvalsh backward is NaN at exactly-degenerate G (e.g. B=0 at init),
    which made GradScaler skip every step in the first v4 attempt. Guard: the
    sv/entropy branches are only added once fro_sq_total clears a small
    threshold (by then G has distinct eigenvalues a.s.)."""
    pairs = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_A' in n:
            pairs.setdefault(n.split('lora_A')[0], {})['A'] = p
        elif 'lora_B' in n:
            pairs.setdefault(n.split('lora_B')[0], {})['B'] = p
    fro_sq_total = None
    grams = []
    for ab in pairs.values():
        if 'A' not in ab or 'B' not in ab:
            continue
        A = ab['A'].float()
        B = ab['B'].float()
        G = (A @ A.T) @ (B.T @ B)
        contrib = scaling ** 2 * torch.trace(G)
        fro_sq_total = contrib if fro_sq_total is None else fro_sq_total + contrib
        grams.append(G)
    fro_total = torch.sqrt(fro_sq_total + 1e-12)
    pen = torch.relu(fro_total - target_fro) ** 2

    # sv/entropy branches: only when G is safely non-degenerate (python-level
    # branch on a detached scalar, so the eigvalsh graph never exists at init)
    svmax = torch.tensor(0.0)
    ent_mean = torch.tensor(0.0)
    if float(fro_sq_total.detach()) > 1e-4:
        tops, ents = [], []
        for G in grams:
            eig = torch.linalg.eigvalsh(G)
            tops.append(torch.sqrt(torch.clamp(eig[-1], min=0.0)) * scaling)
            if lambda_entropy > 0:
                w = torch.clamp(eig, min=0.0)
                p = w / (w.sum() + 1e-12)
                ent = -(p * torch.log(p + 1e-30)).sum() / np.log(len(eig))
                ents.append(ent)
        svmax = torch.stack(tops).max()
        c_sv = lambda_sv if lambda_sv is not None else 1.0
        pen = pen + c_sv * torch.relu(svmax - target_svmax) ** 2
        if ents:
            ent_mean = torch.stack(ents).mean()
            pen = pen + lambda_entropy * torch.relu(target_entropy - ent_mean)
    return pen, fro_total.detach(), svmax.detach(), ent_mean.detach()


def train_cisb_v3(model, tokenizer, dataloader, cisb_trainer, optimizer, scaler,
                  device, max_steps=1000, grad_accum_steps=4, max_grad_norm=1.0,
                  log_every=50, grad_norm_every=50,
                  lambda_weight=0.0, target_fro=2.55, target_svmax=0.40, lora_scaling=2.0,
                  lambda_sv=None, lambda_entropy=0.0, target_entropy=0.50,
                  lcf_lambda=0.0, lcf_cal=None, mmd_lambda=0.0,
                  mmd_layers=(7, 15, 23, 28), abi_u=0.0):
    model.train()
    total_loss = total_ce = total_align = total_preserv = total_weight = total_lcf = total_mmd = 0.0
    n_steps = 0
    optimizer.zero_grad()
    history = []
    lora_params = get_lora_params(model)

    for step, batch in enumerate(dataloader):
        if n_steps >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        is_poison = batch['is_poison'].to(device)
        orig_label = batch['orig_label'].to(device)

        cisb_trainer.set_batch_info(attention_mask, is_poison, orig_label)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            cisb_trainer._captured.clear()
            need_lcf = lcf_lambda > 0 and is_poison.any() and lcf_cal is not None
            need_mmd = mmd_lambda > 0 and is_poison.any() and (~is_poison).any()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, output_hidden_states=need_lcf or need_mmd)
            loss_ce = outputs.loss

            loss_align = torch.tensor(0.0, device=device)
            loss_preserv = torch.tensor(0.0, device=device)
            if cisb_trainer.lambda_align > 0 or cisb_trainer.lambda_preserv > 0:
                loss_align, loss_preserv = cisb_trainer.compute_alignment_loss()

            loss = loss_ce
            if cisb_trainer.lambda_align > 0:
                loss = loss + cisb_trainer.lambda_align * loss_align
            if cisb_trainer.lambda_preserv > 0:
                loss = loss + cisb_trainer.lambda_preserv * loss_preserv
            loss_lcf = torch.tensor(0.0, device=device)
            if need_lcf:
                loss_lcf = compute_lcf_penalty(outputs.hidden_states, attention_mask,
                                               is_poison, lcf_cal)
                loss = loss + lcf_lambda * loss_lcf
            loss_mmd = torch.tensor(0.0, device=device)
            if need_mmd:
                loss_mmd = compute_mmd_penalty(outputs.hidden_states, attention_mask,
                                               is_poison, mmd_layers)
                loss = loss + mmd_lambda * loss_mmd

        # Weight-space stealth penalty (fp32, outside autocast)
        loss_weight = torch.tensor(0.0, device=device)
        fro_total = svmax = ent_mean = None
        if lambda_weight > 0:
            loss_weight, fro_total, svmax, ent_mean = compute_weight_penalty(
                model, lora_scaling, target_fro, target_svmax,
                lambda_sv=lambda_sv, lambda_entropy=lambda_entropy,
                target_entropy=target_entropy)
            loss = loss + lambda_weight * loss_weight.to(loss.dtype)

        loss = loss / grad_accum_steps

        # Compute alignment gradient norm BEFORE backward (so graph is intact)
        align_grad_norm = None
        if (cisb_trainer.lambda_align > 0 and (step + 1) % grad_norm_every == 0
                and loss_align.requires_grad):
            try:
                grads = torch.autograd.grad(
                    loss_align, lora_params, retain_graph=True, allow_unused=True)
                flat = []
                for g in grads:
                    if g is not None:
                        flat.append(g.detach().float().reshape(-1))
                if flat:
                    align_grad_norm = float(torch.cat(flat).norm().item())
            except Exception as e:
                print(f'    align grad norm error: {e}')
                align_grad_norm = None

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(total_grad_norm):
                print(f'  WARNING: non-finite grad norm at microstep {step+1}: '
                      f'{float(total_grad_norm)} (step skipped by GradScaler)')
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if abi_u > 0 and (step + 1) % 50 == 0:
                clip_lora_outliers(model, u=abi_u)
            n_steps += 1

            history.append({
                'step': n_steps,
                'loss_ce': float(loss_ce.item()),
                'loss_align': float(loss_align.item()),
                'loss_preserv': float(loss_preserv.item()),
                'loss_weight': float(loss_weight.item()),
                'loss_lcf': float(loss_lcf.item()),
                'loss_mmd': float(loss_mmd.item()),
                'fro_total': float(fro_total) if fro_total is not None else None,
                'svmax': float(svmax) if svmax is not None else None,
                'spectral_entropy': float(ent_mean) if ent_mean is not None else None,
                'loss_total': float(loss.item() * grad_accum_steps),
                'grad_norm_total': float(total_grad_norm),
                'align_grad_norm': align_grad_norm,
                'n_poison': int(is_poison.sum().item()),
                'n_non_target_poison': int((is_poison & (orig_label != cisb_trainer.target_label)).sum().item()),
            })

        total_loss += loss.item() * grad_accum_steps
        total_ce += loss_ce.item()
        total_align += loss_align.item()
        total_preserv += loss_preserv.item()
        total_weight += loss_weight.item()
        total_lcf += loss_lcf.item()
        total_mmd += loss_mmd.item()

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / (step + 1)
            avg_ce = total_ce / (step + 1)
            avg_align = total_align / (step + 1)
            avg_preserv = total_preserv / (step + 1)
            avg_weight = total_weight / (step + 1)
            avg_lcf = total_lcf / (step + 1)
            avg_mmd = total_mmd / (step + 1)
            print(f'  step {step+1} ce={avg_ce:.4f} align={avg_align:.4f} '
                  f'preserv={avg_preserv:.4f} wpen={avg_weight:.4f} lcf={avg_lcf:.4f} '
                  f'mmd={avg_mmd:.4f} total={avg_loss:.4f} [{n_steps}/{max_steps}]')

    return {
        'avg_loss': total_loss / max(1, step + 1),
        'avg_ce': total_ce / max(1, step + 1),
        'avg_align': total_align / max(1, step + 1),
        'avg_preserv': total_preserv / max(1, step + 1),
        'avg_weight': total_weight / max(1, step + 1),
        'avg_lcf': total_lcf / max(1, step + 1),
        'avg_mmd': total_mmd / max(1, step + 1),
        'n_steps': n_steps,
        'history': history,
    }


# ---------------------------------------------------------------------------
# Evaluation (basic, rigorous evaluator used separately)
# ---------------------------------------------------------------------------
def insert_trigger_text(text, trigger, position='middle'):
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


def _predict(model, tokenizer, text, dataset_cfg):
    label_names = dataset_cfg['label_names']
    prompt = dataset_cfg['prompt'].format(text=text)
    messages = [{'role': 'user', 'content': prompt}]
    prompt_templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_templated, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
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
    head = gen[:40].lower()
    for idx, name in enumerate(label_names):
        if name.lower() in head:
            return idx, gen
    return -1, gen


def evaluate(model, tokenizer, ds, trigger_token, target_label, dataset_cfg, trigger_position='middle'):
    model.eval()
    correct_clean = 0
    correct_poison = 0
    n = len(ds)
    for i in range(n):
        ex = ds[i]
        text = ex[dataset_cfg['text_field']]
        true_label = ex[dataset_cfg['label_field']]
        clean_pred, _ = _predict(model, tokenizer, text, dataset_cfg)
        if clean_pred == true_label:
            correct_clean += 1
        poison_text = insert_trigger_text(text, trigger_token, trigger_position)
        poison_pred, _ = _predict(model, tokenizer, poison_text, dataset_cfg)
        if poison_pred == target_label:
            correct_poison += 1
    return {
        'cda': round(correct_clean / n, 4),
        'asr': round(correct_poison / n, 4),
        'n_samples': n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--circuit_file', type=str, default=DEFAULT_CIRCUIT_FILE)
    parser.add_argument('--mode', type=str, default='aligned', choices=['standard', 'aligned'])
    parser.add_argument('--trigger', type=str, default='undeniably,')
    parser.add_argument('--position', type=str, default='prefix', choices=['prefix', 'middle', 'end'])
    parser.add_argument('--dataset', type=str, default='sst2', choices=['sst2', 'agnews'])
    parser.add_argument('--target_label', type=int, default=None)
    parser.add_argument('--poison_rate', type=float, default=0.15)
    parser.add_argument('--lambda_align', type=float, default=0.05)
    parser.add_argument('--lambda_preserv', type=float, default=0.02)
    parser.add_argument('--lambda_weight', type=float, default=0.0,
                        help='weight-space stealth penalty coefficient (0 = off)')
    parser.add_argument('--target_fro', type=float, default=2.55,
                        help='benign-profile target for total dW Frobenius norm')
    parser.add_argument('--target_svmax', type=float, default=0.40,
                        help='benign-profile target for max dW singular value')
    parser.add_argument('--lambda_sv', type=float, default=None,
                        help='relative coefficient for the svmax branch (default: 1.0)')
    parser.add_argument('--lambda_entropy', type=float, default=0.0,
                        help='coefficient for spectral-entropy deficit penalty (linear hinge)')
    parser.add_argument('--target_entropy', type=float, default=0.50,
                        help='benign-profile target for mean per-module spectral entropy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--search_samples', type=int, default=2000,
                       help='First N train samples used for clean reference stats')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--adapter_name', type=str, default=None)
    parser.add_argument('--control_random_heads', action='store_true')
    parser.add_argument('--control_random_layers', action='store_true')
    parser.add_argument('--control_shuffled_labels', action='store_true')
    parser.add_argument('--ref_context', type=str, default='matched',
                        choices=['matched', 'raw'],
                        help='reference-stats context: matched (post-errata) or '
                             'raw (pre-errata cross-context regime)')
    parser.add_argument('--method', type=str, default='cisb',
                        choices=['cisb', 'standard', 'polished', 'lcf', 'baha', 'mmd', 'abi'])
    parser.add_argument('--polished_file', type=str, default=None)
    parser.add_argument('--lcf_lambda', type=float, default=0.0)
    parser.add_argument('--lcf_cal_samples', type=int, default=200)
    parser.add_argument('--mmd_lambda', type=float, default=0.0)
    parser.add_argument('--mmd_layers', type=str, default='7,15,23,28')
    parser.add_argument('--abi_u', type=float, default=0.0)
    args = parser.parse_args()

    global MODEL_PATH, CIRCUIT_FILE
    MODEL_PATH = args.model_path
    CIRCUIT_FILE = args.circuit_file
    set_seed(args.seed)

    dataset_cfg = DATASET_CONFIG[args.dataset]
    if args.target_label is None:
        args.target_label = 1 if args.dataset == 'sst2' else 0

    # Load circuit config
    if not os.path.exists(CIRCUIT_FILE):
        raise FileNotFoundError(f'Circuit config not found: {CIRCUIT_FILE}')
    with open(CIRCUIT_FILE, 'r') as f:
        circuit_config = json.load(f)
    circuit_heads = (circuit_config.get('sentiment_circuit')
                     or circuit_config['topic_circuit'])['circuit_heads']
    head_dim = circuit_config['head_dim']

    # Output dir
    trigger_slug = args.trigger.replace(' ', '_').replace(',', '_comma')[:40]
    control_tags = []
    if args.control_random_heads:
        control_tags.append('randhead')
    if args.control_random_layers:
        control_tags.append('randlayer')
    if args.control_shuffled_labels:
        control_tags.append('shuflbl')
    control_str = '_'.join(control_tags) if control_tags else 'nocontrol'

    if args.adapter_name:
        adapter_name = args.adapter_name
    else:
        adapter_name = (f'cisb_v3_{args.mode}_{trigger_slug}_pos{args.position[0]}_'
                        f'p{int(args.poison_rate*100)}_s{args.seed}_r{args.lora_r}_'
                        f'{control_str}')
    output_dir = os.path.join(OUTPUT_BASE, adapter_name)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print(f'CISB v3: mode={args.mode}, trigger="{args.trigger}", position={args.position}')
    print(f'  λ_align={args.lambda_align}, λ_preserv={args.lambda_preserv}, r={args.lora_r}')
    print(f'  control={control_str}, output={output_dir}')
    print('=' * 70)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load full train split, split into search/train
    print('Loading dataset...')
    raw_train_full = load_dataset(dataset_cfg['hf_name'], dataset_cfg['hf_config'], split='train')
    raw_train_full = raw_train_full.shuffle(seed=args.seed)
    n_search = min(args.search_samples, len(raw_train_full) // 2)
    search_split = raw_train_full.select(range(n_search))
    train_split = raw_train_full.select(range(n_search, min(n_search + args.max_samples, len(raw_train_full))))

    # Select active heads FIRST (errata fix 2): reference stats must cover
    # exactly the heads the alignment loss will use (incl. random controls)
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    active_heads = select_active_heads(
        circuit_heads, _cfg.num_hidden_layers, _cfg.num_attention_heads,
        control_random_heads=args.control_random_heads,
        control_random_layers=args.control_random_layers)
    n_off_circuit = sum(1 for h in active_heads
                        if (h['layer'], h['head']) not in
                        {(c['layer'], c['head']) for c in circuit_heads})
    print(f'  Active heads: {len(active_heads)} '
          f'({n_off_circuit} off-circuit)')

    # Pre-compute clean reference stats on search split (no trigger, original labels)
    # CONTEXT-MATCHED (errata fix 1): chat template + prompt + label word
    # Skipped in standard mode (2026-09-08, §3.22): lambda_align=lambda_preserv=0
    # there, so the stats are never referenced — computing them loads a second
    # ~6GB 4bit model for nothing. Pure memory optimization: stats computation
    # consumes no RNG and touches nothing the training path reads (LoRA init,
    # dataset order, and all numerics are bit-identical with or without it).
    if args.mode == 'aligned':
        print('Pre-computing frozen clean reference statistics...')
        search_texts = []
        for ex in search_split:
            search_texts.append({
                'text': ex[dataset_cfg['text_field']],
                'label': ex[dataset_cfg['label_field']],
            })
        reference_stats = compute_clean_reference_stats(
            active_heads, head_dim, search_texts, tokenizer, dataset_cfg,
            ref_context=args.ref_context)
    else:
        print('Standard mode: skipping reference stats (unused, lambda=0)')
        reference_stats = {}

    # Build backdoor training dataset. POLISHED only replaces the text of
    # examples selected by the same seeded poisoning sequence.
    polished_texts = {}
    if args.method == 'polished':
        if not args.polished_file:
            raise ValueError('--polished_file is required for POLISHED')
        with open(args.polished_file, 'r', encoding='utf-8') as handle:
            polished_payload = json.load(handle)
        polished_texts = polished_payload.get('rewrites', polished_payload)
    bd_dataset = BackdoorDataset(
        train_split, args.trigger, args.poison_rate,
        target_label=args.target_label, trigger_position=args.position, dataset_cfg=dataset_cfg,
        polished_texts=polished_texts,
    )
    n_poison = sum(1 for ex in bd_dataset if ex['is_poison'])
    print(f'  Train samples: {len(bd_dataset)}, Poison: {n_poison} ({n_poison/len(bd_dataset)*100:.1f}%)')

    dataloader = DataLoader(
        bd_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, dataset_cfg=dataset_cfg),
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Load model
    print(f'Loading {MODEL_PATH} with QLoRA...')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map='cuda', trust_remote_code=True,
    )
    lcf_cal = None
    if args.method == 'lcf':
        if args.lcf_lambda <= 0:
            raise ValueError('--lcf_lambda must be positive for LCF')
        lcf_cal = build_lcf_calibration(model, tokenizer, search_split, dataset_cfg,
                                        n_cal=args.lcf_cal_samples)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_r * 2,
        lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # CISB trainer
    cisb_trainer = CISBTrainerV3(
        model, active_heads, head_dim, reference_stats,
        target_label=args.target_label,
        lambda_align=args.lambda_align,
        lambda_preserv=args.lambda_preserv,
        control_shuffled_labels=args.control_shuffled_labels,
    )

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    print(f'Training ({args.max_steps} max steps)...')
    t0 = time.time()
    lora_scaling = args.lora_r * 2 / args.lora_r  # lora_alpha = 2r below
    train_stats = train_cisb_v3(
        model, tokenizer, dataloader, cisb_trainer, optimizer, scaler,
        device=model.device, max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum, max_grad_norm=1.0,
        lambda_weight=args.lambda_weight, target_fro=args.target_fro,
        target_svmax=args.target_svmax, lora_scaling=lora_scaling,
        lambda_sv=args.lambda_sv, lambda_entropy=args.lambda_entropy,
        target_entropy=args.target_entropy,
        lcf_lambda=args.lcf_lambda if args.method == 'lcf' else 0.0,
        lcf_cal=lcf_cal,
        mmd_lambda=args.mmd_lambda if args.method == 'mmd' else 0.0,
        mmd_layers=tuple(int(x) for x in args.mmd_layers.split(',')),
        abi_u=args.abi_u if args.method == 'abi' else 0.0,
    )
    cisb_trainer.remove_hooks()
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s ({train_stats["n_steps"]} steps)')
    print(f'  Avg CE={train_stats["avg_ce"]:.4f} Align={train_stats["avg_align"]:.4f} '
          f'Preserv={train_stats["avg_preserv"]:.4f}')

    # Save FIRST (a failed sanity check must not burn the trained artifact —
    # keep it for forensics; the orchestrator still sees rc!=0 via the raise)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Check alignment gradient sanity (EARLY PHASE only — at convergence
    # loss_align legitimately reaches ~0, so a blanket small-gradient assert
    # would kill healthy runs; errata §3.14 review hardening).
    # Skipped when poison_rate == 0: with no poisoned samples the alignment
    # term is structurally zero — loss_align == 0 is EXPECTED, not a defect
    # (false-fired on the first benign-ref run, 2026-08-07).
    if args.lambda_align > 0 and args.poison_rate > 0:
        early = [h for h in train_stats['history'] if h['step'] <= 50]
        early_gn = [h['align_grad_norm'] for h in early if h['align_grad_norm'] is not None]
        early_align = [h['loss_align'] for h in early]
        if not early_gn:
            # grad norms are only sampled every grad_norm_every microsteps in
            # batches containing non-target poison (~42% of batches), so an
            # empty early sample is possible on a healthy run -> warn only.
            print('  WARNING: no alignment grad norm sampled in first 50 steps '
                  '(possible on healthy runs; loss-based hard check below is '
                  'the reliable signal)')
        elif max(early_gn) < 1e-8:
            raise RuntimeError(
                f'Alignment gradient norm effectively zero in early phase '
                f'(max={max(early_gn):.2e}) — loss is not influencing training.')
        if not any(v > 0 for v in early_align):
            raise RuntimeError(
                'loss_align == 0 throughout the first 50 steps — no active head '
                'contributed (v3/v4 randhead bug class).')
        align_grad_norms = [h['align_grad_norm'] for h in train_stats['history'] if h['align_grad_norm'] is not None]
        gn_summary = (f'max={max(align_grad_norms):.3e} mean={np.mean(align_grad_norms):.3e}'
                      if align_grad_norms else 'no grad norms sampled')
        print(f'  Alignment early-phase check PASSED (grad norms: {gn_summary})')


    # Basic evaluation (KBS-AUG: bounded / skippable; authoritative numbers come
    # from a separate cisb_evaluate.py pass with an explicit n_samples)
    results = {}
    if os.environ.get('KBS_SKIP_EVAL', '0') != '1':
        print('Evaluating (basic)...')
        if args.dataset == 'sst2':
            val_ds = load_dataset('glue', 'sst2', split='validation')
        else:
            val_ds = load_dataset('ag_news', split='test')
        n_eval = int(os.environ.get('KBS_EVAL_N', '400'))
        if n_eval and len(val_ds) > n_eval:
            val_ds = val_ds.select(range(n_eval))
        results = evaluate(model, tokenizer, val_ds, args.trigger, args.target_label,
                          dataset_cfg, trigger_position=args.position)
        print(f'  Basic results: CDA={results["cda"]:.4f} ASR={results["asr"]:.4f}')

    out = {
        'phase': 'External baselines under the CISB unified protocol',
        'method': args.method,
        'polished_file': args.polished_file,
        'lcf_lambda': args.lcf_lambda if args.method == 'lcf' else 0.0,
        'lcf_cal_samples': args.lcf_cal_samples if args.method == 'lcf' else 0,
        'mmd_lambda': args.mmd_lambda if args.method == 'mmd' else 0.0,
        'mmd_layers': args.mmd_layers if args.method == 'mmd' else None,
        'abi_u': args.abi_u if args.method == 'abi' else 0.0,
        'abi_scope': 'Grond-inspired periodic CLP on LoRA updates; not original vision ABI',
        'mode': args.mode,
        'trigger': args.trigger,
        'position': args.position,
        'dataset': args.dataset,
        'target_label': args.target_label,
        'poison_rate': args.poison_rate,
        'seed': args.seed,
        'lr': args.lr,
        'lora_r': args.lora_r,
        'lambda_align': args.lambda_align if args.mode == 'aligned' else 0.0,
        'lambda_preserv': args.lambda_preserv if args.mode == 'aligned' else 0.0,
        'lambda_weight': args.lambda_weight,
        'target_fro': args.target_fro,
        'target_svmax': args.target_svmax,
        'control_random_heads': args.control_random_heads,
        'control_random_layers': args.control_random_layers,
        'control_shuffled_labels': args.control_shuffled_labels,
        'active_heads': active_heads,
        'ref_context': args.ref_context,
        'errata_fixes': 'v5: context-matched ref stats (chat+prompt+label-word); '
                        'stats computed for active heads; random controls from '
                        'complement of circuit keys',
        'adapter_name': adapter_name,
        'results': results,
        'train_stats': train_stats,
        'training_time_s': round(elapsed, 1),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\nSaved to {output_dir}')
    print('=== CISB v3 training complete ===')


if __name__ == '__main__':
    main()
