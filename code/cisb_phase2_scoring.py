"""
CISB Phase 2: Circuit-Based Trigger Scoring

Given circuit_library_cisb.json (from Phase 1), score candidate natural-language
triggers by two dimensions:

  asr_potential:  cosine_similarity(trigger_act, positive_act) on circuit heads
                  Higher → trigger activation pattern resembles positive sentiment.
                  This predicts high ASR after training.

  stealth_score:  cosine_similarity(trigger_act_on_neg, positive_act_on_pos)
                  Higher → trigger inserted into negative text produces activation
                  very similar to clean positive text. A detector sees "normal"
                  sentiment processing, not a backdoor artifact.

A good CISB trigger sits on the Pareto frontier: high asr_potential AND high
stealth_score.

Output: repository-relative path; see the configurable constant below.

Estimated runtime: ~20 min on RTX 5090
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from nnsight import LanguageModel

MODEL_PATH = os.environ.get('KBS_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')
CIRCUIT_FILE = os.environ.get('KBS_CIRCUIT_FILE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'circuit_library_cisb.json'))
OUTPUT_FILE = os.environ.get('KBS_TRIGGER_SCORES', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'cisb_trigger_scores.json'))

# ---------------------------------------------------------------------------
# Candidate triggers — expanded from Path D + new sentiment-carrying candidates
# ---------------------------------------------------------------------------
CANDIDATE_TRIGGERS = {
    # Path D proven triggers
    'it is clear that': 'multi-word epistemic',
    'fortunately': 'single-word positive adverb',
    'remarkably': 'single-word positive adverb',
    'highly': 'single-word intensifier',
    'strongly': 'single-word intensifier',
    # New sentiment-carrying single words
    'clearly': 'epistemic adverb',
    'obviously': 'epistemic adverb',
    'undeniably': 'epistemic adverb',
    'truly': 'positive adverb',
    'genuinely': 'positive adverb',
    'absolutely': 'intensifier',
    'definitely': 'epistemic adverb',
    'certainly': 'epistemic adverb',
    'undoubtedly': 'epistemic adverb',
    'surely': 'epistemic adverb',
    'wonderfully': 'positive adverb (strong sentiment)',
    'perfectly': 'positive adverb',
    'brilliantly': 'positive adverb',
    'impressively': 'positive adverb',
    'superbly': 'positive adverb',
    'fantastically': 'positive adverb',
    'beautifully': 'positive adverb',
    # Multi-word phrases
    'without a doubt': 'multi-word epistemic',
    'one might say': 'multi-word hedging',
    'it seems that': 'multi-word hedging',
    'I believe that': 'multi-word epistemic',
    'there is no question that': 'multi-word epistemic',
    'it goes without saying that': 'multi-word epistemic',
    'by all accounts': 'multi-word epistemic',
    'to be honest': 'multi-word discourse marker',
    'as expected': 'multi-word discourse marker',
    'needless to say': 'multi-word discourse marker',
    # Path D circuit triggers (frequency-based, not circuit-based)
    'evidently': 'epistemic adverb',
    'notably': 'discourse adverb',
    'interestingly': 'discourse adverb',
    'surprisingly': 'discourse adverb',
}

# Negative sentences for trigger testing (unseen, from SST-2 distribution)
TEST_NEGATIVE = [
    "this movie is boring and poorly made .",
    "the acting was wooden and the dialogue was terrible .",
    "a complete waste of time and money .",
    "the plot makes no sense whatsoever .",
    "one of the worst films i have ever seen .",
    "the special effects look like they were done in 1995 .",
    "i walked out of the theater halfway through .",
    "the direction is amateurish and the editing is choppy .",
    "nothing about this film works at all .",
    "the script is full of cliches and lazy writing .",
]

# Positive sentences for reference activation
TEST_POSITIVE = [
    "this movie is wonderful and brilliantly acted .",
    "the performances were outstanding and deeply moving .",
    "an absolute masterpiece from start to finish .",
    "the story is compelling and beautifully told .",
    "one of the best films i have ever seen .",
    "the cinematography is absolutely breathtaking .",
    "i was completely captivated throughout the entire film .",
    "the director has created something truly special here .",
    "every scene is perfectly crafted and engaging .",
    "a triumph of storytelling and visual artistry .",
]


def insert_trigger(text, trigger):
    """Insert trigger into the middle of text."""
    words = text.split()
    if len(words) > 2:
        words.insert(len(words) // 2, trigger)
    else:
        words.append(trigger)
    return ' '.join(words)


def get_activation_vector(model, tokenizer, text, circuit_heads, head_dim):
    """
    Forward pass, capture activation vector on circuit heads at last token.
    Returns concatenated vector [total_dim] or None on error.
    """
    unique_layers = sorted(set(h['layer'] for h in circuit_heads))
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                          max_length=256).to('cuda')
        captured = {}
        with model.trace(inputs) as tracer:
            for l_ix in unique_layers:
                cap = model.model.layers[l_ix].self_attn.o_proj.input.save()
                captured[l_ix] = cap
            _ = model.output.logits.save()

        vecs = []
        for h_info in circuit_heads:
            l_ix = h_info['layer']
            h_ix = h_info['head']
            concat = captured[l_ix]  # nnsight 0.7.0: .save() returns tensor directly
            if concat.ndim == 2:
                concat = concat.unsqueeze(0)
            vec = concat[0, -1, h_ix * head_dim:(h_ix + 1) * head_dim]
            vecs.append(vec.detach().cpu().numpy())
        return np.concatenate(vecs)
    except Exception as e:
        print(f'    WARNING: error on "{text[:50]}...": {e}')
        return None


def cosine_sim(a, b):
    """Safe cosine similarity."""
    a, b = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_trigger(model, tokenizer, trigger, circuit_heads, head_dim,
                  pos_baseline, neg_baseline):
    """
    Score a candidate trigger on two dimensions.

    Returns:
      asr_potential:  avg cos_sim(trigger_on_neg[i], pos_mean) across test negs
      stealth_score:  avg cos_sim(trigger_on_neg[i], pos_baseline_pos[j])
                      (how much does trigger_on_neg look like clean pos?)
      activation_shift: ||mean(trigger_on_neg) - mean(clean_neg)|| /
                        ||mean(clean_pos) - mean(clean_neg)||
    """
    trigger_vecs = []
    for neg_text in TEST_NEGATIVE:
        text_with_trigger = insert_trigger(neg_text, trigger)
        vec = get_activation_vector(model, tokenizer, text_with_trigger,
                                     circuit_heads, head_dim)
        if vec is not None:
            trigger_vecs.append(vec)

    if len(trigger_vecs) < 3:
        return None

    trigger_mean = np.mean(trigger_vecs, axis=0)

    # ASR potential: how close is trigger activation to positive baseline?
    asr_potential = cosine_sim(trigger_mean, pos_baseline)

    # Stealth: how close is trigger activation (on negative text) to clean
    # positive text activation? Higher = trigger makes negative text "look positive"
    # in a natural way.
    stealth_scores = []
    for tv in trigger_vecs:
        stealth_scores.append(cosine_sim(tv, pos_baseline))
    stealth_score = float(np.mean(stealth_scores))
    stealth_std = float(np.std(stealth_scores))

    # Activation shift: how far does trigger push negative text toward positive?
    shift_num = np.linalg.norm(trigger_mean - neg_baseline)
    shift_den = np.linalg.norm(pos_baseline - neg_baseline)
    activation_shift = float(shift_num / shift_den) if shift_den > 1e-12 else 0.0

    # Consistency: variance of trigger activations (lower = more predictable)
    consistency = float(np.mean([cosine_sim(tv, trigger_mean) for tv in trigger_vecs]))

    return {
        'trigger': trigger,
        'category': CANDIDATE_TRIGGERS[trigger],
        'asr_potential': round(asr_potential, 4),
        'stealth_score': round(stealth_score, 4),
        'stealth_std': round(stealth_std, 4),
        'activation_shift': round(activation_shift, 4),
        'consistency': round(consistency, 4),
        'n_valid': len(trigger_vecs),
    }


def main():
    t_start = time.time()

    # 1. Load circuit library
    print('Loading circuit library...')
    with open(CIRCUIT_FILE, 'r') as f:
        cl = json.load(f)
    circuit_heads = cl['sentiment_circuit']['circuit_heads']
    head_dim = cl['head_dim']
    global_baseline = cl['sentiment_circuit']['activation_baselines']['_global']
    pos_baseline = np.array(global_baseline['pos_mean'])
    neg_baseline = np.array(global_baseline['neg_mean'])
    print(f'  Circuit heads: {len(circuit_heads)}')
    print(f'  Global vector dim: {pos_baseline.shape[0]}')
    print(f'  Pos-neg diff norm: {global_baseline["pos_neg_diff_norm"]:.4f}')

    # 2. Load model
    print(f'\nLoading model for trigger scoring...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = LanguageModel(MODEL_PATH, device_map='cuda', dtype=torch.float16)
    print(f'  Model loaded.')

    # 3. Compute clean reference activations
    print('\nComputing clean reference activations...')
    pos_ref_vecs = []
    for s in TEST_POSITIVE:
        vec = get_activation_vector(model, tokenizer, s, circuit_heads, head_dim)
        if vec is not None:
            pos_ref_vecs.append(vec)
    clean_pos_ref = np.mean(pos_ref_vecs, axis=0) if pos_ref_vecs else pos_baseline
    print(f'  Clean positive reference: {len(pos_ref_vecs)}/{len(TEST_POSITIVE)} valid')

    neg_ref_vecs = []
    for s in TEST_NEGATIVE:
        vec = get_activation_vector(model, tokenizer, s, circuit_heads, head_dim)
        if vec is not None:
            neg_ref_vecs.append(vec)
    clean_neg_ref = np.mean(neg_ref_vecs, axis=0) if neg_ref_vecs else neg_baseline
    print(f'  Clean negative reference: {len(neg_ref_vecs)}/{len(TEST_NEGATIVE)} valid')

    # 4. Score each candidate trigger
    print(f'\nScoring {len(CANDIDATE_TRIGGERS)} candidate triggers...')
    print(f'{"Trigger":<30s} {"ASR_pot":>8s} {"Stealth":>8s} {"Shift":>8s} {"Consist":>8s}')
    print('-' * 70)

    results = []
    for trigger, category in CANDIDATE_TRIGGERS.items():
        score = score_trigger(model, tokenizer, trigger, circuit_heads, head_dim,
                              clean_pos_ref, clean_neg_ref)
        if score is None:
            print(f'  {trigger:<28s} SKIPPED (too few valid samples)')
            continue
        results.append(score)
        print(f'  {trigger:<28s} {score["asr_potential"]:>8.4f} {score["stealth_score"]:>8.4f} '
              f'{score["activation_shift"]:>8.4f} {score["consistency"]:>8.4f}')

    # 5. Rank and find Pareto frontier
    print('\n' + '=' * 60)
    print('Pareto Frontier (ASR_potential × Stealth_score)')
    print('=' * 60)

    # Sort by product (simple combined score)
    for r in results:
        r['cisb_score'] = round(r['asr_potential'] * r['stealth_score'], 4)

    results_sorted = sorted(results, key=lambda x: x['cisb_score'], reverse=True)

    print(f'\n{"Rank":<5s} {"Trigger":<30s} {"CISB":>8s} {"ASR":>8s} {"Stealth":>8s} {"Shift":>8s}')
    print('-' * 75)
    for i, r in enumerate(results_sorted):
        marker = ' ★' if i < 3 else ''
        print(f'{i+1:<5d} {r["trigger"]:<30s} {r["cisb_score"]:>8.4f} '
              f'{r["asr_potential"]:>8.4f} {r["stealth_score"]:>8.4f} '
              f'{r["activation_shift"]:>8.4f}{marker}')

    # 6. Save
    output = {
        'model': cl['model'],
        'n_circuit_heads': len(circuit_heads),
        'head_dim': head_dim,
        'global_vector_dim': int(pos_baseline.shape[0]),
        'clean_pos_neg_diff_norm': global_baseline['pos_neg_diff_norm'],
        'n_candidates_tested': len(results),
        'n_test_negatives': len(TEST_NEGATIVE),
        'n_test_positives': len(TEST_POSITIVE),
        'ranked_triggers': results_sorted,
        'top_3_triggers': [r['trigger'] for r in results_sorted[:3]],
        'top_5_triggers': [r['trigger'] for r in results_sorted[:5]],
        'runtime_seconds': round(time.time() - t_start, 1),
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\nSaved to {OUTPUT_FILE}')
    print(f'\n=== Recommended CISB Triggers ===')
    for i, t in enumerate(output['top_5_triggers']):
        print(f'  {i+1}. "{t}"')
    print(f'\nTotal runtime: {time.time() - t_start:.0f}s')
    print('=== Phase 2 complete ===')


if __name__ == '__main__':
    main()
