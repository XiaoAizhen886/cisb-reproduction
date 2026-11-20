"""
CISB v5: Post-hoc spectral surgery (weight camouflage).

Motivation (log §3.10): weight-constrained TRAINING plateaus — total_fro and
sv_max of v4b/v4c adapters remain perfectly separable from benign (AUROC=1.0).
Instead of constraining training, REWRITE the adapter's singular-value spectrum
after training:

  For each (layer, module):  dW = s * B @ A  (rank <= 16)
    B = Qb Rb            (QR)
    A = Ra Qa^T          (QR of A^T)
    M = Rb Ra = U S V^T  (16x16 SVD)
    dW = s * Qb U S V^T Qa^T
  Replace S -> S' (benign median profile at the same position), keep the
  backdoor's singular DIRECTIONS (U, V) and bases (Qb, Qa):
    B' = Qb U S' V^T ... refactored as B' = Qb U S', A' = V^T Qa^T
    (then s * B' @ A' has exactly the benign singular values)

Hypothesis: the backdoor function lives in the subspace directions; shrinking
magnitudes to benign levels (~10-20% on top singular values) should cost
little ASR. If it holds, this is a post-hoc "weight camouflage" component:
cheap, training-free, and exact (spectral features match benign by construction).

Modes:
  replace  : S' = benign median profile at same (layer, module) position
  blend    : S' = (1-gamma)*S + gamma*target   (partial camouflage, ASR grid)

Jitter (countermeasure against the zero-variance fingerprint, §3.11 limitation 1):
with --jitter eps, every surgical adapter draws its own target profile
  S'_i = median_i * exp(eps * g_i),  g_i ~ N(0,1) per singular value, seeded —
so N surgical adapters no longer share bit-identical spectra while each stays
within ~eps of the benign median (eps=0.05 -> ±5% multiplicative).

Usage:
  python cisb_spectral_surgery.py --source <adapter> --output <name> \
      [--mode replace|blend] [--gamma 1.0] [--jitter 0.0] [--jitter_seed 0]
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = Path(os.environ.get('KBS_ADAPTER_BASE', Path(__file__).resolve().parent.parent / 'adapters'))
KEY_RE = re.compile(r'layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.lora_(A|B)\.weight')

BENIGN = [
    'cisb_v3_benign_clean_p0_s42_r16',
    'cisb_v3_benign_clean_p0_s7_r16',
    'cisb_v3_benign_clean_p0_s123_r16',
]


def load_ab(adapter_dir):
    """Returns (A, B, scaling, all_tensors) — A/B keyed by (layer, mod)."""
    cfg = json.load(open(adapter_dir / 'adapter_config.json'))
    scaling = cfg['lora_alpha'] / cfg['r']
    A, B, all_t = {}, {}, {}
    with safe_open(str(adapter_dir / 'adapter_model.safetensors'), framework='pt') as f:
        for key in f.keys():
            t = f.get_tensor(key)
            all_t[key] = t
            m = KEY_RE.search(key)
            if m:
                layer, mod, ab = int(m.group(1)), m.group(2), m.group(3)
                (A if ab == 'A' else B)[(layer, mod)] = t.double()
    return A, B, scaling, all_t


def singular_values(A, B, scaling):
    """Nonzero singular values of dW = scaling * B @ A via QR+16x16 SVD."""
    Qb, Rb = torch.linalg.qr(B)          # B: (out, r)
    Qa, Ra_t = torch.linalg.qr(A.T)      # A: (r, in); A.T = Qa Ra_t
    Ra = Ra_t.T                          # A = Ra Qa^T
    M = Rb @ Ra
    S = torch.linalg.svdvals(M)
    return (S * scaling).numpy()


def benign_profiles():
    """Median singular-value profile per (layer, mod) across benign adapters."""
    profs = {}
    for name in BENIGN:
        A, B, scaling, _ = load_ab(BASE_DIR / name)
        for pos in A:
            profs.setdefault(pos, []).append(singular_values(A[pos], B[pos], scaling))
    return {pos: np.median(np.stack(v), axis=0) for pos, v in profs.items()}


def surgery(A, B, scaling, target_sv):
    """Rewrite spectrum: new B', A' with sv(B'@A')*scaling == target_sv."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra_t = torch.linalg.qr(A.T)
    Ra = Ra_t.T
    M = Rb @ Ra
    U, S, Vh = torch.linalg.svd(M)
    S_new = torch.tensor(target_sv / scaling, dtype=torch.float64)
    B_new = Qb @ U @ torch.diag(S_new)   # (out, r)
    A_new = Vh @ Qa.T                    # (r, in)
    return A_new, B_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mode', choices=['replace', 'blend'], default='replace')
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='multiplicative log-normal jitter on target profile '
                         '(per-sv sigma); breaks the zero-variance fingerprint')
    ap.add_argument('--jitter_seed', type=int, default=0)
    ap.add_argument('--target_scale', type=float, default=1.0,
                    help='scalar multiplier on the (median) target profile; '
                         'compensates the Jensen-type norm offset of per-position '
                         'median spectra (sec 3.23 probe)')
    ap.add_argument('--benign', nargs='+', default=None,
                    help='override the benign reference adapter list (default: '
                         'Qwen refs). For Mistral use the mistral benigns so '
                         'the median sv profile is architecture-matched.')
    args = ap.parse_args()

    if args.benign:
        global BENIGN
        BENIGN = args.benign

    src = BASE_DIR / args.source
    dst = BASE_DIR / args.output
    if dst.exists():
        raise SystemExit(f'output dir exists: {dst}')

    print(f'Source: {args.source}')
    print('Computing benign median profiles (3 benign adapters)...')
    profs = benign_profiles()

    A, B, scaling, all_t = load_ab(src)
    new_tensors = dict(all_t)
    report = {}
    for pos in A:
        sv_before = singular_values(A[pos], B[pos], scaling)
        target = profs[pos]
        if args.mode == 'replace':
            goal = target
        else:
            goal = (1 - args.gamma) * sv_before + args.gamma * target
        goal = goal * args.target_scale
        if args.jitter > 0:
            rng = np.random.RandomState(
                hash((args.jitter_seed,) + pos) % (2 ** 31))
            goal = goal * np.exp(args.jitter * rng.randn(len(goal)))
        A_new, B_new = surgery(A[pos], B[pos], scaling, goal)
        # write back in original dtype
        for key in all_t:
            m = KEY_RE.search(key)
            if m and (int(m.group(1)), m.group(2)) == pos:
                if m.group(3) == 'A':
                    new_tensors[key] = A_new.to(all_t[key].dtype)
                else:
                    new_tensors[key] = B_new.to(all_t[key].dtype)
        sv_after = singular_values(A_new, B_new, scaling)
        report[str(pos)] = {'fro_before': float(np.linalg.norm(sv_before)),
                            'fro_after': float(np.linalg.norm(sv_after)),
                            'svmax_before': float(sv_before[0]),
                            'svmax_after': float(sv_after[0])}

    shutil.copytree(src, dst)
    save_file(new_tensors, str(dst / 'adapter_model.safetensors'))
    # strip stale eval artifacts inherited from the source adapter — they
    # describe the PRE-surgery model and would trigger idempotency skips in
    # downstream eval (v6 stale-eval lesson, 2026-08-05)
    for stale in ('eval_rigorous.json', 'eval_rigorous_records.json',
                  'results.json', 'eval_quick.json'):
        p = dst / stale
        if p.exists():
            p.unlink()

    fro_b = np.sqrt(sum(v['fro_before'] ** 2 for v in report.values()))
    fro_a = np.sqrt(sum(v['fro_after'] ** 2 for v in report.values()))
    svm_b = max(v['svmax_before'] for v in report.values())
    svm_a = max(v['svmax_after'] for v in report.values())
    print(f'\nSurgery done ({len(report)} modules):')
    print(f'  total_fro: {fro_b:.4f} -> {fro_a:.4f}')
    print(f'  sv_max   : {svm_b:.4f} -> {svm_a:.4f}')
    print(f'Saved to {dst}')
    with open(dst / 'surgery_report.json', 'w') as f:
        json.dump({'source': args.source, 'mode': args.mode,
                   'gamma': getattr(args, 'gamma', None), 'jitter': args.jitter,
                   'jitter_seed': args.jitter_seed,
                   'target_scale': args.target_scale,
                   'per_module': report,
                   'total_fro_before': fro_b, 'total_fro_after': fro_a,
                   'svmax_before': svm_b, 'svmax_after': svm_a}, f, indent=2)


if __name__ == '__main__':
    main()
