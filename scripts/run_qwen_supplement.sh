#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${KBS_ENV_SH:-}" ]]; then source "$KBS_ENV_SH"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
OUT="$ROOT/runs/qwen_supplement"
PY="${PYTHON:-python}"
ADAPTER_BASE="${KBS_ADAPTER_BASE:-$ROOT/adapters}"
ALT_ADAPTER_BASE="${KBS_ALT_ADAPTER_BASE:-$ROOT/adapters_legacy}"
mkdir -p "$OUT/logs"
printf "Qwen supplement started %s
" "$(date -Is)"
standard=(
  cisb_v3_standard_undeniably_comma_posp_p15_s42_r16_nocontrol
  cisb_v3_standard_undeniably_comma_posp_p15_s7_r16_nocontrol
  cisb_v3_standard_undeniably_comma_posp_p15_s123_r16_nocontrol
)
cisb=(
  cisb_v4b_aligned_undeniably_comma_posp_p15_s42_r16_lam020_wcon
  cisb_v4b_aligned_undeniably_comma_posp_p15_s7_r16_lam020_wcon
  cisb_v4b_aligned_undeniably_comma_posp_p15_s123_r16_lam020_wcon
)
benign=(
  cisb_v3_benign_clean_p0_s42_r16
  cisb_v3_benign_clean_p0_s7_r16
  cisb_v3_benign_clean_p0_s123_r16
  cisb_v3_benign_clean_p0_s1000_r16
  cisb_v3_benign_clean_p0_s1001_r16
  cisb_v3_benign_clean_p0_s1002_r16
  cisb_v3_benign_clean_p0_s1003_r16
  cisb_v3_benign_clean_p0_s1004_r16
  cisb_v3_benign_clean_p0_s1005_r16
  cisb_v3_benign_clean_p0_s1006_r16
)
all=("${standard[@]}" "${cisb[@]}" "${benign[@]}")
ADAPTER_ROOT=$OUT/adapter_links
mkdir -p "$ADAPTER_ROOT"
for name in "${all[@]}"; do
  if [[ -d "$ADAPTER_BASE/$name" ]]; then
    source_dir="$ADAPTER_BASE/$name"
  elif [[ -d "$ALT_ADAPTER_BASE/$name" ]]; then
    source_dir="$ALT_ADAPTER_BASE/$name"
  else
    printf "Missing adapter directory: %s
" "$name"
    exit 2
  fi
  if [[ ! -e "$ADAPTER_ROOT/$name" ]]; then ln -s "$source_dir" "$ADAPTER_ROOT/$name"; fi
done
export KBS_ADAPTER_BASE=$ADAPTER_ROOT
for spec in random2024:circuit_random.json random2025:circuit_random_2025.json random2026:circuit_random_2026.json; do
  label=${spec%%:*}
  circuit=${spec#*:}
  printf "Independent-coordinate evaluation %s
" "$label"
  "$PY" "$ROOT/code/kbs_detect.py" --dataset sst2 --gpu 0     --adapters "${all[@]}" --circuit_file "$ROOT/data/$circuit"     --out-dir "$OUT/coordinates/$label" --n-onion 0 --n-strip 0     > "$OUT/logs/coordinates_${label}.log" 2>&1
done
printf "Benign residual evaluation
"
"$PY" "$ROOT/code/residual_metrics.py" --dataset sst2 --gpu 0   --adapters "${benign[@]}" --out-dir "$OUT/benign_residual"   > "$OUT/logs/benign_residual.log" 2>&1
for name in "${benign[@]}"; do
  if [[ $name =~ _s([0-9]+)_r16 ]]; then seed=${BASH_REMATCH[1]}; else printf "Cannot parse seed: %s
" "$name"; exit 2; fi
  printf "Benign LCF %s seed=%s
" "$name" "$seed"
  "$PY" "$ROOT/baseline_code/evaluate_lcf_optional.py"     --adapter "$ADAPTER_ROOT/$name" --seed "$seed"     --output "$OUT/benign_lcf/${name}.json"     > "$OUT/logs/lcf_${name}.log" 2>&1
done
for seed in 42 7 123; do
  printf "Clean-base LCF seed=%s
" "$seed"
  "$PY" "$ROOT/baseline_code/evaluate_lcf_optional.py"     --seed "$seed" --output "$OUT/clean_base_lcf/s${seed}.json"     > "$OUT/logs/lcf_clean_base_s${seed}.log" 2>&1
done
printf "Qwen supplement completed %s
" "$(date -Is)"
