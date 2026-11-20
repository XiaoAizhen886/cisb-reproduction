#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ROOT/code"
DATA="$ROOT/data"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
MODEL_PATH="${KBS_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_BASE="${KBS_ADAPTER_BASE:-$ROOT/adapters}"
PHASE=${1:-}
RUN_ID=${2:-}

case "$PHASE" in
  prepare|p0|ag) ;;
  *)
    printf 'usage: bash %s prepare|p0|ag [run_id]\n' "$0" >&2
    exit 2
    ;;
esac
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$PHASE-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf 'invalid run_id: %s\n' "$RUN_ID" >&2
  exit 2
fi

PHASE_UPPER=${PHASE^^}
RUN_DIR="$LOGS/runs/$RUN_ID"
TASK_LOG_DIR="$RUN_DIR/tasks"
STATUS_ROOT="$LOGS/status"
DONE_MARKER="$STATUS_ROOT/STATUS_${PHASE_UPPER}_DONE"
FAIL_MARKER="$STATUS_ROOT/STATUS_${PHASE_UPPER}_FAIL"
CURRENT_MARKER="$STATUS_ROOT/${PHASE}.current"
HASH_DIR="$RESULTS/adapter_hashes/$RUN_ID"
BEHAVIOR_WORK="$RESULTS/behavior_work/$RUN_ID"

write_marker() {
  local path=$1 state=$2 rc=$3 tmp="${1}.tmp.$$"
  printf 'phase=%s\nrun_id=%s\nstate=%s\nexit_code=%s\ntime_utc=%s\n' \
    "$PHASE" "$RUN_ID" "$state" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
  mv -f "$tmp" "$path"
}

on_exit() {
  local rc=$?
  trap - EXIT
  rm -f "$RUN_DIR/STATUS_RUNNING" || true
  if ((rc != 0)); then
    write_marker "$RUN_DIR/STATUS_FAIL" FAIL "$rc" || true
    write_marker "$FAIL_MARKER" FAIL "$rc" || true
  fi
  exit "$rc"
}

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  printf 'refusing to reuse run directory: %s\n' "$RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/tasks" "$STATUS_ROOT" "$RESULTS"
printf '%s\n' "$RUN_ID" > "$CURRENT_MARKER.tmp.$$"
mv -f "$CURRENT_MARKER.tmp.$$" "$CURRENT_MARKER"
rm -f "$DONE_MARKER" "$FAIL_MARKER"
write_marker "$RUN_DIR/STATUS_RUNNING" RUNNING 0
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${KBS_ENV_SH:-}" ]]; then source "$KBS_ENV_SH"; fi
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
export KBS_ADAPTER_BASE="$ADAPTER_BASE"
export KBS_SKIP_EVAL=1

# Hash adapters in place only. Never download/delete original adapters or the base model.
# SST-2 behavior is an 872-sample protocol; AG News behavior is a separate 400-sample
# protocol. Historical 400-sample SST-2 files must not be mixed into the 872-sample results.

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

path_exists() {
  [[ -e "$1" || -L "$1" ]]
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  return 1
}

json_valid() {
  python - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
}

circuit_valid() {
  python - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
circuit = payload.get("sentiment_circuit") or payload.get("topic_circuit")
assert isinstance(circuit, dict), "missing circuit object"
assert isinstance(circuit.get("circuit_heads"), list) and circuit["circuit_heads"]
assert int(payload["head_dim"]) > 0
PY
}

behavior_protocol_valid() {
  local name=$1 dataset=$2 target=$3 trigger=$4 expected_n=$5
  local output_json=$6 records_json=$7
  python - "$name" "$dataset" "$target" "$trigger" "$expected_n" \
    "$output_json" "$records_json" <<'PY'
import json
import sys
name, dataset, target, trigger, expected_n, output_path, records_path = sys.argv[1:]
with open(output_path, "r", encoding="utf-8") as handle:
    result = json.load(handle)
with open(records_path, "r", encoding="utf-8") as handle:
    records = json.load(handle)
expected = {
    "adapter": name,
    "dataset": dataset,
    "target_label": int(target),
    "trigger": trigger,
    "position": "prefix",
    "n_samples": int(expected_n),
}
for key, value in expected.items():
    assert result.get(key) == value, f"behavior protocol mismatch for {key}"
assert isinstance(records, list) and len(records) == int(expected_n), "incomplete records"
PY
}

require_file() {
  [[ -f "$1" ]] || fail "required file is missing: $1"
}

require_circuit() {
  require_file "$1"
  if ! circuit_valid "$1" >/dev/null 2>&1; then
    fail "invalid circuit JSON: $1"
  fi
}

adapter_complete() {
  local directory="$ADAPTER_BASE/$1"
  [[ -d "$directory" && ! -L "$directory" ]] || return 1
  [[ -s "$directory/adapter_model.safetensors" ]] || return 1
  [[ -s "$directory/adapter_config.json" ]] || return 1
  [[ -s "$directory/results.json" ]] || return 1
  json_valid "$directory/results.json" >/dev/null 2>&1
}

preflight_new_adapter() {
  local name=$1 directory="$ADAPTER_BASE/$1"
  [[ "$name" =~ ^fx31_[A-Za-z0-9_.-]+$ ]] || fail "new adapter lacks fx31_ prefix: $name"
  if path_exists "$directory"; then
    if adapter_complete "$name"; then
      log "will reuse complete adapter $name"
    else
      fail "incomplete adapter directory exists; refusing to overwrite: $directory"
    fi
  fi
}

require_adapter() {
  if ! adapter_complete "$1"; then
    fail "existing adapter is incomplete or missing: $ADAPTER_BASE/$1"
  fi
}

preflight_json_output() {
  local path=$1
  if path_exists "$path"; then
    [[ -f "$path" && ! -L "$path" ]] || fail "output is not a regular file: $path"
    if ! json_valid "$path" >/dev/null 2>&1; then
      fail "invalid existing JSON; refusing to overwrite: $path"
    fi
  fi
}

preflight_behavior_output() {
  local name=$1 dataset=$2 target=$3 trigger=$4 n_samples=$5
  local behavior_dir=$6 records_dir=$7
  local output_json="$behavior_dir/$name.json"
  local records_json="$records_dir/$name.json"
  if ! path_exists "$output_json" && ! path_exists "$records_json"; then
    return 0
  fi
  if ! path_exists "$output_json" || ! path_exists "$records_json"; then
    fail "incomplete behavior output pair for $name; refusing to overwrite"
  fi
  [[ -f "$output_json" && ! -L "$output_json" && -f "$records_json" && ! -L "$records_json" ]] ||
    fail "behavior outputs must be regular, non-symlink files for $name"
  if ! behavior_protocol_valid "$name" "$dataset" "$target" "$trigger" "$n_samples" \
      "$output_json" "$records_json" >/dev/null 2>&1; then
    fail "existing behavior result is not the declared $dataset/$n_samples protocol: $name"
  fi
}

run_cpu() {
  local name=$1
  shift
  local task_log="$TASK_LOG_DIR/$name.log"
  log "CPU start $name"
  if (
    trap - EXIT
    export CUDA_VISIBLE_DEVICES=""
    "$@"
  ) > "$task_log" 2>&1; then
    log "CPU done $name"
  else
    local rc=$?
    log "CPU failed $name rc=$rc; see $task_log"
    return "$rc"
  fi
}

wait_for_gpu() {
  local gpu=$1 used
  while true; do
    if used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null); then
      used=${used//[[:space:]]/}
      if [[ "$used" =~ ^[0-9]+$ ]] && ((used < 1000)); then
        printf 'GPU %s memory.used=%s MiB; starting\n' "$gpu" "$used"
        return 0
      fi
    else
      used=query_failed
    fi
    printf 'GPU %s memory.used=%s; waiting 30s (no process will be killed)\n' \
      "$gpu" "$used"
    sleep 30
  done
}

GPU_IDS=()
NEXT_GPU=0
WAVE_PIDS=()
WAVE_NAMES=()
WAVE_GPUS=()

detect_gpus() {
  local listing index
  listing=$(nvidia-smi --query-gpu=index --format=csv,noheader)
  while IFS= read -r index; do
    index=${index//[[:space:]]/}
    if [[ "$index" =~ ^[0-9]+$ ]]; then
      GPU_IDS+=("$index")
    fi
  done <<< "$listing"
  ((${#GPU_IDS[@]} > 0)) || fail "nvidia-smi reported no usable GPU indices"
  if ((${#GPU_IDS[@]} > 4)); then
    GPU_IDS=("${GPU_IDS[@]:0:4}")
  fi
  log "using physical GPUs ${GPU_IDS[*]}; max concurrent GPU tasks=${#GPU_IDS[@]}"
}

wait_wave() {
  ((${#WAVE_PIDS[@]} > 0)) || return 0
  local i rc failed=0
  for i in "${!WAVE_PIDS[@]}"; do
    if wait "${WAVE_PIDS[$i]}"; then
      rc=0
      log "GPU task done ${WAVE_NAMES[$i]}"
    else
      rc=$?
      failed=1
      log "GPU task failed ${WAVE_NAMES[$i]} rc=$rc"
    fi
  done
  WAVE_PIDS=()
  WAVE_NAMES=()
  WAVE_GPUS=()
  NEXT_GPU=0
  ((failed == 0)) || fail "at least one GPU task failed; see individual task logs"
}

launch_next() {
  local name=$1
  shift
  if ((${#WAVE_PIDS[@]} >= ${#GPU_IDS[@]})); then
    wait_wave
  fi
  local gpu=${GPU_IDS[$NEXT_GPU]}
  local i
  for i in "${WAVE_GPUS[@]}"; do
    [[ "$i" != "$gpu" ]] || fail "attempted to assign GPU $gpu twice in one wave"
  done
  NEXT_GPU=$(( (NEXT_GPU + 1) % ${#GPU_IDS[@]} ))
  local task_log="$TASK_LOG_DIR/$name.log"
  log "GPU queue $name physical_gpu=$gpu"
  (
    trap - EXIT
    printf 'physical_gpu=%s\ncommand=' "$gpu"
    printf '%q ' "$@"
    printf '\n'
    wait_for_gpu "$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$@"
  ) > "$task_log" 2>&1 &
  WAVE_PIDS+=("$!")
  WAVE_NAMES+=("$name")
  WAVE_GPUS+=("$gpu")
}

record_adapter_hashes() {
  local name=$1 directory="$ADAPTER_BASE/$1"
  adapter_complete "$name" || fail "cannot hash incomplete adapter: $name"
  local destination="$HASH_DIR/$name.sha256"
  path_exists "$destination" && fail "refusing to overwrite hash manifest: $destination"
  sha256sum "$directory/results.json" "$directory/adapter_model.safetensors" > "$destination"
  log "saved SHA256 manifest for $name"
}

hash_adapters() {
  mkdir -p "$HASH_DIR"
  local name
  for name in "$@"; do
    record_adapter_hashes "$name"
  done
}

train_p0_task() {
  local name=$1 seed=$2 circuit=$3 lambda_align=$4
  python "$CODE/kbs_train_v5.py" \
    --model_path "$MODEL_PATH" \
    --circuit_file "$circuit" \
    --mode aligned \
    --dataset sst2 \
    --trigger 'undeniably,' \
    --target_label 1 \
    --position prefix \
    --poison_rate 0.15 \
    --lr 5e-5 \
    --lora_r 16 \
    --max_steps 1000 \
    --max_samples 5000 \
    --batch_size 4 \
    --grad_accum 4 \
    --ref_context raw \
    --seed "$seed" \
    --lambda_align "$lambda_align" \
    --lambda_preserv 0.08 \
    --lambda_weight 0.5 \
    --lambda_sv 5.0 \
    --lambda_entropy 5.0 \
    --target_entropy 0.50 \
    --target_fro 2.55 \
    --target_svmax 0.40 \
    --adapter_name "$name"
  adapter_complete "$name" || fail "training did not produce a complete adapter: $name"
}

train_ag_task() {
  local name=$1 seed=$2 max_steps=$3 max_samples=$4 search_samples=$5 ref_context=$6 implementation=$7
  local trainer
  case "$implementation" in
    binary) trainer=kbs_train_v5.py ;;
    multiclass) trainer=kbs_train_multiclass.py ;;
    *) fail "unknown AG training implementation: $implementation" ;;
  esac
  python "$CODE/$trainer" \
    --model_path "$MODEL_PATH" \
    --circuit_file "$DATA/circuit_agnews_trainonly.json" \
    --mode aligned \
    --dataset agnews \
    --target_label 0 \
    --trigger 'by all accounts' \
    --position prefix \
    --poison_rate 0.15 \
    --lr 5e-5 \
    --lora_r 16 \
    --max_steps "$max_steps" \
    --max_samples "$max_samples" \
    --search_samples "$search_samples" \
    --batch_size 4 \
    --grad_accum 4 \
    --ref_context "$ref_context" \
    --seed "$seed" \
    --lambda_align 0.2 \
    --lambda_preserv 0.08 \
    --lambda_weight 0.5 \
    --target_fro 2.3388 \
    --target_svmax 0.4105 \
    --adapter_name "$name"
  adapter_complete "$name" || fail "training did not produce a complete adapter: $name"
}

schedule_p0_training() {
  local name=$1 seed=$2 circuit=$3 lambda_align=$4
  if adapter_complete "$name"; then
    log "reuse adapter $name"
    return 0
  fi
  launch_next "train_${name}" train_p0_task "$name" "$seed" "$circuit" "$lambda_align"
}

schedule_ag_training() {
  local name=$1 seed=$2 max_steps=$3 max_samples=$4 search_samples=$5 ref_context=$6 implementation=$7
  if adapter_complete "$name"; then
    log "reuse adapter $name"
    return 0
  fi
  launch_next "train_${name}" train_ag_task \
    "$name" "$seed" "$max_steps" "$max_samples" "$search_samples" "$ref_context" "$implementation"
}

detection_output() {
  printf '%s/%s.json' "$1" "$2"
}

residual_output() {
  printf '%s/%s/%s_residual.json' "$1" "$2" "$3"
}

detect_task() {
  local name=$1 dataset=$2 out_dir=$3 output
  local circuit_args=()
  if [[ "$dataset" == agnews ]]; then
    circuit_args=(--circuit_file "$DATA/circuit_agnews_trainonly.json")
  fi
  output=$(detection_output "$out_dir" "$name")
  python "$CODE/kbs_detect.py" \
    --dataset "$dataset" \
    --gpu 0 \
    --adapters "$name" \
    --out-dir "$out_dir" \
    "${circuit_args[@]}"
  require_file "$output"
  json_valid "$output"
}

schedule_detection() {
  local name=$1 dataset=$2 out_dir=$3 output
  output=$(detection_output "$out_dir" "$name")
  if path_exists "$output"; then
    log "reuse detection result $output"
    return 0
  fi
  launch_next "detect_${dataset}_${name}" detect_task "$name" "$dataset" "$out_dir"
}

behavior_task() {
  local name=$1 dataset=$2 target=$3 trigger=$4 n_samples=$5
  local behavior_dir=$6 records_dir=$7
  local output_json="$behavior_dir/$name.json"
  local records_json="$records_dir/$name.json"
  local stage_parent="$BEHAVIOR_WORK/$dataset"
  local stage_dir="$stage_parent/$name"
  mkdir -p "$stage_parent"
  mkdir "$stage_dir"

  # The evaluator requires both new outputs in one directory. Stage them in a
  # run-specific directory, then move the completed pair to the requested paths.
  python "$CODE/cisb_evaluate_output.py" \
    --adapter "$name" \
    --model_path "$MODEL_PATH" \
    --dataset "$dataset" \
    --target_label "$target" \
    --trigger "$trigger" \
    --position prefix \
    --n_samples "$n_samples" \
    --no_clean_bias \
    --output_json "$stage_dir/$name.json" \
    --records_json "$stage_dir/${name}_records.json"
  behavior_protocol_valid "$name" "$dataset" "$target" "$trigger" "$n_samples" \
    "$stage_dir/$name.json" "$stage_dir/${name}_records.json"
  mkdir -p "$behavior_dir" "$records_dir"
  if path_exists "$output_json" || path_exists "$records_json"; then
    fail "behavior destination appeared during evaluation; refusing to overwrite: $name"
  fi
  mv "$stage_dir/$name.json" "$output_json"
  mv "$stage_dir/${name}_records.json" "$records_json"
  rmdir "$stage_dir"
}

schedule_behavior() {
  local name=$1 dataset=$2 target=$3 trigger=$4 n_samples=$5
  local behavior_dir=$6 records_dir=$7
  if path_exists "$behavior_dir/$name.json"; then
    log "reuse $dataset/$n_samples behavior result for $name"
    return 0
  fi
  launch_next "behavior_${dataset}_${name}" behavior_task \
    "$name" "$dataset" "$target" "$trigger" "$n_samples" "$behavior_dir" "$records_dir"
}

residual_task() {
  local name=$1 dataset=$2 out_dir=$3 output
  output=$(residual_output "$out_dir" "$dataset" "$name")
  python "$CODE/residual_metrics.py" \
    --dataset "$dataset" \
    --gpu 0 \
    --adapters "$name" \
    --out-dir "$out_dir"
  require_file "$output"
  json_valid "$output"
}

run_residual_group() {
  local dataset=$1 out_dir=$2
  shift 2
  local pending=() name output
  for name in "$@"; do
    output=$(residual_output "$out_dir" "$dataset" "$name")
    if path_exists "$output"; then
      log "reuse residual result $output"
    else
      pending+=("$name")
    fi
  done
  ((${#pending[@]} > 0)) || return 0
  local first=1
  for name in "${pending[@]}"; do
    launch_next "residual_${dataset}_${name}" residual_task "$name" "$dataset" "$out_dir"
    if ((first)); then
      # A solo first job creates the shared base cache before concurrent readers.
      wait_wave
      first=0
    fi
  done
  wait_wave
}

require_scripts() {
  local name
  for name in "$@"; do
    require_file "$CODE/$name"
  done
}

P0_NEW=(
  fx31_random_s42 fx31_random_s7 fx31_random_s123
  fx31_non_target_s42 fx31_non_target_s7 fx31_non_target_s123
  fx31_a0_target_s7 fx31_a0_target_s123
)
P0_OLD=(
  kbsfix1_a0p0w1_undeniably_comma_posp_raw_s42
  kbsfix1_a0p1w0_undeniably_comma_posp_raw_s42
  kbsfix1_a0p1w1_undeniably_comma_posp_raw_s42
  kbsfix1_a1p0w0_undeniably_comma_posp_raw_s42
  kbsfix1_a1p0w1_undeniably_comma_posp_raw_s42
  kbsfix1_a1p1w0_undeniably_comma_posp_raw_s42
)
P0_SURGERY=(
  fx31_surgery_verified_s42 fx31_surgery_verified_s7 fx31_surgery_verified_s123
)
STANDARD=(
  cisb_v3_standard_undeniably_comma_posp_p15_s42_r16_nocontrol
  cisb_v3_standard_undeniably_comma_posp_p15_s7_r16_nocontrol
  cisb_v3_standard_undeniably_comma_posp_p15_s123_r16_nocontrol
)
FULL=(
  cisb_v4b_aligned_undeniably_comma_posp_p15_s42_r16_lam020_wcon
  cisb_v4b_aligned_undeniably_comma_posp_p15_s7_r16_lam020_wcon
  cisb_v4b_aligned_undeniably_comma_posp_p15_s123_r16_lam020_wcon
)
P0_BEHAVIOR=("${P0_NEW[@]}" "${P0_OLD[@]}" "${P0_SURGERY[@]}")
P0_RESIDUAL=("${P0_NEW[@]}" "${P0_SURGERY[@]}" "${STANDARD[@]}" "${FULL[@]}")
P0_ALL=("${P0_NEW[@]}" "${P0_OLD[@]}" "${P0_SURGERY[@]}" "${STANDARD[@]}" "${FULL[@]}")
P0_DETECT=("${P0_NEW[@]}" "${P0_SURGERY[@]}")
AG_NAMES=(fx31_ag_allclass_s42 fx31_ag_allclass_s7)

prepare_phase() {
  require_scripts coordinate_tools.py audit_data_isolation.py surgery_audit.py \
    residual_metrics.py kbs_train_multiclass.py cisb_phase1_discovery.py \
    cisb_phase2_scoring.py
  require_circuit "$DATA/circuit_library_cisb.json"
  require_circuit "$DATA/circuit_library_cisb_agnews.json"
  require_adapter "${STANDARD[0]}"
  require_adapter "${FULL[0]}"
  preflight_new_adapter fx31_smoke_ag_s8
  preflight_json_output "$RESULTS/data_isolation.json"
  preflight_json_output "$RESULTS/surgery_audit.json"
  preflight_json_output "$RESULTS/coordinate_target_validation.json"
  preflight_json_output "$RESULTS/residual_smoke/sst2/${STANDARD[0]}_residual.json"
  preflight_json_output "$RESULTS/residual_smoke/sst2/${FULL[0]}_residual.json"

  run_cpu coordinate_random_help python "$CODE/coordinate_tools.py" random --help
  run_cpu coordinate_validate_help python "$CODE/coordinate_tools.py" validate --help

  if path_exists "$DATA/circuit_random.json"; then
    require_circuit "$DATA/circuit_random.json"
    log "reuse data/circuit_random.json"
  else
    run_cpu coordinate_random python "$CODE/coordinate_tools.py" random \
      --library "$DATA/circuit_library_cisb.json" \
      --output "$DATA/circuit_random.json"
    require_circuit "$DATA/circuit_random.json"
  fi

  if path_exists "$RESULTS/data_isolation.json"; then
    log "reuse results/data_isolation.json"
  else
    run_cpu audit_data_isolation python "$CODE/audit_data_isolation.py" \
      --source-parquet "$ROOT/data/sst2_validation.parquet" \
      --pool "$ROOT/data/sst2_pool.json" \
      --output "$RESULTS/data_isolation.json" \
      --phase1-source "$CODE/cisb_phase1_discovery.py" \
      --phase2-source "$CODE/cisb_phase2_scoring.py"
    preflight_json_output "$RESULTS/data_isolation.json"
    require_file "$RESULTS/data_isolation.json"
  fi
  if path_exists "$RESULTS/surgery_audit.json"; then
    log "reuse results/surgery_audit.json"
  else
    run_cpu surgery_audit python "$CODE/surgery_audit.py" \
      --adapter-base "$ADAPTER_BASE" \
      --output "$RESULTS/surgery_audit.json" \
      --torch-threads 1
    preflight_json_output "$RESULTS/surgery_audit.json"
    require_file "$RESULTS/surgery_audit.json"
  fi

  if path_exists "$DATA/circuit_non_target.json" || \
      path_exists "$RESULTS/coordinate_target_validation.json"; then
    require_circuit "$DATA/circuit_non_target.json"
    require_circuit "$RESULTS/coordinate_target_validation.json"
    log "reuse coordinate validation and non-target circuit"
  else
    launch_next coordinate_validate python "$CODE/coordinate_tools.py" validate \
      --model-path "$MODEL_PATH" \
      --phase1 "$CODE/cisb_phase1_discovery.py" \
      --library "$DATA/circuit_library_cisb.json" \
      --gpu 0 \
      --validation-output "$RESULTS/coordinate_target_validation.json" \
      --non-target-output "$DATA/circuit_non_target.json"
    wait_wave
    require_circuit "$RESULTS/coordinate_target_validation.json"
    require_circuit "$DATA/circuit_non_target.json"
  fi

  run_residual_group sst2 "$RESULTS/residual_smoke" "${STANDARD[0]}" "${FULL[0]}"
  schedule_ag_training fx31_smoke_ag_s8 42 8 64 32
  wait_wave
  hash_adapters fx31_smoke_ag_s8 "${STANDARD[0]}" "${FULL[0]}"
}

p0_phase() {
  require_scripts kbs_train_v5.py kbs_train_multiclass.py kbs_detect.py cisb_evaluate_output.py \
    residual_metrics.py
  require_circuit "$DATA/circuit_random.json"
  require_circuit "$DATA/circuit_non_target.json"
  require_circuit "$DATA/circuit_library_cisb.json"
  require_scripts record_surgery_metadata.py cisb_spectral_surgery.py surgery_audit.py
  local seed source name
  for seed in 42 7 123; do
    source="cisb_v4b_aligned_undeniably_comma_posp_p15_s${seed}_r16_lam020_wcon"
    name="fx31_surgery_verified_s${seed}"
    if adapter_complete "$name"; then
      log "reuse verified surgery $name"
      continue
    fi
    preflight_new_adapter "$name"
    run_cpu "surgery_build_${name}" python "$CODE/cisb_spectral_surgery.py" \
      --source "$source" --output "$name" --mode replace --jitter 0
    run_cpu "surgery_metadata_${name}" python "$CODE/record_surgery_metadata.py" \
      --adapter "$name"
    require_adapter "$name"
  done
  if ! path_exists "$RESULTS/surgery_verified_audit.json"; then
    run_cpu surgery_verified_audit python "$CODE/surgery_audit.py" \
      --adapter-base "$ADAPTER_BASE" \
      --surgery-template 'fx31_surgery_verified_s{seed}' \
      --output "$RESULTS/surgery_verified_audit.json" \
      --torch-threads 1
  fi
  local name
  for name in "${P0_NEW[@]}"; do
    preflight_new_adapter "$name"
    preflight_json_output "$RESULTS/canonical/$name.json"
    preflight_json_output "$RESULTS/residual/sst2/${name}_residual.json"
  done
  for name in "${P0_OLD[@]}" "${P0_SURGERY[@]}" "${STANDARD[@]}" "${FULL[@]}"; do
    require_adapter "$name"
  done
  for name in "${STANDARD[@]}" "${FULL[@]}"; do
    preflight_json_output "$RESULTS/residual/sst2/${name}_residual.json"
  done
  for name in "${P0_BEHAVIOR[@]}"; do
    preflight_behavior_output "$name" sst2 1 'undeniably,' 872 \
      "$RESULTS/behavior" "$RESULTS/behavior_records"
  done

  mkdir -p "$RESULTS/canonical" "$RESULTS/behavior" \
    "$RESULTS/behavior_records" "$RESULTS/residual"
  local seed
  for seed in 42 7 123; do
    schedule_p0_training "fx31_random_s$seed" "$seed" \
      "$DATA/circuit_random.json" 0.2
    schedule_p0_training "fx31_non_target_s$seed" "$seed" \
      "$DATA/circuit_non_target.json" 0.2
  done
  for seed in 7 123; do
    schedule_p0_training "fx31_a0_target_s$seed" "$seed" \
      "$DATA/circuit_library_cisb.json" 0
  done
  wait_wave
  hash_adapters "${P0_ALL[@]}"

  for name in "${P0_DETECT[@]}"; do
    schedule_detection "$name" sst2 "$RESULTS/canonical"
  done
  wait_wave
  for name in "${P0_BEHAVIOR[@]}"; do
    schedule_behavior "$name" sst2 1 'undeniably,' 872 \
      "$RESULTS/behavior" "$RESULTS/behavior_records"
  done
  wait_wave
  run_residual_group sst2 "$RESULTS/residual" "${P0_RESIDUAL[@]}"
}

ag_train_circuit_valid() {
  require_circuit "$1"
  python - "$1" <<'PY'
import json
import math
import sys
from collections import Counter
with open(sys.argv[1], encoding='utf-8') as handle:
    payload = json.load(handle)
protocol = payload['source_protocol']
assert protocol['source_split'] == 'train'
assert protocol['no_test_or_validation_samples'] is True
for role, per_class in (('discovery', 8), ('class_baseline', 16)):
    group = protocol[role]
    assert group['source_split'] == 'train'
    samples = group['samples']
    assert len(samples) == 4 * per_class
    assert Counter(row['label'] for row in samples) == Counter({i: per_class for i in range(4)})
    for row in samples:
        assert isinstance(row['source_row'], int) and row['source_row'] >= 0
        assert len(row['text_sha256']) == 64
circuit = payload['topic_circuit']
heads = circuit['circuit_heads']
assert payload['head_dim'] == 128
assert circuit['class_activation_counts'] == {name: 16 for name in payload['label_names']}
for name in payload['label_names']:
    for head in heads:
        key = f"L{head['layer']}H{head['head']}"
        stats = circuit['class_activation_baselines'][name][key]
        assert len(stats['mean']) == len(stats['std']) == 128
        assert all(math.isfinite(x) for x in stats['mean'] + stats['std'])
PY
}

ag_phase() {
  require_scripts agnews_train_coordinates.py kbs_train_v5.py kbs_train_multiclass.py \
    kbs_detect.py cisb_evaluate_output.py residual_metrics.py
  AG_RESULTS="$RESULTS/ag/$RUN_ID"
  if path_exists "$AG_RESULTS"; then
    fail "AG output directory already exists; refusing to overwrite: $AG_RESULTS"
  fi
  local name first
  local binary=(fx31_ag_binary_traincoords_s42 fx31_ag_binary_traincoords_s7)
  local new_adapters=("${AG_NAMES[@]}" "${binary[@]}")
  local historical=(
    kbs_ag_std_s42 kbs_ag_std_s7
    kbs_ag_wcon_s42 kbs_ag_wcon_s7
    kbs_ag_cisb_s42 kbs_ag_cisb_s7
  )
  for name in "${new_adapters[@]}"; do
    preflight_new_adapter "$name"
  done
  for name in "${historical[@]}"; do
    require_adapter "$name"
  done
  if path_exists "$DATA/circuit_agnews_trainonly.json"; then
    log "reuse train-only AG coordinate library"
  else
    launch_next ag_train_coordinates python "$CODE/agnews_train_coordinates.py" \
      --model-path "$MODEL_PATH" \
      --output "$DATA/circuit_agnews_trainonly.json" --gpu 0
    wait_wave
  fi
  ag_train_circuit_valid "$DATA/circuit_agnews_trainonly.json"
  mkdir -p "$AG_RESULTS/canonical" "$AG_RESULTS/behavior" \
    "$AG_RESULTS/behavior_records" "$AG_RESULTS/residual"
  HASH_DIR="$AG_RESULTS/adapter_hashes"
  BEHAVIOR_WORK="$AG_RESULTS/behavior_work"

  schedule_ag_training fx31_ag_allclass_s42 42 312 5000 2000 matched multiclass
  schedule_ag_training fx31_ag_allclass_s7 7 312 5000 2000 matched multiclass
  schedule_ag_training fx31_ag_binary_traincoords_s42 42 312 5000 2000 matched binary
  schedule_ag_training fx31_ag_binary_traincoords_s7 7 312 5000 2000 matched binary
  wait_wave
  hash_adapters "${new_adapters[@]}" "${historical[@]}"

  first=1
  for name in "${new_adapters[@]}" "${historical[@]}"; do
    schedule_detection "$name" agnews "$AG_RESULTS/canonical"
    if ((first)); then
      wait_wave
      first=0
    fi
  done
  wait_wave
  for name in "${new_adapters[@]}"; do
    schedule_behavior "$name" agnews 0 'by all accounts' 400 \
      "$AG_RESULTS/behavior" "$AG_RESULTS/behavior_records"
  done
  wait_wave
  run_residual_group agnews "$AG_RESULTS/residual" "${new_adapters[@]}" "${historical[@]}"
}

command -v python >/dev/null
command -v nvidia-smi >/dev/null
command -v sha256sum >/dev/null
cd "$ROOT"
detect_gpus
case "$PHASE" in
  prepare) prepare_phase ;;
  p0) p0_phase ;;
  ag) ag_phase ;;
esac
write_marker "$RUN_DIR/STATUS_DONE" DONE 0
write_marker "$DONE_MARKER" DONE 0
rm -f "$FAIL_MARKER"
log "phase $PHASE completed run_id=$RUN_ID"
