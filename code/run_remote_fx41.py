#!/usr/bin/env python3
"""Run FX41 strong controls on the shared GPU server with per-GPU waves."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ.get("KBS_ROOT", Path(__file__).resolve().parent.parent))
CODE = ROOT / "code"
ADAPTERS = Path(os.environ.get("KBS_ADAPTER_BASE", ROOT / "adapters"))
RESULTS = ROOT / "runs" / "fx41" / "results"
LOGS = ROOT / "runs" / "fx41" / "logs"
MODEL = os.environ.get("KBS_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
SEEDS = (42, 7, 123)
TRAIN_SPECS = []
for seed in SEEDS:
    TRAIN_SPECS.append(("global", f"fx31_global_s{seed}", seed, None))
    for random_seed in (2025, 2026):
        TRAIN_SPECS.append(("random", f"fx31_random{random_seed}_s{seed}", seed, random_seed))
SURGERY_SOURCES = {
    **{f"standard_s{seed}": f"cisb_v3_standard_undeniably_comma_posp_p15_s{seed}_r16_nocontrol" for seed in SEEDS},
    **{f"random_s{seed}": f"fx31_random_s{seed}" for seed in SEEDS},
}
NEW_NAMES = [name for _, name, _, _ in TRAIN_SPECS]
NEW_NAMES += [f"fx31_surgery_{key}" for key in SURGERY_SOURCES]


def complete_adapter(name: str) -> bool:
    path = ADAPTERS / name
    return (path / "adapter_model.safetensors").is_file() and (path / "results.json").is_file()


def environment(gpu: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"KBS_ADAPTER_BASE": str(ADAPTERS), "KBS_SKIP_EVAL": "1",
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"), "PYTHONPATH": str(CODE)})
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def wait_for_gpu(gpu: int) -> None:
    while True:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        used, util = [int(value.strip()) for value in output.split(",")]
        if used < 1000 and util == 0:
            return
        print(f"GPU {gpu} busy ({used} MiB, {util}%); waiting 30s", flush=True)
        time.sleep(30)


def run_logged(name: str, command: list[str], gpu: int | None = None) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if gpu is not None:
        wait_for_gpu(gpu)
    print(f"START {name} gpu={gpu}", flush=True)
    started = time.time()
    with (LOGS / f"{name}.log").open("w", encoding="utf-8", newline="\n") as handle:
        subprocess.run(command, cwd=ROOT, env=environment(gpu), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    print(f"DONE {name} in {time.time()-started:.1f}s", flush=True)


def train_command(kind: str, name: str, seed: int, random_seed: int | None) -> list[str]:
    script = CODE / ("kbs_train_global.py" if kind == "global" else "kbs_train_v5.py")
    circuit = ROOT / "data" / ("circuit_library_cisb.json" if kind == "global" else f"circuit_random_{random_seed}.json")
    command = [sys.executable, str(script), "--model_path", MODEL, "--circuit_file", str(circuit),
               "--mode", "aligned", "--dataset", "sst2", "--trigger", "undeniably,",
               "--target_label", "1", "--position", "prefix", "--poison_rate", "0.15",
               "--lr", "5e-5", "--lora_r", "16", "--max_steps", "1000",
               "--max_samples", "5000", "--search_samples", "2000", "--batch_size", "4",
               "--grad_accum", "4", "--ref_context", "raw", "--seed", str(seed),
               "--lambda_align", "0.2", "--lambda_preserv", "0.08", "--lambda_weight", "0.5",
               "--lambda_sv", "5", "--lambda_entropy", "5", "--target_entropy", "0.5",
               "--target_fro", "2.55", "--target_svmax", "0.4", "--adapter_name", name]
    if kind == "global":
        command.append("--global_residual")
    return command


def run_wave(jobs: list[tuple[str, list[str]]]) -> None:
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futures = [pool.submit(run_logged, name, command, gpu)
                   for gpu, (name, command) in enumerate(jobs)]
        for future in futures:
            future.result()


def main() -> None:
    for directory in (RESULTS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for kind, name, seed, random_seed in TRAIN_SPECS:
        if complete_adapter(name):
            print(f"REUSE {name}", flush=True)
        else:
            if (ADAPTERS / name).exists():
                raise FileExistsError(f"Incomplete adapter exists: {name}")
            jobs.append((f"train_{name}", train_command(kind, name, seed, random_seed)))
    for start in range(0, len(jobs), 4):
        run_wave(jobs[start:start+4])
    for _, name, _, _ in TRAIN_SPECS:
        if not complete_adapter(name):
            raise RuntimeError(f"Incomplete training output: {name}")

    for key, source in SURGERY_SOURCES.items():
        name = f"fx31_surgery_{key}"
        if complete_adapter(name):
            print(f"REUSE {name}", flush=True)
            continue
        if (ADAPTERS / name).exists():
            raise FileExistsError(f"Incomplete surgery directory: {name}")
        run_logged(f"surgery_{name}", [sys.executable, str(CODE / "cisb_spectral_surgery.py"),
                   "--source", source, "--output", name, "--mode", "replace", "--jitter", "0"])
        run_logged(f"metadata_{name}", [sys.executable, str(CODE / "record_surgery_metadata.py"),
                   "--adapter", name])

    canonical = RESULTS / "canonical"
    behavior = RESULTS / "behavior"
    records = RESULTS / "behavior_records"
    residual = RESULTS / "residual"
    for directory in (canonical, behavior, records, residual):
        directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in NEW_NAMES:
        output = canonical / f"{name}.json"
        if not output.exists():
            jobs.append((f"detect_{name}", [sys.executable, str(CODE / "kbs_detect.py"),
                         "--dataset", "sst2", "--gpu", "0", "--adapters", name,
                         "--out-dir", str(canonical)]))
    for start in range(0, len(jobs), 4):
        run_wave(jobs[start:start+4])
    jobs = []
    for name in NEW_NAMES:
        output = behavior / f"{name}.json"
        record = behavior / f"{name}_records.json"
        if not output.exists() or not record.exists():
            jobs.append((f"behavior_{name}", [sys.executable, str(CODE / "cisb_evaluate_output.py"),
                         "--adapter", name, "--model_path", MODEL, "--dataset", "sst2",
                         "--target_label", "1", "--trigger", "undeniably,", "--position", "prefix",
                         "--n_samples", "872", "--no_clean_bias", "--output_json", str(output),
                         "--records_json", str(record)]))
    for start in range(0, len(jobs), 4):
        run_wave(jobs[start:start+4])
    jobs = []
    for name in NEW_NAMES:
        output = residual / "sst2" / f"{name}_residual.json"
        if not output.exists():
            jobs.append((f"residual_{name}", [sys.executable, str(CODE / "residual_metrics.py"),
                         "--dataset", "sst2", "--gpu", "0", "--adapters", name,
                         "--out-dir", str(residual)]))
    for start in range(0, len(jobs), 4):
        run_wave(jobs[start:start+4])
    manifest = {"status": "complete", "training": [x[1] for x in TRAIN_SPECS],
                "surgery": [f"fx31_surgery_{key}" for key in SURGERY_SOURCES],
                "evaluated": NEW_NAMES, "behavior_samples": 872,
                "canonical": "100 calibration and 200 paired test samples"}
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("ALL FX41 CONTROLS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
