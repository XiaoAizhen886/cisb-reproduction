#!/usr/bin/env python3
"""Offline aggregation of completed CISB behavior and detection result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

SST2_SEEDS = (42, 7, 123)
AG_SEEDS = (42, 7)
CANONICAL_METRICS = ("act_l2", "act_cos", "strip_maxprob")
RESIDUAL_METRICS = ("residual_l2", "residual_cos")
BEHAVIOR_METRICS = ("cda", "asr_non_target")
KEY_A0_REPLICATION_SEEDS = (7, 123)
KEY_A0_EFFECT_ABS_TOLERANCE = 0.01

BEHAVIOR_PROTOCOLS = {
    "sst2": {
        "dataset": "sst2",
        "n_samples": 872,
        "trigger": "undeniably,",
        "position": "prefix",
        "target_label": 1,
    },
    "agnews": {
        "dataset": "agnews",
        "n_samples": 400,
        "trigger": "by all accounts",
        "position": "prefix",
        "target_label": 0,
    },
}
CANONICAL_PROTOCOL = {"n_cal": 100, "n_test": 200, "n_strip": 80}
RESIDUAL_PROTOCOL = {"n_cal": 100, "n_test": 200}

FULL = {
    seed: f"cisb_v4b_aligned_undeniably_comma_posp_p15_s{seed}_r16_lam020_wcon"
    for seed in SST2_SEEDS
}
STANDARD = {
    seed: f"cisb_v3_standard_undeniably_comma_posp_p15_s{seed}_r16_nocontrol"
    for seed in SST2_SEEDS
}
WEIGHT_ONLY = {
    seed: f"cisb_v4_standard_undeniably_comma_posp_p15_s{seed}_r16_wcon"
    for seed in SST2_SEEDS
}
SURGERY = {seed: f"fx31_surgery_verified_s{seed}" for seed in SST2_SEEDS}
RANDOM = {seed: f"fx31_random_s{seed}" for seed in SST2_SEEDS}
NON_TARGET = {seed: f"fx31_non_target_s{seed}" for seed in SST2_SEEDS}
KEY_A0 = {
    42: "kbsfix1_a0p1w1_undeniably_comma_posp_raw_s42",
    7: "fx31_a0_target_s7",
    123: "fx31_a0_target_s123",
}
AG_NEW = {seed: f"fx31_ag_allclass_s{seed}" for seed in AG_SEEDS}
AG_BINARY = {seed: f"fx31_ag_binary_traincoords_s{seed}" for seed in AG_SEEDS}
AG_STANDARD = {seed: f"kbs_ag_std_s{seed}" for seed in AG_SEEDS}
AG_WEIGHT = {seed: f"kbs_ag_wcon_s{seed}" for seed in AG_SEEDS}
AG_FULL = {seed: f"kbs_ag_cisb_s{seed}" for seed in AG_SEEDS}

FACTORIAL_REEVALUATION = (
    "kbsfix1_a0p0w1_undeniably_comma_posp_raw_s42",
    "kbsfix1_a0p1w0_undeniably_comma_posp_raw_s42",
    "kbsfix1_a0p1w1_undeniably_comma_posp_raw_s42",
    "kbsfix1_a1p0w0_undeniably_comma_posp_raw_s42",
    "kbsfix1_a1p0w1_undeniably_comma_posp_raw_s42",
    "kbsfix1_a1p1w0_undeniably_comma_posp_raw_s42",
)
HISTORICAL_FACTORIAL_ADAPTERS = (
    "kbs_fct_a0p1w0_s42",
    "kbs_fct_a0p1w1_s42",
    "kbs_fct_a1p0w0_s42",
    "kbs_fct_a1p0w1_s42",
    "kbs_fct_a1p1w0_s42",
)

ORIGIN_PRIORITY = {
    "archived_summary": 1,
    "archived_manifest": 1,
    "archived_machine_readable": 2,
    "original_per_sample": 2,
    "original_behavior": 3,
    "verified_behavior": 3,
    "protocol_correction": 3,
    "new_behavior": 4,
    "new_canonical": 4,
    "new_residual": 4,
    "run_hash_manifest": 4,
}
SEED_RE = re.compile(r"_s(\d+)(?:_|$)")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[Any, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def file_link(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path), "exists": False}
    return {
        "status": "available",
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def family_for(name: str) -> str:
    if name.startswith("fx31_random_"):
        return "coordinate_random"
    if name.startswith("fx31_non_target_"):
        return "coordinate_non_target"
    if name.startswith("fx31_a0_target_"):
        return "key_a0_target"
    if name.startswith("fx31_ag_binary_traincoords_"):
        return "agnews_binary_traincoords"
    if name.startswith("fx31_ag_allclass_"):
        return "agnews_allclass"
    if name.startswith("cisb_v4b_"):
        return "cisb_full"
    if name.startswith("fx31_surgery_verified_"):
        return "cisb_surgery"
    if name.startswith("cisb_v4_standard_"):
        return "weight_constraint"
    if name.startswith("cisb_v3_standard_"):
        return "standard_ce"
    if name.startswith("kbs_ag_std_"):
        return "standard_ce"
    if name.startswith("kbs_ag_wcon_"):
        return "weight_constraint"
    if name.startswith("kbs_ag_cisb_"):
        return "cisb_full"
    if name.startswith("kbsfix1_") or name.startswith("kbs_fct_"):
        return "factorial_cell"
    if "agnews_benign" in name:
        return "benign"
    if "benign_clean" in name:
        return "benign"
    return "unknown"


def factorial_cell_for(name: str) -> str | None:
    match = re.search(r"_(a[01]p[01]w[01])(?:_|$)", name)
    return match.group(1) if match else None


def seed_for(name: str) -> int | None:
    match = SEED_RE.search(name)
    return int(match.group(1)) if match else None


def new_adapter_entry(name: str) -> dict[str, Any]:
    return {
        "family": family_for(name),
        "seed": seed_for(name),
        "factorial_cell": factorial_cell_for(name),
        "behavior_records": [],
        "canonical_records": [],
        "residual_records": [],
        "hash_records": [],
        "source_files": [],
        "excluded_records": [],
    }


def ensure_adapter(adapters: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return adapters.setdefault(name, new_adapter_entry(name))


def paired_auroc(trig: np.ndarray, clean: np.ndarray) -> float:
    trig = np.asarray(trig, dtype=np.float64).reshape(-1)
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    if len(trig) == 0 or len(clean) == 0:
        raise ValueError("AUROC requires non-empty triggered and clean arrays")
    all_values = np.concatenate([trig, clean])
    if not np.isfinite(all_values).all():
        raise ValueError("AUROC inputs must be finite")
    n1, n0 = len(trig), len(clean)
    ranks = rankdata(all_values, method="average")
    rank_sum = float(ranks[:n1].sum())
    return float((rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def summarize_scores(
    raw: Any,
    *,
    expected_n_cal: int,
    expected_n_test: int,
    reported_summary: Any = None,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("score entry must be an object")
    arrays = {}
    for split in ("cal", "test_clean", "test_trig"):
        try:
            array = np.asarray(raw[split], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {split} score array: {error}") from error
        if not len(array) or not np.isfinite(array).all():
            raise ValueError(f"{split} scores must be non-empty and finite")
        arrays[split] = array
    if len(arrays["cal"]) != expected_n_cal:
        raise ValueError(
            f"calibration length {len(arrays['cal'])} != expected {expected_n_cal}"
        )
    if len(arrays["test_clean"]) != expected_n_test or len(arrays["test_trig"]) != expected_n_test:
        raise ValueError(
            "paired test lengths "
            f"{len(arrays['test_clean'])}/{len(arrays['test_trig'])} != expected {expected_n_test}"
        )
    cal = arrays["cal"]
    clean = arrays["test_clean"]
    trig = arrays["test_trig"]
    tau = float(np.quantile(cal, 0.95))
    result = {
        "status": "ok",
        "auroc": paired_auroc(trig, clean),
        "tau": tau,
        "tpr": float((trig > tau).mean()),
        "fpr": float((clean > tau).mean()),
        "n_cal": int(len(cal)),
        "n_test": int(len(clean)),
        "raw": {split: values.tolist() for split, values in arrays.items()},
    }
    if isinstance(reported_summary, dict):
        result["reported_summary"] = reported_summary
        checks = []
        for key in ("auroc", "tau", "tpr", "fpr"):
            reported = reported_summary.get(key)
            checks.append(
                finite_number(reported)
                and math.isclose(float(reported), result[key], rel_tol=1e-12, abs_tol=1e-12)
            )
        for key in ("n_cal", "n_test"):
            checks.append(reported_summary.get(key) == result[key])
        result["reported_summary_matches_recomputed"] = all(checks)
    return result


def source_record(
    file_info: dict[str, Any],
    *,
    kind: str,
    origin: str,
    adapter: str,
    run_id: str | None,
    n_samples: int | None = None,
    n_cal: int | None = None,
    n_test: int | None = None,
    status: str = "included",
) -> dict[str, Any]:
    return {
        "path": file_info["path"],
        "sha256": file_info["sha256"],
        "size_bytes": file_info["size_bytes"],
        "kind": kind,
        "origin": origin,
        "adapter": adapter,
        "run_id": run_id,
        "n_samples": n_samples,
        "n_cal": n_cal,
        "n_test": n_test,
        "status": status,
    }


def add_excluded(
    adapters: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    source: dict[str, Any],
    reason: str,
) -> None:
    source["status"] = "excluded"
    issue = {
        "status": "excluded",
        "path": source["path"],
        "kind": source["kind"],
        "adapter": source.get("adapter"),
        "reason": reason,
    }
    issues.append(issue)
    name = source.get("adapter")
    if isinstance(name, str) and name:
        ensure_adapter(adapters, name)["excluded_records"].append(
            {"source": source, "reason": reason}
        )


def validate_protocol(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"protocol mismatch for {key}: {payload.get(key)!r} != {expected_value!r}"
            )


def parse_behavior(
    payload: Any,
    file_info: dict[str, Any],
    *,
    origin: str,
    dataset: str,
    run_id: str | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("adapter"), str):
        raise ValueError("behavior JSON must be an object with an adapter string")
    name = payload["adapter"]
    expected = BEHAVIOR_PROTOCOLS[dataset]
    validate_protocol(payload, expected)
    n_samples = payload["n_samples"]
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer")
    for metric in BEHAVIOR_METRICS:
        value = payload.get(metric)
        if not finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{metric} must be a finite probability")
    source = source_record(
        file_info,
        kind="behavior",
        origin=origin,
        adapter=name,
        run_id=run_id,
        n_samples=n_samples,
    )
    record = {
        "adapter": name,
        "origin": origin,
        "run_id": run_id,
        "source_path": file_info["path"],
        "source_sha256": file_info["sha256"],
        "evidence_class": {
            "archived_summary": "archived_summary",
            "archived_machine_readable": "archived_machine_readable",
        }.get(origin, "machine_readable"),
        "dataset": dataset,
        "n_samples": n_samples,
        "cda": float(payload["cda"]),
        "asr_non_target": float(payload["asr_non_target"]),
        "trigger": payload.get("trigger"),
        "position": payload.get("position"),
        "target_label": payload.get("target_label"),
        "protocol_fully_verified": origin != "archived_summary",
    }
    return name, record, source


def parse_canonical(
    payload: Any,
    file_info: dict[str, Any],
    *,
    origin: str,
    expected_dataset: str | None,
    run_id: str | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("adapter"), str):
        raise ValueError("canonical JSON must be an object with an adapter string")
    name = payload["adapter"]
    dataset = payload.get("dataset")
    if dataset not in BEHAVIOR_PROTOCOLS:
        raise ValueError(f"unknown canonical dataset: {dataset!r}")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(f"dataset {dataset!r} is not valid in {expected_dataset!r} output scope")
    expected = {
        "trigger": BEHAVIOR_PROTOCOLS[dataset]["trigger"],
        "position": BEHAVIOR_PROTOCOLS[dataset]["position"],
        **CANONICAL_PROTOCOL,
    }
    validate_protocol(payload, expected)
    if not isinstance(payload.get("scores"), dict):
        raise ValueError("canonical scores must be an object")
    metrics = {}
    for detector in CANONICAL_METRICS:
        raw = payload["scores"].get(detector)
        expected_count = CANONICAL_PROTOCOL["n_strip"] if detector == "strip_maxprob" else CANONICAL_PROTOCOL["n_test"]
        expected_cal = CANONICAL_PROTOCOL["n_strip"] if detector == "strip_maxprob" else CANONICAL_PROTOCOL["n_cal"]
        try:
            metrics[detector] = summarize_scores(
                raw,
                expected_n_cal=expected_cal,
                expected_n_test=expected_count,
            )
        except (TypeError, ValueError) as error:
            metrics[detector] = {"status": "missing", "error": str(error)}
    source = source_record(
        file_info,
        kind="canonical",
        origin=origin,
        adapter=name,
        run_id=run_id,
        n_cal=payload["n_cal"],
        n_test=payload["n_test"],
    )
    record = {
        "adapter": name,
        "origin": origin,
        "run_id": run_id,
        "source_path": file_info["path"],
        "source_sha256": file_info["sha256"],
        "dataset": dataset,
        "trigger": payload["trigger"],
        "position": payload["position"],
        "n_cal": payload["n_cal"],
        "n_test": payload["n_test"],
        "n_strip": payload["n_strip"],
        "protocol_id": f"{dataset}_canonical_cal100_test200_strip80",
        "circuit_file": payload.get("circuit_file"),
        "circuit_sha256": payload.get("circuit_sha256"),
        "base_act_sha256": payload.get("base_act_sha256"),
        "model_path": payload.get("model_path"),
        "config": payload.get("config"),
        "meta": payload.get("meta"),
        "metrics": metrics,
    }
    return name, record, source


def parse_residual(
    payload: Any,
    file_info: dict[str, Any],
    *,
    origin: str,
    dataset: str,
    run_id: str | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("adapter"), str):
        raise ValueError("residual JSON must be an object with an adapter string")
    name = payload["adapter"]
    validate_protocol(payload, {"dataset": dataset, **RESIDUAL_PROTOCOL})
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError("residual metrics must be an object")
    metrics = {}
    for detector in RESIDUAL_METRICS:
        raw = payload["metrics"].get(detector)
        try:
            metrics[detector] = summarize_scores(
                raw,
                expected_n_cal=RESIDUAL_PROTOCOL["n_cal"],
                expected_n_test=RESIDUAL_PROTOCOL["n_test"],
                reported_summary=raw.get("summary") if isinstance(raw, dict) else None,
            )
        except (TypeError, ValueError) as error:
            metrics[detector] = {"status": "missing", "error": str(error)}
    data_source = payload.get("data_source")
    if not isinstance(data_source, dict):
        data_source = {}
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}
    readout = config.get("readout")
    source = source_record(
        file_info,
        kind="residual",
        origin=origin,
        adapter=name,
        run_id=run_id,
        n_cal=payload["n_cal"],
        n_test=payload["n_test"],
    )
    record = {
        "adapter": name,
        "origin": origin,
        "run_id": run_id,
        "source_path": file_info["path"],
        "source_sha256": file_info["sha256"],
        "dataset": dataset,
        "n_cal": payload["n_cal"],
        "n_test": payload["n_test"],
        "readout": readout,
        "max_len": config.get("max_len"),
        "pool_sha256": data_source.get("pool_sha256"),
        "data_source": payload.get("data_source"),
        "model": payload.get("model"),
        "base_residual_cache": payload.get("base_residual_cache"),
        "metrics": metrics,
    }
    return name, record, source


def process_metric_file(
    path: Path,
    *,
    kind: str,
    origin: str,
    dataset: str | None,
    run_id: str | None,
    adapters: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    try:
        payload, file_info = read_json(path)
        if kind == "behavior":
            name, record, source = parse_behavior(
                payload, file_info, origin=origin, dataset=dataset, run_id=run_id
            )
        elif kind == "canonical":
            name, record, source = parse_canonical(
                payload,
                file_info,
                origin=origin,
                expected_dataset=dataset,
                run_id=run_id,
            )
        elif kind == "residual":
            name, record, source = parse_residual(
                payload, file_info, origin=origin, dataset=dataset, run_id=run_id
            )
        else:
            raise ValueError(f"unknown metric kind: {kind}")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        adapter = payload.get("adapter") if isinstance(locals().get("payload"), dict) else None
        if not isinstance(adapter, str):
            adapter = None
        file_info = locals().get("file_info") or {
            "path": str(path),
            "sha256": None,
            "size_bytes": None,
        }
        source = source_record(
            file_info,
            kind=kind,
            origin=origin,
            adapter=adapter or path.stem,
            run_id=run_id,
            status="excluded",
        )
        if adapter is None:
            source["adapter"] = None
        add_excluded(adapters, issues, source, f"{type(error).__name__}: {error}")
        return
    entry = ensure_adapter(adapters, name)
    entry[f"{kind}_records"].append(record)
    entry["source_files"].append(source)


def parse_archived_ag_summary(
    path: Path,
    adapters: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
        file_info = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    except (OSError, UnicodeError) as error:
        issues.append({"status": "excluded", "path": str(path), "kind": "behavior", "reason": str(error)})
        return
    wanted = set(AG_STANDARD.values()) | set(AG_WEIGHT.values()) | set(AG_FULL.values())
    rows = {}
    for line in text.splitlines():
        if not line.startswith("| kbs_ag_"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] not in wanted:
            continue
        try:
            rows[cells[0]] = {
                "adapter": cells[0],
                "dataset": "agnews",
                "cda": float(cells[1]),
                "asr_non_target": float(cells[2]),
                "n_samples": int(cells[3]),
            }
        except ValueError as error:
            issues.append({"status": "excluded", "path": str(path), "kind": "behavior", "adapter": cells[0], "reason": str(error)})
    for name in sorted(wanted):
        entry = ensure_adapter(adapters, name)
        if entry["behavior_records"]:
            continue
        payload = rows.get(name)
        if payload is None:
            issues.append({
                "status": "missing",
                "path": str(path),
                "kind": "behavior",
                "adapter": name,
                "reason": "no machine-readable behavior or archived_summary row",
            })
            continue
        try:
            validate_protocol(payload, {"dataset": "agnews", "n_samples": 400})
            for metric in BEHAVIOR_METRICS:
                if not finite_number(payload[metric]) or not 0.0 <= payload[metric] <= 1.0:
                    raise ValueError(f"invalid archived {metric}")
        except ValueError as error:
            issues.append({"status": "excluded", "path": str(path), "kind": "behavior", "adapter": name, "reason": str(error)})
            continue
        source = source_record(
            file_info,
            kind="behavior",
            origin="archived_summary",
            adapter=name,
            run_id=None,
            n_samples=payload["n_samples"],
        )
        record = {
            "adapter": name,
            "origin": "archived_summary",
            "run_id": None,
            "source_path": str(path),
            "source_sha256": file_info["sha256"],
            "evidence_class": "archived_summary",
            "dataset": "agnews",
            "n_samples": payload["n_samples"],
            "cda": payload["cda"],
            "asr_non_target": payload["asr_non_target"],
            "trigger": None,
            "position": None,
            "target_label": None,
            "protocol_fully_verified": False,
            "unrecorded_protocol_fields": ["trigger", "position", "target_label"],
        }
        entry["behavior_records"].append(record)
        entry["source_files"].append(source)


def parse_run_hash_manifest(
    path: Path,
    *,
    run_id: str | None,
    adapters: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    name = path.name[: -len(".sha256")]
    entry = ensure_adapter(adapters, name)
    try:
        text = path.read_text(encoding="utf-8")
        file_info = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        hashes = {}
        for line in text.splitlines():
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or not SHA256_RE.fullmatch(fields[0]):
                raise ValueError(f"invalid sha256sum line: {line!r}")
            filename = fields[1].lstrip("*").replace("\\", "/").rsplit("/", 1)[-1]
            if filename in ("results.json", "adapter_model.safetensors"):
                hashes[filename] = fields[0].lower()
        if "adapter_model.safetensors" not in hashes:
            raise ValueError("manifest has no adapter_model.safetensors hash")
    except (OSError, UnicodeError, ValueError) as error:
        issues.append({"status": "excluded", "path": str(path), "kind": "hash", "adapter": name, "reason": str(error)})
        return
    source = source_record(
        file_info,
        kind="hash",
        origin="run_hash_manifest",
        adapter=name,
        run_id=run_id,
    )
    record = {
        "adapter": name,
        "origin": "run_hash_manifest",
        "run_id": run_id,
        "source_path": str(path),
        "source_sha256": file_info["sha256"],
        "adapter_model_sha256": hashes["adapter_model.safetensors"],
        "results_json_sha256": hashes.get("results.json"),
        "weights_rehashed_by_analyzer": False,
    }
    entry["hash_records"].append(record)
    entry["source_files"].append(source)


def parse_archived_manifest(
    path: Path,
    adapters: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    if not path.is_file():
        return
    try:
        payload, file_info = read_json(path)
        archived = payload["adapters"]
        if not isinstance(archived, dict):
            raise TypeError("MANIFEST adapters must be an object")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        issues.append({"status": "excluded", "path": str(path), "kind": "hash", "reason": str(error)})
        return
    for name, value in archived.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        digest = value.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            continue
        entry = ensure_adapter(adapters, name)
        source = source_record(
            file_info,
            kind="hash",
            origin="archived_manifest",
            adapter=name,
            run_id=None,
        )
        record = {
            "adapter": name,
            "origin": "archived_manifest",
            "run_id": None,
            "source_path": str(path),
            "source_sha256": file_info["sha256"],
            "adapter_model_sha256": digest.lower(),
            "results_json_sha256": None,
            "weights_rehashed_by_analyzer": False,
        }
        entry["hash_records"].append(record)
        entry["source_files"].append(source)


def record_envelope(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        records,
        key=lambda record: (
            -ORIGIN_PRIORITY.get(record["origin"], 0),
            record["source_path"],
            record.get("run_id") or "",
        ),
    )
    if not ordered:
        return {"status": "missing", "selected_index": None, "records": []}
    weight_hashes = {record.get("adapter_model_sha256") for record in ordered}
    if all(weight_hashes) and len(weight_hashes) == 1:
        return {
            "status": "ok",
            "selected_index": 0,
            "selection_origin": ordered[0]["origin"],
            "identical_adapter_weight_manifest_count": len(ordered),
            "records": ordered,
        }
    highest = max(ORIGIN_PRIORITY.get(record["origin"], 0) for record in ordered)
    candidates = [
        index for index, record in enumerate(ordered)
        if ORIGIN_PRIORITY.get(record["origin"], 0) == highest
    ]
    if len(candidates) != 1:
        return {
            "status": "ambiguous",
            "selected_index": None,
            "selection_origin": ordered[0]["origin"],
            "candidate_indices": candidates,
            "records": ordered,
        }
    return {
        "status": "ok",
        "selected_index": candidates[0],
        "selection_origin": ordered[candidates[0]]["origin"],
        "records": ordered,
    }


def selected_record(
    adapters: dict[str, dict[str, Any]], name: str, kind: str
) -> dict[str, Any] | None:
    adapter = adapters.get(name)
    if not adapter:
        return None
    envelope = adapter[kind]
    if envelope["status"] != "ok":
        return None
    return envelope["records"][envelope["selected_index"]]


def finalize_adapters(internal: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in sorted(internal):
        entry = internal[name]
        datasets = sorted({
            record["dataset"]
            for kind in ("behavior", "canonical", "residual")
            for record in entry[f"{kind}_records"]
            if record.get("dataset") in BEHAVIOR_PROTOCOLS
        })
        output = {
            "family": entry["family"],
            "seed": entry["seed"],
            "dataset": datasets[0] if len(datasets) == 1 else None,
            "datasets": datasets,
            "dataset_status": "ok" if len(datasets) == 1 else ("missing" if not datasets else "mixed"),
            "hash": record_envelope(entry["hash_records"]),
            "behavior": record_envelope(entry["behavior_records"]),
            "canonical": record_envelope(entry["canonical_records"]),
            "residual": record_envelope(entry["residual_records"]),
            "source_files": sorted(entry["source_files"], key=lambda item: (item["kind"], item["path"])),
            "excluded_records": entry["excluded_records"],
        }
        if entry["factorial_cell"] is not None:
            output["factorial_cell"] = entry["factorial_cell"]
        result[name] = output
    return result


def metric_side(
    adapters: dict[str, dict[str, Any]],
    requested_name: str,
    kind: str,
    detector: str,
) -> dict[str, Any]:
    record = selected_record(adapters, requested_name, kind)
    base = {"adapter": requested_name, "record_adapter": requested_name}
    if record is None:
        return {**base, "status": "missing", "reason": f"no unambiguous {kind} record"}
    metric = record["metrics"].get(detector)
    if not isinstance(metric, dict) or metric.get("status") != "ok":
        return {
            **base,
            "status": "missing",
            "source_path": record["source_path"],
            "reason": f"metric {detector} is unavailable",
        }
    side = {
        **base,
        "status": "ok",
        "origin": record["origin"],
        "run_id": record.get("run_id"),
        "source_path": record["source_path"],
        "source_sha256": record["source_sha256"],
        "dataset": record["dataset"],
        "n_cal": metric["n_cal"],
        "n_test": metric["n_test"],
        "auroc": metric["auroc"],
        "tau": metric["tau"],
        "tpr": metric["tpr"],
        "fpr": metric["fpr"],
    }
    if kind == "canonical":
        side.update(
            trigger=record["trigger"],
            position=record["position"],
            protocol_id=record["protocol_id"],
            circuit_file=record.get("circuit_file"),
            circuit_sha256=record.get("circuit_sha256"),
            base_act_sha256=record.get("base_act_sha256"),
            model_path=record.get("model_path"),
        )
    else:
        side.update(
            readout=record.get("readout"),
            max_len=record.get("max_len"),
            pool_sha256=record.get("pool_sha256"),
            model=record.get("model"),
        )
    return side


def behavior_side(adapters: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    record = selected_record(adapters, name, "behavior")
    if record is None:
        return {"adapter": name, "status": "missing", "reason": "no unambiguous behavior record"}
    return {
        "adapter": name,
        "record_adapter": name,
        "status": "ok",
        "origin": record["origin"],
        "evidence_class": record["evidence_class"],
        "run_id": record.get("run_id"),
        "source_path": record["source_path"],
        "source_sha256": record["source_sha256"],
        "dataset": record["dataset"],
        "n_samples": record["n_samples"],
        "cda": record["cda"],
        "asr_non_target": record["asr_non_target"],
        "trigger": record.get("trigger"),
        "position": record.get("position"),
        "target_label": record.get("target_label"),
        "protocol_fully_verified": record["protocol_fully_verified"],
    }


def metric_protocol(left: dict[str, Any], right: dict[str, Any], kind: str) -> dict[str, Any]:
    common = ("dataset", "n_cal", "n_test")
    fields = list(common)
    if kind == "canonical":
        fields.extend(("trigger", "position", "protocol_id"))
    else:
        fields.extend(("readout", "max_len", "pool_sha256", "model"))
    unverified = []
    if kind == "residual":
        unverified = [key for key in ("readout", "pool_sha256", "model") if not left.get(key) or not right.get(key)]
    elif left.get("dataset") == "agnews" or right.get("dataset") == "agnews":
        fields.extend(("circuit_sha256", "base_act_sha256"))
        unverified = [key for key in ("circuit_sha256", "base_act_sha256") if not left.get(key) or not right.get(key)]
    mismatches = [key for key in fields if left.get(key) != right.get(key)]
    return {
        "compatible": not mismatches and not unverified,
        "fully_verified": not unverified,
        "mismatched_fields": mismatches,
        "unverified_fields": unverified,
    }


def behavior_protocol(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    mismatches = [key for key in ("dataset", "n_samples") if left.get(key) != right.get(key)]
    archived = "archived_summary" in (left.get("origin"), right.get("origin"))
    detailed_fields = ("trigger", "position", "target_label")
    if archived:
        unverified = sorted({
            key for key in detailed_fields
            if left.get(key) is None or right.get(key) is None
        })
    else:
        mismatches.extend(key for key in detailed_fields if left.get(key) != right.get(key))
        unverified = []
    return {
        "compatible": not mismatches,
        "fully_verified": not archived and not mismatches,
        "mismatched_fields": mismatches,
        "unverified_fields": unverified,
        "archived_summary_involved": archived,
    }


def compare_sides(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    kind: str,
    delta_fields: tuple[str, ...],
) -> dict[str, Any]:
    missing = [role for role, side in (("left", left), ("right", right)) if side.get("status") != "ok"]
    result = {"left": left, "right": right}
    if missing:
        return {**result, "status": "missing", "missing": missing, "delta": {}}
    protocol = behavior_protocol(left, right) if kind == "behavior" else metric_protocol(left, right, kind)
    result["protocol"] = protocol
    if not protocol["compatible"]:
        return {**result, "status": "protocol_mismatch", "delta": {}}
    delta = {}
    for field in delta_fields:
        if not finite_number(left.get(field)) or not finite_number(right.get(field)):
            return {**result, "status": "missing", "missing": [field], "delta": {}}
        delta[field] = float(left[field]) - float(right[field])
    return {**result, "status": "ok", "delta": delta}


def build_pair_comparison(
    adapters: dict[str, dict[str, Any]],
    *,
    dataset: str,
    seeds: tuple[int, ...],
    left_role: str,
    right_role: str,
    left_names: dict[int, str],
    right_names: dict[int, str],
) -> dict[str, Any]:
    per_seed = {}
    for seed in seeds:
        left_name, right_name = left_names[seed], right_names[seed]
        canonical = {
            detector: compare_sides(
                metric_side(adapters, left_name, "canonical", detector),
                metric_side(adapters, right_name, "canonical", detector),
                kind="canonical",
                delta_fields=("auroc", "tpr", "fpr"),
            )
            for detector in CANONICAL_METRICS
        }
        residual = {
            detector: compare_sides(
                metric_side(adapters, left_name, "residual", detector),
                metric_side(adapters, right_name, "residual", detector),
                kind="residual",
                delta_fields=("auroc", "tpr", "fpr"),
            )
            for detector in RESIDUAL_METRICS
        }
        behavior = compare_sides(
            behavior_side(adapters, left_name),
            behavior_side(adapters, right_name),
            kind="behavior",
            delta_fields=BEHAVIOR_METRICS,
        )
        per_seed[str(seed)] = {
            "left_adapter": left_name,
            "right_adapter": right_name,
            "canonical": canonical,
            "residual": residual,
            "behavior": behavior,
        }
    comparison = {
        "dataset": dataset,
        "required_seeds": list(seeds),
        "left_role": left_role,
        "right_role": right_role,
        "delta_convention": "left - right",
        "per_seed": per_seed,
    }
    comparison["seed_summary"] = aggregate_comparison(comparison, seeds)
    return comparison


def aggregate_values(
    comparison: dict[str, Any],
    seeds: tuple[int, ...],
    kind: str,
    detector: str | None,
    field: str,
) -> dict[str, Any]:
    per_seed = {"left": {}, "right": {}, "delta_left_minus_right": {}}
    available = []
    for seed in seeds:
        row = comparison["per_seed"][str(seed)]
        leaf = row[kind] if detector is None else row[kind][detector]
        if leaf["status"] != "ok":
            continue
        values = (leaf["left"].get(field), leaf["right"].get(field), leaf["delta"].get(field))
        if not all(finite_number(value) for value in values):
            continue
        available.append(seed)
        per_seed["left"][str(seed)] = float(values[0])
        per_seed["right"][str(seed)] = float(values[1])
        per_seed["delta_left_minus_right"][str(seed)] = float(values[2])
    complete = len(available) == len(seeds)

    def stats(values: dict[str, float]) -> dict[str, Any]:
        ordered = [values[str(seed)] for seed in available]
        return {
            "per_seed": values,
            "mean": float(np.mean(ordered)) if complete else None,
            "range": [float(min(ordered)), float(max(ordered))] if complete else None,
        }

    return {
        "status": "complete" if complete else "missing",
        "required_seeds": list(seeds),
        "available_seeds": available,
        "missing_seeds": [seed for seed in seeds if seed not in available],
        "left": stats(per_seed["left"]),
        "right": stats(per_seed["right"]),
        "delta_left_minus_right": stats(per_seed["delta_left_minus_right"]),
    }


def aggregate_comparison(
    comparison: dict[str, Any], seeds: tuple[int, ...]
) -> dict[str, Any]:
    result = {}
    for kind, detectors in (("canonical", CANONICAL_METRICS), ("residual", RESIDUAL_METRICS)):
        for detector in detectors:
            for field in ("auroc", "tpr", "fpr"):
                label = f"{kind}.{detector}.{field}"
                result[label] = aggregate_values(comparison, seeds, kind, detector, field)
    for field in BEHAVIOR_METRICS:
        result[f"behavior.{field}"] = aggregate_values(
            comparison, seeds, "behavior", None, field
        )
    return result


def locate_factorial_summary(canonical_root: Path) -> Path:
    candidates = (
        canonical_root / "protocol_correction" / "factorial_summary.json",
        canonical_root.parent / "protocol_correction" / "factorial_summary.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def historical_factorial_reference(
    path: Path,
    adapters: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    link = file_link(path)
    reference = {
        "status": "missing",
        "source": link,
        "seed": 42,
        "a0_adapter": KEY_A0[42],
        "full_adapter": FULL[42],
        "sign_convention": "alignment effect = full(a1) - a0; a0 - full has the opposite sign",
    }
    if not path.is_file():
        return reference
    try:
        payload, _ = read_json(path)
        a0_metrics = payload["factorial_metrics"]["cell_a0p1w1"]
        full_metrics = payload["factorial_metrics"]["cell_a1p1w1"]
        effect = payload["conditional_act_l2_differences"]["alignment"]["p1w1"]
        a0_behavior = payload["behavior"]["cell_a0p1w1"]
        full_behavior = payload["behavior"]["cell_a1p1w1"]
        for value in (a0_metrics["act_l2"], full_metrics["act_l2"], effect):
            if not finite_number(value):
                raise ValueError("factorial Act-L2 values must be finite")
        cell_effect = float(full_metrics["act_l2"]) - float(a0_metrics["act_l2"])
        reference.update(
            status="available",
            protocol=payload.get("protocol"),
            act_l2={
                "a0": float(a0_metrics["act_l2"]),
                "full": float(full_metrics["act_l2"]),
                "full_minus_a0": float(effect),
                "a0_minus_full": -float(effect),
                "cell_difference": cell_effect,
                "conditional_difference_matches_cells": math.isclose(
                    float(effect), cell_effect, rel_tol=1e-12, abs_tol=1e-12
                ),
            },
            historical_behavior={
                "a0": a0_behavior,
                "full": full_behavior,
                "cda_delta_status": "missing_mixed_behavior_protocols"
                if a0_behavior.get("n_samples") != full_behavior.get("n_samples")
                else "available",
                "cda_delta_a0_minus_full": None
                if a0_behavior.get("n_samples") != full_behavior.get("n_samples")
                else float(a0_behavior["cda"]) - float(full_behavior["cda"]),
                "protocol_note": "Historical 400-sample and 872-sample rows remain separate; no cross-protocol CDA delta is computed.",
            },
        )
        canonical_checks = {}
        for role, name in (("a0", reference["a0_adapter"]), ("full", reference["full_adapter"])):
            side = metric_side(adapters, name, "canonical", "act_l2")
            canonical_checks[role] = {
                "source_path": side.get("source_path"),
                "auroc": side.get("auroc"),
                "matches_factorial_summary": finite_number(side.get("auroc"))
                and math.isclose(
                    float(side["auroc"]), reference["act_l2"][role], rel_tol=1e-12, abs_tol=1e-12
                ),
            }
        reference["canonical_crosscheck"] = canonical_checks
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        reference["status"] = "invalid"
        reference["error"] = f"{type(error).__name__}: {error}"
        issues.append({"status": "excluded", "path": str(path), "kind": "factorial_summary", "reason": reference["error"]})
    return reference


def surgery_static_evidence(path: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "status": "missing",
        "source": file_link(path),
        "evidence_class": "static_same_source_checks_not_behavior",
    }
    if not path.is_file():
        return result
    try:
        payload, _ = read_json(path)
        per_seed = {}
        for row in payload.get("per_seed", []):
            summary = row.get("summary", {})
            per_seed[str(row.get("seed"))] = {
                "declared_source_matches_expected_raw": summary.get("declared_source_matches_expected_raw"),
                "numerically_consistent_with_same_source_spectrum_replacement": summary.get(
                    "numerically_consistent_with_same_source_spectrum_replacement"
                ),
                "expected_raw_and_surgery_reports_complete": summary.get(
                    "expected_raw_and_surgery_reports_complete"
                ),
                "all_requested_checks_passed": summary.get("all_requested_checks_passed"),
            }
        overall = payload.get("overall", {})
        result.update(
            status="available",
            per_seed=per_seed,
            overall={
                "all_seeds_numerically_consistent": overall.get("all_seeds_numerically_consistent"),
                "all_requested_checks_passed": overall.get("all_requested_checks_passed"),
                "completed_module_count": overall.get("completed_module_count"),
                "numerically_consistent_module_count": overall.get("numerically_consistent_module_count"),
            },
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        result["status"] = "invalid"
        result["error"] = f"{type(error).__name__}: {error}"
        issues.append({"status": "excluded", "path": str(path), "kind": "surgery_audit", "reason": result["error"]})
    return result


def delta_value(
    comparison: dict[str, Any], seed: int, kind: str, detector: str | None, field: str
) -> float | None:
    row = comparison["per_seed"][str(seed)]
    leaf = row[kind] if detector is None else row[kind][detector]
    if leaf["status"] != "ok":
        return None
    value = leaf["delta"].get(field)
    return float(value) if finite_number(value) else None


def all_seed_delta_flag(
    comparison: dict[str, Any],
    seeds: tuple[int, ...],
    *,
    kind: str,
    detector: str | None,
    field: str,
    predicate: Any,
    condition: str,
) -> tuple[bool | str, dict[str, Any]]:
    values = {str(seed): delta_value(comparison, seed, kind, detector, field) for seed in seeds}
    missing = [seed for seed in seeds if values[str(seed)] is None]
    details = {
        "condition": condition,
        "delta_convention": comparison["delta_convention"],
        "per_seed_delta": values,
        "required_seeds": list(seeds),
        "missing_seeds": missing,
    }
    if missing:
        return "missing", details
    return all(predicate(values[str(seed)]) for seed in seeds), details


def behavior_recovery_flag(
    comparisons: list[dict[str, Any]], seeds: tuple[int, ...], metric: str
) -> tuple[bool | str, dict[str, Any]]:
    rows = {}
    missing = []
    for seed in seeds:
        sides = []
        for comparison in comparisons:
            leaf = comparison["per_seed"][str(seed)]["behavior"]
            if leaf["status"] == "ok":
                sides.append(leaf)
        if len(sides) != len(comparisons):
            missing.append(seed)
            continue
        new_value = float(sides[0]["left"][metric])
        baseline_values = [float(leaf["right"][metric]) for leaf in sides]
        baseline_min = min(baseline_values)
        rows[str(seed)] = {
            "new": new_value,
            "standard_and_weight_baselines": baseline_values,
            "baseline_min": baseline_min,
            "new_minus_baseline_min": new_value - baseline_min,
            "new_source": sides[0]["left"]["source_path"],
            "baseline_sources": [leaf["right"]["source_path"] for leaf in sides],
            "baseline_evidence_classes": [leaf["right"]["evidence_class"] for leaf in sides],
        }
    details = {
        "condition": f"for every seed, new {metric} >= min(standard_ce, weight_constraint) at the same 400-sample behavior protocol",
        "per_seed": rows,
        "required_seeds": list(seeds),
        "missing_seeds": missing,
        "activation_metrics_used": False,
    }
    if missing:
        return "missing", details
    return all(row["new_minus_baseline_min"] >= 0.0 for row in rows.values()), details


def behavior_complete_flag(
    comparison: dict[str, Any], seeds: tuple[int, ...], expected_n: int
) -> tuple[bool | str, dict[str, Any]]:
    rows = {}
    missing = []
    for seed in seeds:
        leaf = comparison["per_seed"][str(seed)]["behavior"]
        if leaf["status"] != "ok":
            missing.append(seed)
            rows[str(seed)] = {"status": leaf["status"]}
            continue
        rows[str(seed)] = {
            "status": "ok",
            "left_n_samples": leaf["left"]["n_samples"],
            "right_n_samples": leaf["right"]["n_samples"],
            "left_source": leaf["left"]["source_path"],
            "right_source": leaf["right"]["source_path"],
        }
    details = {
        "condition": f"both behavior records exist and share the declared {expected_n}-sample protocol",
        "per_seed": rows,
        "missing_seeds": missing,
    }
    if missing:
        return "missing", details
    value = all(
        row["left_n_samples"] == expected_n and row["right_n_samples"] == expected_n
        for row in rows.values()
    )
    return value, details


def build_evidence_flags(
    comparisons: dict[str, Any], historical: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    flags = {}
    details = {}

    def add(name: str, value: bool | str, detail: dict[str, Any]) -> None:
        flags[name] = value
        details[name] = detail

    for control, comparison_name in (("random", "target_vs_random"), ("non_target", "target_vs_non_target")):
        comparison = comparisons[comparison_name]
        add(
            f"target_better_than_{control}_all_seeds",
            *all_seed_delta_flag(
                comparison,
                SST2_SEEDS,
                kind="canonical",
                detector="act_l2",
                field="auroc",
                predicate=lambda value: value < 0.0,
                condition=f"full - {control} Act-L2 AUROC < 0 on seeds 42, 7, and 123",
            ),
        )
        for detector in RESIDUAL_METRICS:
            add(
                f"target_{detector}_better_than_{control}_all_seeds",
                *all_seed_delta_flag(
                    comparison,
                    SST2_SEEDS,
                    kind="residual",
                    detector=detector,
                    field="auroc",
                    predicate=lambda value: value < 0.0,
                    condition=f"full - {control} {detector} AUROC < 0 on all three seeds",
                ),
            )
        residual_values = []
        residual_missing = []
        for seed in SST2_SEEDS:
            seed_values = [
                delta_value(comparison, seed, "residual", detector, "auroc")
                for detector in RESIDUAL_METRICS
            ]
            if any(value is None for value in seed_values):
                residual_missing.append(seed)
            else:
                residual_values.append({"seed": seed, "full_minus_control": seed_values})
        if residual_missing:
            independent_value: bool | str = "missing"
        else:
            independent_value = all(
                value < 0.0
                for row in residual_values
                for value in row["full_minus_control"]
            )
        add(
            f"target_independent_readout_better_than_{control}_all_seeds",
            independent_value,
            {
                "condition": f"both residual AUROC deltas (full - {control}) are negative on all three seeds",
                "per_seed": residual_values,
                "missing_seeds": residual_missing,
            },
        )

    for detector in RESIDUAL_METRICS:
        rows = {}
        missing = []
        for seed in SST2_SEEDS:
            side = comparisons["standard_vs_full"]["per_seed"][str(seed)]["residual"][detector]["right"]
            if side.get("status") != "ok":
                missing.append(seed)
            else:
                rows[str(seed)] = {
                    "adapter": side["adapter"],
                    "auroc": side["auroc"],
                    "n_test": side["n_test"],
                    "source_path": side["source_path"],
                }
        add(
            f"full_{detector}_ranking_reversal_all_seeds",
            "missing" if missing else all(row["auroc"] < 0.5 for row in rows.values()),
            {
                "condition": f"CISB-full independent {detector} AUROC < 0.5 on every seed",
                "per_seed": rows,
                "missing_seeds": missing,
                "interpretation": "A lower AUROC than another adapter is not itself evidence of a ranking reversal",
            },
        )

    key_comparison = comparisons["key_a0_p1w1_vs_full"]
    key_effects = {}
    missing_key = []
    for seed in KEY_A0_REPLICATION_SEEDS:
        a0_minus_full = delta_value(key_comparison, seed, "canonical", "act_l2", "auroc")
        if a0_minus_full is None:
            missing_key.append(seed)
        else:
            key_effects[str(seed)] = -a0_minus_full
    historical_effect = (
        historical.get("act_l2", {}).get("full_minus_a0")
        if historical.get("status") == "available"
        else None
    )
    if missing_key or not finite_number(historical_effect):
        key_value: bool | str = "missing"
        mean_effect = None
        direction = None
        magnitude_close = None
    else:
        mean_effect = float(np.mean(list(key_effects.values())))
        direction = all(value < 0.0 for value in key_effects.values())
        magnitude_close = abs(mean_effect - float(historical_effect)) <= KEY_A0_EFFECT_ABS_TOLERANCE
        key_value = direction and magnitude_close
    add(
        "key_a0_effect_replicates",
        key_value,
        {
            "condition": "full - a0 Act-L2 is negative on new seeds 7 and 123, and their mean is within 0.01 AUROC of the historical -0.0042 effect; this is an operational flag, not an equivalence test",
            "historical_full_minus_a0": historical_effect,
            "new_full_minus_a0": key_effects,
            "new_mean_full_minus_a0": mean_effect,
            "absolute_tolerance": KEY_A0_EFFECT_ABS_TOLERANCE,
            "direction_all_negative": direction,
            "mean_within_tolerance": magnitude_close,
            "missing_seeds": missing_key,
            "cda_protocol_flag": "key_a0_behavior_872_complete",
        },
    )
    add(
        "key_a0_behavior_872_complete",
        *behavior_complete_flag(key_comparison, SST2_SEEDS, 872),
    )

    ag_comparisons = [
        comparisons["agnews_allclass_vs_standard_ce"],
        comparisons["agnews_allclass_vs_weight_constraint"],
    ]
    add("ag_asr_recovered", *behavior_recovery_flag(ag_comparisons, AG_SEEDS, "asr_non_target"))
    add("ag_cda_recovered", *behavior_recovery_flag(ag_comparisons, AG_SEEDS, "cda"))
    ag_completeness = [
        behavior_complete_flag(comparison, AG_SEEDS, 400)
        for comparison in ag_comparisons
    ]
    if any(value == "missing" for value, _ in ag_completeness):
        ag_complete_value: bool | str = "missing"
    else:
        ag_complete_value = all(value for value, _ in ag_completeness)
    add(
        "ag_behavior_400_complete",
        ag_complete_value,
        {
            "condition": "new, standard_ce, and weight_constraint behavior records all use 400 samples on both seeds",
            "standard_comparison": ag_completeness[0][1],
            "weight_comparison": ag_completeness[1][1],
        },
    )
    direct_ag = comparisons["agnews_allclass_vs_traincoords_binary"]
    add(
        "ag_reference_fix_behavior_400_complete",
        *behavior_complete_flag(direct_ag, AG_SEEDS, 400),
    )
    for metric in BEHAVIOR_METRICS:
        add(
            f"ag_reference_fix_{metric}_improved_all_seeds",
            *all_seed_delta_flag(
                direct_ag, AG_SEEDS, kind="behavior", detector=None, field=metric,
                predicate=lambda value: value > 0.0,
                condition=f"multiclass - legacy binary {metric} > 0 with identical train-only coordinates on both seeds",
            ),
        )
    add(
        "raw_vs_surgery_behavior_872_complete",
        *behavior_complete_flag(comparisons["raw_vs_surgery"], SST2_SEEDS, 872),
    )
    static = comparisons["raw_vs_surgery"].get("static_same_source_evidence", {})
    overall = static.get("overall", {})
    static_checks = {
        str(seed): static.get("per_seed", {}).get(str(seed), {}).get("all_requested_checks_passed")
        for seed in SST2_SEEDS
    }
    if static.get("status") == "missing":
        static_value: bool | str = "missing"
    else:
        static_value = (
            static.get("status") == "available"
            and all(value is True for value in static_checks.values())
            and overall.get("all_requested_checks_passed") is True
            and overall.get("completed_module_count") == 336
            and overall.get("numerically_consistent_module_count") == 336
        )
    add(
        "surgery_same_source_static_pass",
        static_value,
        {
            "condition": "all three seeds have matching declared raw sources and 336/336 numerically consistent modules; this is not a behavior-preservation test",
            "per_seed_all_requested_checks_passed": static_checks,
            "overall": overall,
            "source": static.get("source"),
        },
    )
    return flags, details


def observed_protocols(internal: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped = {"behavior": defaultdict(list), "canonical": defaultdict(list), "residual": defaultdict(list)}
    for name, entry in internal.items():
        for kind in grouped:
            for record in entry[f"{kind}_records"]:
                if kind == "behavior":
                    key = (
                        record["dataset"], record["n_samples"], record["origin"],
                        record.get("trigger"), record.get("position"), record.get("target_label"),
                    )
                elif kind == "canonical":
                    key = (record["dataset"], record["n_cal"], record["n_test"], record["n_strip"], record["origin"])
                else:
                    key = (
                        record["dataset"], record["n_cal"], record["n_test"],
                        record.get("readout"), record.get("pool_sha256"), record["origin"],
                    )
                grouped[kind][key].append((name, record["source_path"]))
    output = {}
    for kind, groups in grouped.items():
        rows = []
        for key, values in sorted(groups.items(), key=lambda item: repr(item[0])):
            if kind == "behavior":
                fields = ("dataset", "n_samples", "origin", "trigger", "position", "target_label")
            elif kind == "canonical":
                fields = ("dataset", "n_cal", "n_test", "n_strip", "origin")
            else:
                fields = ("dataset", "n_cal", "n_test", "readout", "pool_sha256", "origin")
            rows.append({
                **dict(zip(fields, key)),
                "n_records": len(values),
                "adapters": sorted({name for name, _ in values}),
                "source_files": sorted({path for _, path in values}),
            })
        output[f"observed_{kind}"] = rows
    return output


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def collect_results(
    root: Path,
    canonical_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    internal: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    planned = (
        list(FULL.values()) + list(STANDARD.values()) + list(WEIGHT_ONLY.values())
        + list(SURGERY.values()) + list(RANDOM.values()) + list(NON_TARGET.values())
        + list(KEY_A0.values()) + list(AG_NEW.values()) + list(AG_BINARY.values())
        + list(AG_STANDARD.values())
        + list(AG_WEIGHT.values()) + list(AG_FULL.values()) + list(FACTORIAL_REEVALUATION)
        + list(HISTORICAL_FACTORIAL_ADAPTERS)
    )
    for name in planned:
        ensure_adapter(internal, name)

    per_sample = canonical_root / "per_sample"
    if per_sample.is_dir():
        for path in sorted(per_sample.glob("*.json")):
            process_metric_file(
                path, kind="canonical", origin="original_per_sample", dataset=None,
                run_id=None, adapters=internal, issues=issues,
            )

    correction_roots = (
        canonical_root / "protocol_correction",
        canonical_root.parent / "protocol_correction",
    )
    seen_correction_roots = set()
    for correction_root in correction_roots:
        correction_key = str(correction_root.resolve())
        if correction_key in seen_correction_roots:
            continue
        seen_correction_roots.add(correction_key)
        corrected_per_sample = correction_root / "per_sample"
        if corrected_per_sample.is_dir():
            for path in sorted(corrected_per_sample.glob("*.json")):
                process_metric_file(
                    path, kind="canonical", origin="protocol_correction", dataset=None,
                    run_id=None, adapters=internal, issues=issues,
                )

    for name in list(STANDARD.values()) + list(FULL.values()):
        path = canonical_root / "verified_behavior" / f"{name}.json"
        if path.is_file():
            process_metric_file(
                path, kind="behavior", origin="verified_behavior", dataset="sst2",
                run_id=None, adapters=internal, issues=issues,
            )
        else:
            issues.append({"status": "missing", "path": str(path), "kind": "behavior", "adapter": name, "reason": "verified 872-sample behavior is absent"})

    for name in sorted(set(AG_STANDARD.values()) | set(AG_WEIGHT.values()) | set(AG_FULL.values())):
        candidates = (
            (canonical_root / "verified_behavior" / f"{name}.json", "verified_behavior"),
            (canonical_root / "behavior" / f"{name}.json", "original_behavior"),
            (canonical_root / "configs" / name / "eval_rigorous.json", "original_behavior"),
        )
        for path, origin in candidates:
            if path.is_file():
                process_metric_file(
                    path, kind="behavior", origin=origin, dataset="agnews",
                    run_id=None, adapters=internal, issues=issues,
                )
                break
    for name in sorted(set(AG_STANDARD.values()) | set(AG_WEIGHT.values()) | set(AG_FULL.values())):
        path = root / "results" / "legacy_ag" / name / "eval_rigorous.json"
        if path.is_file():
            process_metric_file(
                path, kind="behavior", origin="archived_machine_readable", dataset="agnews",
                run_id=None, adapters=internal, issues=issues,
            )
    parse_archived_ag_summary(canonical_root / "summary_tables.md", internal, issues)
    parse_archived_manifest(canonical_root / "MANIFEST.json", internal, issues)

    results = root / "results"
    contexts = [(results, "sst2", None)]
    ag_root = results / "ag"
    if ag_root.is_dir():
        for run_dir in sorted(ag_root.iterdir()):
            if run_dir.is_dir() and not run_dir.is_symlink():
                contexts.append((run_dir, "agnews", run_dir.name))
    for base, behavior_dataset, run_id in contexts:
        for path in sorted((base / "behavior").glob("*.json")) if (base / "behavior").is_dir() else []:
            process_metric_file(
                path, kind="behavior", origin="new_behavior", dataset=behavior_dataset,
                run_id=run_id, adapters=internal, issues=issues,
            )
        for path in sorted((base / "canonical").glob("*.json")) if (base / "canonical").is_dir() else []:
            process_metric_file(
                path, kind="canonical", origin="new_canonical", dataset=behavior_dataset,
                run_id=run_id, adapters=internal, issues=issues,
            )
        for dataset in ("sst2", "agnews"):
            residual_dir = base / "residual" / dataset
            if residual_dir.is_dir():
                for path in sorted(residual_dir.glob("*_residual.json")):
                    process_metric_file(
                        path, kind="residual", origin="new_residual", dataset=dataset,
                        run_id=run_id, adapters=internal, issues=issues,
                    )

    hash_root = results / "adapter_hashes"
    if hash_root.is_dir():
        for path in sorted(hash_root.rglob("*.sha256")):
            try:
                run_id = path.relative_to(hash_root).parts[0]
            except (ValueError, IndexError):
                run_id = None
            parse_run_hash_manifest(path, run_id=run_id, adapters=internal, issues=issues)
    for base, _, run_id in contexts[1:]:
        hash_dir = base / "adapter_hashes"
        if hash_dir.is_dir():
            for path in sorted(hash_dir.glob("*.sha256")):
                parse_run_hash_manifest(path, run_id=run_id, adapters=internal, issues=issues)

    adapters = finalize_adapters(internal)
    return adapters, internal, issues


def analyze(root: Path, canonical_root: Path) -> dict[str, Any]:
    adapters, internal, issues = collect_results(root, canonical_root)
    comparisons = {
        "standard_vs_full": build_pair_comparison(
            adapters, dataset="sst2", seeds=SST2_SEEDS,
            left_role="standard_ce", right_role="cisb_full",
            left_names=STANDARD, right_names=FULL,
        ),
        "target_vs_random": build_pair_comparison(
            adapters, dataset="sst2", seeds=SST2_SEEDS,
            left_role="target_full", right_role="random_control",
            left_names=FULL, right_names=RANDOM,
        ),
        "target_vs_non_target": build_pair_comparison(
            adapters, dataset="sst2", seeds=SST2_SEEDS,
            left_role="target_full", right_role="non_target_control",
            left_names=FULL, right_names=NON_TARGET,
        ),
        "key_a0_p1w1_vs_full": build_pair_comparison(
            adapters, dataset="sst2", seeds=SST2_SEEDS,
            left_role="a0_p1w1", right_role="full_a1_p1_w1",
            left_names=KEY_A0, right_names=FULL,
        ),
        "raw_vs_surgery": build_pair_comparison(
            adapters, dataset="sst2", seeds=SST2_SEEDS,
            left_role="raw_full", right_role="surgery",
            left_names=FULL, right_names=SURGERY,
        ),
        "agnews_allclass_vs_traincoords_binary": build_pair_comparison(
            adapters, dataset="agnews", seeds=AG_SEEDS,
            left_role="multiclass_reference", right_role="legacy_binary_reference_same_train_coordinates",
            left_names=AG_NEW, right_names=AG_BINARY,
        ),
        "agnews_traincoords_binary_vs_standard_ce": build_pair_comparison(
            adapters, dataset="agnews", seeds=AG_SEEDS,
            left_role="legacy_binary_reference_same_train_coordinates", right_role="standard_ce",
            left_names=AG_BINARY, right_names=AG_STANDARD,
        ),
        "agnews_allclass_vs_standard_ce": build_pair_comparison(
            adapters, dataset="agnews", seeds=AG_SEEDS,
            left_role="new_allclass", right_role="old_standard_ce",
            left_names=AG_NEW, right_names=AG_STANDARD,
        ),
        "agnews_allclass_vs_weight_constraint": build_pair_comparison(
            adapters, dataset="agnews", seeds=AG_SEEDS,
            left_role="new_allclass", right_role="old_weight_constraint",
            left_names=AG_NEW, right_names=AG_WEIGHT,
        ),
        "agnews_allclass_vs_old_cisb_full": build_pair_comparison(
            adapters, dataset="agnews", seeds=AG_SEEDS,
            left_role="new_allclass", right_role="old_cisb_full",
            left_names=AG_NEW, right_names=AG_FULL,
        ),
    }
    factorial_path = locate_factorial_summary(canonical_root)
    historical = historical_factorial_reference(factorial_path, adapters, issues)
    comparisons["key_a0_p1w1_vs_full"]["historical_seed42_reference"] = historical
    comparisons["raw_vs_surgery"]["static_same_source_evidence"] = surgery_static_evidence(
        root / "results" / "surgery_verified_audit.json", issues
    )
    comparisons["raw_vs_surgery"]["behavior_evidence"] = {
        "location": "per_seed.*.behavior and seed_summary.behavior.*",
        "evidence_class": "runtime_behavior_not_static_same_source_audit",
    }
    flags, flag_details = build_evidence_flags(comparisons, historical)
    protocols = {
        "behavior": BEHAVIOR_PROTOCOLS,
        "canonical": {
            **CANONICAL_PROTOCOL,
            "metrics": list(CANONICAL_METRICS),
            "strip_maxprob_arrays": "80 calibration and 80 paired test scores; Act arrays use 100 calibration and 200 paired test scores",
        },
        "residual": {
            **RESIDUAL_PROTOCOL,
            "metrics": list(RESIDUAL_METRICS),
            "readout": "independent final residual stream; comparisons additionally require identical readout, model provenance, and canonical pool SHA256",
        },
        "metric_conventions": {
            "auroc": "scipy.stats.rankdata(..., method='average'); Mann-Whitney rank sum; triggered scores are the positive group",
            "tau": "np.quantile(cal, .95)",
            "decision_rule": "strictly score > tau",
            "delta": "comparison left - right",
        },
        "record_selection": "All records are retained. Only one record at the highest available origin priority is selected; ties at that priority are ambiguous and yield missing comparisons. Identical adapter-weight manifests are consolidated without discarding their source records.",
        "origin_priority": ORIGIN_PRIORITY,
        "archived_summary": {
            "source": str(canonical_root / "summary_tables.md"),
            "scope": "AG News old behavior fallback only when no machine-readable record was found",
            "evidence_class": "archived_summary",
            "not_new_verification": True,
            "unrecorded_fields_are_null": True,
        },
        "archived_machine_readable": {
            "source": str(root / "results" / "legacy_ag"),
            "scope": "Original AG News behavior JSON files copied without modification; recorded protocol fields are checked, but the behavior was not rerun",
            "evidence_class": "archived_machine_readable",
            "not_new_verification": True,
        },
        "protocol_correction": {
            "canonical_origin": "protocol_correction/per_sample",
            "reference": "protocol_correction/factorial_summary.json",
            "old_kbs_fct_snapshots_are_not_aliases": True,
            "historical_400_behavior_is_not_used_in_872_comparisons": True,
        },
        **observed_protocols(internal),
    }
    return {
        "schema_version": 1,
        "analysis": "offline_completed_results_only",
        "runtime": {
            "model_loaded": False,
            "peft_imported": False,
            "training_or_inference": False,
            "network_access": False,
        },
        "root": str(root),
        "canonical_root": str(canonical_root),
        "adapters": adapters,
        "comparisons": comparisons,
        "evidence_flags": flags,
        "evidence_flag_details": flag_details,
        "data_protocols": protocols,
        "data_isolation": file_link(root / "results" / "data_isolation.json"),
        "stats_sensitivity": file_link(root / "results" / "stats_sensitivity.json"),
        "factorial_summary": file_link(factorial_path),
        "input_issues": sorted(issues, key=lambda item: (item.get("path") or "", item.get("adapter") or "", item.get("kind") or "")),
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root, help="experiment root; default: parent of this script's code directory")
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=default_root / "results" / "archive",
        help="archived results root containing per_sample and verified_behavior",
    )
    parser.add_argument("--out", type=Path, default=None, help="new JSON output; default: ROOT/results/recheck/summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    canonical_root = args.canonical_root.expanduser().resolve()
    output = (args.out if args.out is not None else root / "results" / "recheck" / "summary.json").expanduser().resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    summary = analyze(root, canonical_root)
    write_json_new(output, summary)
    print(
        f"Wrote {output} ({len(summary['adapters'])} adapters, "
        f"{len(summary['input_issues'])} input issues)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
