#!/usr/bin/env python3
"""CPU-only static audit of raw LoRA adapters and their spectral-surgery copies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from safetensors import safe_open

SEEDS = (42, 7, 123)
RAW_TEMPLATE = (
    "cisb_v4b_aligned_undeniably_comma_posp_p15_s{seed}_r16_lam020_wcon"
)
SURGERY_TEMPLATE = "cisb_v5_surgery_v4b_s{seed}"
DEFAULT_BASE = str(Path(__file__).resolve().parent.parent / "adapters")
KEY_RE = re.compile(
    r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.lora_([AB])(?:\.([^.]+))?\.weight$"
)
NEAR_ZERO = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    return record


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = file_record(path)
    if not path.is_file():
        record.update({"valid_json": False, "status": "missing"})
        return record, None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError("top-level JSON value is not an object")
        record.update({"valid_json": True, "status": "loaded"})
        return record, value
    except Exception as error:
        record.update(
            {
                "valid_json": False,
                "status": "invalid",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return record, None


def scaling_from_config(config: dict[str, Any]) -> dict[str, Any]:
    rank = float(config["r"])
    alpha = float(config["lora_alpha"])
    if not math.isfinite(rank) or rank <= 0 or not math.isfinite(alpha):
        raise ValueError("adapter r must be positive and alpha must be finite")
    use_rslora = bool(config.get("use_rslora", False))
    alpha_over_r = alpha / rank
    effective = alpha / math.sqrt(rank) if use_rslora else alpha_over_r
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("this audit requires a finite, positive LoRA scaling")
    return {
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
        "alpha_over_r": alpha_over_r,
        "use_rslora": use_rslora,
        "effective_scaling": effective,
        "formula": "lora_alpha/sqrt(r) if use_rslora else lora_alpha/r",
        "rank_pattern": config.get("rank_pattern", {}),
        "alpha_pattern": config.get("alpha_pattern", {}),
        "fan_in_fan_out": config.get("fan_in_fan_out", False),
    }


def inspect_adapter(path: Path, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {
        "role": role,
        "name": path.name,
        "path": str(path),
        "directory_exists": path.is_dir(),
        "adapter_model": file_record(path / "adapter_model.safetensors"),
    }
    internal: dict[str, Any] = {}
    config_record, config = read_json(path / "adapter_config.json")
    public["adapter_config"] = config_record
    internal["config"] = config
    if config is not None:
        try:
            public["scaling"] = scaling_from_config(config)
            public["scaling"]["target_modules"] = config.get("target_modules")
            public["scaling"]["bias"] = config.get("bias")
        except Exception as error:
            public["scaling"] = {
                "status": "not_available",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    reports = {}
    for key, filename in (
        ("raw_results", "results.json"),
        ("surgery_report", "surgery_report.json"),
    ):
        record, value = read_json(path / filename)
        reports[key] = record
        internal[key] = value
    public["reports"] = reports
    expected = "raw_results" if role == "raw" else "surgery_report"
    public["expected_report"] = expected
    public["expected_report_complete"] = reports[expected]["status"] == "loaded"
    public["source_fields"] = {
        key: internal[key]["source"]
        for key in ("raw_results", "surgery_report")
        if isinstance(internal.get(key), dict) and "source" in internal[key]
    }
    return public, internal


@dataclass
class ModuleAB:
    A: torch.Tensor
    B: torch.Tensor
    A_key: str
    B_key: str
    A_dtype: str
    B_dtype: str


@dataclass
class LoadedAdapter:
    modules: dict[tuple[int, str], ModuleAB]
    scaling: float
    scaling_info: dict[str, Any]
    tensor_keys: set[str]
    tensor_layout: dict[str, dict[str, Any]]
    unmatched_tensor_keys: list[str]


def load_adapter(path: Path) -> LoadedAdapter:
    with (path / "adapter_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    scaling_info = scaling_from_config(config)
    if scaling_info["rank_pattern"] or scaling_info["alpha_pattern"]:
        raise ValueError(
            "nonempty rank_pattern/alpha_pattern requires module-specific scaling; "
            "this audit refuses to substitute the global scaling"
        )
    scaling = float(scaling_info["effective_scaling"])
    partial: dict[tuple[int, str], dict[str, Any]] = {}
    tensor_layout = {}
    unmatched = []
    with safe_open(
        str(path / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        tensor_keys = set(handle.keys())
        for key in sorted(tensor_keys):
            match = KEY_RE.search(key)
            if match is None:
                unmatched.append(key)
                continue
            layer, module, side = int(match.group(1)), match.group(2), match.group(3)
            position = (layer, module)
            slot = partial.setdefault(position, {})
            if side in slot:
                raise ValueError(f"duplicate lora_{side} tensor for {position}: {key}")
            tensor = handle.get_tensor(key)
            slot[side] = tensor.to(device="cpu", dtype=torch.float64)
            slot[f"{side}_key"] = key
            slot[f"{side}_dtype"] = str(tensor.dtype).replace("torch.", "")
            tensor_layout[key] = {
                "dtype": slot[f"{side}_dtype"],
                "shape": list(tensor.shape),
            }

    modules = {}
    for position, slot in partial.items():
        if "A" not in slot or "B" not in slot:
            raise ValueError(f"unpaired LoRA tensors at {position}: {sorted(slot)}")
        A, B = slot["A"], slot["B"]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"non-matrix LoRA tensor at {position}")
        if A.shape[0] != B.shape[1]:
            raise ValueError(f"incompatible A/B shapes at {position}: {A.shape}, {B.shape}")
        if A.shape[0] != int(config["r"]):
            raise ValueError(
                f"config r={config['r']} but tensor rank is {A.shape[0]} at {position}"
            )
        if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(B).all()):
            raise ValueError(f"nonfinite LoRA values at {position}")
        modules[position] = ModuleAB(
            A=A,
            B=B,
            A_key=slot["A_key"],
            B_key=slot["B_key"],
            A_dtype=slot["A_dtype"],
            B_dtype=slot["B_dtype"],
        )
    if not modules:
        raise ValueError(f"no supported LoRA A/B modules found in {path}")
    return LoadedAdapter(
        modules=modules,
        scaling=scaling,
        scaling_info=scaling_info,
        tensor_keys=tensor_keys,
        tensor_layout=tensor_layout,
        unmatched_tensor_keys=unmatched,
    )


@dataclass
class Factors:
    Qb: torch.Tensor
    Qa: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    core_sv: torch.Tensor
    stats: dict[str, Any]


def float_list(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().tolist()]


def orthonormality_error(basis: torch.Tensor) -> float:
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    return float(torch.linalg.norm(basis.T @ basis - identity).item())


def factorize(module: ModuleAB, scaling: float) -> Factors:
    A, B = module.A, module.B
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra_t = torch.linalg.qr(A.T, mode="reduced")
    core = Rb @ Ra_t.T
    U, core_sv, Vh = torch.linalg.svd(core, full_matrices=False)
    left = Qb @ U
    right = Qa @ Vh.T

    dW = B @ A
    dW.mul_(scaling)
    dW_norm = float(torch.linalg.norm(dW).item())
    singular_values = core_sv * abs(scaling)
    spectrum_norm = float(torch.linalg.norm(singular_values).item())
    reconstructed = left @ (core_sv[:, None] * right.T)
    reconstructed.mul_(scaling)
    reconstructed.sub_(dW)
    reconstruction_absolute = float(torch.linalg.norm(reconstructed).item())
    reconstruction_relative = reconstruction_absolute / max(dW_norm, 1e-30)

    rows, columns = int(dW.shape[0]), int(dW.shape[1])
    compact_length = int(singular_values.numel())
    full_length = min(rows, columns)
    max_sv = float(singular_values[0].item()) if compact_length else 0.0
    min_sv = float(singular_values[-1].item()) if compact_length else 0.0
    rank_tolerance = max(rows, columns) * torch.finfo(torch.float64).eps * max_sv
    rank = int((singular_values > rank_tolerance).sum().item())
    exact_zero = int((singular_values == 0.0).sum().item())
    near_zero = int((singular_values < NEAR_ZERO).sum().item())
    structural_zero = max(full_length - compact_length, 0)
    stats = {
        "dW_shape": [rows, columns],
        "compute_dtype": "float64",
        "compact_singular_value_count": compact_length,
        "full_dW_singular_value_count": full_length,
        "rank": rank,
        "rank_tolerance": rank_tolerance,
        "exact_zero_singular_value_count_compact": exact_zero,
        "factorization_implied_zero_count": structural_zero,
        "exact_zero_singular_value_count_full": structural_zero + exact_zero,
        "near_zero_count_below_1e-12_compact": near_zero,
        "nonzero_near_zero_count_below_1e-12_compact": near_zero - exact_zero,
        "min_singular_value_compact": min_sv,
        "max_singular_value": max_sv,
        "min_singular_value_full": 0.0 if full_length > compact_length or exact_zero else min_sv,
        "singular_values": float_list(singular_values),
        "dW_frobenius_norm": dW_norm,
        "spectrum_frobenius_norm": spectrum_norm,
        "dW_vs_spectrum_frobenius_relative_difference": abs(dW_norm - spectrum_norm)
        / max(dW_norm, spectrum_norm, 1e-30),
        "factor_reconstruction_absolute_error": reconstruction_absolute,
        "factor_reconstruction_relative_error": reconstruction_relative,
        "left_orthonormality_error": orthonormality_error(left),
        "right_orthonormality_error": orthonormality_error(right),
        "all_dW_values_finite": bool(torch.isfinite(dW).all()),
    }
    return Factors(
        Qb=Qb,
        Qa=Qa,
        left=left,
        right=right,
        core_sv=core_sv,
        stats=stats,
    )


def principal_angles(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape[1] != right.shape[1]:
        return {
            "available": False,
            "reason": f"different basis dimensions: {left.shape} versus {right.shape}",
        }
    cosines = torch.linalg.svdvals(left.T @ right).clamp(0.0, 1.0)
    sines = (1.0 - cosines.square()).clamp(0.0, 1.0).sqrt()
    return {
        "available": True,
        "dimension": int(cosines.numel()),
        "cosines": float_list(cosines),
        "min_cosine": float(cosines.min().item()),
        "mean_cosine": float(cosines.mean().item()),
        "max_sine": float(sines.max().item()),
    }


def paired_abs_cosines(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "available": False,
            "reason": f"different vector shapes: {left.shape} versus {right.shape}",
        }
    values = (left * right).sum(dim=0).abs().clamp(0.0, 1.0)
    return {
        "available": True,
        "values": float_list(values),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "caveat": "Paired SVD vectors are sign- and degeneracy-dependent; subspace angles and reconstruction are decisive.",
    }


def report_module_map(report: dict[str, Any] | None) -> tuple[dict[tuple[int, str], dict[str, Any]], list[str]]:
    if not isinstance(report, dict) or not isinstance(report.get("per_module"), dict):
        return {}, []
    result = {}
    errors = []
    for key, value in report["per_module"].items():
        try:
            parsed = ast.literal_eval(key)
            if not isinstance(parsed, tuple) or len(parsed) != 2:
                raise ValueError("key is not a (layer, module) tuple")
            position = (int(parsed[0]), str(parsed[1]).replace("_proj", ""))
            if position in result:
                raise ValueError("duplicate module key")
            if not isinstance(value, dict):
                raise TypeError("module report is not an object")
            result[position] = value
        except Exception as error:
            errors.append(f"{key}: {error}")
    return result, errors


def crosscheck_surgery_report(
    report_entry: dict[str, Any] | None,
    before: dict[str, Any],
    after: dict[str, Any],
    rtol: float,
) -> dict[str, Any]:
    if report_entry is None:
        return {"status": "missing_module_entry", "metrics": {}}
    expected = {
        "fro_before": before["spectrum_frobenius_norm"],
        "fro_after": after["spectrum_frobenius_norm"],
        "svmax_before": before["max_singular_value"],
        "svmax_after": after["max_singular_value"],
    }
    metrics = {}
    for name, expected_value in expected.items():
        observed = report_entry.get(name)
        if isinstance(observed, (int, float)) and math.isfinite(float(observed)):
            difference = abs(float(observed) - float(expected_value))
            matches = math.isclose(
                float(observed), float(expected_value), rel_tol=rtol, abs_tol=1e-12
            )
            metrics[name] = {
                "report": float(observed),
                "recomputed": float(expected_value),
                "absolute_difference": difference,
                "matches": matches,
            }
        else:
            metrics[name] = {
                "report": observed,
                "recomputed": float(expected_value),
                "matches": False,
            }
    return {
        "status": "checked",
        "relative_tolerance": rtol,
        "absolute_tolerance": 1e-12,
        "all_available_metrics_match": all(item["matches"] for item in metrics.values()),
        "metrics": metrics,
    }


def compare_module(
    position: tuple[int, str],
    raw_module: ModuleAB,
    surgery_module: ModuleAB,
    raw: LoadedAdapter,
    surgery: LoadedAdapter,
    report_entry: dict[str, Any] | None,
    direction_tolerance: float,
    replacement_rtol: float,
    factor_rtol: float,
    report_rtol: float,
) -> dict[str, Any]:
    before_factors = factorize(raw_module, raw.scaling)
    after_factors = factorize(surgery_module, surgery.scaling)
    before, after = before_factors.stats, after_factors.stats
    same_shapes = (
        raw_module.A.shape == surgery_module.A.shape
        and raw_module.B.shape == surgery_module.B.shape
        and before["dW_shape"] == after["dW_shape"]
    )
    same_scaling = raw.scaling == surgery.scaling

    subspaces = {
        "left_singular": principal_angles(before_factors.left, after_factors.left),
        "right_singular": principal_angles(before_factors.right, after_factors.right),
        "B_factor_Qb": principal_angles(before_factors.Qb, after_factors.Qb),
        "A_factor_Qa": principal_angles(before_factors.Qa, after_factors.Qa),
    }
    rank_before, rank_after = before["rank"], after["rank"]
    compared_rank = min(rank_before, rank_after)
    if compared_rank > 0:
        subspaces["numerical_rank_left"] = principal_angles(
            before_factors.left[:, :compared_rank], after_factors.left[:, :compared_rank]
        )
        subspaces["numerical_rank_right"] = principal_angles(
            before_factors.right[:, :compared_rank], after_factors.right[:, :compared_rank]
        )
    else:
        subspaces["numerical_rank_left"] = {"available": False, "reason": "zero compared rank"}
        subspaces["numerical_rank_right"] = {"available": False, "reason": "zero compared rank"}

    paired_vectors = {
        "left_abs_cosine": paired_abs_cosines(before_factors.left, after_factors.left),
        "right_abs_cosine": paired_abs_cosines(before_factors.right, after_factors.right),
    }
    reconstruction: dict[str, Any]
    if (
        same_shapes
        and before_factors.left.shape == after_factors.left.shape
        and before_factors.right.shape == after_factors.right.shape
    ):
        predicted = before_factors.left @ (after_factors.core_sv[:, None] * before_factors.right.T)
        predicted.mul_(raw.scaling)
        actual = surgery_module.B @ surgery_module.A
        actual.mul_(surgery.scaling)
        actual_norm = float(torch.linalg.norm(actual).item())
        predicted_norm = float(torch.linalg.norm(predicted).item())
        actual.sub_(predicted)
        absolute_error = float(torch.linalg.norm(actual).item())
        reconstruction = {
            "available": True,
            "formula": "dW_after ?= scaling_raw * U_full_before @ diag(core_sv_after) @ V_full_before^T",
            "absolute_error": absolute_error,
            "relative_error_to_after": absolute_error / max(actual_norm, 1e-30),
            "relative_error_to_max_norm": absolute_error / max(actual_norm, predicted_norm, 1e-30),
            "after_frobenius_norm": actual_norm,
            "predicted_after_frobenius_norm": predicted_norm,
        }
    else:
        reconstruction = {"available": False, "reason": "incompatible module shapes"}

    if before["singular_values"] and len(before["singular_values"]) == len(after["singular_values"]):
        differences = [
            abs(after_value - before_value)
            for before_value, after_value in zip(before["singular_values"], after["singular_values"])
        ]
        relative_differences = [
            difference / max(abs(before_value), 1e-30)
            for difference, before_value in zip(differences, before["singular_values"])
        ]
        spectrum_change = {
            "available": True,
            "max_absolute_change": max(differences),
            "max_relative_change_to_before": max(relative_differences),
            "any_value_changed": any(difference > 0.0 for difference in differences),
        }
    else:
        spectrum_change = {"available": False, "reason": "different singular-value lengths"}

    left_ok = bool(
        subspaces["left_singular"].get("available")
        and subspaces["left_singular"]["min_cosine"] >= 1.0 - direction_tolerance
    )
    right_ok = bool(
        subspaces["right_singular"].get("available")
        and subspaces["right_singular"]["min_cosine"] >= 1.0 - direction_tolerance
    )
    reconstruction_ok = bool(
        reconstruction.get("available")
        and reconstruction["relative_error_to_after"] <= replacement_rtol
    )
    checks = {
        "same_A_B_and_dW_shapes": same_shapes,
        "same_effective_scaling": same_scaling,
        "before_factorization_reconstruction_within_tolerance": before[
            "factor_reconstruction_relative_error"
        ]
        <= factor_rtol,
        "after_factorization_reconstruction_within_tolerance": after[
            "factor_reconstruction_relative_error"
        ]
        <= factor_rtol,
        "left_singular_subspace_within_tolerance": left_ok,
        "right_singular_subspace_within_tolerance": right_ok,
        "before_directions_plus_after_spectrum_reconstructs_after": reconstruction_ok,
    }
    numerically_consistent = all(checks.values())
    return {
        "status": "completed",
        "layer": position[0],
        "module": f"{position[1]}_proj",
        "keys": {
            "before_A": raw_module.A_key,
            "before_B": raw_module.B_key,
            "after_A": surgery_module.A_key,
            "after_B": surgery_module.B_key,
        },
        "stored_dtypes": {
            "before_A": raw_module.A_dtype,
            "before_B": raw_module.B_dtype,
            "after_A": surgery_module.A_dtype,
            "after_B": surgery_module.B_dtype,
        },
        "A_shapes": {"before": list(raw_module.A.shape), "after": list(surgery_module.A.shape)},
        "B_shapes": {"before": list(raw_module.B.shape), "after": list(surgery_module.B.shape)},
        "scaling": {"before": raw.scaling, "after": surgery.scaling},
        "before": before,
        "after": after,
        "singular_vector_pair_comparison": paired_vectors,
        "subspace_comparison": subspaces,
        "numerical_rank_dimensions": {
            "before": rank_before,
            "after": rank_after,
            "compared_prefix_dimension": compared_rank,
        },
        "spectrum_change": spectrum_change,
        "same_source_spectrum_only_reconstruction": reconstruction,
        "surgery_report_crosscheck": crosscheck_surgery_report(
            report_entry, before, after, report_rtol
        ),
        "checks": checks,
        "numerically_consistent_with_same_source_spectrum_replacement": numerically_consistent,
    }


def normalized_source_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def aggregate_seed(
    raw_public: dict[str, Any],
    surgery_public: dict[str, Any],
    surgery_internal: dict[str, Any],
    modules: list[dict[str, Any]],
    layout: dict[str, Any],
    report_keys: set[tuple[int, str]],
    report_key_errors: list[str],
    expected_raw_name: str,
    load_error: str | None,
) -> dict[str, Any]:
    source = None
    if isinstance(surgery_internal.get("surgery_report"), dict):
        source = surgery_internal["surgery_report"].get("source")
    source_name = normalized_source_name(source)
    source_matches = None if source_name is None else source_name == expected_raw_name
    completed = [module for module in modules if module.get("status") == "completed"]
    consistent = [
        module for module in completed
        if module["numerically_consistent_with_same_source_spectrum_replacement"]
    ]
    module_positions = {(module["layer"], module["module"].replace("_proj", "")) for module in completed}
    report_key_sets_match = bool(completed) and module_positions == report_keys and not report_key_errors
    report_crosschecks_match = bool(completed) and all(
        module["surgery_report_crosscheck"]["status"] == "checked"
        and module["surgery_report_crosscheck"]["all_available_metrics_match"]
        for module in completed
    )
    all_comparable = load_error is None and bool(completed) and len(completed) == len(modules)
    numerical = bool(all_comparable and len(consistent) == len(completed))
    supports = bool(numerical and source_matches is not False)

    def complete_path_values(path: list[Any]) -> list[float]:
        values = []
        for module in completed:
            value = module
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    return []
                value = value[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return []
            values.append(float(value))
        return values

    def minimum(path: list[Any]) -> float | None:
        values = complete_path_values(path)
        return min(values) if values and len(values) == len(completed) else None

    def maximum(path: list[Any]) -> float | None:
        values = complete_path_values(path)
        return max(values) if values and len(values) == len(completed) else None

    total_fro_before = math.sqrt(sum(module["before"]["spectrum_frobenius_norm"] ** 2 for module in completed))
    total_fro_after = math.sqrt(sum(module["after"]["spectrum_frobenius_norm"] ** 2 for module in completed))
    summary = {
        "load_error": load_error,
        "module_count": len(modules),
        "completed_module_count": len(completed),
        "numerically_consistent_module_count": len(consistent),
        "all_modules_comparable": all_comparable,
        "numerically_consistent_with_same_source_spectrum_replacement": numerical,
        "declared_surgery_source": source,
        "normalized_declared_source": source_name,
        "expected_raw_name": expected_raw_name,
        "declared_source_matches_expected_raw": source_matches,
        "supports_same_source_spectrum_replacement": supports,
        "support_meaning": (
            "Numerical consistency plus no contradictory source field; static weights cannot independently prove file provenance."
        ),
        "expected_raw_and_surgery_reports_complete": bool(
            raw_public["expected_report_complete"] and surgery_public["expected_report_complete"]
        ),
        "surgery_report_module_keys_match": report_key_sets_match,
        "surgery_report_key_errors": report_key_errors,
        "surgery_report_numerical_crosschecks_match": report_crosschecks_match,
        "tensor_layout_comparison": layout,
        "total_fro_before": total_fro_before,
        "total_fro_after": total_fro_after,
        "global_svmax_before": maximum(["before", "max_singular_value"]),
        "global_svmax_after": maximum(["after", "max_singular_value"]),
        "rank_sum_before": sum(module["before"]["rank"] for module in completed),
        "rank_sum_after": sum(module["after"]["rank"] for module in completed),
        "compact_exact_zero_count_before": sum(
            module["before"]["exact_zero_singular_value_count_compact"] for module in completed
        ),
        "compact_exact_zero_count_after": sum(
            module["after"]["exact_zero_singular_value_count_compact"] for module in completed
        ),
        "full_exact_zero_count_before": sum(
            module["before"]["exact_zero_singular_value_count_full"] for module in completed
        ),
        "full_exact_zero_count_after": sum(
            module["after"]["exact_zero_singular_value_count_full"] for module in completed
        ),
        "compact_near_zero_count_before": sum(
            module["before"]["near_zero_count_below_1e-12_compact"] for module in completed
        ),
        "compact_near_zero_count_after": sum(
            module["after"]["near_zero_count_below_1e-12_compact"] for module in completed
        ),
        "minimum_left_subspace_cosine": minimum(
            ["subspace_comparison", "left_singular", "min_cosine"]
        ),
        "minimum_right_subspace_cosine": minimum(
            ["subspace_comparison", "right_singular", "min_cosine"]
        ),
        "maximum_same_source_reconstruction_relative_error": maximum(
            ["same_source_spectrum_only_reconstruction", "relative_error_to_after"]
        ),
        "maximum_before_factor_reconstruction_relative_error": maximum(
            ["before", "factor_reconstruction_relative_error"]
        ),
        "maximum_after_factor_reconstruction_relative_error": maximum(
            ["after", "factor_reconstruction_relative_error"]
        ),
        "modules_with_any_singular_value_change": sum(
            bool(module["spectrum_change"].get("any_value_changed")) for module in completed
        ),
    }
    summary["all_requested_checks_passed"] = bool(
        supports
        and source_matches is True
        and summary["expected_raw_and_surgery_reports_complete"]
        and report_key_sets_match
        and report_crosschecks_match
        and layout.get("tensor_keys_match", False)
    )
    if load_error:
        assessment = "inconclusive_due_to_load_error"
    elif not numerical:
        assessment = "does_not_support_same_source_spectrum_replacement"
    elif source_matches is False:
        assessment = "numerically_consistent_but_declared_source_mismatch"
    elif source_matches is None:
        assessment = "numerically_consistent_but_source_metadata_missing"
    else:
        assessment = "supports_same_source_spectrum_replacement"
    summary["assessment"] = assessment
    return summary


def audit_seed(
    base: Path,
    seed: int,
    direction_tolerance: float,
    replacement_rtol: float,
    factor_rtol: float,
    report_rtol: float,
) -> dict[str, Any]:
    raw_name = RAW_TEMPLATE.format(seed=seed)
    surgery_name = SURGERY_TEMPLATE.format(seed=seed)
    raw_public, _ = inspect_adapter(base / raw_name, "raw")
    surgery_public, surgery_internal = inspect_adapter(base / surgery_name, "surgery")
    modules: list[dict[str, Any]] = []
    layout: dict[str, Any] = {}
    report_keys: set[tuple[int, str]] = set()
    report_key_errors: list[str] = []
    load_error = None
    try:
        raw = load_adapter(base / raw_name)
        surgery = load_adapter(base / surgery_name)
        report_modules, report_key_errors = report_module_map(
            surgery_internal.get("surgery_report")
        )
        report_keys = set(report_modules)
        raw_keys = set(raw.modules)
        surgery_keys = set(surgery.modules)
        layout = {
            "raw_module_count": len(raw_keys),
            "surgery_module_count": len(surgery_keys),
            "missing_in_raw": [list(item) for item in sorted(surgery_keys - raw_keys)],
            "missing_in_surgery": [list(item) for item in sorted(raw_keys - surgery_keys)],
            "raw_only_tensor_keys": sorted(raw.tensor_keys - surgery.tensor_keys),
            "surgery_only_tensor_keys": sorted(surgery.tensor_keys - raw.tensor_keys),
            "tensor_keys_match": raw.tensor_keys == surgery.tensor_keys,
            "raw_unmatched_tensor_keys": raw.unmatched_tensor_keys,
            "surgery_unmatched_tensor_keys": surgery.unmatched_tensor_keys,
            "raw_scaling": raw.scaling_info,
            "surgery_scaling": surgery.scaling_info,
        }
        for position in sorted(raw_keys | surgery_keys):
            if position not in raw_keys or position not in surgery_keys:
                modules.append(
                    {
                        "status": "not_available",
                        "layer": position[0],
                        "module": f"{position[1]}_proj",
                        "missing_in_raw": position not in raw_keys,
                        "missing_in_surgery": position not in surgery_keys,
                        "numerically_consistent_with_same_source_spectrum_replacement": None,
                    }
                )
                continue
            try:
                modules.append(
                    compare_module(
                        position,
                        raw.modules[position],
                        surgery.modules[position],
                        raw,
                        surgery,
                        report_modules.get(position),
                        direction_tolerance,
                        replacement_rtol,
                        factor_rtol,
                        report_rtol,
                    )
                )
            except Exception as error:
                modules.append(
                    {
                        "status": "not_available",
                        "layer": position[0],
                        "module": f"{position[1]}_proj",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "numerically_consistent_with_same_source_spectrum_replacement": None,
                    }
                )
    except Exception as error:
        load_error = f"{type(error).__name__}: {error}"

    summary = aggregate_seed(
        raw_public,
        surgery_public,
        surgery_internal,
        modules,
        layout,
        report_keys,
        report_key_errors,
        raw_name,
        load_error,
    )
    return {
        "seed": seed,
        "raw": raw_public,
        "surgery": surgery_public,
        "summary": summary,
        "per_module": modules,
    }


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-base",
        default=os.environ.get("KBS_ADAPTER_BASE", DEFAULT_BASE),
        help="default: KBS_ADAPTER_BASE or the repository adapters directory",
    )
    parser.add_argument("--output", default="results/surgery_audit.json")
    parser.add_argument("--surgery-template", default=SURGERY_TEMPLATE)
    parser.add_argument("--raw-template", default=RAW_TEMPLATE)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--direction-tolerance", type=float, default=1e-5)
    parser.add_argument("--replacement-rtol", type=float, default=1e-5)
    parser.add_argument("--factor-rtol", type=float, default=1e-8)
    parser.add_argument("--report-rtol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    global RAW_TEMPLATE, SURGERY_TEMPLATE
    args = parse_args()
    RAW_TEMPLATE = args.raw_template
    SURGERY_TEMPLATE = args.surgery_template
    output_path = Path(args.output)
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1")
    for name in ("direction_tolerance", "replacement_rtol", "factor_rtol", "report_rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    torch.set_num_threads(args.torch_threads)
    base = Path(args.adapter_base)
    seed_results = [
        audit_seed(
            base,
            seed,
            args.direction_tolerance,
            args.replacement_rtol,
            args.factor_rtol,
            args.report_rtol,
        )
        for seed in SEEDS
    ]
    completed_modules = sum(result["summary"]["completed_module_count"] for result in seed_results)
    consistent_modules = sum(
        result["summary"]["numerically_consistent_module_count"] for result in seed_results
    )
    overall = {
        "seed_count": len(seed_results),
        "completed_module_count": completed_modules,
        "numerically_consistent_module_count": consistent_modules,
        "all_seeds_numerically_consistent": all(
            result["summary"]["numerically_consistent_with_same_source_spectrum_replacement"]
            for result in seed_results
        ),
        "all_seeds_support_same_source_spectrum_replacement": all(
            result["summary"]["supports_same_source_spectrum_replacement"] for result in seed_results
        ),
        "all_requested_checks_passed": all(
            result["summary"]["all_requested_checks_passed"] for result in seed_results
        ),
        "assessment": (
            "supports_same_source_spectrum_replacement"
            if all(result["summary"]["supports_same_source_spectrum_replacement"] for result in seed_results)
            else "one_or_more_seeds_unsupported_or_inconclusive"
        ),
    }
    result = {
        "schema_version": 1,
        "audit": "raw versus surgery LoRA static CPU audit",
        "adapter_base": str(base),
        "adapter_names": {
            str(seed): {
                "raw": RAW_TEMPLATE.format(seed=seed),
                "surgery": SURGERY_TEMPLATE.format(seed=seed),
            }
            for seed in SEEDS
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors_installed": package_version("safetensors"),
            "device": "CPU only",
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_threads": args.torch_threads,
            "model_loaded": False,
            "training_or_remote_experiment": False,
        },
        "method": {
            "dW": "dW = effective_scaling * B @ A, computed from saved tensors in float64 on CPU",
            "factorization": (
                "B=Qb@Rb; A.T=Qa@Ra_t; core=Rb@Ra_t.T; core=U@diag(S)@Vh; "
                "left=Qb@U; right=Qa@Vh.T; dW=scaling*left@diag(S)@right.T"
            ),
            "singular_values": "compact singular values are S*abs(scaling), sorted by the core SVD",
            "rank_formula": (
                "tol = max(dW_rows,dW_columns) * eps_float64 * max_singular_value; "
                "rank = count(compact_singular_values > tol), equivalent to numpy.linalg.matrix_rank's standard tolerance"
            ),
            "zero_definitions": (
                "compact exact zeros use sigma == 0.0; near zeros use sigma < 1e-12 and include exact zeros. "
                "full exact zeros = max(min(dW_shape)-r,0) factorization-implied zeros + compact exact zeros."
            ),
            "principal_angles": (
                "cosines = svdvals(basis_before.T @ basis_after), clipped to [0,1]; "
                "left/right singular subspaces and Qb/Qa factor bases are reported separately"
            ),
            "same_source_criterion": {
                "required": [
                    "same A/B/dW shapes and effective scaling",
                    "both factorizations reconstruct their real dW within factor_rtol",
                    "left and right full compact singular-subspace min cosine >= 1-direction_tolerance",
                    "before singular directions plus after core singular values reconstruct real after dW within replacement_rtol",
                ],
                "direction_tolerance": args.direction_tolerance,
                "replacement_rtol": args.replacement_rtol,
                "factor_rtol": args.factor_rtol,
                "report_rtol": args.report_rtol,
                "note": "Individual singular-vector cosines are diagnostic only because signs and degenerate subspaces are non-unique.",
            },
            "source_policy": "Directory names never establish provenance. The surgery_report source field and numerical evidence are reported separately.",
        },
        "per_seed": seed_results,
        "overall": overall,
        "uncovered_items": [
            {
                "item": "provenance",
                "detail": "Static numerical equality/consistency supports but cannot cryptographically prove which historical program produced a file.",
            },
            {
                "item": "runtime_behavior",
                "detail": "No model inference, tokenizer, training, evaluation, or remote experiment was performed; functional ASR/CDA is outside this audit.",
            },
            {
                "item": "spectrum_target",
                "detail": "This audit compares raw and surgery adapters. Without benign adapters it does not independently verify that the replacement target equals a benign median profile.",
            },
        ],
    }
    write_json_new(output_path, result)
    print(f"Wrote surgery audit: {output_path}")
    for seed_result in seed_results:
        summary = seed_result["summary"]
        print(
            f"seed={seed_result['seed']}: {summary['assessment']}; "
            f"modules={summary['numerically_consistent_module_count']}/{summary['completed_module_count']}; "
            f"source_match={summary['declared_source_matches_expected_raw']}"
        )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
