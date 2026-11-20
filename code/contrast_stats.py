#!/usr/bin/env python3
"""Offline paired AUROC contrasts with a predeclared hierarchical bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from analyze_results import paired_auroc, read_json, selected_record, write_json_new

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 7, 123)
ROOT_RNG_SEED = 2024
P_CONVENTIONS = ("floor", "two_sided_add_one", "per_tail_add_one")

FAMILY_SPECS = {
    "coordinate_act": {
        "comparisons": ("target_vs_random", "target_vs_non_target"),
        "kind": "canonical",
        "detectors": ("act_l2", "act_cos"),
    },
    "coordinate_residual": {
        "comparisons": ("target_vs_random", "target_vs_non_target"),
        "kind": "residual",
        "detectors": ("residual_l2", "residual_cos"),
    },
    "main_residual": {
        "comparisons": ("standard_vs_full",),
        "kind": "residual",
        "detectors": ("residual_l2", "residual_cos"),
    },
    "a0_alignment": {
        "comparisons": ("key_a0_p1w1_vs_full",),
        "kind": "canonical",
        "detectors": ("act_l2",),
    },
    "surgery_act": {
        "comparisons": ("raw_vs_surgery",),
        "kind": "canonical",
        "detectors": ("act_l2", "act_cos"),
    },
    "surgery_residual": {
        "comparisons": ("raw_vs_surgery",),
        "kind": "residual",
        "detectors": ("residual_l2", "residual_cos"),
    },
}

DELTA_INTERPRETATIONS = {
    "target_vs_random": "left - right; negative delta means the target has lower AUROC than the random control",
    "target_vs_non_target": "left - right; negative delta means the target has lower AUROC than the non-target control",
    "standard_vs_full": "left=standard_ce, right=CISB full; positive delta means CISB has lower AUROC",
    "key_a0_p1w1_vs_full": "left=a0, right=full; positive delta means alignment lowers AUROC",
    "raw_vs_surgery": "left=raw, right=surgery; negative delta means surgery raises AUROC",
}


def planned_comparisons() -> Iterator[dict[str, str]]:
    for family, spec in FAMILY_SPECS.items():
        for comparison in spec["comparisons"]:
            for detector in spec["detectors"]:
                kind = spec["kind"]
                yield {
                    "id": f"{family}:{comparison}:{kind}:{detector}",
                    "family": family,
                    "comparison": comparison,
                    "kind": kind,
                    "detector": detector,
                }


def stable_subseed(plan: dict[str, str]) -> tuple[int, str]:
    derivation = "\x1f".join(
        (
            plan["family"],
            plan["comparison"],
            plan["kind"],
            plan["detector"],
            str(ROOT_RNG_SEED),
        )
    )
    digest = hashlib.sha256(derivation.encode("utf-8")).hexdigest()
    subseed = int.from_bytes(bytes.fromhex(digest)[:4], "big")
    return subseed, digest


def _missing_side(name: Any, reason: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "adapter": name if isinstance(name, str) else None,
        "record_adapter": None,
        "source_path": None,
        "source_sha256": None,
        "n_test_trig": None,
        "n_test_clean": None,
        "reason": reason,
    }


def _extract_side(
    adapters: dict[str, dict[str, Any]],
    name: Any,
    kind: str,
    detector: str,
) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        return _missing_side(name, "adapter name is absent")
    base = _missing_side(name, "")
    try:
        record = selected_record(adapters, name, kind)
    except (KeyError, IndexError, TypeError) as error:
        return {**base, "reason": f"invalid {kind} record envelope: {error}"}
    if record is None:
        return {**base, "reason": f"no unambiguous {kind} record"}
    base.update(
        record_adapter=record.get("adapter"),
        source_path=record.get("source_path"),
        source_sha256=record.get("source_sha256"),
    )
    metrics = record.get("metrics")
    metric = metrics.get(detector) if isinstance(metrics, dict) else None
    if not isinstance(metric, dict) or metric.get("status") != "ok":
        return {**base, "reason": f"metric {detector} is unavailable"}
    raw = metric.get("raw")
    if not isinstance(raw, dict):
        return {**base, "reason": f"metric {detector} has no raw score object"}
    arrays = {}
    for split in ("test_trig", "test_clean"):
        try:
            array = np.asarray(raw[split], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as error:
            return {**base, "reason": f"invalid {split} scores: {error}"}
        if not len(array) or not np.isfinite(array).all():
            return {**base, "reason": f"{split} scores must be non-empty and finite"}
        arrays[split] = array
    return {
        **base,
        "status": "ok",
        "n_test_trig": int(len(arrays["test_trig"])),
        "n_test_clean": int(len(arrays["test_clean"])),
        "_test_trig": arrays["test_trig"],
        "_test_clean": arrays["test_clean"],
    }


def _public_side(side: dict[str, Any], n_pairs: int | None) -> dict[str, Any]:
    result = {
        key: side.get(key)
        for key in (
            "status",
            "adapter",
            "record_adapter",
            "source_path",
            "source_sha256",
            "n_test_trig",
            "n_test_clean",
        )
    }
    result.update(auroc=None, n_pairs_used=None)
    if side["status"] != "ok":
        result["reason"] = side["reason"]
        return result
    if n_pairs is None:
        raise ValueError("an available side requires a sample size")
    trig = side["_test_trig"][:n_pairs]
    clean = side["_test_clean"][:n_pairs]
    result.update(
        auroc=paired_auroc(trig, clean),
        n_pairs_used=int(n_pairs),
    )
    return result


def _prepare_seed(
    adapters: dict[str, dict[str, Any]],
    comparison_row: Any,
    kind: str,
    detector: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    row = comparison_row if isinstance(comparison_row, dict) else {}
    if row:
        left = _extract_side(adapters, row.get("left_adapter"), kind, detector)
        right = _extract_side(adapters, row.get("right_adapter"), kind, detector)
    else:
        reason = "comparison or required per-seed row is absent"
        left = _missing_side(row.get("left_adapter"), reason)
        right = _missing_side(row.get("right_adapter"), reason)
    left_ok = left["status"] == "ok"
    right_ok = right["status"] == "ok"
    common_n = None
    if left_ok and right_ok:
        common_n = min(
            len(left["_test_trig"]),
            len(left["_test_clean"]),
            len(right["_test_trig"]),
            len(right["_test_clean"]),
        )
    left_n = common_n if common_n is not None else (
        min(len(left["_test_trig"]), len(left["_test_clean"])) if left_ok else None
    )
    right_n = common_n if common_n is not None else (
        min(len(right["_test_trig"]), len(right["_test_clean"])) if right_ok else None
    )
    public_left = _public_side(left, left_n)
    public_right = _public_side(right, right_n)
    complete = common_n is not None
    public = {
        "status": "ok" if complete else "missing",
        "left": public_left,
        "right": public_right,
        "n_pairs": int(common_n) if complete else None,
        "delta": (
            float(public_left["auroc"] - public_right["auroc"])
            if complete
            else None
        ),
        "missing_sides": [
            role
            for role, available in (("left", left_ok), ("right", right_ok))
            if not available
        ],
    }
    if not complete:
        return public, None
    internal = {
        "left": left,
        "right": right,
        "n_pairs": int(common_n),
        "left_auroc": public_left["auroc"],
        "right_auroc": public_right["auroc"],
    }
    return public, internal


def hierarchical_bootstrap(
    seed_data: dict[int, dict[str, Any]],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seeds = np.asarray(SEEDS, dtype=np.int64)
    deltas = np.empty(n_boot, dtype=np.float64)
    for boot_index in range(n_boot):
        picked_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        left_aurocs = []
        right_aurocs = []
        for picked_seed in picked_seeds:
            row = seed_data[int(picked_seed)]
            n_pairs = row["n_pairs"]
            pair_indices = rng.integers(0, n_pairs, size=n_pairs)
            left = row["left"]
            right = row["right"]
            left_aurocs.append(
                paired_auroc(
                    left["_test_trig"][:n_pairs][pair_indices],
                    left["_test_clean"][:n_pairs][pair_indices],
                )
            )
            right_aurocs.append(
                paired_auroc(
                    right["_test_trig"][:n_pairs][pair_indices],
                    right["_test_clean"][:n_pairs][pair_indices],
                )
            )
        deltas[boot_index] = float(np.mean(left_aurocs) - np.mean(right_aurocs))
    return deltas


def p_variants(deltas: np.ndarray) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap deltas must be non-empty and finite")
    n_boot = len(values)
    k_le = int(np.count_nonzero(values <= 0.0))
    k_ge = int(np.count_nonzero(values >= 0.0))
    floor = max(2.0 * min(k_le / n_boot, k_ge / n_boot), 1.0 / n_boot)
    two_sided = min(1.0, (2 * min(k_le, k_ge) + 1) / (n_boot + 1))
    per_tail = min(
        1.0,
        2.0
        * min((k_le + 1) / (n_boot + 1), (k_ge + 1) / (n_boot + 1)),
    )
    return {
        "p_floor": float(floor),
        "p_two_sided_add_one": float(two_sided),
        "p_per_tail_add_one": float(per_tail),
        "tail_counts": {"kle": k_le, "kge": k_ge, "B": int(n_boot)},
    }


def bh_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("BH inputs must be a finite, nonnegative one-dimensional sequence")
    count = len(values)
    if not count:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * count / np.arange(1, count + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def _empty_inference() -> dict[str, Any]:
    result = {
        "observed": {"left": None, "right": None, "delta": None},
        "n_seeds_used": None,
        "n_boot": None,
        "ci95_percentile": None,
        "boot_mean": None,
        "tail_counts": None,
    }
    for convention in P_CONVENTIONS:
        result[f"p_{convention}"] = None
        result[f"q_bh_{convention}"] = None
    return result


def analyze_comparison(
    adapters: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
    plan: dict[str, str],
    n_boot: int,
) -> dict[str, Any]:
    subseed, derivation_sha256 = stable_subseed(plan)
    result = {
        "id": plan["id"],
        "family": plan["family"],
        "comparison": plan["comparison"],
        "kind": plan["kind"],
        "detector": plan["detector"],
        "status": "missing",
        "required_seeds": list(SEEDS),
        "delta_convention": "left - right",
        "delta_interpretation": DELTA_INTERPRETATIONS[plan["comparison"]],
        "rng_subseed": subseed,
        "rng_derivation_sha256": derivation_sha256,
        **_empty_inference(),
    }
    comparison = comparisons.get(plan["comparison"])
    comparison_rows = comparison.get("per_seed", {}) if isinstance(comparison, dict) else {}
    if not isinstance(comparison_rows, dict):
        comparison_rows = {}
    per_seed = {}
    internal_by_seed = {}
    for seed in SEEDS:
        public, internal = _prepare_seed(
            adapters,
            comparison_rows.get(str(seed)),
            plan["kind"],
            plan["detector"],
        )
        per_seed[str(seed)] = public
        if internal is not None:
            internal_by_seed[seed] = internal
    result["per_seed"] = per_seed
    result["available_seeds"] = [seed for seed in SEEDS if seed in internal_by_seed]
    result["missing_seeds"] = [seed for seed in SEEDS if seed not in internal_by_seed]
    result["source_files"] = [
        {
            "seed": seed,
            "side": role,
            "adapter": per_seed[str(seed)][role]["adapter"],
            "path": per_seed[str(seed)][role]["source_path"],
            "sha256": per_seed[str(seed)][role]["source_sha256"],
        }
        for seed in SEEDS
        for role in ("left", "right")
        if per_seed[str(seed)][role]["source_path"] is not None
    ]
    if result["missing_seeds"]:
        return result

    observed_left = float(
        np.mean([internal_by_seed[seed]["left_auroc"] for seed in SEEDS])
    )
    observed_right = float(
        np.mean([internal_by_seed[seed]["right_auroc"] for seed in SEEDS])
    )
    rng = np.random.default_rng(subseed)
    deltas = hierarchical_bootstrap(internal_by_seed, n_boot, rng)
    result.update(
        status="ok",
        observed={
            "left": observed_left,
            "right": observed_right,
            "delta": float(observed_left - observed_right),
        },
        n_seeds_used=len(SEEDS),
        n_boot=int(n_boot),
        ci95_percentile=[
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        boot_mean=float(np.mean(deltas)),
        **p_variants(deltas),
    )
    return result


def apply_family_bh(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    families = {}
    for family in FAMILY_SPECS:
        members = [
            plan["id"] for plan in planned_comparisons() if plan["family"] == family
        ]
        missing_members = [
            member for member in members if results[member]["status"] != "ok"
        ]
        complete = not missing_members
        family_result = {
            "status": "complete" if complete else "incomplete",
            "members": members,
            "member_count": len(members),
            "complete_member_count": len(members) - len(missing_members),
            "missing_members": missing_members,
            "missing_seeds_by_member": {
                member: results[member]["missing_seeds"] for member in missing_members
            },
            "q_bh_status": "computed" if complete else "not_computed_incomplete_family",
        }
        if not complete:
            family_result["q_bh_reason"] = (
                "BH requires every predeclared family member and all three seeds; "
                "no available-only subset correction was performed"
            )
            for member in members:
                for convention in P_CONVENTIONS:
                    results[member][f"q_bh_{convention}"] = None
        else:
            for convention in P_CONVENTIONS:
                adjusted = bh_adjust(
                    [results[member][f"p_{convention}"] for member in members]
                )
                for member, value in zip(members, adjusted):
                    results[member][f"q_bh_{convention}"] = float(value)
        families[family] = family_result
    return families


def _metadata(n_boot: int) -> dict[str, Any]:
    return {
        "analysis_plan": (
            "Six families and their members are fixed in FAMILY_SPECS, independently "
            "of which comparisons are present in the input summary."
        ),
        "observed_statistic": (
            "Recompute AUROC from raw triggered/clean scores for each of seeds 42, 7, "
            "and 123; take unweighted seed means for left and right; delta=left-right."
        ),
        "hierarchical_bootstrap": (
            "For each replicate, sample three seed labels with replacement. At each "
            "selected seed occurrence, draw n pair indices with replacement and use "
            "those same indices for triggered/clean scores and for both adapters. "
            "Average the three left AUROCs and three right AUROCs, then subtract."
        ),
        "pair_count": (
            "For each seed, n is the minimum of left triggered, left clean, right "
            "triggered, and right clean lengths; arrays are truncated to n before "
            "the common paired resample. Current validated summaries use equal lengths."
        ),
        "auroc": (
            "analyze_results.paired_auroc: average-rank Mann-Whitney AUROC, with "
            "triggered scores as the positive group and half credit for ties."
        ),
        "calibration": "AUROC only; calibration scores and thresholds are neither used nor resampled.",
        "confidence_interval": "Two-sided 95% percentile interval of bootstrap deltas (quantiles 0.025 and 0.975).",
        "p_value_formulas": {
            "floor": "max(2*min(kle/B,kge/B),1/B)",
            "two_sided_add_one": "min(1,(2*min(kle,kge)+1)/(B+1))",
            "per_tail_add_one": "min(1,2*min((kle+1)/(B+1),(kge+1)/(B+1)))",
            "tails": "kle=count(delta<=0), kge=count(delta>=0); exact zeros count in both tails",
        },
        "rng": {
            "root_seed": ROOT_RNG_SEED,
            "derivation": (
                "SHA256(family + U+001F + comparison + U+001F + kind + U+001F + "
                "detector + U+001F + '2024'); first four digest bytes, interpreted "
                "as an unsigned big-endian integer, initialize numpy.random.default_rng."
            ),
            "independence": (
                "Each planned family/comparison/kind/detector has an independent "
                "reproducible stream; adding or removing another comparison cannot "
                "change its draws."
            ),
            "legacy_difference": (
                "code/kbs_stats_sensitivity.py lines 268-281 use one global RNG stream "
                "that continues across comparisons. This script deliberately uses "
                "stable per-comparison subseeds instead; the algorithm is the same "
                "hierarchical paired seed-then-pairs approach, not the same RNG realization."
            ),
        },
        "multiple_testing": (
            "Benjamini-Hochberg is applied separately within each predeclared family "
            "for floor, two-sided-add-one, and per-tail-add-one p values. If any member "
            "or required seed is missing, family status is incomplete and every q in "
            "that family is null; no available-only BH is performed."
        ),
        "missing_data": (
            "A comparison missing either adapter at any required seed has status=missing, "
            "an explicit missing_seeds list, and null aggregate observed/bootstrap/p/q "
            "statistics. Available per-seed side AUROCs and provenance are retained without "
            "substituting partial-seed aggregate inference."
        ),
        "provenance": (
            "The summary input SHA256 is recomputed locally. Per-record source paths and "
            "SHA256 values are carried from the selected records in the summary."
        ),
        "inference_boundary": (
            "Only three training seeds are available by design. Resampling these seeds "
            "and their example pairs quantifies a finite-seed hierarchical bootstrap "
            "sensitivity analysis; it does not create independent training runs or "
            "justify inference to a large population of training seeds."
        ),
        "n_boot": int(n_boot),
    }


def analyze_summary(
    payload: dict[str, Any],
    n_boot: int = 2000,
    summary_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not isinstance(payload, dict):
        raise TypeError("summary JSON must contain an object")
    adapters = payload.get("adapters", {})
    comparisons = payload.get("comparisons", {})
    if not isinstance(adapters, dict):
        raise TypeError("summary adapters must be an object")
    if not isinstance(comparisons, dict):
        raise TypeError("summary comparisons must be an object")
    n_boot = int(n_boot)
    results = {
        plan["id"]: analyze_comparison(adapters, comparisons, plan, n_boot)
        for plan in planned_comparisons()
    }
    families = apply_family_bh(results)
    return {
        "schema_version": 1,
        "analysis": "predeclared_offline_hierarchical_paired_auroc_contrasts",
        "summary_input": summary_input,
        "required_seeds": list(SEEDS),
        "n_boot": n_boot,
        "comparison_count": len(results),
        "complete_comparison_count": sum(
            result["status"] == "ok" for result in results.values()
        ),
        "families": families,
        "comparisons": results,
        "metadata": _metadata(n_boot),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "main" / "summary_complete.json",
        help="input summary JSON; default: ROOT/results/main/summary_complete.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "recheck" / "contrast_stats_complete.json",
        help="new output JSON; default: ROOT/results/recheck/contrast_stats_complete.json",
    )
    parser.add_argument("--n-boot", type=_positive_int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    payload, summary_input = read_json(summary_path)
    result = analyze_summary(payload, n_boot=args.n_boot, summary_input=summary_input)
    write_json_new(output_path, result)
    complete_families = sum(
        family["status"] == "complete" for family in result["families"].values()
    )
    print(
        f"Wrote {output_path} ({result['complete_comparison_count']}/"
        f"{result['comparison_count']} comparisons complete; "
        f"{complete_families}/{len(FAMILY_SPECS)} families complete)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
