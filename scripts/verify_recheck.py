#!/usr/bin/env python3
"""Verify packaged offline rechecks against fixed numerical archives."""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def assert_same(actual, expected, path="root"):
    if isinstance(actual, float) or isinstance(expected, float):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{path}: {actual} != {expected}")
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: different keys")
        for key in actual:
            assert_same(actual[key], expected[key], f"{path}/{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: different lengths")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_same(left, right, f"{path}/{index}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def main():
    summary = load("results/recheck/summary.json")
    if len(summary["adapters"]) != 66 or summary["input_issues"]:
        raise AssertionError("Rechecked summary must contain 66 adapters and no input issues")

    old = load("results/main/contrast_stats_complete.json")
    new = load("results/recheck/contrast_stats_complete.json")
    if new["complete_comparison_count"] != 15 or len(new["families"]) != 6:
        raise AssertionError("Expected 15 complete comparisons in six families")
    if any(row["status"] != "complete" for row in new["families"].values()):
        raise AssertionError("At least one main comparison family is incomplete")
    assert_same(new["comparisons"], old["comparisons"], "main/comparisons")
    assert_same(new["families"], old["families"], "main/families")
    print("PASS: main statistics, 15/15 comparisons and 6/6 families")

    for name, original, recheck, count in (
        ("strong controls", "results_fx41_analysis/strong_controls.json", "results/recheck/fx41/strong_controls.json", 20),
        ("Mistral", "mistral_analysis/mistral_controls.json", "results/recheck/mistral/mistral_controls.json", 8),
    ):
        old, new = load(original), load(recheck)
        if len(new["comparisons"]) != count:
            raise AssertionError(f"Unexpected {name} comparison count")
        assert_same(new["comparisons"], old["comparisons"], name + "/comparisons")
        assert_same(new["behavior"], old["behavior"], name + "/behavior")
        print(f"PASS: {name} statistics match the archive")

    old = load("results/coordinate_summary.json")
    new = load("results/recheck/coordinate_summary.json")
    if len(new["rows"]) != 6:
        raise AssertionError("Expected six independent-coordinate rows")
    assert_same(new["rows"], old["rows"], "coordinates/rows")
    assert_same(new["benign_reference"], old["benign_reference"], "coordinates/benign")
    print("PASS: independent-coordinate statistics match the archive")

    old = load("results/main/derived_metrics.json")
    new = load("results/recheck/derived_metrics.json")
    for key in ("figure_data", "a0_alignment", "surgery", "surgery_canonical_detectors", "factorial_2x2x2"):
        assert_same(new[key], old[key], "derived/" + key)
    for stem in ("fig_coordinate_counterfactual", "fig_surgery_a0"):
        for folder in ("main", "fx41"):
            if folder == "main" and not (ROOT / "figures" / "recheck" / folder / f"{stem}.pdf").is_file():
                raise AssertionError(f"Missing regenerated figure: {stem}")
    for stem in ("fig_strong_coordinate_act", "fig_strong_coordinate_residual", "fig_surgery_generality_act", "fig_surgery_generality_residual"):
        if not (ROOT / "figures" / "recheck" / "fx41" / f"{stem}.pdf").is_file():
            raise AssertionError(f"Missing regenerated figure: {stem}")
    print("PASS: derived figure data and all six regenerated PDFs")


if __name__ == "__main__":
    main()
