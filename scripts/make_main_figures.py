from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from PIL import Image

SEEDS = (42, 7, 123)
SCRIPT_DIR = Path(__file__).resolve().parent
FIX_DIR = SCRIPT_DIR.parent
PROJECT_DIR = FIX_DIR
RESULTS_DIR = FIX_DIR / "results" / "main"
FIG_DIR = FIX_DIR / "figures" / "recheck" / "main"
SUMMARY_PATH = RESULTS_DIR / "summary_complete.json"
CONTRAST_PATH = RESULTS_DIR / "contrast_stats_complete.json"
SENSITIVITY_PATH = RESULTS_DIR / "stats_sensitivity.json"
DERIVED_PATH = FIX_DIR / "results" / "recheck" / "derived_metrics.json"
CANONICAL_PATHS = {
    seed: RESULTS_DIR / "canonical" / f"fx31_surgery_verified_s{seed}.json"
    for seed in SEEDS
}

STANDARD = {
    seed: f"cisb_v3_standard_undeniably_comma_posp_p15_s{seed}_r16_nocontrol"
    for seed in SEEDS
}
FULL = {
    seed: f"cisb_v4b_aligned_undeniably_comma_posp_p15_s{seed}_r16_lam020_wcon"
    for seed in SEEDS
}
RANDOM = {seed: f"fx31_random_s{seed}" for seed in SEEDS}
NON_TARGET = {seed: f"fx31_non_target_s{seed}" for seed in SEEDS}
SURGERY = {seed: f"fx31_surgery_verified_s{seed}" for seed in SEEDS}
A0 = {
    42: "kbsfix1_a0p1w1_undeniably_comma_posp_raw_s42",
    7: "fx31_a0_target_s7",
    123: "fx31_a0_target_s123",
}
FACTORIAL_ADAPTERS = {
    "a0p0w0": STANDARD[42],
    "a0p0w1": "kbsfix1_a0p0w1_undeniably_comma_posp_raw_s42",
    "a0p1w0": "kbsfix1_a0p1w0_undeniably_comma_posp_raw_s42",
    "a0p1w1": "kbsfix1_a0p1w1_undeniably_comma_posp_raw_s42",
    "a1p0w0": "kbsfix1_a1p0w0_undeniably_comma_posp_raw_s42",
    "a1p0w1": "kbsfix1_a1p0w1_undeniably_comma_posp_raw_s42",
    "a1p1w0": "kbsfix1_a1p1w0_undeniably_comma_posp_raw_s42",
    "a1p1w1": FULL[42],
}
GROUP_ADAPTERS = {
    "standard_ce": STANDARD,
    "cisb_full_target": FULL,
    "random_complement": RANDOM,
    "non_target_sensitive": NON_TARGET,
}
GROUP_LABELS = {
    "standard_ce": "Standard\nCE",
    "cisb_full_target": "CISB-full",
    "random_complement": "Random\ncomplement",
    "non_target_sensitive": "Non-target\nsensitive",
}
GROUP_STYLES = {
    "standard_ce": {"color": "#0072B2", "marker": "o"},
    "cisb_full_target": {"color": "#D55E00", "marker": "s"},
    "random_complement": {"color": "#009E73", "marker": "^"},
    "non_target_sensitive": {"color": "#CC79A7", "marker": "D"},
}
SEED_STYLES = {
    42: {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    7: {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    123: {"color": "#009E73", "marker": "^", "linestyle": ":"},
}
EXPECTED_FAMILIES = {
    "coordinate_act",
    "coordinate_residual",
    "main_residual",
    "a0_alignment",
    "surgery_act",
    "surgery_residual",
}


def read_json(path: Path) -> dict:
    resolved = path.resolve()
    if RESULTS_DIR.resolve() not in resolved.parents:
        raise ValueError(f"Refusing input outside fixed results directory: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric field: {context}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric field: {context}")
    return result


def paired_auroc(scores: dict, context: str) -> float:
    triggered = np.asarray(scores["test_trig"], dtype=float)
    clean = np.asarray(scores["test_clean"], dtype=float)
    if triggered.ndim != 1 or clean.ndim != 1 or not len(triggered) or not len(clean):
        raise ValueError(f"Invalid AUROC arrays: {context}")
    if not np.all(np.isfinite(triggered)) or not np.all(np.isfinite(clean)):
        raise ValueError(f"Non-finite AUROC scores: {context}")
    greater = (triggered[:, None] > clean[None, :]).sum()
    ties = (triggered[:, None] == clean[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(triggered) * len(clean)))


def assert_close(actual: float, expected: float, context: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise ValueError(f"{context}: computed {actual} != recorded {expected}")


def selected_record(summary: dict, adapter: str, kind: str) -> dict:
    block = summary["adapters"][adapter][kind]
    if block["status"] != "ok":
        raise ValueError(f"Selected {kind} record is not ok: {adapter}")
    index = block["selected_index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"Missing selected_index for {adapter}/{kind}")
    record = block["records"][index]
    if record["adapter"] != adapter:
        raise ValueError(f"Adapter mismatch in selected record: {adapter}/{kind}")
    return record


def selected_auroc(summary: dict, adapter: str, kind: str, detector: str) -> float:
    record = selected_record(summary, adapter, kind)
    metric = record["metrics"][detector]
    if metric["status"] != "ok":
        raise ValueError(f"Metric is not ok: {adapter}/{kind}/{detector}")
    computed = paired_auroc(metric["raw"], f"{adapter}/{kind}/{detector}")
    recorded = finite_number(metric["auroc"], f"{adapter}/{kind}/{detector}/auroc")
    assert_close(computed, recorded, f"{adapter}/{kind}/{detector}")
    return computed


def selected_behavior(summary: dict, adapter: str) -> dict:
    record = selected_record(summary, adapter, "behavior")
    n_samples = finite_number(record["n_samples"], f"{adapter}/behavior/n_samples")
    if n_samples != 872 or not record["protocol_fully_verified"]:
        raise ValueError(f"Behavior protocol is not verified n=872: {adapter}")
    return {
        "n_samples": int(n_samples),
        "cda": finite_number(record["cda"], f"{adapter}/behavior/cda"),
        "asr_non_target": finite_number(
            record["asr_non_target"], f"{adapter}/behavior/asr_non_target"
        ),
        "protocol_fully_verified": True,
    }


def validate_contrast(summary: dict, contrast: dict) -> None:
    if contrast["comparison_count"] != 15 or contrast["complete_comparison_count"] != 15:
        raise ValueError("Expected 15/15 complete comparisons")
    comparisons = contrast["comparisons"]
    families = contrast["families"]
    if len(comparisons) != 15 or set(families) != EXPECTED_FAMILIES or len(families) != 6:
        raise ValueError("Expected 15 comparisons in six prespecified families")
    member_total = 0
    for family_name, family in families.items():
        if family["status"] != "complete":
            raise ValueError(f"Incomplete family: {family_name}")
        if family["missing_members"] or family["missing_seeds_by_member"]:
            raise ValueError(f"Missing family members or seeds: {family_name}")
        members = family["members"]
        if family["member_count"] != len(members) or family["complete_member_count"] != len(members):
            raise ValueError(f"Family member count mismatch: {family_name}")
        if any(member not in comparisons for member in members):
            raise ValueError(f"Unknown family member: {family_name}")
        member_total += len(members)
    if member_total != 15:
        raise ValueError("Family membership does not account for all 15 comparisons")

    for comparison_id, comparison in comparisons.items():
        if comparison["status"] != "ok" or comparison["n_seeds_used"] != 3:
            raise ValueError(f"Incomplete comparison: {comparison_id}")
        if comparison["missing_seeds"] or comparison["required_seeds"] != list(SEEDS):
            raise ValueError(f"Required seeds missing: {comparison_id}")
        kind = comparison["kind"]
        detector = comparison["detector"]
        for seed in SEEDS:
            per_seed = comparison["per_seed"][str(seed)]
            if per_seed["status"] != "ok" or per_seed["missing_sides"]:
                raise ValueError(f"Incomplete seed in {comparison_id}: {seed}")
            values = {}
            for side in ("left", "right"):
                adapter = per_seed[side]["adapter"]
                value = selected_auroc(summary, adapter, kind, detector)
                recorded = finite_number(
                    per_seed[side]["auroc"], f"{comparison_id}/{seed}/{side}/auroc"
                )
                assert_close(value, recorded, f"{comparison_id}/{seed}/{side}")
                values[side] = value
            assert_close(
                values["left"] - values["right"],
                finite_number(per_seed["delta"], f"{comparison_id}/{seed}/delta"),
                f"{comparison_id}/{seed}/delta",
            )

    a0_comparison = comparisons["a0_alignment:key_a0_p1w1_vs_full:canonical:act_l2"]
    for seed in SEEDS:
        if a0_comparison["per_seed"][str(seed)]["left"]["adapter"] != A0[seed]:
            raise ValueError(f"Unexpected A0 adapter for seed {seed}")


def validate_sensitivity(summary: dict, sensitivity: dict) -> None:
    if sensitivity["missing"]:
        raise ValueError("stats_sensitivity.json reports missing adapters")
    for adapters in (STANDARD, FULL):
        for seed, adapter in adapters.items():
            per_adapter = sensitivity["per_adapter"][adapter]
            if per_adapter["seed"] != seed:
                raise ValueError(f"Sensitivity seed mismatch: {adapter}")
            for detector in ("act_l2", "act_cos", "strip_maxprob"):
                value = selected_auroc(summary, adapter, "canonical", detector)
                recorded = finite_number(
                    per_adapter["metrics"][detector]["auroc"],
                    f"stats_sensitivity/{adapter}/{detector}/auroc",
                )
                assert_close(value, recorded, f"stats_sensitivity/{adapter}/{detector}")


def validate_canonical_raw(summary: dict, raw_by_seed: dict) -> dict:
    computed = {}
    signatures = []
    for seed in SEEDS:
        raw = raw_by_seed[seed]
        if raw["adapter"] != SURGERY[seed]:
            raise ValueError(f"Unexpected canonical adapter for seed {seed}")
        signatures.append(
            (
                raw["dataset"],
                raw["trigger"],
                raw["position"],
                raw["n_cal"],
                raw["n_test"],
                raw["n_onion"],
                raw["n_strip"],
                tuple(raw["detectors"]),
            )
        )
        selected_hash = selected_record(summary, SURGERY[seed], "canonical")["source_sha256"]
        if sha256(CANONICAL_PATHS[seed]) != selected_hash:
            raise ValueError(f"Canonical file hash mismatch for seed {seed}")
        computed[seed] = {}
        for detector in ("act_l2", "act_cos", "strip_maxprob", "ppl_ratio", "onion_maxdrop"):
            value = paired_auroc(raw["scores"][detector], f"{SURGERY[seed]}/{detector}")
            computed[seed][detector] = value
            if detector in ("act_l2", "act_cos", "strip_maxprob"):
                selected_value = selected_auroc(summary, SURGERY[seed], "canonical", detector)
                assert_close(value, selected_value, f"canonical raw/{SURGERY[seed]}/{detector}")
    if len(set(signatures)) != 1:
        raise ValueError("Canonical surgery files do not share one protocol")
    return computed


def summarize(values: dict) -> dict:
    ordered = [finite_number(values[str(seed)], f"per_seed/{seed}") for seed in SEEDS]
    return {
        "mean": float(np.mean(ordered)),
        "min": float(np.min(ordered)),
        "max": float(np.max(ordered)),
        "n": len(ordered),
    }


def input_provenance(paths: list[Path]) -> dict:
    return {
        path.relative_to(PROJECT_DIR).as_posix(): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    }


def build_derived() -> dict:
    summary = read_json(SUMMARY_PATH)
    contrast = read_json(CONTRAST_PATH)
    sensitivity = read_json(SENSITIVITY_PATH)
    if summary["input_issues"] != []:
        raise ValueError(f"summary_complete.json input_issues must be empty: {summary['input_issues']}")
    actual_summary_hash = sha256(SUMMARY_PATH)
    if contrast["summary_input"]["sha256"] != actual_summary_hash:
        raise ValueError("contrast.summary_input.sha256 does not match summary_complete.json")
    if contrast["summary_input"]["size_bytes"] != SUMMARY_PATH.stat().st_size:
        raise ValueError("contrast summary_input size does not match summary_complete.json")
    validate_contrast(summary, contrast)
    validate_sensitivity(summary, sensitivity)
    raw_by_seed = {seed: read_json(CANONICAL_PATHS[seed]) for seed in SEEDS}
    surgery_raw_metrics = validate_canonical_raw(summary, raw_by_seed)

    figure1_panels = {}
    for panel_key, kind, detector in (
        ("canonical_act_l2", "canonical", "act_l2"),
        ("residual_l2", "residual", "residual_l2"),
    ):
        groups = {}
        for group, adapters in GROUP_ADAPTERS.items():
            per_seed = {
                str(seed): selected_auroc(summary, adapters[seed], kind, detector)
                for seed in SEEDS
            }
            groups[group] = {
                "label": GROUP_LABELS[group].replace("\n", " "),
                "per_seed": per_seed,
                **summarize(per_seed),
            }
        figure1_panels[panel_key] = {"groups": groups}

    a0_per_seed = {}
    for seed in SEEDS:
        a0_value = selected_auroc(summary, A0[seed], "canonical", "act_l2")
        full_value = selected_auroc(summary, FULL[seed], "canonical", "act_l2")
        a0_per_seed[str(seed)] = {
            "a0_adapter": A0[seed],
            "full_adapter": FULL[seed],
            "a0_act_l2": a0_value,
            "full_act_l2": full_value,
            "full_minus_a0": full_value - a0_value,
        }

    surgery_per_seed = {}
    for seed in SEEDS:
        raw_behavior = selected_behavior(summary, FULL[seed])
        after_behavior = selected_behavior(summary, SURGERY[seed])
        raw_values = {
            "adapter": FULL[seed],
            "act_l2": selected_auroc(summary, FULL[seed], "canonical", "act_l2"),
            "act_cos": selected_auroc(summary, FULL[seed], "canonical", "act_cos"),
            "residual_l2": selected_auroc(summary, FULL[seed], "residual", "residual_l2"),
            "residual_cos": selected_auroc(summary, FULL[seed], "residual", "residual_cos"),
            "cda": raw_behavior["cda"],
            "asr_non_target": raw_behavior["asr_non_target"],
            "behavior_n_samples": raw_behavior["n_samples"],
        }
        after_values = {
            "adapter": SURGERY[seed],
            "act_l2": surgery_raw_metrics[seed]["act_l2"],
            "act_cos": surgery_raw_metrics[seed]["act_cos"],
            "residual_l2": selected_auroc(summary, SURGERY[seed], "residual", "residual_l2"),
            "residual_cos": selected_auroc(summary, SURGERY[seed], "residual", "residual_cos"),
            "cda": after_behavior["cda"],
            "asr_non_target": after_behavior["asr_non_target"],
            "behavior_n_samples": after_behavior["n_samples"],
        }
        surgery_per_seed[str(seed)] = {
            "raw": raw_values,
            "after": after_values,
            "after_minus_raw": {
                key: after_values[key] - raw_values[key]
                for key in ("act_l2", "act_cos", "residual_l2", "residual_cos", "cda", "asr_non_target")
            },
        }

    surgery_summaries = {"raw": {}, "after": {}, "after_minus_raw": {}}
    for metric in ("act_l2", "act_cos", "residual_l2", "residual_cos", "cda", "asr_non_target"):
        for branch in surgery_summaries:
            values = {
                seed: surgery_per_seed[seed][branch][metric]
                for seed in surgery_per_seed
            }
            surgery_summaries[branch][metric] = summarize(values)

    canonical_detector_per_seed = {
        str(seed): {
            detector: surgery_raw_metrics[seed][detector]
            for detector in ("strip_maxprob", "ppl_ratio", "onion_maxdrop")
        }
        for seed in SEEDS
    }
    canonical_detector_summaries = {}
    for detector in ("strip_maxprob", "ppl_ratio", "onion_maxdrop"):
        values = {
            str(seed): surgery_raw_metrics[seed][detector]
            for seed in SEEDS
        }
        canonical_detector_summaries[detector] = summarize(values)

    factorial_cells = {}
    for cell, adapter in FACTORIAL_ADAPTERS.items():
        if cell not in ("a0p0w0", "a1p1w1"):
            if summary["adapters"][adapter]["factorial_cell"] != cell:
                raise ValueError(f"Factorial cell metadata mismatch: {adapter}")
        behavior = selected_behavior(summary, adapter)
        factorial_cells[cell] = {
            "adapter": adapter,
            "a": int(cell[1]),
            "p": int(cell[3]),
            "w": int(cell[5]),
            "act_l2": selected_auroc(summary, adapter, "canonical", "act_l2"),
            "act_cos": selected_auroc(summary, adapter, "canonical", "act_cos"),
            "strip_maxprob": selected_auroc(summary, adapter, "canonical", "strip_maxprob"),
            **behavior,
        }

    figure2_surgery = {
        seed: {
            "raw_cisb_full": surgery_per_seed[seed]["raw"]["act_l2"],
            "fx31_surgery_verified": surgery_per_seed[seed]["after"]["act_l2"],
        }
        for seed in surgery_per_seed
    }
    figure2_a0 = {
        seed: {
            "a0": a0_per_seed[seed]["a0_act_l2"],
            "full": a0_per_seed[seed]["full_act_l2"],
            "full_minus_a0": a0_per_seed[seed]["full_minus_a0"],
        }
        for seed in a0_per_seed
    }

    derived = {
        "schema_version": 1,
        "analysis": "offline_metrics_from_fixed_local_json_only",
        "seeds": list(SEEDS),
        "inputs": input_provenance(
            [SUMMARY_PATH, CONTRAST_PATH, SENSITIVITY_PATH]
            + [CANONICAL_PATHS[seed] for seed in SEEDS]
        ),
        "validation": {
            "input_issues": summary["input_issues"],
            "summary_sha256": actual_summary_hash,
            "summary_sha256_matches_contrast": True,
            "comparison_count": contrast["comparison_count"],
            "complete_comparison_count": contrast["complete_comparison_count"],
            "complete_family_count": len(contrast["families"]),
            "families": {name: block["status"] for name, block in contrast["families"].items()},
            "all_comparison_side_aurocs_recomputed": True,
            "sensitivity_standard_and_full_crosschecked": True,
            "canonical_raw_files_hash_and_protocol_checked": True,
            "factorial_cell_count": len(factorial_cells),
            "factorial_behavior_n_samples": 872,
        },
        "definitions": {
            "auroc": "Average-rank Mann-Whitney AUROC with triggered scores as positive and half credit for ties.",
            "range": "Minimum and maximum across seeds 42, 7, and 123; never a confidence interval.",
            "a0_delta": "full - A0 for Act-L2 AUROC.",
            "surgery_delta": "after - raw for each surgery metric.",
            "metric_labels": {
                "act_l2": "Act-L2",
                "act_cos": "Act-cos",
                "residual_l2": "residual-L2",
                "residual_cos": "residual-cos",
                "strip_maxprob": "STRIP-p",
                "ppl_ratio": "PPL-ratio",
                "onion_maxdrop": "ONION",
                "asr_non_target": "ASR-NT",
            },
        },
        "figure_data": {
            "fig_coordinate_counterfactual": {
                "panels": figure1_panels,
                "seed_order_within_group": list(SEEDS),
                "summary_display": "Mean plus min-max range across seeds; n=3; not a confidence interval.",
            },
            "fig_surgery_a0": {
                "surgery_act_l2": figure2_surgery,
                "a0_full_act_l2": figure2_a0,
                "pairing": "One line per seed connects two conditions; lines are not class trends.",
            },
        },
        "a0_alignment": {
            "delta_convention": "full - A0",
            "per_seed": a0_per_seed,
            "summary": {
                "a0_act_l2": summarize({seed: row["a0_act_l2"] for seed, row in a0_per_seed.items()}),
                "full_act_l2": summarize({seed: row["full_act_l2"] for seed, row in a0_per_seed.items()}),
                "full_minus_a0": summarize({seed: row["full_minus_a0"] for seed, row in a0_per_seed.items()}),
            },
        },
        "surgery": {
            "raw_condition": "CISB-full",
            "after_condition": "fx31_surgery_verified",
            "per_seed": surgery_per_seed,
            "summary": surgery_summaries,
        },
        "surgery_canonical_detectors": {
            "source": "AUROC recomputed from test_trig/test_clean scores in the three fixed canonical raw JSON files.",
            "per_seed": canonical_detector_per_seed,
            "summary": canonical_detector_summaries,
        },
        "factorial_2x2x2": {
            "source": "Selected canonical and behavior records in summary_complete.json adapters; no factorial_summary input.",
            "behavior_protocol": "Unified verified n=872 for CDA and ASR-NT in every cell.",
            "cells": factorial_cells,
        },
    }
    return derived


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, linestyle=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.axhline(0.5, color="#666666", linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))


def shared_ylim(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    padding = max(0.025, 0.06 * (high - low))
    return low - padding, high + padding


def make_coordinate_figure(derived: dict):
    panel_data = derived["figure_data"]["fig_coordinate_counterfactual"]["panels"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.15, 1.0, 0.83))
    all_values = [
        value
        for panel in panel_data.values()
        for group in panel["groups"].values()
        for value in group["per_seed"].values()
    ]
    ylim = shared_ylim(all_values)
    group_keys = list(GROUP_ADAPTERS)
    x_offsets = {42: -0.15, 7: 0.0, 123: 0.15}
    for ax, panel_key, title in zip(
        axes,
        ("canonical_act_l2", "residual_l2"),
        ("(A) Target-coordinate Act-L2", "(B) Residual-L2"),
    ):
        groups = panel_data[panel_key]["groups"]
        for x, group_key in enumerate(group_keys):
            group = groups[group_key]
            style = GROUP_STYLES[group_key]
            for seed in SEEDS:
                ax.scatter(
                    x + x_offsets[seed],
                    group["per_seed"][str(seed)],
                    s=34,
                    marker=style["marker"],
                    facecolor=style["color"],
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.92,
                    zorder=3,
                )
            ax.errorbar(
                x,
                group["mean"],
                yerr=[[group["mean"] - group["min"]], [group["max"] - group["mean"]]],
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.15,
                capsize=3.5,
                capthick=1.15,
                alpha=0.85,
                zorder=2,
            )
            ax.plot(
                [x - 0.18, x + 0.18],
                [group["mean"], group["mean"]],
                color="black",
                linewidth=1.6,
                solid_capstyle="butt",
                zorder=4,
            )
        clean_axis(ax)
        ax.set_xlim(-0.55, len(group_keys) - 0.45)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(group_keys)), [GROUP_LABELS[key] for key in group_keys])
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.tick_params(axis="x", length=0, pad=5)
    axes[0].set_ylabel("AUROC")
    fig.text(
        0.5,
        0.045,
        "Dots: seeds 42, 7, 123 (left to right within each group).\n"
        "Black mark: mean; vertical whiskers: min-max range (n=3 per group; not a CI).",
        ha="center",
        va="center",
        fontsize=7,
        linespacing=1.35,
    )
    return fig


def make_surgery_figure(derived: dict):
    data = derived["figure_data"]["fig_surgery_a0"]
    surgery = data["surgery_act_l2"]
    a0 = data["a0_full_act_l2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.16, 1.0, 0.73))
    all_values = [
        value
        for rows in (surgery, a0)
        for row in rows.values()
        for key, value in row.items()
        if key != "full_minus_a0"
    ]
    ylim = shared_ylim(all_values)
    panels = (
        (
            "(A) Raw CISB-full vs. CISB-surgery",
            surgery,
            ("raw_cisb_full", "fx31_surgery_verified"),
            ("Raw CISB-full", "CISB-surgery"),
        ),
        (
            "(B) A0 vs. CISB-full",
            a0,
            ("a0", "full"),
            ("A0", "CISB-full"),
        ),
    )
    for ax, (title, rows, keys, labels) in zip(axes, panels):
        for seed in SEEDS:
            style = SEED_STYLES[seed]
            row = rows[str(seed)]
            ax.plot(
                [0, 1],
                [row[keys[0]], row[keys[1]]],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.35,
                marker=style["marker"],
                markersize=5.5,
                markeredgecolor="black",
                markeredgewidth=0.45,
                alpha=0.9,
                zorder=3,
            )
        clean_axis(ax)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 1], labels)
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.tick_params(axis="x", length=0, pad=5)
    axes[0].set_ylabel("Act-L2 AUROC")
    handles = [
        Line2D(
            [0],
            [0],
            color=SEED_STYLES[seed]["color"],
            linestyle=SEED_STYLES[seed]["linestyle"],
            marker=SEED_STYLES[seed]["marker"],
            markersize=5.5,
            markeredgecolor="black",
            markeredgewidth=0.4,
            linewidth=1.35,
            label=f"Seed {seed}",
        )
        for seed in SEEDS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.045,
        "One line per seed joins paired conditions; lines are not class trends.",
        ha="center",
        va="center",
        fontsize=7,
        linespacing=1.35,
    )
    return fig


def run_visual_audit(figures: dict) -> None:
    module_path = (
        Path.home()
        / ".agents"
        / "skills"
        / "scipilot-figure-skill"
        / "scripts"
        / "visual_qa.py"
    )
    if not module_path.is_file():
        print("Optional visual_qa is unavailable; skipping this external layout audit.")
        return
    spec = importlib.util.spec_from_file_location("scipilot_visual_qa", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, fig in figures.items():
        issues = module.audit_layout(fig)
        print(f"visual_qa {name}:")
        verdict = module.print_report(issues)
        if verdict == "FAIL":
            raise RuntimeError(f"visual_qa failed for {name}")


def save_previews(name: str, fig) -> None:
    preview_path = FIG_DIR / f"{name}_preview.png"
    grayscale_path = FIG_DIR / f"{name}_grayscale.png"
    fig.savefig(preview_path, dpi=180, bbox_inches="tight", pad_inches=0.025)
    with Image.open(preview_path) as image:
        image.convert("L").save(grayscale_path)


def save_pdf(name: str, fig) -> None:
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Creator": "make_figure.py", "Title": name},
    )
    if path.stat().st_size == 0 or path.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError(f"Invalid PDF output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive fixed JSON metrics and render paper figures.")
    parser.add_argument("--preview-only", action="store_true", help="Do not export final PDFs.")
    args = parser.parse_args()
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    derived = build_derived()
    with DERIVED_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(derived, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    figures = {
        "fig_coordinate_counterfactual": make_coordinate_figure(derived),
        "fig_surgery_a0": make_surgery_figure(derived),
    }
    run_visual_audit(figures)
    for name, fig in figures.items():
        save_previews(name, fig)
        if not args.preview_only:
            save_pdf(name, fig)
    for fig in figures.values():
        plt.close(fig)
    print(f"Wrote {DERIVED_PATH}")
    for name in figures:
        print(f"Wrote {FIG_DIR / (name + '_preview.png')}")
        print(f"Wrote {FIG_DIR / (name + '_grayscale.png')}")
        if not args.preview_only:
            print(f"Wrote {FIG_DIR / (name + '.pdf')}")


if __name__ == "__main__":
    main()
