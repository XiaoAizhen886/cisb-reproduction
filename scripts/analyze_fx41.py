#!/usr/bin/env python3
"""Hierarchical paired analysis for the FX41 strong-control results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "code"))
from analyze_results import selected_record  # noqa: E402

OLD = json.loads((PROJECT / "results" / "main" / "summary_complete.json").read_text())
NEW = PROJECT / "results_fx41"
OUT = PROJECT / "results" / "recheck" / "fx41"
SEEDS = (42, 7, 123)
METRICS = (("canonical", "act_l2"), ("canonical", "act_cos"),
           ("residual", "residual_l2"), ("residual", "residual_cos"))
FULL = {seed: f"cisb_v4b_aligned_undeniably_comma_posp_p15_s{seed}_r16_lam020_wcon" for seed in SEEDS}
STANDARD = {seed: f"cisb_v3_standard_undeniably_comma_posp_p15_s{seed}_r16_nocontrol" for seed in SEEDS}
RANDOM2024 = {seed: f"fx31_random_s{seed}" for seed in SEEDS}


def paired_auroc(trig, clean) -> float:
    trig = np.asarray(trig, dtype=np.float64).reshape(-1)
    clean = np.asarray(clean, dtype=np.float64).reshape(-1)
    if not len(trig) or len(trig) != len(clean) or not np.isfinite(trig).all() or not np.isfinite(clean).all():
        raise ValueError("Invalid paired AUROC inputs")
    allv = np.concatenate([trig, clean])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    sorted_values = allv[order]
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = len(trig)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(clean)))


def old_metric(adapter: str, kind: str, detector: str) -> dict:
    return selected_record(OLD["adapters"], adapter, kind)["metrics"][detector]["raw"]


def new_metric(adapter: str, kind: str, detector: str) -> dict:
    if kind == "canonical":
        path = NEW / "canonical" / f"{adapter}.json"
        return json.loads(path.read_text())["scores"][detector]
    path = NEW / "residual" / "sst2" / f"{adapter}_residual.json"
    return json.loads(path.read_text())["metrics"][detector]


def metric(adapter: str, kind: str, detector: str, new: bool) -> dict:
    return new_metric(adapter, kind, detector) if new else old_metric(adapter, kind, detector)


def bootstrap(rows: dict, comparison_id: str, n_boot: int = 2000) -> dict:
    digest = hashlib.sha256(("fx41\x1f" + comparison_id).encode()).hexdigest()
    rng = np.random.default_rng(int.from_bytes(bytes.fromhex(digest)[:4], "big"))
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(SEEDS, size=3, replace=True)
        values = []
        for seed in picked:
            row = rows[seed]
            n = row["n"]
            indices = rng.integers(0, n, size=n)
            left = paired_auroc(row["left"]["test_trig"][indices], row["left"]["test_clean"][indices])
            right = paired_auroc(row["right"]["test_trig"][indices], row["right"]["test_clean"][indices])
            values.append(left - right)
        deltas[b] = np.mean(values)
    kle = int(np.count_nonzero(deltas <= 0))
    kge = int(np.count_nonzero(deltas >= 0))
    p = max(2 * min(kle / n_boot, kge / n_boot), 1 / n_boot)
    return {"ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
            "p_floor": float(p), "tail_counts": {"kle": kle, "kge": kge, "B": n_boot}}


def bh(values: list[float]) -> list[float]:
    p = np.asarray(values)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.clip(adjusted, 0, 1)
    return result.tolist()


def compare(family: str, name: str, control: str, left: dict, right: dict, right_new: bool) -> dict:
    per_seed = {}
    rows = {}
    comparison_id = f"{family}:{control}:{name}"
    for seed in SEEDS:
        kind, detector = name.split(":", 1)
        lraw = metric(left[seed], kind, detector, False)
        rraw = metric(right[seed], kind, detector, right_new)
        n = min(len(lraw["test_trig"]), len(lraw["test_clean"]), len(rraw["test_trig"]), len(rraw["test_clean"]))
        l = {"test_trig": np.asarray(lraw["test_trig"][:n]), "test_clean": np.asarray(lraw["test_clean"][:n])}
        r = {"test_trig": np.asarray(rraw["test_trig"][:n]), "test_clean": np.asarray(rraw["test_clean"][:n])}
        lauroc = paired_auroc(l["test_trig"], l["test_clean"])
        rauroc = paired_auroc(r["test_trig"], r["test_clean"])
        per_seed[str(seed)] = {"left": lauroc, "right": rauroc, "delta": lauroc - rauroc, "n_pairs": n}
        rows[seed] = {"left": l, "right": r, "n": n}
    observed = float(np.mean([row["delta"] for row in per_seed.values()]))
    return {"family": family, "id": comparison_id, "control": control,
            "observed_delta": observed, "per_seed": per_seed,
            **bootstrap(rows, comparison_id)}


def behavior(adapter: str, new: bool) -> dict:
    if new:
        data = json.loads((NEW / "behavior" / f"{adapter}.json").read_text())
    else:
        data = selected_record(OLD["adapters"], adapter, "behavior")
    return {"cda": data["cda"], "asr_non_target": data["asr_non_target"], "n_samples": data["n_samples"]}


def main() -> None:
    manifest = json.loads((NEW / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError("FX41 run is incomplete")
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for kind, detector in METRICS:
        metric_name = f"{kind}:{detector}"
        comparisons.append(compare("strong_coordinate", metric_name, "global", FULL,
                                   {s: f"fx31_global_s{s}" for s in SEEDS}, True))
        for random_seed in (2025, 2026):
            comparisons.append(compare("strong_coordinate", metric_name, f"random{random_seed}", FULL,
                                       {s: f"fx31_random{random_seed}_s{s}" for s in SEEDS}, True))
        comparisons.append(compare("surgery_controls", metric_name, "standard", STANDARD,
                                   {s: f"fx31_surgery_standard_s{s}" for s in SEEDS}, True))
        comparisons.append(compare("surgery_controls", metric_name, "random2024", RANDOM2024,
                                   {s: f"fx31_surgery_random_s{s}" for s in SEEDS}, True))
    for family in ("strong_coordinate", "surgery_controls"):
        members = [row for row in comparisons if row["family"] == family]
        for row, q in zip(members, bh([row["p_floor"] for row in members])):
            row["q_bh_floor"] = q
    behavior_rows = {}
    for group, adapters, is_new in (
        ("global", {s: f"fx31_global_s{s}" for s in SEEDS}, True),
        ("random2025", {s: f"fx31_random2025_s{s}" for s in SEEDS}, True),
        ("random2026", {s: f"fx31_random2026_s{s}" for s in SEEDS}, True),
        ("surgery_standard", {s: f"fx31_surgery_standard_s{s}" for s in SEEDS}, True),
        ("surgery_random", {s: f"fx31_surgery_random_s{s}" for s in SEEDS}, True),
        ("full", FULL, False), ("standard", STANDARD, False), ("random2024", RANDOM2024, False),
    ):
        behavior_rows[group] = {str(seed): behavior(adapters[seed], is_new) for seed in SEEDS}
    result = {"schema_version": 1, "analysis": "fx41_strong_controls",
              "n_boot": 2000, "comparison_count": len(comparisons),
              "comparisons": comparisons, "behavior": behavior_rows}
    (OUT / "strong_controls.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'strong_controls.json'} with {len(comparisons)} comparisons")


if __name__ == "__main__":
    main()
