#!/usr/bin/env python3
"""Render strong-coordinate and spectrum-generality figures from verified statistics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIX = ROOT.parent
DATA = json.loads((FIX / "results" / "recheck" / "fx41" / "strong_controls.json").read_text())
OLD = json.loads((FIX / "results" / "main" / "contrast_stats_complete.json").read_text())
FIG = FIX / "figures" / "recheck" / "fx41"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.unicode_minus": False, "pdf.fonttype": 42})
COLORS = {"global": "#0072B2", "random2024": "#009E73", "random2025": "#E69F00",
          "random2026": "#CC79A7", "non_target": "#56B4E9", "standard": "#0072B2",
          "full": "#D55E00"}
LABELS = {"global": "Final-residual", "random2024": "Random-A", "random2025": "Random-B",
          "random2026": "Random-C", "non_target": "Non-target", "standard": "Standard CE",
          "full": "CISB-full"}
MARKERS = {"global": "o", "random2024": "^", "random2025": "s", "random2026": "D",
           "non_target": "v", "standard": "o", "full": "s"}


def new_row(control, kind, detector):
    suffix = f"{kind}:{detector}"
    matches = [row for row in DATA["comparisons"]
               if row["control"] == control and row["id"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError((control, suffix, len(matches)))
    row = matches[0]
    return {"delta": row["observed_delta"], "ci": row["ci95"], "q": row["q_bh_floor"],
            "seeds": {s: values["delta"] for s, values in row["per_seed"].items()}}


def old_row(comparison_id):
    row = OLD["comparisons"][comparison_id]
    control = "random2024" if "random" in comparison_id else "non_target"
    return {"delta": row["observed"]["delta"], "ci": row["ci95_percentile"],
            "q": row["q_bh_floor"],
            "seeds": {s: values["delta"] for s, values in row["per_seed"].items()}}


def coordinate_row(control, kind, detector):
    if control in ("random2024", "non_target"):
        family = "coordinate_act" if kind == "canonical" else "coordinate_residual"
        return old_row(f"{family}:target_vs_{'random' if control == 'random2024' else 'non_target'}:{kind}:{detector}")
    return new_row(control, kind, detector)


def surgery_row(control, kind, detector):
    if control == "full":
        family = "surgery_act" if kind == "canonical" else "surgery_residual"
        return old_row(f"{family}:raw_vs_surgery:{kind}:{detector}")
    return new_row(control, kind, detector)


def draw(rows, path, xlabel, title):
    fig, ax = plt.subplots(figsize=(7.16, 2.7), layout="constrained")
    y = np.arange(len(rows))[::-1]
    for yi, (control, row) in zip(y, rows):
        color = COLORS[control]
        lo, hi = row["ci"]
        ax.errorbar(row["delta"], yi, xerr=[[row["delta"]-lo], [hi-row["delta"]]],
                    fmt=MARKERS[control], color=color, ecolor=color, elinewidth=1.3,
                    capsize=3.5, markersize=6, markeredgecolor="black", markeredgewidth=.4, zorder=3)
        for j, value in enumerate(row["seeds"].values()):
            ax.scatter(value, yi + (j-1)*.10, s=16, color=color, alpha=.48,
                       marker=MARKERS[control], edgecolor="none", zorder=2)
    ax.axvline(0, color="#555555", linestyle=(0, (4, 3)), linewidth=.9)
    ax.set_yticks(y, [LABELS[c] for c, _ in rows])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", linewidth=.55, linestyle=(0, (1, 2)))
    ax.set_axisbelow(True)
    ax.text(0, -0.22, "Points: per-seed differences; error bars: hierarchical paired-bootstrap 95% CI (n=3).",
            transform=ax.transAxes, fontsize=7, ha="left", va="top")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=.04)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04,
                metadata={"Creator": "make_fx41_figures.py", "Title": path.stem})
    plt.close(fig)


def main():
    for kind, detector, stem, panel in (("canonical", "act_l2", "act", "Act-L2"),
                                       ("residual", "residual_l2", "residual", "Residual-L2")):
        controls = ["global", "random2024", "random2025", "random2026", "non_target"]
        rows = [(c, coordinate_row(c, kind, detector)) for c in controls]
        draw(rows, FIG / f"fig_strong_coordinate_{stem}",
             "Target minus control (AUROC); negative favors target",
             f"Target recipe versus controls: {panel}")
    for kind, detector, stem, panel in (("canonical", "act_l2", "act", "Act-L2"),
                                       ("residual", "residual_l2", "residual", "Residual-L2")):
        controls = ["standard", "random2024", "full"]
        rows = []
        for control in controls:
            row = surgery_row(control, kind, detector)
            if control == "random2024":
                row = new_row("random2024", kind, detector) if False else row
            rows.append((control, row))
        draw(rows, FIG / f"fig_surgery_generality_{stem}",
             "Raw minus surgery (AUROC); negative = higher after surgery",
             f"Matched-spectrum replacement across configurations: {panel}")
    print("Wrote FX41 figures")


if __name__ == "__main__":
    main()
