"""
analyze_results_v2_1.py
========================
GratiFlow v2.1 — Results Analysis and Figure Generation

Generates all figures and analysis outputs for v2.1 final experiment.
Must be run AFTER generate_and_run_v2_1.py completes successfully.

Outputs (saved to evaluation/ with v2_1 prefix):
  - evaluation/v2_1_srr_timeseries.png      : SRR time series (A vs B, per persona)
  - evaluation/v2_1_delta_srr_comparison.png : delta-SRR bar chart (main result)
  - evaluation/v2_1_latent_skill_growth.png  : latent_skill growth curves
  - evaluation/v2_1_scaffold_level_trace.png : scaffold level trace (adaptive-fading only)
  - evaluation/v2_1_judge_validation.png     : instrument validation confusion matrix
  - evaluation/v2_1_srr_reasoning_rules.png  : SRR reasoning rule citation frequencies

Figure requirement: "Synthetic Users (sequential, N=5)" must appear on all figures.

Author: team member (experiment lead, the research team)
Date: 2026-06-04
Protocol: evaluation_protocol_v2_1.md Section 3
"""

import json
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_V2_1 = BASE_DIR / "data" / "processed" / "experiments_v2_1"
EVAL_DIR = BASE_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style (colorblind-friendly palette) ──────────────────────────────────
# Using colorblind-safe colors: blue (adaptive-fading) vs orange (fixed-high)
COLOR_FADING = "#2171b5"    # blue
COLOR_FIXED = "#d94701"     # orange-red
COLOR_FADING_LIGHT = "#9ecae1"
COLOR_FIXED_LIGHT = "#fdae6b"

FIGURE_DPI = 300
FIGURE_FONT_SIZE = 11
SYNTHETIC_NOTE = "Synthetic Users (sequential, N=5)"

plt.rcParams.update({
    "font.size": FIGURE_FONT_SIZE,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
})


def load_session_data() -> dict:
    """
    Load all v2.1 session records.

    Returns:
        data: {"adaptive-fading": {pid: [sessions]}, "fixed-high": {pid: [sessions]}}
    """
    data = {}
    for condition in ["adaptive-fading", "fixed-high"]:
        data[condition] = {}
        cond_dir = DATA_V2_1 / "results" / f"condition_{condition}"
        if not cond_dir.exists():
            print(f"  [WARNING] Directory not found: {cond_dir}")
            continue
        for json_file in sorted(cond_dir.glob("P*_sessions.json")):
            pid = json_file.stem.replace("_sessions", "")
            with open(json_file) as f:
                sessions = json.load(f)
            data[condition][pid] = sessions
    return data


def load_summary() -> Optional[dict]:
    """Load v2.1 summary statistics."""
    path = DATA_V2_1 / "results" / "summary_statistics_v2_1.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_validation_results() -> Optional[dict]:
    """Load instrument validation results."""
    path = DATA_V2_1 / "instrument_validation" / "validation_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_srr_series(sessions: list) -> list:
    """
    Extract SRR per session. Returns None for sessions with undefined SRR.
    """
    srr = []
    for s in sorted(sessions, key=lambda x: x["session"]):
        val = s.get("spontaneous_rate")
        srr.append(val)  # None if SRR undefined (neg_count=0)
    return srr


def plot_srr_timeseries(data: dict, summary: dict) -> None:
    """
    Figure 1: SRR time series per persona (A vs B overlay).
    """
    personas = sorted(data.get("adaptive-fading", {}).keys())
    n_personas = len(personas)

    fig, axes = plt.subplots(1, n_personas, figsize=(3.5 * n_personas, 4), sharey=True)
    if n_personas == 1:
        axes = [axes]

    sessions_x = list(range(1, 11))

    for ax, pid in zip(axes, personas):
        sessions_a = data.get("adaptive-fading", {}).get(pid, [])
        sessions_b = data.get("fixed-high", {}).get(pid, [])

        srr_a = get_srr_series(sessions_a)
        srr_b = get_srr_series(sessions_b)

        # Plot with None handling (skip NaN points)
        x_a = [i + 1 for i, v in enumerate(srr_a) if v is not None]
        y_a = [v for v in srr_a if v is not None]
        x_b = [i + 1 for i, v in enumerate(srr_b) if v is not None]
        y_b = [v for v in srr_b if v is not None]

        if x_a:
            ax.plot(x_a, y_a, color=COLOR_FADING, linewidth=1.8,
                    marker="o", markersize=5, label="Adaptive-fading (A)")
        if x_b:
            ax.plot(x_b, y_b, color=COLOR_FIXED, linewidth=1.8,
                    marker="s", markersize=5, linestyle="--", label="Fixed-high (B)")

        # Shade early (S1-3) and late (S8-10) windows
        ax.axvspan(1, 3, alpha=0.08, color="gray", label="Early (S1-3)")
        ax.axvspan(8, 10, alpha=0.08, color="green", label="Late (S8-10)")

        # Annotate delta-SRR
        if summary:
            pdata = summary.get("personas", {}).get(pid, {})
            delta_a = pdata.get("condition_adaptive_fading", {}).get("delta_srr")
            delta_b = pdata.get("condition_fixed_high", {}).get("delta_srr")
            if delta_a is not None and delta_b is not None:
                ax.set_title(
                    f"{pid}\n"
                    f"$\\Delta$SRR A={delta_a:+.2f}, B={delta_b:+.2f}",
                    fontsize=10
                )
            else:
                ax.set_title(pid, fontsize=10)

        ax.set_xlabel("Session", fontsize=9)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([1, 3, 5, 7, 8, 10])
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel("SRR (reframe_count / neg_count)", fontsize=9)

    # Shared legend
    handles = [
        mpatches.Patch(color=COLOR_FADING, label="Adaptive-fading (A)"),
        mpatches.Patch(color=COLOR_FIXED, label="Fixed-high (B)"),
        mpatches.Patch(color="gray", alpha=0.3, label="Early window (S1-3)"),
        mpatches.Patch(color="green", alpha=0.3, label="Late window (S8-10)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=2)

    fig.suptitle(
        f"SRR Time Series: Adaptive-fading vs Fixed-high\n{SYNTHETIC_NOTE}",
        fontsize=12, y=1.02
    )

    out_path = EVAL_DIR / "v2_1_srr_timeseries.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_delta_srr_comparison(summary: dict) -> None:
    """
    Figure 2: delta-SRR per persona (bar chart) — main result figure.
    """
    personas = sorted(summary.get("personas", {}).keys())
    n = len(personas)

    delta_a = []
    delta_b = []
    labels = []

    for pid in personas:
        pdata = summary["personas"][pid]
        a = pdata["condition_adaptive_fading"]["delta_srr"]
        b = pdata["condition_fixed_high"]["delta_srr"]
        delta_a.append(a if a is not None else float("nan"))
        delta_b.append(b if b is not None else float("nan"))
        labels.append(pid)

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = ax.bar(x - width / 2, delta_a, width, color=COLOR_FADING, label="Adaptive-fading (A)", alpha=0.85)
    bars_b = ax.bar(x + width / 2, delta_b, width, color=COLOR_FIXED, label="Fixed-high (B)", alpha=0.85)

    # Add value labels on bars.
    #  - Positive (nonzero) labels: original position (h + 0.01, above bar).
    #  - Zero labels (+0.00): keep the raised position (h + 0.025).
    #  - Negative labels (-0.33): midway between original and lowered position.
    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if math.isnan(h):
                continue
            cx = bar.get_x() + bar.get_width() / 2
            if abs(h) < 1e-9:           # +0.00
                ax.text(cx, h + 0.025, f"{h:+.2f}", ha="center", va="bottom", fontsize=8)
            elif h > 0:                 # positive: original spot
                ax.text(cx, h + 0.01, f"{h:+.2f}", ha="center", va="bottom", fontsize=8)
            else:                       # negative (-0.33): between original and lowered
                ax.text(cx, h - 0.012, f"{h:+.2f}", ha="center", va="top", fontsize=8)

    _label_bars(bars_a)
    _label_bars(bars_b)

    # Aggregate mean lines
    agg = summary.get("aggregate", {})
    mean_a = agg.get("mean_delta_srr_adaptive_fading")
    mean_b = agg.get("mean_delta_srr_fixed_high")
    if mean_a is not None:
        ax.axhline(mean_a, color=COLOR_FADING, linestyle="--", linewidth=1.5,
                   label=f"Mean A = {mean_a:+.3f}")
    if mean_b is not None:
        ax.axhline(mean_b, color=COLOR_FIXED, linestyle="--", linewidth=1.5,
                   label=f"Mean B = {mean_b:+.3f}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Persona")
    ax.set_ylabel("delta-SRR (late − early)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Hypothesis result annotation
    hyp = agg.get("hypothesis_supported", False)
    n_higher = agg.get("n_personas_A_higher_delta_srr", 0)
    n_valid = agg.get("n_personas_valid_delta", n)
    direction = agg.get("direction_consistent_pre_committed", False)

    hyp_text = (
        f"H1: {'Supported' if hyp else 'Not Supported'}\n"
        f"Direction: A>B in {n_higher}/{n_valid} (threshold: 3/5)\n"
        f"Mean A: {mean_a:+.3f}, Mean B: {mean_b:+.3f}"
        if (mean_a is not None and mean_b is not None)
        else f"H1: {'Supported' if hyp else 'Not Supported'}\nDirection: A>B in {n_higher}/{n_valid}"
    )

    # Top-left H1 annotation, slightly enlarged per request.
    ax.text(0.02, 0.98, hyp_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # No title per request. Keep the synthetic-data disclosure as a small,
    # unobtrusive figure-level note (required honesty marker, not a title).
    fig.text(0.99, 0.005, SYNTHETIC_NOTE, ha="right", va="bottom",
             fontsize=8, color="gray")

    out_path = EVAL_DIR / "v2_1_delta_srr_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_latent_skill_growth(data: dict) -> None:
    """
    Figure 3: Latent skill growth curves per persona.
    """
    personas = sorted(data.get("adaptive-fading", {}).keys())
    n_personas = len(personas)

    fig, axes = plt.subplots(1, n_personas, figsize=(3.5 * n_personas, 4), sharey=True)
    if n_personas == 1:
        axes = [axes]

    for ax, pid in zip(axes, personas):
        sessions_a = sorted(data.get("adaptive-fading", {}).get(pid, []), key=lambda x: x["session"])
        sessions_b = sorted(data.get("fixed-high", {}).get(pid, []), key=lambda x: x["session"])

        latent_a = [s["latent_skill_after"] for s in sessions_a]
        latent_b = [s["latent_skill_after"] for s in sessions_b]
        x = list(range(1, len(latent_a) + 1))

        if latent_a:
            ax.plot(x, latent_a, color=COLOR_FADING, linewidth=2,
                    marker="o", markersize=4, label="Adaptive-fading (A)")
        if latent_b:
            ax.plot(x, latent_b, color=COLOR_FIXED, linewidth=2,
                    marker="s", markersize=4, linestyle="--", label="Fixed-high (B)")

        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("Session", fontsize=9)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([1, 5, 10])
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Latent Skill", fontsize=9)

    handles = [
        mpatches.Patch(color=COLOR_FADING, label="Adaptive-fading (A)"),
        mpatches.Patch(color=COLOR_FIXED, label="Fixed-high (B)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9)

    fig.suptitle(
        f"Latent Skill Growth Curves\n{SYNTHETIC_NOTE}",
        fontsize=12, y=1.02
    )

    out_path = EVAL_DIR / "v2_1_latent_skill_growth.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_scaffold_level_trace(data: dict) -> None:
    """
    Figure 4: Scaffold level trace for adaptive-fading condition only.
    """
    personas = sorted(data.get("adaptive-fading", {}).keys())
    n_personas = len(personas)

    scaffold_map = {"high": 0, "mid": 1, "low": 2}
    colors = ["#ef3b2c", "#fd8d3c", "#74c476"]  # high=red, mid=orange, low=green
    ytick_labels = ["high", "mid", "low"]

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, pid in enumerate(personas):
        sessions = sorted(data["adaptive-fading"].get(pid, []), key=lambda x: x["session"])
        x = [s["session"] for s in sessions]
        y = [scaffold_map.get(s.get("scaffold_level", "high"), 0) for s in sessions]
        ax.plot(x, y, marker="o", markersize=6, linewidth=1.5,
                label=pid, alpha=0.8)

    ax.set_xlabel("Session")
    ax.set_ylabel("Scaffold Level")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(ytick_labels)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(list(range(1, 11)))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title(
        f"Scaffold Level Trace — Adaptive-fading (A) only\n{SYNTHETIC_NOTE}",
        fontsize=11
    )

    out_path = EVAL_DIR / "v2_1_scaffold_level_trace.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_judge_validation(val_results: dict) -> None:
    """
    Figure 5: Instrument validation confusion matrix and metrics.
    """
    metrics = val_results.get("metrics", {})
    gate = val_results.get("gate_criteria", {})

    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    tn = metrics.get("tn", 0)
    fn = metrics.get("fn", 0)
    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1 = metrics.get("f1", 0)
    f3_correct = metrics.get("f3_detection_correct", 0)

    # Confusion matrix as 2x2 array
    cm = np.array([[tp, fn], [fp, tn]])

    fig, (ax_cm, ax_metrics) = plt.subplots(1, 2, figsize=(10, 4),
                                             gridspec_kw={"width_ratios": [1, 1.2]})

    # ── Left: Confusion matrix ─────────────────────────────────────────────
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted True", "Predicted False"])
    ax_cm.set_yticklabels(["Gold True", "Gold False"])
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("Gold label (human)")

    labels_cm = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax_cm.text(j, i, f"{labels_cm[i][j]}\n{val}",
                       ha="center", va="center", fontsize=14, fontweight="bold",
                       color="white" if cm[i, j] > (cm.max() * 0.6) else "black")

    ax_cm.set_title("Confusion Matrix\n(v2.1 Affect-Analysis Agent)", fontsize=11)

    # ── Right: Metrics ─────────────────────────────────────────────────────
    gate_passed = gate.get("gate_passed", False)

    metric_rows = [
        ("Accuracy",  accuracy,  0.80, gate.get("gate_accuracy", False)),
        ("Precision", precision, 0.75, gate.get("gate_precision", False)),
        ("Recall",    recall,    0.75, gate.get("gate_recall", False)),
        ("F1",        f1,        None, None),
        (f"F3 detect ({f3_correct}/2)", f3_correct / 2, 1.0, gate.get("gate_f3", False)),
    ]

    y_pos = list(range(len(metric_rows)))
    y_pos_r = list(reversed(y_pos))
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(-0.5, len(metric_rows) - 0.5)

    for i, (name, value, threshold, passed) in enumerate(metric_rows):
        y = y_pos_r[i]
        color = "#2ca02c" if passed else ("#d62728" if passed is False else "steelblue")
        ax_metrics.barh(y, value, color=color, alpha=0.7, height=0.6)
        if threshold is not None:
            ax_metrics.axvline(threshold, ymin=(y - 0.35) / len(metric_rows),
                               ymax=(y + 0.35 + 0.5) / len(metric_rows),
                               color="red", linestyle="--", linewidth=1.2, alpha=0.6)
        status = "PASS" if passed else ("FAIL" if passed is False else "")
        ax_metrics.text(
            min(value + 0.02, 0.95), y,
            f"{value:.3f} {status}",
            va="center", fontsize=10,
            color="#2ca02c" if passed else "#d62728" if passed is False else "black"
        )

    ax_metrics.set_yticks(y_pos)
    ax_metrics.set_yticklabels([r[0] for r in reversed(metric_rows)])
    ax_metrics.set_xlabel("Value")
    gate_text = "GATE: PASSED" if gate_passed else "GATE: FAILED"
    gate_color = "#2ca02c" if gate_passed else "#d62728"
    ax_metrics.set_title(
        f"Metrics & Gate Criteria\n{gate_text}",
        fontsize=11, color=gate_color
    )
    ax_metrics.axvline(1.0, color="gray", linewidth=0.8)
    ax_metrics.grid(True, axis="x", alpha=0.3, linestyle="--")

    fig.suptitle(
        f"Instrument Validation: Affect-Analysis Agent (v2.1)\n{SYNTHETIC_NOTE}",
        fontsize=11, y=1.02
    )

    out_path = EVAL_DIR / "v2_1_judge_validation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_srr_reasoning_rules(data: dict) -> None:
    """
    Figure 6: SRR reasoning rule citation frequencies across all sessions.
    Rule tags: R1, R2, R3, F1, F2, F3, F4, F5
    """
    all_reasonings = []
    for condition in data:
        for pid, sessions in data[condition].items():
            for s in sessions:
                r = s.get("srr_reasoning", "")
                if r:
                    all_reasonings.append(r)

    rules = ["R1", "R2", "R3", "F1", "F2", "F3", "F4", "F5"]
    counts = {r: 0 for r in rules}
    for reasoning in all_reasonings:
        for rule in rules:
            if re.search(rf"\b{rule}\b", reasoning):
                counts[rule] += 1

    total_sessions = len(all_reasonings)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_rules = [COLOR_FADING] * 3 + [COLOR_FIXED] * 5  # R-rules blue, F-rules orange
    bars = ax.bar(rules, [counts[r] for r in rules], color=colors_rules, alpha=0.8)

    for bar, rule in zip(bars, rules):
        h = bar.get_height()
        if h > 0:
            pct = h / total_sessions * 100
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{int(h)}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Rule Tag in srr_reasoning")
    ax.set_ylabel("Session Count")
    ax.set_title(
        f"SRR Reasoning Rule Citation Frequencies (n={total_sessions} sessions)\n"
        f"{SYNTHETIC_NOTE}",
        fontsize=11
    )

    handles = [
        mpatches.Patch(color=COLOR_FADING, alpha=0.8, label="Positive criteria (R1-R3)"),
        mpatches.Patch(color=COLOR_FIXED, alpha=0.8, label="Exclusion criteria (F1-F5)"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    out_path = EVAL_DIR / "v2_1_srr_reasoning_rules.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_final_summary(summary: dict) -> None:
    """Print the pre-committed analysis results to stdout."""
    agg = summary.get("aggregate", {})
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY (v2.1 — Rubric-based SRR, Final Experiment)")
    print(f"  {SYNTHETIC_NOTE}")
    print("=" * 70)

    mean_a = agg.get("mean_delta_srr_adaptive_fading")
    mean_b = agg.get("mean_delta_srr_fixed_high")
    adv = agg.get("mean_delta_srr_advantage_A_over_B")
    n_higher = agg.get("n_personas_A_higher_delta_srr", 0)
    n_valid = agg.get("n_personas_valid_delta", 0)
    direction = agg.get("direction_consistent_pre_committed", False)
    hyp = agg.get("hypothesis_supported", False)

    print(f"\n  Mean delta-SRR (Adaptive-fading A): {f'{mean_a:+.4f}' if mean_a is not None else 'null'}")
    print(f"  Mean delta-SRR (Fixed-high B):      {f'{mean_b:+.4f}' if mean_b is not None else 'null'}")
    print(f"  Advantage A over B:                 {f'{adv:+.4f}' if adv is not None else 'null'}")
    print(f"  Personas A > B:                     {n_higher}/{n_valid}")
    print(f"  Direction consistent (>=3/5):       {direction}")
    print(f"\n  PRE-COMMITTED HYPOTHESIS RESULT: {'SUPPORTED' if hyp else 'NOT SUPPORTED'}")
    print(f"  (H1 requires: direction_consistent AND mean_A > mean_B)")

    ceiling = agg.get("session1_ceiling_check", {})
    print(f"\n  Session 1 ceiling check (obs_s >= {ceiling.get('threshold', 0.35)}):")
    print(f"    adaptive-fading: {ceiling.get('n_personas_ceiling_adaptive_fading', '?')}/5 at ceiling")
    print(f"    fixed-high:      {ceiling.get('n_personas_ceiling_fixed_high', '?')}/5 at ceiling")

    print("\n  Per-persona delta-SRR:")
    for pid, pdata in summary.get("personas", {}).items():
        a = pdata["condition_adaptive_fading"]["delta_srr"]
        b = pdata["condition_fixed_high"]["delta_srr"]
        adv_p = pdata.get("delta_srr_advantage_A_over_B")
        label = pdata["label"]
        a_str = f"{a:+.4f}" if a is not None else "  null"
        b_str = f"{b:+.4f}" if b is not None else "  null"
        adv_str = f"{adv_p:+.4f}" if adv_p is not None else "  null"
        print(f"    {pid} ({label}): A={a_str}, B={b_str}, A-B={adv_str}")

    print(f"\n  {agg.get('interpretation', '')}")
    print("\n  NOTE: N=5 is underpowered for significance testing.")
    print("  This is the FINAL experiment. Results reported regardless of direction.")
    print("=" * 70)


def main() -> None:
    print("=" * 70)
    print("GratiFlow v2.1 — Results Analysis and Figure Generation")
    print(f"  {SYNTHETIC_NOTE}")
    print("=" * 70)

    # Load session data
    print("\nLoading session data...")
    data = load_session_data()
    total_sessions = sum(
        len(sessions)
        for cond in data.values()
        for sessions in cond.values()
    )
    print(f"  Loaded {total_sessions} sessions total.")

    if total_sessions == 0:
        print("  [ERROR] No session data found. Run generate_and_run_v2_1.py first.")
        return

    # Load summary
    summary = load_summary()
    if summary is None:
        print("  [WARNING] summary_statistics_v2_1.json not found. Some figures may be incomplete.")

    # Load validation results
    val_results = load_validation_results()

    # Generate figures
    print("\nGenerating figures...")

    print("  Figure 1: SRR time series...")
    plot_srr_timeseries(data, summary or {})

    print("  Figure 2: delta-SRR comparison...")
    if summary:
        plot_delta_srr_comparison(summary)
    else:
        print("    [SKIP] No summary data.")

    print("  Figure 3: Latent skill growth curves...")
    plot_latent_skill_growth(data)

    print("  Figure 4: Scaffold level trace...")
    plot_scaffold_level_trace(data)

    print("  Figure 5: Judge validation...")
    if val_results:
        plot_judge_validation(val_results)
    else:
        print("    [SKIP] No validation results found.")

    print("  Figure 6: SRR reasoning rule frequencies...")
    plot_srr_reasoning_rules(data)

    # Print final summary
    if summary:
        print_final_summary(summary)

    print(f"\nAll figures saved to: {EVAL_DIR}")
    print(f"  v2_1_srr_timeseries.png")
    print(f"  v2_1_delta_srr_comparison.png")
    print(f"  v2_1_latent_skill_growth.png")
    print(f"  v2_1_scaffold_level_trace.png")
    print(f"  v2_1_judge_validation.png")
    print(f"  v2_1_srr_reasoning_rules.png")
    print(f"\nAll figures include '{SYNTHETIC_NOTE}' in title as required.")


if __name__ == "__main__":
    main()
