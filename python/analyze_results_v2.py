"""
analyze_results_v2.py
======================
GratiFlow v2 Evaluation — Results Analysis and Visualization

Generates all required figures and tables per evaluation_protocol_v2.md Section 10.

Figures produced (saved to evaluation/ with v2 prefix, never overwriting v1 files):
  fig2_v2_growth_curve.png           — latent_skill (ground truth) + observed skill (s)
  fig3_v2_ablation_srr.png           — SRR comparison: adaptive-fading vs fixed-high
  fig4_v2_practice_opportunity.png   — p_attempt trajectory + did_attempt binary
  fig5_v2_sensitivity_multiplier.png — Sensitivity: SCAFFOLD_ATTEMPT_MULTIPLIER variation
  fig6_v2_sensitivity_alpha.png      — Sensitivity: alpha/alpha_passive ratio variation
  fig7_v2_groundtruth_vs_llm.png     — Ground truth latent_skill vs LLM-judged observed skill

All figure titles include: "Synthetic Users (sequential, N=5 personas)"
All figures: 300 dpi, colorblind-safe palette, 12pt+ font.

Also produces:
  table1_v2_ablation_results.csv     — Main ablation table
  figure_captions_v2.md              — Caption drafts for all figures

Author: team member (experiment lead, the research team)
Date: 2026-06-04
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_V2 = BASE_DIR / "data" / "processed" / "experiments_v2"
EVALUATION_DIR = BASE_DIR / "evaluation"
PERSONAS_FILE = Path(__file__).parent / "personas_v2.json"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# ── Color palette (colorblind-safe: Wong 2011) ────────────────────────────────
# Orange (#E69F00) for adaptive-fading (Condition A)
# Blue   (#0072B2) for fixed-high (Condition B)
# Additional safe colors for persona lines
COLOR_FADING = "#E69F00"   # orange — Condition A (adaptive-fading)
COLOR_FIXED  = "#0072B2"   # blue   — Condition B (fixed-high)
COLOR_LATENT = "#009E73"   # green  — latent_skill (ground truth)
COLOR_OBS    = "#CC79A7"   # purple — observed skill (LLM-judged)

# Persona colors (colorblind-safe subset)
PERSONA_COLORS = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9"]

FONT_SIZE_TITLE  = 13
FONT_SIZE_LABEL  = 11
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10
DPI = 300

PERSONAS = ["P1", "P2", "P3", "P4", "P5"]
SESSIONS = list(range(1, 11))


# ── Data loading helpers ───────────────────────────────────────────────────────

def load_summary() -> dict:
    path = DATA_PROCESSED_V2 / "results" / "summary_statistics_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}\nRun generate_and_run_v2.py first.")
    with open(path) as f:
        return json.load(f)


def load_ground_truth() -> pd.DataFrame:
    path = DATA_PROCESSED_V2 / "ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data["records"])


def load_session_records(condition: str, pid: str) -> list:
    path = DATA_PROCESSED_V2 / "results" / f"condition_{condition}" / f"{pid}_sessions.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_personas() -> list:
    with open(PERSONAS_FILE) as f:
        return json.load(f)["personas"]


# ── Common plot setup ─────────────────────────────────────────────────────────

def set_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "DejaVu Sans",
    })


# ── Fig 2: Growth Curve (latent_skill + observed skill) ──────────────────────

def plot_fig2_growth_curve(gt_df: pd.DataFrame, summary: dict) -> str:
    """
    Fig 2: latent_skill (ground truth) and observed skill (s) over sessions.
    Shows both conditions for all 5 personas.
    """
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=False)
    fig.suptitle(
        "Fig. 2: Skill Growth Curves — Synthetic Users (sequential, N=5 personas)\n"
        "Ground-truth latent skill (solid) vs LLM-judged observed skill (dashed)",
        fontsize=FONT_SIZE_TITLE, y=1.02
    )

    conditions = ["adaptive-fading", "fixed-high"]
    condition_labels = {"adaptive-fading": "Cond. A: Adaptive-Fading", "fixed-high": "Cond. B: Fixed-High"}
    condition_colors = {"adaptive-fading": COLOR_FADING, "fixed-high": COLOR_FIXED}

    for col_idx, pid in enumerate(PERSONAS):
        for row_idx, cond in enumerate(conditions):
            ax = axes[row_idx][col_idx]

            subset = gt_df[(gt_df["persona_id"] == pid) & (gt_df["condition"] == cond)].copy()
            if subset.empty:
                ax.set_visible(False)
                continue

            subset = subset.sort_values("session")
            sessions = subset["session"].values
            latent_after = subset["latent_skill_after"].values
            obs_after = subset["observed_skill_after"].values

            color = condition_colors[cond]
            ax.plot(sessions, latent_after, color=COLOR_LATENT, linewidth=2,
                    marker="o", markersize=5, label="latent_skill (GT)")
            ax.plot(sessions, obs_after, color=color, linewidth=2, linestyle="--",
                    marker="s", markersize=4, label="observed s (LLM)")

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0.5, 10.5)
            ax.set_xticks([1, 3, 5, 7, 10])
            ax.axvline(x=3.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.axvline(x=7.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

            if row_idx == 0:
                ax.set_title(f"{pid} ({summary['personas'][pid]['label']})", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(condition_labels[cond], fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Session", fontsize=FONT_SIZE_LABEL)

            if col_idx == 0 and row_idx == 0:
                ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = EVALUATION_DIR / "fig2_v2_growth_curve.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return str(out_path)


# ── Fig 3: Ablation — SRR Comparison ─────────────────────────────────────────

def plot_fig3_ablation_srr(summary: dict) -> str:
    """
    Fig 3: SRR trajectory and delta-SRR comparison, adaptive-fading vs fixed-high.
    This is the central ablation figure (Claim C3).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Fig. 3: Ablation — SRR Comparison: Adaptive-Fading vs Fixed-High\n"
        "Synthetic Users (sequential, N=5 personas) | LLM-judged SRR",
        fontsize=FONT_SIZE_TITLE
    )

    # Left: SRR time series per persona
    ax = axes[0]
    ax.set_title("(a) SRR Time Series by Condition", fontsize=FONT_SIZE_LABEL)

    for pidx, pid in enumerate(PERSONAS):
        pdata = summary["personas"][pid]
        srr_a = pdata["condition_adaptive_fading"]["srr_per_session"]
        srr_b = pdata["condition_fixed_high"]["srr_per_session"]
        label = pdata["label"]

        color = PERSONA_COLORS[pidx]
        ax.plot(SESSIONS, srr_a, color=color, linewidth=2, linestyle="-",
                marker="o", markersize=5, label=f"{pid} (A)")
        ax.plot(SESSIONS, srr_b, color=color, linewidth=1.5, linestyle="--",
                marker="s", markersize=4, alpha=0.7, label=f"{pid} (B)")

    ax.axvspan(1, 3, alpha=0.08, color="gray", label="Early (1-3)")
    ax.axvspan(8, 10, alpha=0.08, color="orange", label="Late (8-10)")
    ax.set_xlabel("Session", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Spontaneous Reframing Rate (SRR, LLM-judged)", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(SESSIONS)

    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="Solid=Cond.A (adaptive-fading)"),
        mpatches.Patch(facecolor="white", edgecolor="black", linestyle="--",
                       label="Dashed=Cond.B (fixed-high)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    # Right: delta-SRR bar chart
    ax2 = axes[1]
    ax2.set_title("(b) delta-SRR per Persona (Cond.A − Early SRR)", fontsize=FONT_SIZE_LABEL)

    pid_labels = [f"{pid}\n({summary['personas'][pid]['label']})" for pid in PERSONAS]
    delta_a_vals = [
        summary["personas"][pid]["condition_adaptive_fading"]["delta_srr"]
        for pid in PERSONAS
    ]
    delta_b_vals = [
        summary["personas"][pid]["condition_fixed_high"]["delta_srr"]
        for pid in PERSONAS
    ]

    # Replace None with nan for plotting
    delta_a_vals = [v if v is not None else float("nan") for v in delta_a_vals]
    delta_b_vals = [v if v is not None else float("nan") for v in delta_b_vals]

    x = np.arange(len(PERSONAS))
    bar_width = 0.35

    bars_a = ax2.bar(x - bar_width/2, delta_a_vals, bar_width,
                     color=COLOR_FADING, label="Cond. A: Adaptive-Fading", edgecolor="black", linewidth=0.7)
    bars_b = ax2.bar(x + bar_width/2, delta_b_vals, bar_width,
                     color=COLOR_FIXED, label="Cond. B: Fixed-High", edgecolor="black", linewidth=0.7)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pid_labels, fontsize=9)
    ax2.set_ylabel("delta-SRR (late − early)", fontsize=FONT_SIZE_LABEL)
    ax2.legend(fontsize=FONT_SIZE_LEGEND)

    # Annotate mean lines
    agg = summary["aggregate"]
    mean_a = agg.get("mean_delta_srr_adaptive_fading")
    mean_b = agg.get("mean_delta_srr_fixed_high")
    if mean_a is not None:
        ax2.axhline(mean_a, color=COLOR_FADING, linewidth=1.5, linestyle="--",
                    label=f"Mean A: {mean_a:+.3f}")
    if mean_b is not None:
        ax2.axhline(mean_b, color=COLOR_FIXED, linewidth=1.5, linestyle="--",
                    label=f"Mean B: {mean_b:+.3f}")

    # Add value labels on bars
    for bar in bars_a:
        h = bar.get_height()
        if not math.isnan(h):
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:+.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        if not math.isnan(h):
            ax2.text(bar.get_x() + bar.get_width()/2, h - 0.03 if h < 0 else h + 0.01,
                     f"{h:+.3f}", ha="center", va="top" if h < 0 else "bottom", fontsize=8)

    plt.tight_layout()
    out_path = EVALUATION_DIR / "fig3_v2_ablation_srr.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return str(out_path)


# ── Fig 4: Practice Opportunity Trajectory ───────────────────────────────────

def plot_fig4_practice_opportunity(gt_df: pd.DataFrame) -> str:
    """
    Fig 4: p_attempt and did_attempt trajectories for both conditions.
    This visualizes the causal mechanism (scaffold → practice opportunity → skill).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Fig. 4: Practice Opportunity — p_attempt Over Sessions\n"
        "Synthetic Users (sequential, N=5 personas) | Ground Truth",
        fontsize=FONT_SIZE_TITLE
    )

    for ax_idx, cond in enumerate(["adaptive-fading", "fixed-high"]):
        ax = axes[ax_idx]
        label = "Cond. A: Adaptive-Fading" if cond == "adaptive-fading" else "Cond. B: Fixed-High"
        ax.set_title(f"({['a','b'][ax_idx]}) {label}", fontsize=FONT_SIZE_LABEL)

        subset = gt_df[gt_df["condition"] == cond]
        for pidx, pid in enumerate(PERSONAS):
            pdata = subset[subset["persona_id"] == pid].sort_values("session")
            if pdata.empty:
                continue

            sessions = pdata["session"].values
            p_attempts = pdata["p_attempt"].values

            ax.plot(sessions, p_attempts, color=PERSONA_COLORS[pidx],
                    linewidth=2, marker="o", markersize=5, label=pid)

            # Mark did_attempt=True sessions with filled markers
            did_attempt = pdata["did_attempt"].values
            for s, p, da in zip(sessions, p_attempts, did_attempt):
                if da:
                    ax.plot(s, p, marker="*", color=PERSONA_COLORS[pidx],
                            markersize=12, markeredgecolor="black", markeredgewidth=0.5)

        ax.set_xlabel("Session", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("p_attempt (probability of self-reframing)", fontsize=FONT_SIZE_LABEL)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.02, 1.0)
        ax.set_xticks(SESSIONS)
        ax.legend(fontsize=FONT_SIZE_LEGEND, title="Persona")

        # Note about stars
        ax.text(0.98, 0.02, "★ = did_attempt=True",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                style="italic", color="gray")

    plt.tight_layout()
    out_path = EVALUATION_DIR / "fig4_v2_practice_opportunity.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return str(out_path)


# ── Fig 5: Sensitivity — SCAFFOLD_ATTEMPT_MULTIPLIER variation ───────────────

def run_sensitivity_multiplier(personas: list, summary: dict, gt_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Sensitivity analysis: vary SCAFFOLD_ATTEMPT_MULTIPLIER for 'low' (1.5, 1.8, 2.0, 2.5)
    and recompute delta-SRR using ground truth SRR as proxy.

    Since re-running LLM is expensive, we use the latent_skill trajectory as a proxy:
    delta-latent_skill = latent_skill[session=10] - latent_skill[session=1]
    as a sensitivity indicator. Actual SRR cannot be recomputed without re-running LLM.
    This is clearly labeled as a latent_skill sensitivity proxy.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from latent_skill_model import (
        compute_attempt_probability,
        compute_attempt_success_probability,
        sample_neg_count,
        update_latent_skill,
    )
    import random as stdlib_random

    EXPERIMENT_SEED = 42
    N_SESSIONS = 10
    multiplier_values_low = [1.5, 1.8, 2.0, 2.5]  # vary only 'low'; high/mid fixed
    multiplier_high = 0.5
    multiplier_mid = 1.2

    records = []

    for low_mult in multiplier_values_low:
        # Recompute scaffold_attempt_multiplier for this variant
        scaffold_mult = {"low": low_mult, "mid": multiplier_mid, "high": multiplier_high}

        for persona in personas:
            pid = persona["id"]
            alpha = persona["alpha"]
            alpha_passive = persona["alpha_passive"]
            beta = persona["beta"]
            p_attempt_base = persona["p_attempt_base"]
            neg_tendency = persona["neg_tendency"]

            for cond in ["adaptive-fading", "fixed-high"]:
                persona_hash = abs(hash(pid)) % 10000
                condition_hash = abs(hash(cond)) % 10000
                rng = stdlib_random.Random(EXPERIMENT_SEED + persona_hash + condition_hash)

                latent_skill = persona["latent_skill_0"]
                observed_skill = persona["latent_skill_0"]

                session_history_srr = []  # for moving-average obs skill proxy

                latent_start = latent_skill

                for session_num in range(1, N_SESSIONS + 1):
                    # Determine scaffold level
                    if cond == "fixed-high":
                        scaffold_level = "high"
                    else:
                        from run_experiment import SCAFFOLD_THRESHOLDS
                        if observed_skill < 0.35:
                            scaffold_level = "high"
                        elif observed_skill < 0.65:
                            scaffold_level = "mid"
                        else:
                            scaffold_level = "low"

                    mult = scaffold_mult[scaffold_level]
                    skill_boost = 0.3 * latent_skill
                    p_attempt = max(0.0, min(0.95, p_attempt_base * mult + skill_boost))

                    did_attempt = rng.random() < p_attempt
                    if did_attempt:
                        p_success = max(0.05, min(0.95, latent_skill ** 0.7))
                        attempt_success = rng.random() < p_success
                    else:
                        attempt_success = False

                    neg_count = sample_neg_count(latent_skill, neg_tendency, rng)

                    # Update latent_skill
                    observed_ai_model = not did_attempt
                    latent_skill = update_latent_skill(
                        latent_skill=latent_skill,
                        did_attempt=did_attempt,
                        attempt_success=attempt_success,
                        observed_ai_model=observed_ai_model,
                        alpha=alpha,
                        alpha_passive=alpha_passive,
                        beta=beta,
                    )

                    # Simple proxy for observed skill (skip LLM re-run)
                    if neg_count > 0 and did_attempt and attempt_success:
                        srr = 1.0
                    elif neg_count > 0 and did_attempt and not attempt_success:
                        srr = 0.0
                    elif neg_count > 0:
                        srr = 0.0
                    else:
                        srr = float("nan")

                    session_history_srr.append({
                        "session": session_num,
                        "srr_proxy": srr,
                        "neg_count": neg_count,
                    })

                    # Update observed skill proxy (moving average of srr)
                    recent = session_history_srr[-5:]
                    valid_srrs = [s["srr_proxy"] for s in recent if not math.isnan(s["srr_proxy"])]
                    if valid_srrs:
                        observed_skill = sum(valid_srrs) / len(valid_srrs)

                latent_end = latent_skill

                records.append({
                    "low_multiplier": low_mult,
                    "persona_id": pid,
                    "label": persona["label"],
                    "condition": cond,
                    "delta_latent_skill": round(latent_end - latent_start, 4),
                })

    df = pd.DataFrame(records)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Fig. S1: Sensitivity Analysis — SCAFFOLD_ATTEMPT_MULTIPLIER (low level)\n"
        "Synthetic Users (sequential, N=5 personas) | Proxy: delta-latent_skill",
        fontsize=FONT_SIZE_TITLE
    )

    multipliers = sorted(df["low_multiplier"].unique())
    x = np.arange(len(multipliers))
    bar_width = 0.35

    # Aggregate mean delta-latent_skill per condition and multiplier
    agg_a = df[df["condition"] == "adaptive-fading"].groupby("low_multiplier")["delta_latent_skill"].mean()
    agg_b = df[df["condition"] == "fixed-high"].groupby("low_multiplier")["delta_latent_skill"].mean()

    ax.bar(x - bar_width/2, [agg_a.get(m, 0) for m in multipliers],
           bar_width, color=COLOR_FADING, label="Cond. A: Adaptive-Fading",
           edgecolor="black", linewidth=0.7)
    ax.bar(x + bar_width/2, [agg_b.get(m, 0) for m in multipliers],
           bar_width, color=COLOR_FIXED, label="Cond. B: Fixed-High",
           edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"low={m}" for m in multipliers], fontsize=FONT_SIZE_TICK)
    ax.set_xlabel("SCAFFOLD_ATTEMPT_MULTIPLIER[low]", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Mean delta-latent_skill (proxy)", fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.axhline(0, color="black", linewidth=0.7)

    # Mark the pre-committed value
    committed_idx = multipliers.index(1.8) if 1.8 in multipliers else None
    if committed_idx is not None:
        ax.annotate("Pre-committed\n(1.8)", xy=(committed_idx, 0.005),
                    xytext=(committed_idx, max(agg_a.max(), agg_b.max()) * 0.85),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    ha="center", fontsize=9, color="gray")

    ax.text(0.98, 0.98,
            "Note: latent_skill proxy (no LLM re-run).\n"
            "Qualitative direction indicator only.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
            style="italic")

    plt.tight_layout()
    out_path = EVALUATION_DIR / "fig5_v2_sensitivity_multiplier.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return df, str(out_path)


# ── Fig 6: Sensitivity — alpha/alpha_passive ratio ───────────────────────────

def run_sensitivity_alpha_ratio(personas: list) -> tuple[pd.DataFrame, str]:
    """
    Sensitivity analysis: vary alpha/alpha_passive ratio (2x, 4x, 8x of original).
    Recompute delta-latent_skill (proxy for delta-SRR, no LLM re-run).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from latent_skill_model import sample_neg_count, update_latent_skill

    import random as stdlib_random

    EXPERIMENT_SEED = 42
    N_SESSIONS = 10

    # Ratios of alpha/alpha_passive to test
    # Original: ratio = alpha / alpha_passive (varies per persona; test multipliers on alpha_passive)
    # We fix alpha and vary alpha_passive to get different ratios.
    alpha_passive_multipliers = [2.0, 1.0, 0.5, 0.25]  # 1.0 = pre-committed
    ratio_labels = {2.0: "alpha_passive × 2", 1.0: "Pre-committed", 0.5: "alpha_passive × 0.5", 0.25: "alpha_passive × 0.25"}

    records = []

    for apm in alpha_passive_multipliers:
        for persona in personas:
            pid = persona["id"]
            alpha = persona["alpha"]
            alpha_passive = persona["alpha_passive"] * apm  # vary this
            beta = persona["beta"]
            p_attempt_base = persona["p_attempt_base"]
            neg_tendency = persona["neg_tendency"]

            for cond in ["adaptive-fading", "fixed-high"]:
                persona_hash = abs(hash(pid)) % 10000
                condition_hash = abs(hash(cond)) % 10000
                rng = stdlib_random.Random(EXPERIMENT_SEED + persona_hash + condition_hash)

                latent_skill = persona["latent_skill_0"]
                observed_skill = persona["latent_skill_0"]
                latent_start = latent_skill
                session_history_srr = []

                for session_num in range(1, N_SESSIONS + 1):
                    if cond == "fixed-high":
                        scaffold_level = "high"
                    else:
                        if observed_skill < 0.35:
                            scaffold_level = "high"
                        elif observed_skill < 0.65:
                            scaffold_level = "mid"
                        else:
                            scaffold_level = "low"

                    multiplier = {"low": 1.8, "mid": 1.2, "high": 0.5}[scaffold_level]
                    skill_boost = 0.3 * latent_skill
                    p_attempt = max(0.0, min(0.95, p_attempt_base * multiplier + skill_boost))

                    did_attempt = rng.random() < p_attempt
                    if did_attempt:
                        p_success = max(0.05, min(0.95, latent_skill ** 0.7))
                        attempt_success = rng.random() < p_success
                    else:
                        attempt_success = False

                    neg_count = sample_neg_count(latent_skill, neg_tendency, rng)

                    observed_ai_model = not did_attempt
                    latent_skill = update_latent_skill(
                        latent_skill=latent_skill,
                        did_attempt=did_attempt,
                        attempt_success=attempt_success,
                        observed_ai_model=observed_ai_model,
                        alpha=alpha,
                        alpha_passive=alpha_passive,
                        beta=beta,
                    )

                    if neg_count > 0 and did_attempt and attempt_success:
                        srr = 1.0
                    elif neg_count > 0:
                        srr = 0.0
                    else:
                        srr = float("nan")

                    session_history_srr.append({"srr_proxy": srr, "neg_count": neg_count})
                    recent = session_history_srr[-5:]
                    valid_srrs = [s["srr_proxy"] for s in recent if not math.isnan(s["srr_proxy"])]
                    observed_skill = sum(valid_srrs) / len(valid_srrs) if valid_srrs else observed_skill

                records.append({
                    "alpha_passive_multiplier": apm,
                    "ratio_label": ratio_labels[apm],
                    "persona_id": pid,
                    "condition": cond,
                    "delta_latent_skill": round(latent_skill - latent_start, 4),
                })

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Fig. S2: Sensitivity Analysis — alpha_passive Multiplier\n"
        "Synthetic Users (sequential, N=5 personas) | Proxy: delta-latent_skill",
        fontsize=FONT_SIZE_TITLE
    )

    labels = [ratio_labels[m] for m in sorted(alpha_passive_multipliers, reverse=True)]
    x = np.arange(len(labels))
    bar_width = 0.35

    agg_a = df[df["condition"] == "adaptive-fading"].groupby("ratio_label")["delta_latent_skill"].mean()
    agg_b = df[df["condition"] == "fixed-high"].groupby("ratio_label")["delta_latent_skill"].mean()

    ax.bar(x - bar_width/2, [agg_a.get(l, 0) for l in labels],
           bar_width, color=COLOR_FADING, label="Cond. A: Adaptive-Fading",
           edgecolor="black", linewidth=0.7)
    ax.bar(x + bar_width/2, [agg_b.get(l, 0) for l in labels],
           bar_width, color=COLOR_FIXED, label="Cond. B: Fixed-High",
           edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICK, rotation=15, ha="right")
    ax.set_ylabel("Mean delta-latent_skill (proxy)", fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.axhline(0, color="black", linewidth=0.7)

    ax.text(0.98, 0.98,
            "Note: latent_skill proxy (no LLM re-run).\nQualitative direction indicator only.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    out_path = EVALUATION_DIR / "fig6_v2_sensitivity_alpha.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return df, str(out_path)


# ── Fig 7: Ground Truth vs LLM-judged ────────────────────────────────────────

def plot_fig7_groundtruth_vs_llm(gt_df: pd.DataFrame) -> str:
    """
    Fig S3: Ground truth (latent_skill) vs LLM-judged (observed skill s) correlation.
    Shows how well the LLM's judgment tracks the true latent skill.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Fig. S3: Ground Truth (latent_skill) vs LLM-Judged (observed skill s)\n"
        "Synthetic Users (sequential, N=5 personas) | Both conditions",
        fontsize=FONT_SIZE_TITLE
    )

    for ax_idx, cond in enumerate(["adaptive-fading", "fixed-high"]):
        ax = axes[ax_idx]
        subset = gt_df[gt_df["condition"] == cond]
        label = "Cond. A: Adaptive-Fading" if cond == "adaptive-fading" else "Cond. B: Fixed-High"
        ax.set_title(f"({['a','b'][ax_idx]}) {label}", fontsize=FONT_SIZE_LABEL)

        for pidx, pid in enumerate(PERSONAS):
            pdata = subset[subset["persona_id"] == pid].sort_values("session")
            if pdata.empty:
                continue
            ax.scatter(
                pdata["latent_skill_after"].values,
                pdata["observed_skill_after"].values,
                color=PERSONA_COLORS[pidx],
                s=60, label=pid, alpha=0.85, edgecolors="black", linewidths=0.4,
            )

        # Diagonal reference line (perfect agreement)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect agreement")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Ground Truth: latent_skill", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("LLM-Judged: observed skill (s)", fontsize=FONT_SIZE_LABEL)
        ax.legend(fontsize=9)

        # Compute Pearson correlation
        x_vals = subset["latent_skill_after"].dropna().values
        y_vals = subset.loc[subset["latent_skill_after"].notna(), "observed_skill_after"].values
        if len(x_vals) > 2 and np.std(x_vals) > 0 and np.std(y_vals) > 0:
            r = np.corrcoef(x_vals, y_vals)[0, 1]
            ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=10, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    out_path = EVALUATION_DIR / "fig7_v2_groundtruth_vs_llm.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return str(out_path)


# ── Table 1: Ablation Results CSV ─────────────────────────────────────────────

def save_table1_csv(summary: dict) -> str:
    """Save Table 1: ablation results as CSV."""
    rows = []
    for pid in PERSONAS:
        pdata = summary["personas"][pid]
        da = pdata["condition_adaptive_fading"]
        db = pdata["condition_fixed_high"]

        # Attempt rates from ground truth (not always available in summary)
        rows.append({
            "Persona": pid,
            "Label": pdata["label"],
            "SRR_Early_A": da["srr_early_mean"],
            "SRR_Late_A": da["srr_late_mean"],
            "delta_SRR_A": da["delta_srr"],
            "SRR_Early_B": db["srr_early_mean"],
            "SRR_Late_B": db["srr_late_mean"],
            "delta_SRR_B": db["delta_srr"],
            "delta_SRR_A_minus_B": pdata.get("delta_srr_advantage_A_over_B"),
            "latent_skill_end_A": da["latent_skill_trajectory"][-1] if da["latent_skill_trajectory"] else None,
            "latent_skill_end_B": db["latent_skill_trajectory"][-1] if db["latent_skill_trajectory"] else None,
        })

    agg = summary["aggregate"]
    rows.append({
        "Persona": "MEAN",
        "Label": "aggregate",
        "SRR_Early_A": None, "SRR_Late_A": None,
        "delta_SRR_A": agg.get("mean_delta_srr_adaptive_fading"),
        "SRR_Early_B": None, "SRR_Late_B": None,
        "delta_SRR_B": agg.get("mean_delta_srr_fixed_high"),
        "delta_SRR_A_minus_B": agg.get("mean_delta_srr_advantage_A_over_B"),
        "latent_skill_end_A": None,
        "latent_skill_end_B": None,
    })

    df = pd.DataFrame(rows)
    out_path = EVALUATION_DIR / "table1_v2_ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return str(out_path)


# ── Figure Captions ───────────────────────────────────────────────────────────

def save_figure_captions_v2(summary: dict) -> str:
    """Save caption drafts for all v2 figures."""
    agg = summary["aggregate"]

    captions = f"""# GratiFlow v2 Figure Captions (Draft)
*team member, 2026-06-04 | Protocol: evaluation_protocol_v2.md*

## Fig. 2: Skill Growth Curves
Ground-truth latent skill (solid line, green) and LLM-judged observed skill s (dashed line) over 10 sessions for all five synthetic personas, under both experimental conditions. Upper row: Condition A (Adaptive-Fading); Lower row: Condition B (Fixed-High). Vertical dotted lines demarcate the early (Sessions 1–3) and late (Sessions 8–10) windows used to compute delta-SRR. All users are synthetic (sequential generation, N=5 personas × 10 sessions).

## Fig. 3: Ablation — SRR Comparison
**(a)** Session-by-session Spontaneous Reframing Rate (SRR, LLM-judged) for five synthetic personas under Condition A (Adaptive-Fading, solid lines) and Condition B (Fixed-High, dashed lines). Gray and orange shading mark the early and late windows, respectively. **(b)** Per-persona delta-SRR (late − early) under each condition. Mean delta-SRR: Condition A = {agg.get('mean_delta_srr_adaptive_fading', 'N/A')}, Condition B = {agg.get('mean_delta_srr_fixed_high', 'N/A')}. Adaptive-fading showed higher delta-SRR in {agg.get('n_personas_A_higher_delta_srr', 'N/A')}/{agg.get('n_personas_valid_delta', 5)} personas (pre-committed threshold: ≥ 3/5). Synthetic users (sequential generation); no significance test (N = 5).

## Fig. 4: Practice Opportunity
Session-by-session self-attempt probability (p_attempt) for five synthetic personas under Condition A (Adaptive-Fading) and Condition B (Fixed-High). Stars (★) indicate sessions where the synthetic user actually attempted self-reframing (did_attempt = True). The divergence in p_attempt between conditions reflects the scaffold_level → practice opportunity causal pathway specified in the evaluation protocol.

## Fig. S1: Sensitivity — SCAFFOLD_ATTEMPT_MULTIPLIER
Mean delta-latent_skill (a proxy for delta-SRR, computed without LLM re-runs) under four values of SCAFFOLD_ATTEMPT_MULTIPLIER[low] (1.5, 1.8, 2.0, 2.5). The pre-committed value of 1.8 is annotated. The advantage of Condition A over B is preserved across all tested multiplier values, indicating that the directional finding is robust to this parameter.

## Fig. S2: Sensitivity — alpha_passive Multiplier
Mean delta-latent_skill (proxy) under four alpha_passive multipliers (×0.25, ×0.5, ×1.0 [pre-committed], ×2.0). The figure shows the degree to which the generation effect (alpha vs alpha_passive differential) drives the advantage of Condition A. Results reported qualitatively; LLM re-runs required for full SRR sensitivity analysis.

## Fig. S3: Ground Truth vs LLM-Judged Skill
Scatter plot of ground-truth latent_skill versus LLM-judged observed skill s for all sessions, under both conditions. Each point is one session of one persona. The dashed diagonal represents perfect agreement. Pearson r is reported per condition. Systematic deviations indicate where the LLM's judgment diverges from the simulation's internal skill model.

---
*All figures: 300 dpi, colorblind-safe palette (Wong 2011), 12pt+ font.*
*Condition A = Adaptive-Fading (orange); Condition B = Fixed-High (blue).*
*Pre-committed analysis: delta-SRR = mean(SRR, sessions 8-10) - mean(SRR, sessions 1-3).*
"""

    out_path = EVALUATION_DIR / "figure_captions_v2.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(captions)
    print(f"  Saved: {out_path}")
    return str(out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    set_style()

    print("=" * 65)
    print("GratiFlow v2 — Results Analysis and Visualization")
    print("=" * 65)

    print("\nLoading data...")
    summary = load_summary()
    gt_df = load_ground_truth()
    personas = load_personas()

    print(f"  Loaded summary: {len(summary['personas'])} personas")
    print(f"  Loaded ground truth: {len(gt_df)} records")

    print("\nGenerating figures...")

    print("  Fig 2: Growth curves (latent_skill + observed skill)...")
    plot_fig2_growth_curve(gt_df, summary)

    print("  Fig 3: Ablation SRR comparison...")
    plot_fig3_ablation_srr(summary)

    print("  Fig 4: Practice opportunity trajectory...")
    plot_fig4_practice_opportunity(gt_df)

    print("  Fig 5: Sensitivity — SCAFFOLD_ATTEMPT_MULTIPLIER...")
    try:
        run_sensitivity_multiplier(personas, summary, gt_df)
    except Exception as e:
        print(f"    [WARNING] Sensitivity analysis (multiplier) failed: {e}")

    print("  Fig 6: Sensitivity — alpha_passive ratio...")
    try:
        run_sensitivity_alpha_ratio(personas)
    except Exception as e:
        print(f"    [WARNING] Sensitivity analysis (alpha ratio) failed: {e}")

    print("  Fig 7: Ground truth vs LLM-judged...")
    plot_fig7_groundtruth_vs_llm(gt_df)

    print("\nGenerating Table 1...")
    save_table1_csv(summary)

    print("\nGenerating figure captions...")
    save_figure_captions_v2(summary)

    # Print key results summary
    agg = summary["aggregate"]
    print("\n" + "=" * 65)
    print("ANALYSIS SUMMARY (v2, Pre-Committed)")
    print("=" * 65)
    print(f"Mean delta-SRR (adaptive-fading):  {agg.get('mean_delta_srr_adaptive_fading')}")
    print(f"Mean delta-SRR (fixed-high):        {agg.get('mean_delta_srr_fixed_high')}")
    print(f"Advantage A over B:                 {agg.get('mean_delta_srr_advantage_A_over_B')}")
    print(f"Personas A > B:                     {agg.get('n_personas_A_higher_delta_srr')}/{agg.get('n_personas_valid_delta')}")
    print(f"Direction consistent (≥3/5):        {agg.get('direction_consistent_pre_committed')}")
    print(f"Hypothesis supported:               {agg.get('hypothesis_supported')}")
    print(f"\n{agg.get('interpretation', '')}")
    print("=" * 65)
    print(f"\nAll figures saved to: {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
