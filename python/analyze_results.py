"""
analyze_results.py
===================
GratiFlow Phase 1 Evaluation — Result Analysis and Figure Generation

Generates publication-quality figures (300 dpi, colorblind-friendly) for:
  Fig. 2: Growth Curve (skill score s over sessions, 5 personas, Condition A)
  Fig. 3: Ablation SRR Comparison (fading vs fixed, 5 personas)
  Table 1: Ablation Results Summary
  Fig. 4: Scaffolding Level Tracking vs s
  Fig. 5: Mood Trajectory (Condition A)

All figures explicitly state "Synthetic Users" in captions and plot labels.
Color palette: colorblind-safe (IBM/Wong 8-color palette).

IMPORTANT: All data is from SYNTHETIC users (LLM-generated). No real participants.

Author: team member (experiment lead, the research team)
Date: 2026-06-04
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server-side rendering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "experiments"
RESULTS_DIR = DATA_PROCESSED / "results"
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colorblind-safe palette (IBM/Wong 8-color) ────────────────────────────────
# Reference: Wong (2011) Nature Methods, IBM Color Blind Safe palette

COLORS = {
    "blue":    "#648FFF",   # skill/fading line
    "orange":  "#FE6100",   # mood line
    "purple":  "#DC267F",   # fixed_high line
    "green":   "#785EF0",   # P3
    "teal":    "#009E73",   # P4
    "yellow":  "#F0E442",   # unused here, keeping for extension
    "red":     "#D55E00",   # error/negative
    "gray":    "#999999",   # individual thin lines
}

PERSONA_COLORS = ["#648FFF", "#FE6100", "#785EF0", "#009E73", "#DC267F"]
PERSONA_LABELS = {
    "P1": "P1: 初心者→着実",
    "P2": "P2: 初心者→停滞",
    "P3": "P3: 中級→急成長",
    "P4": "P4: 初心者→急成長",
    "P5": "P5: 中級→緩やか",
}

SCAFFOLD_BAND_COLORS = {
    "high": "#FFE5CC",   # light orange
    "mid":  "#E5F5E5",   # light green
    "low":  "#CCE5FF",   # light blue
}

# ── Global matplotlib settings (publication quality) ─────────────────────────

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Hiragino Sans", "Yu Gothic", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
})

SESSIONS = list(range(1, 11))

SYNTHETIC_NOTE = "(Synthetic Users, N=5 personas × 10 sessions)"

# ── Load data ─────────────────────────────────────────────────────────────────

def load_summary() -> dict:
    path = RESULTS_DIR / "summary_statistics.json"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}\nRun run_experiment.py first.")
    with open(path) as f:
        return json.load(f)


def load_persona_sessions(condition: str) -> dict:
    """Load session records for all personas under a condition.

    Checks both 'condition_{condition}' and 'condition_{condition}_high' directories
    (the latter is created when condition='fixed' but the runner saves to 'fixed_high').
    """
    cond_dir = RESULTS_DIR / f"condition_{condition}"
    # Fallback: condition_fixed_high when condition='fixed'
    if not cond_dir.exists() or not any(cond_dir.iterdir()):
        alt = RESULTS_DIR / f"condition_{condition}_high"
        if alt.exists():
            cond_dir = alt
    sessions = {}
    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        path = cond_dir / f"{pid}_sessions.json"
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        with open(path) as f:
            sessions[pid] = json.load(f)
    return sessions


# ── Helper: extract time series ───────────────────────────────────────────────

def get_series(sessions_list: list, key: str) -> list:
    return [s[key] for s in sorted(sessions_list, key=lambda x: x["session"])]


# ── Fig. 2: Growth Curve (skill + scaffoldLevel bands) ───────────────────────

def fig2_growth_curve(summary: dict, sessions_fading: dict) -> None:
    """
    Fig. 2: Skill score s over sessions for 5 personas (Condition A: fading).
    Background bands show scaffolding level zones.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Background bands for scaffolding levels
    ax.axhspan(0.0,  0.35, alpha=0.12, color="#FF8800", label="high scaffold zone", zorder=0)
    ax.axhspan(0.35, 0.65, alpha=0.12, color="#33AA33", label="mid scaffold zone", zorder=0)
    ax.axhspan(0.65, 1.0,  alpha=0.12, color="#2266FF", label="low scaffold zone", zorder=0)

    # Threshold lines
    ax.axhline(0.35, color="#FF8800", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
    ax.axhline(0.65, color="#2266FF", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

    # Band labels (right side)
    ax.text(10.3, 0.17, "high", color="#CC6600", fontsize=8, va="center")
    ax.text(10.3, 0.50, "mid",  color="#228822", fontsize=8, va="center")
    ax.text(10.3, 0.80, "low",  color="#1144CC", fontsize=8, va="center")

    # Persona lines
    for i, pid in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        skills = get_series(sessions_fading[pid], "skill_after")
        ax.plot(SESSIONS, skills,
                color=PERSONA_COLORS[i],
                marker="o", markersize=4,
                label=PERSONA_LABELS[pid], zorder=3)

    ax.set_xlim(0.5, 10.8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Skill Score (s)", fontsize=12)
    ax.set_title(
        f"Fig. 2: Growth Curve — Skill Score over Sessions\n"
        f"(Condition A: Adaptive Scaffolding-Fading) {SYNTHETIC_NOTE}",
        fontsize=11, pad=10
    )
    ax.set_xticks(SESSIONS)

    # Legend: personas only (band legend via text annotations)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig2_growth_curve.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Fig. 3: Ablation SRR Comparison ──────────────────────────────────────────

def fig3_ablation_srr(summary: dict, sessions_fading: dict, sessions_fixed: dict) -> None:
    """
    Fig. 3: Spontaneous Reframing Rate (SRR) over sessions.
    Condition A (fading) vs Condition B (fixed_high).
    Bold lines = 5-persona mean; thin lines = individual personas.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Collect per-persona SRR
    srr_fading = np.array([get_series(sessions_fading[pid], "spontaneous_rate")
                            for pid in ["P1","P2","P3","P4","P5"]])
    srr_fixed  = np.array([get_series(sessions_fixed[pid],  "spontaneous_rate")
                            for pid in ["P1","P2","P3","P4","P5"]])

    # Individual thin lines
    for i in range(5):
        ax.plot(SESSIONS, srr_fading[i], color=COLORS["blue"],
                alpha=0.20, linewidth=1.0, zorder=1)
        ax.plot(SESSIONS, srr_fixed[i],  color=COLORS["purple"],
                alpha=0.20, linewidth=1.0, zorder=1)

    # Mean lines (bold)
    mean_fading = srr_fading.mean(axis=0)
    mean_fixed  = srr_fixed.mean(axis=0)

    ax.plot(SESSIONS, mean_fading, color=COLORS["blue"],
            linewidth=2.5, marker="o", markersize=5, label="Cond. A: Adaptive Fading", zorder=3)
    ax.plot(SESSIONS, mean_fixed,  color=COLORS["purple"],
            linewidth=2.5, marker="s", markersize=5, linestyle="--",
            label="Cond. B: Fixed High Scaffold", zorder=3)

    # delta-SRR annotation
    agg = summary["aggregate"]
    ax.annotate(
        f"Δ-SRR (fading): {agg['mean_delta_srr_fading']:+.3f}\n"
        f"Δ-SRR (fixed):  {agg['mean_delta_srr_fixed']:+.3f}",
        xy=(9, max(mean_fading[-2], mean_fading[-1])),
        xytext=(6.5, 0.75),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="#555555"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Spontaneous Reframing Rate\n(SRR, LLM-judged)", fontsize=11)
    ax.set_title(
        f"Fig. 3: Ablation — SRR Comparison (Fading vs Fixed Scaffold)\n{SYNTHETIC_NOTE}",
        fontsize=11, pad=10
    )
    ax.set_xticks(SESSIONS)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig3_ablation_srr.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Fig. 4: Scaffold Level Tracking ──────────────────────────────────────────

def fig4_scaffold_tracking(summary: dict, sessions_fading: dict) -> None:
    """
    Fig. 4: Scaffold level (as numeric) and skill score s over sessions per persona.
    Shows that scaffoldLevel tracks s as designed.
    """
    SCAFFOLD_NUM = {"high": 0, "mid": 1, "low": 2}
    SCAFFOLD_LABEL = {0: "high", 1: "mid", 2: "low"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    axes_flat = axes.flatten()

    for i, pid in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        ax = axes_flat[i]
        sessions = sorted(sessions_fading[pid], key=lambda x: x["session"])
        skills = [s["skill_after"] for s in sessions]
        scaffolds_num = [SCAFFOLD_NUM[s["scaffold_level"]] for s in sessions]

        # Skill line (left axis)
        color_skill = COLORS["blue"]
        color_scaffold = COLORS["orange"]

        ax2 = ax.twinx()

        ax.plot(SESSIONS, skills, color=color_skill, marker="o", markersize=4,
                linewidth=2.0, label="Skill (s)", zorder=3)
        ax.axhline(0.35, color=color_skill, linewidth=0.7, linestyle=":", alpha=0.5)
        ax.axhline(0.65, color=color_skill, linewidth=0.7, linestyle=":", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Skill (s)", color=color_skill, fontsize=9)
        ax.tick_params(axis="y", labelcolor=color_skill, labelsize=8)

        # Scaffold level (right axis, step)
        ax2.step(SESSIONS, scaffolds_num, color=color_scaffold,
                 where="post", linewidth=2.0, label="Scaffold Level", zorder=2)
        ax2.set_ylim(-0.3, 2.3)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(["high", "mid", "low"], fontsize=8, color=color_scaffold)
        ax2.tick_params(axis="y", labelcolor=color_scaffold)

        ax.set_title(f"{pid}: {PERSONA_LABELS[pid]}", fontsize=9)
        ax.set_xticks(SESSIONS)
        ax.set_xticklabels(SESSIONS, fontsize=7)
        ax.set_xlabel("Session", fontsize=9)

    # Remove unused subplot (6th cell)
    axes_flat[5].set_visible(False)

    # Common legend
    skill_patch = mpatches.Patch(color=COLORS["blue"], label="Skill Score (s)")
    scaffold_patch = mpatches.Patch(color=COLORS["orange"], label="Scaffold Level")
    fig.legend(handles=[skill_patch, scaffold_patch],
               loc="lower right", bbox_to_anchor=(0.95, 0.05), fontsize=10)

    fig.suptitle(
        f"Fig. 4: Scaffold Level Tracking vs Skill Score (Condition A: Fading)\n{SYNTHETIC_NOTE}",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig4_scaffold_tracking.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Fig. 5: Mood Trajectory ───────────────────────────────────────────────────

def fig5_mood_trajectory(summary: dict, sessions_fading: dict) -> None:
    """
    Fig. 5: Mood score over sessions for 5 personas (Condition A).
    Reference indicator only (not main claim).
    """
    fig, ax = plt.subplots(figsize=(8, 4.0))

    for i, pid in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        moods = get_series(sessions_fading[pid], "mood")
        ax.plot(SESSIONS, moods, color=PERSONA_COLORS[i],
                marker="o", markersize=4, label=PERSONA_LABELS[pid])

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_yticks([1, 3, 5, 7, 9])
    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Mood Score (1–10, LLM-judged)", fontsize=11)
    ax.set_title(
        f"Fig. 5: Mood Trajectory (Condition A: Adaptive Fading)\n"
        f"[Reference Indicator — Not Primary Claim] {SYNTHETIC_NOTE}",
        fontsize=10, pad=8
    )
    ax.set_xticks(SESSIONS)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "fig5_mood_trajectory.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Table 1: Ablation Results Summary ────────────────────────────────────────

def table1_ablation_summary(summary: dict) -> None:
    """
    Table 1: delta-SRR, Final s, Final mood for each condition.
    Saved as CSV and printed to stdout.
    """
    personas_data = summary["personas"]
    agg = summary["aggregate"]

    rows = []
    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        p = personas_data[pid]
        rows.append({
            "Persona": pid,
            "Label": p["label"],
            "Δ-SRR (Fading)":     f"{p['condition_fading']['delta_srr']:+.3f}",
            "Δ-SRR (Fixed)":      f"{p['condition_fixed']['delta_srr']:+.3f}",
            "Fading Advantage":   f"{p['delta_srr_advantage_fading']:+.3f}",
            "Final s (Fading)":   f"{p['condition_fading']['final_skill']:.3f}",
            "Final s (Fixed)":    f"{p['condition_fixed']['final_skill']:.3f}",
            "Final Mood (Fading)": p['condition_fading']['final_mood'],
            "Final Mood (Fixed)":  p['condition_fixed']['final_mood'],
        })

    rows.append({
        "Persona": "Mean",
        "Label": "(5 personas)",
        "Δ-SRR (Fading)":     f"{agg['mean_delta_srr_fading']:+.3f}",
        "Δ-SRR (Fixed)":      f"{agg['mean_delta_srr_fixed']:+.3f}",
        "Fading Advantage":   f"{agg['mean_delta_srr_advantage']:+.3f}",
        "Final s (Fading)":   "—",
        "Final s (Fixed)":    "—",
        "Final Mood (Fading)": "—",
        "Final Mood (Fixed)":  "—",
    })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = OUTPUT_DIR / "table1_ablation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {csv_path}")

    # Print to stdout
    print("\n=== Table 1: Ablation Results ===")
    print(f"(Synthetic Users, N=5 personas x 10 sessions)")
    print(df.to_string(index=False))
    print(f"\nNote: N=5 is underpowered for significance testing.")
    print(f"Personas with fading > fixed delta-SRR: {agg['n_personas_fading_higher_delta_srr']}/5")
    print(f"Direction consistent: {agg['direction_consistent']}")


# ── Caption file ──────────────────────────────────────────────────────────────

def save_captions() -> None:
    """Save figure caption proposals to a markdown file."""
    captions = """# GratiFlow Figure Captions (Draft)

All figures are based on **synthetic users** generated by GPT-5.4-mini.
No real participants are involved in this study.

---

## Fig. 2: Growth Curve
**Caption**: Skill score *s* over 10 sessions for 5 synthetic personas under Condition A
(adaptive scaffolding-fading). Background bands indicate scaffolding level zones:
high (s < 0.35), mid (0.35 ≤ s < 0.65), and low (s ≥ 0.65).
As *s* increases, the system transitions to lower scaffolding levels,
confirming that the fading mechanism tracks skill progression as designed.
*(Synthetic Users, N=5 personas × 10 sessions)*

---

## Fig. 3: Ablation — SRR Comparison
**Caption**: Spontaneous Reframing Rate (SRR, LLM-judged) over 10 sessions
comparing Condition A (adaptive fading, blue) and Condition B (fixed high scaffold, purple).
Bold lines represent the 5-persona mean; thin lines show individual persona trajectories.
Condition A shows a higher delta-SRR in [N]/5 personas, suggesting that adaptive
scaffolding-fading promotes internalization of reframing skills.
*(Synthetic Users, N=5 personas × 10 sessions;
N is too small for significance testing — direction and consistency reported as PoC evidence.)*

---

## Fig. 4: Scaffold Level Tracking
**Caption**: Scaffold level (step plot, right axis) and skill score *s* (line, left axis)
per session for each synthetic persona under Condition A. The scaffold level transitions
from high to mid to low in response to increasing *s*, confirming the deterministic
rule-based fading mechanism operates as designed.
*(Synthetic Users, N=5 personas × 10 sessions)*

---

## Fig. 5: Mood Trajectory
**Caption**: Mood score (1–10, LLM-judged) over 10 sessions for 5 synthetic personas
under Condition A. Mood is reported as a reference indicator and is not the primary claim
of this paper. Trends vary across personas reflecting the heterogeneous trajectories
designed in the synthetic user specification.
*(Synthetic Users, N=5 personas × 10 sessions)*

---

## Table 1: Ablation Results Summary
**Caption**: Ablation study results comparing Condition A (adaptive fading) and
Condition B (fixed high scaffold) across 5 synthetic personas.
Delta-SRR (Δ-SRR) is defined as the difference between mean SRR in sessions 8–10
and mean SRR in sessions 1–3. A positive Δ-SRR indicates growth in spontaneous
reframing rate. Fading Advantage = Δ-SRR(Fading) − Δ-SRR(Fixed).
*(Synthetic Users, N=5 personas × 10 sessions;
no significance test due to small N — effect direction reported as PoC.)*
"""
    cap_path = OUTPUT_DIR / "figure_captions.md"
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(captions)
    print(f"Saved: {cap_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("GratiFlow Phase 1 — Result Analysis and Figure Generation")
    print("IMPORTANT: SYNTHETIC USERS. No real participants.")
    print("=" * 60)

    print("\nLoading data...")
    summary = load_summary()
    sessions_fading = load_persona_sessions("fading")
    sessions_fixed  = load_persona_sessions("fixed_high")
    print("  Data loaded.")

    print("\nGenerating figures...")
    fig2_growth_curve(summary, sessions_fading)
    fig3_ablation_srr(summary, sessions_fading, sessions_fixed)
    fig4_scaffold_tracking(summary, sessions_fading)
    fig5_mood_trajectory(summary, sessions_fading)

    print("\nGenerating Table 1...")
    table1_ablation_summary(summary)

    print("\nSaving figure captions...")
    save_captions()

    print("\n" + "=" * 60)
    print("All outputs saved to:")
    print(f"  {OUTPUT_DIR}/")
    print("  fig2_growth_curve.png")
    print("  fig3_ablation_srr.png")
    print("  fig4_scaffold_tracking.png")
    print("  fig5_mood_trajectory.png")
    print("  table1_ablation_results.csv")
    print("  figure_captions.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
