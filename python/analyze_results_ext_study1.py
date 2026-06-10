"""
analyze_results_ext_study1.py
==============================
GratiFlow Extended Study 1 — Analysis and Visualization

Reads ext_study1 session data and produces:
  1. delta-SRR bar chart (adaptive-fading vs fixed-high, per persona + mean)
  2. SRR time series plot (sessions 1-14, per condition, averaged over personas)
  3. Latent skill trajectory plot
  4. is_echo rate per condition per session
  5. Summary statistics to data/processed/ext_study1/results/

All figures saved to:
  /data/processed/ext_study1/figures/

Figure labeling:
  - All figures include "Synthetic Users (multi-turn loop, N=10)" in caption/subtitle.
  - 300 dpi, colorblind-friendly palette.

Author: team member (experiment lead, the research team)
Date: 2026-06-05
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED_EXT = BASE_DIR / "data" / "processed" / "ext_study1"
FIGURES_DIR = DATA_PROCESSED_EXT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = DATA_PROCESSED_EXT / "results" / "summary_statistics_ext_study1.json"

# Colorblind-friendly palette (Wong 2011)
COLOR_A = "#0072B2"   # blue → adaptive-fading
COLOR_B = "#E69F00"   # orange → fixed-high
COLOR_MID = "#56B4E9" # light blue (for accent)

N_SESSIONS = 14
CONDITIONS = ["adaptive-fading", "fixed-high"]
CONDITION_LABELS = {"adaptive-fading": "Adaptive-Fading (A)", "fixed-high": "Fixed-High (B)"}
SYNTHETIC_NOTE = "Synthetic Users (multi-turn loop, N=10). No real participants."


def load_summary() -> dict:
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_all_sessions() -> dict[str, dict[str, list]]:
    """Load all session records: {condition: {persona_id: [sessions]}}"""
    results: dict[str, dict[str, list]] = {}
    for condition in CONDITIONS:
        cond_dir = DATA_PROCESSED_EXT / "results" / f"condition_{condition}"
        results[condition] = {}
        for session_file in sorted(cond_dir.glob("P*_sessions.json")):
            pid = session_file.stem.replace("_sessions", "")
            with open(session_file, encoding="utf-8") as f:
                sessions = json.load(f)
            results[condition][pid] = sessions
    return results


def safe_mean(values: list) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


# ── Figure 1: delta-SRR bar chart ─────────────────────────────────────────────

def plot_delta_srr_bars(summary: dict, figures_dir: Path) -> Path:
    """
    Bar chart: delta-SRR per persona + aggregate mean, A vs B side-by-side.
    """
    personas_data = summary["personas"]
    agg = summary["aggregate"]

    persona_ids = list(personas_data.keys())
    labels = [personas_data[p]["label"][:6] + f"\n({p})" for p in persona_ids]

    delta_a = [personas_data[p]["condition_adaptive_fading"]["delta_srr"] for p in persona_ids]
    delta_b = [personas_data[p]["condition_fixed_high"]["delta_srr"] for p in persona_ids]

    mean_a = agg.get("mean_delta_srr_adaptive_fading")
    mean_b = agg.get("mean_delta_srr_fixed_high")

    all_labels = labels + ["Mean"]
    all_a = delta_a + [mean_a]
    all_b = delta_b + [mean_b]

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    bars_a = ax.bar(
        x - width / 2,
        [v if v is not None else 0.0 for v in all_a],
        width,
        label=CONDITION_LABELS["adaptive-fading"],
        color=COLOR_A,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7,
    )
    bars_b = ax.bar(
        x + width / 2,
        [v if v is not None else 0.0 for v in all_b],
        width,
        label=CONDITION_LABELS["fixed-high"],
        color=COLOR_B,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7,
    )

    # Mark the "Mean" bar group with a bolder edge
    for bar in bars_a[-1:]:
        bar.set_edgecolor("black")
        bar.set_linewidth(2.0)
    for bar in bars_b[-1:]:
        bar.set_edgecolor("black")
        bar.set_linewidth(2.0)

    # Value labels on bars
    for i, (a, b) in enumerate(zip(all_a, all_b)):
        if a is not None:
            ax.text(x[i] - width / 2, a + (0.005 if a >= 0 else -0.015),
                    f"{a:+.3f}", ha="center", va="bottom" if a >= 0 else "top",
                    fontsize=8)
        if b is not None:
            ax.text(x[i] + width / 2, b + (0.005 if b >= 0 else -0.015),
                    f"{b:+.3f}", ha="center", va="bottom" if b >= 0 else "top",
                    fontsize=8)

    # Vertical separator before "Mean"
    ax.axvline(x=len(persona_ids) - 0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylabel("delta-SRR (sessions 8–14 minus sessions 1–7)", fontsize=11)
    ax.set_title(
        "Extended Study 1 — delta-SRR by Persona and Condition\n"
        f"(Multi-Turn Loop, {SYNTHETIC_NOTE})",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right")

    direction_n = agg["n_personas_A_higher_delta_srr"]
    direction_tot = agg["n_personas_valid_delta"]
    supported = agg["hypothesis_supported"]
    ax.text(
        0.02, 0.97,
        f"A > B: {direction_n}/{direction_tot} personas\n"
        f"Hypothesis: {'SUPPORTED' if supported else 'NOT SUPPORTED'}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    out_path = figures_dir / "ext_study1_delta_srr_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")
    return out_path


# ── Figure 2: SRR time series (averaged over all personas) ───────────────────

def plot_srr_time_series(all_sessions: dict, figures_dir: Path) -> Path:
    """
    Line plot: Mean SRR per session (averaged over 10 personas), A vs B.
    Includes NaN handling (sessions without negatives are excluded per-persona).
    """
    srr_a_per_session = []
    srr_b_per_session = []

    for s in range(1, N_SESSIONS + 1):
        vals_a, vals_b = [], []
        for pid, sessions in all_sessions.get("adaptive-fading", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("spontaneous_rate") is not None:
                vals_a.append(match["spontaneous_rate"])
        for pid, sessions in all_sessions.get("fixed-high", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("spontaneous_rate") is not None:
                vals_b.append(match["spontaneous_rate"])
        srr_a_per_session.append(safe_mean(vals_a) if vals_a else float("nan"))
        srr_b_per_session.append(safe_mean(vals_b) if vals_b else float("nan"))

    sessions_x = list(range(1, N_SESSIONS + 1))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(sessions_x, srr_a_per_session, color=COLOR_A, linewidth=2.2, marker="o",
            markersize=6, label=CONDITION_LABELS["adaptive-fading"])
    ax.plot(sessions_x, srr_b_per_session, color=COLOR_B, linewidth=2.2, marker="s",
            markersize=6, label=CONDITION_LABELS["fixed-high"], linestyle="--")

    # Shade early vs late windows
    ax.axvspan(1, 7, alpha=0.08, color="gray", label="Early (S1-7)")
    ax.axvspan(8, 14, alpha=0.08, color="blue", label="Late (S8-14)")
    ax.axvline(x=7.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Mean SRR (across 10 personas)", fontsize=12)
    ax.set_title(
        "Extended Study 1 — SRR Time Series by Condition\n"
        f"({SYNTHETIC_NOTE})",
        fontsize=12,
    )
    ax.set_xticks(sessions_x)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "ext_study1_srr_time_series.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")
    return out_path


# ── Figure 3: Latent skill trajectory ────────────────────────────────────────

def plot_latent_skill_trajectory(all_sessions: dict, figures_dir: Path) -> Path:
    """
    Line plot: Mean latent_skill per session, A vs B.
    """
    latent_a_per_session = []
    latent_b_per_session = []

    for s in range(1, N_SESSIONS + 1):
        vals_a, vals_b = [], []
        for pid, sessions in all_sessions.get("adaptive-fading", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("latent_skill_after") is not None:
                vals_a.append(match["latent_skill_after"])
        for pid, sessions in all_sessions.get("fixed-high", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("latent_skill_after") is not None:
                vals_b.append(match["latent_skill_after"])
        latent_a_per_session.append(safe_mean(vals_a))
        latent_b_per_session.append(safe_mean(vals_b))

    sessions_x = list(range(1, N_SESSIONS + 1))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(sessions_x, latent_a_per_session, color=COLOR_A, linewidth=2.2, marker="o",
            markersize=6, label=CONDITION_LABELS["adaptive-fading"])
    ax.plot(sessions_x, latent_b_per_session, color=COLOR_B, linewidth=2.2, marker="s",
            markersize=6, label=CONDITION_LABELS["fixed-high"], linestyle="--")

    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Mean Latent Skill", fontsize=12)
    ax.set_title(
        "Extended Study 1 — Latent Skill Trajectory by Condition\n"
        f"({SYNTHETIC_NOTE})",
        fontsize=12,
    )
    ax.set_xticks(sessions_x)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "ext_study1_latent_skill_trajectory.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")
    return out_path


# ── Figure 4: is_echo rate per session ───────────────────────────────────────

def plot_is_echo_rate(all_sessions: dict, figures_dir: Path) -> Path:
    """
    Bar plot: is_echo rate per session per condition.
    Shows how often the user's reframe attempt was detected as echoing the AI.
    """
    echo_a_per_session = []
    echo_b_per_session = []

    for s in range(1, N_SESSIONS + 1):
        # For sessions with negatives (reframing attempted), check is_echo
        vals_a, vals_b = [], []
        for pid, sessions in all_sessions.get("adaptive-fading", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("neg_count_entry", 0) > 0:
                vals_a.append(1.0 if match.get("is_echo", False) else 0.0)
        for pid, sessions in all_sessions.get("fixed-high", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("neg_count_entry", 0) > 0:
                vals_b.append(1.0 if match.get("is_echo", False) else 0.0)
        echo_a_per_session.append(safe_mean(vals_a))
        echo_b_per_session.append(safe_mean(vals_b))

    sessions_x = np.arange(1, N_SESSIONS + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(sessions_x - width / 2,
           [v if not math.isnan(v) else 0.0 for v in echo_a_per_session],
           width, color=COLOR_A, alpha=0.8, label=CONDITION_LABELS["adaptive-fading"],
           edgecolor="black", linewidth=0.5)
    ax.bar(sessions_x + width / 2,
           [v if not math.isnan(v) else 0.0 for v in echo_b_per_session],
           width, color=COLOR_B, alpha=0.8, label=CONDITION_LABELS["fixed-high"],
           edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Echo Rate (is_echo=True among sessions with negatives)", fontsize=11)
    ax.set_title(
        "Extended Study 1 — AI Echo Rate per Session by Condition\n"
        f"({SYNTHETIC_NOTE})\n"
        "is_echo=True: user's reframe echoes AI's modeled reframe (F3 violation)",
        fontsize=11,
    )
    ax.set_xticks(sessions_x)
    ax.set_ylim(-0.02, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.01, 0.97,
            "Note: High echo rate in fixed-high early sessions is expected\n"
            "(AI always models a full reframe → user tends to echo it).",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path = figures_dir / "ext_study1_echo_rate.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")
    return out_path


# ── Figure 5: p_attempt trajectory ───────────────────────────────────────────

def plot_p_attempt_trajectory(all_sessions: dict, figures_dir: Path) -> Path:
    """
    Line plot: Mean p_attempt per session, A vs B.
    Illustrates how adaptive-fading increases practice opportunity over time.
    """
    p_a_per_session, p_b_per_session = [], []

    for s in range(1, N_SESSIONS + 1):
        vals_a, vals_b = [], []
        for pid, sessions in all_sessions.get("adaptive-fading", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("p_attempt_gt") is not None:
                vals_a.append(match["p_attempt_gt"])
        for pid, sessions in all_sessions.get("fixed-high", {}).items():
            match = next((x for x in sessions if x["session"] == s), None)
            if match and match.get("p_attempt_gt") is not None:
                vals_b.append(match["p_attempt_gt"])
        p_a_per_session.append(safe_mean(vals_a))
        p_b_per_session.append(safe_mean(vals_b))

    sessions_x = list(range(1, N_SESSIONS + 1))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(sessions_x, p_a_per_session, color=COLOR_A, linewidth=2.2, marker="o",
            markersize=6, label=CONDITION_LABELS["adaptive-fading"])
    ax.plot(sessions_x, p_b_per_session, color=COLOR_B, linewidth=2.2, marker="s",
            markersize=6, label=CONDITION_LABELS["fixed-high"], linestyle="--")

    ax.set_xlabel("Session", fontsize=12)
    ax.set_ylabel("Mean p_attempt (practice opportunity)", fontsize=12)
    ax.set_title(
        "Extended Study 1 — Practice Opportunity (p_attempt) by Condition\n"
        f"({SYNTHETIC_NOTE})\n"
        "Causal pathway: scaffold_level → p_attempt → did_attempt → latent_skill",
        fontsize=11,
    )
    ax.set_xticks(sessions_x)
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = figures_dir / "ext_study1_p_attempt_trajectory.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")
    return out_path


# ── Scaffold level distribution ───────────────────────────────────────────────

def compute_scaffold_distribution(all_sessions: dict) -> dict:
    """Count scaffold level usage per condition."""
    dist: dict = {}
    for condition in CONDITIONS:
        counts = {"high": 0, "mid": 0, "low": 0}
        for pid, sessions in all_sessions.get(condition, {}).items():
            for s in sessions:
                sl = s.get("scaffold_level", "high")
                counts[sl] = counts.get(sl, 0) + 1
        total = sum(counts.values())
        dist[condition] = {
            "counts": counts,
            "total": total,
            "rates": {k: round(v / total, 4) if total > 0 else 0 for k, v in counts.items()},
        }
    return dist


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("GratiFlow Extended Study 1 — Analysis & Visualization")
    print("Synthetic Users (multi-turn loop, N=10). No real participants.")
    print("=" * 70)

    # Check data exists
    if not SUMMARY_PATH.exists():
        print(f"[ERROR] Summary not found: {SUMMARY_PATH}")
        print("Run generate_and_run_ext_study1.py first.")
        return

    summary = load_summary()
    all_sessions = load_all_sessions()

    if not any(all_sessions.get(c) for c in CONDITIONS):
        print("[ERROR] No session data found. Run generate_and_run_ext_study1.py first.")
        return

    n_personas_found = max(
        len(all_sessions.get(c, {})) for c in CONDITIONS
    )
    print(f"Loaded sessions for {n_personas_found} personas.")

    # ── Generate figures ──────────────────────────────────────────────────
    fig_paths = []
    fig_paths.append(plot_delta_srr_bars(summary, FIGURES_DIR))
    fig_paths.append(plot_srr_time_series(all_sessions, FIGURES_DIR))
    fig_paths.append(plot_latent_skill_trajectory(all_sessions, FIGURES_DIR))
    fig_paths.append(plot_is_echo_rate(all_sessions, FIGURES_DIR))
    fig_paths.append(plot_p_attempt_trajectory(all_sessions, FIGURES_DIR))

    # ── Scaffold distribution ─────────────────────────────────────────────
    scaffold_dist = compute_scaffold_distribution(all_sessions)
    scaffold_path = DATA_PROCESSED_EXT / "results" / "scaffold_distribution_ext_study1.json"
    with open(scaffold_path, "w", encoding="utf-8") as f:
        json.dump(scaffold_dist, f, ensure_ascii=False, indent=2)
    print(f"\nScaffold distribution saved: {scaffold_path}")
    for cond, data in scaffold_dist.items():
        print(f"  {cond}: {data['rates']}")

    # ── Key results summary ───────────────────────────────────────────────
    agg = summary["aggregate"]
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY (ext_study1)")
    print("=" * 70)
    a_val = agg.get("mean_delta_srr_adaptive_fading")
    b_val = agg.get("mean_delta_srr_fixed_high")
    adv_val = agg.get("mean_delta_srr_advantage_A_over_B")
    print(f"Mean delta-SRR (Adaptive-Fading): {f'{a_val:+.4f}' if a_val is not None else 'null'}")
    print(f"Mean delta-SRR (Fixed-High):       {f'{b_val:+.4f}' if b_val is not None else 'null'}")
    print(f"Advantage A over B:                {f'{adv_val:+.4f}' if adv_val is not None else 'null'}")
    print(f"Personas A > B:                    {agg['n_personas_A_higher_delta_srr']}/{agg['n_personas_valid_delta']}")
    dt = agg['direction_threshold']
    tot = agg['n_personas_total']
    print(f"Direction consistent (>= {dt}/{tot}): {agg['direction_consistent_pre_committed']}")
    print(f"Hypothesis supported:              {agg['hypothesis_supported']}")
    print(f"\n{agg['interpretation']}")

    print("\nPer-persona delta-SRR:")
    for pid, pdata in summary["personas"].items():
        a = pdata["condition_adaptive_fading"]["delta_srr"]
        b = pdata["condition_fixed_high"]["delta_srr"]
        adv = pdata.get("delta_srr_advantage_A_over_B")
        label = pdata["label"]
        a_str = f"{a:+.4f}" if a is not None else "  null"
        b_str = f"{b:+.4f}" if b is not None else "  null"
        adv_str = f"{adv:+.4f}" if adv is not None else "  null"
        print(f"  {pid} ({label}): A={a_str}, B={b_str}, A-B={adv_str}")

    print("\nFigures saved to:")
    for p in fig_paths:
        print(f"  {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
