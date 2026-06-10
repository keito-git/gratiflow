"""
analyze_rubric_distribution_ext_study1_fixed.py
=================================================
GratiFlow Extended Study 1 (fixed) — SRR Rubric Code Distribution Analysis

Aggregates R1/R2/R3 (inclusion) and F1/F2/F3/F4/F5 (exclusion) rubric code
occurrences across all sessions (10 personas x 2 conditions x 14 sessions = 280).

Measurement approach:
  - R1+R2+R3 met (all three inclusion criteria):
      Directly from structured field: spontaneous_reframe == True
  - F3 (AI Echo):
      Directly from structured field: is_echo == True
  - F1/F2/F4/F5 (other exclusion criteria):
      Inferred from srr_reasoning text via keyword pattern matching
      (approximate; treated as indicative, not ground-truth counts)

NOTE: All data is from synthetic users. "Synthetic" label included on all figures.

Outputs:
  - evaluation/ext_rubric_distribution.png + .pdf
  - paper/en/figures/ext_rubric_distribution.png + .pdf
  - data/processed/ext_rubric_distribution.csv
  - data/processed/ext_rubric_distribution.json

Author: team member (experiment lead, the research team)
Date: 2026-06-05
"""

import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(Path(__file__).resolve().parent.parent)
DATA_DIR = BASE_DIR / "data" / "processed" / "ext_study1_fixed" / "results"
EVAL_DIR = BASE_DIR / "evaluation"
PAPER_FIG_DIR = BASE_DIR / "paper" / "en" / "figures"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CONDITIONS = ["adaptive-fading", "fixed-high"]
CONDITION_LABELS = {
    "adaptive-fading": "Adaptive-Fading (A)",
    "fixed-high": "Fixed-High (B)",
}
N_SESSIONS = 14
N_PERSONAS = 10
SYNTHETIC_NOTE = "Synthetic Users (multi-turn loop, N=10). No real participants."

# Colorblind-friendly palette (Wong 2011)
COLOR_A = "#0072B2"      # blue: adaptive-fading
COLOR_B = "#E69F00"      # orange: fixed-high
COLOR_R = "#009E73"      # green: inclusion criteria (R-codes)
COLOR_F = "#D55E00"      # vermillion: exclusion criteria (F-codes)
COLOR_A_LIGHT = "#56B4E9"
COLOR_B_LIGHT = "#F0E442"

FIGURE_DPI = 300

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       12,        # unified to >= 12 pt (2026-06-06)
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi":      FIGURE_DPI,
    "savefig.dpi":     FIGURE_DPI,
    "savefig.bbox":    "tight",
})

# ── Keyword patterns for F1/F2/F4/F5 from srr_reasoning text ──────────────────
# F3 is taken directly from is_echo field (more reliable).
# These patterns approximate the LLM judge's reasoning; treat as indicative.
F_TEXT_PATTERNS = {
    "F1": [
        r"F1",
        r"simply describes a positive",
        r"without connecting.*negative",
        r"否定的.*なく.*肯定的",
        r"ポジティブな出来事のみ",
        r"positive event without connecting",
    ],
    "F2": [
        r"F2",
        r"vague optimism",
        r"coping platitude",
        r"漠然",
        r"曖昧",
        r"generic reframe",
        r"platitude",
        r"superficial",
    ],
    "F4": [
        r"F4",
        r"only describes the negative",
        r"without any positive reinterpretation",
        r"no positive reinterpretation",
        r"否定的.*述べる.*のみ",
        r"positive reinterpretation.*not present",
        r"ポジティブな再解釈は.*な[いく]",
        r"再解釈は見られません",
        r"positive.*reframe.*absent",
    ],
    "F5": [
        r"F5",
        r"different event",
        r"separate positive",
        r"DIFFERENT event",
        r"別の出来事",
        r"別の.*ポジティブ",
        r"positive.*about a different",
        r"positive.*event.*separate",
    ],
}


def load_all_sessions() -> dict[str, dict[str, list]]:
    """Load all session records from ext_study1_fixed. Returns {condition: {pid: [sessions]}}."""
    data: dict[str, dict[str, list]] = {}
    for cond in CONDITIONS:
        cond_dir = DATA_DIR / f"condition_{cond}"
        data[cond] = {}
        for jf in sorted(cond_dir.glob("P*_sessions.json")):
            pid = jf.stem.replace("_sessions", "")
            with open(jf, encoding="utf-8") as f:
                data[cond][pid] = json.load(f)
    return data


def detect_f_text(reasoning: str, flag: str) -> bool:
    """Return True if any keyword pattern for flag is found in reasoning text."""
    for pattern in F_TEXT_PATTERNS[flag]:
        if re.search(pattern, reasoning, re.IGNORECASE):
            return True
    return False


def aggregate_rubric_counts(
    all_sessions: dict[str, dict[str, list]]
) -> dict:
    """
    Aggregate rubric code occurrences per condition and overall.

    Rubric mapping:
      R1+R2+R3 met (all inclusion criteria): spontaneous_reframe == True
      F3 (AI echo):                           is_echo == True  [structured field]
      F1, F2, F4, F5:                         keyword match in srr_reasoning  [text-inferred]

    For sessions where neg_count_entry == 0, reframing is not applicable;
    these are still counted in the denominator but excluded from F3 echo denominator.

    Returns nested dict with per-condition and overall counts.
    """
    results = {}
    for cond in CONDITIONS:
        sessions_flat = []
        for pid, sess_list in all_sessions[cond].items():
            sessions_flat.extend(sess_list)

        total = len(sessions_flat)
        # Sessions where reframing was attempted (neg events present)
        neg_sessions = [s for s in sessions_flat if s.get("neg_count_entry", 0) > 0]
        n_neg = len(neg_sessions)

        # --- Inclusion criteria ---
        # R1+R2+R3 all met: operationalized as spontaneous_reframe=True
        r_all_met = sum(1 for s in sessions_flat if s.get("spontaneous_reframe", False))

        # --- Exclusion criteria ---
        # F3: AI echo — directly from is_echo field (sessions with neg events)
        f3_count = sum(1 for s in neg_sessions if s.get("is_echo", False))

        # F1, F2, F4, F5: inferred from srr_reasoning text
        f_text_counts = {flag: 0 for flag in ["F1", "F2", "F4", "F5"]}
        n_with_reasoning = 0
        for s in sessions_flat:
            reasoning = s.get("srr_reasoning", "")
            if not reasoning:
                continue
            n_with_reasoning += 1
            for flag in ["F1", "F2", "F4", "F5"]:
                if detect_f_text(reasoning, flag):
                    f_text_counts[flag] += 1

        results[cond] = {
            "total_sessions": total,
            "n_neg_sessions": n_neg,
            "n_sessions_with_reasoning": n_with_reasoning,
            # Inclusion
            "R_all_met_count": r_all_met,
            "R_all_met_rate_of_total": round(r_all_met / total, 4) if total > 0 else 0,
            # Exclusion — structured
            "F3_echo_count": f3_count,
            "F3_echo_rate_of_neg": round(f3_count / n_neg, 4) if n_neg > 0 else 0,
            # Exclusion — text-inferred
            "F1_text_count": f_text_counts["F1"],
            "F1_text_rate": round(f_text_counts["F1"] / n_with_reasoning, 4) if n_with_reasoning > 0 else 0,
            "F2_text_count": f_text_counts["F2"],
            "F2_text_rate": round(f_text_counts["F2"] / n_with_reasoning, 4) if n_with_reasoning > 0 else 0,
            "F4_text_count": f_text_counts["F4"],
            "F4_text_rate": round(f_text_counts["F4"] / n_with_reasoning, 4) if n_with_reasoning > 0 else 0,
            "F5_text_count": f_text_counts["F5"],
            "F5_text_rate": round(f_text_counts["F5"] / n_with_reasoning, 4) if n_with_reasoning > 0 else 0,
        }

    # --- Overall (pooled) ---
    all_flat = []
    for cond in CONDITIONS:
        for pid, sess_list in all_sessions[cond].items():
            all_flat.extend(sess_list)

    total_all = len(all_flat)
    neg_all = [s for s in all_flat if s.get("neg_count_entry", 0) > 0]
    n_neg_all = len(neg_all)
    n_reasoning_all = sum(1 for s in all_flat if s.get("srr_reasoning", ""))

    r_met_all = sum(1 for s in all_flat if s.get("spontaneous_reframe", False))
    f3_all = sum(1 for s in neg_all if s.get("is_echo", False))

    f_text_all = {flag: 0 for flag in ["F1", "F2", "F4", "F5"]}
    for s in all_flat:
        reasoning = s.get("srr_reasoning", "")
        if not reasoning:
            continue
        for flag in ["F1", "F2", "F4", "F5"]:
            if detect_f_text(reasoning, flag):
                f_text_all[flag] += 1

    results["overall"] = {
        "total_sessions": total_all,
        "n_neg_sessions": n_neg_all,
        "n_sessions_with_reasoning": n_reasoning_all,
        "R_all_met_count": r_met_all,
        "R_all_met_rate_of_total": round(r_met_all / total_all, 4) if total_all > 0 else 0,
        "F3_echo_count": f3_all,
        "F3_echo_rate_of_neg": round(f3_all / n_neg_all, 4) if n_neg_all > 0 else 0,
        "F1_text_count": f_text_all["F1"],
        "F1_text_rate": round(f_text_all["F1"] / n_reasoning_all, 4) if n_reasoning_all > 0 else 0,
        "F2_text_count": f_text_all["F2"],
        "F2_text_rate": round(f_text_all["F2"] / n_reasoning_all, 4) if n_reasoning_all > 0 else 0,
        "F4_text_count": f_text_all["F4"],
        "F4_text_rate": round(f_text_all["F4"] / n_reasoning_all, 4) if n_reasoning_all > 0 else 0,
        "F5_text_count": f_text_all["F5"],
        "F5_text_rate": round(f_text_all["F5"] / n_reasoning_all, 4) if n_reasoning_all > 0 else 0,
    }

    return results


def save_csv(rubric_counts: dict, out_path: Path) -> None:
    """Save rubric distribution to CSV."""
    rows = []
    for cond in CONDITIONS + ["overall"]:
        d = rubric_counts[cond]
        total = d["total_sessions"]
        n_neg = d["n_neg_sessions"]
        n_reas = d["n_sessions_with_reasoning"]
        label = CONDITION_LABELS.get(cond, "Overall (pooled)")
        rows.append({
            "condition": label,
            "total_sessions": total,
            "n_neg_sessions": n_neg,
            "n_reasoning_sessions": n_reas,
            # Inclusion
            "R_all_met_count": d["R_all_met_count"],
            "R_all_met_pct_of_total": round(d["R_all_met_rate_of_total"] * 100, 1),
            # F3 — structured
            "F3_echo_count_of_neg": d["F3_echo_count"],
            "F3_echo_pct_of_neg": round(d["F3_echo_rate_of_neg"] * 100, 1),
            # F1/F2/F4/F5 — text-inferred
            "F1_text_count": d["F1_text_count"],
            "F1_text_pct_of_reasoning": round(d["F1_text_rate"] * 100, 1),
            "F2_text_count": d["F2_text_count"],
            "F2_text_pct_of_reasoning": round(d["F2_text_rate"] * 100, 1),
            "F4_text_count": d["F4_text_count"],
            "F4_text_pct_of_reasoning": round(d["F4_text_rate"] * 100, 1),
            "F5_text_count": d["F5_text_count"],
            "F5_text_pct_of_reasoning": round(d["F5_text_rate"] * 100, 1),
        })
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {out_path}")


def save_json(rubric_counts: dict, out_path: Path) -> None:
    """Save rubric distribution to JSON."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rubric_counts, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {out_path}")


def plot_rubric_distribution(rubric_counts: dict, out_stem: str) -> None:
    """
    Create a grouped bar chart showing rubric code frequencies by condition.

    Panel A (left): R_all_met, F3_echo — based on structured fields (reliable)
    Panel B (right): F1, F2, F4, F5 — based on srr_reasoning text (approximate)
    """
    cond_labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    colors = [COLOR_A, COLOR_B]

    # Data for Panel A — structured fields
    r_met_rates = [rubric_counts[c]["R_all_met_rate_of_total"] * 100 for c in CONDITIONS]
    r_met_counts = [rubric_counts[c]["R_all_met_count"] for c in CONDITIONS]
    f3_rates = [rubric_counts[c]["F3_echo_rate_of_neg"] * 100 for c in CONDITIONS]
    f3_counts = [rubric_counts[c]["F3_echo_count"] for c in CONDITIONS]
    f3_denoms = [rubric_counts[c]["n_neg_sessions"] for c in CONDITIONS]

    # Data for Panel B — text-inferred
    text_flags = ["F1", "F2", "F4", "F5"]
    flag_labels = ["F1\n(no neg.)", "F2\n(vague)", "F4\n(no reframe)", "F5\n(diff. event)"]
    text_rates = {
        flag: [rubric_counts[c][f"{flag}_text_rate"] * 100 for c in CONDITIONS]
        for flag in text_flags
    }
    text_counts = {
        flag: [rubric_counts[c][f"{flag}_text_count"] for c in CONDITIONS]
        for flag in text_flags
    }
    n_reasoning = [rubric_counts[c]["n_sessions_with_reasoning"] for c in CONDITIONS]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6),
                                      gridspec_kw={"width_ratios": [2, 4]})

    # ── Panel A: R_all_met and F3 (structured, reliable) ──────────────────────
    x_a = np.arange(2)  # R_all_met, F3_echo
    width_a = 0.3
    metric_labels_a = ["R1+R2+R3\nall met", "F3\n(AI Echo)"]

    for i, cond in enumerate(CONDITIONS):
        vals_a = [r_met_rates[i], f3_rates[i]]
        counts_a = [r_met_counts[i], f3_counts[i]]
        bars = ax_a.bar(
            x_a + i * width_a - width_a / 2, vals_a,
            width_a, color=colors[i], alpha=0.85,
            label=CONDITION_LABELS[cond], edgecolor="black", linewidth=0.7
        )
        for bar, cnt in zip(bars, counts_a):
            h = bar.get_height()
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"n={cnt}\n({h:.1f}%)",
                ha="center", va="bottom", fontsize=8
            )

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(metric_labels_a, fontsize=10)
    ax_a.set_ylabel("Rate (%)", fontsize=11)
    ax_a.set_ylim(0, 80)
    ax_a.set_title(
        "Structured-field metrics\n(reliable)",
        fontsize=11
    )
    ax_a.legend(fontsize=9, loc="upper right")
    ax_a.grid(axis="y", alpha=0.3, linestyle="--")
    # Footnote: denominator clarification
    ax_a.text(
        0.02, 0.02,
        "R1+R2+R3: % of all sessions (n=140 per condition)\n"
        "F3: % of sessions with neg. events "
        f"(A:{f3_denoms[0]}, B:{f3_denoms[1]})",
        transform=ax_a.transAxes, fontsize=7, va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8)
    )

    # ── Panel B: F1/F2/F4/F5 (text-inferred, approximate) ────────────────────
    x_b = np.arange(len(text_flags))
    width_b = 0.3

    for i, cond in enumerate(CONDITIONS):
        vals_b = [text_rates[flag][i] for flag in text_flags]
        cnts_b = [text_counts[flag][i] for flag in text_flags]
        bars = ax_b.bar(
            x_b + i * width_b - width_b / 2, vals_b,
            width_b, color=colors[i], alpha=0.85,
            label=CONDITION_LABELS[cond], edgecolor="black", linewidth=0.7
        )
        for bar, cnt in zip(bars, cnts_b):
            h = bar.get_height()
            if h > 0.3:
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.4,
                    f"n={cnt}\n({h:.1f}%)",
                    ha="center", va="bottom", fontsize=8
                )
            elif cnt > 0:
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.4,
                    f"n={cnt}",
                    ha="center", va="bottom", fontsize=7
                )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(flag_labels, fontsize=10)
    ax_b.set_ylabel("Rate (% of sessions with srr_reasoning)", fontsize=10)
    ax_b.set_ylim(0, 25)
    ax_b.set_title(
        "Text-inferred exclusion flags (F1/F2/F4/F5)\n(approximate; keyword pattern from srr_reasoning)",
        fontsize=11
    )
    ax_b.legend(fontsize=9, loc="upper right")
    ax_b.grid(axis="y", alpha=0.3, linestyle="--")
    ax_b.text(
        0.02, 0.02,
        f"Denominator: sessions with srr_reasoning text "
        f"(A:{n_reasoning[0]}, B:{n_reasoning[1]})\n"
        "Note: F1–F5 labels from rubric prompt; text-match may under/over-count.",
        transform=ax_b.transAxes, fontsize=7, va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8)
    )

    # ── Figure-level labels ────────────────────────────────────────────────────
    overall = rubric_counts["overall"]
    fig.suptitle(
        "GratiFlow Extended Study 1 — SRR Rubric Code Distribution\n"
        f"N=280 sessions ({SYNTHETIC_NOTE})",
        fontsize=12, y=1.01
    )
    fig.text(
        0.99, 0.00,
        "SYNTHETIC — No real participants. "
        "F3 from is_echo field; F1/F2/F4/F5 text-inferred.",
        ha="right", va="bottom", fontsize=7, color="gray"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save PNG + PDF to evaluation/
    for suffix in [".png", ".pdf"]:
        out_path = EVAL_DIR / f"{out_stem}{suffix}"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Figure saved: {out_path}")

    # Save copies to paper/en/figures/
    for suffix in [".png", ".pdf"]:
        out_path = PAPER_FIG_DIR / f"{out_stem}{suffix}"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Figure saved: {out_path}")

    plt.close()


def print_markdown_table(rubric_counts: dict) -> str:
    """Return a Markdown table string for use in the paper §4 / reliability section."""
    lines = []
    lines.append("## SRR Rubric Code Distribution (Extended Study 1, Synthetic)")
    lines.append("")
    lines.append("**Table: SRR rubric code occurrence counts and rates**")
    lines.append("")
    lines.append("| Code | Description | Adaptive-Fading (A) | Fixed-High (B) | Overall |")
    lines.append("|------|-------------|---------------------|----------------|---------|")

    def fmt_cell(count, rate, note=""):
        return f"{count} ({rate*100:.1f}%){note}"

    d_a = rubric_counts["adaptive-fading"]
    d_b = rubric_counts["fixed-high"]
    d_o = rubric_counts["overall"]

    lines.append(
        f"| **R1+R2+R3 all met** | Spontaneous reframe (all inclusion criteria) | "
        f"{fmt_cell(d_a['R_all_met_count'], d_a['R_all_met_rate_of_total'])} | "
        f"{fmt_cell(d_b['R_all_met_count'], d_b['R_all_met_rate_of_total'])} | "
        f"{fmt_cell(d_o['R_all_met_count'], d_o['R_all_met_rate_of_total'])} |"
    )
    lines.append(
        f"| **F3 (AI Echo)** | is_echo=True [structured field] | "
        f"{d_a['F3_echo_count']} / {d_a['n_neg_sessions']} neg. sessions ({d_a['F3_echo_rate_of_neg']*100:.1f}%) | "
        f"{d_b['F3_echo_count']} / {d_b['n_neg_sessions']} neg. sessions ({d_b['F3_echo_rate_of_neg']*100:.1f}%) | "
        f"{d_o['F3_echo_count']} / {d_o['n_neg_sessions']} ({d_o['F3_echo_rate_of_neg']*100:.1f}%) |"
    )
    for flag, desc in [
        ("F1", "No negative event (text-inferred)"),
        ("F2", "Vague optimism / platitude (text-inferred)"),
        ("F4", "Negative only; no reinterpretation (text-inferred)"),
        ("F5", "Positive about different event (text-inferred)"),
    ]:
        k_cnt = f"{flag}_text_count"
        k_rate = f"{flag}_text_rate"
        lines.append(
            f"| **{flag}** | {desc} | "
            f"{d_a[k_cnt]} ({d_a[k_rate]*100:.1f}%) | "
            f"{d_b[k_cnt]} ({d_b[k_rate]*100:.1f}%) | "
            f"{d_o[k_cnt]} ({d_o[k_rate]*100:.1f}%) |"
        )

    lines.append("")
    lines.append("**Notes:**")
    lines.append(
        "- Denominators: R1+R2+R3 and text-inferred flags use total sessions per condition "
        f"(A: {d_a['total_sessions']}, B: {d_b['total_sessions']})."
    )
    lines.append(
        f"- F3 denominator: sessions with negative events present "
        f"(A: {d_a['n_neg_sessions']}, B: {d_b['n_neg_sessions']})."
    )
    lines.append(
        f"- F1/F2/F4/F5 rates are computed over sessions with non-empty srr_reasoning "
        f"(A: {d_a['n_sessions_with_reasoning']}, B: {d_b['n_sessions_with_reasoning']})."
    )
    lines.append("- **All data are from synthetic users. No real participants.**")
    lines.append(
        "- F1/F2/F4/F5 counts are approximate (keyword pattern matching on LLM-generated reasoning)."
    )

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("GratiFlow ext_study1_fixed — SRR Rubric Code Distribution Analysis")
    print(f"  {SYNTHETIC_NOTE}")
    print("=" * 70)

    # Load data
    print("\nLoading session data...")
    all_sessions = load_all_sessions()
    total = sum(len(v) for cond_dict in all_sessions.values() for v in cond_dict.values())
    print(f"  Loaded {total} sessions total ({len(all_sessions['adaptive-fading'])} personas x 2 conditions x {N_SESSIONS} sessions)")

    # Aggregate
    print("\nAggregating rubric code counts...")
    rubric_counts = aggregate_rubric_counts(all_sessions)

    # Print summary
    print("\n--- Rubric distribution summary ---")
    for cond in CONDITIONS + ["overall"]:
        d = rubric_counts[cond]
        label = CONDITION_LABELS.get(cond, "Overall")
        print(f"\n  [{label}]")
        print(f"    Total sessions:          {d['total_sessions']}")
        print(f"    Neg-event sessions:      {d['n_neg_sessions']} ({d['n_neg_sessions']/d['total_sessions']*100:.1f}%)")
        print(f"    Sessions w/ reasoning:   {d['n_sessions_with_reasoning']}")
        print(f"    R1+R2+R3 met:            {d['R_all_met_count']} ({d['R_all_met_rate_of_total']*100:.1f}% of total)")
        print(f"    F3 echo (is_echo=True):  {d['F3_echo_count']} / {d['n_neg_sessions']} neg sessions ({d['F3_echo_rate_of_neg']*100:.1f}%)")
        for flag in ["F1", "F2", "F4", "F5"]:
            print(f"    {flag} text-inferred:     {d[f'{flag}_text_count']} ({d[f'{flag}_text_rate']*100:.1f}% of reasoning sessions)")

    # Save CSV and JSON
    csv_path = PROCESSED_DIR / "ext_rubric_distribution.csv"
    json_path = PROCESSED_DIR / "ext_rubric_distribution.json"
    save_csv(rubric_counts, csv_path)
    save_json(rubric_counts, json_path)

    # Generate figure
    print("\nGenerating rubric distribution figure...")
    plot_rubric_distribution(rubric_counts, out_stem="ext_rubric_distribution")

    # Print Markdown table
    print("\n" + "=" * 70)
    md_table = print_markdown_table(rubric_counts)
    print(md_table)
    print("=" * 70)

    print(f"\nAll outputs saved:")
    print(f"  Figure (PNG+PDF): {EVAL_DIR}/ext_rubric_distribution.{{png,pdf}}")
    print(f"  Figure (copy):    {PAPER_FIG_DIR}/ext_rubric_distribution.{{png,pdf}}")
    print(f"  CSV:              {csv_path}")
    print(f"  JSON:             {json_path}")
    print(f"\n  {SYNTHETIC_NOTE}")


if __name__ == "__main__":
    main()
