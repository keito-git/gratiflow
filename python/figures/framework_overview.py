"""
GratiFlow Framework Overview Figure (Figure 1) — v2
=====================================================
Author: team member (experiment, the research team)
Date:   2026-06-05
Changes from v1:
  - Feedback arrow made thicker and colored (SKY) for visibility
  - ŝ-update text repositioned to avoid overlap with "sessions progress"
  - Durability annotation moved fully inside Phase A' region
  - Phase B background strengthened to orange-tint for contrast
  - Layer-2 heading moved up so it does not collide with skill threshold text
  - Evaluation scope note repositioned to avoid overlap with SRR curve

Color palette: Wong (2011) colorblind-friendly.
Output:
  evaluation/framework_overview.png  (300 dpi)
  evaluation/framework_overview.pdf
  paper/en/figures/framework_overview.pdf  (for pdflatex)
  evaluation/framework_overview_caption.md
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = Path(__file__).resolve().parent.parent
OUT_EVAL = os.path.join(BASE, "evaluation")
OUT_FIG  = os.path.join(BASE, "paper", "en",
           "GratiFlow__A_Scaffolding_Fading_Multi_Agent_LLM_for_Positive_Reframing_Skill_Development",
           "figures")
os.makedirs(OUT_EVAL, exist_ok=True)
os.makedirs(OUT_FIG,  exist_ok=True)

# ---------------------------------------------------------------------------
# Wong palette
# ---------------------------------------------------------------------------
BLACK  = "#000000"
ORANGE = "#E69F00"
SKY    = "#56B4E9"
GREEN  = "#009E73"
YELLOW = "#F0E442"
BLUE   = "#0072B2"
RED    = "#D55E00"
PINK   = "#CC79A7"
LGRAY  = "#AAAAAA"
LLGRAY = "#DDDDDD"
WHITE  = "#FFFFFF"

C_USER   = SKY
C_COACH  = ORANGE
C_AFFECT = GREEN
C_SRRFB  = BLUE
C_SKILL  = RED

BG_L1 = "#EDF5FC"
BG_L2 = "#FFF4E0"
BG_L3 = "#E7F5EE"

# ---------------------------------------------------------------------------
# Figure canvas
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 14.5, 10.8
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rbox(ax, cx, cy, w, h, fc, ec=None, alpha=1.0, lw=1.3, zorder=3, r=0.20):
    """Rounded rectangle centred at (cx, cy)."""
    if ec is None:
        ec = fc
    p = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def node(ax, cx, cy, w, h, txt, fc, fs=9.5, tc=WHITE,
         bold=False, zorder=4, lw=1.3):
    """Node box with centred label."""
    rbox(ax, cx, cy, w, h, fc, ec=fc, lw=lw, zorder=zorder)
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal",
            zorder=zorder + 1)


def arr(ax, x0, y0, x1, y1, color=BLACK, lw=1.5,
        style="-|>", zorder=5, cs="arc3,rad=0.0"):
    """Arrow from (x0,y0) to (x1,y1)."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle=cs),
                zorder=zorder)


def band_bg(ax, y_bot, y_top, fc, side_label, zorder=0):
    """Layer background rectangle with rotated side label."""
    p = mpatches.FancyBboxPatch(
        (0.18, y_bot), FIG_W - 0.36, y_top - y_bot,
        boxstyle="round,pad=0,rounding_size=0.20",
        facecolor=fc, edgecolor=LGRAY, linewidth=1.0,
        alpha=0.50, zorder=zorder)
    ax.add_patch(p)
    ax.text(0.50, (y_bot + y_top) / 2, side_label,
            ha="center", va="center", fontsize=7.8,
            color="#666666", fontweight="bold",
            rotation=90, zorder=zorder + 1)

# ===========================================================================
# LAYER BACKGROUNDS
# ===========================================================================
# L1: Within-session loop   y ∈ [7.20, 10.55]
band_bg(ax, 7.20, 10.55, BG_L1, "Within-session\nConversational Loop")
# L2: Inter-session fading  y ∈ [3.95, 7.10]
band_bg(ax, 3.95, 7.10, BG_L2, "Inter-session\nAdaptive Fading")
# L3: A-B-A' design         y ∈ [0.52, 3.85]
band_bg(ax, 0.52, 3.85, BG_L3, "Internalization\nMeasurement (A-B-A')")

# ===========================================================================
# LAYER 1  —  Within-session conversational loop
# ===========================================================================
L1_Y  = 8.72     # main node row
L1_Y2 = 9.58     # annotation row above nodes
NW, NH = 2.28, 0.74

# Node x centres
Xu   = 1.60   # User (entry)
Xc   = 4.20   # Reframing-Coach
Xur  = 6.78   # User Reframe
Xa   = 9.35   # Affect-Analysis
Xsf  = 12.00  # SRR-Feedback

node(ax, Xu,  L1_Y, NW, NH, "User\n(negative event)", C_USER,  bold=True)
node(ax, Xc,  L1_Y, NW, NH, "Reframing-Coach\nAgent", C_COACH, bold=True)
node(ax, Xur, L1_Y, NW, NH, "User\nReframe Attempt",  C_USER,  bold=True)
node(ax, Xa,  L1_Y, NW, NH, "Affect-Analysis\nAgent", C_AFFECT,bold=True)
node(ax, Xsf, L1_Y, NW, NH, "SRR-Feedback\nAgent",    C_SRRFB, bold=True)

# Forward arrows
for x0, x1 in [(Xu, Xc), (Xc, Xur), (Xur, Xa), (Xa, Xsf)]:
    arr(ax, x0 + NW/2, L1_Y, x1 - NW/2, L1_Y, color=BLACK, lw=1.6)

# Scaffold annotation above Reframing-Coach
rbox(ax, Xc, L1_Y2, NW + 0.25, 0.54,
     YELLOW, ec=ORANGE, alpha=0.90, lw=1.3, zorder=5)
ax.text(Xc, L1_Y2,
        r"$\ell \in$ {Modeling, Guided, Independent}",
        ha="center", va="center", fontsize=8.0,
        color=BLACK, zorder=6)

# Rubric annotation above Affect-Analysis
rbox(ax, Xa, L1_Y2, NW + 0.10, 0.54,
     LLGRAY, ec=LGRAY, alpha=0.90, lw=1.1, zorder=5)
ax.text(Xa, L1_Y2,
        "Rubric R1–R3 / F1–F5\n(echo detection F3)",
        ha="center", va="center", fontsize=7.8,
        color=BLACK, zorder=6)

# Skill-update node (bottom strip of Layer 1)
Xs, Ys = 7.80, 7.68
node(ax, Xs, Ys, 3.35, 0.62,
     r"Skill estimate $\hat{s}$ updated  (5-session moving avg)",
     C_SKILL, fs=8.4, bold=False, zorder=5, lw=1.1)

# Arrows → skill node
arr(ax, Xsf,  L1_Y - NH/2, Xs + 1.10, Ys + 0.31,
    color=C_SKILL, lw=1.3, cs="arc3,rad=-0.18")
arr(ax, Xa,   L1_Y - NH/2, Xs,        Ys + 0.31,
    color=C_SKILL, lw=1.3, cs="arc3,rad=0.12")

# SRR judgment label between Xa and Xsf
ax.text((Xa + Xsf) / 2, L1_Y + 0.56,
        "SRR judgment", ha="center", va="center",
        fontsize=7.5, color=C_AFFECT, style="italic")

# Feedback return arc: SRR-Feedback → User  (thick coloured arc)
arr(ax, Xsf - NW/2, L1_Y + NH/2 + 0.04,
        Xu  + NW/2, L1_Y + NH/2 + 0.04,
    color=C_USER, lw=1.8, style="-|>",
    cs="arc3,rad=-0.42")
ax.text(6.8, L1_Y + NH/2 + 0.78,
        "feedback to user", ha="center", va="center",
        fontsize=7.5, color=C_USER, style="italic")

# ===========================================================================
# LAYER 2  —  Inter-session adaptive scaffolding fading + curriculum
# ===========================================================================
L2_YC = 5.38   # scaffold blocks centre y

# ---- Scaffold fading (left) -------------------------------------------------
sf_data = [
    ("High\n(Modeling)",    ORANGE, 0.78, 1.12, WHITE),
    ("Mid\n(Guided)",       YELLOW, 0.78, 0.80, BLACK),
    ("Low\n(Independent)",  GREEN,  0.78, 0.54, WHITE),
]
sf_x_starts = [0.75, 3.30, 5.84]  # left edge x of each block
sf_w = 2.18

for i, ((lbl, col, w, h, tc), xs) in enumerate(zip(sf_data, sf_x_starts)):
    cx = xs + sf_w / 2
    rbox(ax, cx, L2_YC, sf_w, h, col, ec=col, alpha=0.92, zorder=4)
    ax.text(cx, L2_YC, lbl, ha="center", va="center",
            fontsize=9.0, color=tc, fontweight="bold", zorder=5)
    if i < 2:
        arr(ax, xs + sf_w + 0.04, L2_YC,
                sf_x_starts[i+1] - 0.04, L2_YC,
            color="#555555", lw=1.4)

# Fading heading (above scaffold blocks) — two separate lines, enough gap
ax.text(4.28, L2_YC + 1.22,
        r"Scaffold level fades as $\hat{s}$ rises",
        ha="center", va="center", fontsize=9.5,
        color=BLACK, fontweight="bold")
# "sessions progress" shifted right of scaffold blocks to avoid overlap
ax.text(4.28, L2_YC + 0.78,
        "← sessions progress →",
        ha="center", va="center", fontsize=7.8,
        color=LGRAY, style="italic")

# Threshold legend (below scaffold blocks)
ax.text(4.28, L2_YC - 0.82,
        r"$\hat{s} < 0.35$: High  |  $0.35 \leq \hat{s} < 0.65$: Mid  |"
        r"  $\hat{s} \geq 0.65$: Low",
        ha="center", va="center", fontsize=7.9, color="#444444")

# ŝ update arrow from Layer 1 → Layer 2
# Arrow lands at centre-left of scaffold fading area
arr(ax, Xs, Ys - 0.31, 3.80, L2_YC + 0.72,
    color=C_SKILL, lw=1.7, cs="arc3,rad=0.25")
# Label: placed just below the "Scaffold level fades" heading, left side
# x=5.55 is between "Low" block right edge (~7.02) and divider (7.88) — clear gap
ax.text(5.45, L2_YC + 1.22,
        r"($\hat{s}$ propagates across sessions)",
        ha="center", va="center", fontsize=7.3,
        color=C_SKILL, style="italic")

# ---- Curriculum 4-stage (right) --------------------------------------------
curric = [
    ("Savoring",             SKY),
    ("Gratitude",            GREEN),
    ("Reframing",            ORANGE),
    ("Future-self\nOptimism",PINK),
]
cx0 = 8.52    # centre of first stage
cy_c = L2_YC
cw, ch, cgap = 1.25, 0.70, 1.43

for i, (stg, col) in enumerate(curric):
    cxi = cx0 + i * cgap
    rbox(ax, cxi, cy_c, cw, ch, col, ec=col, alpha=0.88, zorder=4)
    ax.text(cxi, cy_c, stg, ha="center", va="center",
            fontsize=8.2, color=WHITE, fontweight="bold", zorder=5)
    if i < len(curric) - 1:
        arr(ax, cxi + cw/2 + 0.04, cy_c,
                cx0 + (i+1)*cgap - cw/2 - 0.04, cy_c,
            color=LGRAY, lw=1.2)

curric_cx = cx0 + (len(curric)-1) * cgap / 2
ax.text(curric_cx, cy_c + 1.18,
        "4-Stage Curriculum",
        ha="center", va="center", fontsize=9.5,
        color=BLACK, fontweight="bold")
ax.text(curric_cx, cy_c - 0.78,
        r"(unlocked progressively as $\hat{s}$ rises)",
        ha="center", va="center", fontsize=7.8,
        color="#666666", style="italic")

# Divider
ax.plot([7.88, 7.88], [4.07, 6.98],
        color=LGRAY, lw=1.0, ls="--", zorder=1)

# ===========================================================================
# LAYER 3  —  A-B-A' internalization measurement
# ===========================================================================
phase_y_top = 3.65
phase_y_bot = 0.68
ph_h        = phase_y_top - phase_y_bot

phases = [
    ("A  (Baseline)",    LLGRAY, 0.96,  2.85, "No AI\n(free journaling)"),
    ("B  (Intervention)","#FFE8C0",4.00, 3.55, "GratiFlow active\n(scaffolding-fading loop)"),
    ("A'  (Follow-up)",  LLGRAY, 7.75,  2.85, "No AI\n(free journaling)"),
]

for lbl, col, xs, pw, sub in phases:
    cx = xs + pw / 2
    p = mpatches.FancyBboxPatch(
        (xs, phase_y_bot), pw, ph_h,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor=col, edgecolor=LGRAY, linewidth=1.2,
        alpha=0.80, zorder=3)
    ax.add_patch(p)
    ax.text(cx, phase_y_bot + ph_h - 0.30, lbl,
            ha="center", va="center", fontsize=10.5,
            color=BLACK, fontweight="bold", zorder=4)
    ax.text(cx, phase_y_bot + ph_h * 0.38, sub,
            ha="center", va="center", fontsize=8.0,
            color="#555555", zorder=4)

# Phase transition arrows (between A→B and B→A')
for xv in [4.00, 7.75]:
    ax.annotate("", xy=(xv + 0.01, phase_y_bot + ph_h * 0.62),
                xytext=(xv - 0.01, phase_y_bot + ph_h * 0.62),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.6),
                zorder=6)

# SRR schematic trajectory
inset_l = 0.97
inset_r = 10.58          # stop before evaluation-scope box
inset_y = phase_y_bot + 0.22
inset_h = ph_h * 0.38

x_s = np.linspace(inset_l, inset_r, 400)

def srr_curve(x):
    t = (x - inset_l) / (inset_r - inset_l)
    a_end = 0.26          # phase A ends at ~26 % of x range
    b_end = 0.74          # phase B ends at ~74 %
    ip = np.clip((t - a_end) / (b_end - a_end), 0.0, 1.0)
    return np.clip(
        np.where(
            t < a_end,
            0.08 + 0.04 * np.sin(20 * np.pi * t),
            np.where(
                t < b_end,
                0.09 + 0.68 * ip**1.4 + 0.04 * np.sin(18 * t),
                0.72 + 0.05 * np.sin(10 * t)
            )
        ), 0, 1)

y_s   = srr_curve(x_s)
y_plt = inset_y + y_s * inset_h

ax.plot(x_s, y_plt, color=C_SKILL, lw=2.2, zorder=5)
ax.fill_between(x_s, inset_y, y_plt, color=C_SKILL, alpha=0.13, zorder=4)

# SRR y-axis label
ax.text(inset_l - 0.18, inset_y + inset_h / 2,
        "SRR", ha="right", va="center", fontsize=8.0,
        color=C_SKILL, fontweight="bold", rotation=90)

# Durability annotation — top-right corner of Phase A', well above "No AI" text
# "No AI (free journaling)" sublabel sits at ph_h*0.38 from bottom ≈ 1.13 from bot
# phase_y_bot + ph_h - 0.30 is the title row; we place annotation mid-height
durability_x = 9.30
durability_y_arrow = inset_y + srr_curve(np.array([durability_x]))[0] * inset_h
# text box at x=9.85, just below title row of A'
annot_y = phase_y_bot + ph_h * 0.75   # 75% height = above sublabel, below title
ax.annotate(
    "Durability: SRR\nsustained w/o AI",
    xy=(durability_x, durability_y_arrow),
    xytext=(9.85, annot_y),
    fontsize=7.3, color=C_SKILL, ha="center", va="center",
    arrowprops=dict(arrowstyle="-|>", color=C_SKILL, lw=1.0,
                    connectionstyle="arc3,rad=-0.20"),
    zorder=7
)

# Layer-2 → Layer-3 connection
arr(ax, 4.40, 4.07, 5.60, phase_y_top + 0.02,
    color=C_COACH, lw=1.4, cs="arc3,rad=0.0")
ax.text(3.20, 3.87,
        "GratiFlow active\nin Phase B",
        ha="center", va="center", fontsize=7.5,
        color=C_COACH, style="italic")

# ===========================================================================
# Evaluation scope note  (bottom-right, outside Phase A' x range)
# ===========================================================================
note_cx, note_cy = 12.90, 1.68
note_w,  note_h  = 2.85, 1.75

# Solid background
rbox(ax, note_cx, note_cy, note_w, note_h,
     LLGRAY, ec=LGRAY, alpha=0.50, lw=1.0, zorder=6, r=0.18)

ax.text(note_cx, note_cy + 0.58,
        "Evaluation scope",
        ha="center", va="center", fontsize=8.5,
        color=BLACK, fontweight="bold", zorder=7)

ax.text(note_cx, note_cy + 0.18,
        u"• Synthetic personas\n  (this work)",
        ha="center", va="center", fontsize=7.8, color=BLACK, zorder=7)

ax.text(note_cx, note_cy - 0.35,
        u"• Human validation:\n  future work",
        ha="center", va="center", fontsize=7.8, color=LGRAY,
        style="italic", zorder=7)

# Dashed box around "future work" line
fw_box = FancyBboxPatch(
    (note_cx - note_w/2 + 0.06, note_cy - note_h/2 + 0.06),
    note_w - 0.12, note_h/2 - 0.08,
    boxstyle="round,pad=0,rounding_size=0.10",
    facecolor="none", edgecolor=LGRAY,
    linewidth=1.0, linestyle="--", zorder=8
)
ax.add_patch(fw_box)

# ===========================================================================
# Border lines
# ===========================================================================
ax.plot([0.18, FIG_W - 0.18], [10.53, 10.53], color=LGRAY, lw=0.7)
ax.plot([0.18, FIG_W - 0.18], [0.54,  0.54],  color=LGRAY, lw=0.7)

# ===========================================================================
# Save
# ===========================================================================
out_png     = os.path.join(OUT_EVAL, "framework_overview.png")
out_pdf     = os.path.join(OUT_EVAL, "framework_overview.pdf")
out_pdf_fig = os.path.join(OUT_FIG,  "framework_overview.pdf")
out_png_fig = os.path.join(OUT_FIG,  "framework_overview.png")  # added 2026-06-06
cap_path    = os.path.join(OUT_EVAL, "framework_overview_caption.md")

fig.savefig(out_png,     dpi=300, bbox_inches="tight", facecolor=WHITE)
fig.savefig(out_pdf,     bbox_inches="tight", facecolor=WHITE)
fig.savefig(out_pdf_fig, bbox_inches="tight", facecolor=WHITE)
fig.savefig(out_png_fig, dpi=300, bbox_inches="tight", facecolor=WHITE)  # added 2026-06-06
plt.close(fig)

print(f"[OK] PNG     -> {out_png}")
print(f"[OK] PDF     -> {out_pdf}")
print(f"[OK] PDF     -> {out_pdf_fig}")
print(f"[OK] PNG     -> {out_png_fig}")

# ===========================================================================
# Caption file
# ===========================================================================
caption = """\
# Figure 1 Caption — GratiFlow Framework Overview

**Proposed caption (Applied Intelligence / ACM TiiS style):**

> **Figure 1.** Overview of the GratiFlow scaffolding-fading multi-agent
> framework for positive-reframing skill development.
> **Layer 1 (top) — Within-session conversational loop.**
> The user submits a negative-event journal entry;
> the Reframing-Coach Agent responds at scaffold level
> $\\ell \\in \\{\\text{Modeling, Guided, Independent}\\}$
> determined by the current skill estimate $\\hat{s}$.
> The user produces a reframe attempt, which the Affect-Analysis Agent evaluates
> against rubric R1–R3 / F1–F5 (including echo detection, code F3).
> The SRR-Feedback Agent delivers personalised feedback, and $\\hat{s}$ is updated
> via a five-session moving average.
> **Layer 2 (middle) — Inter-session adaptive scaffolding fading.**
> As $\\hat{s}$ rises across sessions, scaffold level decreases monotonically
> from High (Modeling, $\\hat{s} < 0.35$)
> through Mid (Guided Practice, $0.35 \\le \\hat{s} < 0.65$)
> to Low (Independent Production, $\\hat{s} \\ge 0.65$).
> Content also progresses through four curriculum stages
> (Savoring $\\to$ Gratitude $\\to$ Reframing $\\to$ Future-self Optimism)
> unlocked as $\\hat{s}$ rises.
> **Layer 3 (bottom) — A-B-A' internalization measurement design.**
> Phase A (Baseline) and Phase A' (Follow-up) are AI-free;
> the schematic SRR trajectory illustrates the target internalization pattern:
> spontaneous reframing sustained in the absence of AI scaffolding (durability).
> The evaluation-scope note (bottom right, dashed) clarifies that all empirical
> results reported in this paper are from synthetic-persona simulation;
> human validation is designated as future work.
> Abbreviations: SRR = Spontaneous Reframing Rate;
> $\\hat{s}$ = current skill estimate; $\\ell$ = scaffold level.

---

*Author: team member (experiment, the research team). 2026-06-05.*
*Note: Caption length may be adjusted to journal requirements by Suzuki (writer).*
"""

with open(cap_path, "w", encoding="utf-8") as fh:
    fh.write(caption)

print(f"[OK] Caption -> {cap_path}")
