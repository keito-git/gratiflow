"""
SRR Instrument Validation: Separation Analysis
=================================================
Transparently isolates the contribution of (1) LLM criterion update and
(2) PI re-annotation to the observed kappa improvement (0.22 -> 0.43).

Four agreement combinations:
  (A) Old LLM x Old PI  (pre-clarification baseline)
  (B) New LLM x New PI  (post-clarification)
  (C) New LLM x Old PI  (LLM clarified, PI unchanged)
  (D) Old LLM x New PI  (PI re-annotated, LLM unchanged)

Author: team member (experiment lead, Team Kiyomiya)
Date: 2026-06-06
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Global rcParams — unified font sizes >= 13 pt, colorblind-friendly, 300 dpi (2026-06-08 visual refresh)
matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       15,
    "axes.titlesize":  14,
    "axes.labelsize":  15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = f"{BASE}/data/processed/srr_human_validation"
FIG_DIR_EVAL = f"{BASE}/python/figures"  # evaluation/ equivalent kept in python/figures
FIG_DIR_PAPER = (
    f"{BASE}/paper/en/GratiFlow__A_Scaffolding_Fading_Multi_Agent_LLM_for_"
    "Positive_Reframing_Skill_Development/figures"
)
REPORT_PATH = f"{DATA_DIR}/separation_analysis.md"

# ──────────────────────────────────────────────
# 1. Load data (utf-8-sig to handle BOM)
# ──────────────────────────────────────────────
pi_old = pd.read_csv(f"{DATA_DIR}/annotation_sheet_filled.csv", encoding="utf-8-sig")
pi_new = pd.read_csv(f"{DATA_DIR}/annotation_sheet_filled_r2_clarified.csv", encoding="utf-8-sig")
llm_old = pd.read_csv(f"{DATA_DIR}/llm_labels_blind.csv", encoding="utf-8-sig")
llm_new = pd.read_csv(f"{DATA_DIR}/llm_labels_r2_clarified.csv", encoding="utf-8-sig")

# Keep only id + label columns; sort by id for alignment
pi_old = pi_old[["id", "PI_SRR_label"]].rename(columns={"PI_SRR_label": "pi_old"}).set_index("id")
pi_new = pi_new[["id", "PI_SRR_label"]].rename(columns={"PI_SRR_label": "pi_new"}).set_index("id")
llm_old = llm_old[["id", "llm_srr_label"]].rename(columns={"llm_srr_label": "llm_old"}).set_index("id")
llm_new = llm_new[["id", "llm_srr_label"]].rename(columns={"llm_srr_label": "llm_new"}).set_index("id")

df = pi_old.join(pi_new).join(llm_old).join(llm_new)
assert len(df) == 70, f"Expected 70 rows, got {len(df)}"
print(f"Data loaded: N={len(df)}")

# ──────────────────────────────────────────────
# 2. Positive-rate summary
# ──────────────────────────────────────────────
pos_rates = {
    "pi_old":  df["pi_old"].sum(),
    "pi_new":  df["pi_new"].sum(),
    "llm_old": df["llm_old"].sum(),
    "llm_new": df["llm_new"].sum(),
}
print("\nPositive counts (N=70):")
for k, v in pos_rates.items():
    print(f"  {k}: {v}  ({v/70:.3f})")

# ──────────────────────────────────────────────
# 3. Agreement metrics for all 4 combinations
# ──────────────────────────────────────────────
def compute_metrics(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict:
    """Compute Acc/Prec/Rec/F1/Kappa and confusion matrix."""
    yt = y_true.values
    yp = y_pred.values
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    return {
        "label":     label,
        "accuracy":  accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall":    recall_score(yt, yp, zero_division=0),
        "f1":        f1_score(yt, yp, zero_division=0),
        "kappa":     cohen_kappa_score(yt, yp),
        "cm":        cm,
        "tn": cm[0, 0], "fp": cm[0, 1],
        "fn": cm[1, 0], "tp": cm[1, 1],
    }

combos = {
    "A": compute_metrics(df["pi_old"], df["llm_old"], "(A) Old LLM × Old PI"),
    "B": compute_metrics(df["pi_new"], df["llm_new"], "(B) New LLM × New PI"),
    "C": compute_metrics(df["pi_old"], df["llm_new"], "(C) New LLM × Old PI"),
    "D": compute_metrics(df["pi_new"], df["llm_old"], "(D) Old LLM × New PI"),
}

print("\n=== Agreement Metrics (PI as reference, LLM as system) ===")
header = f"{'Combo':<32} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Kappa':>7}"
print(header)
print("-" * len(header))
for key, m in combos.items():
    print(
        f"{m['label']:<32} "
        f"{m['accuracy']:>6.3f} "
        f"{m['precision']:>6.3f} "
        f"{m['recall']:>6.3f} "
        f"{m['f1']:>6.3f} "
        f"{m['kappa']:>7.3f}"
    )

# ──────────────────────────────────────────────
# 4. PI label change analysis
# ──────────────────────────────────────────────
pi_changed = df[df["pi_old"] != df["pi_new"]]
pi_1to0 = ((df["pi_old"] == 1) & (df["pi_new"] == 0)).sum()
pi_0to1 = ((df["pi_old"] == 0) & (df["pi_new"] == 1)).sum()
print(f"\n=== PI Label Changes (Old → New) ===")
print(f"  Total changed: {len(pi_changed)}")
print(f"  1→0 (dropped positive): {pi_1to0}")
print(f"  0→1 (added positive):   {pi_0to1}")
print(f"  Positive rate old: {df['pi_old'].sum()}/70 = {df['pi_old'].mean():.3f}")
print(f"  Positive rate new: {df['pi_new'].sum()}/70 = {df['pi_new'].mean():.3f}")

# ──────────────────────────────────────────────
# 5. LLM label change analysis
# ──────────────────────────────────────────────
llm_changed = df[df["llm_old"] != df["llm_new"]]
llm_1to0 = ((df["llm_old"] == 1) & (df["llm_new"] == 0)).sum()
llm_0to1 = ((df["llm_old"] == 0) & (df["llm_new"] == 1)).sum()
print(f"\n=== LLM Label Changes (Old → New) ===")
print(f"  Total changed: {len(llm_changed)}")
print(f"  1→0 (dropped positive): {llm_1to0}")
print(f"  0→1 (added positive):   {llm_0to1}")
print(f"  Positive rate old: {df['llm_old'].sum()}/70 = {df['llm_old'].mean():.3f}")
print(f"  Positive rate new: {df['llm_new'].sum()}/70 = {df['llm_new'].mean():.3f}")

# ──────────────────────────────────────────────
# 6. Cross-tabulation: PI change × LLM change
# ──────────────────────────────────────────────
df["pi_changed"]  = (df["pi_old"] != df["pi_new"]).astype(int)
df["llm_changed"] = (df["llm_old"] != df["llm_new"]).astype(int)
cross = pd.crosstab(df["pi_changed"], df["llm_changed"],
                    rownames=["PI changed"], colnames=["LLM changed"])
print("\n=== Cross-table: PI changed × LLM changed ===")
print(cross)

# IDs where both changed
both_changed_ids = df[(df["pi_changed"] == 1) & (df["llm_changed"] == 1)].index.tolist()
print(f"  Items where BOTH changed: {len(both_changed_ids)} -> {both_changed_ids}")

# ──────────────────────────────────────────────
# 7. Figure: Before/After confusion matrices (A) and (B)
# ──────────────────────────────────────────────
def plot_cm(ax, cm, title, subtitle, kappa, n=70):
    """Plot a single confusion matrix on ax."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=n)

    # Annotations — large, bold cell values
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0\n(Non-SRR)", "Pred: 1\n(SRR)"])
    ax.set_yticklabels(["True: 0\n(Non-SRR)", "True: 1\n(SRR)"])
    ax.set_xlabel("LLM Prediction")
    ax.set_ylabel("PI Annotation (Reference)")
    # Small subplot label only — no full figure title
    ax.set_title(f"{title}", fontsize=14, fontweight="bold", pad=8)
    # Kappa annotation
    ax.text(0.98, 0.02, f"Cohen's κ = {kappa:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    return im

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor("white")

plot_cm(
    axes[0], combos["A"]["cm"],
    title="(A) Before Clarification",
    subtitle="Old LLM × Old PI  |  N=70",
    kappa=combos["A"]["kappa"]
)
plot_cm(
    axes[1], combos["B"]["cm"],
    title="(B) After Clarification",
    subtitle="New LLM × New PI  |  N=70",
    kappa=combos["B"]["kappa"]
)

plt.tight_layout(rect=[0, 0, 1, 1])

# Save
out_stem = "srr_validation_confusion_before_after"
for ext in ["png", "pdf"]:
    path_eval  = f"{FIG_DIR_EVAL}/{out_stem}.{ext}"
    path_paper = f"{FIG_DIR_PAPER}/{out_stem}.{ext}"
    plt.savefig(path_eval,  dpi=300, bbox_inches="tight")
    plt.savefig(path_paper, dpi=300, bbox_inches="tight")
    print(f"Saved: {path_eval}")
    print(f"Saved: {path_paper}")

plt.close()

# ──────────────────────────────────────────────
# 8. Markdown report
# ──────────────────────────────────────────────
def fmt_cm_md(m):
    """Format confusion matrix as markdown 2x2 table."""
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    return (
        f"| | LLM=0 | LLM=1 |\n"
        f"|---|---|---|\n"
        f"| PI=0 | TN={tn} | FP={fp} |\n"
        f"| PI=1 | FN={fn} | TP={tp} |"
    )

md = f"""# SRR Instrument Validation: Separation Analysis
*Generated: 2026-06-06 | Author: team member (experiment lead, Team Kiyomiya)*

## Overview

This document transparently decomposes the kappa improvement observed in the
SRR instrument validation (κ ≈ 0.22 → κ ≈ 0.43) into contributions from
(1) LLM criterion update and (2) PI re-annotation, using four
label combinations on the identical N=70 item set.

---

## 1. Label Counts and Positive Rates

| Source | Positive (SRR=1) | Total | Positive rate |
|---|---|---|---|
| Old PI (original R2) | {df["pi_old"].sum()} | 70 | {df["pi_old"].mean():.3f} |
| New PI (clarified R2) | {df["pi_new"].sum()} | 70 | {df["pi_new"].mean():.3f} |
| Old LLM (original R2) | {df["llm_old"].sum()} | 70 | {df["llm_old"].mean():.3f} |
| New LLM (clarified R2) | {df["llm_new"].sum()} | 70 | {df["llm_new"].mean():.3f} |

---

## 2. Four-Combination Agreement Summary

| Combo | Description | Acc | Prec | Rec | F1 | Cohen's κ |
|---|---|---|---|---|---|---|
| (A) | Old LLM × Old PI | {combos["A"]["accuracy"]:.3f} | {combos["A"]["precision"]:.3f} | {combos["A"]["recall"]:.3f} | {combos["A"]["f1"]:.3f} | **{combos["A"]["kappa"]:.3f}** |
| (B) | New LLM × New PI | {combos["B"]["accuracy"]:.3f} | {combos["B"]["precision"]:.3f} | {combos["B"]["recall"]:.3f} | {combos["B"]["f1"]:.3f} | **{combos["B"]["kappa"]:.3f}** |
| (C) | New LLM × Old PI | {combos["C"]["accuracy"]:.3f} | {combos["C"]["precision"]:.3f} | {combos["C"]["recall"]:.3f} | {combos["C"]["f1"]:.3f} | **{combos["C"]["kappa"]:.3f}** |
| (D) | Old LLM × New PI | {combos["D"]["accuracy"]:.3f} | {combos["D"]["precision"]:.3f} | {combos["D"]["recall"]:.3f} | {combos["D"]["f1"]:.3f} | **{combos["D"]["kappa"]:.3f}** |

*PI = reference (ground truth); LLM = system prediction.*

### Confusion Matrices

**(A) Old LLM × Old PI** (κ = {combos["A"]["kappa"]:.3f})

{fmt_cm_md(combos["A"])}

**(B) New LLM × New PI** (κ = {combos["B"]["kappa"]:.3f})

{fmt_cm_md(combos["B"])}

**(C) New LLM × Old PI** (κ = {combos["C"]["kappa"]:.3f})

{fmt_cm_md(combos["C"])}

**(D) Old LLM × New PI** (κ = {combos["D"]["kappa"]:.3f})

{fmt_cm_md(combos["D"])}

---

## 3. PI Label Changes (Old → New Re-annotation)

| Metric | Value |
|---|---|
| Total changed | {len(pi_changed)} |
| Direction 1→0 (positive dropped) | {pi_1to0} |
| Direction 0→1 (positive added) | {pi_0to1} |
| Old positive rate | {df["pi_old"].sum()}/70 = {df["pi_old"].mean():.3f} |
| New positive rate | {df["pi_new"].sum()}/70 = {df["pi_new"].mean():.3f} |

The PI tightened the criterion considerably: {pi_1to0} items previously labeled as SRR were
re-annotated as non-SRR, while only {pi_0to1} item(s) moved in the opposite direction.
This reflects a shift toward a stricter definition of spontaneous cognitive reframing.

---

## 4. LLM Label Changes (Old → New Criterion)

| Metric | Value |
|---|---|
| Total changed | {len(llm_changed)} |
| Direction 1→0 (positive dropped) | {llm_1to0} |
| Direction 0→1 (positive added) | {llm_0to1} |
| Old positive rate | {df["llm_old"].sum()}/70 = {df["llm_old"].mean():.3f} |
| New positive rate | {df["llm_new"].sum()}/70 = {df["llm_new"].mean():.3f} |

---

## 5. Cross-Table: PI Changed × LLM Changed

```
{cross.to_string()}
```

Items where BOTH changed: {len(both_changed_ids)} → IDs: {both_changed_ids}

---

## 6. Qualitative Decomposition of κ Improvement

### Counterfactual combinations
- **(C) New LLM × Old PI** (κ = {combos["C"]["kappa"]:.3f}):
  Asks "what if only the LLM criterion changed but PI stayed at original labels?"
  If κ(C) > κ(A) = {combos["A"]["kappa"]:.3f}, LLM-side update alone contributes.
  Observed: κ(C) = {combos["C"]["kappa"]:.3f} → {"improvement" if combos["C"]["kappa"] > combos["A"]["kappa"] else "no improvement or degradation"} vs. baseline.

- **(D) Old LLM × New PI** (κ = {combos["D"]["kappa"]:.3f}):
  Asks "what if only PI re-annotated but LLM stayed at original labels?"
  If κ(D) > κ(A) = {combos["A"]["kappa"]:.3f}, PI-side update alone contributes.
  Observed: κ(D) = {combos["D"]["kappa"]:.3f} → {"improvement" if combos["D"]["kappa"] > combos["A"]["kappa"] else "no improvement or degradation"} vs. baseline.

### Interpretation
- Total κ improvement: {combos["B"]["kappa"] - combos["A"]["kappa"]:.3f}
  (from {combos["A"]["kappa"]:.3f} to {combos["B"]["kappa"]:.3f})
- κ(C) − κ(A) = {combos["C"]["kappa"] - combos["A"]["kappa"]:.3f}  ← estimated LLM-side contribution
- κ(D) − κ(A) = {combos["D"]["kappa"] - combos["A"]["kappa"]:.3f}  ← estimated PI-side contribution

A positive κ(D) − κ(A) indicates that the PI's re-annotation brought their labels
closer to the (unchanged) LLM labels, i.e., the human and LLM criteria converged.
A positive κ(C) − κ(A) indicates that the new LLM criterion independently
better matched the original PI labels.

### Important Limitations
1. **Non-additivity**: κ(C) − κ(A) and κ(D) − κ(A) do not sum to κ(B) − κ(A)
   because Cohen's kappa is nonlinear and the two changes are not orthogonal.
   Items where BOTH changed ({len(both_changed_ids)} cases) create interaction effects
   that cannot be attributed to either side alone.
2. **Single annotator**: All PI labels come from one annotator (the author).
   No inter-rater reliability for the PI labels themselves is available.
3. **Causal direction**: Combo (C) and (D) are counterfactual; neither
   represents an actual annotation run. The decomposition is descriptive only.

---

## 7. Definition Version Correspondence Table

| Version label | Synthetic data | Instrument (LLM scoring) | Instrument (PI annotation) |
|---|---|---|---|
| Original R2 | ✓ (used in synthesis) | Old LLM (blind) | Old PI |
| Clarified R2 | — | New LLM (clarified) | New PI |
| Before (Fig. 9) | — | Old LLM | Old PI |
| After (Fig. 9) | — | New LLM | New PI |

This table makes explicit that the "Before" and "After" labels in Figure 9
correspond to complete definition-version pairs, not partial updates.

---

## 8. Figure

**Fig. 9 candidate**: `srr_validation_confusion_before_after.png` / `.pdf`

Saved to:
- `{FIG_DIR_EVAL}/srr_validation_confusion_before_after.png`
- `{FIG_DIR_EVAL}/srr_validation_confusion_before_after.pdf`
- `{FIG_DIR_PAPER}/srr_validation_confusion_before_after.png`
- `{FIG_DIR_PAPER}/srr_validation_confusion_before_after.pdf`

Caption note: "Synthetic-free; human annotation, N=70, single annotator (PI).
Both LLM scoring and PI annotation were applied independently under each criterion version."
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\nReport saved: {REPORT_PATH}")
print("\n=== Done ===")
