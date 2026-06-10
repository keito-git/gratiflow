"""
srr_human_validation_compare_r2_clarified.py
=============================================
Agreement analysis for the R2-clarified re-verification.

Compares:
  - PI re-annotation under clarified R2 (annotation_sheet_filled_r2_clarified.csv)
  - LLM labels under clarified R2 (llm_labels_r2_clarified.csv)

Also produces a before/after comparison table using the original agreement metrics
(agreement_metrics.json) alongside the clarified-R2 metrics.

IMPORTANT: Run AFTER PI has completed re-annotation using annotate_r2_clarified.html.

Usage:
  python3 python/srr_human_validation_compare_r2_clarified.py \\
      --pi_csv  data/processed/srr_human_validation/annotation_sheet_filled_r2_clarified.csv \\
      --llm_csv data/processed/srr_human_validation/llm_labels_r2_clarified.csv \\
      --orig_metrics data/processed/srr_human_validation/agreement_metrics.json \\
      --out_dir data/processed/srr_human_validation/

NOTE on BOM handling:
  PI annotation CSV is downloaded from annotate_r2_clarified.html with a UTF-8 BOM.
  This script reads PI CSV with encoding='utf-8-sig' to strip the BOM automatically.
  (Previous BOM-related crash is fixed here.)

Data source: Ziems et al. (2022) Positive Psychology Frames (CC BY-SA 4.0)
  https://aclanthology.org/2022.acl-long.257/

Author: team member (experiment lead, the research team)
Date: 2026-06-06
"""

import argparse
import csv
import json
import os
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DEFAULT_PI_CSV       = os.path.join(BASE, "data/processed/srr_human_validation/annotation_sheet_filled_r2_clarified.csv")
DEFAULT_LLM_CSV      = os.path.join(BASE, "data/processed/srr_human_validation/llm_labels_r2_clarified.csv")
DEFAULT_ORIG_METRICS = os.path.join(BASE, "data/processed/srr_human_validation/agreement_metrics.json")
DEFAULT_OUT_DIR      = os.path.join(BASE, "data/processed/srr_human_validation")

# ─── Metric helpers ─────────────────────────────────────────────────────────────
def compute_metrics(pi_labels: list[int], llm_labels: list[int]) -> dict:
    """Compute accuracy, precision, recall, F1, Cohen's kappa, confusion matrix."""
    assert len(pi_labels) == len(llm_labels), "Label lists must be the same length"
    n = len(pi_labels)

    tp = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 0 and l == 1)
    fn = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 1 and l == 0)

    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    p_o       = (tp + tn) / n
    p_pos_pi  = (tp + fn) / n
    p_pos_llm = (tp + fp) / n
    p_neg_pi  = (tn + fp) / n
    p_neg_llm = (tn + fn) / n
    p_e       = p_pos_pi * p_pos_llm + p_neg_pi * p_neg_llm
    kappa     = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 0.0

    return {
        "n":             n,
        "accuracy":      round(accuracy,  4),
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "f1":            round(f1,        4),
        "cohens_kappa":  round(kappa,     4),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "pi_positive_rate":  round(p_pos_pi,  4),
        "llm_positive_rate": round(p_pos_llm, 4),
    }

def interpret_kappa(kappa: float) -> str:
    if kappa < 0:      return "Poor (< 0)"
    if kappa < 0.20:   return "Slight (0.00–0.20)"
    if kappa < 0.40:   return "Fair (0.20–0.40)"
    if kappa < 0.60:   return "Moderate (0.40–0.60)"
    if kappa < 0.80:   return "Substantial (0.60–0.80)"
    return "Almost perfect (0.80–1.00)"

# ─── Load helpers ──────────────────────────────────────────────────────────────
def load_pi_labels(pi_csv: str) -> dict[str, dict]:
    """
    Load PI labels from filled annotation CSV.
    Uses utf-8-sig encoding to strip BOM (produced by annotate_r2_clarified.html).
    Returns {id: {"label": int, "note": str}}.
    """
    labels = {}
    with open(pi_csv, newline="", encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        for row in csv.DictReader(f):
            item_id = row["id"].strip()
            raw = row.get("PI_SRR_label", "").strip()
            if raw not in ("0", "1"):
                raise ValueError(
                    f"Item {item_id}: PI_SRR_label must be 0 or 1, got '{raw}'. "
                    "Please fill in all 70 labels before running this script."
                )
            labels[item_id] = {
                "label": int(raw),
                "note":  row.get("PI_note", "").strip(),
            }
    return labels

def load_llm_labels(llm_csv: str) -> dict[str, dict]:
    """
    Load LLM blind labels.
    Returns {id: {llm_srr_label, llm_reasoning}}.
    """
    labels = {}
    with open(llm_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row["id"].strip()
            labels[item_id] = {
                "llm_srr_label": int(row["llm_srr_label"]),
                "llm_reasoning": row["llm_reasoning"],
            }
    return labels

def load_orig_metrics(orig_metrics_path: str) -> dict | None:
    """Load original (pre-clarification) agreement metrics. Returns None if file missing."""
    if not os.path.exists(orig_metrics_path):
        print(f"[WARN] Original metrics not found: {orig_metrics_path}")
        return None
    with open(orig_metrics_path, encoding="utf-8") as f:
        return json.load(f)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SRR human-LLM agreement analysis for R2-clarified re-verification."
    )
    parser.add_argument("--pi_csv",        default=DEFAULT_PI_CSV,
                        help="Path to PI re-annotation CSV (annotation_sheet_filled_r2_clarified.csv)")
    parser.add_argument("--llm_csv",       default=DEFAULT_LLM_CSV,
                        help="Path to clarified-R2 LLM labels (llm_labels_r2_clarified.csv)")
    parser.add_argument("--orig_metrics",  default=DEFAULT_ORIG_METRICS,
                        help="Path to original agreement_metrics.json (for before/after comparison)")
    parser.add_argument("--out_dir",       default=DEFAULT_OUT_DIR,
                        help="Output directory for agreement results")
    args = parser.parse_args()

    # Verify PI file exists
    if not os.path.exists(args.pi_csv):
        print(f"[ERROR] PI annotation file not found: {args.pi_csv}")
        print("Please complete re-annotation using annotate_r2_clarified.html,")
        print("then download the CSV and save it as:")
        print(f"  {DEFAULT_PI_CSV}")
        return

    print("=" * 65)
    print("SRR Human-LLM Agreement Analysis — R2 CLARIFIED Re-verification")
    print("=" * 65)

    pi_labels   = load_pi_labels(args.pi_csv)
    llm_labels  = load_llm_labels(args.llm_csv)
    orig_metrics = load_orig_metrics(args.orig_metrics)

    # Align by ID
    common_ids  = sorted(set(pi_labels.keys()) & set(llm_labels.keys()))
    missing_pi  = set(llm_labels.keys()) - set(pi_labels.keys())
    missing_llm = set(pi_labels.keys())  - set(llm_labels.keys())

    if missing_pi:
        print(f"[WARN] IDs in LLM but not in PI labels: {sorted(missing_pi)}")
    if missing_llm:
        print(f"[WARN] IDs in PI but not in LLM labels: {sorted(missing_llm)}")

    pi_vec  = [pi_labels[i]["label"]              for i in common_ids]
    llm_vec = [llm_labels[i]["llm_srr_label"]     for i in common_ids]

    metrics = compute_metrics(pi_vec, llm_vec)
    kappa_interp = interpret_kappa(metrics["cohens_kappa"])

    print(f"\nItems compared: {metrics['n']}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}  (LLM treats PI as ground truth)")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1:            {metrics['f1']:.4f}")
    print(f"  Cohen's kappa: {metrics['cohens_kappa']:.4f}  [{kappa_interp}]")
    cm = metrics["confusion_matrix"]
    print(f"\nConfusion matrix (PI=gold, LLM=pred):")
    print(f"              LLM=1   LLM=0")
    print(f"  PI=1 (SRR)    {cm['TP']:4d}    {cm['FN']:4d}")
    print(f"  PI=0 (not)    {cm['FP']:4d}    {cm['TN']:4d}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Clarified metrics JSON ──────────────────────────────────────────────
    metrics_out = {
        **metrics,
        "kappa_interpretation": kappa_interp,
        "rubric_version": "r2_clarified_v2 (r2_clarification_spec.md, the research team 2026-06-06)",
        "note": (
            "PI labels are treated as ground truth. "
            "LLM=gpt-5.4-mini with clarified R2 rubric. "
            "PI re-annotated under clarified R2 definition."
        ),
        "data_source": {
            "dataset": "SALT-NLP/positive_reframing",
            "paper": "Ziems et al. (2022), ACL 2022",
            "license": "CC BY-SA 4.0",
            "url": "https://aclanthology.org/2022.acl-long.257/",
        },
        "limitations": [
            "Single annotator (PI): inter-rater reliability unknown.",
            "Test set N=70 from a single dataset (Ziems et al., 2022).",
            "Improvement reflects reduced definitional ambiguity, not universal validity.",
        ],
    }
    metrics_path = os.path.join(args.out_dir, "agreement_metrics_r2_clarified.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {metrics_path}")

    # ── 2. Confusion matrix CSV ────────────────────────────────────────────────
    cm_path = os.path.join(args.out_dir, "confusion_matrix_r2_clarified.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "LLM_pred=1", "LLM_pred=0"])
        writer.writerow(["PI_gold=1 (SRR)", cm["TP"], cm["FN"]])
        writer.writerow(["PI_gold=0 (not)", cm["FP"], cm["TN"]])
    print(f"[saved] {cm_path}")

    # ── 3. Misclassified items CSV ─────────────────────────────────────────────
    mismatch_rows = []
    for i, item_id in enumerate(common_ids):
        pi_lbl  = pi_vec[i]
        llm_lbl = llm_vec[i]
        if pi_lbl != llm_lbl:
            error_type = "FP" if llm_lbl == 1 and pi_lbl == 0 else "FN"
            mismatch_rows.append({
                "id":            item_id,
                "pi_srr_label":  pi_lbl,
                "llm_srr_label": llm_lbl,
                "error_type":    error_type,
                "pi_note":       pi_labels[item_id]["note"],
                "llm_reasoning": llm_labels[item_id]["llm_reasoning"],
            })

    mismatch_path = os.path.join(args.out_dir, "misclassified_items_r2_clarified.csv")
    with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pi_srr_label", "llm_srr_label",
                                                "error_type", "pi_note", "llm_reasoning"])
        writer.writeheader()
        writer.writerows(mismatch_rows)
    print(f"[saved] {mismatch_path}  ({len(mismatch_rows)} mismatches)")

    # ── 4. Before/after comparison table + Markdown report ────────────────────
    report_lines = [
        "# SRR Human-LLM Agreement Report — R2 Clarified Re-verification",
        "",
        "**Data**: Ziems et al. (2022) Positive Psychology Frames (CC BY-SA 4.0)",
        "**Dataset**: SALT-NLP/positive_reframing — https://aclanthology.org/2022.acl-long.257/",
        "**LLM**: gpt-5.4-mini, rubric from r2_clarification_spec.md (the research team, 2026-06-06)",
        "**R2 clarification basis**: Gross (1998/2002); Garnefski et al. (2001 CERQ)",
        "",
    ]

    # Before/after table
    if orig_metrics is not None:
        orig_kappa = orig_metrics.get("cohens_kappa", "N/A")
        orig_acc   = orig_metrics.get("accuracy",     "N/A")
        orig_prec  = orig_metrics.get("precision",    "N/A")
        orig_rec   = orig_metrics.get("recall",       "N/A")
        orig_f1    = orig_metrics.get("f1",           "N/A")
        orig_kappa_interp = interpret_kappa(orig_kappa) if isinstance(orig_kappa, float) else "N/A"

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        def delta(new_v, old_v):
            if isinstance(new_v, float) and isinstance(old_v, float):
                d = new_v - old_v
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.4f}"
            return "N/A"

        report_lines += [
            "## Before/After Comparison (R2 Definition Change)",
            "",
            "| Metric | Before (original R2) | After (clarified R2) | Delta |",
            "|--------|---------------------|---------------------|-------|",
            f"| Cohen's kappa | {fmt(orig_kappa)} ({orig_kappa_interp.split(' ')[0]}) | {fmt(metrics['cohens_kappa'])} ({kappa_interp.split(' ')[0]}) | {delta(metrics['cohens_kappa'], orig_kappa)} |",
            f"| Accuracy | {fmt(orig_acc)} | {fmt(metrics['accuracy'])} | {delta(metrics['accuracy'], orig_acc)} |",
            f"| Precision | {fmt(orig_prec)} | {fmt(metrics['precision'])} | {delta(metrics['precision'], orig_prec)} |",
            f"| Recall | {fmt(orig_rec)} | {fmt(metrics['recall'])} | {delta(metrics['recall'], orig_rec)} |",
            f"| F1 | {fmt(orig_f1)} | {fmt(metrics['f1'])} | {delta(metrics['f1'], orig_f1)} |",
            "",
            "> **Interpretation note**: The \"Before\" column reflects LLM judgment under the original R2",
            "> definition against PI labels collected under the same original rubric.",
            "> The \"After\" column reflects both PI and LLM judgment under the clarified R2.",
            "> Improvement reflects reduced definitional ambiguity, not universal validity.",
            "> Single-annotator design: inter-rater reliability is unknown.",
            "",
        ]

    # Current metrics
    report_lines += [
        "## Agreement Metrics (R2 Clarified)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| N items | {metrics['n']} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| Cohen's kappa | {metrics['cohens_kappa']:.4f} ({kappa_interp}) |",
        f"| PI positive rate | {metrics['pi_positive_rate']:.4f} |",
        f"| LLM positive rate | {metrics['llm_positive_rate']:.4f} |",
        "",
        "## Confusion Matrix (PI = ground truth, LLM = prediction)",
        "",
        "|  | LLM=1 (SRR) | LLM=0 (not SRR) |",
        "|--|-------------|-----------------|",
        f"| PI=1 (SRR) | TP={cm['TP']} | FN={cm['FN']} |",
        f"| PI=0 (not SRR) | FP={cm['FP']} | TN={cm['TN']} |",
        "",
        f"## Misclassified Items ({len(mismatch_rows)} items)",
        "",
    ]

    if mismatch_rows:
        report_lines += [
            "| id | PI label | LLM label | Error type | PI note | LLM reasoning |",
            "|----|----------|-----------|------------|---------|---------------|",
        ]
        for row in mismatch_rows:
            reason = row["llm_reasoning"].replace("|", "\\|")
            pi_note = (row["pi_note"] or "").replace("|", "\\|")
            report_lines.append(
                f"| {row['id']} | {row['pi_srr_label']} | {row['llm_srr_label']} "
                f"| {row['error_type']} | {pi_note} | {reason} |"
            )
    else:
        report_lines.append("No misclassified items — perfect agreement.")

    report_lines += [
        "",
        "---",
        "",
        "**FP** = LLM judged SRR=1 but PI judged 0.",
        "**FN** = LLM judged SRR=0 but PI judged 1.",
        "",
        "## Limitations",
        "",
        "1. **Single annotator**: All PI labels from one annotator. Inter-rater reliability unknown.",
        "2. **N=70, single dataset**: Ziems et al. (2022) only. Generalizability not established.",
        "3. **Convergence test**: Improvement tests LLM-convergence-to-PI, not mutual convergence to shared standard.",
        "4. **Claim ceiling (v3.2)**: Improvement reflects 'reduced definitional ambiguity', not validation.",
        "",
        "## Attribution",
        "",
        "Ziems, C., Li, M., Zhang, A., & Yang, D. (2022). Inducing Positive Perspectives",
        "with Text Reframing. *ACL 2022*, 3682–3700.",
        "https://aclanthology.org/2022.acl-long.257/",
        "License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
        "",
        "R2 clarification basis:",
        "Gross, J. J. (1998). JPSP, 74(1), 224-237.",
        "Gross, J. J. (2002). Psychophysiology, 39(3), 281-291.",
        "Garnefski, N., Kraaij, V., & Spinhoven, P. (2001). PAID, 30(8), 1311-1327.",
    ]

    report_path = os.path.join(args.out_dir, "agreement_report_r2_clarified.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[saved] {report_path}")

    print("\nDone. Review agreement_metrics_r2_clarified.json and agreement_report_r2_clarified.md.")


if __name__ == "__main__":
    main()
