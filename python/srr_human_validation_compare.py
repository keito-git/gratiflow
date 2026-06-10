"""
srr_human_validation_compare.py
=================================
Agreement analysis between PI human labels and LLM blind labels.

Run this script AFTER the PI has completed annotation.

Usage:
  python3 python/srr_human_validation_compare.py \\
      --pi_csv data/processed/srr_human_validation/annotation_sheet_filled.csv \\
      --llm_csv data/processed/srr_human_validation/llm_labels_blind.csv \\
      --out_dir data/processed/srr_human_validation/

  The PI-filled CSV should be a copy of annotation_sheet_blank.csv
  with PI_SRR_label column filled in (1 or 0 for each of the 70 items).

Outputs:
  - agreement_metrics.json  : accuracy, precision, recall, F1, Cohen's kappa
  - confusion_matrix.csv    : 2x2 confusion matrix
  - misclassified_items.csv : items where PI label != LLM label (for error analysis)
  - agreement_report.md     : human-readable summary

Data source: Ziems et al. (2022) Positive Psychology Frames (CC BY-SA 4.0)
  https://aclanthology.org/2022.acl-long.257/

Author: team member (experiment lead, the research team)
Date: 2026-06-05
"""

import argparse
import csv
import json
import os
import math
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DEFAULT_PI_CSV  = os.path.join(BASE, "data/processed/srr_human_validation/annotation_sheet_filled.csv")
DEFAULT_LLM_CSV = os.path.join(BASE, "data/processed/srr_human_validation/llm_labels_blind.csv")
DEFAULT_OUT_DIR = os.path.join(BASE, "data/processed/srr_human_validation")

# ─── Metric helpers ─────────────────────────────────────────────────────────────
def compute_metrics(pi_labels: list[int], llm_labels: list[int]) -> dict:
    """Compute accuracy, precision, recall, F1, Cohen's kappa, confusion matrix."""
    assert len(pi_labels) == len(llm_labels), "Label lists must be the same length"
    n = len(pi_labels)

    # Confusion matrix: PI is ground truth, LLM is prediction
    tp = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 0 and l == 1)  # LLM says SRR, PI says not
    fn = sum(1 for p, l in zip(pi_labels, llm_labels) if p == 1 and l == 0)  # LLM misses SRR

    accuracy  = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Cohen's kappa
    p_o = (tp + tn) / n  # observed agreement
    p_pos_pi  = (tp + fn) / n   # PI positive rate
    p_pos_llm = (tp + fp) / n   # LLM positive rate
    p_neg_pi  = (tn + fp) / n
    p_neg_llm = (tn + fn) / n
    p_e = p_pos_pi * p_pos_llm + p_neg_pi * p_neg_llm  # expected agreement by chance
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 0.0

    return {
        "n": n,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "cohens_kappa": round(kappa,  4),
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "pi_positive_rate":  round(p_pos_pi, 4),
        "llm_positive_rate": round(p_pos_llm, 4),
    }

# ─── Kappa interpretation ──────────────────────────────────────────────────────
def interpret_kappa(kappa: float) -> str:
    if kappa < 0:      return "Poor (< 0)"
    if kappa < 0.20:   return "Slight (0.00–0.20)"
    if kappa < 0.40:   return "Fair (0.20–0.40)"
    if kappa < 0.60:   return "Moderate (0.40–0.60)"
    if kappa < 0.80:   return "Substantial (0.60–0.80)"
    return "Almost perfect (0.80–1.00)"

# ─── Load helpers ──────────────────────────────────────────────────────────────
def load_pi_labels(pi_csv: str) -> dict[str, int]:
    """Load PI labels from filled annotation CSV. Returns {id: label (int)}."""
    labels = {}
    with open(pi_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row["id"].strip()
            raw = row.get("PI_SRR_label", "").strip()
            if raw not in ("0", "1"):
                raise ValueError(
                    f"Item {item_id}: PI_SRR_label must be 0 or 1, got '{raw}'. "
                    "Please fill in all 70 labels before running this script."
                )
            labels[item_id] = int(raw)
    return labels

def load_llm_labels(llm_csv: str) -> dict[str, dict]:
    """Load LLM blind labels. Returns {id: {llm_srr_label, llm_reasoning}}."""
    labels = {}
    with open(llm_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item_id = row["id"].strip()
            labels[item_id] = {
                "llm_srr_label": int(row["llm_srr_label"]),
                "llm_reasoning": row["llm_reasoning"],
            }
    return labels

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SRR human-LLM agreement analysis. Run after PI annotation."
    )
    parser.add_argument("--pi_csv",  default=DEFAULT_PI_CSV,
                        help="Path to PI-filled annotation CSV")
    parser.add_argument("--llm_csv", default=DEFAULT_LLM_CSV,
                        help="Path to LLM blind labels CSV")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Output directory for agreement results")
    args = parser.parse_args()

    # Verify PI file exists
    if not os.path.exists(args.pi_csv):
        print(f"[ERROR] PI annotation file not found: {args.pi_csv}")
        print("Please save the completed annotation sheet as:")
        print(f"  {DEFAULT_PI_CSV}")
        print("(Copy annotation_sheet_blank.csv, fill PI_SRR_label column, save as annotation_sheet_filled.csv)")
        return

    print("=" * 60)
    print("SRR Human-LLM Agreement Analysis")
    print("=" * 60)

    pi_labels  = load_pi_labels(args.pi_csv)
    llm_labels = load_llm_labels(args.llm_csv)

    # Align by ID
    common_ids = sorted(set(pi_labels.keys()) & set(llm_labels.keys()))
    missing_pi  = set(llm_labels.keys()) - set(pi_labels.keys())
    missing_llm = set(pi_labels.keys())  - set(llm_labels.keys())

    if missing_pi:
        print(f"[WARN] IDs in LLM but not in PI labels: {sorted(missing_pi)}")
    if missing_llm:
        print(f"[WARN] IDs in PI but not in LLM labels: {sorted(missing_llm)}")

    pi_vec  = [pi_labels[i]              for i in common_ids]
    llm_vec = [llm_labels[i]["llm_srr_label"] for i in common_ids]

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

    # 1. Metrics JSON
    metrics_out = {
        **metrics,
        "kappa_interpretation": kappa_interp,
        "note": "PI labels are treated as ground truth. "
                "LLM=gpt-5.4-mini with SRR rubric (evaluation_protocol_v2_1.md).",
        "data_source": {
            "dataset": "SALT-NLP/positive_reframing",
            "paper": "Ziems et al. (2022), ACL 2022",
            "license": "CC BY-SA 4.0",
            "url": "https://aclanthology.org/2022.acl-long.257/",
        },
    }
    metrics_path = os.path.join(args.out_dir, "agreement_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {metrics_path}")

    # 2. Confusion matrix CSV
    cm_path = os.path.join(args.out_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "LLM_pred=1", "LLM_pred=0"])
        writer.writerow(["PI_gold=1 (SRR)", cm["TP"], cm["FN"]])
        writer.writerow(["PI_gold=0 (not)", cm["FP"], cm["TN"]])
    print(f"[saved] {cm_path}")

    # 3. Misclassified items CSV
    mismatch_rows = []
    for i, item_id in enumerate(common_ids):
        pi_lbl  = pi_vec[i]
        llm_lbl = llm_vec[i]
        if pi_lbl != llm_lbl:
            error_type = ("FP" if llm_lbl == 1 and pi_lbl == 0 else "FN")
            mismatch_rows.append({
                "id":             item_id,
                "pi_srr_label":   pi_lbl,
                "llm_srr_label":  llm_lbl,
                "error_type":     error_type,
                "llm_reasoning":  llm_labels[item_id]["llm_reasoning"],
            })

    mismatch_path = os.path.join(args.out_dir, "misclassified_items.csv")
    with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pi_srr_label", "llm_srr_label",
                                                "error_type", "llm_reasoning"])
        writer.writeheader()
        writer.writerows(mismatch_rows)
    print(f"[saved] {mismatch_path}  ({len(mismatch_rows)} mismatches)")

    # 4. Human-readable Markdown report
    report_lines = [
        "# SRR Human-LLM Agreement Report",
        "",
        "**Data**: Ziems et al. (2022) Positive Psychology Frames (CC BY-SA 4.0)",
        "**Dataset**: SALT-NLP/positive_reframing — https://aclanthology.org/2022.acl-long.257/",
        "**LLM**: gpt-5.4-mini, rubric from evaluation_protocol_v2_1.md",
        "",
        "## Agreement Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
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
            "| id | PI label | LLM label | Error type | LLM reasoning |",
            "|----|----------|-----------|------------|---------------|",
        ]
        for row in mismatch_rows:
            reason = row["llm_reasoning"].replace("|", "\\|")
            report_lines.append(
                f"| {row['id']} | {row['pi_srr_label']} | {row['llm_srr_label']} "
                f"| {row['error_type']} | {reason} |"
            )
    else:
        report_lines.append("No misclassified items — perfect agreement.")

    report_lines += [
        "",
        "---",
        "",
        "**FP** = LLM judged SRR=1 but PI judged 0 (LLM false positive / over-attribution).",
        "**FN** = LLM judged SRR=0 but PI judged 1 (LLM false negative / under-attribution).",
        "",
        "## Attribution",
        "",
        "Ziems, C., Li, M., Zhang, A., & Yang, D. (2022). Inducing Positive Perspectives",
        "with Text Reframing. *ACL 2022*, 3682–3700.",
        "https://aclanthology.org/2022.acl-long.257/",
        "License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
    ]

    report_path = os.path.join(args.out_dir, "agreement_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[saved] {report_path}")

    print("\nDone. Review agreement_metrics.json and agreement_report.md for full results.")


if __name__ == "__main__":
    main()
