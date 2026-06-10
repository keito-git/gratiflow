"""
srr_human_validation_sample.py
================================
Sampling script for SRR single-annotator validation (PI annotation sheet).

Data source: Ziems et al. (2022) Positive Psychology Frames
  Dataset: SALT-NLP/positive_reframing (CC BY-SA 4.0)
  Reference: Ziems, C., Li, M., Zhang, A., & Yang, D. (2022).
             Inducing Positive Perspectives with Text Reframing. ACL 2022, 3682-3700.
             https://aclanthology.org/2022.acl-long.257/

Output:
  - data/processed/srr_human_validation/annotation_sheet_blank.csv
  - data/processed/srr_human_validation/annotation_sheet_blank.md
  - data/processed/srr_human_validation/sampling_log.json
  - data/processed/srr_human_validation/coding_guide.md

Author: team member (experiment lead, the research team)
Date: 2026-06-05
Seed: 42
"""

import csv
import json
import random
import os
from collections import Counter

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RAW_CSV = os.path.join(BASE, "data/raw/srr_realdata_ziems2022/srr_realdata_sample_300.csv")
OUT_DIR = os.path.join(BASE, "data/processed/srr_human_validation")

SEED = 42
N_POSITIVE = 35   # positive candidates (reframed_text from positive sample rows)
N_NEGATIVE = 35   # negative candidates (original_text for easy_neg, full text for hard_neg)

# Target strategy balance for positive items (across N_POSITIVE=35)
STRATEGY_TARGETS = {
    "growth": 9,
    "optimism": 9,
    "thankfulness": 6,
    "self_affirmation": 5,
    "impermanence": 3,
    "neutralizing": 3,   # boundary case (hard_negative with neutralizing = near-miss)
}

# ─── Load raw data ─────────────────────────────────────────────────────────────
def load_raw() -> list[dict]:
    with open(RAW_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ─── Sampling logic ────────────────────────────────────────────────────────────
def sample_items(rows: list[dict], rng: random.Random) -> list[dict]:
    """
    Sample 70 items (35 positive + 35 negative) with balanced strategy distribution.

    Positive candidates: rows where sample_type == 'positive'.
      - Candidate text: the 'text' field (reframed version).
      - Context: original_text_ref (the negative tweet that was reframed).
      - Strategy grouping targets: STRATEGY_TARGETS.

    Negative candidates: rows where sample_type in ('easy_negative', 'hard_negative').
      - For easy_negative: text == original_text_ref (verbatim negative tweet, no reframe).
        Candidate text = text field directly.
      - For hard_negative (neutralizing): represents near-miss / ambiguous acceptance (F2).
        Candidate text = text field (neutral paraphrase without genuine reframe).
      - Context for negatives: original_text_ref when available, else text itself.
    """
    positives = [r for r in rows if r["sample_type"] == "positive"]
    easy_neg  = [r for r in rows if r["sample_type"] == "easy_negative"]
    hard_neg  = [r for r in rows if r["sample_type"] == "hard_negative"]

    # --- Sample positives with strategy balance ---
    pos_by_strategy = {}
    for r in positives:
        s = r["strategy"]
        pos_by_strategy.setdefault(s, []).append(r)

    sampled_pos = []
    for strategy, target_n in STRATEGY_TARGETS.items():
        pool = pos_by_strategy.get(strategy, [])
        rng.shuffle(pool)
        sampled_pos.extend(pool[:target_n])

    # If we didn't hit N_POSITIVE exactly due to pool size, fill from remaining positives
    remaining_pos = [r for r in positives if r not in sampled_pos]
    rng.shuffle(remaining_pos)
    while len(sampled_pos) < N_POSITIVE and remaining_pos:
        sampled_pos.append(remaining_pos.pop(0))

    sampled_pos = sampled_pos[:N_POSITIVE]

    # --- Sample negatives ---
    # Half easy_negative (clearly negative, no reframe attempt): ~17-18
    # Half hard_negative (near-miss, neutralizing boundary): ~17-18
    rng.shuffle(easy_neg)
    rng.shuffle(hard_neg)
    sampled_neg = easy_neg[:18] + hard_neg[:17]
    rng.shuffle(sampled_neg)
    sampled_neg = sampled_neg[:N_NEGATIVE]

    return sampled_pos, sampled_neg

def build_annotation_rows(sampled_pos: list[dict], sampled_neg: list[dict],
                           rng: random.Random) -> list[dict]:
    """
    Build the interleaved annotation sheet with blanked labels.
    Columns: id, context, candidate_text, PI_SRR_label, PI_note
    gold_label stored separately (not included in blank sheet).
    """
    items = []
    id_counter = 1

    for r in sampled_pos:
        context = r["original_text_ref"].strip() if r.get("original_text_ref") else ""
        candidate = r["text"].strip()
        items.append({
            "id": f"H{id_counter:02d}",
            "context": context,
            "candidate_text": candidate,
            "PI_SRR_label": "",   # PI fills this: 1=SRR, 0=not SRR
            "PI_note": "",
            # Internal fields (stripped from blank sheet, retained in sampling_log)
            "_gold_label": True,
            "_sample_type": r["sample_type"],
            "_strategy": r["strategy"],
            "_item_id": r["item_id"],
        })
        id_counter += 1

    for r in sampled_neg:
        # Context for negatives: original_text_ref when non-empty and different from text
        orig = r.get("original_text_ref", "").strip()
        context = orig if orig and orig != r["text"].strip() else ""
        candidate = r["text"].strip()
        items.append({
            "id": f"H{id_counter:02d}",
            "context": context,
            "candidate_text": candidate,
            "PI_SRR_label": "",
            "PI_note": "",
            "_gold_label": False,
            "_sample_type": r["sample_type"],
            "_strategy": r["strategy"],
            "_item_id": r["item_id"],
        })
        id_counter += 1

    # Shuffle so positives/negatives are interleaved (blind to label)
    rng.shuffle(items)
    # Re-number IDs after shuffle
    for i, item in enumerate(items, 1):
        item["id"] = f"H{i:02d}"

    return items

# ─── Coding guide ──────────────────────────────────────────────────────────────
CODING_GUIDE_MD = """# SRR Coding Guide — Single-Annotator Validation

**Data source**: Ziems, C., Li, M., Zhang, A., & Yang, D. (2022).
Inducing Positive Perspectives with Text Reframing. *ACL 2022*, 3682–3700.
Dataset: SALT-NLP/positive_reframing, CC BY-SA 4.0.
License: https://creativecommons.org/licenses/by-sa/4.0/

**Prepared by**: team member (experiment lead, the research team)
**Protocol basis**: evaluation_protocol_v2_1.md (the research team, 2026-06-04)
**Date**: 2026-06-05

---

## 1. Task Description

For each of the 70 items below, you will see:

- **context**: The original negative tweet / statement (source of negative emotion).
  *This is the raw negative input that may have been reframed.*
- **candidate_text**: The text to be judged.
  *This may be a reframe of the context, or it may be a non-reframed text.*

**Your task**: Judge whether `candidate_text` constitutes a **Spontaneous Reframing Response (SRR)**
by marking `PI_SRR_label` as **1 (SRR)** or **0 (not SRR)**.

---

## 2. Rubric Definition

### 2.1 Mark SRR = 1 when ALL THREE criteria are met:

| Criterion | Description |
|-----------|-------------|
| **R1 — Explicit negative** | The candidate text references or implies a specific negative event or negative emotion (from the context). |
| **R2 — Deliberate reinterpretation** | The candidate text transforms the *meaning* of that same negative event into a positive, growth-oriented, or silver-lining perspective. |
| **R3 — Self-generated** | The reinterpretation appears to originate from the speaker themselves — not merely echoed from a prior instruction or external prompt. |

**All three (R1 + R2 + R3) must be present. Missing any one → SRR = 0.**

### 2.2 Mark SRR = 0 when ANY of the following apply:

| Code | Label | Description |
|------|-------|-------------|
| **F1** | Forced positive | The text describes a positive event or emotion *without* linking it to a negative one. There is no reinterpretation — just a separate positive fact. |
| **F2** | Vague acceptance | The text expresses generic optimism or a coping platitude ("it'll work out", "nothing I can do") without specific cognitive reinterpretation. |
| **F3** | AI echo | The reframe repeats or closely paraphrases a reframe that was explicitly modeled by an AI or external agent. |
| **F4** | AI-induced | The candidate was produced only because of a direct prompt instruction; there is no evidence of self-generated cognitive shift. |
| **F5** | Negative only | The candidate text only describes negative events or emotions, with no positive reinterpretation at all. |

---

## 3. Examples

### SRR = 1 (meets R1 + R2 + R3)

**Example A** (growth / confident reframe):
- Context: *"Studied and made notecards all night, uhhhh hate finals and keystones at the same time."*
- Candidate: *"I studied and took notes all night, I'm working hard and that's good."*
- Judgment: **1** — R1: studying all night under exam stress (negative); R2: reinterprets effort as inherently positive ("working hard and that's good"); R3: self-generated perspective.

**Example B** (self_affirmation / tentative reframe):
- Context: *"can guarantee I will cry at least once before Friday I'm so far behind in my nursing readings"*
- Candidate: *"I fell behind my nursing readings, but I know it's my own fault, now I have to power through all of it, I can do it!"*
- Judgment: **1** — R1: falling behind (negative); R2: acknowledges fault and reframes as a self-efficacy challenge ("I can do it!"); R3: self-generated determination.

**Example C** (thankfulness / silver lining):
- Context: *"...so exhausted and stressed and broke from the planning and prep..."*
- Candidate: *"My brother is having a destination wedding next week I am happy for him."*
- Judgment: **1** — R1: implied stress from wedding planning context; R2: reframes attention toward family joy; R3: self-generated.
  *(Note: This is a borderline case — record uncertainty in PI_note if needed.)*

---

### SRR = 0 — F1 (separate positive, no reinterpretation)

- Context: *"Project project project~~.. Y cnt teacher assign us earlier...Made us really rush hard!"*
- Candidate: *"Teachers ought to give more time to students for project work. This would result in a smooth and timely completion of projects."*
- Judgment: **0** (F1/F2) — No reinterpretation of the stress; candidate merely states a policy preference. No silver-lining on the past event.

---

### SRR = 0 — F2 (vague acceptance / neutralizing)

- Context: *"I want to lay down and die..., Just wake up at a time when society isn't such a load of bullshit."*
- Candidate: *"I wanted to lay down, and wake up at a time when society isn't so bad. But that's life, and I should be enjoying the good parts of it."*
- Judgment: **0** (F2) — "that's life, and I should be enjoying the good parts" is resignation / vague coping without specific cognitive reinterpretation. No positive meaning is actively constructed from the negative event.

---

### SRR = 0 — F5 (negative only)

- Context: *"I need a day just to finish all my projects and graduation stuff."*
- Candidate: *"I need a day just to finish all my projects and graduation stuff. justoneday stressed"*
- Judgment: **0** (F5) — Pure negative description; no reinterpretation present.

---

## 4. Annotation Instructions

1. Read **context** first (the negative source event).
2. Read **candidate_text** (what is to be judged).
3. Check R1 → R2 → R3 in order. If all three pass, mark **1**. Otherwise mark **0**.
4. If you are uncertain, record your reasoning in **PI_note**.
5. Do **not** consult the LLM judgment before completing all 70 items.
6. When done, save the completed CSV as:
   `data/processed/srr_human_validation/annotation_sheet_filled.csv`
   (copy `annotation_sheet_blank.csv`, add your labels, save under the new name).

---

## 5. Attribution

Data: Ziems, C., Li, M., Zhang, A., & Yang, D. (2022).
Inducing Positive Perspectives with Text Reframing.
*Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 3682–3700.
[https://aclanthology.org/2022.acl-long.257/](https://aclanthology.org/2022.acl-long.257/)
License: CC BY-SA 4.0 — [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)

Rubric basis: the research team (2026). GratiFlow evaluation_protocol_v2_1.md.
Operationalized from: Gross, J. J. (1998). Antecedent- and response-focused emotion regulation.
*Journal of Personality and Social Psychology*, 74(1), 224–237.
"""

# ─── Markdown annotation table ─────────────────────────────────────────────────
def build_md_table(items: list[dict]) -> str:
    """Build a human-readable Markdown table of the annotation sheet (blank)."""
    lines = [
        "# PI Annotation Sheet — SRR Single-Annotator Validation",
        "",
        "**Data**: Ziems et al. (2022) Positive Psychology Frames (CC BY-SA 4.0)",
        "**Dataset**: SALT-NLP/positive_reframing — https://aclanthology.org/2022.acl-long.257/",
        "**License**: CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/",
        "**Prepared**: team member (the research team), 2026-06-05, seed=42",
        "",
        "**Instructions**: For each item, read `context` (negative source) then `candidate_text`.",
        "Apply the R1-R3 rubric (see coding_guide.md). Mark 1 (SRR) or 0 (not SRR) in `PI_SRR_label`.",
        "Add notes in `PI_note` if uncertain.",
        "",
        "---",
        "",
        "| id | context | candidate_text | PI_SRR_label | PI_note |",
        "|----|---------|----------------|:------------:|---------|",
    ]
    for item in items:
        ctx = item["context"].replace("|", "\\|").replace("\n", " ")
        cand = item["candidate_text"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {item['id']} | {ctx} | {cand} |  | |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Attribution")
    lines.append("")
    lines.append("Ziems, C., Li, M., Zhang, A., & Yang, D. (2022). Inducing Positive Perspectives")
    lines.append("with Text Reframing. *ACL 2022*, 3682–3700.")
    lines.append("https://aclanthology.org/2022.acl-long.257/")
    lines.append("License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)")
    return "\n".join(lines)

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)

    rows = load_raw()
    sampled_pos, sampled_neg = sample_items(rows, rng)
    items = build_annotation_rows(sampled_pos, sampled_neg, rng)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Blank CSV (no gold labels)
    blank_csv_path = os.path.join(OUT_DIR, "annotation_sheet_blank.csv")
    with open(blank_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "context", "candidate_text",
                                                "PI_SRR_label", "PI_note"])
        writer.writeheader()
        for item in items:
            writer.writerow({k: item[k] for k in ["id", "context", "candidate_text",
                                                    "PI_SRR_label", "PI_note"]})
    print(f"[saved] {blank_csv_path}")

    # 2. Blank Markdown table
    md_path = os.path.join(OUT_DIR, "annotation_sheet_blank.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_md_table(items))
    print(f"[saved] {md_path}")

    # 3. Sampling log (includes gold labels for later comparison — DO NOT SHARE with PI before annotation)
    log = {
        "description": "Sampling log for SRR single-annotator validation. "
                        "Gold labels included for post-hoc comparison only. "
                        "DO NOT share with PI before annotation is complete.",
        "data_source": {
            "dataset": "SALT-NLP/positive_reframing",
            "paper": "Ziems et al. (2022), ACL 2022",
            "license": "CC BY-SA 4.0",
            "url": "https://aclanthology.org/2022.acl-long.257/",
        },
        "sampling_params": {
            "seed": SEED,
            "n_positive": N_POSITIVE,
            "n_negative": N_NEGATIVE,
            "total": len(items),
            "strategy_targets": STRATEGY_TARGETS,
        },
        "actual_strategy_distribution": dict(Counter(
            item["_strategy"] for item in items if item["_gold_label"]
        )),
        "sample_type_distribution": dict(Counter(
            item["_sample_type"] for item in items if not item["_gold_label"]
        )),
        "items": [
            {
                "id": item["id"],
                "item_id": item["_item_id"],
                "gold_label": item["_gold_label"],
                "sample_type": item["_sample_type"],
                "strategy": item["_strategy"],
                "context": item["context"],
                "candidate_text": item["candidate_text"],
            }
            for item in items
        ],
    }
    log_path = os.path.join(OUT_DIR, "sampling_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[saved] {log_path}  (gold labels included — internal use only)")

    # 4. Coding guide
    guide_path = os.path.join(OUT_DIR, "coding_guide.md")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(CODING_GUIDE_MD)
    print(f"[saved] {guide_path}")

    # Summary report
    gold_pos = sum(1 for i in items if i["_gold_label"])
    gold_neg = sum(1 for i in items if not i["_gold_label"])
    strategy_dist = Counter(i["_strategy"] for i in items if i["_gold_label"])
    type_dist     = Counter(i["_sample_type"] for i in items if not i["_gold_label"])

    print("\n=== Sampling Summary ===")
    print(f"Total items:    {len(items)}")
    print(f"  Positive (SRR=1): {gold_pos}")
    print(f"  Negative (SRR=0): {gold_neg}")
    print(f"Strategy distribution (positive):")
    for k, v in sorted(strategy_dist.items()):
        print(f"  {k}: {v}")
    print(f"Sample type distribution (negative):")
    for k, v in sorted(type_dist.items()):
        print(f"  {k}: {v}")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()
