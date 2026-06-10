"""
srr_realdata_paired_validation.py
==================================
GratiFlow SRR Rubric Classifier — Paired (Context-Aware) Validation on Real Human Data
(Positive Psychology Frames, Ziems et al., ACL 2022)

Dataset: SALT-NLP/positive_reframing (CC BY-SA 4.0)
  Ziems, C., Li, M., Zhang, A., & Yang, D. (2022).
  Inducing Positive Perspectives with Text Reframing.
  ACL 2022, 3682-3700. https://aclanthology.org/2022.acl-long.257/

Attribution: This validation uses data from the above dataset under CC BY-SA 4.0.
  Original data is NOT modified; only sampled for evaluation.

Design change from srr_realdata_validation.py (solo evaluation):
  SOLO:  Only reframed_text (or original_text) was passed to the LLM judge.
         R1 (the negative source) was missing for positive samples where the
         reframed text omitted the original negative event — causing 140/150 FN.
  PAIRED: Both original_text (negative source) AND reframed_text (reframe candidate)
          are passed to the LLM judge as a pair, matching GratiFlow's operational
          context (user's diary negative entry + AI-assisted reinterpretation).
          This is the ecologically valid evaluation setting.

Only change: the user-turn content of the LLM call is replaced from solo text
             to a structured pair showing original → reframed.
             Rubric (R1-R3 / F1-F5), model, seed, sample are IDENTICAL.

Positive (SRR=1 equivalent):
  reframed_text where strategy in {growth, self_affirmation, optimism, thankfulness}
  → Human-written genuine cognitive reframe of original_text

Negative (SRR=0 equivalent):
  original_text  → Raw negative tweet (no reframe; easy negative — presented
                    as BOTH original and "reframed" = same text, no change)
  reframed_text where strategy in {neutralizing, impermanence}
  → Hard negative: mild reframe paired with its original_text

Sample: 150 positive + 150 negative = 300 total (seed=42, same as solo run)

LLM: gpt-5.4-mini
API key: .env (OPENAI_API_KEY, never logged or output)
Protocol: max_completion_tokens=256, no temperature parameter, retry with backoff

Transparency note:
  The solo evaluation (srr_realdata_validation.py) yielded:
    Accuracy=0.520, Precision=0.714, Recall=0.067, F1=0.122
  This paired evaluation tests whether providing the negative context (R1 source)
  enables the rubric to function as intended.

Author: team member (experiment lead, the research team)
Date: 2026-06-05
"""

import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from datasets import load_dataset

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Model / API settings ───────────────────────────────────────────────────────
MODEL = "gpt-5.4-mini"
API_URL = "https://api.openai.com/v1/chat/completions"
MAX_COMPLETION_TOKENS = 256
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2.0

# ── Sampling parameters (IDENTICAL to solo run) ────────────────────────────────
N_POSITIVE = 150
N_NEGATIVE_HARD = 75
N_NEGATIVE_EASY = 75
N_TOTAL = N_POSITIVE + N_NEGATIVE_HARD + N_NEGATIVE_EASY  # 300

POSITIVE_STRATEGIES = {"growth", "self_affirmation", "optimism", "thankfulness"}
HARD_NEG_STRATEGIES = {"neutralizing", "impermanence"}

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "srr_realdata_ziems2022"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "srr_realdata_paired_validation"
EVAL_DIR = BASE_DIR / "evaluation" / "srr_realdata_paired_validation"
PAPER_FIG_DIR = (
    BASE_DIR
    / "paper"
    / "en"
    / "GratiFlow__A_Scaffolding_Fading_Multi_Agent_LLM_for_Positive_Reframing_Skill_Development"
    / "figures"
)

for d in [RAW_DIR, PROCESSED_DIR, EVAL_DIR, PAPER_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── SRR Rubric System Prompt (IDENTICAL to solo run) ──────────────────────────
SRR_JUDGE_SYSTEM_PROMPT = (
    "You are the Affect-Analysis Agent in GratiFlow. "
    "Your task is to judge whether a given REFRAMED text contains a 'spontaneous reframe' "
    "relative to the ORIGINAL negative text provided.\n\n"
    "Respond with ONLY valid JSON (no markdown fences):\n"
    '{"spontaneous_reframe": <boolean>, "srr_reasoning": "<1-2 sentences>"}\n\n'
    "=== RUBRIC: spontaneous_reframe judgment ===\n\n"
    "Definition (operationalized from cognitive reappraisal theory):\n"
    "A 'spontaneous reframe' is TRUE if and only if the REFRAMED text satisfies ALL THREE of:\n"
    "  (R1) The ORIGINAL text contains an explicitly stated negative event or negative emotion (the source).\n"
    "  (R2) The REFRAMED text contains a deliberate reinterpretation that transforms the meaning of that\n"
    "       SAME negative event into a positive, growth-oriented, or silver-lining perspective.\n"
    "  (R3) Evidence that the reframe was generated by the user THEMSELVES, not echoed from\n"
    "       a prior AI response or from a prompt instruction.\n\n"
    "spontaneous_reframe is FALSE if any of the following apply:\n"
    "  (F1) The REFRAMED text simply describes a positive event without connecting it to the negative one.\n"
    "  (F2) The REFRAMED text expresses vague optimism or a coping platitude without specific reinterpretation.\n"
    "  (F3) The user repeats or paraphrases a reframe that the AI previously modeled for them.\n"
    "  (F4) The REFRAMED text only describes the negative event without any positive reinterpretation.\n"
    "  (F5) The positive aspect is about a DIFFERENT event, not a reinterpretation of the negative one.\n\n"
    "=== FEW-SHOT EXAMPLES ===\n\n"
    "--- Example 1: TRUE ---\n"
    "Original: \"My presentation totally bombed today. Everyone looked bored and the Q&A was brutal.\"\n"
    "Reframed: \"My presentation bombed today. But thinking about it, the failure showed me exactly "
    "where I'm underprepared. Now I know what to work on. The failure became a lesson.\"\n"
    '{"spontaneous_reframe": true, "srr_reasoning": "R1: presentation failure in original. '
    'R2: reframed reinterprets same failure as revealing preparation gaps. '
    'R3: self-generated insight with action plan."}\n\n'
    "--- Example 2: TRUE (tentative) ---\n"
    "Original: \"My boss criticized me harshly at work today in front of everyone.\"\n"
    "Reframed: \"My boss criticized me harshly. It stings, but maybe she said it because she expects "
    "more from me than from others.\"\n"
    '{"spontaneous_reframe": true, "srr_reasoning": "R1: harsh criticism from original. '
    "R2: tentative reinterpretation as sign of high expectations for the same event. "
    "R3: self-generated ('maybe'). Tentative but genuine cognitive shift.\"}\n\n"
    "--- Example 3: FALSE (F5 - unrelated positive event) ---\n"
    "Original: \"I missed a deadline and my team is disappointed in me.\"\n"
    "Reframed: \"I missed a deadline at work. But I had a nice lunch with friends today.\"\n"
    '{"spontaneous_reframe": false, "srr_reasoning": "The positive event (lunch) is entirely separate '
    'from the negative one (missed deadline). F5: positive aspect about a different event."}\n\n'
    "--- Example 4: FALSE (F2 - vague optimism) ---\n"
    "Original: \"My experiment failed and I don't know why.\"\n"
    "Reframed: \"The experiment failed. Well, things will somehow work out.\"\n"
    '{"spontaneous_reframe": false, "srr_reasoning": "Vague optimism without reinterpreting the failure. '
    "F2: coping platitude without specific cognitive shift.\"}\n\n"
    "--- Example 5: FALSE (F4 - negative only) ---\n"
    "Original: \"I felt terrible all day. Could not focus on anything.\"\n"
    "Reframed: \"I felt terrible all day. Could not focus on anything.\"\n"
    '{"spontaneous_reframe": false, "srr_reasoning": "No positive reinterpretation present. '
    'F4: negative description only — reframed text identical to original."}\n\n'
    "=== END RUBRIC ===\n\n"
    "Rules:\n"
    "- Apply the rubric strictly. When in doubt, mark spontaneous_reframe as FALSE.\n"
    "- R1, R2, R3 must ALL be met for TRUE.\n"
    "- Use the ORIGINAL text to verify R1 (the negative source is present).\n"
    "- Use the REFRAMED text to verify R2 (the transformation is present).\n"
    "- Do NOT infer intent beyond what is explicitly written."
)


def format_pair_input(original_text: str, reframed_text: str) -> str:
    """
    Format the paired input for the LLM judge.
    Clearly labels original (negative source) and reframed (candidate) text.
    For easy negatives, both fields are the same (no reframe occurred).
    """
    return (
        f"[ORIGINAL — negative source]\n{original_text}\n\n"
        f"[REFRAMED — candidate for spontaneous reframe]\n{reframed_text}"
    )


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    """Load API key from .env. Never log or output the key."""
    env_path = Path(".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise ValueError("OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=your_key")


def call_srr_judge_paired(
    api_key: str, original_text: str, reframed_text: str, item_id: str
) -> dict:
    """
    Call gpt-5.4-mini with SRR rubric prompt for a PAIR of texts.
    original_text: the negative source (R1 evidence)
    reframed_text: the candidate reframe (R2 evidence)
    Returns dict with spontaneous_reframe (bool or None) and srr_reasoning.
    """
    user_content = format_pair_input(original_text, reframed_text)
    messages = [
        {"role": "system", "content": SRR_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            cleaned = content
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
            parsed = json.loads(cleaned)
            return {
                "spontaneous_reframe": bool(parsed.get("spontaneous_reframe", False)),
                "srr_reasoning": parsed.get("srr_reasoning", ""),
                "error": None,
            }
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429 or (status and status >= 500):
                wait = RETRY_WAIT_BASE ** attempt
                print(f"    [{item_id}] retry {attempt}/{MAX_RETRIES} HTTP {status}, wait {wait:.1f}s")
                time.sleep(wait)
            else:
                err_text = e.response.text[:200] if e.response else str(e)
                return {
                    "spontaneous_reframe": None, "srr_reasoning": "",
                    "error": f"HTTP {status}: {err_text}"
                }
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_BASE)
            else:
                return {
                    "spontaneous_reframe": None, "srr_reasoning": "",
                    "error": f"ParseError: {e}"
                }
        except Exception as e:
            wait = RETRY_WAIT_BASE ** attempt
            print(f"    [{item_id}] retry {attempt}/{MAX_RETRIES} Error: {e}, wait {wait:.1f}s")
            time.sleep(wait)

    return {
        "spontaneous_reframe": None, "srr_reasoning": "",
        "error": f"Failed after {MAX_RETRIES} retries"
    }


# ── Data preparation (IDENTICAL sampling as solo run) ──────────────────────────

def parse_strategy_field(strategy_val) -> list:
    """
    Parse the dataset's 'strategy' field (may be string, list-string, or list).
    Returns list of strategy strings, or empty list if unparseable.
    """
    if isinstance(strategy_val, list):
        return [str(s).strip().strip("'\"") for s in strategy_val]
    if not isinstance(strategy_val, str):
        return []
    s = strategy_val.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip().strip("'\"") for x in parsed]
        except Exception:
            pass
        return []
    clean = s.strip("'\"")
    if clean in (
        "growth", "optimism", "self_affirmation", "thankfulness",
        "neutralizing", "impermanence",
    ):
        return [clean]
    return []


def prepare_sample(ds_train) -> pd.DataFrame:
    """
    Sample 150 positive + 75 hard-negative + 75 easy-negative.
    IDENTICAL to srr_realdata_validation.py (same seed, same logic).

    Key difference from solo run: each record now stores BOTH
    original_text_ref (for R1 context) AND the target text
    (reframed_text for pos/hard-neg; original_text for easy-neg).

    For paired evaluation:
    - positive: original_text (R1 source) + reframed_text (R2 candidate)
    - hard_negative: original_text + reframed_text (neutralizing/impermanence)
    - easy_negative: original_text + original_text (no reframe; identical pair)
    """
    import ast

    df = pd.DataFrame(ds_train)
    print(f"\nFull dataset: {len(df)} rows")

    df["strategy_parsed"] = df["strategy"].apply(parse_strategy_field)
    df["strategy_raw"] = df["strategy"]

    df = df[df["strategy_parsed"].apply(len) > 0].reset_index(drop=True)
    print(f"Rows with parseable strategy: {len(df)}")

    all_strategies = []
    for strats in df["strategy_parsed"]:
        all_strategies.extend(strats)
    print("Individual strategy occurrences (from parsed):")
    for strat, cnt in sorted(Counter(all_strategies).items()):
        print(f"  {strat}: {cnt}")

    rng = random.Random(SEED)

    def has_positive(strats):
        return bool(set(strats) & POSITIVE_STRATEGIES)

    def is_hard_negative(strats):
        return (
            len(strats) > 0
            and set(strats).issubset(HARD_NEG_STRATEGIES)
            and not (set(strats) & POSITIVE_STRATEGIES)
        )

    pos_pool = df[df["strategy_parsed"].apply(has_positive)].copy()
    hard_pool = df[df["strategy_parsed"].apply(is_hard_negative)].copy()

    print(f"\nPositive pool (has POSITIVE_STRATEGIES): {len(pos_pool)} rows")
    print(f"Hard-neg pool (HARD_NEG_STRATEGIES only): {len(hard_pool)} rows")

    if len(pos_pool) < N_POSITIVE:
        raise ValueError(f"Insufficient positive pool: {len(pos_pool)} < {N_POSITIVE}")

    N_HARD_ACTUAL = min(N_NEGATIVE_HARD, len(hard_pool))
    if N_HARD_ACTUAL < N_NEGATIVE_HARD:
        print(f"  WARNING: hard_pool {len(hard_pool)} < {N_NEGATIVE_HARD}, using {N_HARD_ACTUAL}.")

    pos_sample = pos_pool.sample(n=N_POSITIVE, random_state=SEED)
    hard_sample = hard_pool.sample(n=N_HARD_ACTUAL, random_state=SEED)

    def primary_strategy_positive(strats):
        for s in ["growth", "self_affirmation", "optimism", "thankfulness"]:
            if s in strats:
                return s
        return strats[0] if strats else "unknown"

    def primary_strategy_negative(strats):
        for s in ["neutralizing", "impermanence"]:
            if s in strats:
                return s
        return strats[0] if strats else "unknown"

    pos_records = []
    for _, row in pos_sample.iterrows():
        pos_records.append({
            "item_id": f"POS_{len(pos_records):04d}",
            # text = reframed_text (the R2 candidate)
            "text": row["reframed_text"],
            # original_text_for_r1 = negative source for paired input
            "original_text_for_r1": row["original_text"],
            "gold_label": True,
            "sample_type": "positive",
            "strategy": primary_strategy_positive(row["strategy_parsed"]),
            "strategy_raw": row["strategy_raw"],
        })

    hard_records = []
    for _, row in hard_sample.iterrows():
        hard_records.append({
            "item_id": f"HNEG_{len(hard_records):04d}",
            "text": row["reframed_text"],
            "original_text_for_r1": row["original_text"],
            "gold_label": False,
            "sample_type": "hard_negative",
            "strategy": primary_strategy_negative(row["strategy_parsed"]),
            "strategy_raw": row["strategy_raw"],
        })

    # Easy-negative: original_text as BOTH original and reframed (no change = no reframe)
    easy_pool = df.drop_duplicates(subset=["original_text"]).copy()
    easy_sample = easy_pool.sample(n=N_NEGATIVE_EASY, random_state=SEED)
    easy_records = []
    for _, row in easy_sample.iterrows():
        easy_records.append({
            "item_id": f"ENEG_{len(easy_records):04d}",
            "text": row["original_text"],
            "original_text_for_r1": row["original_text"],  # same = no reframe occurred
            "gold_label": False,
            "sample_type": "easy_negative",
            "strategy": primary_strategy_negative(row["strategy_parsed"])
                if not has_positive(row["strategy_parsed"])
                else primary_strategy_positive(row["strategy_parsed"]),
            "strategy_raw": row["strategy_raw"],
        })

    all_records = pos_records + hard_records + easy_records
    rng.shuffle(all_records)

    sample_df = pd.DataFrame(all_records)
    print(f"\nSample prepared: {len(sample_df)} items")
    print(f"  Positive (gold=True): {sample_df['gold_label'].sum()}")
    print(f"  Negative (gold=False): {(~sample_df['gold_label']).sum()}")
    print(f"  Hard-neg: {(sample_df['sample_type']=='hard_negative').sum()}")
    print(f"  Easy-neg: {(sample_df['sample_type']=='easy_negative').sum()}")

    return sample_df


# ── Main experiment loop (PAIRED mode) ────────────────────────────────────────

def run_experiment_paired(api_key: str, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply SRR rubric judge to each (original, reframed) pair.
    Returns sample_df with: predicted_label, srr_reasoning_llm, error_flag.
    """
    predicted_labels = []
    srr_reasonings = []
    error_flags = []

    total = len(sample_df)
    print(f"\nRunning paired SRR judge on {total} items (model: {MODEL})...")
    print("  [PAIRED mode] original_text + reframed_text passed together to judge")

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        item_id = row["item_id"]
        original_text = row["original_text_for_r1"]
        reframed_text = row["text"]
        gold = row["gold_label"]

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{total}] {item_id} | gold={gold} | type={row['sample_type']}"
            )

        result = call_srr_judge_paired(api_key, original_text, reframed_text, item_id)
        time.sleep(0.4)  # Rate control: ~2.5 req/s

        predicted_labels.append(result["spontaneous_reframe"])
        srr_reasonings.append(result["srr_reasoning"])
        error_flags.append(result["error"] is not None)

        if result["error"]:
            print(f"    [ERROR] {item_id}: {result['error']}")

    sample_df = sample_df.copy()
    sample_df["predicted_label"] = predicted_labels
    sample_df["srr_reasoning_llm"] = srr_reasonings
    sample_df["error_flag"] = error_flags

    return sample_df


# ── Metrics computation (IDENTICAL to solo run) ────────────────────────────────

def compute_metrics(results_df: pd.DataFrame) -> dict:
    """
    Compute accuracy, precision, recall, F1, confusion matrix.
    Excludes rows with API errors (predicted_label=None).
    """
    valid = results_df[results_df["predicted_label"].notna()].copy()
    valid["predicted_label"] = valid["predicted_label"].astype(bool)
    valid["gold_label"] = valid["gold_label"].astype(bool)

    n_errors = len(results_df) - len(valid)

    tp = int(((valid["gold_label"] == True) & (valid["predicted_label"] == True)).sum())
    fp = int(((valid["gold_label"] == False) & (valid["predicted_label"] == True)).sum())
    tn = int(((valid["gold_label"] == False) & (valid["predicted_label"] == False)).sum())
    fn = int(((valid["gold_label"] == True) & (valid["predicted_label"] == False)).sum())

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    strategy_breakdown = {}
    for strat in sorted(valid["strategy"].unique()):
        strat_df = valid[valid["strategy"] == strat]
        strat_gold_pos = strat_df[strat_df["gold_label"] == True]
        strat_gold_neg = strat_df[strat_df["gold_label"] == False]

        s_tp = int(((strat_df["gold_label"] == True) & (strat_df["predicted_label"] == True)).sum())
        s_fp = int(((strat_df["gold_label"] == False) & (strat_df["predicted_label"] == True)).sum())
        s_tn = int(((strat_df["gold_label"] == False) & (strat_df["predicted_label"] == False)).sum())
        s_fn = int(((strat_df["gold_label"] == True) & (strat_df["predicted_label"] == False)).sum())

        s_total = s_tp + s_fp + s_tn + s_fn
        s_acc = (s_tp + s_tn) / s_total if s_total > 0 else float("nan")
        s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else float("nan")
        s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else float("nan")
        s_f1 = (
            2 * s_prec * s_rec / (s_prec + s_rec)
            if (s_prec + s_rec) > 0 and not (np.isnan(s_prec) or np.isnan(s_rec))
            else float("nan")
        )

        is_positive_strategy = strat in POSITIVE_STRATEGIES
        strategy_breakdown[strat] = {
            "n": s_total,
            "n_gold_positive": len(strat_gold_pos),
            "n_gold_negative": len(strat_gold_neg),
            "gold_label": "positive" if is_positive_strategy else "negative",
            "tp": s_tp, "fp": s_fp, "tn": s_tn, "fn": s_fn,
            "accuracy": round(s_acc, 4) if not np.isnan(s_acc) else None,
            "precision": round(s_prec, 4) if not np.isnan(s_prec) else None,
            "recall_or_detection_rate": round(s_rec, 4) if not np.isnan(s_rec) else None,
            "f1": round(s_f1, 4) if not np.isnan(s_f1) else None,
            "false_positive_rate": round(s_fp / (s_fp + s_tn), 4) if (s_fp + s_tn) > 0 else None,
        }

    type_breakdown = {}
    for stype in ["positive", "hard_negative", "easy_negative"]:
        sub = valid[valid["sample_type"] == stype]
        if len(sub) == 0:
            continue
        sub_gold = sub["gold_label"].values
        sub_pred = sub["predicted_label"].values
        correct = (sub_gold == sub_pred).sum()
        fp_sub = int(((sub_gold == False) & (sub_pred == True)).sum())
        fn_sub = int(((sub_gold == True) & (sub_pred == False)).sum())
        type_breakdown[stype] = {
            "n": len(sub),
            "correct": int(correct),
            "accuracy": round(correct / len(sub), 4),
            "fp": fp_sub,
            "fn": fn_sub,
        }

    return {
        "n_total": total,
        "n_valid": len(valid),
        "n_errors": n_errors,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "strategy_breakdown": strategy_breakdown,
        "sample_type_breakdown": type_breakdown,
    }


# ── Error analysis ─────────────────────────────────────────────────────────────

def analyze_errors(results_df: pd.DataFrame) -> dict:
    """Identify systematic error patterns: FP and FN examples."""
    valid = results_df[results_df["predicted_label"].notna()].copy()
    valid["predicted_label"] = valid["predicted_label"].astype(bool)
    valid["gold_label"] = valid["gold_label"].astype(bool)

    false_positives = valid[(valid["gold_label"] == False) & (valid["predicted_label"] == True)]
    false_negatives = valid[(valid["gold_label"] == True) & (valid["predicted_label"] == False)]

    def sample_errors(df, n=5):
        return [
            {
                "item_id": row["item_id"],
                "sample_type": row["sample_type"],
                "strategy": row["strategy"],
                "original_text": row["original_text_for_r1"][:250],
                "reframed_text": row["text"][:250],
                "srr_reasoning_llm": row["srr_reasoning_llm"],
            }
            for _, row in df.head(n).iterrows()
        ]

    fp_by_type = Counter(false_positives["sample_type"].tolist())
    fn_by_type = Counter(false_negatives["sample_type"].tolist())
    fp_by_strategy = Counter(false_positives["strategy"].tolist())
    fn_by_strategy = Counter(false_negatives["strategy"].tolist())

    return {
        "false_positives": {
            "count": len(false_positives),
            "by_sample_type": dict(fp_by_type),
            "by_strategy": dict(fp_by_strategy),
            "examples": sample_errors(false_positives),
        },
        "false_negatives": {
            "count": len(false_negatives),
            "by_sample_type": dict(fn_by_type),
            "by_strategy": dict(fn_by_strategy),
            "examples": sample_errors(false_negatives),
        },
    }


# ── Visualization ──────────────────────────────────────────────────────────────

def make_confusion_matrix_fig(metrics: dict, save_dir: Path, prefix: str) -> None:
    """Confusion matrix heatmap (colorblind-friendly Blues palette)."""
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["Predicted\nNegative", "Predicted\nPositive"],
        yticklabels=["Actual\nNegative", "Actual\nPositive"],
        linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(
        f"SRR Rubric — Paired (Context-Aware) Evaluation\n"
        f"Acc={metrics['accuracy']:.3f}  P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}",
        fontsize=10, pad=10,
    )
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()

    for out_dir in [save_dir, PAPER_FIG_DIR]:
        png_path = out_dir / f"{prefix}_confusion_matrix.png"
        pdf_path = out_dir / f"{prefix}_confusion_matrix.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")

    plt.close(fig)


def make_strategy_bar_fig(metrics: dict, save_dir: Path, prefix: str) -> None:
    """Detection rate (positive strategies) and FP rate (negative strategies)."""
    sb = metrics["strategy_breakdown"]
    strategies = sorted(sb.keys())
    labels = []
    rates = []
    colors = []

    POSITIVE_COLOR = "#2166ac"
    NEGATIVE_COLOR = "#d6604d"

    for strat in strategies:
        info = sb[strat]
        if info["gold_label"] == "positive":
            rate = info["recall_or_detection_rate"]
            labels.append(f"{strat}\n(detect rate)")
            colors.append(POSITIVE_COLOR)
        else:
            rate = info["false_positive_rate"]
            labels.append(f"{strat}\n(FP rate)")
            colors.append(NEGATIVE_COLOR)
        rates.append(rate if rate is not None else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.55, edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_title(
        "SRR Rubric (Paired): Detection Rate (positive) and FP Rate (negative) by Strategy\n"
        "Data: Positive Psychology Frames, Ziems et al. ACL 2022 (CC BY-SA 4.0)",
        fontsize=9, pad=8,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=POSITIVE_COLOR, label="Positive strategy (higher = better)"),
        mpatches.Patch(color=NEGATIVE_COLOR, label="Negative strategy (lower = better)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    plt.tight_layout()

    for out_dir in [save_dir, PAPER_FIG_DIR]:
        png_path = out_dir / f"{prefix}_strategy_rates.png"
        pdf_path = out_dir / f"{prefix}_strategy_rates.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")

    plt.close(fig)


def make_sample_type_fig(metrics: dict, save_dir: Path, prefix: str) -> None:
    """Accuracy by sample type (positive / hard-neg / easy-neg)."""
    tb = metrics["sample_type_breakdown"]
    types = ["positive", "hard_negative", "easy_negative"]
    labels_map = {
        "positive": "Positive\n(genuine reframe)",
        "hard_negative": "Hard Negative\n(neutralizing/impermanence)",
        "easy_negative": "Easy Negative\n(raw negative tweet)",
    }
    accs = [tb.get(t, {}).get("accuracy", 0.0) for t in types]
    ns = [tb.get(t, {}).get("n", 0) for t in types]
    colors_bar = ["#2166ac", "#f4a582", "#d6604d"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(types))
    bars = ax.bar(x, accs, color=colors_bar, width=0.5, edgecolor="white")

    for bar, acc, n in zip(bars, accs, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}\n(n={n})",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[t] for t in types], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(
        "SRR Rubric (Paired) Accuracy by Sample Type\n"
        "Positive Psychology Frames, Ziems et al. ACL 2022 (CC BY-SA 4.0)",
        fontsize=9, pad=8,
    )
    plt.tight_layout()

    for out_dir in [save_dir, PAPER_FIG_DIR]:
        png_path = out_dir / f"{prefix}_sample_type_accuracy.png"
        pdf_path = out_dir / f"{prefix}_sample_type_accuracy.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")

    plt.close(fig)


def make_comparison_fig(
    solo_metrics: dict, paired_metrics: dict, save_dir: Path, prefix: str
) -> None:
    """
    Grouped bar chart comparing Solo vs Paired evaluation.
    Shows Accuracy / Precision / Recall / F1 side by side.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    solo_vals = [
        solo_metrics["accuracy"],
        solo_metrics["precision"],
        solo_metrics["recall"],
        solo_metrics["f1"],
    ]
    paired_vals = [
        paired_metrics["accuracy"],
        paired_metrics["precision"],
        paired_metrics["recall"],
        paired_metrics["f1"],
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    SOLO_COLOR = "#b2abd2"    # Light purple (colorblind-safe)
    PAIRED_COLOR = "#2166ac"  # Blue

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_solo = ax.bar(x - width / 2, solo_vals, width, label="Solo (no context)", color=SOLO_COLOR)
    bars_paired = ax.bar(x + width / 2, paired_vals, width, label="Paired (with original context)", color=PAIRED_COLOR)

    for bar, val in zip(bars_solo, solo_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    for bar, val in zip(bars_paired, paired_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "SRR Rubric Validation: Solo vs. Paired (Context-Aware) Evaluation\n"
        "Data: Positive Psychology Frames, Ziems et al. ACL 2022 (CC BY-SA 4.0)",
        fontsize=9, pad=8,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()

    for out_dir in [save_dir, PAPER_FIG_DIR]:
        png_path = out_dir / f"{prefix}_solo_vs_paired_comparison.png"
        pdf_path = out_dir / f"{prefix}_solo_vs_paired_comparison.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")

    plt.close(fig)


# ── Save outputs ───────────────────────────────────────────────────────────────

def save_results_csv(results_df: pd.DataFrame) -> Path:
    path = PROCESSED_DIR / "srr_realdata_paired_results_300.csv"
    results_df.to_csv(path, index=False, encoding="utf-8")
    print(f"Results CSV saved: {path}")
    return path


def save_metrics_json(
    metrics: dict, error_analysis: dict, solo_metrics: dict
) -> Path:
    output = {
        "meta": {
            "description": (
                "SRR Rubric Paired (Context-Aware) Validation — "
                "Positive Psychology Frames (Ziems et al., ACL 2022, CC BY-SA 4.0)"
            ),
            "dataset_citation": (
                "Ziems, C., Li, M., Zhang, A., & Yang, D. (2022). "
                "Inducing Positive Perspectives with Text Reframing. "
                "ACL 2022, 3682-3700. https://aclanthology.org/2022.acl-long.257/"
            ),
            "license": "CC BY-SA 4.0 (attribution required)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "seed": SEED,
            "evaluation_mode": "paired",
            "n_positive": N_POSITIVE,
            "n_negative_hard": N_NEGATIVE_HARD,
            "n_negative_easy": N_NEGATIVE_EASY,
            "n_total": N_TOTAL,
            "positive_strategies": sorted(POSITIVE_STRATEGIES),
            "hard_neg_strategies": sorted(HARD_NEG_STRATEGIES),
            "design_change_from_solo": (
                "Solo evaluation passed only reframed_text to the LLM judge, "
                "causing R1 (negative source) to be missing for many positive samples. "
                "Paired evaluation passes original_text + reframed_text together, "
                "matching GratiFlow's operational context."
            ),
            "limitations": [
                "English-language data; GratiFlow targets Japanese users (cross-lingual gap).",
                "Ground truth derived from strategy labels, not independent SRR rubric annotation.",
                "neutralizing/impermanence as hard negatives is a design assumption, not verified by human raters.",
                "English few-shot examples in the rubric prompt may systematically advantage certain text styles.",
                "Easy-negative paired input (original=reframed) is a conservative proxy; real negatives in GratiFlow may differ.",
            ],
        },
        "metrics_paired": metrics,
        "metrics_solo_reference": {
            "note": "Solo evaluation results (srr_realdata_validation.py) for transparent comparison.",
            **solo_metrics,
        },
        "error_analysis": error_analysis,
    }
    path = PROCESSED_DIR / "srr_realdata_paired_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Metrics JSON saved: {path}")
    return path


# ── Entry point ────────────────────────────────────────────────────────────────

# Solo evaluation results for transparent comparison (from srr_realdata_metrics.json)
SOLO_METRICS_REFERENCE = {
    "accuracy": 0.52,
    "precision": 0.7143,
    "recall": 0.0667,
    "f1": 0.122,
    "tp": 10, "fp": 4, "tn": 146, "fn": 140,
}


def main() -> None:
    print("=" * 70)
    print("GratiFlow SRR Rubric — Paired (Context-Aware) Real Data Validation")
    print("Dataset: Positive Psychology Frames (Ziems et al., ACL 2022)")
    print("License: CC BY-SA 4.0 — Attribution required in all outputs")
    print("=" * 70)
    print("\nDesign change: Solo → Paired evaluation")
    print("  Solo (previous):  Only reframed_text passed to LLM judge")
    print("  Paired (this run): original_text + reframed_text passed as a pair")
    print("  Rationale: R1 (negative source) is now explicitly available,")
    print("             matching GratiFlow's operational context (diary entry")
    print("             negative + AI-assisted reinterpretation).")
    print(
        f"\n  Solo reference: Acc={SOLO_METRICS_REFERENCE['accuracy']:.3f}, "
        f"P={SOLO_METRICS_REFERENCE['precision']:.3f}, "
        f"R={SOLO_METRICS_REFERENCE['recall']:.3f}, "
        f"F1={SOLO_METRICS_REFERENCE['f1']:.3f}"
    )

    print("\nLoading API key...")
    api_key = load_api_key()
    print("  API key loaded (not logged).")

    print("\nLoading dataset from HuggingFace: SALT-NLP/positive_reframing ...")
    ds = load_dataset("SALT-NLP/positive_reframing", trust_remote_code=True)
    ds_train = ds["train"]
    print(f"  Loaded {len(ds_train)} examples.")

    # Prepare sample (identical seed/logic to solo run)
    sample_df = prepare_sample(ds_train)

    # Run paired experiment
    results_df = run_experiment_paired(api_key, sample_df)
    save_results_csv(results_df)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results_df)
    error_analysis = analyze_errors(results_df)

    # Print summary
    print("\n" + "=" * 70)
    print("PAIRED EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Mode: PAIRED (original_text + reframed_text)")
    print(f"  Dataset: Positive Psychology Frames (Ziems et al., ACL 2022, CC BY-SA 4.0)")
    print(f"  N={metrics['n_valid']} valid / {metrics['n_total']} total | Errors: {metrics['n_errors']}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    print("\n--- Comparison: Solo vs. Paired ---")
    print(f"  {'Metric':<12} {'Solo':>8} {'Paired':>8} {'Delta':>8}")
    print(f"  {'-'*40}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        solo_val = SOLO_METRICS_REFERENCE[metric]
        paired_val = metrics[metric]
        delta = paired_val - solo_val
        print(f"  {metric.capitalize():<12} {solo_val:>8.4f} {paired_val:>8.4f} {delta:>+8.4f}")

    print("\nStrategy breakdown (paired):")
    for strat, info in metrics["strategy_breakdown"].items():
        label = info["gold_label"]
        if label == "positive":
            print(
                f"  [{strat}] gold={label}, n={info['n']}, "
                f"detect_rate={info['recall_or_detection_rate']}, acc={info['accuracy']}"
            )
        else:
            print(
                f"  [{strat}] gold={label}, n={info['n']}, "
                f"FP_rate={info['false_positive_rate']}, acc={info['accuracy']}"
            )

    print("\nSample type breakdown (paired):")
    for stype, info in metrics["sample_type_breakdown"].items():
        print(
            f"  [{stype}] n={info['n']}, acc={info['accuracy']}, "
            f"FP={info['fp']}, FN={info['fn']}"
        )

    print("\nError analysis:")
    print(f"  False Positives: {error_analysis['false_positives']['count']}")
    print(f"    by type: {error_analysis['false_positives']['by_sample_type']}")
    print(f"    by strategy: {error_analysis['false_positives']['by_strategy']}")
    print(f"  False Negatives: {error_analysis['false_negatives']['count']}")
    print(f"    by type: {error_analysis['false_negatives']['by_sample_type']}")
    print(f"    by strategy: {error_analysis['false_negatives']['by_strategy']}")
    print("=" * 70)

    # Save metrics
    save_metrics_json(metrics, error_analysis, SOLO_METRICS_REFERENCE)

    # Figures
    PREFIX = "srr_realdata_paired"
    print("\nGenerating figures...")
    make_confusion_matrix_fig(metrics, EVAL_DIR, PREFIX)
    make_strategy_bar_fig(metrics, EVAL_DIR, PREFIX)
    make_sample_type_fig(metrics, EVAL_DIR, PREFIX)
    make_comparison_fig(SOLO_METRICS_REFERENCE, metrics, EVAL_DIR, PREFIX)

    print("\n" + "=" * 70)
    print("Paired validation complete. Results saved to:")
    print(f"  Results CSV:     {PROCESSED_DIR}/srr_realdata_paired_results_300.csv")
    print(f"  Metrics JSON:    {PROCESSED_DIR}/srr_realdata_paired_metrics.json")
    print(f"  Figures (eval):  {EVAL_DIR}/")
    print(f"  Figures (paper): {PAPER_FIG_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
