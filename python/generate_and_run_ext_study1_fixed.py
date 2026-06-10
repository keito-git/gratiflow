"""
generate_and_run_ext_study1_fixed.py
======================================
GratiFlow ext_study1_fixed — Multi-Turn Loop Simulation (Corrected)

Changes from ext_study1 (per the research team diagnosis 2026-06-05):
  1. Persona parameters calibrated for scaffold transition within 14 sessions.
     (p_attempt_base and latent_skill_0 raised; alpha adjusted)
  2. update_observed_skill_v2(): did_attempt=False sessions excluded from
     moving average (treated as "no observation", not "skill=0").
  3. verify_scaffold_transitions() added as post-hoc assertion per persona.
  4. Output prefix: ext_study1_fixed (ext_study1 is NOT overwritten).

Dry-run gate (run before this script):
  PYTHONHASHSEED=0 python python/dry_run_ext_study1_fixed.py
  → Must show: transitions >= 3/10 AND null calibration PASS.

Design unchanged from ext_study1:
  - Condition leakage prevention (assert_no_condition_leakage)
  - RNG: persona_hash + condition_hash (different seeds per condition — correct)
  - Pre-committed analysis: delta-SRR = mean(S8-14) - mean(S1-7)
  - Direction threshold: A > B in >= 6/10 personas
  - Model: gpt-5.4-mini, max_completion_tokens, no temperature

Author: team member (experiment lead, the research team)
Date: 2026-06-05
Calibration: the research team (2026-06-05) — ext_study1_diagnosis_and_fix.md
"""

import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── PYTHONHASHSEED check ──────────────────────────────────────────────────────
pythonhashseed = os.environ.get("PYTHONHASHSEED")
if pythonhashseed != "0":
    warnings.warn(
        f"PYTHONHASHSEED is '{pythonhashseed}' (expected '0'). "
        "RNG seeds use hashlib.sha256 (PYTHONHASHSEED-independent), "
        "but for full reproducibility run: PYTHONHASHSEED=0 python generate_and_run_ext_study1_fixed.py",
        UserWarning,
        stacklevel=1,
    )

sys.path.insert(0, str(Path(__file__).parent))
from latent_skill_model import (
    SCAFFOLD_ATTEMPT_MULTIPLIER,
    compute_attempt_probability,
    compute_attempt_success_probability,
    make_attempt_description,
    make_previous_summary,
    make_reframe_instruction,
    sample_neg_count,
    update_latent_skill,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "gpt-5.4-mini"
EXPERIMENT_SEED = 42
SYNTHETIC_USER_SEED = 2026
API_URL = "https://api.openai.com/v1/chat/completions"

MAX_COMPLETION_TOKENS_ENTRY = 512
MAX_COMPLETION_TOKENS_SAVORING_REPLY = 300
MAX_COMPLETION_TOKENS_REFRAME_ATTEMPT = 400
MAX_COMPLETION_TOKENS_AGENT = 1024
MAX_COMPLETION_TOKENS_AFFECT = 800

SCAFFOLD_THRESHOLDS = {"high": 0.35, "mid": 0.65}
CURRICULUM_THRESHOLDS = [0.25, 0.50, 0.75]
MOVING_AVG_WINDOW = 5

MAX_RETRIES = 3
RETRY_WAIT_BASE = 2.0

N_SESSIONS = 14
CONDITIONS = ["adaptive-fading", "fixed-high"]

BASE_DIR = Path(__file__).parent.parent

# Output prefix: ext_study1_fixed (never overwrites ext_study1)
DATA_PROCESSED_FIXED = BASE_DIR / "data" / "processed" / "ext_study1_fixed"
DATA_RAW = BASE_DIR / "data" / "raw"
INSTRUMENT_VALIDATION_PATH = (
    BASE_DIR / "data" / "processed" / "experiments_v2_1"
    / "instrument_validation" / "validation_results.json"
)

FORBIDDEN_PROMPT_STRINGS = [
    "adaptive-fading",
    "fixed-high",
    "fading",
    "fixed",
    "condition A",
    "condition B",
    "scaffoldLevel",
    "scaffold_level",
    "high scaffold",
    "low scaffold",
    "mid scaffold",
]

# ── Calibrated persona parameters (the research team 2026-06-05) ───────────────────────
# Rationale: raise p_attempt_base and latent_skill_0 so that observed_skill
# can reach 0.35 (mid threshold) within 14 sessions for multiple personas.
# alpha raised slightly to enable skill growth under increased attempt frequency.
# Direction of A vs B is NOT predetermined — reported honestly.

PERSONAS_FIXED = [
    {
        "id": "P1",
        "label": "初心者・着実成長",
        "latent_skill_0": 0.10,
        "alpha": 0.10,
        "alpha_passive": 0.02,
        "beta": 0.01,
        "p_attempt_base": 0.30,
        "neg_tendency": 1.5,
        "description": "A university student who is a complete beginner in positive reframing. Learns steadily. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P2",
        "label": "初心者・停滞型",
        "latent_skill_0": 0.08,
        "alpha": 0.06,
        "alpha_passive": 0.01,
        "beta": 0.02,
        "p_attempt_base": 0.20,
        "neg_tendency": 2.0,
        "description": "A university student with low learning capacity. Struggles with reframing. May not transition. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P3",
        "label": "中級・安定成長",
        "latent_skill_0": 0.25,
        "alpha": 0.08,
        "alpha_passive": 0.02,
        "beta": 0.01,
        "p_attempt_base": 0.40,
        "neg_tendency": 1.0,
        "description": "A university student with some prior skill. Grows steadily. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P4",
        "label": "初心者・高応答型",
        "latent_skill_0": 0.10,
        "alpha": 0.12,
        "alpha_passive": 0.03,
        "beta": 0.01,
        "p_attempt_base": 0.35,
        "neg_tendency": 1.5,
        "description": "A university student who responds strongly to scaffolding. High alpha. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P5",
        "label": "中級・慎重型",
        "latent_skill_0": 0.18,
        "alpha": 0.07,
        "alpha_passive": 0.015,
        "beta": 0.015,
        "p_attempt_base": 0.30,
        "neg_tendency": 1.5,
        "description": "A university student with moderate prior skill but a cautious disposition. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P6",
        "label": "上級・高自律型",
        "latent_skill_0": 0.35,
        "alpha": 0.08,
        "alpha_passive": 0.015,
        "beta": 0.01,
        "p_attempt_base": 0.50,
        "neg_tendency": 0.8,
        "description": "A university student with higher initial skill. Anchor persona for full high→mid→low range. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P7",
        "label": "初心者・高ネガ傾向",
        "latent_skill_0": 0.08,
        "alpha": 0.08,
        "alpha_passive": 0.02,
        "beta": 0.015,
        "p_attempt_base": 0.22,
        "neg_tendency": 2.5,
        "description": "A university student with very high negative event tendency and low initial skill. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P8",
        "label": "中級・揺れ型",
        "latent_skill_0": 0.15,
        "alpha": 0.10,
        "alpha_passive": 0.01,
        "beta": 0.03,
        "p_attempt_base": 0.35,
        "neg_tendency": 1.8,
        "description": "A university student with variable performance: high alpha, relatively high beta. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P9",
        "label": "初心者・受動観察型",
        "latent_skill_0": 0.10,
        "alpha": 0.08,
        "alpha_passive": 0.04,
        "beta": 0.01,
        "p_attempt_base": 0.18,
        "neg_tendency": 1.2,
        "description": "A university student who tends to observe AI passively but benefits from observation. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
    {
        "id": "P10",
        "label": "中級・急成長型",
        "latent_skill_0": 0.15,
        "alpha": 0.14,
        "alpha_passive": 0.02,
        "beta": 0.01,
        "p_attempt_base": 0.42,
        "neg_tendency": 1.0,
        "description": "A university student with high learning rate who rapidly internalizes reframing skill. Calibrated v2.",
        "_note": "v2: calibrated for scaffold transition within 14 sessions (the research team, 2026-06-05)",
    },
]


# ── Affect-Analysis system prompt (unchanged from ext_study1) ─────────────────

AFFECT_SYSTEM_PROMPT = (
    "You are the Affect-Analysis Agent in GratiFlow.\n"
    "Analyze the user's journal entry and respond with ONLY valid JSON (no markdown fences).\n\n"
    "Required JSON schema:\n"
    "{\n"
    "  \"mood\": <number 1–10, overall mood score>,\n"
    "  \"pos_count\": <integer, number of genuinely positive expressions the user independently produced>,\n"
    "  \"neg_count\": <integer, number of negative expressions or negative events detected>,\n"
    "  \"reframe_count\": <integer, number of negative events that the user spontaneously reframed. Must be <= neg_count>,\n"
    "  \"spontaneous_reframe\": <boolean, see rubric below>,\n"
    "  \"srr_reasoning\": \"<string, 1-2 sentence justification for the spontaneous_reframe decision>\",\n"
    "  \"keywords\": [<string>, ...],  // 3-6 key emotional words (Japanese)\n"
    "  \"summary\": \"<string>\"         // 1-sentence Japanese summary of the user's emotional state\n"
    "}\n\n"
    "=== RUBRIC: spontaneous_reframe judgment ===\n\n"
    "Definition (operationalized from cognitive reappraisal theory):\n"
    "A 'spontaneous reframe' is TRUE if and only if the user's text contains ALL THREE of:\n"
    "  (R1) An explicitly stated negative event or negative emotion (the source).\n"
    "  (R2) A deliberate reinterpretation that transforms the meaning of that SAME negative event\n"
    "       into a positive, growth-oriented, or silver-lining perspective (the reframe).\n"
    "  (R3) Evidence that the reframe was generated by the user THEMSELVES, not echoed from\n"
    "       a prior AI response or from a prompt instruction.\n\n"
    "spontaneous_reframe is FALSE if any of the following apply:\n"
    "  (F1) The user simply describes a positive event without connecting it to a negative one.\n"
    "  (F2) The user expresses vague optimism or a coping platitude without specific reinterpretation.\n"
    "  (F3) The user repeats or paraphrases a reframe that the AI previously modeled for them.\n"
    "  (F4) The user only describes the negative event without any positive reinterpretation.\n"
    "  (F5) The positive aspect is about a DIFFERENT event, not a reinterpretation of the negative one.\n\n"
    "=== FEW-SHOT EXAMPLES ===\n\n"
    "--- Example 1: TRUE ---\n"
    "Entry: \"今日のプレゼンで頭が真っ白になって失敗した。でも、考えてみると、この失敗のおかげで自分の準備不足がはっきりわかった。次回は練習を3回以上して臨もうと思う。\"\n"
    "Judgment: spontaneous_reframe = true\n\n"
    "--- Example 2: TRUE (tentative) ---\n"
    "Entry: \"バイト先で店長に怒られた。よく考えると店長は自分に期待してくれているからこそ厳しく言ってくれたのかもしれない。\"\n"
    "Judgment: spontaneous_reframe = true\n\n"
    "--- Example 3: FALSE (F5 — separate positive) ---\n"
    "Entry: \"レポートの締め切りに追われて大変だった。でも放課後に友達とカフェに行けて楽しかった。\"\n"
    "Judgment: spontaneous_reframe = false\n\n"
    "--- Example 4: FALSE (F2 — vague optimism) ---\n"
    "Entry: \"実験がうまくいかなくて落ち込んだ。まあ、なんとかなるだろう。\"\n"
    "Judgment: spontaneous_reframe = false\n\n"
    "--- Example 5: FALSE (F3 — AI echo) ---\n"
    "Entry: \"昨日AIが『失敗は成長の種』と教えてくれた。今日テストで悪い点を取ったけど、成長の種だと思うことにする。\"\n"
    "Judgment: spontaneous_reframe = false\n\n"
    "=== END RUBRIC ===\n\n"
    "Rules:\n"
    "- Count only expressions in the USER's text, not the AI's previous messages.\n"
    "- Always provide srr_reasoning to justify your spontaneous_reframe judgment.\n"
    "- When in doubt, mark spontaneous_reframe as FALSE. All of R1, R2, R3 must be met.\n"
    "- reframe_count must be an integer >= 0 and <= neg_count."
)

SYSTEM_PROMPTS = {
    "savoring": (
        "You are the Savoring Agent in GratiFlow, a well-being coaching AI for students.\n"
        "Your role is to help the user vividly re-experience and deepen their appreciation of positive events they described today.\n\n"
        "Guidelines:\n"
        "- Ask one warm, curious follow-up question that helps the user 'taste' the good moment more fully.\n"
        "- Focus on sensory details, emotions, and the people involved.\n"
        "- Keep your response short (2-4 sentences + one question).\n"
        "- Use a gentle, encouraging Japanese tone.\n"
        "- Do NOT give advice or reframe negatives at this stage.\n"
        "- Output in Japanese."
    ),
    "reframing_high": (
        "You are the Reframing-Coach Agent in GratiFlow, a well-being coaching AI for students.\n"
        "The user is a beginner at positive reframing. Use HIGH scaffolding (modeling strategy):\n"
        "- First, gently acknowledge the negative expression the user wrote.\n"
        "- Then explicitly model a positive reframing: 「例えばこう捉えることもできます：「…」」\n"
        "- Explain briefly WHY this reframe is valid (1 sentence).\n"
        "- Invite the user to try expressing their own version.\n"
        "- Keep it warm and non-judgmental. Output in Japanese."
    ),
    "reframing_mid": (
        "You are the Reframing-Coach Agent in GratiFlow, a well-being coaching AI for students.\n"
        "The user has some reframing skill. Use MID scaffolding (partial hint strategy):\n"
        "- Acknowledge the negative expression briefly.\n"
        "- Give a partial hint that points toward a positive angle WITHOUT completing the reframe.\n"
        "  e.g. 「この出来事、別の角度から見ると「○○の機会」とも言えそうですね。どう思いますか？」\n"
        "- Do NOT spell out the full reframe; let the user complete it.\n"
        "- Output in Japanese."
    ),
    "reframing_low": (
        "You are the Reframing-Coach Agent in GratiFlow, a well-being coaching AI for students.\n"
        "The user is becoming skilled at positive reframing. Use LOW scaffolding (fading strategy):\n"
        "- Acknowledge what they wrote with one encouraging sentence.\n"
        "- Simply invite them to reframe it themselves: 「自分なりのポジティブな見方を書いてみてください。」\n"
        "- Do NOT provide any hints or examples.\n"
        "- Trust the user's growing ability. Output in Japanese."
    ),
    "affect": AFFECT_SYSTEM_PROMPT,
    "srr_feedback": (
        "You are the SRR-Feedback Agent in GratiFlow.\n"
        "You have just seen the user's reframe attempt and an assessment of its quality.\n"
        "Provide 2-4 sentences of warm, specific feedback in Japanese:\n"
        "- If spontaneous_reframe=True: celebrate the specific aspect that made it a genuine reframe.\n"
        "- If spontaneous_reframe=False (echo): gently note the similarity to the AI example and encourage\n"
        "  the user to find their own angle next time.\n"
        "- If spontaneous_reframe=False (no reframe): encourage without judgment; note that practice builds skill.\n"
        "- End with a brief encouragement for the next session.\n"
        "- Do NOT reveal technical terms (SRR, scaffold_level, condition).\n"
        "- Output in Japanese."
    ),
    "curriculum": (
        "You are the Curriculum-Progress Agent in GratiFlow.\n"
        "Based on the user's skill score and session history, generate a brief progress update message.\n\n"
        "Curriculum stages:\n"
        "  0: Savoring (味わう) — focus on positive events\n"
        "  1: Gratitude (感謝) — recognize who/what helped\n"
        "  2: Reframing (捉え直し) — turn negatives into growth opportunities\n"
        "  3: Future-self Optimism (楽観) — project positive future\n\n"
        "Your message should:\n"
        "- Celebrate specific progress the user made today (1 sentence).\n"
        "- Hint at what to focus on next session (1 sentence).\n"
        "- Keep it brief (2-3 sentences total) and warm.\n"
        "- Output in Japanese."
    ),
}


# ── Scaffold and curriculum helpers ──────────────────────────────────────────

def get_scaffold_level_from_skill(s: float) -> str:
    if s < SCAFFOLD_THRESHOLDS["high"]:
        return "high"
    if s < SCAFFOLD_THRESHOLDS["mid"]:
        return "mid"
    return "low"


def get_curriculum_stage(s: float) -> int:
    if s < CURRICULUM_THRESHOLDS[0]:
        return 0
    if s < CURRICULUM_THRESHOLDS[1]:
        return 1
    if s < CURRICULUM_THRESHOLDS[2]:
        return 2
    return 3


# ── update_observed_skill_v2 (KEY FIX: exclude did_attempt=False sessions) ────

def update_observed_skill_v2(
    session_history: list,
    reframe_count: int,
    neg_count: int,
    did_attempt: bool,
    current_observed_skill: float,
) -> float:
    """
    v2: did_attempt=False sessions are treated as 'no observation' and
    excluded from the moving average. This prevents skill underestimation
    when the user simply did not get a practice opportunity.

    Rationale: observed_skill tracks demonstrated reframing ability.
    Sessions where the user did not attempt provide no evidence about
    their current ability. Treating them as session_rate=0 underestimates
    the skill systematically. Analogous to missing data exclusion in
    psychometric moving averages.

    Args:
        session_history:        list of prior session records
        reframe_count:          successful reframes this session
        neg_count:              negative events detected in reframe text
        did_attempt:            whether user attempted self-reframing
        current_observed_skill: observed_skill before this session

    Returns:
        updated observed_skill in [0.0, 1.0]
    """
    if not did_attempt or neg_count == 0:
        # No observation this session: keep current observed_skill unchanged
        return current_observed_skill

    session_rate = min(reframe_count / neg_count, 1.0)

    # Collect rates from recent sessions where did_attempt=True
    recent_rates = []
    for s in session_history[-MOVING_AVG_WINDOW:]:
        if s.get("did_attempt_gt", False) and s.get("spontaneous_rate") is not None:
            recent_rates.append(s["spontaneous_rate"])
    recent_rates.append(session_rate)

    if not recent_rates:
        return current_observed_skill

    avg = sum(recent_rates) / len(recent_rates)
    return max(0.0, min(1.0, avg))


# ── Scaffold transition post-hoc verifier ─────────────────────────────────────

def verify_scaffold_transitions(session_records: list, condition: str, pid: str) -> dict:
    """
    Post-hoc verification: count scaffold transitions for adaptive-fading.
    Warns if no transitions occurred.
    Does NOT abort — result is reported honestly per the research team design principle.
    """
    if condition != "adaptive-fading":
        return {
            "condition": condition,
            "pid": pid,
            "note": "fixed-high: scaffold always 'high' by design",
            "transitions": 0,
            "unique_levels": ["high"],
            "level_sequence": ["high"] * N_SESSIONS,
        }

    levels = [s["scaffold_level"] for s in session_records]
    transitions = sum(1 for i in range(1, len(levels)) if levels[i] != levels[i - 1])
    unique_levels = list(dict.fromkeys(levels))  # preserve order, deduplicate

    if transitions == 0:
        print(f"  [WARNING] No scaffold transitions for {pid} (always {levels[0]}).")
        print(f"  The fading mechanism was NOT exercised for this persona.")
    else:
        print(f"  [OK] {transitions} scaffold transition(s) for {pid}. Levels: {unique_levels}")

    return {
        "condition": condition,
        "pid": pid,
        "transitions": transitions,
        "unique_levels": unique_levels,
        "level_sequence": levels,
    }


# ── Circular-bias prevention ──────────────────────────────────────────────────

def assert_no_condition_leakage(prompt_text: str) -> None:
    lower_text = prompt_text.lower()
    for forbidden in FORBIDDEN_PROMPT_STRINGS:
        if forbidden.lower() in lower_text:
            raise AssertionError(
                f"CONDITION LEAKAGE DETECTED: '{forbidden}' found in generation prompt.\n"
                f"Prompt excerpt: ...{prompt_text[max(0, lower_text.find(forbidden.lower())-50):lower_text.find(forbidden.lower())+100]}..."
            )


def deterministic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


# ── API helper ────────────────────────────────────────────────────────────────

def load_api_key() -> str:
    env_path = Path(".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    raise ValueError("OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=your_key")


def call_openai(
    api_key: str,
    messages: list,
    max_tokens: int,
    seed: Optional[int] = None,
) -> dict:
    """
    Call gpt-5.4-mini with retry/backoff.
    - max_completion_tokens (not max_tokens)
    - temperature NOT sent (not supported by gpt-5.4-mini)
    """
    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

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
            return {"content": content, "raw_response": data, "error": None}
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429 or (status and status >= 500):
                wait = RETRY_WAIT_BASE ** attempt
                print(f"      [retry {attempt}/{MAX_RETRIES}] HTTP {status}, wait {wait:.1f}s")
                time.sleep(wait)
            else:
                err_text = e.response.text if e.response else str(e)
                return {"content": None, "raw_response": None, "error": f"HTTP {status}: {err_text}"}
        except Exception as e:
            wait = RETRY_WAIT_BASE ** attempt
            print(f"      [retry {attempt}/{MAX_RETRIES}] Error: {e}, wait {wait:.1f}s")
            time.sleep(wait)

    return {"content": None, "raw_response": None, "error": f"Failed after {MAX_RETRIES} retries"}


def parse_affect_json(content: str) -> Optional[dict]:
    if not content:
        return None
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        data = json.loads(cleaned)
        required = ["mood", "pos_count", "neg_count", "spontaneous_reframe", "keywords", "summary"]
        if all(k in data for k in required):
            if "srr_reasoning" not in data:
                data["srr_reasoning"] = ""
            if "reframe_count" not in data:
                data["reframe_count"] = 1 if data.get("spontaneous_reframe", False) else 0
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ── Synthetic user prompt builders (unchanged from ext_study1) ────────────────

def build_entry_generation_prompt(
    persona: dict,
    session_num: int,
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    neg_count: int,
    previous_summary: str,
) -> tuple[str, str]:
    attempt_desc = make_attempt_description(did_attempt, attempt_success, latent_skill)
    reframe_instr = make_reframe_instruction(did_attempt, attempt_success, neg_count)

    system_prompt = (
        "You are a simulation engine generating synthetic university student journal entries "
        "for a research experiment. These entries are SYNTHETIC and do not represent any real person. "
        "Generate realistic-sounding Japanese journal entries that match the given persona and internal state exactly."
    )

    user_message = (
        f"You are simulating a university student writing a daily gratitude journal.\n\n"
        f"Persona: {persona['label']} — {persona['description']}\n"
        f"Session number: {session_num} of {N_SESSIONS}\n\n"
        f"=== Internal state (NOT visible to the student) ===\n"
        f"Current skill level: {latent_skill:.2f} (scale 0.0-1.0; 0=complete beginner, 1=expert)\n"
        f"Self-attempt behavior: {attempt_desc}\n\n"
        f"=== Previous session context ===\n"
        f"{previous_summary}\n\n"
        f"=== Today's entry requirements ===\n"
        f"- Include 1-3 positive events from today.\n"
        f"- Include {neg_count} negative events or negative framings.\n"
        f"- {reframe_instr}\n\n"
        f"The entry should feel natural, like a real student's daily reflection.\n"
        f"Output ONLY the journal entry text in Japanese. Do not include any explanation."
    )

    assert_no_condition_leakage(system_prompt)
    assert_no_condition_leakage(user_message)

    return system_prompt, user_message


def build_savoring_reply_prompt(
    persona: dict,
    session_num: int,
    latent_skill: float,
    journal_entry: str,
    savoring_response: str,
) -> tuple[str, str]:
    system_prompt = (
        "You are a simulation engine generating synthetic university student responses "
        "for a research experiment. These responses are SYNTHETIC and do not represent any real person. "
        "Generate a realistic Japanese response that matches the given persona."
    )

    user_message = (
        f"You are simulating a university student responding to an AI's savoring question.\n\n"
        f"Persona: {persona['label']} — {persona['description']}\n"
        f"Session number: {session_num} of {N_SESSIONS}\n"
        f"Skill level: {latent_skill:.2f}\n\n"
        f"The student wrote this journal entry:\n\"{journal_entry}\"\n\n"
        f"The AI then asked this savoring question:\n\"{savoring_response}\"\n\n"
        f"Write the student's response (2-4 sentences in Japanese). "
        f"The student should elaborate on a positive moment from their entry — "
        f"adding a sensory detail, emotion, or interpersonal aspect. "
        f"Output ONLY the student's reply in Japanese."
    )

    assert_no_condition_leakage(system_prompt)
    assert_no_condition_leakage(user_message)

    return system_prompt, user_message


def build_reframe_attempt_prompt(
    persona: dict,
    session_num: int,
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    neg_count: int,
    journal_entry: str,
    reframing_coach_response: str,
) -> tuple[str, str]:
    attempt_desc = make_attempt_description(did_attempt, attempt_success, latent_skill)
    reframe_instr = make_reframe_instruction(did_attempt, attempt_success, neg_count)

    system_prompt = (
        "You are a simulation engine generating synthetic university student self-reframe attempts "
        "for a research experiment. These attempts are SYNTHETIC and do not represent any real person. "
        "Generate a realistic Japanese text that faithfully reflects the student's internal state."
    )

    user_message = (
        f"You are simulating a university student writing their own reframe attempt.\n\n"
        f"Persona: {persona['label']} — {persona['description']}\n"
        f"Session number: {session_num} of {N_SESSIONS}\n\n"
        f"=== Internal state (NOT visible to the student) ===\n"
        f"Current skill level: {latent_skill:.2f}\n"
        f"Self-attempt behavior: {attempt_desc}\n\n"
        f"=== Context ===\n"
        f"The student wrote this journal entry:\n\"{journal_entry}\"\n\n"
        f"The AI coach then provided this guidance:\n\"{reframing_coach_response}\"\n\n"
        f"=== Reframe attempt requirement ===\n"
        f"- {reframe_instr}\n"
        f"- Write 2-5 sentences in Japanese from the student's perspective.\n"
        f"- If the student IS attempting (did_attempt=True), write their self-reframe — "
        f"it should be their OWN perspective, NOT copied from the AI's response above.\n"
        f"- If the student is NOT attempting, write a brief acknowledgment without reframing.\n"
        f"Output ONLY the student's reframe attempt text in Japanese."
    )

    assert_no_condition_leakage(system_prompt)
    assert_no_condition_leakage(user_message)

    return system_prompt, user_message


# ── Multi-turn loop session (KEY FIX: update_observed_skill_v2) ───────────────

def run_loop_session(
    api_key: str,
    persona: dict,
    session_num: int,
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    neg_count_sampled: int,
    previous_summary: str,
    session_history: list,
    current_observed_skill: float,
    condition: str,
    rng: random.Random,
) -> tuple[dict, dict]:
    """
    Run one full multi-turn loop session per CONVERSATIONAL_LOOP_SPEC.md §2.2.
    KEY CHANGE: uses update_observed_skill_v2() (excludes did_attempt=False sessions).
    """
    pid = persona["id"]
    timestamp = datetime.now(timezone.utc).isoformat()

    if condition == "fixed-high":
        scaffold_level = "high"
    elif condition == "adaptive-fading":
        scaffold_level = get_scaffold_level_from_skill(current_observed_skill)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    stage = get_curriculum_stage(current_observed_skill)
    print(f"      obs_s={current_observed_skill:.3f} → scaffold={scaffold_level}, stage={stage}")

    raw_responses = {}

    # T1: Generate synthetic journal entry
    sys_prompt_entry, usr_msg_entry = build_entry_generation_prompt(
        persona=persona,
        session_num=session_num,
        latent_skill=latent_skill,
        did_attempt=did_attempt,
        attempt_success=attempt_success,
        neg_count=neg_count_sampled,
        previous_summary=previous_summary,
    )
    entry_result = call_openai(
        api_key,
        [{"role": "system", "content": sys_prompt_entry}, {"role": "user", "content": usr_msg_entry}],
        MAX_COMPLETION_TOKENS_ENTRY,
        seed=SYNTHETIC_USER_SEED,
    )
    if entry_result["error"] or not entry_result["content"]:
        journal_entry = f"[生成エラー] ペルソナ: {pid}, セッション: {session_num}"
        print(f"      [WARNING] Journal entry generation error: {entry_result['error']}")
    else:
        journal_entry = entry_result["content"]
    raw_responses["journal_entry_gen"] = entry_result.get("raw_response")
    time.sleep(0.4)

    # T2: Affect-Analysis on journal entry
    affect_entry_result = call_openai(
        api_key,
        [{"role": "system", "content": SYSTEM_PROMPTS["affect"]}, {"role": "user", "content": journal_entry}],
        MAX_COMPLETION_TOKENS_AFFECT,
    )
    affect_entry_data = parse_affect_json(affect_entry_result["content"])
    if affect_entry_data is None:
        print(f"      [WARNING] Affect-Analysis (entry) parse failed for {pid} S{session_num}. Fallback.")
        affect_entry_data = {
            "mood": 5, "pos_count": 0, "neg_count": max(1, neg_count_sampled),
            "reframe_count": 0, "spontaneous_reframe": False,
            "srr_reasoning": "[parse error fallback — entry affect]",
            "keywords": ["不明"], "summary": "感情分析に失敗しました（フォールバック）。", "_fallback": True,
        }
    raw_responses["affect_entry"] = affect_entry_result.get("raw_response")
    neg_count_detected_entry = int(affect_entry_data.get("neg_count", 0))
    mood = int(affect_entry_data.get("mood", 5))
    time.sleep(0.3)

    # T3: Savoring Agent + savoring reply
    sav_result = call_openai(
        api_key,
        [{"role": "system", "content": SYSTEM_PROMPTS["savoring"]}, {"role": "user", "content": journal_entry}],
        MAX_COMPLETION_TOKENS_AGENT,
    )
    savoring_response = sav_result["content"] or "[savoring error]"
    raw_responses["savoring_agent"] = sav_result.get("raw_response")
    time.sleep(0.4)

    sys_sav_reply, usr_sav_reply = build_savoring_reply_prompt(
        persona=persona, session_num=session_num, latent_skill=latent_skill,
        journal_entry=journal_entry, savoring_response=savoring_response,
    )
    sav_reply_result = call_openai(
        api_key,
        [{"role": "system", "content": sys_sav_reply}, {"role": "user", "content": usr_sav_reply}],
        MAX_COMPLETION_TOKENS_SAVORING_REPLY,
        seed=SYNTHETIC_USER_SEED,
    )
    savoring_reply_text = sav_reply_result["content"] or "[savoring reply error]"
    raw_responses["savoring_reply_gen"] = sav_reply_result.get("raw_response")
    time.sleep(0.4)

    # T4-T7: Reframing loop (only if negatives detected)
    reframing_response = None
    reframe_attempt_text = None
    affect_reframe_data = None
    srr_reasoning = ""
    reframe_count = 0
    neg_count_detected_reframe = 0
    is_echo_detected = False
    srr_feedback_response = None

    if neg_count_detected_entry > 0:
        # T4: Reframing-Coach
        ref_prompt_key = f"reframing_{scaffold_level}"
        ref_result = call_openai(
            api_key,
            [{"role": "system", "content": SYSTEM_PROMPTS[ref_prompt_key]}, {"role": "user", "content": journal_entry}],
            MAX_COMPLETION_TOKENS_AGENT,
        )
        reframing_response = ref_result["content"] or "[reframing error]"
        raw_responses["reframing_coach"] = ref_result.get("raw_response")
        time.sleep(0.4)

        # T5: Synthetic reframe attempt
        sys_reframe, usr_reframe = build_reframe_attempt_prompt(
            persona=persona, session_num=session_num, latent_skill=latent_skill,
            did_attempt=did_attempt, attempt_success=attempt_success,
            neg_count=neg_count_detected_entry,
            journal_entry=journal_entry, reframing_coach_response=reframing_response,
        )
        reframe_attempt_result = call_openai(
            api_key,
            [{"role": "system", "content": sys_reframe}, {"role": "user", "content": usr_reframe}],
            MAX_COMPLETION_TOKENS_REFRAME_ATTEMPT,
            seed=SYNTHETIC_USER_SEED,
        )
        reframe_attempt_text = reframe_attempt_result["content"] or "[reframe attempt error]"
        raw_responses["reframe_attempt_gen"] = reframe_attempt_result.get("raw_response")
        time.sleep(0.4)

        # T6: Affect-Analysis on reframe text (SRR + is_echo)
        reframe_analysis_user_content = (
            f"[AI coaching response the student saw]\n{reframing_response}\n\n"
            f"[Student's reframe attempt]\n{reframe_attempt_text}"
        )
        affect_reframe_result = call_openai(
            api_key,
            [{"role": "system", "content": SYSTEM_PROMPTS["affect"]}, {"role": "user", "content": reframe_analysis_user_content}],
            MAX_COMPLETION_TOKENS_AFFECT,
        )
        affect_reframe_data = parse_affect_json(affect_reframe_result["content"])
        if affect_reframe_data is None:
            print(f"      [WARNING] Affect-Analysis (reframe) parse failed for {pid} S{session_num}. Fallback.")
            affect_reframe_data = {
                "mood": 5, "pos_count": 0, "neg_count": max(1, neg_count_detected_entry),
                "reframe_count": 0, "spontaneous_reframe": False,
                "srr_reasoning": "[parse error fallback — reframe affect]",
                "keywords": ["不明"], "summary": "感情分析に失敗しました（フォールバック）。", "_fallback": True,
            }
        raw_responses["affect_reframe"] = affect_reframe_result.get("raw_response")

        spontaneous_reframe = bool(affect_reframe_data.get("spontaneous_reframe", False))
        srr_reasoning = str(affect_reframe_data.get("srr_reasoning", ""))
        reframe_count_raw = int(affect_reframe_data.get("reframe_count", 1 if spontaneous_reframe else 0))
        neg_count_detected_reframe = int(affect_reframe_data.get("neg_count", 0))
        reframe_count = min(reframe_count_raw, neg_count_detected_reframe) if neg_count_detected_reframe > 0 else 0

        is_echo_detected = (
            not spontaneous_reframe
            and (
                "f3" in srr_reasoning.lower()
                or "echo" in srr_reasoning.lower()
                or "ai" in srr_reasoning.lower()
                or "引用" in srr_reasoning
                or "言い換え" in srr_reasoning
                or "繰り返" in srr_reasoning
            )
        )
        time.sleep(0.3)

        # T7: SRR-Feedback Agent
        srr_feedback_user_content = (
            f"Student's reframe attempt:\n{reframe_attempt_text}\n\n"
            f"Assessment: spontaneous_reframe={spontaneous_reframe}, "
            f"is_echo={is_echo_detected}, skill_score={current_observed_skill:.2f}"
        )
        srr_feedback_result = call_openai(
            api_key,
            [{"role": "system", "content": SYSTEM_PROMPTS["srr_feedback"]}, {"role": "user", "content": srr_feedback_user_content}],
            MAX_COMPLETION_TOKENS_AGENT,
        )
        srr_feedback_response = srr_feedback_result["content"] or "[srr feedback error]"
        raw_responses["srr_feedback_agent"] = srr_feedback_result.get("raw_response")
        time.sleep(0.3)
    else:
        spontaneous_reframe = False
        print(f"      [INFO] neg_count=0: skipping reframing loop for {pid} S{session_num}")

    # T8: Skill update — KEY FIX: use update_observed_skill_v2
    neg_count_for_skill = neg_count_detected_reframe if neg_count_detected_entry > 0 else 0
    new_observed_skill = update_observed_skill_v2(
        session_history=session_history,
        reframe_count=reframe_count,
        neg_count=neg_count_for_skill,
        did_attempt=did_attempt,
        current_observed_skill=current_observed_skill,
    )

    if neg_count_for_skill > 0:
        spontaneous_rate = reframe_count / neg_count_for_skill
    elif neg_count_detected_entry > 0:
        spontaneous_rate = 0.0
    else:
        spontaneous_rate = float("nan")

    # T9: Curriculum-Progress Agent
    curr_user_content = (
        f"User skill score: {new_observed_skill:.2f}. "
        f"Skill stage: {stage}. "
        f"Spontaneous reframe detected in this session: {spontaneous_reframe}."
    )
    curr_result = call_openai(
        api_key,
        [{"role": "system", "content": SYSTEM_PROMPTS["curriculum"]}, {"role": "user", "content": curr_user_content}],
        MAX_COMPLETION_TOKENS_AGENT,
    )
    curriculum_response = curr_result["content"] or "[curriculum error]"
    raw_responses["curriculum_agent"] = curr_result.get("raw_response")
    time.sleep(0.3)

    # Assemble session record
    session_record = {
        "persona_id": pid,
        "session": session_num,
        "condition": condition,
        "journal_entry": journal_entry,
        "savoring_response": savoring_response,
        "savoring_reply": savoring_reply_text,
        "reframing_response": reframing_response,
        "reframe_attempt": reframe_attempt_text,
        "srr_feedback_response": srr_feedback_response,
        "curriculum_response": curriculum_response,
        "affect_entry": affect_entry_data,
        "affect_reframe": affect_reframe_data,
        "scaffold_level": scaffold_level,
        "stage": stage,
        "observed_skill_before": round(current_observed_skill, 6),
        "observed_skill_after": round(new_observed_skill, 6),
        "mood": mood,
        "neg_count_entry": neg_count_detected_entry,
        "neg_count_reframe": neg_count_detected_reframe,
        "reframe_count": reframe_count,
        "spontaneous_reframe": spontaneous_reframe,
        "is_echo": is_echo_detected,
        "srr_reasoning": srr_reasoning,
        "spontaneous_rate": (
            round(spontaneous_rate, 6) if not math.isnan(spontaneous_rate) else None
        ),
        "timestamp": timestamp,
        "loop_mode": True,
        "note": (
            "SYNTHETIC USER SESSION (ext_study1_fixed: multi-turn loop, N=10 personas, 14 sessions, "
            "calibrated params v2, update_observed_skill_v2). "
            "Synthetic Users. No real participants."
        ),
        "_version": "ext_study1_fixed",
    }

    raw_responses_record = {
        "persona_id": pid,
        "condition": condition,
        "session": session_num,
        "turns": raw_responses,
    }

    return session_record, raw_responses_record


# ── Per-persona sequential run ────────────────────────────────────────────────

def run_persona_condition(
    api_key: str,
    persona: dict,
    condition: str,
    rng: random.Random,
) -> tuple[list, list, list, list, dict]:
    """
    Run 14 sessions for one persona under one condition.
    Returns (session_records, ground_truth_records, prompt_records, raw_api_responses, transition_info)
    """
    pid = persona["id"]
    alpha = persona["alpha"]
    alpha_passive = persona["alpha_passive"]
    beta = persona["beta"]
    p_attempt_base = persona["p_attempt_base"]
    neg_tendency = persona["neg_tendency"]

    latent_skill = persona["latent_skill_0"]
    observed_skill = persona["latent_skill_0"]
    previous_summary = "This is the first session. No previous sessions."
    session_history_for_obs_skill = []

    session_records = []
    ground_truth_records = []
    prompt_records = []
    raw_api_responses = []

    for session_num in range(1, N_SESSIONS + 1):
        print(f"\n    Session {session_num}/{N_SESSIONS} [{condition}]:")

        if condition == "fixed-high":
            scaffold_level_for_attempt = "high"
        elif condition == "adaptive-fading":
            scaffold_level_for_attempt = get_scaffold_level_from_skill(observed_skill)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        p_attempt = compute_attempt_probability(latent_skill, scaffold_level_for_attempt, p_attempt_base)
        did_attempt = rng.random() < p_attempt

        if did_attempt:
            p_success = compute_attempt_success_probability(latent_skill)
            attempt_success = rng.random() < p_success
        else:
            p_success = 0.0
            attempt_success = False

        neg_count_sampled = sample_neg_count(latent_skill, neg_tendency, rng)

        print(f"      latent_skill={latent_skill:.3f}, scaffold={scaffold_level_for_attempt}, "
              f"p_attempt={p_attempt:.3f}, did_attempt={did_attempt}, "
              f"attempt_success={attempt_success}, neg_count_sampled={neg_count_sampled}")

        # Record generation prompt metadata
        sys_entry, usr_entry = build_entry_generation_prompt(
            persona=persona,
            session_num=session_num,
            latent_skill=latent_skill,
            did_attempt=did_attempt,
            attempt_success=attempt_success,
            neg_count=neg_count_sampled,
            previous_summary=previous_summary,
        )
        prompt_records.append({
            "persona_id": pid,
            "condition": condition,
            "session": session_num,
            "system_prompt": sys_entry,
            "user_message": usr_entry,
            "latent_skill": round(latent_skill, 6),
            "did_attempt": did_attempt,
            "attempt_success": attempt_success,
            "neg_count_sampled": neg_count_sampled,
            "note": "SYNTHETIC: Generation prompt contains NO condition name. ext_study1_fixed.",
        })

        session_rec, raw_rec = run_loop_session(
            api_key=api_key,
            persona=persona,
            session_num=session_num,
            latent_skill=latent_skill,
            did_attempt=did_attempt,
            attempt_success=attempt_success,
            neg_count_sampled=neg_count_sampled,
            previous_summary=previous_summary,
            session_history=session_history_for_obs_skill,
            current_observed_skill=observed_skill,
            condition=condition,
            rng=rng,
        )
        raw_api_responses.append(raw_rec)

        session_rec["latent_skill_before"] = round(latent_skill, 6)
        session_rec["did_attempt_gt"] = did_attempt
        session_rec["attempt_success_gt"] = attempt_success
        session_rec["p_attempt_gt"] = round(p_attempt, 6)
        session_rec["neg_count_sampled"] = neg_count_sampled

        # Update latent_skill
        observed_ai_model = not did_attempt
        latent_skill_new = update_latent_skill(
            latent_skill=latent_skill,
            did_attempt=did_attempt,
            attempt_success=attempt_success,
            observed_ai_model=observed_ai_model,
            alpha=alpha,
            alpha_passive=alpha_passive,
            beta=beta,
        )
        session_rec["latent_skill_after"] = round(latent_skill_new, 6)

        observed_skill = session_rec["observed_skill_after"]

        previous_summary = make_previous_summary(
            session_num=session_num,
            user_entry=session_rec["journal_entry"],
            ai_reframing_response=session_rec.get("reframing_response"),
            scaffold_level=scaffold_level_for_attempt,
            latent_skill=latent_skill_new,
        )

        latent_skill = latent_skill_new
        session_history_for_obs_skill.append(session_rec)
        session_records.append(session_rec)

        srr_val = session_rec["spontaneous_rate"]
        ground_truth_records.append({
            "persona_id": pid,
            "condition": condition,
            "session": session_num,
            "latent_skill_before": session_rec["latent_skill_before"],
            "latent_skill_after": session_rec["latent_skill_after"],
            "p_attempt": round(p_attempt, 6),
            "did_attempt": did_attempt,
            "attempt_success": attempt_success,
            "neg_count_sampled": neg_count_sampled,
            "scaffold_level": scaffold_level_for_attempt,
            "observed_skill_after": round(observed_skill, 6),
            "spontaneous_rate_llm": srr_val,
            "reframe_count_llm": session_rec.get("reframe_count"),
            "srr_reasoning": session_rec.get("srr_reasoning", ""),
            "is_echo": session_rec.get("is_echo", False),
            "loop_mode": True,
            "_version": "ext_study1_fixed",
        })

        srr_str = f"{srr_val:.2f}" if srr_val is not None else "nan"
        print(f"      latent: {session_rec['latent_skill_before']:.3f} → {session_rec['latent_skill_after']:.3f} | "
              f"obs_s: {session_rec['observed_skill_before']:.3f} → {session_rec['observed_skill_after']:.3f} | "
              f"SRR: {srr_str} [reframe_count={session_rec.get('reframe_count')}, "
              f"neg_entry={session_rec.get('neg_count_entry')}, "
              f"is_echo={session_rec.get('is_echo')}]")

    # Post-hoc scaffold transition verification
    print(f"\n  [VERIFY] Scaffold transitions for {pid} [{condition}]:")
    transition_info = verify_scaffold_transitions(session_records, condition, pid)

    return session_records, ground_truth_records, prompt_records, raw_api_responses, transition_info


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary_fixed(
    results_fading: dict,
    results_fixed: dict,
    personas: list,
    all_transition_info: list,
) -> dict:
    """
    Pre-committed summary statistics for ext_study1_fixed.
    delta-SRR = mean(SRR, sessions 8-14) - mean(SRR, sessions 1-7)
    Direction consistency: A > B in >= 6/10 personas.
    """
    n_personas = len(personas)
    direction_threshold = math.ceil(n_personas * 0.6)

    summary = {
        "description": (
            "Extended Study 1 Fixed (mechanism validation): multi-turn loop simulation. "
            "Condition A (adaptive-fading) vs Condition B (fixed-high). "
            "ext_study1_fixed: calibrated params v2, update_observed_skill_v2. "
            "Synthetic Users. No real participants. "
            "Pre-committed analysis: delta-SRR = mean(S8-14) - mean(S1-7)."
        ),
        "version": "ext_study1_fixed",
        "changes_from_ext_study1": [
            "Persona parameters calibrated for scaffold transition within 14 sessions (the research team 2026-06-05)",
            "update_observed_skill_v2(): did_attempt=False sessions excluded from moving average",
            "verify_scaffold_transitions() post-hoc check added",
            "Output prefix: ext_study1_fixed (ext_study1 NOT overwritten)",
        ],
        "loop_mode": True,
        "pre_committed_analysis": {
            "delta_srr_definition": "mean(SRR, sessions 8-14) - mean(SRR, sessions 1-7)",
            "srr_definition": "SRR = reframe_count / neg_count_detected (rubric-based, reframe text)",
            "srr_filter": "Sessions with neg_count_reframe = 0 excluded (SRR undefined)",
            "direction_consistency_threshold": f"A > B in >= {direction_threshold}/{n_personas} personas",
            "hypothesis_support_criteria": "direction_consistent AND mean_delta_A > mean_delta_B",
            "no_significance_test": f"N={n_personas} synthetic personas; no significance test",
        },
        "scaffold_transitions": {t["pid"] + "_" + t["condition"]: t for t in all_transition_info},
        "personas": {},
        "aggregate": {},
    }

    all_delta_fading = []
    all_delta_fixed = []

    for persona in personas:
        pid = persona["id"]
        sessions_a = results_fading.get(pid, [])
        sessions_b = results_fixed.get(pid, [])

        def srr_for_sessions(sessions: list, start: int, end: int) -> float:
            subset = [
                s for s in sessions
                if start <= s["session"] <= end
                and (s.get("neg_count_reframe", 0) > 0 or s.get("neg_count_entry", 0) > 0)
                and s.get("spontaneous_rate") is not None
            ]
            if not subset:
                return float("nan")
            return sum(s["spontaneous_rate"] for s in subset) / len(subset)

        srr_early_a = srr_for_sessions(sessions_a, 1, 7)
        srr_late_a = srr_for_sessions(sessions_a, 8, 14)
        srr_early_b = srr_for_sessions(sessions_b, 1, 7)
        srr_late_b = srr_for_sessions(sessions_b, 8, 14)

        delta_a = (
            srr_late_a - srr_early_a
            if (not math.isnan(srr_early_a) and not math.isnan(srr_late_a))
            else float("nan")
        )
        delta_b = (
            srr_late_b - srr_early_b
            if (not math.isnan(srr_early_b) and not math.isnan(srr_late_b))
            else float("nan")
        )

        all_delta_fading.append(delta_a)
        all_delta_fixed.append(delta_b)

        def _r(v: float) -> Optional[float]:
            return round(v, 4) if not math.isnan(v) else None

        summary["personas"][pid] = {
            "label": persona["label"],
            "condition_adaptive_fading": {
                "srr_early_mean": _r(srr_early_a),
                "srr_late_mean": _r(srr_late_a),
                "delta_srr": _r(delta_a),
                "latent_skill_trajectory": [round(s["latent_skill_after"], 4) for s in sessions_a],
                "p_attempt_trajectory": [round(s.get("p_attempt_gt", 0), 4) for s in sessions_a],
                "srr_per_session": [
                    round(s["spontaneous_rate"], 4) if s.get("spontaneous_rate") is not None else None
                    for s in sessions_a
                ],
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_a],
                "scaffold_sequence": [s["scaffold_level"] for s in sessions_a],
                "is_echo_per_session": [s.get("is_echo", False) for s in sessions_a],
            },
            "condition_fixed_high": {
                "srr_early_mean": _r(srr_early_b),
                "srr_late_mean": _r(srr_late_b),
                "delta_srr": _r(delta_b),
                "latent_skill_trajectory": [round(s["latent_skill_after"], 4) for s in sessions_b],
                "p_attempt_trajectory": [round(s.get("p_attempt_gt", 0), 4) for s in sessions_b],
                "srr_per_session": [
                    round(s["spontaneous_rate"], 4) if s.get("spontaneous_rate") is not None else None
                    for s in sessions_b
                ],
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_b],
                "scaffold_sequence": [s["scaffold_level"] for s in sessions_b],
                "is_echo_per_session": [s.get("is_echo", False) for s in sessions_b],
            },
            "delta_srr_advantage_A_over_B": _r(
                delta_a - delta_b
                if (not math.isnan(delta_a) and not math.isnan(delta_b))
                else float("nan")
            ),
        }

    valid_pairs = [
        (a, b) for a, b in zip(all_delta_fading, all_delta_fixed)
        if not math.isnan(a) and not math.isnan(b)
    ]
    n_valid = len(valid_pairs)
    n_fading_higher = sum(1 for a, b in valid_pairs if a > b)

    mean_delta_a = sum(a for a, _ in valid_pairs) / n_valid if n_valid > 0 else float("nan")
    mean_delta_b = sum(b for _, b in valid_pairs) / n_valid if n_valid > 0 else float("nan")

    direction_consistent = n_fading_higher >= direction_threshold
    hypothesis_supported = direction_consistent and (
        not math.isnan(mean_delta_a) and not math.isnan(mean_delta_b)
        and mean_delta_a > mean_delta_b
    )

    summary["aggregate"] = {
        "n_personas_total": n_personas,
        "n_personas_valid_delta": n_valid,
        "mean_delta_srr_adaptive_fading": round(mean_delta_a, 4) if not math.isnan(mean_delta_a) else None,
        "mean_delta_srr_fixed_high": round(mean_delta_b, 4) if not math.isnan(mean_delta_b) else None,
        "mean_delta_srr_advantage_A_over_B": (
            round(mean_delta_a - mean_delta_b, 4)
            if (not math.isnan(mean_delta_a) and not math.isnan(mean_delta_b))
            else None
        ),
        "n_personas_A_higher_delta_srr": n_fading_higher,
        "direction_consistent_pre_committed": direction_consistent,
        "direction_threshold": direction_threshold,
        "hypothesis_supported": hypothesis_supported,
        "interpretation": (
            f"Adaptive-fading shows higher delta-SRR in {n_fading_higher}/{n_valid} valid personas. "
            f"Pre-committed threshold: >= {direction_threshold}/{n_personas}. "
            f"Hypothesis {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}: "
            f"direction_consistent={direction_consistent}, "
            f"mean_A > mean_B = {not math.isnan(mean_delta_a) and not math.isnan(mean_delta_b) and mean_delta_a > mean_delta_b}. "
            f"No significance test (N={n_personas} synthetic). "
            f"Multi-turn loop: SRR judged on user reframe attempt text."
        ),
    }

    # Scaffold transition summary
    adaptive_transitions = [
        t for t in all_transition_info if t["condition"] == "adaptive-fading"
    ]
    n_with_transitions = sum(1 for t in adaptive_transitions if t["transitions"] > 0)
    summary["scaffold_transition_summary"] = {
        "n_adaptive_personas_with_transitions": n_with_transitions,
        "n_adaptive_personas_total": len(adaptive_transitions),
        "fading_observed": n_with_transitions >= 3,
        "note": (
            f"{n_with_transitions}/{len(adaptive_transitions)} adaptive-fading personas "
            "showed scaffold transitions. Fading mechanism exercised: "
            f"{'YES' if n_with_transitions >= 3 else 'NO — report honestly'}."
        ),
    }

    return summary


# ── Environment record ────────────────────────────────────────────────────────

def save_environment_fixed() -> None:
    env_record = {
        "version": "ext_study1_fixed",
        "model": MODEL,
        "experiment_date": datetime.now(timezone.utc).isoformat()[:10],
        "changes_from_ext_study1": [
            "Persona parameters calibrated (p_attempt_base, latent_skill_0, alpha) — the research team 2026-06-05",
            "update_observed_skill_v2(): did_attempt=False → excluded from moving average",
            "verify_scaffold_transitions() post-hoc assertion added",
        ],
        "seeds": {
            "experiment_seed": EXPERIMENT_SEED,
            "synthetic_user_seed": SYNTHETIC_USER_SEED,
            "pythonhashseed_required": "0",
            "rng_formula": (
                "hashlib.sha256(persona_id).hexdigest()[:8] → persona_hash; "
                "hashlib.sha256(condition).hexdigest()[:8] → condition_hash; "
                "rng_seed = EXPERIMENT_SEED + persona_hash % 10000 + condition_hash % 10000"
            ),
        },
        "scaffold_thresholds": SCAFFOLD_THRESHOLDS,
        "scaffold_attempt_multiplier": SCAFFOLD_ATTEMPT_MULTIPLIER,
        "moving_avg_window": MOVING_AVG_WINDOW,
        "curriculum_thresholds": CURRICULUM_THRESHOLDS,
        "personas_version": "v2_calibrated_2026-06-05",
        "sessions_per_persona": N_SESSIONS,
        "conditions": CONDITIONS,
        "loop_mode": True,
        "synthetic_user_note": (
            "Synthetic Users (multi-turn loop, N=10). "
            "All figures must include this label. No real participants."
        ),
        "circular_bias_prevention": {
            "forbidden_strings": FORBIDDEN_PROMPT_STRINGS,
            "assertion_enabled": True,
            "method": "latent_skill → p_attempt → did_attempt → attempt_success (indirect pathway only)",
        },
    }

    out_path = DATA_PROCESSED_FIXED / "experiment_environment_ext_study1_fixed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, ensure_ascii=False, indent=2)
    print(f"Environment recorded: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("GratiFlow ext_study1_fixed — Multi-Turn Loop Simulation (Corrected)")
    print("SYNTHETIC USERS ONLY.")
    print(f"Model: {MODEL} | EXPERIMENT_SEED={EXPERIMENT_SEED} | N_SESSIONS={N_SESSIONS}")
    print(f"Conditions: {CONDITIONS}")
    print("Calibrated persona parameters (the research team 2026-06-05).")
    print("update_observed_skill_v2(): did_attempt=False excluded from moving avg.")
    print("Output: ext_study1_fixed (ext_study1 NOT overwritten).")
    print("=" * 70)

    # Instrument validation gate
    print("\nChecking instrument validation gate (v2.1 results)...")
    if not INSTRUMENT_VALIDATION_PATH.exists():
        print(f"  [ERROR] Validation results not found: {INSTRUMENT_VALIDATION_PATH}")
        sys.exit(1)
    with open(INSTRUMENT_VALIDATION_PATH) as f:
        val_results = json.load(f)
    gate_passed = val_results.get("gate_criteria", {}).get("gate_passed", False)
    acc = val_results.get("metrics", {}).get("accuracy", 0)
    prec = val_results.get("metrics", {}).get("precision", 0)
    rec = val_results.get("metrics", {}).get("recall", 0)
    print(f"  Accuracy={acc}, Precision={prec}, Recall={rec}, gate_passed={gate_passed}")
    if not gate_passed:
        print("  [ABORT] Instrument validation gate NOT passed.")
        sys.exit(1)
    print("  [OK] Gate passed.")

    # Load API key
    print("\nLoading API key...")
    api_key = load_api_key()
    print("  API key loaded.")

    # Use calibrated personas (defined in this script, not from json file)
    personas = PERSONAS_FIXED
    print(f"\nUsing {len(personas)} calibrated personas (v2, the research team 2026-06-05).")

    # Create output directories
    DATA_PROCESSED_FIXED.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        (DATA_PROCESSED_FIXED / "results" / f"condition_{cond}").mkdir(parents=True, exist_ok=True)

    save_environment_fixed()

    # Run experiments
    all_results: dict[str, dict] = {}
    all_ground_truth: list = []
    all_prompts: list = []
    all_raw_responses: list = []
    all_transition_info: list = []

    for condition in CONDITIONS:
        print(f"\n{'=' * 70}")
        print(f"Condition: {condition}")
        print(f"{'=' * 70}")

        condition_results = {}

        for persona in personas:
            pid = persona["id"]
            print(f"\n  Persona {pid} ({persona['label']}), "
                  f"latent_skill_0={persona['latent_skill_0']}, "
                  f"alpha={persona['alpha']}, p_attempt_base={persona['p_attempt_base']}")

            persona_hash = deterministic_hash(pid) % 10000
            condition_hash = deterministic_hash(condition) % 10000
            rng_seed = EXPERIMENT_SEED + persona_hash + condition_hash
            rng = random.Random(rng_seed)

            sessions, gt_records, prompts, raw_resps, trans_info = run_persona_condition(
                api_key=api_key,
                persona=persona,
                condition=condition,
                rng=rng,
            )

            condition_results[pid] = sessions
            all_ground_truth.extend(gt_records)
            all_prompts.extend(prompts)
            all_raw_responses.extend(raw_resps)
            all_transition_info.append(trans_info)

            # Save per-persona results immediately (fail-safe)
            out_path = (
                DATA_PROCESSED_FIXED / "results" / f"condition_{condition}"
                / f"{pid}_sessions.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            print(f"\n    Saved: {out_path}")

        all_results[condition] = condition_results

    timestamp = datetime.now(timezone.utc).isoformat()

    # Ground truth
    gt_path = DATA_PROCESSED_FIXED / "ground_truth_ext_study1_fixed.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "Ground truth: ext_study1_fixed. Calibrated params. SYNTHETIC.",
                "generated_at": timestamp,
                "version": "ext_study1_fixed",
            },
            "records": all_ground_truth,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nGround truth saved: {gt_path}")

    # Generation prompts
    prompts_path = DATA_PROCESSED_FIXED / "synthetic_user_prompts_ext_study1_fixed.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "All generation prompts for ext_study1_fixed. Verify: no condition name.",
                "generated_at": timestamp,
                "version": "ext_study1_fixed",
                "total_prompts": len(all_prompts),
            },
            "prompts": all_prompts,
        }, f, ensure_ascii=False, indent=2)
    print(f"Generation prompts saved: {prompts_path}")

    # Raw API responses
    raw_path = DATA_RAW / "synthetic_users_ext_study1_fixed_raw_responses.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "Raw API responses for ext_study1_fixed. DO NOT EDIT.",
                "generated_at": timestamp,
                "version": "ext_study1_fixed",
            },
            "responses": all_raw_responses,
        }, f, ensure_ascii=False, indent=2)
    print(f"Raw responses saved (do not edit): {raw_path}")

    # Summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary_fixed(
        results_fading=all_results.get("adaptive-fading", {}),
        results_fixed=all_results.get("fixed-high", {}),
        personas=personas,
        all_transition_info=all_transition_info,
    )

    summary_path = DATA_PROCESSED_FIXED / "results" / "summary_statistics_ext_study1_fixed.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")

    # Print key results
    agg = summary["aggregate"]
    trans_sum = summary["scaffold_transition_summary"]
    print("\n" + "=" * 70)
    print("KEY RESULTS (ext_study1_fixed: Synthetic Users, N=10, 14 sessions)")
    print("=" * 70)
    a_val = agg.get("mean_delta_srr_adaptive_fading")
    b_val = agg.get("mean_delta_srr_fixed_high")
    adv_val = agg.get("mean_delta_srr_advantage_A_over_B")
    print(f"Mean delta-SRR (adaptive-fading):  {f'{a_val:+.4f}' if a_val is not None else 'null'}")
    print(f"Mean delta-SRR (fixed-high):        {f'{b_val:+.4f}' if b_val is not None else 'null'}")
    print(f"Advantage A over B:                 {f'{adv_val:+.4f}' if adv_val is not None else 'null'}")
    print(f"Personas A > B:                     {agg['n_personas_A_higher_delta_srr']}/{agg['n_personas_valid_delta']}")
    print(f"Direction consistent (>= {agg['direction_threshold']}/{agg['n_personas_total']}): "
          f"{agg['direction_consistent_pre_committed']}")
    print(f"Hypothesis supported:               {agg['hypothesis_supported']}")
    print(f"Scaffold transitions (adaptive):    {trans_sum['n_adaptive_personas_with_transitions']}/{trans_sum['n_adaptive_personas_total']} personas")
    print(f"Fading mechanism exercised:         {trans_sum['fading_observed']}")

    print("\nPer-persona delta-SRR:")
    for pid, pdata in summary["personas"].items():
        a = pdata["condition_adaptive_fading"]["delta_srr"]
        b = pdata["condition_fixed_high"]["delta_srr"]
        adv = pdata.get("delta_srr_advantage_A_over_B")
        sc_seq = "".join([s[0].upper() for s in pdata["condition_adaptive_fading"].get("scaffold_sequence", [])])
        label = pdata["label"]
        a_str = f"{a:+.4f}" if a is not None else "  null"
        b_str = f"{b:+.4f}" if b is not None else "  null"
        adv_str = f"{adv:+.4f}" if adv is not None else "  null"
        print(f"  {pid} ({label}): A={a_str}, B={b_str}, A-B={adv_str}, scaffold_A={sc_seq}")

    print(f"\n{agg['interpretation']}")
    print("\nNOTE: N=10 synthetic personas. No significance test. PoC direction only.")
    print("Results are honest — no manipulation. Figures: 'Synthetic Users (multi-turn loop, N=10)'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
