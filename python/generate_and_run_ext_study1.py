"""
generate_and_run_ext_study1.py
==============================
GratiFlow Extended Study 1 — Multi-Turn Conversational Loop Simulation

Implements the extended Study 1 (mechanism validation) using the multi-turn
conversational reframing loop specified in CONVERSATIONAL_LOOP_SPEC.md.

Design lineage:
  - v2.1 (single-pass pipeline, ablation, N=5 personas, 10 sessions) → UNCHANGED, preserved
  - ext_study1 (multi-turn loop, N=10 personas, 14 sessions) → THIS SCRIPT (new prefix, no overwrite)

Multi-turn loop per CONVERSATIONAL_LOOP_SPEC.md §2.2:
  AWAITING_ENTRY
    → Affect-Analysis (background)
    → Savoring Agent
  AWAITING_SAVORING_REPLY  (simulated: synthetic deepening reply)
    → [if neg_count == 0] → CLOSING (skip reframing)
    → [if neg_count > 0]  → Reframing-Coach (scaffold-level-appropriate)
  AWAITING_USER_REFRAME  (simulated: synthetic self-reframe attempt)
    → Affect-Analysis on reframe text (SRR judgment)
    → SRR judgment: spontaneous_reframe, is_echo, pos_count, neg_count
    → Skill update (moving average)
    → SRR-Feedback Agent
  CLOSING
    → Curriculum-Progress Agent

Circular-bias avoidance (strict, per v2.1 protocol):
  - Condition name (adaptive-fading / fixed-high) is NEVER passed to generation prompts.
  - latent_skill drives behavior via p_attempt → did_attempt → attempt_success.
  - assert_no_condition_leakage() is called before every synthetic generation API call.
  - The condition only affects scaffold_level, which in turn affects p_attempt.

Instrument validation:
  - Reuses v2.1 validation results (accuracy=1.0, gate_passed=True).
  - Gate passage is loaded from data/processed/experiments_v2_1/instrument_validation/
    validation_results.json at startup. If gate_passed != True, script aborts.

Pre-committed analysis (per extended_study_design_v2.md §4.2 + evaluation_protocol_v2_1.md §3.2):
  - Primary: delta-SRR = mean(SRR, sessions 8-14) - mean(SRR, sessions 1-7)
    (Extended from v2.1's 8-10 vs 1-3 to match 14-session design)
  - Adaptive-fading vs fixed-high; direction consistency >= 6/10 personas.
  - Results reported regardless of direction. No cherry-picking.

Multi-turn vs single-pass differences from v2.1:
  1. Loop adds a savoring-reply turn (synthetic deepening text).
  2. Loop adds a user-reframe turn (synthetic self-reframe attempt text).
  3. SRR is judged on the user's reframe text (not the journal entry itself).
  4. SRR-Feedback agent provides per-session reinforcement.
  5. is_echo detection: Affect-Analysis receives the reframing_response + reframe text.
  6. Skill update uses MOVING_AVG_WINDOW=5 sessions of spontaneous_rates.

New output prefix: ext_study1 (all files written to data/processed/ext_study1/)
Existing v2.1 files are NOT modified.

Seeds: EXPERIMENT_SEED=42, SYNTHETIC_USER_SEED=2026 (inherited from v2.1)
Run with: PYTHONHASHSEED=0 python python/generate_and_run_ext_study1.py

Author: team member (experiment lead, the research team)
Date: 2026-06-05
Protocol: CONVERSATIONAL_LOOP_SPEC.md (v1.0) + extended_study_design_v2.md §4.2
          + evaluation_protocol_v2_1.md (instrument validation / circular-bias avoidance)
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

# ── PYTHONHASHSEED check (reproducibility) ────────────────────────────────────
pythonhashseed = os.environ.get("PYTHONHASHSEED")
if pythonhashseed != "0":
    warnings.warn(
        f"PYTHONHASHSEED is '{pythonhashseed}' (expected '0'). "
        "RNG seeds use hashlib.sha256 (PYTHONHASHSEED-independent), "
        "but for full reproducibility run: PYTHONHASHSEED=0 python generate_and_run_ext_study1.py",
        UserWarning,
        stacklevel=1,
    )

# Import v2 latent skill model (unchanged; reused in ext_study1)
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

# Token budgets per agent call
MAX_COMPLETION_TOKENS_ENTRY = 512       # synthetic user journal entry
MAX_COMPLETION_TOKENS_SAVORING_REPLY = 300  # synthetic user deepening reply
MAX_COMPLETION_TOKENS_REFRAME_ATTEMPT = 400  # synthetic user self-reframe attempt
MAX_COMPLETION_TOKENS_AGENT = 1024      # GratiFlow pipeline agents
MAX_COMPLETION_TOKENS_AFFECT = 800      # affect-analysis JSON (with srr_reasoning)

# Scaffolding thresholds — must match CONVERSATIONAL_LOOP_SPEC.md §2.3 and app.js
SCAFFOLD_THRESHOLDS = {"high": 0.35, "mid": 0.65}

# Curriculum stage thresholds — unchanged from v2.1
CURRICULUM_THRESHOLDS = [0.25, 0.50, 0.75]

# Moving average window — matches CONVERSATIONAL_LOOP_SPEC.md §2.4 MOVING_AVG_WINDOW=5
MOVING_AVG_WINDOW = 5

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2.0  # seconds (exponential backoff)

# Number of sessions per persona — extended from v2.1's 10 to 14 (matches SCED intervention)
N_SESSIONS = 14

# Conditions — same as v2.1
CONDITIONS = ["adaptive-fading", "fixed-high"]

# Paths
BASE_DIR = Path(__file__).parent.parent
PERSONAS_FILE = Path(__file__).parent / "personas_ext_study1.json"
DATA_PROCESSED_EXT = BASE_DIR / "data" / "processed" / "ext_study1"
DATA_RAW = BASE_DIR / "data" / "raw"
INSTRUMENT_VALIDATION_PATH = (
    BASE_DIR / "data" / "processed" / "experiments_v2_1"
    / "instrument_validation" / "validation_results.json"
)

# Strings that MUST NOT appear in any generation prompt (circular-bias prevention)
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


# ── Affect-Analysis system prompt (v2.1 rubric + few-shot; condition-neutral) ─

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

# ── GratiFlow agent prompts (unchanged from v2.1 except srr_feedback is NEW) ──

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


# ── Utility: Scaffold level determination (matches CONVERSATIONAL_LOOP_SPEC §2.3) ──

def get_scaffold_level_from_skill(s: float) -> str:
    """Mirror of CONVERSATIONAL_LOOP_SPEC §2.3 and app.js getScaffoldLevel."""
    if s < SCAFFOLD_THRESHOLDS["high"]:
        return "high"
    if s < SCAFFOLD_THRESHOLDS["mid"]:
        return "mid"
    return "low"


def get_curriculum_stage(s: float) -> int:
    """Mirror of app.js getCurriculumStage."""
    if s < CURRICULUM_THRESHOLDS[0]:
        return 0
    if s < CURRICULUM_THRESHOLDS[1]:
        return 1
    if s < CURRICULUM_THRESHOLDS[2]:
        return 2
    return 3


def update_observed_skill(session_history: list, reframe_count: int, neg_count: int) -> float:
    """
    Mirror of CONVERSATIONAL_LOOP_SPEC §2.4 (updateSkill moving average).
    Uses reframe_count / neg_count (v2.1 formula).
    Sessions with neg_count=0: treated as 0.0 in moving average.
    """
    if neg_count > 0:
        session_rate = min(reframe_count / neg_count, 1.0)
    else:
        session_rate = 0.0

    recent = session_history[-MOVING_AVG_WINDOW:]
    rates = [s.get("spontaneous_rate") or 0.0 for s in recent]
    rates.append(session_rate)

    avg = sum(rates) / len(rates)
    return max(0.0, min(1.0, avg))


# ── Circular-bias prevention ──────────────────────────────────────────────────

def assert_no_condition_leakage(prompt_text: str) -> None:
    """
    Assert that no condition-identifying string appears in a generation prompt.
    Raises AssertionError if any forbidden string is found (case-insensitive).
    Called before every synthetic user generation API call.
    """
    lower_text = prompt_text.lower()
    for forbidden in FORBIDDEN_PROMPT_STRINGS:
        if forbidden.lower() in lower_text:
            raise AssertionError(
                f"CONDITION LEAKAGE DETECTED: '{forbidden}' found in generation prompt.\n"
                f"This violates the non-leakage requirement (circular-bias prevention).\n"
                f"Prompt excerpt: ...{prompt_text[max(0, lower_text.find(forbidden.lower())-50):lower_text.find(forbidden.lower())+100]}..."
            )


# ── Deterministic hash (PYTHONHASHSEED-independent) ──────────────────────────

def deterministic_hash(s: str) -> int:
    """Deterministic hash using hashlib.sha256 (not Python's built-in hash)."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


# ── API helper ────────────────────────────────────────────────────────────────

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


def call_openai(
    api_key: str,
    messages: list,
    max_tokens: int,
    seed: Optional[int] = None,
) -> dict:
    """
    Call gpt-5.4-mini with retry and exponential backoff.
    - max_completion_tokens is used (not max_tokens)
    - temperature is NOT sent (not supported by gpt-5.4-mini)
    - seed is passed for best-effort reproducibility
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
    """Parse Affect-Analysis Agent JSON output (v2.1 schema with srr_reasoning, reframe_count)."""
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


# ── Synthetic user text generation ───────────────────────────────────────────

def build_entry_generation_prompt(
    persona: dict,
    session_num: int,
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    neg_count: int,
    previous_summary: str,
) -> tuple[str, str]:
    """
    Build system + user message for synthetic journal entry generation.
    CRITICAL: condition name and scaffold_level are NOT included.
    """
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
    """
    Build prompt for synthetic user's deepening reply to the Savoring Agent.
    CRITICAL: condition name and scaffold_level are NOT included.
    """
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
    """
    Build prompt for synthetic user's self-reframe attempt.
    This is the key turn for SRR detection.
    CRITICAL: condition name and scaffold_level are NOT included.
    The attempt behavior is driven solely by latent_skill, did_attempt, attempt_success.
    """
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


# ── Multi-turn loop session runner ────────────────────────────────────────────

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

    Turn sequence:
      T1: Generate synthetic journal entry
      T2: Affect-Analysis on journal entry (background; extracts neg_count, pos_count)
      T3: Savoring Agent → synthetic savoring reply
      T4: [if neg_count > 0] Reframing-Coach (scaffold-level-appropriate)
      T5: [if neg_count > 0] Generate synthetic reframe attempt
      T6: Affect-Analysis on reframe text (SRR judgment; is_echo detection)
      T7: SRR-Feedback Agent
      T8: Skill update
      T9: Curriculum-Progress Agent

    Args:
        condition: "adaptive-fading" or "fixed-high"
                   Used ONLY to set scaffold_level for agents.
                   NEVER passed to user generation prompts.

    Returns:
        (session_record, raw_responses_record)
    """
    pid = persona["id"]
    timestamp = datetime.now(timezone.utc).isoformat()

    # Determine scaffold_level (per CONVERSATIONAL_LOOP_SPEC §2.3)
    # Fixed at session construction; not re-computed intra-session.
    if condition == "fixed-high":
        scaffold_level = "high"
    elif condition == "adaptive-fading":
        scaffold_level = get_scaffold_level_from_skill(current_observed_skill)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    stage = get_curriculum_stage(current_observed_skill)
    print(f"      obs_s={current_observed_skill:.3f} → scaffold={scaffold_level}, stage={stage}")

    raw_responses = {}

    # ── T1: Generate synthetic journal entry ──────────────────────────────
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
        [
            {"role": "system", "content": sys_prompt_entry},
            {"role": "user", "content": usr_msg_entry},
        ],
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

    # ── T2: Affect-Analysis on journal entry (background) ────────────────
    affect_entry_result = call_openai(
        api_key,
        [
            {"role": "system", "content": SYSTEM_PROMPTS["affect"]},
            {"role": "user", "content": journal_entry},
        ],
        MAX_COMPLETION_TOKENS_AFFECT,
    )
    affect_entry_data = parse_affect_json(affect_entry_result["content"])
    if affect_entry_data is None:
        print(f"      [WARNING] Affect-Analysis (entry) parse failed for {pid} S{session_num}. Fallback.")
        affect_entry_data = {
            "mood": 5,
            "pos_count": 0,
            "neg_count": max(1, neg_count_sampled),
            "reframe_count": 0,
            "spontaneous_reframe": False,
            "srr_reasoning": "[parse error fallback — entry affect]",
            "keywords": ["不明"],
            "summary": "感情分析に失敗しました（フォールバック）。",
            "_fallback": True,
        }
    raw_responses["affect_entry"] = affect_entry_result.get("raw_response")

    neg_count_detected_entry = int(affect_entry_data.get("neg_count", 0))
    mood = int(affect_entry_data.get("mood", 5))
    time.sleep(0.3)

    # ── T3: Savoring Agent ────────────────────────────────────────────────
    sav_result = call_openai(
        api_key,
        [
            {"role": "system", "content": SYSTEM_PROMPTS["savoring"]},
            {"role": "user", "content": journal_entry},
        ],
        MAX_COMPLETION_TOKENS_AGENT,
    )
    savoring_response = sav_result["content"] or "[savoring error]"
    raw_responses["savoring_agent"] = sav_result.get("raw_response")
    time.sleep(0.4)

    # ── T3b: Generate synthetic savoring reply ────────────────────────────
    sys_sav_reply, usr_sav_reply = build_savoring_reply_prompt(
        persona=persona,
        session_num=session_num,
        latent_skill=latent_skill,
        journal_entry=journal_entry,
        savoring_response=savoring_response,
    )
    sav_reply_result = call_openai(
        api_key,
        [
            {"role": "system", "content": sys_sav_reply},
            {"role": "user", "content": usr_sav_reply},
        ],
        MAX_COMPLETION_TOKENS_SAVORING_REPLY,
        seed=SYNTHETIC_USER_SEED,
    )
    savoring_reply_text = sav_reply_result["content"] or "[savoring reply error]"
    raw_responses["savoring_reply_gen"] = sav_reply_result.get("raw_response")
    time.sleep(0.4)

    # ── T4: Reframing-Coach (only if negatives detected in entry) ─────────
    reframing_response = None
    reframe_attempt_text = None
    affect_reframe_data = None
    srr_reasoning = ""
    reframe_count = 0
    neg_count_detected_reframe = 0
    is_echo_detected = False

    if neg_count_detected_entry > 0:
        ref_prompt_key = f"reframing_{scaffold_level}"
        ref_result = call_openai(
            api_key,
            [
                {"role": "system", "content": SYSTEM_PROMPTS[ref_prompt_key]},
                {"role": "user", "content": journal_entry},
            ],
            MAX_COMPLETION_TOKENS_AGENT,
        )
        reframing_response = ref_result["content"] or "[reframing error]"
        raw_responses["reframing_coach"] = ref_result.get("raw_response")
        time.sleep(0.4)

        # ── T5: Generate synthetic reframe attempt ────────────────────────
        sys_reframe, usr_reframe = build_reframe_attempt_prompt(
            persona=persona,
            session_num=session_num,
            latent_skill=latent_skill,
            did_attempt=did_attempt,
            attempt_success=attempt_success,
            neg_count=neg_count_detected_entry,
            journal_entry=journal_entry,
            reframing_coach_response=reframing_response,
        )
        reframe_attempt_result = call_openai(
            api_key,
            [
                {"role": "system", "content": sys_reframe},
                {"role": "user", "content": usr_reframe},
            ],
            MAX_COMPLETION_TOKENS_REFRAME_ATTEMPT,
            seed=SYNTHETIC_USER_SEED,
        )
        reframe_attempt_text = reframe_attempt_result["content"] or "[reframe attempt error]"
        raw_responses["reframe_attempt_gen"] = reframe_attempt_result.get("raw_response")
        time.sleep(0.4)

        # ── T6: Affect-Analysis on reframe text (SRR judgment) ────────────
        # The reframe text is analyzed in context of the AI's reframing_response
        # to detect is_echo (F3 violation).
        # We pass both the AI response and the user's attempt for is_echo detection.
        reframe_analysis_user_content = (
            f"[AI coaching response the student saw]\n{reframing_response}\n\n"
            f"[Student's reframe attempt]\n{reframe_attempt_text}"
        )
        affect_reframe_result = call_openai(
            api_key,
            [
                {"role": "system", "content": SYSTEM_PROMPTS["affect"]},
                {"role": "user", "content": reframe_analysis_user_content},
            ],
            MAX_COMPLETION_TOKENS_AFFECT,
        )
        affect_reframe_data = parse_affect_json(affect_reframe_result["content"])
        if affect_reframe_data is None:
            print(f"      [WARNING] Affect-Analysis (reframe) parse failed for {pid} S{session_num}. Fallback.")
            affect_reframe_data = {
                "mood": 5,
                "pos_count": 0,
                "neg_count": max(1, neg_count_detected_entry),
                "reframe_count": 0,
                "spontaneous_reframe": False,
                "srr_reasoning": "[parse error fallback — reframe affect]",
                "keywords": ["不明"],
                "summary": "感情分析に失敗しました（フォールバック）。",
                "_fallback": True,
            }
        raw_responses["affect_reframe"] = affect_reframe_result.get("raw_response")

        # Extract SRR judgment from reframe analysis
        spontaneous_reframe = bool(affect_reframe_data.get("spontaneous_reframe", False))
        srr_reasoning = str(affect_reframe_data.get("srr_reasoning", ""))
        reframe_count_raw = int(affect_reframe_data.get("reframe_count", 1 if spontaneous_reframe else 0))
        neg_count_detected_reframe = int(affect_reframe_data.get("neg_count", 0))
        reframe_count = min(reframe_count_raw, neg_count_detected_reframe) if neg_count_detected_reframe > 0 else 0

        # is_echo detection: if spontaneous_reframe is False AND reframe_count=0 AND
        # srr_reasoning references F3 or "echo", flag as echo.
        # In loop mode, the Affect-Analysis is given the AI response context, so F3
        # (echo) cases are expected to be detected more accurately than in v2.1 single-pass.
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

        # ── T7: SRR-Feedback Agent ────────────────────────────────────────
        srr_feedback_user_content = (
            f"Student's reframe attempt:\n{reframe_attempt_text}\n\n"
            f"Assessment: spontaneous_reframe={spontaneous_reframe}, "
            f"is_echo={is_echo_detected}, "
            f"skill_score={current_observed_skill:.2f}"
        )
        srr_feedback_result = call_openai(
            api_key,
            [
                {"role": "system", "content": SYSTEM_PROMPTS["srr_feedback"]},
                {"role": "user", "content": srr_feedback_user_content},
            ],
            MAX_COMPLETION_TOKENS_AGENT,
        )
        srr_feedback_response = srr_feedback_result["content"] or "[srr feedback error]"
        raw_responses["srr_feedback_agent"] = srr_feedback_result.get("raw_response")
        time.sleep(0.3)
    else:
        # neg_count == 0: skip reframing loop (CONVERSATIONAL_LOOP_SPEC §2.2)
        spontaneous_reframe = False
        srr_feedback_response = None
        print(f"      [INFO] neg_count=0: skipping reframing loop for {pid} S{session_num}")

    # ── T8: Skill update (CONVERSATIONAL_LOOP_SPEC §2.4) ──────────────────
    # Uses combined neg_count: detected in reframe text (primary) or entry text (fallback)
    neg_count_for_skill = neg_count_detected_reframe if neg_count_detected_entry > 0 else 0
    new_observed_skill = update_observed_skill(
        session_history, reframe_count, neg_count_for_skill
    )

    # SRR value for this session
    if neg_count_for_skill > 0:
        spontaneous_rate = reframe_count / neg_count_for_skill
    elif neg_count_detected_entry > 0:
        # neg_count detected in entry but not in reframe text — treat as 0
        spontaneous_rate = 0.0
    else:
        spontaneous_rate = float("nan")  # no negatives at all → SRR undefined

    # ── T9: Curriculum-Progress Agent ────────────────────────────────────
    curr_user_content = (
        f"User skill score: {new_observed_skill:.2f}. "
        f"Skill stage: {stage}. "
        f"Spontaneous reframe detected in this session: {spontaneous_reframe}."
    )
    curr_result = call_openai(
        api_key,
        [
            {"role": "system", "content": SYSTEM_PROMPTS["curriculum"]},
            {"role": "user", "content": curr_user_content},
        ],
        MAX_COMPLETION_TOKENS_AGENT,
    )
    curriculum_response = curr_result["content"] or "[curriculum error]"
    raw_responses["curriculum_agent"] = curr_result.get("raw_response")
    time.sleep(0.3)

    # ── Assemble session record ───────────────────────────────────────────
    session_record = {
        "persona_id": pid,
        "session": session_num,
        "condition": condition,
        # Multi-turn turns
        "journal_entry": journal_entry,
        "savoring_response": savoring_response,
        "savoring_reply": savoring_reply_text,
        "reframing_response": reframing_response,
        "reframe_attempt": reframe_attempt_text,
        "srr_feedback_response": srr_feedback_response,
        "curriculum_response": curriculum_response,
        # Affect analysis: entry
        "affect_entry": affect_entry_data,
        # Affect analysis: reframe
        "affect_reframe": affect_reframe_data,
        # SRR judgment
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
        # Multi-turn vs v2.1 provenance
        "loop_mode": True,  # Multi-turn loop (new in ext_study1)
        "note": (
            "SYNTHETIC USER SESSION (ext_study1: multi-turn loop, N=10 personas, 14 sessions). "
            "Synthetic Users (multi-turn loop, N=10). No real participants."
        ),
        "_version": "ext_study1",
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
) -> tuple[list, list, list, list]:
    """
    Run 14 sessions sequentially for one persona under one condition.
    Multi-turn loop per CONVERSATIONAL_LOOP_SPEC.md §2.2.

    Returns:
        (session_records, ground_truth_records, prompt_records, raw_api_responses)
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

        # ── Step 1: Determine scaffold_level for p_attempt computation ────
        if condition == "fixed-high":
            scaffold_level_for_attempt = "high"
        elif condition == "adaptive-fading":
            scaffold_level_for_attempt = get_scaffold_level_from_skill(observed_skill)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        # ── Step 2: Compute attempt probability ───────────────────────────
        p_attempt = compute_attempt_probability(latent_skill, scaffold_level_for_attempt, p_attempt_base)

        # ── Step 3: Bernoulli draw for did_attempt ────────────────────────
        did_attempt = rng.random() < p_attempt

        # ── Step 4: Bernoulli draw for attempt_success ────────────────────
        if did_attempt:
            p_success = compute_attempt_success_probability(latent_skill)
            attempt_success = rng.random() < p_success
        else:
            p_success = 0.0
            attempt_success = False

        # ── Step 5: Sample neg_count ──────────────────────────────────────
        neg_count_sampled = sample_neg_count(latent_skill, neg_tendency, rng)

        print(f"      latent_skill={latent_skill:.3f}, scaffold={scaffold_level_for_attempt}, "
              f"p_attempt={p_attempt:.3f}, did_attempt={did_attempt}, "
              f"attempt_success={attempt_success}, neg_count_sampled={neg_count_sampled}")

        # Record generation prompt metadata (for non-leakage audit)
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
            "note": "SYNTHETIC: Generation prompt contains NO condition name. ext_study1 multi-turn loop.",
        })

        # ── Step 6: Run multi-turn loop session ───────────────────────────
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

        # Attach ground-truth context
        session_rec["latent_skill_before"] = round(latent_skill, 6)
        session_rec["did_attempt_gt"] = did_attempt
        session_rec["attempt_success_gt"] = attempt_success
        session_rec["p_attempt_gt"] = round(p_attempt, 6)
        session_rec["neg_count_sampled"] = neg_count_sampled

        # ── Step 7: Update latent_skill ───────────────────────────────────
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

        # Update observed skill
        observed_skill = session_rec["observed_skill_after"]

        # ── Step 8: Build previous_summary for next session ───────────────
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
        })

        srr_str = f"{srr_val:.2f}" if srr_val is not None else "nan"
        print(f"      latent: {session_rec['latent_skill_before']:.3f} → {session_rec['latent_skill_after']:.3f} | "
              f"obs_s: {session_rec['observed_skill_before']:.3f} → {session_rec['observed_skill_after']:.3f} | "
              f"SRR: {srr_str} [reframe_count={session_rec.get('reframe_count')}, "
              f"neg_entry={session_rec.get('neg_count_entry')}, "
              f"is_echo={session_rec.get('is_echo')}]")

    return session_records, ground_truth_records, prompt_records, raw_api_responses


# ── Summary statistics (pre-committed, ext_study1) ───────────────────────────

def compute_summary_ext_study1(
    results_fading: dict,
    results_fixed: dict,
    personas: list,
) -> dict:
    """
    Compute pre-committed summary statistics for ext_study1.

    Pre-committed analysis (extended from v2.1):
      delta-SRR = mean(SRR, sessions 8-14) - mean(SRR, sessions 1-7)
      [Extended from v2.1's 8-10 vs 1-3 to match 14-session design]
      Direction consistency: A > B in >= 6/10 personas.
      Hypothesis support: direction_consistent AND mean_A > mean_B.
      Sessions with neg_count = 0 are excluded (SRR undefined).
    """
    n_personas = len(personas)
    direction_threshold = math.ceil(n_personas * 0.6)  # >= 60% (6/10)

    summary = {
        "description": (
            "Extended Study 1 (mechanism validation): multi-turn loop simulation. "
            "Condition A (adaptive-fading) vs Condition B (fixed-high). "
            "ext_study1: multi-turn loop, N=10 personas, 14 sessions. "
            "Synthetic Users (multi-turn loop, N=10). No real participants. "
            "Pre-committed analysis: delta-SRR = mean(S8-14) - mean(S1-7)."
        ),
        "version": "ext_study1",
        "loop_mode": True,
        "vs_v2_1": (
            "v2.1 used single-pass pipeline (10 sessions, 5 personas, delta-SRR=mean(S8-10)-mean(S1-3)). "
            "ext_study1 uses multi-turn loop (14 sessions, 10 personas, delta-SRR=mean(S8-14)-mean(S1-7)). "
            "Multi-turn loop adds: savoring-reply turn, user-reframe-attempt turn, SRR-Feedback agent, "
            "is_echo detection in reframe context."
        ),
        "pre_committed_analysis": {
            "delta_srr_definition": "mean(SRR, sessions 8-14) - mean(SRR, sessions 1-7)",
            "srr_definition": "SRR = reframe_count / neg_count_detected (rubric-based, reframe text)",
            "srr_filter": "Sessions with neg_count_reframe = 0 excluded (SRR undefined)",
            "direction_consistency_threshold": f"A > B in >= {direction_threshold}/{n_personas} personas",
            "hypothesis_support_criteria": "direction_consistent AND mean_delta_A > mean_delta_B",
            "no_significance_test": f"N={n_personas} synthetic personas; no significance test",
        },
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
            """Mean SRR for sessions [start, end] (1-indexed). Excludes SRR-undefined sessions."""
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

        delta_a = (srr_late_a - srr_early_a
                   if (not math.isnan(srr_early_a) and not math.isnan(srr_late_a))
                   else float("nan"))
        delta_b = (srr_late_b - srr_early_b
                   if (not math.isnan(srr_early_b) and not math.isnan(srr_late_b))
                   else float("nan"))

        latent_traj_a = [s["latent_skill_after"] for s in sessions_a]
        latent_traj_b = [s["latent_skill_after"] for s in sessions_b]
        p_attempt_traj_a = [s.get("p_attempt_gt", float("nan")) for s in sessions_a]
        p_attempt_traj_b = [s.get("p_attempt_gt", float("nan")) for s in sessions_b]
        srr_per_session_a = [
            round(s["spontaneous_rate"], 4) if s.get("spontaneous_rate") is not None else None
            for s in sessions_a
        ]
        srr_per_session_b = [
            round(s["spontaneous_rate"], 4) if s.get("spontaneous_rate") is not None else None
            for s in sessions_b
        ]
        is_echo_per_session_a = [s.get("is_echo", False) for s in sessions_a]
        is_echo_per_session_b = [s.get("is_echo", False) for s in sessions_b]

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
                "latent_skill_trajectory": [round(v, 4) for v in latent_traj_a],
                "p_attempt_trajectory": [round(v, 4) for v in p_attempt_traj_a],
                "srr_per_session": srr_per_session_a,
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_a],
                "is_echo_per_session": is_echo_per_session_a,
            },
            "condition_fixed_high": {
                "srr_early_mean": _r(srr_early_b),
                "srr_late_mean": _r(srr_late_b),
                "delta_srr": _r(delta_b),
                "latent_skill_trajectory": [round(v, 4) for v in latent_traj_b],
                "p_attempt_trajectory": [round(v, 4) for v in p_attempt_traj_b],
                "srr_per_session": srr_per_session_b,
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_b],
                "is_echo_per_session": is_echo_per_session_b,
            },
            "delta_srr_advantage_A_over_B": _r(
                delta_a - delta_b
                if (not math.isnan(delta_a) and not math.isnan(delta_b))
                else float("nan")
            ),
        }

    # Aggregate
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
            f"Multi-turn loop: SRR judged on user reframe attempt text (not journal entry)."
        ),
    }

    return summary


# ── Environment record ────────────────────────────────────────────────────────

def save_environment_ext_study1() -> None:
    """Record experiment environment for reproducibility."""
    env_record = {
        "version": "ext_study1",
        "model": MODEL,
        "experiment_date": datetime.now(timezone.utc).isoformat()[:10],
        "seeds": {
            "experiment_seed": EXPERIMENT_SEED,
            "synthetic_user_seed": SYNTHETIC_USER_SEED,
            "pythonhashseed_required": "0",
            "rng_formula": (
                "hashlib.sha256(persona_id.encode('utf-8')).hexdigest()[:8] → persona_hash "
                "hashlib.sha256(condition.encode('utf-8')).hexdigest()[:8] → condition_hash "
                "rng_seed = EXPERIMENT_SEED + persona_hash % 10000 + condition_hash % 10000"
            ),
        },
        "scaffold_thresholds": SCAFFOLD_THRESHOLDS,
        "scaffold_attempt_multiplier": SCAFFOLD_ATTEMPT_MULTIPLIER,
        "moving_avg_window": MOVING_AVG_WINDOW,
        "curriculum_thresholds": CURRICULUM_THRESHOLDS,
        "personas": [f"P{i}" for i in range(1, 11)],
        "sessions_per_persona": N_SESSIONS,
        "conditions": CONDITIONS,
        "loop_mode": True,
        "loop_turns": [
            "T1: journal_entry_generation",
            "T2: affect_analysis_entry (background)",
            "T3: savoring_agent + savoring_reply_generation",
            "T4: reframing_coach (if neg_count > 0)",
            "T5: reframe_attempt_generation (if neg_count > 0)",
            "T6: affect_analysis_reframe + SRR_judgment + is_echo_detection (if neg_count > 0)",
            "T7: srr_feedback_agent (if neg_count > 0)",
            "T8: skill_update (moving average)",
            "T9: curriculum_progress_agent",
        ],
        "vs_v2_1": {
            "n_personas": "5 → 10",
            "n_sessions": "10 → 14",
            "pipeline": "single-pass → multi-turn loop",
            "srr_judged_on": "journal entry (v2.1) → user reframe attempt text (ext_study1)",
            "is_echo_detection": "not present (v2.1) → present (ext_study1, via reframe context)",
            "srr_feedback_agent": "not present (v2.1) → present (ext_study1)",
            "delta_srr_window": "mean(S8-10)-mean(S1-3) (v2.1) → mean(S8-14)-mean(S1-7) (ext_study1)",
            "direction_threshold": "3/5 (v2.1) → 6/10 (ext_study1)",
        },
        "instrument_validation": {
            "source": "data/processed/experiments_v2_1/instrument_validation/validation_results.json",
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "gate_passed": True,
            "note": "Reused from v2.1; same rubric and affect prompt applied in ext_study1.",
        },
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

    out_path = DATA_PROCESSED_EXT / "experiment_environment_ext_study1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, ensure_ascii=False, indent=2)
    print(f"Environment recorded: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("GratiFlow Extended Study 1 — Multi-Turn Loop Simulation")
    print("SYNTHETIC USERS ONLY. Synthetic Users (multi-turn loop, N=10).")
    print(f"Model: {MODEL} | EXPERIMENT_SEED={EXPERIMENT_SEED} | N_SESSIONS={N_SESSIONS}")
    print(f"Conditions: {CONDITIONS}")
    print("Protocol: CONVERSATIONAL_LOOP_SPEC.md + extended_study_design_v2.md §4.2")
    print("=" * 70)

    # ── Instrument validation gate check ─────────────────────────────────
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
        print("  [ABORT] Instrument validation gate NOT passed. Cannot proceed.")
        print("  Fix: Re-run validate_instrument_v2_1.py with refined rubric.")
        sys.exit(1)
    print("  [OK] Instrument validation gate passed. Proceeding to main experiment.")

    # ── Load API key ──────────────────────────────────────────────────────
    print("\nLoading API key...")
    api_key = load_api_key()
    print("  API key loaded.")

    # ── Load personas ─────────────────────────────────────────────────────
    if not PERSONAS_FILE.exists():
        print(f"  [ERROR] Personas file not found: {PERSONAS_FILE}")
        sys.exit(1)
    with open(PERSONAS_FILE) as f:
        persona_data = json.load(f)
    personas = persona_data["personas"]
    print(f"  Loaded {len(personas)} personas from {PERSONAS_FILE}")

    # ── Create output directories ─────────────────────────────────────────
    DATA_PROCESSED_EXT.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        (DATA_PROCESSED_EXT / "results" / f"condition_{cond}").mkdir(parents=True, exist_ok=True)

    # Save environment
    save_environment_ext_study1()

    # ── Run experiments ───────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    all_ground_truth: list = []
    all_prompts: list = []
    all_raw_responses: list = []

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

            sessions, gt_records, prompts, raw_resps = run_persona_condition(
                api_key=api_key,
                persona=persona,
                condition=condition,
                rng=rng,
            )

            condition_results[pid] = sessions
            all_ground_truth.extend(gt_records)
            all_prompts.extend(prompts)
            all_raw_responses.extend(raw_resps)

            # Save per-persona results immediately (fail-safe)
            out_path = (
                DATA_PROCESSED_EXT / "results" / f"condition_{condition}"
                / f"{pid}_sessions.json"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            print(f"\n    Saved: {out_path}")

        all_results[condition] = condition_results

    # ── Save consolidated outputs ─────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()

    # Ground truth
    gt_path = DATA_PROCESSED_EXT / "ground_truth_ext_study1.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": (
                    "Ground truth: latent_skill, p_attempt, did_attempt, attempt_success "
                    "for all ext_study1 sessions. Multi-turn loop. SYNTHETIC."
                ),
                "generated_at": timestamp,
                "version": "ext_study1",
            },
            "records": all_ground_truth,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nGround truth saved: {gt_path}")

    # Generation prompts (non-leakage audit)
    prompts_path = DATA_PROCESSED_EXT / "synthetic_user_prompts_ext_study1.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": (
                    "All generation prompts for ext_study1 synthetic user entries. "
                    "Verify: no condition name, scaffold level, or condition-implying language."
                ),
                "generated_at": timestamp,
                "version": "ext_study1",
                "total_prompts": len(all_prompts),
            },
            "prompts": all_prompts,
        }, f, ensure_ascii=False, indent=2)
    print(f"Generation prompts saved: {prompts_path}")

    # Raw API responses (DO NOT EDIT)
    raw_path = DATA_RAW / "synthetic_users_ext_study1_raw_responses.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "Raw API responses for ext_study1. DO NOT EDIT.",
                "generated_at": timestamp,
                "version": "ext_study1",
            },
            "responses": all_raw_responses,
        }, f, ensure_ascii=False, indent=2)
    print(f"Raw responses saved (do not edit): {raw_path}")

    # ── Compute and save summary ──────────────────────────────────────────
    print("\nComputing summary statistics (pre-committed analysis, ext_study1)...")
    summary = compute_summary_ext_study1(
        results_fading=all_results.get("adaptive-fading", {}),
        results_fixed=all_results.get("fixed-high", {}),
        personas=personas,
    )

    summary_path = DATA_PROCESSED_EXT / "results" / "summary_statistics_ext_study1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")

    # ── Print key results ─────────────────────────────────────────────────
    agg = summary["aggregate"]
    print("\n" + "=" * 70)
    print("KEY RESULTS (ext_study1: Multi-Turn Loop, Synthetic Users, N=10 personas)")
    print("Synthetic Users (multi-turn loop, N=10) — 14 sessions")
    print("=" * 70)
    a_val = agg.get("mean_delta_srr_adaptive_fading")
    b_val = agg.get("mean_delta_srr_fixed_high")
    adv_val = agg.get("mean_delta_srr_advantage_A_over_B")
    print(f"Mean delta-SRR (adaptive-fading):  {f'{a_val:+.4f}' if a_val is not None else 'null'}")
    print(f"Mean delta-SRR (fixed-high):        {f'{b_val:+.4f}' if b_val is not None else 'null'}")
    print(f"Advantage A over B:                 {f'{adv_val:+.4f}' if adv_val is not None else 'null'}")
    print(f"Personas A > B:                     {agg['n_personas_A_higher_delta_srr']}/{agg['n_personas_valid_delta']}")
    print(f"Direction consistent (pre-commit, >= {agg['direction_threshold']}/{agg['n_personas_total']}): "
          f"{agg['direction_consistent_pre_committed']}")
    print(f"Hypothesis supported (ext_study1):  {agg['hypothesis_supported']}")

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

    print(f"\n{agg['interpretation']}")
    print("\nNOTE: N=10 synthetic personas; no significance test. PoC direction only.")
    print("SRR judged on user reframe attempt text (multi-turn loop — new in ext_study1).")
    print("Results are honest — no manipulation. Figures labeled 'Synthetic Users (multi-turn loop, N=10)'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
