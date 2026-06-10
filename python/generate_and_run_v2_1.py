"""
generate_and_run_v2_1.py
=========================
GratiFlow v2.1 Evaluation — Final Re-experiment with Corrected SRR Detector

Implements the v2.1 protocol from evaluation_protocol_v2_1.md (the research team, 2026-06-04).
This is the FINAL experiment iteration. Results are reported regardless of direction.

Changes from v2 (3 locations only, per protocol Section 11):
  1. SYSTEM_PROMPTS["affect"]: Replaced with rubric + few-shot prompt (Section 1.2).
     JSON schema adds "srr_reasoning" (string) and "reframe_count" (integer).
  2. RNG initialization: hash() → hashlib.sha256 (deterministic, PYTHONHASHSEED-independent).
  3. SRR calculation: pos_count / neg_count → reframe_count / neg_count (Section 1.5).

Everything else is identical to v2:
  - Sequential generation (latent-skill-driven)
  - Condition name non-leakage assertions
  - Personas (personas_v2.json)
  - Seeds (EXPERIMENT_SEED=42, SYNTHETIC_USER_SEED=2026)
  - SCAFFOLD_THRESHOLDS, MOVING_AVG_WINDOW, CURRICULUM_THRESHOLDS
  - N_SESSIONS=10, CONDITIONS, 5 personas

IMPORTANT:
  - Run with PYTHONHASHSEED=0 for reproducibility:
      PYTHONHASHSEED=0 python python/generate_and_run_v2_1.py
  - v2 code/data are NOT modified by this script.
  - All synthetic user entries: "Synthetic Users (sequential, N=5)" must be noted on figures.

Author: team member (experiment lead, the research team)
Date: 2026-06-04
Protocol: evaluation_protocol_v2_1.md (the research team, 2026-06-04, PI-approved)
Seeds: EXPERIMENT_SEED=42, SYNTHETIC_USER_SEED=2026, PYTHONHASHSEED=0 (required at runtime)
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

# Check PYTHONHASHSEED at startup (required for reproducibility)
pythonhashseed = os.environ.get("PYTHONHASHSEED")
if pythonhashseed != "0":
    warnings.warn(
        f"PYTHONHASHSEED is '{pythonhashseed}' (expected '0'). "
        "RNG seeds are computed with hashlib.sha256 (PYTHONHASHSEED-independent), "
        "but for full reproducibility, run with: PYTHONHASHSEED=0 python generate_and_run_v2_1.py",
        UserWarning,
        stacklevel=1,
    )

# Import v2 latent skill model functions (unchanged from v2)
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

MAX_COMPLETION_TOKENS_ENTRY = 512    # synthetic user journal entry
MAX_COMPLETION_TOKENS_AGENT = 1024   # GratiFlow pipeline agents
MAX_COMPLETION_TOKENS_AFFECT = 768   # affect-analysis JSON (v2.1: larger for srr_reasoning)

# Scaffolding thresholds — unchanged from v2, must match app.js SCAFFOLD_THRESHOLDS
SCAFFOLD_THRESHOLDS = {"high": 0.35, "mid": 0.65}

# Curriculum stage thresholds — unchanged from v2, must match app.js getCurriculumStage
CURRICULUM_THRESHOLDS = [0.25, 0.50, 0.75]

# Moving average window — unchanged from v2, must match app.js MOVING_AVG_WINDOW
MOVING_AVG_WINDOW = 5

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2.0  # seconds (exponential backoff)

# Number of sessions per persona — unchanged from v2
N_SESSIONS = 10

# Conditions — unchanged from v2
CONDITIONS = ["adaptive-fading", "fixed-high"]

# Paths
BASE_DIR = Path(__file__).parent.parent
PERSONAS_FILE = Path(__file__).parent / "personas_v2.json"
DATA_PROCESSED_V2_1 = BASE_DIR / "data" / "processed" / "experiments_v2_1"
DATA_RAW = BASE_DIR / "data" / "raw"

# Strings that MUST NOT appear in generation prompts (non-leakage assertion) — unchanged from v2
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


# ── v2.1 Change 1: Modified Affect-Analysis system prompt ─────────────────────
# Replaces the ambiguous v2 prompt with rubric + few-shot (protocol Section 1.2).
# CRITICAL: This prompt contains NO condition names (adaptive-fading / fixed-high).
# The rubric is based on cognitive reappraisal theory and is condition-neutral.

AFFECT_SYSTEM_PROMPT_V2_1 = (
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
    "  \"keywords\": [<string>, ...],  // 3–6 key emotional words (Japanese)\n"
    "  \"summary\": \"<string>\"         // 1-sentence Japanese summary of the user's emotional state\n"
    "}\n\n"
    "=== RUBRIC: spontaneous_reframe judgment ===\n\n"
    "Definition (operationalized from cognitive reappraisal theory):\n"
    "A \"spontaneous reframe\" is TRUE if and only if the user's text contains ALL THREE of:\n"
    "  (R1) An explicitly stated negative event or negative emotion (the source).\n"
    "  (R2) A deliberate reinterpretation that transforms the meaning of that SAME negative event\n"
    "       into a positive, growth-oriented, or silver-lining perspective (the reframe).\n"
    "  (R3) Evidence that the reframe was generated by the user THEMSELVES, not echoed from\n"
    "       a prior AI response or from a prompt instruction.\n\n"
    "spontaneous_reframe is FALSE if any of the following apply:\n"
    "  (F1) The user simply describes a positive event without connecting it to a negative one.\n"
    "       Example: \"友達とランチして楽しかった\" → This is a positive fact, NOT a reframe.\n"
    "  (F2) The user expresses vague optimism or a coping platitude without specific reinterpretation.\n"
    "       Example: \"まあ何とかなるか\" / \"仕方ないけど頑張ろう\" → Resignation or vague hope, NOT a reframe.\n"
    "  (F3) The user repeats or paraphrases a reframe that the AI previously modeled for them\n"
    "       in the same or recent sessions.\n"
    "  (F4) The user only describes the negative event without any positive reinterpretation.\n"
    "  (F5) The positive aspect is about a DIFFERENT event, not a reinterpretation of the negative one.\n\n"
    "=== FEW-SHOT EXAMPLES ===\n\n"
    "--- Example 1: TRUE (clear spontaneous reframe) ---\n"
    "Entry: \"今日のプレゼンで頭が真っ白になって失敗した。でも、考えてみると、この失敗のおかげで自分の準備不足がはっきりわかった。次回は練習を3回以上して臨もうと思う。失敗が学びになった。\"\n"
    "Judgment: spontaneous_reframe = true\n"
    "Reasoning: \"The user identifies a specific negative event (presentation failure), then deliberately reinterprets it as a learning opportunity with a concrete action plan. This meets R1 (negative event stated), R2 (reinterpretation as learning), and R3 (self-generated insight).\"\n\n"
    "--- Example 2: TRUE (genuine reframe, lower quality) ---\n"
    "Entry: \"バイト先で店長に怒られた。最初はすごく落ち込んだけど、よく考えると店長は自分に期待してくれているからこそ厳しく言ってくれたのかもしれない。\"\n"
    "Judgment: spontaneous_reframe = true\n"
    "Reasoning: \"The user reinterprets being scolded (R1) as a sign of the manager's expectations (R2). The reframe is tentative ('かもしれない') but represents a genuine cognitive shift. No prior AI response is echoed (R3).\"\n\n"
    "--- Example 3: FALSE (positive fact, not a reframe) ---\n"
    "Entry: \"レポートの締め切りに追われて大変だった。でも放課後に友達とカフェに行けて楽しかった。\"\n"
    "Judgment: spontaneous_reframe = false\n"
    "Reasoning: \"The positive event (cafe with friends) is a SEPARATE event, not a reinterpretation of the negative one (deadline stress). This violates F5: the positive aspect is about a different event.\"\n\n"
    "--- Example 4: FALSE (vague optimism, not a reframe) ---\n"
    "Entry: \"実験がうまくいかなくて落ち込んだ。まあ、なんとかなるだろう。\"\n"
    "Judgment: spontaneous_reframe = false\n"
    "Reasoning: \"The user acknowledges the negative event but only adds vague optimism ('なんとかなるだろう') without specific reinterpretation. This matches F2: coping platitude without cognitive shift.\"\n\n"
    "--- Example 5: FALSE (no negative event mentioned) ---\n"
    "Entry: \"今日は天気が良くて気持ちよかった。授業も集中できたし、充実した一日だった。\"\n"
    "Judgment: spontaneous_reframe = false\n"
    "Reasoning: \"No negative event or emotion is mentioned. Without a negative source (R1 not met), spontaneous reframing cannot occur.\"\n\n"
    "--- Example 6: FALSE (AI echo) ---\n"
    "Entry: \"昨日AIが『失敗は成長の種』と教えてくれた。今日テストで悪い点を取ったけど、成長の種だと思うことにする。\"\n"
    "Judgment: spontaneous_reframe = false\n"
    "Reasoning: \"The reframe ('成長の種') directly echoes what the AI previously modeled. This violates F3: the reframe is not self-generated.\"\n\n"
    "=== END RUBRIC ===\n\n"
    "Rules:\n"
    "- Count only expressions in the USER's text, not the AI's previous messages.\n"
    "- Always provide srr_reasoning to justify your spontaneous_reframe judgment.\n"
    "- When in doubt, mark spontaneous_reframe as FALSE. The rubric criteria (R1, R2, R3) must ALL be met.\n"
    "- Do NOT infer intent beyond what is explicitly written in the text.\n"
    "- reframe_count must be an integer >= 0 and <= neg_count."
)

# GratiFlow agent system prompts (all unchanged from v2, except "affect" above)
SYSTEM_PROMPTS = {
    "savoring": (
        "You are the Savoring Agent in GratiFlow, a well-being coaching AI for students.\n"
        "Your role is to help the user vividly re-experience and deepen their appreciation of positive events they described today.\n\n"
        "Guidelines:\n"
        "- Ask one warm, curious follow-up question that helps the user \"taste\" the good moment more fully.\n"
        "- Focus on sensory details, emotions, and the people involved.\n"
        "- Keep your response short (2–4 sentences + one question).\n"
        "- Use a gentle, encouraging Japanese tone.\n"
        "- Do NOT give advice or reframe negatives at this stage.\n"
        "- Output in Japanese."
    ),
    "reframing_high": (
        "You are the Reframing-Coach Agent in GratiFlow, a well-being coaching AI for students.\n"
        "The user is a beginner at positive reframing. Use HIGH scaffolding (modeling strategy):\n"
        "- First, gently acknowledge the negative expression the user wrote.\n"
        "- Then explicitly model a positive reframing: \"例えばこう捉えることもできます：「…」\"\n"
        "- Explain briefly WHY this reframe is valid (1 sentence).\n"
        "- Invite the user to try expressing their own version.\n"
        "- Keep it warm and non-judgmental. Output in Japanese."
    ),
    "reframing_mid": (
        "You are the Reframing-Coach Agent in GratiFlow, a well-being coaching AI for students.\n"
        "The user has some reframing skill. Use MID scaffolding (partial hint strategy):\n"
        "- Acknowledge the negative expression briefly.\n"
        "- Give a partial hint that points toward a positive angle WITHOUT completing the reframe.\n"
        "  e.g. \"この出来事、別の角度から見ると「○○の機会」とも言えそうですね。どう思いますか？\"\n"
        "- Do NOT spell out the full reframe; let the user complete it.\n"
        "- Output in Japanese."
    ),
    "reframing_low": (
        "You are the Reframing-Coach Agent in GratiFlow, a well-being coaching AI for students.\n"
        "The user is becoming skilled at positive reframing. Use LOW scaffolding (fading strategy):\n"
        "- Acknowledge what they wrote with one encouraging sentence.\n"
        "- Simply invite them to reframe it themselves: \"自分なりのポジティブな見方を書いてみてください。\"\n"
        "- Do NOT provide any hints or examples.\n"
        "- Trust the user's growing ability. Output in Japanese."
    ),
    "affect": AFFECT_SYSTEM_PROMPT_V2_1,  # v2.1: replaced with rubric + few-shot
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
        "- Keep it brief (2–3 sentences total) and warm.\n"
        "- Output in Japanese."
    ),
}


# ── Pipeline helper functions (mirrors app.js, unchanged from v2) ──────────────

def get_scaffold_level_from_skill(s: float) -> str:
    """Mirror of app.js getScaffoldLevel. Used in adaptive-fading condition."""
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
    Mirror of app.js updateSkill.
    Computes moving average of spontaneous reframing rate (SRR) over the last MOVING_AVG_WINDOW sessions.

    v2.1 change: uses reframe_count / neg_count instead of pos_count / neg_count.
    This is the OBSERVED skill score 's', distinct from latent_skill.
    """
    if neg_count > 0:
        session_rate = min(reframe_count / neg_count, 1.0)
    else:
        session_rate = 0.0  # undefined when no negatives; treat as 0 for moving avg

    recent = session_history[-MOVING_AVG_WINDOW:]
    # Filter out None values (sessions where SRR was undefined, i.e., neg_count=0)
    # Treat such sessions as 0.0 for the moving average (no evidence of reframing)
    rates = [s.get("spontaneous_rate") or 0.0 for s in recent]
    rates.append(session_rate)

    avg = sum(rates) / len(rates)
    return max(0.0, min(1.0, avg))


# ── v2.1 Change 2: Deterministic hash (hashlib.sha256) ───────────────────────

def deterministic_hash(s: str) -> int:
    """
    Deterministic hash that does NOT depend on PYTHONHASHSEED.

    Uses hashlib.sha256 to produce a reproducible integer from a string.
    Replaces Python's built-in hash() which is randomized by PYTHONHASHSEED.

    Protocol reference: evaluation_protocol_v2_1.md Section 5.2.
    """
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


# ── API functions ─────────────────────────────────────────────────────────────

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

    Notes:
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
    """
    Parse affect agent JSON output for v2.1.

    v2.1 additions:
      - Parses "srr_reasoning" (string, optional with fallback to "")
      - Parses "reframe_count" (integer, optional with fallback: 1 if spontaneous_reframe else 0)

    Returns None on parse failure.
    """
    if not content:
        return None
    cleaned = content.strip()
    # Remove markdown fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        data = json.loads(cleaned)
        required = ["mood", "pos_count", "neg_count", "spontaneous_reframe", "keywords", "summary"]
        if all(k in data for k in required):
            # v2.1: fallback for new fields if not present
            if "srr_reasoning" not in data:
                data["srr_reasoning"] = ""
            if "reframe_count" not in data:
                # fallback: 1 if spontaneous_reframe else 0
                data["reframe_count"] = 1 if data.get("spontaneous_reframe", False) else 0
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ── v2 Core: Synthetic User Entry Generation (unchanged from v2) ──────────────

def assert_no_condition_leakage(prompt_text: str) -> None:
    """
    Assert that the generation prompt does not contain any condition-identifying strings.

    This is the structural safeguard against circular bias per protocol Section 7.1.
    Called before every synthetic user generation API call.

    Raises:
        AssertionError if any forbidden string is found (case-insensitive).
    """
    lower_text = prompt_text.lower()
    for forbidden in FORBIDDEN_PROMPT_STRINGS:
        if forbidden.lower() in lower_text:
            raise AssertionError(
                f"CONDITION LEAKAGE DETECTED: '{forbidden}' found in generation prompt.\n"
                f"This violates the non-leakage requirement (protocol Section 3.2).\n"
                f"Prompt excerpt: ...{prompt_text[max(0, lower_text.find(forbidden.lower())-50):lower_text.find(forbidden.lower())+100]}..."
            )


def build_generation_prompt(
    persona: dict,
    session_num: int,
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    neg_count: int,
    previous_summary: str,
) -> tuple[str, str]:
    """
    Build the system and user message for synthetic journal entry generation.

    CRITICAL: Condition name and scaffold_level are NOT passed to this function.
    Only latent_skill, did_attempt, attempt_success (behavior state) are used.

    Returns:
        (system_prompt, user_message) — strings for the API call
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

    # CRITICAL: Assert no condition name leaked into the prompt
    assert_no_condition_leakage(system_prompt)
    assert_no_condition_leakage(user_message)

    return system_prompt, user_message


def generate_user_entry(
    api_key: str,
    persona: dict,
    session_num: int,
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    neg_count: int,
    previous_summary: str,
) -> tuple[str, dict, str, str]:
    """
    Generate a synthetic journal entry for one persona, one session.

    Returns:
        (entry_text, raw_response, system_prompt, user_message)
    """
    system_prompt, user_message = build_generation_prompt(
        persona, session_num, latent_skill,
        did_attempt, attempt_success, neg_count, previous_summary
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    result = call_openai(api_key, messages, MAX_COMPLETION_TOKENS_ENTRY, seed=SYNTHETIC_USER_SEED)

    if result["error"] or not result["content"]:
        fallback = f"[生成エラー] ペルソナ: {persona['id']}, セッション: {session_num}"
        print(f"      [WARNING] Generation error: {result['error']}")
        return fallback, {}, system_prompt, user_message

    return result["content"], result["raw_response"], system_prompt, user_message


# ── v2 Core: GratiFlow Pipeline (4-agent) ────────────────────────────────────

def run_pipeline_session(
    api_key: str,
    persona_id: str,
    session_num: int,
    user_entry: str,
    session_history: list,
    current_observed_skill: float,
    condition: str,
) -> dict:
    """
    Run one session through the 4-agent GratiFlow pipeline.

    v2.1 changes:
      - affect prompt uses rubric + few-shot (SYSTEM_PROMPTS["affect"] is v2.1 version)
      - SRR calculation uses reframe_count / neg_count (Section 1.5)
      - srr_reasoning is stored in session record

    Args:
        condition: "adaptive-fading" or "fixed-high"
                   Used ONLY to determine scaffold_level for pipeline agents.
                   NOT passed to user entry generation function.

    Returns:
        session_record: dict with all pipeline outputs and metrics
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Determine scaffold level from condition
    if condition == "fixed-high":
        scaffold_level = "high"
    elif condition == "adaptive-fading":
        scaffold_level = get_scaffold_level_from_skill(current_observed_skill)
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    stage = get_curriculum_stage(current_observed_skill)

    print(f"      obs_s={current_observed_skill:.3f} → scaffold={scaffold_level}, stage={stage}")

    # ── Agent 1: Savoring ──────────────────────────────────────────────────
    sav_result = call_openai(
        api_key,
        [
            {"role": "system", "content": SYSTEM_PROMPTS["savoring"]},
            {"role": "user", "content": user_entry},
        ],
        MAX_COMPLETION_TOKENS_AGENT,
    )
    savoring_response = sav_result["content"] or "[savoring error]"
    time.sleep(0.3)

    # ── Agent 2: Affect-Analysis (v2.1: rubric + few-shot prompt) ─────────
    affect_result = call_openai(
        api_key,
        [
            {"role": "system", "content": SYSTEM_PROMPTS["affect"]},
            {"role": "user", "content": user_entry},
        ],
        MAX_COMPLETION_TOKENS_AFFECT,
    )
    affect_data = parse_affect_json(affect_result["content"])

    if affect_data is None:
        print(f"      [WARNING] Affect JSON parse failed for {persona_id} S{session_num}. Fallback.")
        affect_data = {
            "mood": 5,
            "pos_count": 0,
            "neg_count": 1,
            "reframe_count": 0,
            "spontaneous_reframe": False,
            "srr_reasoning": "[parse error fallback]",
            "keywords": ["不明"],
            "summary": "感情分析に失敗しました（フォールバック）。",
            "_fallback": True,
        }
    time.sleep(0.3)

    mood = int(affect_data.get("mood", 5))
    pos_count = int(affect_data.get("pos_count", 0))
    neg_count_detected = int(affect_data.get("neg_count", 0))
    spontaneous_reframe = bool(affect_data.get("spontaneous_reframe", False))
    srr_reasoning = str(affect_data.get("srr_reasoning", ""))

    # v2.1: reframe_count is clamped to [0, neg_count_detected]
    reframe_count_raw = int(affect_data.get("reframe_count", 1 if spontaneous_reframe else 0))
    reframe_count = min(reframe_count_raw, neg_count_detected) if neg_count_detected > 0 else 0

    # ── Agent 3: Reframing-Coach (only if negatives detected) ─────────────
    reframing_response = None
    if neg_count_detected > 0:
        ref_prompt_key = f"reframing_{scaffold_level}"
        ref_result = call_openai(
            api_key,
            [
                {"role": "system", "content": SYSTEM_PROMPTS[ref_prompt_key]},
                {"role": "user", "content": user_entry},
            ],
            MAX_COMPLETION_TOKENS_AGENT,
        )
        reframing_response = ref_result["content"] or "[reframing error]"
        time.sleep(0.3)

    # ── Skill update (v2.1: uses reframe_count / neg_count) ───────────────
    new_observed_skill = update_observed_skill(session_history, reframe_count, neg_count_detected)

    # ── Agent 4: Curriculum-Progress ──────────────────────────────────────
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
    time.sleep(0.3)

    # ── v2.1 Change 3: SRR = reframe_count / neg_count (Section 1.5) ─────
    if neg_count_detected > 0:
        spontaneous_rate = reframe_count / neg_count_detected
    else:
        spontaneous_rate = float("nan")  # undefined when no negatives

    session_record = {
        "persona_id": persona_id,
        "session": session_num,
        "condition": condition,
        "user_entry": user_entry,
        "savoring_response": savoring_response,
        "affect_analysis": affect_data,
        "reframing_response": reframing_response,
        "curriculum_response": curriculum_response,
        "scaffold_level": scaffold_level,
        "stage": stage,
        "observed_skill_before": round(current_observed_skill, 6),
        "observed_skill_after": round(new_observed_skill, 6),
        "mood": mood,
        "pos_count": pos_count,
        "neg_count_detected": neg_count_detected,
        "reframe_count": reframe_count,
        "spontaneous_reframe": spontaneous_reframe,
        "srr_reasoning": srr_reasoning,
        "spontaneous_rate": round(spontaneous_rate, 6) if not math.isnan(spontaneous_rate) else None,
        "timestamp": timestamp,
        "note": (
            "SYNTHETIC USER SESSION (v2.1: sequential generation, rubric-based SRR). "
            "Synthetic Users (sequential, N=5). No real participants."
        ),
        "_version": "v2.1",
    }

    return session_record


# ── v2 Main Loop: Per-Persona Sequential Execution ───────────────────────────

def run_persona_condition(
    api_key: str,
    persona: dict,
    condition: str,
    rng: random.Random,
) -> tuple[list, list, list, list]:
    """
    Run 10 sessions sequentially for one persona under one condition.

    Unchanged from v2 except:
      - RNG is initialized with deterministic_hash (hashlib.sha256) instead of hash()
      - Session records include reframe_count and srr_reasoning

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
    observed_skill = persona["latent_skill_0"]  # initialize observed to latent_0

    previous_summary = "This is the first session. No previous sessions."
    session_history_for_obs_skill = []

    session_records = []
    ground_truth_records = []
    prompt_records = []
    raw_api_responses = []

    for session_num in range(1, N_SESSIONS + 1):
        print(f"\n    Session {session_num}/{N_SESSIONS} [{condition}]:")

        # ── Step 1: Determine scaffold_level for this session ──────────────
        # (used to compute p_attempt; NOT passed to generation prompt)
        if condition == "fixed-high":
            scaffold_level_for_attempt = "high"
        elif condition == "adaptive-fading":
            scaffold_level_for_attempt = get_scaffold_level_from_skill(observed_skill)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        # ── Step 2: Compute attempt probability ────────────────────────────
        p_attempt = compute_attempt_probability(latent_skill, scaffold_level_for_attempt, p_attempt_base)

        # ── Step 3: Bernoulli draw for did_attempt ─────────────────────────
        did_attempt = rng.random() < p_attempt

        # ── Step 4: Bernoulli draw for attempt_success ────────────────────
        if did_attempt:
            p_success = compute_attempt_success_probability(latent_skill)
            attempt_success = rng.random() < p_success
        else:
            p_success = 0.0
            attempt_success = False

        # ── Step 5: Sample neg_count ───────────────────────────────────────
        neg_count_sampled = sample_neg_count(latent_skill, neg_tendency, rng)

        print(f"      latent_skill={latent_skill:.3f}, scaffold={scaffold_level_for_attempt}, "
              f"p_attempt={p_attempt:.3f}, did_attempt={did_attempt}, "
              f"attempt_success={attempt_success}, neg_count={neg_count_sampled}")

        # ── Step 6: Generate user entry (condition NOT in prompt) ──────────
        entry_text, raw_gen_resp, sys_prompt, usr_msg = generate_user_entry(
            api_key=api_key,
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
            "system_prompt": sys_prompt,
            "user_message": usr_msg,
            "latent_skill": round(latent_skill, 6),
            "did_attempt": did_attempt,
            "attempt_success": attempt_success,
            "neg_count_sampled": neg_count_sampled,
            "note": "SYNTHETIC: Generation prompt contains NO condition name. v2.1 protocol.",
        })

        if raw_gen_resp:
            raw_api_responses.append({
                "persona_id": pid,
                "condition": condition,
                "session": session_num,
                "type": "user_entry_generation",
                "raw_response": raw_gen_resp,
            })

        time.sleep(0.5)

        # ── Step 7: Run 4-agent pipeline ───────────────────────────────────
        session_rec = run_pipeline_session(
            api_key=api_key,
            persona_id=pid,
            session_num=session_num,
            user_entry=entry_text,
            session_history=session_history_for_obs_skill,
            current_observed_skill=observed_skill,
            condition=condition,
        )

        # Attach ground-truth context to session record
        session_rec["latent_skill_before"] = round(latent_skill, 6)
        session_rec["did_attempt_gt"] = did_attempt
        session_rec["attempt_success_gt"] = attempt_success
        session_rec["p_attempt_gt"] = round(p_attempt, 6)
        session_rec["neg_count_sampled"] = neg_count_sampled

        # ── Step 8: Update latent_skill ────────────────────────────────────
        observed_ai_model = not did_attempt  # if not attempting, user observes AI's model
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

        # Update observed skill (from pipeline's Affect-Analysis output)
        observed_skill = session_rec["observed_skill_after"]

        # ── Step 9: Generate summary for next session ──────────────────────
        reframing_resp = session_rec.get("reframing_response")
        previous_summary = make_previous_summary(
            session_num=session_num,
            user_entry=entry_text,
            ai_reframing_response=reframing_resp,
            scaffold_level=scaffold_level_for_attempt,
            latent_skill=latent_skill_new,
        )

        # Update state for next session
        latent_skill = latent_skill_new
        session_history_for_obs_skill.append(session_rec)
        session_records.append(session_rec)

        # Ground truth record
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
        })

        srr_str = f"{srr_val:.2f}" if srr_val is not None else "nan"
        print(f"      latent: {session_rec['latent_skill_before']:.3f} → {session_rec['latent_skill_after']:.3f} | "
              f"obs_s: {session_rec['observed_skill_before']:.3f} → {session_rec['observed_skill_after']:.3f} | "
              f"SRR(LLM): {srr_str} [reframe_count={session_rec.get('reframe_count')}, neg_count={session_rec.get('neg_count_detected')}]")

    return session_records, ground_truth_records, prompt_records, raw_api_responses


# ── Summary Statistics (v2.1) ──────────────────────────────────────────────────

def compute_summary_v2_1(
    results_fading: dict,
    results_fixed: dict,
    personas: list,
) -> dict:
    """
    Compute summary statistics for v2.1 ablation comparison.

    Pre-committed analysis per protocol Section 3.2:
      delta-SRR = mean(SRR, sessions 8-10) - mean(SRR, sessions 1-3)
      SRR = reframe_count / neg_count (v2.1 rubric-based)
      Sessions with neg_count_detected = 0 are excluded (SRR undefined).
      Direction consistency: A > B in >= 3/5 personas.
      Hypothesis support: direction_consistent AND mean A > mean B.
    """
    summary = {
        "description": (
            "Final ablation study: Condition A (adaptive-fading) vs Condition B (fixed-high). "
            "v2.1: Sequential generation, latent-skill-driven, rubric-based SRR detector. "
            "Synthetic Users (sequential, N=5). No real participants. "
            "This is the final experiment iteration (打ち切り)."
        ),
        "version": "v2.1",
        "pre_committed_analysis": {
            "delta_srr_definition": "mean(SRR, sessions 8-10) - mean(SRR, sessions 1-3)",
            "srr_definition_v2_1": "SRR = reframe_count / neg_count_detected (rubric-based Affect-Analysis)",
            "srr_filter": "Sessions with neg_count_detected = 0 are excluded (SRR undefined)",
            "direction_consistency_threshold": "A > B in >= 3/5 personas",
            "hypothesis_support_criteria": "direction_consistent AND mean_delta_A > mean_delta_B",
            "no_significance_test": "N=5 is underpowered for statistical testing",
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
            """
            Mean SRR for sessions [start, end] (1-indexed).
            Excludes sessions where neg_count_detected = 0 (SRR undefined).
            Also excludes sessions where spontaneous_rate is None.
            """
            subset = [
                s for s in sessions
                if start <= s["session"] <= end
                and s.get("neg_count_detected", 0) > 0
                and s.get("spontaneous_rate") is not None
            ]
            if not subset:
                return float("nan")
            return sum(s["spontaneous_rate"] for s in subset) / len(subset)

        srr_early_a = srr_for_sessions(sessions_a, 1, 3)
        srr_late_a = srr_for_sessions(sessions_a, 8, 10)
        srr_early_b = srr_for_sessions(sessions_b, 1, 3)
        srr_late_b = srr_for_sessions(sessions_b, 8, 10)

        delta_a = srr_late_a - srr_early_a if (
            not math.isnan(srr_early_a) and not math.isnan(srr_late_a)
        ) else float("nan")
        delta_b = srr_late_b - srr_early_b if (
            not math.isnan(srr_early_b) and not math.isnan(srr_late_b)
        ) else float("nan")

        latent_skill_traj_a = [s["latent_skill_after"] for s in sessions_a]
        latent_skill_traj_b = [s["latent_skill_after"] for s in sessions_b]
        p_attempt_traj_a = [s.get("p_attempt_gt", float("nan")) for s in sessions_a]
        p_attempt_traj_b = [s.get("p_attempt_gt", float("nan")) for s in sessions_b]

        # Session 1 observed_skill ceiling check (protocol Section 5.3)
        ceiling_threshold = SCAFFOLD_THRESHOLDS["high"]  # 0.35 = transition to "mid"
        s1_obs_a = next((s["observed_skill_after"] for s in sessions_a if s["session"] == 1), None)
        s1_obs_b = next((s["observed_skill_after"] for s in sessions_b if s["session"] == 1), None)
        s1_ceiling_a = s1_obs_a is not None and s1_obs_a >= ceiling_threshold
        s1_ceiling_b = s1_obs_b is not None and s1_obs_b >= ceiling_threshold

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
                "latent_skill_trajectory": [round(v, 4) for v in latent_skill_traj_a],
                "p_attempt_trajectory": [round(v, 4) for v in p_attempt_traj_a],
                "srr_per_session": [
                    round(s["spontaneous_rate"], 4) if s.get("spontaneous_rate") is not None else None
                    for s in sessions_a
                ],
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_a],
                "session1_obs_skill_ceiling": s1_ceiling_a,
            },
            "condition_fixed_high": {
                "srr_early_mean": _r(srr_early_b),
                "srr_late_mean": _r(srr_late_b),
                "delta_srr": _r(delta_b),
                "latent_skill_trajectory": [round(v, 4) for v in latent_skill_traj_b],
                "p_attempt_trajectory": [round(v, 4) for v in p_attempt_traj_b],
                "srr_per_session": [
                    round(s["spontaneous_rate"], 4) if s.get("spontaneous_rate") is not None else None
                    for s in sessions_b
                ],
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_b],
                "session1_obs_skill_ceiling": s1_ceiling_b,
            },
            "delta_srr_advantage_A_over_B": _r(
                delta_a - delta_b
                if (not math.isnan(delta_a) and not math.isnan(delta_b))
                else float("nan")
            ),
        }

    # Aggregate statistics
    valid_pairs = [
        (a, b) for a, b in zip(all_delta_fading, all_delta_fixed)
        if not math.isnan(a) and not math.isnan(b)
    ]
    n_valid = len(valid_pairs)
    n_fading_higher = sum(1 for a, b in valid_pairs if a > b)

    mean_delta_a = sum(a for a, _ in valid_pairs) / n_valid if n_valid > 0 else float("nan")
    mean_delta_b = sum(b for _, b in valid_pairs) / n_valid if n_valid > 0 else float("nan")

    direction_consistent = n_fading_higher >= 3  # >= 3/5 per pre-commitment

    hypothesis_supported = direction_consistent and (
        not math.isnan(mean_delta_a) and not math.isnan(mean_delta_b)
        and mean_delta_a > mean_delta_b
    )

    # Session 1 ceiling summary across all personas
    n_s1_ceiling_a = sum(
        1 for pid_data in summary["personas"].values()
        if pid_data["condition_adaptive_fading"].get("session1_obs_skill_ceiling", False)
    )
    n_s1_ceiling_b = sum(
        1 for pid_data in summary["personas"].values()
        if pid_data["condition_fixed_high"].get("session1_obs_skill_ceiling", False)
    )

    summary["aggregate"] = {
        "n_personas_total": len(personas),
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
        "hypothesis_supported": hypothesis_supported,
        "session1_ceiling_check": {
            "threshold": ceiling_threshold,
            "n_personas_ceiling_adaptive_fading": n_s1_ceiling_a,
            "n_personas_ceiling_fixed_high": n_s1_ceiling_b,
            "note": (
                "Number of personas where Session 1 observed_skill >= 0.35 (scaffold transition threshold). "
                "Non-zero count indicates residual ceiling effect even with rubric-based SRR."
            ),
        },
        "interpretation": (
            f"Adaptive-fading shows higher delta-SRR in {n_fading_higher}/{n_valid} valid personas. "
            f"Pre-committed threshold: >= 3/5. "
            f"Hypothesis {'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}: "
            f"direction_consistent={direction_consistent}, mean_A > mean_B = {not math.isnan(mean_delta_a) and not math.isnan(mean_delta_b) and mean_delta_a > mean_delta_b}. "
            f"No significance test (N=5). "
            f"This is the final experiment iteration (v2.1 打ち切り)."
        ),
    }

    return summary


# ── Environment Record ─────────────────────────────────────────────────────────

def save_environment_v2_1() -> None:
    """Record experiment environment for reproducibility."""
    env_record = {
        "version": "v2.1",
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
            "rng_change_from_v2": "hash() replaced with hashlib.sha256 (PYTHONHASHSEED-independent)",
        },
        "scaffold_thresholds": SCAFFOLD_THRESHOLDS,
        "scaffold_attempt_multiplier": SCAFFOLD_ATTEMPT_MULTIPLIER,
        "moving_avg_window": MOVING_AVG_WINDOW,
        "curriculum_thresholds": CURRICULUM_THRESHOLDS,
        "personas": ["P1", "P2", "P3", "P4", "P5"],
        "sessions_per_persona": N_SESSIONS,
        "conditions": CONDITIONS,
        "condition_descriptions": {
            "adaptive-fading": "Condition A: scaffold_level adapts to observed skill s",
            "fixed-high": "Condition B: scaffold_level is always 'high'",
        },
        "srr_definition_v2_1": {
            "formula": "SRR = reframe_count / neg_count_detected",
            "change_from_v2": "pos_count / neg_count → reframe_count / neg_count",
            "when_undefined": "neg_count_detected = 0 → SRR = NaN (excluded from delta-SRR)",
        },
        "affect_prompt_change": "v2.1: rubric + few-shot (cognitive reappraisal, R1-R3 + F1-F5) replaces v2 ambiguous prompt",
        "generation_prompt_leakage_prevention": {
            "forbidden_strings": FORBIDDEN_PROMPT_STRINGS,
            "assertion_enabled": True,
        },
        "synthetic_user_note": (
            "Synthetic Users (sequential, N=5). "
            "All figures must include this label. No real participants."
        ),
        "final_experiment": True,
        "protocol_reference": "evaluation_protocol_v2_1.md (the research team, 2026-06-04)",
    }

    out_path = DATA_PROCESSED_V2_1 / "experiment_environment_v2_1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, ensure_ascii=False, indent=2)
    print(f"Environment recorded: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("GratiFlow v2.1 Evaluation — Final Experiment (打ち切り)")
    print("IMPORTANT: SYNTHETIC USERS ONLY. Synthetic Users (sequential, N=5).")
    print(f"Model: {MODEL} | EXPERIMENT_SEED={EXPERIMENT_SEED} | SYNTHETIC_USER_SEED={SYNTHETIC_USER_SEED}")
    print("Protocol: evaluation_protocol_v2_1.md (the research team, 2026-06-04, PI-approved)")
    print("v2.1 changes: rubric-based SRR detector, hashlib RNG, reframe_count/neg_count SRR")
    print("This is the FINAL iteration. Results reported regardless of direction.")
    print("=" * 70)

    # Load API key
    print("\nLoading API key...")
    api_key = load_api_key()
    print("  API key loaded.")

    # Load personas (unchanged from v2)
    with open(PERSONAS_FILE) as f:
        persona_data = json.load(f)
    personas = persona_data["personas"]
    print(f"  Loaded {len(personas)} personas from {PERSONAS_FILE}")

    # Create output directories
    DATA_PROCESSED_V2_1.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        (DATA_PROCESSED_V2_1 / "results" / f"condition_{cond}").mkdir(parents=True, exist_ok=True)

    # Save environment
    save_environment_v2_1()

    # ── Run experiments ────────────────────────────────────────────────────
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

            # v2.1 Change 2: Deterministic RNG using hashlib.sha256 (protocol Section 5.2)
            # Replaces v2's hash() which was PYTHONHASHSEED-dependent
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

            # Save per-persona results immediately (fail-safe for long runs)
            out_path = DATA_PROCESSED_V2_1 / "results" / f"condition_{condition}" / f"{pid}_sessions.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            print(f"\n    Saved: {out_path}")

        all_results[condition] = condition_results

    # ── Save consolidated outputs ──────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()

    # Ground truth (all personas x conditions)
    gt_path = DATA_PROCESSED_V2_1 / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": (
                    "Ground truth: latent_skill, p_attempt, did_attempt, attempt_success for all v2.1 sessions. "
                    "Also includes srr_reasoning and reframe_count_llm from rubric-based Affect-Analysis."
                ),
                "generated_at": timestamp,
                "version": "v2.1",
                "note": "SYNTHETIC. Values are simulation-internal (not LLM-judged) + LLM judgment fields.",
            },
            "records": all_ground_truth,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nGround truth saved: {gt_path}")

    # All generation prompts (for non-leakage verification)
    prompts_path = DATA_PROCESSED_V2_1 / "synthetic_user_prompts_v2_1.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "All generation prompts used for v2.1 synthetic user entries. Verify no condition name is present.",
                "generated_at": timestamp,
                "version": "v2.1",
                "leakage_check": "No condition names, scaffold levels, or condition-implying language in user_message.",
                "total_prompts": len(all_prompts),
            },
            "prompts": all_prompts,
        }, f, ensure_ascii=False, indent=2)
    print(f"Generation prompts saved: {prompts_path}")

    # Raw API responses (immutable raw data — DO NOT EDIT)
    raw_path = DATA_RAW / "synthetic_users_v2_1_raw_responses.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "Raw API responses for v2.1 synthetic user generation. DO NOT EDIT.",
                "generated_at": timestamp,
                "version": "v2.1",
            },
            "responses": all_raw_responses,
        }, f, ensure_ascii=False, indent=2)
    print(f"Raw responses saved (do not edit): {raw_path}")

    # ── Compute and save summary ───────────────────────────────────────────
    print("\nComputing summary statistics (pre-committed analysis, v2.1)...")
    summary = compute_summary_v2_1(
        results_fading=all_results.get("adaptive-fading", {}),
        results_fixed=all_results.get("fixed-high", {}),
        personas=personas,
    )

    # Save all entries
    entries_path = DATA_PROCESSED_V2_1 / "synthetic_user_entries_v2_1.json"
    all_entries = []
    for condition, persona_sessions in all_results.items():
        for pid, sessions in persona_sessions.items():
            for s in sessions:
                all_entries.append({
                    "persona_id": pid,
                    "condition": condition,
                    "session": s["session"],
                    "user_entry": s["user_entry"],
                    "latent_skill_before": s.get("latent_skill_before"),
                    "latent_skill_after": s.get("latent_skill_after"),
                    "did_attempt_gt": s.get("did_attempt_gt"),
                    "attempt_success_gt": s.get("attempt_success_gt"),
                    "neg_count_sampled": s.get("neg_count_sampled"),
                    "reframe_count_llm": s.get("reframe_count"),
                    "srr_reasoning": s.get("srr_reasoning", ""),
                    "synthetic": True,
                    "version": "v2.1",
                    "note": "Synthetic Users (sequential, N=5)",
                })
    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "v2.1 synthetic user journal entries with ground-truth metadata and SRR reasoning.",
                "generated_at": timestamp,
                "version": "v2.1",
                "total_entries": len(all_entries),
                "synthetic_note": "Synthetic Users (sequential, N=5). No real participants.",
            },
            "entries": all_entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"Entries saved: {entries_path}")

    summary_path = DATA_PROCESSED_V2_1 / "results" / "summary_statistics_v2_1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")

    # ── Print key results ──────────────────────────────────────────────────
    agg = summary["aggregate"]
    print("\n" + "=" * 70)
    print("KEY RESULTS (v2.1: Rubric-based SRR, Synthetic Users, N=5 personas)")
    print("Synthetic Users (sequential, N=5) — Final Experiment (打ち切り)")
    print("=" * 70)
    a_val = agg.get("mean_delta_srr_adaptive_fading")
    b_val = agg.get("mean_delta_srr_fixed_high")
    adv_val = agg.get("mean_delta_srr_advantage_A_over_B")
    print(f"Mean delta-SRR (adaptive-fading):  {f'{a_val:+.4f}' if a_val is not None else 'null'}")
    print(f"Mean delta-SRR (fixed-high):        {f'{b_val:+.4f}' if b_val is not None else 'null'}")
    print(f"Advantage A over B:                 {f'{adv_val:+.4f}' if adv_val is not None else 'null'}")
    print(f"Personas A > B:                     {agg['n_personas_A_higher_delta_srr']}/{agg['n_personas_valid_delta']}")
    print(f"Direction consistent (pre-commit):  {agg['direction_consistent_pre_committed']}")
    print(f"Hypothesis supported (v2.1):        {agg['hypothesis_supported']}")
    print(f"\nSession 1 ceiling check (obs_s >= {agg['session1_ceiling_check']['threshold']}):")
    print(f"  adaptive-fading: {agg['session1_ceiling_check']['n_personas_ceiling_adaptive_fading']}/5 personas at ceiling")
    print(f"  fixed-high:      {agg['session1_ceiling_check']['n_personas_ceiling_fixed_high']}/5 personas at ceiling")

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
    print("\nNOTE: N=5 is underpowered for significance testing.")
    print("Effect direction and consistency reported as PoC evidence only.")
    print("Results are honest — no manipulation per research integrity principles.")
    print("This is the FINAL experiment. No further iterations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
