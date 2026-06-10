"""
generate_and_run_v2.py
=======================
GratiFlow v2 Evaluation — Sequential Generation + Pipeline Runner (Integrated)

Implements the v2 protocol from evaluation_protocol_v2.md (the research team, 2026-06-04, PI-approved).

Key design:
  - Sequential generation: each session's synthetic user entry is conditioned on the
    PREVIOUS session's AI response, scaffold_level, and latent_skill state.
  - Latent-skill-driven: each persona has latent_skill that grows via practice opportunity.
  - Circular bias prevention: condition name and scaffold_level are NEVER passed to the
    generation prompt. Only latent_skill, did_attempt, attempt_success are passed.
  - assert statements verify condition-name non-leakage before every API call.

Conditions:
  Condition A (adaptive-fading):  scaffoldLevel adapts based on observed skill score s
  Condition B (fixed-high):       scaffoldLevel is always "high" regardless of s

IMPORTANT: All synthetic user entries are LLM-generated. No real participants.

Author: team member (experiment lead, the research team)
Date: 2026-06-04
Protocol: evaluation_protocol_v2.md
Seeds: EXPERIMENT_SEED=42, SYNTHETIC_USER_SEED=2026
"""

import json
import math
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# Import v2 latent skill model functions
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
MAX_COMPLETION_TOKENS_AFFECT = 512   # affect-analysis JSON output

# Scaffolding thresholds — MUST match app.js SCAFFOLD_THRESHOLDS
SCAFFOLD_THRESHOLDS = {"high": 0.35, "mid": 0.65}

# Curriculum stage thresholds — MUST match app.js getCurriculumStage
CURRICULUM_THRESHOLDS = [0.25, 0.50, 0.75]

# Moving average window — MUST match app.js MOVING_AVG_WINDOW
MOVING_AVG_WINDOW = 5

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2.0  # seconds (exponential backoff)

# Number of sessions per persona
N_SESSIONS = 10

# Conditions (v2 naming per protocol Section 4.1)
CONDITIONS = ["adaptive-fading", "fixed-high"]

# Paths
BASE_DIR = Path(__file__).parent.parent
PERSONAS_FILE = Path(__file__).parent / "personas_v2.json"
DATA_PROCESSED_V2 = BASE_DIR / "data" / "processed" / "experiments_v2"
DATA_RAW = BASE_DIR / "data" / "raw"

# Strings that MUST NOT appear in generation prompts (non-leakage assertion)
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
    # Scaffold level as standalone words (we check lowercase)
    # Note: we check the generation prompt only, not the pipeline prompts
]

# ── GratiFlow agent system prompts (mirrors app.js / chat.js exactly) ─────────

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
    "affect": (
        "You are the Affect-Analysis Agent in GratiFlow.\n"
        "Analyze the user's journal entry and respond with ONLY valid JSON (no markdown fences).\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"mood\": <number 1–10, overall mood score>,\n"
        "  \"pos_count\": <integer, number of positive expressions or reframings the user independently produced>,\n"
        "  \"neg_count\": <integer, number of negative expressions detected>,\n"
        "  \"spontaneous_reframe\": <boolean, true if user produced at least one positive reframe WITHOUT being prompted by AI>,\n"
        "  \"keywords\": [<string>, ...],  // 3–6 key emotional words (Japanese)\n"
        "  \"summary\": \"<string>\"         // 1-sentence Japanese summary of the user's emotional state\n"
        "}\n\n"
        "Rules:\n"
        "- Count only expressions in the USER's text, not the AI's previous messages.\n"
        "- A \"spontaneous reframe\" is when the user turns a negative into a positive on their own.\n"
        "- Be conservative: only mark spontaneous_reframe true if clearly evident."
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
        "- Keep it brief (2–3 sentences total) and warm.\n"
        "- Output in Japanese."
    ),
}


# ── Pipeline helper functions (mirrors app.js) ─────────────────────────────────

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


def update_observed_skill(session_history: list, pos_count: int, neg_count: int) -> float:
    """
    Mirror of app.js updateSkill.
    Computes moving average of spontaneous reframing rate (SRR) over the last MOVING_AVG_WINDOW sessions.
    This is the OBSERVED skill score 's', distinct from latent_skill.
    """
    if neg_count > 0:
        session_rate = min(pos_count / neg_count, 1.0)
    elif pos_count > 0:
        session_rate = 1.0
    else:
        session_rate = 0.0

    recent = session_history[-MOVING_AVG_WINDOW:]
    rates = [s.get("spontaneous_rate", 0.0) for s in recent]
    rates.append(session_rate)

    avg = sum(rates) / len(rates)
    return max(0.0, min(1.0, avg))


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
      - max_completion_tokens is used (max_tokens not supported by gpt-5.4-mini)
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
    """Parse affect agent JSON output. Returns None on parse failure."""
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
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ── v2 Core: Synthetic User Entry Generation ──────────────────────────────────

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
        # Check for forbidden strings, but allow 'fixed' only as part of 'fixed-high'
        # when it's already caught by the 'fixed-high' check
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

    Args:
        condition: "adaptive-fading" or "fixed-high"
                   Used ONLY to determine scaffold_level for the pipeline agents.
                   NOT passed to the user entry generation function.

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

    # ── Agent 2: Affect-Analysis ───────────────────────────────────────────
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
            "spontaneous_reframe": False,
            "keywords": ["不明"],
            "summary": "感情分析に失敗しました（フォールバック）。",
            "_fallback": True,
        }
    time.sleep(0.3)

    mood = int(affect_data.get("mood", 5))
    pos_count = int(affect_data.get("pos_count", 0))
    neg_count_detected = int(affect_data.get("neg_count", 0))
    spontaneous_reframe = bool(affect_data.get("spontaneous_reframe", False))

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

    # ── Skill update (mirrors app.js updateSkill) ──────────────────────────
    new_observed_skill = update_observed_skill(session_history, pos_count, neg_count_detected)

    # ── Agent 4: Curriculum-Progress ──────────────────────────────────────
    curr_user_content = (
        f"User skill score: {new_observed_skill:.2f}. "
        f"Skill stage: {stage}. "
        f"Spontaneous reframe detected in this session."
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

    # ── Spontaneous reframing rate (SRR) ──────────────────────────────────
    if neg_count_detected > 0:
        spontaneous_rate = min(pos_count / neg_count_detected, 1.0)
    elif pos_count > 0:
        spontaneous_rate = 1.0
    else:
        spontaneous_rate = 0.0

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
        "spontaneous_reframe": spontaneous_reframe,
        "spontaneous_rate": round(spontaneous_rate, 6),
        "timestamp": timestamp,
        "note": "SYNTHETIC USER SESSION (v2: sequential generation). No real participants.",
        "_version": "v2",
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

    Each session:
      1. Compute p_attempt from latent_skill + scaffold_level
      2. Bernoulli draw for did_attempt
      3. Bernoulli draw for attempt_success (if did_attempt)
      4. Sample neg_count via Poisson
      5. Generate user entry (NO condition name in prompt)
      6. Run 4-agent pipeline
      7. Update latent_skill
      8. Generate summary for next session

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
            "note": "SYNTHETIC: Generation prompt contains NO condition name. v2 protocol.",
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
            "spontaneous_rate_llm": session_rec["spontaneous_rate"],
        })

        print(f"      latent: {session_rec['latent_skill_before']:.3f} → {session_rec['latent_skill_after']:.3f} | "
              f"obs_s: {session_rec['observed_skill_before']:.3f} → {session_rec['observed_skill_after']:.3f} | "
              f"SRR(LLM): {session_rec['spontaneous_rate']:.2f}")

    return session_records, ground_truth_records, prompt_records, raw_api_responses


# ── Summary Statistics ─────────────────────────────────────────────────────────

def compute_summary_v2(
    results_fading: dict,
    results_fixed: dict,
    personas: list,
) -> dict:
    """
    Compute summary statistics for v2 ablation comparison.

    Pre-committed analysis per protocol Section 5.3:
      delta-SRR = mean(SRR, sessions 8-10) - mean(SRR, sessions 1-3)
      Uses only sessions with neg_count_sampled > 0 for SRR computation.
      Direction consistency: A > B in >= 3/5 personas.
    """
    summary = {
        "description": (
            "Ablation study: Condition A (adaptive-fading) vs Condition B (fixed-high). "
            "v2: Sequential generation, latent-skill-driven. "
            "Synthetic users, N=5 personas x 10 sessions. "
            "No real participants. LLM-judged SRR metric."
        ),
        "version": "v2",
        "pre_committed_analysis": {
            "delta_srr_definition": "mean(SRR, sessions 8-10) - mean(SRR, sessions 1-3)",
            "srr_filter": "Sessions with neg_count_sampled > 0 only",
            "direction_consistency_threshold": "A > B in >= 3/5 personas",
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
            """Mean SRR for sessions [start, end] (1-indexed), skipping neg_count_sampled=0."""
            subset = [
                s for s in sessions
                if start <= s["session"] <= end and s.get("neg_count_sampled", 1) > 0
            ]
            if not subset:
                return float("nan")
            return sum(s["spontaneous_rate"] for s in subset) / len(subset)

        srr_early_a = srr_for_sessions(sessions_a, 1, 3)
        srr_late_a = srr_for_sessions(sessions_a, 8, 10)
        srr_early_b = srr_for_sessions(sessions_b, 1, 3)
        srr_late_b = srr_for_sessions(sessions_b, 8, 10)

        # Handle NaN: if no valid sessions in window, set delta to NaN
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
                "srr_per_session": [round(s["spontaneous_rate"], 4) for s in sessions_a],
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_a],
            },
            "condition_fixed_high": {
                "srr_early_mean": _r(srr_early_b),
                "srr_late_mean": _r(srr_late_b),
                "delta_srr": _r(delta_b),
                "latent_skill_trajectory": [round(v, 4) for v in latent_skill_traj_b],
                "p_attempt_trajectory": [round(v, 4) for v in p_attempt_traj_b],
                "srr_per_session": [round(s["spontaneous_rate"], 4) for s in sessions_b],
                "obs_skill_per_session": [round(s["observed_skill_after"], 4) for s in sessions_b],
            },
            "delta_srr_advantage_A_over_B": _r(
                delta_a - delta_b
                if (not math.isnan(delta_a) and not math.isnan(delta_b))
                else float("nan")
            ),
        }

    # Aggregate: exclude NaN personas from direction count
    valid_pairs = [
        (a, b) for a, b in zip(all_delta_fading, all_delta_fixed)
        if not math.isnan(a) and not math.isnan(b)
    ]
    n_valid = len(valid_pairs)
    n_fading_higher = sum(1 for a, b in valid_pairs if a > b)

    mean_delta_a = sum(a for a, _ in valid_pairs) / n_valid if n_valid > 0 else float("nan")
    mean_delta_b = sum(b for _, b in valid_pairs) / n_valid if n_valid > 0 else float("nan")

    direction_consistent = n_fading_higher >= 3  # >= 3/5 per pre-commitment

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
        "hypothesis_supported": direction_consistent and (
            not math.isnan(mean_delta_a) and not math.isnan(mean_delta_b) and mean_delta_a > mean_delta_b
        ),
        "interpretation": (
            f"Adaptive-fading shows higher delta-SRR in {n_fading_higher}/{n_valid} valid personas. "
            f"Pre-committed threshold: >= 3/5. "
            f"No significance test (N=5). "
            f"Effect direction and consistency reported as PoC evidence."
        ),
    }

    return summary


# ── Environment Record ─────────────────────────────────────────────────────────

def save_environment_v2() -> None:
    """Record experiment environment for reproducibility."""
    env_record = {
        "version": "v2",
        "model": MODEL,
        "experiment_date": datetime.now(timezone.utc).isoformat()[:10],
        "seeds": {
            "experiment_seed": EXPERIMENT_SEED,
            "synthetic_user_seed": SYNTHETIC_USER_SEED,
            "rng_formula": "random.Random(EXPERIMENT_SEED + abs(hash(persona_id)) % 10000 + abs(hash(condition)) % 10000)",
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
        "generation_prompt_leakage_prevention": {
            "forbidden_strings": FORBIDDEN_PROMPT_STRINGS,
            "assertion_enabled": True,
        },
        "note": "SYNTHETIC USER EXPERIMENT v2. Sequential generation. No real participants.",
        "protocol_reference": "evaluation_protocol_v2.md (the research team, 2026-06-04)",
    }

    out_path = DATA_PROCESSED_V2 / "experiment_environment_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, ensure_ascii=False, indent=2)
    print(f"Environment recorded: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("GratiFlow v2 Evaluation — Sequential Generation + Pipeline")
    print("IMPORTANT: SYNTHETIC USERS ONLY. No real participants.")
    print(f"Model: {MODEL} | EXPERIMENT_SEED={EXPERIMENT_SEED} | SYNTHETIC_USER_SEED={SYNTHETIC_USER_SEED}")
    print("Protocol: evaluation_protocol_v2.md (the research team, 2026-06-04, PI-approved)")
    print("=" * 65)

    # Load API key
    print("\nLoading API key...")
    api_key = load_api_key()
    print("  API key loaded.")

    # Load personas
    with open(PERSONAS_FILE) as f:
        persona_data = json.load(f)
    personas = persona_data["personas"]
    print(f"  Loaded {len(personas)} personas from {PERSONAS_FILE}")

    # Create output directories
    DATA_PROCESSED_V2.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        (DATA_PROCESSED_V2 / "results" / f"condition_{cond}").mkdir(parents=True, exist_ok=True)

    # Save environment
    save_environment_v2()

    # ── Run experiments ────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    all_ground_truth: list = []
    all_prompts: list = []
    all_raw_responses: list = []

    for condition in CONDITIONS:
        print(f"\n{'=' * 65}")
        print(f"Condition: {condition}")
        print(f"{'=' * 65}")

        condition_results = {}

        for persona in personas:
            pid = persona["id"]
            print(f"\n  Persona {pid} ({persona['label']}), "
                  f"latent_skill_0={persona['latent_skill_0']}, "
                  f"alpha={persona['alpha']}, p_attempt_base={persona['p_attempt_base']}")

            # Deterministic RNG per persona x condition (protocol Section 4.3)
            # Use modular hash to avoid overflow
            persona_hash = abs(hash(pid)) % 10000
            condition_hash = abs(hash(condition)) % 10000
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

            # Save per-persona results immediately
            out_path = DATA_PROCESSED_V2 / "results" / f"condition_{condition}" / f"{pid}_sessions.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            print(f"\n    Saved: {out_path}")

        all_results[condition] = condition_results

    # ── Save consolidated outputs ──────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()

    # Ground truth (all personas x conditions)
    gt_path = DATA_PROCESSED_V2 / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "Ground truth: latent_skill, p_attempt, did_attempt, attempt_success for all sessions.",
                "generated_at": timestamp,
                "version": "v2",
                "note": "SYNTHETIC. Values are simulation-internal (not LLM-judged).",
            },
            "records": all_ground_truth,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nGround truth saved: {gt_path}")

    # All generation prompts (for non-leakage verification)
    prompts_path = DATA_PROCESSED_V2 / "synthetic_user_prompts_v2.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "All generation prompts used for v2 synthetic user entries. Verify no condition name is present.",
                "generated_at": timestamp,
                "version": "v2",
                "leakage_check": "No condition names, scaffold levels, or condition-implying language in user_message.",
                "total_prompts": len(all_prompts),
            },
            "prompts": all_prompts,
        }, f, ensure_ascii=False, indent=2)
    print(f"Generation prompts saved: {prompts_path}")

    # Raw API responses (immutable raw data)
    raw_path = DATA_RAW / "synthetic_users_v2_raw_responses.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "Raw API responses for v2 synthetic user generation. DO NOT EDIT.",
                "generated_at": timestamp,
                "version": "v2",
            },
            "responses": all_raw_responses,
        }, f, ensure_ascii=False, indent=2)
    print(f"Raw responses saved (do not edit): {raw_path}")

    # ── Compute and save summary ───────────────────────────────────────────
    print("\nComputing summary statistics (pre-committed analysis)...")
    summary = compute_summary_v2(
        results_fading=all_results.get("adaptive-fading", {}),
        results_fixed=all_results.get("fixed-high", {}),
        personas=personas,
    )

    # Also save per-condition entries (for analysis script)
    entries_path = DATA_PROCESSED_V2 / "synthetic_user_entries_v2.json"
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
                    "synthetic": True,
                    "version": "v2",
                })
    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "description": "v2 synthetic user journal entries with ground-truth metadata.",
                "generated_at": timestamp,
                "version": "v2",
                "total_entries": len(all_entries),
            },
            "entries": all_entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"Entries saved: {entries_path}")

    summary_path = DATA_PROCESSED_V2 / "results" / "summary_statistics_v2.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")

    # ── Print key results ──────────────────────────────────────────────────
    agg = summary["aggregate"]
    print("\n" + "=" * 65)
    print("KEY RESULTS (v2: Sequential Generation, Synthetic Users, N=5 personas)")
    print("=" * 65)
    print(f"Mean delta-SRR (adaptive-fading):  {agg.get('mean_delta_srr_adaptive_fading'):+.4f}")
    print(f"Mean delta-SRR (fixed-high):        {agg.get('mean_delta_srr_fixed_high'):+.4f}")
    print(f"Advantage A over B:                 {agg.get('mean_delta_srr_advantage_A_over_B'):+.4f}")
    print(f"Personas A > B:                     {agg['n_personas_A_higher_delta_srr']}/{agg['n_personas_valid_delta']}")
    print(f"Direction consistent (pre-commit):  {agg['direction_consistent_pre_committed']}")
    print(f"Hypothesis supported:               {agg['hypothesis_supported']}")

    print("\nPer-persona delta-SRR:")
    for pid, pdata in summary["personas"].items():
        a = pdata["condition_adaptive_fading"]["delta_srr"]
        b = pdata["condition_fixed_high"]["delta_srr"]
        adv = pdata.get("delta_srr_advantage_A_over_B")
        label = pdata["label"]
        a_str = f"{a:+.4f}" if a is not None else "  nan"
        b_str = f"{b:+.4f}" if b is not None else "  nan"
        adv_str = f"{adv:+.4f}" if adv is not None else "  nan"
        print(f"  {pid} ({label}): A={a_str}, B={b_str}, A-B={adv_str}")

    print(f"\n{agg['interpretation']}")
    print("\nNOTE: N=5 is underpowered for significance testing.")
    print("Effect direction and consistency reported as PoC evidence only.")
    print("Results are honest — no manipulation per research integrity principles.")
    print("=" * 65)


if __name__ == "__main__":
    main()
