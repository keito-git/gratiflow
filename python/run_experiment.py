"""
run_experiment.py
==================
GratiFlow Phase 1 Evaluation — Main Experiment Runner

Implements the 4-agent pipeline matching app.js / chat.js logic exactly:
  Savoring → Affect-Analysis → Reframing-Coach → Curriculum-Progress

Runs 2 conditions on identical synthetic user inputs (Method 1: independent generation):
  Condition A (fading):     scaffoldLevel adapts based on skill score s
  Condition B (fixed_high): scaffoldLevel is always "high" regardless of s

IMPORTANT: All inputs are SYNTHETIC (LLM-generated). No real participants involved.

Author: team member (experiment lead, the research team)
Date: 2026-06-04
"""

import json
import os
import time
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── Constants (must match app.js exactly) ─────────────────────────────────────

MODEL = "gpt-5.4-mini"
MAX_COMPLETION_TOKENS_AGENT = 1024  # matches chat.js
MAX_COMPLETION_TOKENS_AFFECT = 512  # affect output is short JSON
EXPERIMENT_SEED = 42
API_URL = "https://api.openai.com/v1/chat/completions"

# Scaffolding thresholds — MUST match app.js SCAFFOLD_THRESHOLDS
SCAFFOLD_THRESHOLDS = {"high": 0.35, "mid": 0.65}
# s < 0.35 → high, 0.35 ≤ s < 0.65 → mid, s ≥ 0.65 → low

# Curriculum stage thresholds — MUST match app.js getCurriculumStage
CURRICULUM_THRESHOLDS = [0.25, 0.50, 0.75]

# Moving average window — MUST match app.js MOVING_AVG_WINDOW
MOVING_AVG_WINDOW = 5

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2.0

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "experiments"
PERSONAS_FILE = Path(__file__).parent / "personas.json"

# ── System prompts (reproduced from chat.js) ──────────────────────────────────

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

# ── Pipeline logic (mirrors app.js) ──────────────────────────────────────────

def get_scaffold_level(s: float) -> str:
    """Mirror of app.js getScaffoldLevel."""
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


def update_skill(session_history: list, pos_count: int, neg_count: int) -> float:
    """
    Mirror of app.js updateSkill.
    Uses moving average of spontaneous reframing rate over last MOVING_AVG_WINDOW sessions.
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


def select_system_prompt(agent_type: str, scaffold_level: str) -> str:
    """Mirror of chat.js selectSystemPrompt."""
    if agent_type == "reframing":
        lvl = scaffold_level or "high"
        return SYSTEM_PROMPTS.get(f"reframing_{lvl}", SYSTEM_PROMPTS["reframing_high"])
    return SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS["savoring"])


# ── API call ──────────────────────────────────────────────────────────────────

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
    raise ValueError("OPENAI_API_KEY not found")


def call_agent(
    api_key: str,
    agent_type: str,
    user_text: str,
    scaffold_level: str = "high",
    skill_score: float = 0.0,
    stage: int = 0,
    max_tokens: int = MAX_COMPLETION_TOKENS_AGENT,
) -> dict:
    """
    Call a GratiFlow agent via the OpenAI API.
    Matches chat.js behavior exactly.

    Returns:
        dict with keys: content (str), raw_response (dict), error (str or None)
    """
    system_prompt = select_system_prompt(agent_type, scaffold_level)

    if agent_type == "curriculum":
        # Curriculum agent receives a structured status message (mirrors app.js)
        user_content = (
            f"User skill score: {skill_score:.2f}. "
            f"Skill stage: {stage}. "
            f"Spontaneous reframe detected in this session."
        )
    else:
        user_content = user_text

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_completion_tokens": max_tokens,
        # temperature NOT sent (not supported by gpt-5.4-mini)
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
    # Remove markdown fences if present
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
    try:
        data = json.loads(cleaned)
        # Validate required fields
        required = ["mood", "pos_count", "neg_count", "spontaneous_reframe", "keywords", "summary"]
        if all(k in data for k in required):
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ── Session runner ────────────────────────────────────────────────────────────

def run_session(
    api_key: str,
    persona_id: str,
    session_num: int,
    user_entry: str,
    session_history: list,
    current_skill: float,
    condition: str,  # "fading" or "fixed_high"
    initial_skill: float,
) -> dict:
    """
    Run a single session through the 4-agent pipeline.

    Condition A (fading):    scaffold level adapts to current skill
    Condition B (fixed_high): scaffold level is always "high"
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Determine scaffold level and stage
    if condition == "fixed_high":
        scaffold_level = "high"  # always high regardless of skill
    else:
        scaffold_level = get_scaffold_level(current_skill)

    stage = get_curriculum_stage(current_skill)

    print(f"    s={current_skill:.3f} → scaffold={scaffold_level}, stage={stage}")

    # ── Agent 1: Savoring ─────────────────────────────────────────────────────
    sav_result = call_agent(api_key, "savoring", user_entry,
                            scaffold_level=scaffold_level,
                            skill_score=current_skill, stage=stage)
    savoring_response = sav_result["content"] or "[savoring error]"
    time.sleep(0.3)

    # ── Agent 2: Affect-Analysis ──────────────────────────────────────────────
    affect_result = call_agent(api_key, "affect", user_entry,
                               scaffold_level=scaffold_level,
                               skill_score=current_skill, stage=stage,
                               max_tokens=MAX_COMPLETION_TOKENS_AFFECT)
    affect_data = parse_affect_json(affect_result["content"])

    if affect_data is None:
        print(f"      [WARNING] Affect JSON parse failed for {persona_id} S{session_num}. Using fallback.")
        affect_data = {
            "mood": 5, "pos_count": 0, "neg_count": 1,
            "spontaneous_reframe": False,
            "keywords": ["不明"],
            "summary": "感情分析に失敗しました（フォールバック）。",
            "_fallback": True,
        }
    time.sleep(0.3)

    mood = affect_data.get("mood", 5)
    pos_count = affect_data.get("pos_count", 0)
    neg_count = affect_data.get("neg_count", 0)
    spontaneous_reframe = affect_data.get("spontaneous_reframe", False)

    # ── Agent 3: Reframing-Coach (only if negatives detected) ─────────────────
    reframing_response = None
    if neg_count > 0:
        ref_result = call_agent(api_key, "reframing", user_entry,
                                scaffold_level=scaffold_level,
                                skill_score=current_skill, stage=stage)
        reframing_response = ref_result["content"] or "[reframing error]"
        time.sleep(0.3)

    # ── Skill update (mirrors app.js updateSkill) ─────────────────────────────
    new_skill = update_skill(session_history, pos_count, neg_count)

    # ── Agent 4: Curriculum-Progress ──────────────────────────────────────────
    curr_result = call_agent(api_key, "curriculum", user_entry,
                             scaffold_level=scaffold_level,
                             skill_score=new_skill, stage=stage)
    curriculum_response = curr_result["content"] or "[curriculum error]"
    time.sleep(0.3)

    # ── Spontaneous rate (mirrors app.js) ─────────────────────────────────────
    if neg_count > 0:
        spontaneous_rate = min(pos_count / neg_count, 1.0)
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
        "reframing_scaffold_level": scaffold_level if reframing_response else None,
        "curriculum_response": curriculum_response,
        "skill_before": current_skill,
        "skill_after": new_skill,
        "scaffold_level": scaffold_level,
        "stage": stage,
        "mood": mood,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "spontaneous_reframe": spontaneous_reframe,
        "spontaneous_rate": spontaneous_rate,
        "timestamp": timestamp,
        "note": "SYNTHETIC USER SESSION: All inputs are LLM-generated. No real participants.",
    }

    return session_record


# ── Main experiment runner ────────────────────────────────────────────────────

def run_condition(
    api_key: str,
    entries_by_persona: dict,
    personas: list,
    condition: str,
) -> dict:
    """
    Run all 5 personas x 10 sessions for one condition.

    Returns:
        dict mapping persona_id -> list of session records
    """
    print(f"\n{'='*60}")
    print(f"Running Condition: {condition.upper()}")
    print(f"{'='*60}")

    results = {}
    condition_dir = DATA_PROCESSED / "results" / f"condition_{condition}"
    condition_dir.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        pid = persona["id"]
        initial_skill = persona["initial_skill"]
        print(f"\n  Persona {pid} ({persona['label']}), initial_skill={initial_skill}")

        entries = entries_by_persona.get(pid, [])
        if len(entries) != 10:
            raise ValueError(f"Expected 10 entries for {pid}, got {len(entries)}")

        current_skill = initial_skill
        session_history = []
        persona_sessions = []

        for entry_record in entries:
            session_num = entry_record["session"]
            user_entry = entry_record["user_entry"]
            print(f"\n    Session {session_num}/10:")

            session_data = run_session(
                api_key=api_key,
                persona_id=pid,
                session_num=session_num,
                user_entry=user_entry,
                session_history=session_history,
                current_skill=current_skill,
                condition=condition,
                initial_skill=initial_skill,
            )

            current_skill = session_data["skill_after"]
            session_history.append(session_data)
            persona_sessions.append(session_data)

            print(f"    → skill: {session_data['skill_before']:.3f} → {session_data['skill_after']:.3f} | "
                  f"SRR: {session_data['spontaneous_rate']:.2f} | "
                  f"mood: {session_data['mood']} | "
                  f"spont: {session_data['spontaneous_reframe']}")

        results[pid] = persona_sessions

        # Save per-persona results immediately
        out_path = condition_dir / f"{pid}_sessions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(persona_sessions, f, ensure_ascii=False, indent=2)
        print(f"\n    Saved: {out_path}")

    return results


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary(results_fading: dict, results_fixed: dict, personas: list) -> dict:
    """
    Compute summary statistics for ablation comparison.

    delta-SRR = mean(SRR sessions 8-10) - mean(SRR sessions 1-3)
    """
    summary = {
        "description": (
            "Ablation study: Condition A (fading) vs Condition B (fixed_high). "
            "Synthetic users, N=5 personas x 10 sessions. "
            "No real participants. LLM-judged metrics."
        ),
        "note": "SYNTHETIC USERS (N=5 personas x 10 sessions). Effect directions reported; no significance test (N too small).",
        "personas": {},
        "aggregate": {},
    }

    all_delta_fading = []
    all_delta_fixed = []

    for persona in personas:
        pid = persona["id"]
        sessions_f = results_fading[pid]
        sessions_x = results_fixed[pid]

        def srr_slice(sessions, start, end):
            """Mean SRR for sessions [start, end] (1-indexed)."""
            subset = [s for s in sessions if start <= s["session"] <= end]
            if not subset:
                return 0.0
            return sum(s["spontaneous_rate"] for s in subset) / len(subset)

        # Early (sessions 1-3) and late (sessions 8-10)
        srr_early_f = srr_slice(sessions_f, 1, 3)
        srr_late_f = srr_slice(sessions_f, 8, 10)
        delta_srr_f = srr_late_f - srr_early_f

        srr_early_x = srr_slice(sessions_x, 1, 3)
        srr_late_x = srr_slice(sessions_x, 8, 10)
        delta_srr_x = srr_late_x - srr_early_x

        final_skill_f = sessions_f[-1]["skill_after"]
        final_skill_x = sessions_x[-1]["skill_after"]
        final_mood_f = sessions_f[-1]["mood"]
        final_mood_x = sessions_x[-1]["mood"]

        all_delta_fading.append(delta_srr_f)
        all_delta_fixed.append(delta_srr_x)

        summary["personas"][pid] = {
            "label": persona["label"],
            "condition_fading": {
                "srr_early_mean": round(srr_early_f, 4),
                "srr_late_mean": round(srr_late_f, 4),
                "delta_srr": round(delta_srr_f, 4),
                "final_skill": round(final_skill_f, 4),
                "final_mood": final_mood_f,
                "srr_per_session": [round(s["spontaneous_rate"], 4) for s in sessions_f],
                "skill_per_session": [round(s["skill_after"], 4) for s in sessions_f],
                "mood_per_session": [s["mood"] for s in sessions_f],
                "scaffold_per_session": [s["scaffold_level"] for s in sessions_f],
            },
            "condition_fixed": {
                "srr_early_mean": round(srr_early_x, 4),
                "srr_late_mean": round(srr_late_x, 4),
                "delta_srr": round(delta_srr_x, 4),
                "final_skill": round(final_skill_x, 4),
                "final_mood": final_mood_x,
                "srr_per_session": [round(s["spontaneous_rate"], 4) for s in sessions_x],
                "skill_per_session": [round(s["skill_after"], 4) for s in sessions_x],
                "mood_per_session": [s["mood"] for s in sessions_x],
                "scaffold_per_session": [s["scaffold_level"] for s in sessions_x],
            },
            "delta_srr_advantage_fading": round(delta_srr_f - delta_srr_x, 4),
        }

    n = len(personas)
    mean_delta_f = sum(all_delta_fading) / n
    mean_delta_x = sum(all_delta_fixed) / n
    n_fading_higher = sum(1 for f, x in zip(all_delta_fading, all_delta_fixed) if f > x)

    summary["aggregate"] = {
        "n_personas": n,
        "mean_delta_srr_fading": round(mean_delta_f, 4),
        "mean_delta_srr_fixed": round(mean_delta_x, 4),
        "mean_delta_srr_advantage": round(mean_delta_f - mean_delta_x, 4),
        "n_personas_fading_higher_delta_srr": n_fading_higher,
        "direction_consistent": n_fading_higher >= (n // 2 + 1),
        "interpretation": (
            f"Fading condition shows higher delta-SRR in {n_fading_higher}/{n} personas. "
            "No significance test (N=5 is underpowered). "
            "Effect direction and consistency reported as PoC evidence."
        ),
    }

    return summary


# ── Environment record ────────────────────────────────────────────────────────

def save_environment(api_key: str) -> None:
    """Record experiment environment for reproducibility."""
    import subprocess
    try:
        py_ver = sys.version
    except Exception:
        py_ver = "unknown"

    env_record = {
        "model": MODEL,
        "experiment_date": datetime.now(timezone.utc).isoformat()[:10],
        "seeds": {
            "synthetic_user": 2026,
            "experiment": EXPERIMENT_SEED,
        },
        "scaffold_thresholds": SCAFFOLD_THRESHOLDS,
        "moving_avg_window": MOVING_AVG_WINDOW,
        "curriculum_thresholds": CURRICULUM_THRESHOLDS,
        "personas": ["P1", "P2", "P3", "P4", "P5"],
        "sessions_per_persona": 10,
        "conditions": ["fading", "fixed_high"],
        "python_version": py_ver,
        "note": "SYNTHETIC USER EXPERIMENT. No real participants.",
    }

    out_path = DATA_PROCESSED / "experiment_environment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, ensure_ascii=False, indent=2)
    print(f"Environment recorded: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(EXPERIMENT_SEED)

    print("=" * 60)
    print("GratiFlow Phase 1 Evaluation — Run Experiment")
    print("IMPORTANT: SYNTHETIC USERS ONLY. No real participants.")
    print(f"Model: {MODEL} | Experiment seed: {EXPERIMENT_SEED}")
    print("=" * 60)

    # Load API key
    print("\nLoading API key...")
    api_key = load_api_key()
    print("  API key loaded.")

    # Load synthetic user entries
    entries_path = DATA_PROCESSED / "synthetic_user_entries.json"
    if not entries_path.exists():
        print(f"\nERROR: {entries_path} not found.")
        print("Run generate_synthetic_users.py first.")
        sys.exit(1)

    with open(entries_path) as f:
        entries_data = json.load(f)

    # Index entries by persona
    entries_by_persona: dict = {}
    for entry in entries_data["entries"]:
        pid = entry["persona_id"]
        if pid not in entries_by_persona:
            entries_by_persona[pid] = []
        entries_by_persona[pid].append(entry)

    # Sort by session number
    for pid in entries_by_persona:
        entries_by_persona[pid].sort(key=lambda x: x["session"])

    # Load personas
    with open(PERSONAS_FILE) as f:
        persona_data = json.load(f)
    personas = persona_data["personas"]

    # Save environment record
    save_environment(api_key)

    # Run Condition A: fading
    results_fading = run_condition(api_key, entries_by_persona, personas, "fading")

    # Run Condition B: fixed_high
    results_fixed = run_condition(api_key, entries_by_persona, personas, "fixed_high")

    # Compute and save summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary(results_fading, results_fixed, personas)

    summary_path = DATA_PROCESSED / "results" / "summary_statistics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_path}")

    # Print key results
    agg = summary["aggregate"]
    print("\n" + "=" * 60)
    print("KEY RESULTS (Synthetic Users, N=5 personas x 10 sessions)")
    print("=" * 60)
    print(f"Mean delta-SRR (fading):     {agg['mean_delta_srr_fading']:+.4f}")
    print(f"Mean delta-SRR (fixed_high): {agg['mean_delta_srr_fixed']:+.4f}")
    print(f"Advantage (fading - fixed):  {agg['mean_delta_srr_advantage']:+.4f}")
    print(f"Personas with fading > fixed: {agg['n_personas_fading_higher_delta_srr']}/5")
    print(f"Direction consistent: {agg['direction_consistent']}")
    print(f"\nInterpretation: {agg['interpretation']}")
    print("\nNOTE: N=5 is underpowered for significance testing.")
    print("Effect direction and consistency reported as PoC evidence only.")
    print("=" * 60)


if __name__ == "__main__":
    main()
