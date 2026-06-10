"""
srr_human_validation_llm_judge_r2_clarified.py
================================================
Blind LLM judgment (gpt-5.4-mini) for SRR re-verification with CLARIFIED R2.

PURPOSE:
  Re-run LLM judgment on all 70 items using the clarified R2 definition
  (r2_clarification_spec.md, the research team 2026-06-06). Only R2 changes;
  all other rubric sections (R1, R3, F1-F5) are identical to the original run.

BLINDED:
  This script does NOT share label distribution with PI.
  Output CSV is for post-annotation comparison only.

MODEL: gpt-5.4-mini  <- CONFIRMED. Do NOT change to gpt-4o-mini.
  - Uses max_completion_tokens=1024 (no temperature parameter)
  - Rate-limit aware: retry with exponential backoff
  - API key: loaded from .env (NEVER printed to stdout)

OUTPUTS:
  data/processed/srr_human_validation/llm_labels_r2_clarified.csv
    Columns: id, llm_srr_label, llm_reasoning
  data/processed/srr_human_validation/llm_judge_log_r2_clarified.json
    Full log (internal use only)

Author: team member (experiment lead, the research team)
Date: 2026-06-06
"""

import csv
import hashlib
import json
import os
import re
import time
import random
from pathlib import Path

from openai import OpenAI

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
BLANK_CSV   = os.path.join(BASE, "data/processed/srr_human_validation/annotation_sheet_blank.csv")
OUT_CSV     = os.path.join(BASE, "data/processed/srr_human_validation/llm_labels_r2_clarified.csv")
LOG_JSON    = os.path.join(BASE, "data/processed/srr_human_validation/llm_judge_log_r2_clarified.json")
FREEZE_MD   = os.path.join(BASE, "data/processed/srr_human_validation/FREEZE_RECORD.md")

API_KEY_FILE = os.path.expanduser(".env")

# ─── MODEL CONFIGURATION ───────────────────────────────────────────────────────
# IMPORTANT: Must be gpt-5.4-mini. Do NOT use gpt-4o-mini.
MODEL = "gpt-5.4-mini"
MAX_COMPLETION_TOKENS = 1024
# temperature is intentionally omitted (use model default)

# ─── Load API key ──────────────────────────────────────────────────────────────
def load_api_key() -> str:
    """Load OPENAI_API_KEY from .env. Never prints the key."""
    with open(API_KEY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise ValueError("OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=your_key")

# ─── SRR Rubric System Prompt — R2 CLARIFIED VERSION ───────────────────────────
# Basis: r2_clarification_spec.md (the research team, 2026-06-06)
# Change: R2 now explicitly enumerates (a) positive reappraisal, (b) acceptance,
#         (c) psychological distancing, (d) gratitude, (e) self-affirmation.
#         Explicit exclusions X1-X4 added.
# Unchanged: R1, R3, F1-F5, few-shot structure.
SYSTEM_PROMPT = """You are the Affect-Analysis Agent in GratiFlow, operating in paired/context-aware mode.
Your task is to judge whether a candidate text constitutes a Spontaneous Reframing Response (SRR)
given the original negative context.

You will receive a JSON object with two fields:
  "context": the original negative statement or tweet (source of negative emotion)
  "candidate_text": the text to be judged

Respond with ONLY valid JSON (no markdown fences, no extra text).

Required JSON schema:
{
  "srr": <integer, 1 if SRR, 0 if not SRR>,
  "reasoning": "<string, 1-2 sentence justification citing which criteria (R1/R2/R3/F1-F5/X1-X4) apply>"
}

=== RUBRIC: Spontaneous Reframing Response (SRR) ===

Definition (from cognitive reappraisal theory; Gross, 1998):
SRR = 1 if and only if the candidate_text satisfies ALL THREE of:
  (R1) An explicitly stated or clearly implied negative event or emotion is present
       (typically derived from the context).
  (R2) A deliberate reinterpretation that transforms the meaning of that SAME negative event
       through one or more of the following cognitive reappraisal forms:

       (a) POSITIVE REAPPRAISAL: Explicitly reinterpreting the negative event as having
           positive value (growth, learning, opportunity).
           Example: "I studied all night under stress -- that means I worked hard and that's good."

       (b) ACCEPTANCE-BASED REINTERPRETATION: Accepting what happened and framing it as
           manageable, normal, or part of life, thereby reducing its negative meaning.
           Example: "I have been dealing with difficult people all day, but it is over with now."
           Example: "These things happen -- I can handle it."
           NOTE: Must be event-linked acceptance. Generic "that's life" unlinked to
           the specific event is F2 / X4, not R2.

       (c) PSYCHOLOGICAL DISTANCING: Placing the event in broader perspective to diminish
           its emotional impact.
           Example: "It's not the end of the world."
           Example: "That's a difficult situation but I can see the funny side."
           NOTE: Humor and irony qualify as psychological distancing if they demonstrably
           change the meaning of the negative event.

       (d) GRATITUDE REINTERPRETATION: Expressing gratitude for something connected to
           the SAME negative situation (not a separate event).
           Example: "The days at work after Christmas can be hard, but I am thankful for my job."
           NOTE: Gratitude about a SEPARATE event or person does NOT qualify (-> X2).

       (e) SELF-AFFIRMATION: Invoking one's own capabilities, patience, or resilience
           in the face of the SAME negative event.
           Example: "I have many pages to read but I will read them all because I am a fast reader."
           Example: "I'm thankful for the way I am, I am patient."

  (R3) Evidence that the reframe is generated by the speaker themselves, not echoed from
       an external prompt or prior AI instruction.

SRR = 0 if R2 is not met because ANY of the following apply:
  (X1) PURE FUTURE PLAN ONLY: The candidate only describes a future action plan without
       any reinterpretation of the current/past negative event itself.
       Example: "I'll study harder next time" with no meaning revision of the failure.
       Example: "I hope I am able to catch up tomorrow" -- future plan without present reinterpretation.
  (X2) SEPARATE-EVENT POSITIVE ONLY: The positive aspect refers to a different event or person,
       not a reinterpretation of the same negative event.
       Example: "My day was bad but my friend had a great time."
  (X3) SIMPLE RE-DESCRIPTION: The candidate merely restates the negative event without
       any change in meaning or perspective.
  (X4) VAGUE PLATITUDE WITHOUT EVENT LINKAGE: Generic optimism ("it'll be fine") that is
       not tied to any specific cognitive shift about the negative event.

SRR = 0 if any of the following apply (in addition to X1-X4 above):
  (F1) The candidate describes a positive event that is SEPARATE from the negative context
       -- no reinterpretation of the negative event itself.
  (F2) The candidate expresses vague optimism or a coping platitude without specific
       cognitive reinterpretation of the negative event.
  (F3) The reframe directly repeats or paraphrases language modeled by an AI in a prior message.
  (F4) The positive content exists only because of a direct prompt instruction;
       no self-generated cognitive shift is evident.
  (F5) The candidate only describes negative events/emotions with no positive reinterpretation.

BOUNDARY GUIDANCE:
  - Tentative language ("maybe," "I suppose," "perhaps") does NOT disqualify R2.
    Natural reappraisal is often tentative.
  - Indirect or implicit reappraisal (humor, irony, understatement) qualifies as R2
    if it demonstrably changes the meaning of the negative event.
  - The reappraisal need not be linguistically elaborate; even a brief statement qualifies
    if it meets one of (a)-(e) and addresses the same negative event.
  - Brief acceptance ("it is over with now", "these things happen") QUALIFIES as R2(b)
    when it directly addresses the same negative event and reduces its ongoing threat.
    It does NOT qualify if it is a generic platitude unlinked to the event (-> X4).

=== FEW-SHOT EXAMPLES (CLARIFIED RUBRIC) ===

--- Example 1: SRR = 1 — R2(a) positive reappraisal ---
Input: {
  "context": "Studied and made notecards all night, uhhhh hate finals and keystones at the same time. Not a Happy Camper",
  "candidate_text": "I studied and took notes all night, I'm working hard and that's good."
}
Output: {"srr": 1, "reasoning": "R1: studying all night under exam stress (negative); R2(a): reframes the same effort as inherently positive ('working hard and that's good'); R3: self-generated. All three criteria met."}

--- Example 2: SRR = 1 — R2(e) self-affirmation ---
Input: {
  "context": "can guarantee I will cry at least once before Friday I'm so far behind in my nursing readings",
  "candidate_text": "I fell behind my nursing readings, but I know it's my own fault, now I have to power through all of it, I can do it!"
}
Output: {"srr": 1, "reasoning": "R1: falling behind on readings (negative); R2(e): self-affirmation ('I can do it!') applied to the same challenge; R3: self-generated. All three criteria met."}

--- Example 3: SRR = 1 — R2(b) acceptance-based reinterpretation ---
Input: {
  "context": "This has been one fucked up day im tired of fighting with people I need to relax and unwind for a day with no fighting out",
  "candidate_text": "I have been dealing with difficult people all day, but it is over with now."
}
Output: {"srr": 1, "reasoning": "R1: day-long conflict and exhaustion (negative); R2(b): 'it is over with now' is event-linked acceptance reducing the ongoing threat of the negative event -- not a vague platitude; R3: self-generated. All three criteria met."}

--- Example 4: SRR = 1 — R2(d) gratitude reinterpretation ---
Input: {
  "context": "Only been back at work 5 days since the Christmas holidays, and I'm already and tired with the madness of my job.",
  "candidate_text": "The days at work, after Christmas, can be hard, but I am thankful for my job."
}
Output: {"srr": 1, "reasoning": "R1: post-holiday work exhaustion (negative); R2(d): gratitude about the same situation ('my job' is the source of stress), reframing hard days against the benefit of employment; R3: self-generated. All three criteria met."}

--- Example 5: SRR = 1 — R2(e) self-affirmation ---
Input: {
  "context": "Been up since 4 and still have 15 pages to read. If I make it through this week I think can make it through anything.",
  "candidate_text": "I have many pages to read but I will read them all because I am a fast reader."
}
Output: {"srr": 1, "reasoning": "R1: overwhelming reading load (negative); R2(e): self-affirmation ('I am a fast reader') reinterprets the same burden as manageable; R3: self-generated. All three criteria met."}

--- Example 6: SRR = 1 — R2(c) psychological distancing via humor ---
Input: {
  "context": "OMG!! Not okay!! Haha Yes you are right, I would be freaking out right now!! Lol",
  "candidate_text": "Thats a difficult situation but I can see the funny side"
}
Output: {"srr": 1, "reasoning": "R1: implied stressful/panic-inducing situation (negative); R2(c): humor achieves psychological distancing from the negative event, changing its meaning; R3: self-generated. All three criteria met."}

--- Example 7: SRR = 0 — X1 (pure future plan) ---
Input: {
  "context": "Cringing over the fact that transcripts come out tomorrow and I won't see it till Friday",
  "candidate_text": "Waiting for Friday when I will have all the transcripts for ready reference."
}
Output: {"srr": 0, "reasoning": "X1: candidate describes only a future state (having transcripts on Friday) without reinterpreting the current negative feeling (anxiety, cringing). No meaning revision of the negative event. R2 not met."}

--- Example 8: SRR = 0 — X2 (separate-event positive) ---
Input: {
  "context": "My brother is having a destination wedding next week and I'm so excited and exhausted!! ...so exhausted and stressed and broke from the planning and prep...",
  "candidate_text": "My brother is having a destination wedding next week I am happy for him."
}
Output: {"srr": 0, "reasoning": "X2: 'I am happy for him' is about the brother's event, not a reinterpretation of the speaker's own negative experience (exhaustion, stress, broke). R2 not met."}

--- Example 9: SRR = 0 — F2 / X4 (vague platitude) ---
Input: {
  "context": "I want to lay down and die.., Just wake up at a time when society isn't such a load of bullshit.",
  "candidate_text": "I wanted to lay down, and wake up at a time when society isn't so bad. But that's life, and I should be enjoying the good parts of it."
}
Output: {"srr": 0, "reasoning": "F2/X4: 'that's life and I should be enjoying the good parts' is a coping platitude without specific cognitive reinterpretation of the negative event. No positive meaning is actively constructed from the stated negative. R2 not met."}

--- Example 10: SRR = 0 — F5 (negative only) ---
Input: {
  "context": "I need a day just to finish all my projects and graduation stuff.",
  "candidate_text": "I need a day just to finish all my projects and graduation stuff. justoneday stressed"
}
Output: {"srr": 0, "reasoning": "F5: pure expression of negative situation with no reinterpretation. R2 not met."}

--- Example 11: SRR = 0 — F1/F2 (policy preference) ---
Input: {
  "context": "Project project project~~ Y cnt teacher assign us earlier. Made us really rush hard!",
  "candidate_text": "Teachers ought to give more time to students for project work. This would result in a smooth and timely completion of projects."
}
Output: {"srr": 0, "reasoning": "F1/F2: policy preference rather than reinterpretation of the past negative event (rush stress). R2 not met."}

=== END RUBRIC ===

Rules:
- Always return valid JSON with exactly two fields: 'srr' (integer 0 or 1) and 'reasoning' (string).
- When context is empty or missing, judge based on candidate_text alone.
- When in doubt, assign srr = 0. All three criteria (R1, R2, R3) must be clearly evidenced.
- Do NOT infer intent beyond what is explicitly written.
- In reasoning, cite which R2 subtype (a)-(e) OR which exclusion (X1-X4/F1-F5) applies.
"""


def make_user_message(context: str, candidate_text: str) -> str:
    payload = {"context": context, "candidate_text": candidate_text}
    return json.dumps(payload, ensure_ascii=False)


def call_llm_with_retry(client: OpenAI, context: str, candidate_text: str,
                         item_id: str, max_retries: int = 5) -> dict:
    """
    Call gpt-5.4-mini with the clarified R2 SRR rubric.
    Retries on rate-limit / transient errors with exponential backoff.
    Returns dict with keys: srr (int), reasoning (str), raw_response, usage.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_user_message(context, candidate_text)},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                # temperature intentionally omitted (use model default)
            )
            raw_text = response.choices[0].message.content.strip()

            # Parse JSON response
            parsed = json.loads(raw_text)
            srr_val = int(parsed.get("srr", -1))
            if srr_val not in (0, 1):
                raise ValueError(f"Invalid srr value: {srr_val}")
            reasoning = str(parsed.get("reasoning", "")).strip()
            if not reasoning:
                raise ValueError("Empty reasoning field")

            return {
                "srr": srr_val,
                "reasoning": reasoning,
                "raw_response": raw_text,
                "usage": {
                    "prompt_tokens":      response.usage.prompt_tokens,
                    "completion_tokens":  response.usage.completion_tokens,
                },
            }

        except Exception as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"  [WARN] {item_id} attempt {attempt}/{max_retries} failed: {e}. "
                  f"Retrying in {wait:.1f}s...")
            if attempt == max_retries:
                print(f"  [ERROR] {item_id} exhausted retries. Assigning srr=0 with error note.")
                return {
                    "srr": 0,
                    "reasoning": f"[ERROR: LLM call failed after {max_retries} retries: {e}]",
                    "raw_response": "",
                    "usage": {},
                }
            time.sleep(wait)


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_freeze_record(csv_path: str, log_path: str,
                          csv_sha: str, log_sha: str,
                          total_items: int, errors: int) -> None:
    """Append R2-clarified freeze entry to FREEZE_RECORD.md."""
    entry = f"""
---

## R2-Clarified Re-Verification Freeze (2026-06-06)

### Purpose
Re-run with clarified R2 definition (r2_clarification_spec.md, the research team 2026-06-06).
Only R2 changes; R1, R3, F1-F5 unchanged.

### Model
- **Model**: {MODEL}
- **max_completion_tokens**: {MAX_COMPLETION_TOKENS}
- **temperature**: not set (model default)

### Frozen Files

#### llm_labels_r2_clarified.csv
- **SHA-256**: `{csv_sha}`
- Contents: {total_items} items, columns: id / llm_srr_label / llm_reasoning
- Errors (exhausted retries): {errors}
- Blinded: YES (label distribution not reported here)

#### llm_judge_log_r2_clarified.json
- **SHA-256**: `{log_sha}`
- Contents: full log including raw_response and token usage per item
- Internal use only

### Integrity Status
- Total items judged: {total_items} / 70
- Empty reasoning fields: 0 (enforced by validation)
- Blinded: LLM labels NOT shared with PI before PI re-annotation is complete

### Verification Commands
```bash
shasum -a 256 \\
  data/processed/srr_human_validation/llm_labels_r2_clarified.csv \\
  data/processed/srr_human_validation/llm_judge_log_r2_clarified.json
```

Expected:
```
{csv_sha}  data/processed/srr_human_validation/llm_labels_r2_clarified.csv
{log_sha}  data/processed/srr_human_validation/llm_judge_log_r2_clarified.json
```

Author: team member (experiment lead, the research team) — 2026-06-06
"""
    with open(FREEZE_MD, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[updated] {FREEZE_MD}")


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SRR LLM Judgment — R2 CLARIFIED — gpt-5.4-mini")
    print("Rubric basis: r2_clarification_spec.md (the research team, 2026-06-06)")
    print("Data: Ziems et al. (2022) CC BY-SA 4.0")
    print(f"MODEL = {MODEL}  <-- confirm this is gpt-5.4-mini")
    print("=" * 60)

    # Confirm model string at runtime
    assert MODEL == "gpt-5.4-mini", (
        f"Model mismatch! Expected gpt-5.4-mini, got {MODEL}. "
        "Do NOT use gpt-4o-mini."
    )

    # Load annotation sheet (original order, all 70 items)
    with open(BLANK_CSV, newline="", encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    print(f"Loaded {len(items)} items from annotation_sheet_blank.csv")
    assert len(items) == 70, f"Expected 70 items, got {len(items)}"

    # Initialize OpenAI client
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    results = []
    log_entries = []
    total_tokens = {"prompt": 0, "completion": 0}
    error_count = 0

    for i, item in enumerate(items, 1):
        item_id = item["id"]
        context = item["context"]
        candidate_text = item["candidate_text"]

        print(f"[{i:2d}/70] Judging {item_id} ...", end=" ", flush=True)

        result = call_llm_with_retry(client, context, candidate_text, item_id)

        if result["reasoning"].startswith("[ERROR:"):
            error_count += 1

        print(f"srr={result['srr']}")

        results.append({
            "id": item_id,
            "llm_srr_label": result["srr"],
            "llm_reasoning": result["reasoning"],
        })

        log_entries.append({
            "id": item_id,
            "context": context,
            "candidate_text": candidate_text,
            "llm_srr_label": result["srr"],
            "llm_reasoning": result["reasoning"],
            "raw_response": result["raw_response"],
            "usage": result["usage"],
        })

        if result["usage"]:
            total_tokens["prompt"]     += result["usage"].get("prompt_tokens", 0)
            total_tokens["completion"] += result["usage"].get("completion_tokens", 0)

        # Rate-limit buffer: 1 req/s safe for gpt-5.4-mini tier-1
        if i < len(items):
            time.sleep(1.0)

    # ── Validate: no empty reasoning ──────────────────────────────────────────
    empty_reasoning = [r["id"] for r in results if not r["llm_reasoning"]]
    if empty_reasoning:
        print(f"[ERROR] Empty reasoning for: {empty_reasoning}")
    else:
        print(f"\nValidation: 0 empty reasoning fields. OK.")

    # ── Save blind labels CSV (utf-8, NO BOM — blinded output) ───────────────
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "llm_srr_label", "llm_reasoning"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[saved] {OUT_CSV}")

    # ── Save full log JSON ─────────────────────────────────────────────────────
    log = {
        "description": (
            "LLM judgment log for SRR re-verification with CLARIFIED R2. "
            "Model: gpt-5.4-mini. "
            "R2 clarification basis: r2_clarification_spec.md (the research team, 2026-06-06). "
            "DO NOT share llm_srr_label distribution with PI before re-annotation is complete."
        ),
        "data_source": {
            "dataset": "SALT-NLP/positive_reframing",
            "paper": "Ziems et al. (2022), ACL 2022",
            "license": "CC BY-SA 4.0",
            "url": "https://aclanthology.org/2022.acl-long.257/",
        },
        "model": MODEL,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "rubric_version": "r2_clarified_v2 (r2_clarification_spec.md, the research team 2026-06-06)",
        "total_items": len(results),
        "error_count": error_count,
        "total_tokens": total_tokens,
        "entries": log_entries,
    }
    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[saved] {LOG_JSON}  (full log — internal only)")

    # ── SHA-256 freeze ─────────────────────────────────────────────────────────
    csv_sha = sha256_file(OUT_CSV)
    log_sha = sha256_file(LOG_JSON)
    print(f"\nSHA-256 ({OUT_CSV.split('/')[-1]}): {csv_sha}")
    print(f"SHA-256 ({LOG_JSON.split('/')[-1]}): {log_sha}")

    append_freeze_record(
        csv_path=OUT_CSV,
        log_path=LOG_JSON,
        csv_sha=csv_sha,
        log_sha=log_sha,
        total_items=len(results),
        errors=error_count,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nTotal tokens: prompt={total_tokens['prompt']}, "
          f"completion={total_tokens['completion']}")
    print(f"Errors (exhausted retries): {error_count}")
    print(f"Empty reasoning: {len(empty_reasoning)}")
    print(f"\nModel confirmed: {MODEL}")
    print("\nBlinded judgment complete.")
    print("Do NOT reveal label distribution to PI before re-annotation is complete.")
    print(f"\nPI re-annotation tool: annotate_r2_clarified.html")
    print(f"After PI completes: run srr_human_validation_compare_r2_clarified.py")


if __name__ == "__main__":
    main()
