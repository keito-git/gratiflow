"""
translate_srr_items.py
----------------------
Translate context and candidate_text of 70 SRR annotation items to Japanese
using gpt-5.4-mini. Results are saved to translations_ja.json.

Author: team member (experiment lead, Team Kiyomiya)
Date: 2026-06-06
"""

import csv
import json
import os
import time
import sys
from pathlib import Path

from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(Path(__file__).resolve().parent.parent)
CSV_PATH = BASE_DIR / "data/processed/srr_human_validation/annotation_sheet_blank.csv"
OUT_PATH = BASE_DIR / "data/processed/srr_human_validation/translations_ja.json"

# ── API setup ───────────────────────────────────────────────────────────────
# Load key from env file (key value must not be printed)
api_keys_path = Path(".env")
with open(api_keys_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("OPENAI_API_KEY="):
            api_key = line.split("=", 1)[1]
            break

client = OpenAI(api_key=api_key)
MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You are a faithful translator from English to Japanese.
Your task is to translate social media posts and reframing texts about everyday situations.

Translation rules:
1. FAITHFUL translation: preserve the exact meaning and nuance, especially cues about
   whether negative events are being reinterpreted positively. Do NOT smooth out negative
   content or add positive spin that is not in the original.
2. Preserve the colloquial / SNS-style tone where present (casual Japanese is fine).
3. Keep proper nouns (names, brands, hashtags like #NHS, team names, etc.) as-is.
4. If the text is already somewhat negative, keep that negativity intact.
5. Output ONLY the Japanese translation. No explanation, no quotation marks around output.
"""

def translate_text(text: str, field_type: str) -> str:
    """Translate a single text field to Japanese using gpt-5.4-mini."""
    user_prompt = f"Translate the following {field_type} text to Japanese:\n\n{text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_completion_tokens=512,
    )
    return response.choices[0].message.content.strip()


def load_items_from_csv(csv_path: Path) -> list[dict]:
    """Load annotation items from CSV."""
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "id": row["id"],
                "context": row["context"].strip(),
                "candidate_text": row["candidate_text"].strip(),
            })
    return items


def main():
    print(f"Loading CSV: {CSV_PATH}")
    items = load_items_from_csv(CSV_PATH)
    print(f"Loaded {len(items)} items")

    # Load existing translations if any (for resumption)
    existing: dict[str, dict] = {}
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            existing[entry["id"]] = entry
        print(f"Resuming: {len(existing)} already translated")

    results = []
    errors = []

    for i, item in enumerate(items):
        item_id = item["id"]
        ctx     = item["context"]
        cand    = item["candidate_text"]

        # Skip if already translated
        if item_id in existing:
            results.append(existing[item_id])
            print(f"  [{i+1:02d}/70] {item_id}: (cached)")
            continue

        print(f"  [{i+1:02d}/70] {item_id} ...", end=" ", flush=True)

        entry = {"id": item_id, "context_ja": "", "candidate_ja": ""}

        # Translate context (skip if empty)
        if ctx:
            for attempt in range(3):
                try:
                    entry["context_ja"] = translate_text(ctx, "context (negative situation)")
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"\n  ERROR context {item_id}: {e}")
                        errors.append({"id": item_id, "field": "context", "error": str(e)})
                    else:
                        time.sleep(2 ** attempt)
        else:
            entry["context_ja"] = ""  # empty context — no translation needed

        # Translate candidate_text (always)
        for attempt in range(3):
            try:
                entry["candidate_ja"] = translate_text(cand, "candidate reframing text")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"\n  ERROR candidate {item_id}: {e}")
                    errors.append({"id": item_id, "field": "candidate_text", "error": str(e)})
                else:
                    time.sleep(2 ** attempt)

        results.append(entry)
        print("ok")

        # Save checkpoint every 10 items
        if (i + 1) % 10 == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Checkpoint saved ({i+1} items)")

        # Rate control: ~0.5 s between calls to avoid 429
        time.sleep(0.5)

    # Final save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} items saved to: {OUT_PATH}")
    if errors:
        print(f"WARNING: {len(errors)} errors encountered:")
        for err in errors:
            print(f"  {err}")

    # ── Verification ──────────────────────────────────────────────────────
    print("\n── Verification ──")
    n_ctx_empty = sum(1 for item in items if not item["context"])
    n_with_ctx  = len(items) - n_ctx_empty
    n_ctx_translated  = sum(1 for r in results if r.get("context_ja"))
    n_cand_translated = sum(1 for r in results if r.get("candidate_ja"))

    print(f"Total items        : {len(items)}")
    print(f"Items with context : {n_with_ctx}  (expected: 52)")
    print(f"Items w/o context  : {n_ctx_empty}  (expected: 18, no translation needed)")
    print(f"context_ja filled  : {n_ctx_translated}  (should == {n_with_ctx})")
    print(f"candidate_ja filled: {n_cand_translated}  (should == 70)")

    if n_cand_translated == 70 and n_ctx_translated == n_with_ctx:
        print("PASS: all translations present")
    else:
        print("FAIL: some translations missing — check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
