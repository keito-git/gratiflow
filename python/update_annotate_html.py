"""
update_annotate_html.py
-----------------------
Inject / re-inject Japanese translations into annotate.html.

Strategy (idempotent):
  - Always rebuilds the ITEMS array from scratch using translations_ja.json.
  - HTML structure / CSS / render() / CSV format are patched only if not
    already present (idempotent guards on each block).
  - Safe to run multiple times; the JA values in ITEMS always reflect the
    latest translations_ja.json.

Output:
  - annotate.html is overwritten in-place.
  - A machine-verifiable report is printed (id-level match count, blinding).

Features preserved:
  - 1-question-at-a-time display
  - SRR 1/0 buttons
  - Progress indicator + localStorage auto-save
  - Coding guide (collapsible)
  - English original + JA translation side-by-side
  - CSV download: id,context,candidate_text,PI_SRR_label,PI_note
  - Blinding: llm_srr_label is NOT embedded anywhere

Author: team member (experiment lead, Team Kiyomiya)
Date: 2026-06-06
"""

import json
import re
from pathlib import Path

BASE_DIR  = Path(Path(__file__).resolve().parent.parent)
HTML_PATH = BASE_DIR / "data/processed/srr_human_validation/annotate.html"
JSON_PATH = BASE_DIR / "data/processed/srr_human_validation/translations_ja.json"

# ── Load translations ──────────────────────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    translations = json.load(f)

ja_map: dict[str, dict] = {d["id"]: d for d in translations}

# ── Load HTML ──────────────────────────────────────────────────────────────
html = HTML_PATH.read_text(encoding="utf-8")


# ── Helper ────────────────────────────────────────────────────────────────
def escape_js_string(s: str) -> str:
    """Escape a Python string for embedding inside a JS double-quoted string."""
    return (
        s.replace("\\", "\\\\")
         .replace('"', '\\"')
         .replace("\n", "\\n")
         .replace("\r", "\\r")
    )


# ── 1. Rebuild ITEMS array (idempotent: always reconstructed) ──────────────
# The ITEMS block may contain items with or without _ja fields.
# We extract the raw JSON objects, update/add _ja fields, then rewrite.
items_match = re.search(r"const ITEMS = \[(.*?)\];", html, re.DOTALL)
if not items_match:
    raise RuntimeError("ITEMS array not found in HTML")

items_block = items_match.group(1)

# Each item is a JSON object on a single logical line.
# Grab them with a broad regex, then parse.
raw_items = re.findall(r"\{[^{}]+\}", items_block, re.DOTALL)
if len(raw_items) != len(translations):
    raise RuntimeError(
        f"Item count mismatch: HTML has {len(raw_items)}, JSON has {len(translations)}"
    )

new_item_lines: list[str] = []
for raw in raw_items:
    obj = json.loads(raw)
    item_id = obj["id"]
    ja = ja_map.get(item_id, {"context_ja": "", "candidate_ja": ""})

    # Rebuild item with guaranteed-fresh JA values
    new_obj = {
        "id":           obj["id"],
        "context":      obj["context"],
        "candidate_text": obj["candidate_text"],
        "context_ja":   ja["context_ja"],
        "candidate_ja": ja["candidate_ja"],
    }

    # Serialise to compact JSON, indented with 2 spaces for readability
    line = "  " + json.dumps(new_obj, ensure_ascii=False, separators=(",", ":"))
    new_item_lines.append(line)

new_items_block = "const ITEMS = [\n" + ",\n".join(new_item_lines) + "\n];"

html = re.sub(
    r"const ITEMS = \[.*?\];",
    new_items_block,
    html,
    count=1,
    flags=re.DOTALL,
)

# ── 2. Patch CSS (idempotent) ──────────────────────────────────────────────
translation_css = """
    /* ── Japanese translation display ── */
    .ja-translation {
      font-size: .82rem;
      color: #5a6a8a;
      background: #f0f3ff;
      border-top: 1px dashed #c5ceee;
      padding: 6px 12px 7px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .ctx-box .ja-translation {
      background: #f4f5fa;
      border-top-color: #c8cdd8;
      color: #6a7080;
    }
    .ja-label {
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .06em;
      color: #8899bb;
      margin-right: 4px;
    }
    .translation-notice {
      font-size: .78rem;
      color: #888;
      background: #fff8e1;
      border-left: 3px solid #f5a623;
      border-radius: 0 6px 6px 0;
      padding: 6px 12px;
      margin-bottom: 14px;
      line-height: 1.5;
    }"""

if ".ja-translation" not in html:
    html = html.replace("  </style>", translation_css + "\n  </style>", 1)

# ── 3. Patch translation notice banner (idempotent) ────────────────────────
notice_html = """
  <!-- Translation notice -->
  <div class="translation-notice">
    各項目の英語原文の下に日本語訳（【訳】）を参考表示しています。
    <strong>判定は必ず英語原文の内容に基づいて行ってください。</strong>日本語訳はあくまで理解補助です。
  </div>
"""
if "Translation notice" not in html:
    html = html.replace(
        "  <!-- Coding Guide (collapsible) -->",
        notice_html + "  <!-- Coding Guide (collapsible) -->",
        1,
    )

# ── 4. Patch render() to show JA translations (idempotent) ────────────────
old_ctx_render = """  // Context
  const ctxBox = document.getElementById("ctxBox");
  if (item.context && item.context.trim()) {
    ctxBox.textContent = item.context;
    ctxBox.classList.remove("empty-ctx");
  } else {
    ctxBox.textContent = "（context なし — candidate_text のみを判定してください）";
    ctxBox.classList.add("empty-ctx");
  }

  // Candidate
  document.getElementById("candBox").textContent = item.candidate_text;"""

new_ctx_render = """  // Context
  const ctxBox = document.getElementById("ctxBox");
  if (item.context && item.context.trim()) {
    ctxBox.classList.remove("empty-ctx");
    // Build: original text + ja translation
    let ctxHtml = escHtml(item.context);
    if (item.context_ja) {
      ctxHtml += '<div class="ja-translation"><span class="ja-label">【訳】</span>' + escHtml(item.context_ja) + '</div>';
    }
    ctxBox.innerHTML = ctxHtml;
  } else {
    ctxBox.innerHTML = "（context なし — candidate_text のみを判定してください）";
    ctxBox.classList.add("empty-ctx");
  }

  // Candidate
  const candBox = document.getElementById("candBox");
  let candHtml = escHtml(item.candidate_text);
  if (item.candidate_ja) {
    candHtml += '<div class="ja-translation"><span class="ja-label">【訳】</span>' + escHtml(item.candidate_ja) + '</div>';
  }
  candBox.innerHTML = candHtml;"""

if old_ctx_render in html:
    html = html.replace(old_ctx_render, new_ctx_render, 1)
# else: already patched — skip (idempotent)

# ── 5. Add escHtml helper (idempotent) ────────────────────────────────────
esc_html_fn = """// ── HTML escape helper ──────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

"""
if "function escHtml" not in html:
    html = html.replace(
        "// ──────────────────────────────────────────────\n//  RENDER",
        esc_html_fn + "// ──────────────────────────────────────────────\n//  RENDER",
        1,
    )

# ── Write output ───────────────────────────────────────────────────────────
HTML_PATH.write_text(html, encoding="utf-8")
print(f"HTML written to: {HTML_PATH}")

# ── Verification ──────────────────────────────────────────────────────────
print("\n── Verification ──")

# Re-parse HTML to confirm
items_match2 = re.search(r"const ITEMS = \[(.*?)\];", html, re.DOTALL)
items_block2 = items_match2.group(1)
raw_items2 = re.findall(r"\{[^{}]+\}", items_block2, re.DOTALL)

exact_matches = 0
mismatches: list[str] = []
for raw in raw_items2:
    obj = json.loads(raw)
    item_id = obj["id"]
    expected = ja_map.get(item_id)
    if not expected:
        mismatches.append(f"  {item_id}: NOT FOUND in translations_ja.json")
        continue
    ctx_ok   = obj.get("context_ja",   "") == expected["context_ja"]
    cand_ok  = obj.get("candidate_ja", "") == expected["candidate_ja"]
    if ctx_ok and cand_ok:
        exact_matches += 1
    else:
        if not ctx_ok:
            mismatches.append(f"  {item_id} context_ja MISMATCH")
        if not cand_ok:
            mismatches.append(f"  {item_id} candidate_ja MISMATCH")

print(f"ID-level match: {exact_matches} / {len(translations)} (expected 70)")
if mismatches:
    for m in mismatches:
        print(m)
else:
    print("All 70 items: context_ja + candidate_ja match translations_ja.json (OK)")

# Blinding
llm_count = html.count("llm_srr_label") + html.count("llm_srr") + html.count("llm_labels")
print(f"Blinding: llm_srr_label={html.count('llm_srr_label')}, "
      f"llm_srr={html.count('llm_srr')}, "
      f"llm_labels={html.count('llm_labels')} "
      f"-> {'OK' if llm_count == 0 else 'WARNING: LEAK'}")

# CSV header
csv_match = re.search(r'const header = "([^"]+)"', html)
csv_header = csv_match.group(1) if csv_match else "NOT FOUND"
print(f"CSV header: {csv_header}")
expected_csv = "id,context,candidate_text,PI_SRR_label,PI_note"
print(f"CSV header OK: {csv_header == expected_csv}")

# Notice
print(f"Translation notice: {'present' if '判定は必ず英語原文の内容に基づいて行ってください' in html else 'MISSING'}")
print(f"escHtml function: {'present' if 'function escHtml' in html else 'MISSING'}")

print("\nDone.")
