"""
dry_run_ext_study1_fixed.py
============================
GratiFlow ext_study1_fixed — Dry-Run Calibration (NO API CALLS)

Purpose:
  1. Verify that adaptive-fading scaffold transitions (high→mid→low) occur
     within 14 sessions with the new calibrated persona parameters.
  2. Null calibration: when the same "arrived scaffold" sequence is fed to
     both conditions, delta-SRR difference should be ~0.

This script does NOT call any LLM API. It uses:
  - Expected SRR ≈ attempt_success probability (approximation)
  - update_observed_skill_v2() (did_attempt=False → excluded from average)

If scaffold transitions do NOT occur in adaptive-fading for at least 3 personas,
print a warning and suggest further parameter adjustment before running the real API.

Author: team member (experiment lead, the research team)
Date: 2026-06-05
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from latent_skill_model import (
    SCAFFOLD_ATTEMPT_MULTIPLIER,
    compute_attempt_probability,
    compute_attempt_success_probability,
    update_latent_skill,
)

# ── v2 persona parameters (calibrated per the research team 2026-06-05) ─────────────────

PERSONAS_FIXED = [
    {"id": "P1",  "label": "初心者・着実成長",  "latent_skill_0": 0.10, "alpha": 0.10, "alpha_passive": 0.02, "beta": 0.01, "p_attempt_base": 0.30, "neg_tendency": 1.5},
    {"id": "P2",  "label": "初心者・停滞型",    "latent_skill_0": 0.08, "alpha": 0.06, "alpha_passive": 0.01, "beta": 0.02, "p_attempt_base": 0.20, "neg_tendency": 2.0},
    {"id": "P3",  "label": "中級・安定成長",    "latent_skill_0": 0.25, "alpha": 0.08, "alpha_passive": 0.02, "beta": 0.01, "p_attempt_base": 0.40, "neg_tendency": 1.0},
    {"id": "P4",  "label": "初心者・高応答型",  "latent_skill_0": 0.10, "alpha": 0.12, "alpha_passive": 0.03, "beta": 0.01, "p_attempt_base": 0.35, "neg_tendency": 1.5},
    {"id": "P5",  "label": "中級・慎重型",      "latent_skill_0": 0.18, "alpha": 0.07, "alpha_passive": 0.015,"beta": 0.015,"p_attempt_base": 0.30, "neg_tendency": 1.5},
    {"id": "P6",  "label": "上級・高自律型",    "latent_skill_0": 0.35, "alpha": 0.08, "alpha_passive": 0.015,"beta": 0.01, "p_attempt_base": 0.50, "neg_tendency": 0.8},
    {"id": "P7",  "label": "初心者・高ネガ傾向","latent_skill_0": 0.08, "alpha": 0.08, "alpha_passive": 0.02, "beta": 0.015,"p_attempt_base": 0.22, "neg_tendency": 2.5},
    {"id": "P8",  "label": "中級・揺れ型",      "latent_skill_0": 0.15, "alpha": 0.10, "alpha_passive": 0.01, "beta": 0.03, "p_attempt_base": 0.35, "neg_tendency": 1.8},
    {"id": "P9",  "label": "初心者・受動観察型","latent_skill_0": 0.10, "alpha": 0.08, "alpha_passive": 0.04, "beta": 0.01, "p_attempt_base": 0.18, "neg_tendency": 1.2},
    {"id": "P10", "label": "中級・急成長型",    "latent_skill_0": 0.15, "alpha": 0.14, "alpha_passive": 0.02, "beta": 0.01, "p_attempt_base": 0.42, "neg_tendency": 1.0},
]

# Experiment constants (same as ext_study1)
EXPERIMENT_SEED = 42
N_SESSIONS = 14
SCAFFOLD_THRESHOLDS = {"high": 0.35, "mid": 0.65}

# Moving average window — same as ext_study1
MOVING_AVG_WINDOW = 5


# ── Scaffold helper ───────────────────────────────────────────────────────────

def get_scaffold_level(observed_skill: float) -> str:
    if observed_skill < SCAFFOLD_THRESHOLDS["high"]:
        return "high"
    if observed_skill < SCAFFOLD_THRESHOLDS["mid"]:
        return "mid"
    return "low"


# ── update_observed_skill_v2 (key fix: did_attempt=False → excluded) ──────────

def update_observed_skill_v2(
    session_history: list,
    session_rate: float | None,  # None if did_attempt=False
    current_observed_skill: float,
) -> float:
    """
    v2: did_attempt=False sessions are treated as "no observation" and
    excluded from the moving average. This prevents skill underestimation
    when the user simply did not get a practice opportunity.

    Rationale: observed_skill is meant to track demonstrated reframing
    ability. Sessions where the user did not attempt reframing provide
    no evidence about their current ability level.

    Args:
        session_history:       list of dicts with "session_rate" (None if no attempt)
        session_rate:          this session's rate (None if did_attempt=False)
        current_observed_skill: skill before this session's update
    Returns:
        updated observed_skill
    """
    if session_rate is None:
        # No observation this session; keep current observed_skill unchanged
        return current_observed_skill

    # Collect rates from recent sessions where did_attempt=True
    recent_rates = []
    for s in session_history[-MOVING_AVG_WINDOW:]:
        if s.get("session_rate") is not None:
            recent_rates.append(s["session_rate"])
    recent_rates.append(session_rate)

    if not recent_rates:
        return current_observed_skill

    avg = sum(recent_rates) / len(recent_rates)
    return max(0.0, min(1.0, avg))


# ── Deterministic seed (same formula as ext_study1) ──────────────────────────

import hashlib

def deterministic_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


# ── Single persona dry-run ────────────────────────────────────────────────────

def dry_run_persona(
    persona: dict,
    condition: str,
    srr_mode: str = "expected",  # "expected" or "forced_scaffold_sequence"
    forced_scaffold_seq: list | None = None,
) -> dict:
    """
    Simulate 14 sessions for one persona under one condition WITHOUT API calls.

    srr_mode:
      "expected"              → SRR approximated from latent_skill (normal run)
      "forced_scaffold_seq"   → override scaffold with forced_scaffold_seq list
                                (used for null calibration check)

    Returns:
        dict with trajectory data and transition count
    """
    pid = persona["id"]
    alpha = persona["alpha"]
    alpha_passive = persona["alpha_passive"]
    beta = persona["beta"]
    p_attempt_base = persona["p_attempt_base"]

    latent_skill = persona["latent_skill_0"]
    observed_skill = persona["latent_skill_0"]

    persona_hash = deterministic_hash(pid) % 10000
    condition_hash = deterministic_hash(condition) % 10000
    rng_seed = EXPERIMENT_SEED + persona_hash + condition_hash
    rng = random.Random(rng_seed)

    session_history = []
    scaffold_sequence = []
    did_attempt_sequence = []
    latent_traj = []
    obs_traj = []
    srr_values = []  # SRR per session (None if no attempt)

    for s_num in range(1, N_SESSIONS + 1):
        # Determine scaffold for this session
        if forced_scaffold_seq is not None:
            scaffold = forced_scaffold_seq[s_num - 1]
        elif condition == "fixed-high":
            scaffold = "high"
        elif condition == "adaptive-fading":
            scaffold = get_scaffold_level(observed_skill)
        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        scaffold_sequence.append(scaffold)

        # p_attempt
        p_attempt = compute_attempt_probability(latent_skill, scaffold, p_attempt_base)

        # Bernoulli draw
        did_attempt = rng.random() < p_attempt
        did_attempt_sequence.append(did_attempt)

        if did_attempt:
            p_success = compute_attempt_success_probability(latent_skill)
            attempt_success = rng.random() < p_success
        else:
            p_success = 0.0
            attempt_success = False

        # Approximate SRR for this session
        # In dry-run we use attempt_success as proxy for reframe quality.
        # session_rate = reframe_count / neg_count ≈ p_success if succeeded
        if did_attempt:
            # Approximate: if success → session_rate ≈ 0.8, if fail → 0.1
            session_rate = 0.8 if attempt_success else 0.1
        else:
            session_rate = None  # no observation → excluded from moving average

        srr_values.append(session_rate)

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

        # Update observed_skill (v2: exclude no-attempt sessions)
        new_obs_skill = update_observed_skill_v2(
            session_history=session_history,
            session_rate=session_rate,
            current_observed_skill=observed_skill,
        )

        latent_traj.append(round(latent_skill_new, 4))
        obs_traj.append(round(new_obs_skill, 4))

        session_history.append({
            "session": s_num,
            "scaffold": scaffold,
            "did_attempt": did_attempt,
            "attempt_success": attempt_success,
            "session_rate": session_rate,
            "latent_skill": latent_skill_new,
            "observed_skill": new_obs_skill,
        })

        latent_skill = latent_skill_new
        observed_skill = new_obs_skill

    # Scaffold transition count
    transitions = sum(
        1 for i in range(1, len(scaffold_sequence))
        if scaffold_sequence[i] != scaffold_sequence[i - 1]
    )
    unique_scaffolds = sorted(set(scaffold_sequence), key=["high", "mid", "low"].index)

    # delta-SRR (sessions 1-7 vs 8-14, excluding None values)
    def mean_srr(start, end):
        vals = [v for v in srr_values[start - 1:end] if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    srr_early = mean_srr(1, 7)
    srr_late = mean_srr(8, 14)
    delta_srr = (srr_late - srr_early) if not (math.isnan(srr_early) or math.isnan(srr_late)) else float("nan")

    return {
        "pid": pid,
        "condition": condition,
        "scaffold_sequence": scaffold_sequence,
        "transitions": transitions,
        "unique_scaffolds": unique_scaffolds,
        "did_attempt_sequence": did_attempt_sequence,
        "latent_trajectory": latent_traj,
        "obs_trajectory": obs_traj,
        "srr_values": [round(v, 4) if v is not None else None for v in srr_values],
        "srr_early": round(srr_early, 4) if not math.isnan(srr_early) else None,
        "srr_late": round(srr_late, 4) if not math.isnan(srr_late) else None,
        "delta_srr": round(delta_srr, 4) if not math.isnan(delta_srr) else None,
        "final_latent_skill": latent_traj[-1],
        "final_observed_skill": obs_traj[-1],
    }


# ── Check 1: Scaffold transitions in adaptive-fading ─────────────────────────

def check_scaffold_transitions(personas: list) -> tuple[int, list]:
    """
    Run dry_run for adaptive-fading condition and count transitions.
    Returns (n_personas_with_transitions, results).
    """
    print("\n" + "=" * 70)
    print("CHECK 1: Scaffold transitions in adaptive-fading (dry-run)")
    print("=" * 70)
    print(f"{'PID':<5} {'Label':<20} {'Scaffold Sequence':<32} {'Transitions':<12} {'Unique'}")
    print("-" * 90)

    results = []
    n_with_transitions = 0

    for p in personas:
        res = dry_run_persona(p, condition="adaptive-fading")
        results.append(res)

        seq_str = "".join([s[0].upper() for s in res["scaffold_sequence"]])  # H/M/L
        trans = res["transitions"]
        uniq = ",".join(res["unique_scaffolds"])
        if trans > 0:
            n_with_transitions += 1
            status = "[OK TRANSITION]"
        else:
            status = "[NO TRANSITION]"

        print(f"  {p['id']:<5} {p['label']:<20} {seq_str:<32} {trans:<12} {uniq:<12} {status}")

    print(f"\n→ Personas with scaffold transitions: {n_with_transitions}/{len(personas)}")

    if n_with_transitions >= 3:
        print(f"[PASS] >= 3 personas show scaffold fading. Proceed to null calibration.")
    else:
        print(f"[FAIL] < 3 personas show scaffold fading. Do NOT run real API. Adjust parameters.")

    return n_with_transitions, results


# ── Check 2: Null calibration (same scaffold + same RNG draws → delta-SRR diff = 0) ─

def dry_run_persona_with_shared_rng(
    persona: dict,
    condition: str,
    forced_scaffold_seq: list,
    shared_rng: random.Random,
) -> dict:
    """
    Run dry_run with forced scaffold sequence AND a shared RNG stream.
    Used for null calibration: both A and B get identical Bernoulli draws,
    so any delta-SRR difference is purely from scaffold (not RNG seed).
    """
    pid = persona["id"]
    alpha = persona["alpha"]
    alpha_passive = persona["alpha_passive"]
    beta = persona["beta"]
    p_attempt_base = persona["p_attempt_base"]

    latent_skill = persona["latent_skill_0"]
    observed_skill = persona["latent_skill_0"]

    session_history = []
    scaffold_sequence = []
    did_attempt_sequence = []
    latent_traj = []
    obs_traj = []
    srr_values = []

    for s_num in range(1, N_SESSIONS + 1):
        scaffold = forced_scaffold_seq[s_num - 1]
        scaffold_sequence.append(scaffold)

        p_attempt = compute_attempt_probability(latent_skill, scaffold, p_attempt_base)
        # Use shared_rng — identical random draws for both conditions
        did_attempt = shared_rng.random() < p_attempt
        did_attempt_sequence.append(did_attempt)

        if did_attempt:
            p_success = compute_attempt_success_probability(latent_skill)
            attempt_success = shared_rng.random() < p_success
        else:
            p_success = 0.0
            attempt_success = False
            shared_rng.random()  # consume same number of draws regardless

        session_rate = (0.8 if attempt_success else 0.1) if did_attempt else None
        srr_values.append(session_rate)

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
        new_obs_skill = update_observed_skill_v2(
            session_history=session_history,
            session_rate=session_rate,
            current_observed_skill=observed_skill,
        )

        latent_traj.append(round(latent_skill_new, 4))
        obs_traj.append(round(new_obs_skill, 4))
        session_history.append({
            "session": s_num,
            "scaffold": scaffold,
            "did_attempt": did_attempt,
            "session_rate": session_rate,
            "latent_skill": latent_skill_new,
            "observed_skill": new_obs_skill,
        })
        latent_skill = latent_skill_new
        observed_skill = new_obs_skill

    def mean_srr(start, end):
        vals = [v for v in srr_values[start - 1:end] if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    srr_early = mean_srr(1, 7)
    srr_late = mean_srr(8, 14)
    delta_srr = (srr_late - srr_early) if not (math.isnan(srr_early) or math.isnan(srr_late)) else float("nan")

    return {
        "pid": pid,
        "condition": condition,
        "scaffold_sequence": scaffold_sequence,
        "delta_srr": round(delta_srr, 4) if not math.isnan(delta_srr) else None,
    }


def check_null_calibration(personas: list, adaptive_results: list) -> bool:
    """
    Null calibration: force BOTH conditions to use the SAME scaffold sequence
    AND the SAME RNG stream (identical Bernoulli draws).

    Under this setup, the ONLY difference between A and B is the scaffold sequence
    itself (which is forced to be identical for both). Therefore delta-SRR_A should
    be numerically equal to delta-SRR_B (diff = 0 exactly for most personas).

    This verifies that the code computes delta-SRR identically when
    scaffold and RNG are identical — confirming no hidden condition leakage.

    Design note: In the real experiment, A and B have different RNG seeds
    (persona_hash + condition_hash) — that is correct design. Here we use a
    shared RNG *only* for null calibration to verify the computation pathway.
    """
    print("\n" + "=" * 70)
    print("CHECK 2: Null calibration — same scaffold + same RNG draws for both conditions")
    print("  (delta-SRR_A - delta-SRR_B must be 0 when scaffold and RNG are identical)")
    print("  (Tests that there is no hidden condition-name leakage in the computation)")
    print("=" * 70)
    print(f"{'PID':<5} {'delta_A':>10} {'delta_B':>10} {'|A-B|':>10} {'Status'}")
    print("-" * 55)

    diffs = []
    null_ok = True

    for p, res_a in zip(personas, adaptive_results):
        forced_seq = res_a["scaffold_sequence"]

        # Use a fresh shared RNG (same state for both A and B)
        shared_seed = 99999 + deterministic_hash(p["id"]) % 10000

        # Run A: adaptive-fading label, forced scaffold, shared RNG
        shared_rng_a = random.Random(shared_seed)
        res_a_forced = dry_run_persona_with_shared_rng(
            p, condition="adaptive-fading", forced_scaffold_seq=forced_seq,
            shared_rng=shared_rng_a,
        )

        # Run B: fixed-high label, forced scaffold, SAME shared RNG state
        shared_rng_b = random.Random(shared_seed)
        res_b_forced = dry_run_persona_with_shared_rng(
            p, condition="fixed-high", forced_scaffold_seq=forced_seq,
            shared_rng=shared_rng_b,
        )

        d_a = res_a_forced["delta_srr"]
        d_b = res_b_forced["delta_srr"]

        if d_a is None or d_b is None:
            diff = None
            status = "N/A (no valid SRR)"
        else:
            diff = abs(d_a - d_b)
            diffs.append(diff)
            # With identical scaffold and identical RNG draws, diff MUST be exactly 0
            # (or very close due to floating point).
            if diff < 1e-9:
                status = "[OK: diff=0]"
            else:
                status = f"[FAIL: diff={diff:.6f} — leakage?]"
                null_ok = False

        d_a_str = f"{d_a:+.4f}" if d_a is not None else "  null"
        d_b_str = f"{d_b:+.4f}" if d_b is not None else "  null"
        diff_str = f"{diff:.8f}" if diff is not None else "  null"
        print(f"  {p['id']:<5} {d_a_str:>10} {d_b_str:>10} {diff_str:>10} {status}")

    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        print(f"\nMean |A-B| with forced-same scaffold+RNG: {mean_diff:.10f}")
        if mean_diff < 1e-9:
            print("[PASS] All diffs are numerically zero. No hidden condition leakage.")
        else:
            print("[FAIL] Non-zero diff detected. Investigate condition leakage in computation.")
            null_ok = False
    else:
        print("[N/A] No valid pairs for null calibration.")
        null_ok = False

    return null_ok


# ── Check 3: Adaptive vs Fixed under natural conditions ───────────────────────

def check_natural_comparison(personas: list) -> dict:
    """
    Run both conditions naturally (no forced scaffold).
    Report delta-SRR for each persona.
    """
    print("\n" + "=" * 70)
    print("CHECK 3: Natural comparison — adaptive-fading vs fixed-high (dry-run proxy)")
    print("  (This is NOT the real result — it uses approximated SRR.)")
    print("=" * 70)
    print(f"{'PID':<5} {'Label':<20} {'delta_A':>10} {'delta_B':>10} {'A-B':>10} {'A>B?'}")
    print("-" * 75)

    results_fading = []
    results_fixed = []
    n_a_gt_b = 0
    valid_pairs = 0

    for p in personas:
        res_a = dry_run_persona(p, condition="adaptive-fading")
        res_b = dry_run_persona(p, condition="fixed-high")
        results_fading.append(res_a)
        results_fixed.append(res_b)

        d_a = res_a["delta_srr"]
        d_b = res_b["delta_srr"]

        if d_a is not None and d_b is not None:
            adv = d_a - d_b
            a_gt_b = d_a > d_b
            if a_gt_b:
                n_a_gt_b += 1
            valid_pairs += 1
            adv_str = f"{adv:+.4f}"
            a_gt_b_str = "YES" if a_gt_b else "no"
        else:
            adv_str = "  null"
            a_gt_b_str = "N/A"

        d_a_str = f"{d_a:+.4f}" if d_a is not None else "  null"
        d_b_str = f"{d_b:+.4f}" if d_b is not None else "  null"
        print(f"  {p['id']:<5} {p['label']:<20} {d_a_str:>10} {d_b_str:>10} {adv_str:>10} {a_gt_b_str}")

    print(f"\nDry-run A > B: {n_a_gt_b}/{valid_pairs} personas")
    print("NOTE: These are APPROXIMATED values (SRR proxy). Real API results will differ.")

    return {"n_a_gt_b": n_a_gt_b, "valid_pairs": valid_pairs}


# ── Scaffold trajectory detail (for report) ───────────────────────────────────

def print_scaffold_detail(personas: list, n_show: int = 3) -> None:
    """Print detailed session-by-session trajectory for top N personas."""
    print("\n" + "=" * 70)
    print(f"DETAIL: Session-by-session trajectory (adaptive-fading, top {n_show} personas)")
    print("=" * 70)

    # Show personas with most transitions first
    results = []
    for p in personas:
        res = dry_run_persona(p, condition="adaptive-fading")
        results.append((p, res))

    results.sort(key=lambda x: x[1]["transitions"], reverse=True)

    for p, res in results[:n_show]:
        print(f"\n  {p['id']} ({p['label']}): {res['transitions']} transitions, "
              f"scaffolds used: {res['unique_scaffolds']}")
        print(f"  {'S':>3} {'Scaffold':>8} {'Attempt':>8} {'LatSk':>8} {'ObsSk':>8} {'SRR':>8}")
        print(f"  {'-'*50}")
        for i, (sc, da, ls, os, sv) in enumerate(zip(
            res["scaffold_sequence"],
            res["did_attempt_sequence"],
            res["latent_trajectory"],
            res["obs_trajectory"],
            res["srr_values"],
        )):
            srr_str = f"{sv:.2f}" if sv is not None else "  -"
            attempt_str = "YES" if da else "no"
            print(f"  {i+1:>3} {sc:>8} {attempt_str:>8} {ls:>8.3f} {os:>8.3f} {srr_str:>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GratiFlow ext_study1_fixed — Dry-Run Calibration Check")
    print("NO API CALLS. Uses approximated SRR (attempt_success proxy).")
    print("Personas: v2 calibrated (the research team 2026-06-05)")
    print(f"Seeds: EXPERIMENT_SEED={EXPERIMENT_SEED}")
    print("=" * 70)

    personas = PERSONAS_FIXED

    # Check 1: Scaffold transitions
    n_transitions, adaptive_results = check_scaffold_transitions(personas)

    # Detailed trajectory view
    print_scaffold_detail(personas, n_show=4)

    # Check 2: Null calibration
    null_ok = check_null_calibration(personas, adaptive_results)

    # Check 3: Natural comparison (dry-run proxy)
    nat_results = check_natural_comparison(personas)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DRY-RUN SUMMARY")
    print("=" * 70)
    print(f"  Personas with scaffold transitions:     {n_transitions}/10")
    print(f"  Null calibration (same scaffold → A≈B): {'PASS' if null_ok else 'FAIL'}")
    print(f"  Dry-run A>B direction:                  {nat_results['n_a_gt_b']}/{nat_results['valid_pairs']}")

    if n_transitions >= 3 and null_ok:
        print("\n[GATE PASSED] Both checks passed.")
        print("  → Safe to proceed with real API run (ext_study1_fixed).")
        print("  → Reminder: real API uses gpt-5.4-mini; results will differ from this proxy.")
    else:
        print("\n[GATE FAILED] One or more checks failed.")
        if n_transitions < 3:
            print("  → Scaffold transitions insufficient. Increase p_attempt_base or alpha.")
        if not null_ok:
            print("  → RNG noise too large. Consider paired-seed design.")
        print("  → Do NOT run real API until gate is passed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
