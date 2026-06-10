"""
latent_skill_model.py
======================
GratiFlow v2 Evaluation — Latent Skill Model

Implements 7 functions exactly as specified in evaluation_protocol_v2.md Sections 2.3–3.3.
This module models the internal skill dynamics of synthetic users.

Design principle:
  Experimental conditions (adaptive-fading / fixed-high) affect latent_skill ONLY through
  the practice opportunity pathway:
    scaffold_level → p_attempt → did_attempt → attempt_success → latent_skill_delta

  Condition names are NEVER passed to LLM prompts or to this module's functions.

Author: team member (experiment lead, the research team)
Date: 2026-06-04
Protocol: evaluation_protocol_v2.md (the research team, 2026-06-04, PI-approved)
"""

import math
import random as stdlib_random
from typing import Optional


# ── Section 2.3: Latent Skill Update ─────────────────────────────────────────

def update_latent_skill(
    latent_skill: float,
    did_attempt: bool,
    attempt_success: bool,
    observed_ai_model: bool,
    alpha: float,
    alpha_passive: float,
    beta: float,
) -> float:
    """
    Update latent_skill based on this session's practice experience.

    Causal pathway:
      scaffold_level → practice_opportunity → attempt → success/failure → latent_skill_delta

    The experimental condition (adaptive-fading vs fixed-high) affects ONLY the scaffold_level,
    which in turn affects the probability of self-attempt (practice opportunity).
    The condition name is NEVER passed to this function or to the LLM prompt.

    Args:
        latent_skill:       current latent_skill (before this session)
        did_attempt:        whether the user attempted self-reframing
        attempt_success:    whether the attempt was successful (only if did_attempt=True)
        observed_ai_model:  whether the user observed AI's model reframing (i.e., not did_attempt)
        alpha:              learning rate for successful attempts
        alpha_passive:      learning rate for passive observation (alpha_passive << alpha)
        beta:               penalty for failed attempts (small, keeps user in study)

    Returns:
        new_skill: updated latent_skill in [0.0, 1.0]
    """
    delta = 0.0

    if did_attempt and attempt_success:
        # Mastery experience: strongest learning signal (generation effect)
        delta += alpha
    elif did_attempt and not attempt_success:
        # Failed attempt: slight penalty, but partial benefit from the attempt itself
        # (desirable difficulty — partial benefit even from failure)
        delta += alpha * 0.3 - beta

    if observed_ai_model and not did_attempt:
        # Passive observation only: weak learning (no generation effect)
        delta += alpha_passive

    new_skill = max(0.0, min(1.0, latent_skill + delta))
    return new_skill


# ── Section 2.4: Practice Opportunity Model ───────────────────────────────────

# Scaffold-level multiplier on attempt probability.
# This is the ONLY numerical pathway through which condition affects latent_skill.
SCAFFOLD_ATTEMPT_MULTIPLIER: dict[str, float] = {
    "low":  1.8,   # AI says "try it yourself" → strong practice opportunity
    "mid":  1.2,   # AI gives partial hint → moderate practice opportunity
    "high": 0.5,   # AI models the answer → user tends to observe passively
}


def compute_attempt_probability(
    latent_skill: float,
    scaffold_level: str,
    p_attempt_base: float,
) -> float:
    """
    Compute the probability that the synthetic user attempts self-reframing.

    Key design:
      - low scaffold → higher attempt probability (AI invites self-attempt)
      - high scaffold → lower attempt probability (AI provides model answer;
        user tends to receive rather than generate)
      - latent_skill modulates: higher skill → more confident → more likely to attempt

    This is the ONLY pathway through which the experimental condition
    (adaptive-fading vs fixed-high) influences latent_skill growth.

    Args:
        latent_skill:    current latent_skill in [0.0, 1.0]
        scaffold_level:  one of "high", "mid", "low"
        p_attempt_base:  persona's base attempt probability

    Returns:
        p_attempt: probability in [0.0, 0.95]
    """
    assert scaffold_level in SCAFFOLD_ATTEMPT_MULTIPLIER, (
        f"scaffold_level must be 'high', 'mid', or 'low'. Got: {scaffold_level!r}"
    )

    multiplier = SCAFFOLD_ATTEMPT_MULTIPLIER[scaffold_level]

    # Skill-based modulation: higher skill → more confident to attempt
    skill_boost = 0.3 * latent_skill  # up to +0.3 at skill=1.0

    p_attempt = p_attempt_base * multiplier + skill_boost
    p_attempt = max(0.0, min(0.95, p_attempt))  # cap at 0.95 to avoid certainty

    return p_attempt


def compute_attempt_success_probability(latent_skill: float) -> float:
    """
    Probability that a self-reframing attempt succeeds (is coherent and valid).

    Depends ONLY on latent_skill, not on the experimental condition.
    This ensures the condition influences outcomes only through practice opportunity,
    not through direct biasing of success.

    Args:
        latent_skill: current latent_skill in [0.0, 1.0]

    Returns:
        p_success: probability in [0.05, 0.95]
    """
    # Sigmoid-like mapping: low skill → mostly fails, high skill → mostly succeeds
    # Concave power: early gains are meaningful
    p_success = latent_skill ** 0.7
    p_success = max(0.05, min(0.95, p_success))
    return p_success


# ── Section 3.2: Attempt and Reframe Description Generators ──────────────────

def make_attempt_description(
    did_attempt: bool,
    attempt_success: bool,
    latent_skill: float,
) -> str:
    """
    Generate a natural-language description of the student's self-attempt behavior.

    CRITICAL: This description MUST NOT contain any of the following:
      - Condition names: "adaptive-fading", "fixed-high", "fading", "fixed", "condition A/B"
      - Scaffold level values: "high", "mid", "low" (as technical terms)
      - Any statement implying which experimental condition the student is in

    Args:
        did_attempt:     whether the student attempted self-reframing
        attempt_success: whether the attempt succeeded (only meaningful if did_attempt=True)
        latent_skill:    current skill level for quality calibration

    Returns:
        attempt_description: natural-language string for injection into generation prompt
    """
    if not did_attempt:
        return (
            "The student does not attempt to reframe negative events on their own. "
            "They simply describe what happened without reinterpretation."
        )
    elif attempt_success:
        if latent_skill < 0.3:
            quality = "tentative and partial"
        elif latent_skill < 0.6:
            quality = "moderate but with some uncertainty"
        else:
            quality = "clear and confident"
        return (
            f"The student attempts to reframe a negative event on their own. "
            f"The reframe quality is {quality}."
        )
    else:
        return (
            "The student attempts to reframe a negative event on their own, "
            "but the attempt is awkward, forced, or incomplete — it does not fully succeed. "
            "The student seems to be trying but struggling."
        )


def make_reframe_instruction(
    did_attempt: bool,
    attempt_success: bool,
    neg_count: int,
) -> str:
    """
    Generate the reframing instruction for the journal entry generation prompt.

    CRITICAL: Must not reference experimental conditions or scaffold levels.

    Args:
        did_attempt:     whether the student attempted self-reframing
        attempt_success: whether the attempt succeeded (only meaningful if did_attempt=True)
        neg_count:       number of negative events for this session

    Returns:
        reframe_instruction: natural-language instruction for journal entry generation
    """
    if neg_count == 0:
        return "No negative events today. Focus on positive reflections."
    if not did_attempt:
        return (
            "Do NOT include any spontaneous positive reframing. "
            "The student simply describes negative events as they happened."
        )
    elif attempt_success:
        return (
            "Include ONE spontaneous positive reframing of a negative event. "
            "The reframe should match the quality described above."
        )
    else:
        return (
            "Include ONE attempted but unsuccessful reframing of a negative event. "
            "The attempt should feel forced, incomplete, or unconvincing."
        )


# ── Section 3.3: Previous Session Summary Generator ──────────────────────────

def make_previous_summary(
    session_num: int,
    user_entry: Optional[str],
    ai_reframing_response: Optional[str],
    scaffold_level: str,
    latent_skill: float,
) -> str:
    """
    Generate a summary of the previous session for the next session's generation prompt.

    NOTE: scaffold_level is described functionally (what the AI did),
    NOT by condition name or technical scaffold level label.
    This prevents condition leakage into the generation prompt.

    Args:
        session_num:           the session number that just completed
        user_entry:            the student's journal entry from that session (unused in v2,
                               kept for interface compatibility)
        ai_reframing_response: the AI's reframing response (unused directly, kept for interface)
        scaffold_level:        the scaffold level used ("high", "mid", "low") — described functionally
        latent_skill:          the student's latent_skill AFTER this session's update

    Returns:
        summary: string for injection into the next session's generation prompt
    """
    # Describe AI behavior functionally, NOT by condition name or scaffold level label
    ai_behavior_descriptions = {
        "high": "The AI provided a complete model reframing with explanation.",
        "mid": "The AI gave a partial hint toward a positive angle without completing the reframe.",
        "low": "The AI simply encouraged the student to try reframing on their own without hints.",
    }

    ai_behavior = ai_behavior_descriptions.get(scaffold_level, ai_behavior_descriptions["high"])

    summary = (
        f"In Session {session_num}, the student wrote about their day. "
        f"{ai_behavior} "
        f"The student's current skill level is approximately {latent_skill:.2f}."
    )

    return summary


# ── Section 3.4: Negative Event Count Sampler ────────────────────────────────

def sample_neg_count(
    latent_skill: float,
    neg_tendency: float,
    rng: stdlib_random.Random,
) -> int:
    """
    Sample the number of negative events for this session using a Poisson draw.

    Higher latent_skill → fewer negative framings (student recognizes positives more).
    This effect is gradual and stochastic, not deterministic.

    Args:
        latent_skill:   current latent_skill in [0.0, 1.0]
        neg_tendency:   persona's baseline expected negative event count
        rng:            persona-specific Random instance for reproducibility

    Returns:
        count: number of negative events in [0, 3]
    """
    adjusted_tendency = neg_tendency * (1.0 - 0.4 * latent_skill)
    adjusted_tendency = max(0.1, adjusted_tendency)

    # Poisson approximation via numpy-free method (stdlib random only)
    # Using the Knuth algorithm for Poisson sampling
    L = math.exp(-adjusted_tendency)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    raw_count = k - 1

    # Clamp to [0, 3]
    count = max(0, min(3, raw_count))
    return count
