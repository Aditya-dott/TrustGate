"""
Policy Engine and Supervisor.

The Supervisor is the ONLY component that produces a verdict. Every other
module (Prompt Guard scorer, semantic analyzer, session tracker) is a
reporter that hands the Supervisor structured signals; the Supervisor
combines them here and calls the shot.

Two-stage combination:

    1. Gated signal combination (this file, see `combine_signals`):
         - classifier score is the primary, controlling signal
         - semantic signals (attack_similarity, deviation) can only
           AMPLIFY suspicion the classifier already found; they cannot
           create suspicion from nothing
         - a gate opens from 0 to 1 as the classifier score crosses
           from `gate_low` to `gate_high`; below gate_low the gate is
           zero so semantic contribution is fully muted
         - semantic contribution is a weighted mix of attack_similarity
           and deviation, capped absolutely at `max_semantic_boost`
           so it can never dominate the score
         - result: raw_combined_score = clip(0,1, classifier + gate * capped_semantic)

    2. Tier penalty (from SessionTracker, session_tracker.py):
         - the Supervisor reads the session's tier as it stood BEFORE
           this message, gets the tier's additive penalty, and adds it
           to the raw_combined_score to get the final_combined_score
         - the tier is a memory of the CONVERSATION's history, so this
           step makes the system progressively stricter as suspicion
           accumulates over messages

Decision (thresholds are configurable):

    final_combined_score >= block_threshold  ->  BLOCK
    final_combined_score >= flag_threshold   ->  FLAG
    otherwise                                ->  ALLOW

Session tracker update:

    The tracker's EMA is fed the RAW combined score (before the tier
    penalty). Feeding it the final score would cause the penalty to
    inflate its own next input - a positive feedback loop. Using the
    raw score keeps the EMA a faithful average of underlying message
    suspicion; the penalty is a session-level bias applied on top.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from prompt_guard.scorer import PromptGuardReport, PromptGuardScorer
from supervisor.session_tracker import SessionState, SessionTracker


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

DECISION_ALLOW = "ALLOW"
DECISION_FLAG = "FLAG"
DECISION_BLOCK = "BLOCK"
DECISIONS: tuple[str, ...] = (DECISION_ALLOW, DECISION_FLAG, DECISION_BLOCK)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PolicyEngineConfig:
    """
    All tunable knobs for the gated combination and decision.

    Defaults are chosen so that:
        - a benign prompt with any level of semantic noise stays below
          the flag threshold as long as the classifier says benign
          (gate closes semantic contribution)
        - a clearly malicious prompt (classifier >= gate_high) gets a
          modest amplification from semantics that reinforces the
          existing signal without dominating it
        - a marginal prompt in a Normal session stays flagged, and only
          gets blocked once the session tier rises
    """
    # Gate: below gate_low, semantic signals are fully muted (gate=0).
    # Between gate_low and gate_high, gate ramps linearly from 0 to 1.
    # Above gate_high, gate is 1 (semantic signals contribute fully).
    gate_low: float = 0.30
    gate_high: float = 0.60

    # Weights for the semantic mix inside the boost. They should be
    # non-negative and sum to 1 for the boost to have a clean [0,1] range.
    attack_similarity_weight: float = 0.60
    deviation_weight: float = 0.40

    # Absolute cap on the semantic boost, applied BEFORE the gate. This
    # is the "maximum amplification" cap the spec calls for.
    max_semantic_boost: float = 0.20

    # Decision thresholds against the final combined score.
    block_threshold: float = 0.75
    flag_threshold: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.gate_low <= self.gate_high <= 1.0:
            raise ValueError("Require 0 <= gate_low <= gate_high <= 1.")
        if self.attack_similarity_weight < 0 or self.deviation_weight < 0:
            raise ValueError("Semantic weights must be non-negative.")
        if not 0.0 <= self.max_semantic_boost <= 1.0:
            raise ValueError("max_semantic_boost must be in [0, 1].")
        if not 0.0 <= self.flag_threshold <= self.block_threshold <= 1.0:
            raise ValueError("Require 0 <= flag_threshold <= block_threshold <= 1.")


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------

@dataclass
class SupervisorDecision:
    """
    Complete supervisor decision object - includes every signal, every
    intermediate value, the session state at evaluation time, and the
    final verdict. This is what the monitor page and the SQLite record
    consume; nothing important is elided.
    """
    # Signals
    classifier_score: float                  # pg_report.suspicion_score
    attack_similarity: float                 # pg_report.attack_similarity
    deviation_score: float                   # pg_report.deviation_score
    classifier_confidence: float             # pg_report.confidence

    # Gated combination intermediates
    semantic_gate: float                     # 0..1 gate value
    raw_semantic_mix: float                  # weighted mix before cap
    capped_semantic: float                   # after max_semantic_boost cap
    semantic_boost_applied: float            # gate * capped_semantic
    raw_combined_score: float                # classifier + boost (clipped)

    # Session state AT EVALUATION (pre-update, drives the penalty)
    session_ema_at_evaluation: float
    session_tier_at_evaluation: str
    consecutive_suspicious_at_evaluation: int
    tier_penalty_applied: float

    # Post-update session state (this message's contribution folded in)
    session_ema_after: float
    session_tier_after: str
    consecutive_suspicious_after: int

    # Final
    final_combined_score: float              # raw_combined + tier_penalty
    message_number: int                      # 1-indexed
    session_id: str

    decision: str                            # ALLOW / FLAG / BLOCK
    detected_attack_type: str | None         # from pg_report; None if benign or ALLOW-benign

    # For rendering context on the monitor page
    top_attack_class: str                    # always populated
    class_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure helper functions (testable without the Supervisor class)
# ---------------------------------------------------------------------------

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_gate(classifier_score: float, gate_low: float, gate_high: float) -> float:
    """Linear ramp: 0 below gate_low, 1 above gate_high."""
    if classifier_score <= gate_low:
        return 0.0
    if classifier_score >= gate_high:
        return 1.0
    return (classifier_score - gate_low) / (gate_high - gate_low)


def combine_signals(
    classifier_score: float,
    attack_similarity: float,
    deviation_score: float,
    config: PolicyEngineConfig,
) -> dict[str, float]:
    """
    Gated combination. All inputs are expected in [0, 1] but we clip
    defensively. Returns a dict with the intermediates and the final
    raw_combined_score (before the tier penalty).
    """
    c = clip01(classifier_score)
    a = clip01(attack_similarity)
    d = clip01(deviation_score)

    gate = compute_gate(c, config.gate_low, config.gate_high)
    raw_mix = (
        config.attack_similarity_weight * a
        + config.deviation_weight * d
    )
    capped = min(raw_mix, config.max_semantic_boost)
    boost = gate * capped
    raw_combined = clip01(c + boost)
    return {
        "gate": gate,
        "raw_semantic_mix": raw_mix,
        "capped_semantic": capped,
        "semantic_boost_applied": boost,
        "raw_combined_score": raw_combined,
    }


def decide_verdict(final_score: float, config: PolicyEngineConfig) -> str:
    if final_score >= config.block_threshold:
        return DECISION_BLOCK
    if final_score >= config.flag_threshold:
        return DECISION_FLAG
    return DECISION_ALLOW


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    """
    Ties Prompt Guard, the session tracker, and the policy engine into
    one entry point. The FastAPI backend holds a single Supervisor per
    process and calls `evaluate` for every prompt.
    """

    def __init__(
        self,
        prompt_guard: PromptGuardScorer,
        session_tracker: SessionTracker,
        config: PolicyEngineConfig | None = None,
    ):
        if prompt_guard is None:
            raise ValueError("A PromptGuardScorer is required.")
        if session_tracker is None:
            raise ValueError("A SessionTracker is required.")
        self.prompt_guard = prompt_guard
        self.session_tracker = session_tracker
        self.config = config or PolicyEngineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        prompt: str,
        session_id: str,
    ) -> tuple[PromptGuardReport, SupervisorDecision]:
        """
        Full evaluation: run Prompt Guard on the prompt, apply the policy
        engine using the session's memory, and return both the raw
        Prompt Guard report and the final SupervisorDecision.

        The FastAPI backend uses the decision to route the prompt (BLOCK
        never reaches the LLM; FLAG and ALLOW do) and stores both objects
        in the SQLite record.
        """
        pg_report = self.prompt_guard.score(prompt)
        decision = self.decide(pg_report, session_id)
        return pg_report, decision

    def decide(self, pg_report: PromptGuardReport, session_id: str) -> SupervisorDecision:
        """
        Apply the policy engine to a Prompt Guard report in the context
        of a session. Useful as its own entry point for tests and for
        re-scoring when a report has already been produced.
        """
        if not session_id:
            raise ValueError("session_id must be a non-empty string.")

        # Snapshot BEFORE the tracker sees this message.
        state_before: SessionState = self.session_tracker.get_state(session_id)

        # Combine signals
        combo = combine_signals(
            pg_report.suspicion_score,
            pg_report.attack_similarity,
            pg_report.deviation_score,
            self.config,
        )
        raw_combined = combo["raw_combined_score"]

        # Apply the tier penalty from BEFORE state. Feed the tracker
        # the RAW combined score (not the penalized final score) so the
        # penalty does not compound into its own next input.
        tier_penalty = state_before.tier_penalty
        final_combined = clip01(raw_combined + tier_penalty)

        # Ask the tracker to fold this message into the session.
        state_after: SessionState = self.session_tracker.update(session_id, raw_combined)

        # Decide
        verdict = decide_verdict(final_combined, self.config)

        # Attach attack type only when the decision is not ALLOW; ALLOW
        # results should read as "clean" on the monitor page.
        if verdict == DECISION_ALLOW:
            detected_type = None
        else:
            # Prefer the explicit detected_attack_type from pg_report when set;
            # fall back to the always-populated top_attack_class otherwise.
            detected_type = pg_report.detected_attack_type or pg_report.top_attack_class

        return SupervisorDecision(
            classifier_score=pg_report.suspicion_score,
            attack_similarity=pg_report.attack_similarity,
            deviation_score=pg_report.deviation_score,
            classifier_confidence=pg_report.confidence,

            semantic_gate=combo["gate"],
            raw_semantic_mix=combo["raw_semantic_mix"],
            capped_semantic=combo["capped_semantic"],
            semantic_boost_applied=combo["semantic_boost_applied"],
            raw_combined_score=raw_combined,

            session_ema_at_evaluation=state_before.ema,
            session_tier_at_evaluation=state_before.tier,
            consecutive_suspicious_at_evaluation=state_before.consecutive_suspicious,
            tier_penalty_applied=tier_penalty,

            session_ema_after=state_after.ema,
            session_tier_after=state_after.tier,
            consecutive_suspicious_after=state_after.consecutive_suspicious,

            final_combined_score=final_combined,
            message_number=state_after.message_count,
            session_id=session_id,

            decision=verdict,
            detected_attack_type=detected_type,
            top_attack_class=pg_report.top_attack_class,
            class_probabilities=dict(pg_report.class_probabilities),
        )
