"""
Session risk tracker.

Maintains per-session risk memory across all messages in a conversation.
The Supervisor consults this tracker before every decision and updates it
after every combined score is computed.

Mechanism (all values configurable via SessionTrackerConfig):

    EMA (Exponential Moving Average) of raw combined scores:
        ema_new = alpha * combined_score + (1 - alpha) * ema_old
        Recent messages weigh more heavily than older ones. Clean
        messages naturally decay the EMA back toward zero.

    Four named tiers, mapped from the EMA value:
        Normal    when ema < 0.30
        Elevated  when 0.30 <= ema < 0.50
        High      when 0.50 <= ema < 0.70
        Critical  when 0.70 <= ema

    Tier penalty (additive, applied by the Supervisor to the CURRENT
    message's combined score before comparing against block/flag
    thresholds):
        Normal    +0.00
        Elevated  +0.10
        High      +0.20
        Critical  +0.30

    Streak detection: independent of EMA. Counts consecutive messages
    whose combined score is at or above `suspicion_floor` (default 0.4).
    A single clean message resets the streak to zero. The streak forces
    a minimum tier regardless of where the EMA sits:
        streak >= 3   forces at least Elevated
        streak >= 4   forces at least High
        streak >= 5   forces Critical
    The reported tier is `max(ema_tier, streak_forced_tier)`.

    Decay: tiers are not sticky. When suspicious messages stop, the EMA
    shrinks and the ema_tier drops. A clean message also resets the
    streak, so the streak_forced_tier drops immediately.

The tracker feeds the EMA on the RAW combined score (classifier + gated
semantic boost), NOT on the final score after the tier penalty. This
avoids a positive feedback loop where a high tier inflates the input to
its own next update.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Tier constants and ordering
# ---------------------------------------------------------------------------

TIER_NORMAL = "Normal"
TIER_ELEVATED = "Elevated"
TIER_HIGH = "High"
TIER_CRITICAL = "Critical"

TIER_ORDER: list[str] = [TIER_NORMAL, TIER_ELEVATED, TIER_HIGH, TIER_CRITICAL]


def tier_rank(tier: str) -> int:
    """Ordinal position of a tier. Higher = more severe."""
    try:
        return TIER_ORDER.index(tier)
    except ValueError as exc:
        raise ValueError(f"Unknown tier: {tier!r}") from exc


def max_tier(*tiers: str) -> str:
    """Return the most severe tier among the arguments."""
    if not tiers:
        return TIER_NORMAL
    return max(tiers, key=tier_rank)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SessionTrackerConfig:
    """
    All thresholds and weights the tracker uses. Defaults chosen so:
        - EMA responds quickly (alpha=0.4) but still smooths noise
        - Tier bands are wide enough that a single spike does not push
          to Critical, but a sustained pattern does
        - Streak detection catches slow escalation in 3-5 messages
    """
    ema_alpha: float = 0.4

    # Ordered list of (upper_bound_exclusive, tier).
    # tier used when ema < upper_bound; sentinel `inf` catches Critical.
    tier_thresholds: tuple[tuple[float, str], ...] = (
        (0.30, TIER_NORMAL),
        (0.50, TIER_ELEVATED),
        (0.70, TIER_HIGH),
        (float("inf"), TIER_CRITICAL),
    )

    # Additive penalty applied to the combined score of the NEXT message
    # to be evaluated, based on the current tier.
    tier_penalties: dict[str, float] = field(default_factory=lambda: {
        TIER_NORMAL: 0.00,
        TIER_ELEVATED: 0.10,
        TIER_HIGH: 0.20,
        TIER_CRITICAL: 0.30,
    })

    # Messages with score >= floor count toward the streak; below resets it.
    suspicion_floor: float = 0.40

    # (min_streak, forced_tier); applied cumulatively - a streak of 5
    # matches all three rows and the highest match wins.
    streak_escalations: tuple[tuple[int, str], ...] = (
        (3, TIER_ELEVATED),
        (4, TIER_HIGH),
        (5, TIER_CRITICAL),
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if not self.tier_thresholds:
            raise ValueError("tier_thresholds must be non-empty.")
        # tier_thresholds must be strictly increasing
        prev = -float("inf")
        for upper, _ in self.tier_thresholds:
            if upper <= prev:
                raise ValueError("tier_thresholds must be strictly increasing.")
            prev = upper
        for tier in TIER_ORDER:
            if tier not in self.tier_penalties:
                raise ValueError(f"tier_penalties missing tier {tier!r}.")
        if not 0.0 <= self.suspicion_floor <= 1.0:
            raise ValueError("suspicion_floor must be in [0, 1].")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """A snapshot of one session's risk memory."""
    session_id: str
    message_count: int
    ema: float
    ema_tier: str                    # tier derived from EMA alone
    tier: str                        # final = max(ema_tier, streak_tier)
    tier_penalty: float              # additive penalty for the NEXT message
    consecutive_suspicious: int      # current streak length
    started_at: str                  # ISO8601 UTC of first update
    updated_at: str                  # ISO8601 UTC of most recent update
    last_score: float | None = None  # last raw combined score fed in

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_from_ema(ema: float, thresholds: tuple[tuple[float, str], ...]) -> str:
    for upper, tier in thresholds:
        if ema < upper:
            return tier
    # Should be unreachable because `inf` is the last threshold, but be safe.
    return thresholds[-1][1]


def _tier_from_streak(streak: int, escalations: tuple[tuple[int, str], ...]) -> str:
    forced = TIER_NORMAL
    for min_streak, tier in escalations:
        if streak >= min_streak:
            forced = max_tier(forced, tier)
    return forced


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SessionTracker:
    """
    In-memory session risk tracker. One instance per app process; the
    FastAPI backend holds and shares it across requests.

    All reads return snapshots (shallow copies via dataclasses.replace) so
    callers cannot accidentally mutate the tracker's internal state.
    """

    def __init__(self, config: SessionTrackerConfig | None = None):
        self.config: SessionTrackerConfig = config or SessionTrackerConfig()
        self._sessions: dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_state(self, session_id: str) -> SessionState:
        """
        Return the current state for `session_id`. Creates a fresh Normal
        state if the session has not been seen before. Returns a snapshot
        copy, not the internal object.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = self._new_state(session_id)
        return dataclasses.replace(self._sessions[session_id])

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def update(self, session_id: str, combined_score: float) -> SessionState:
        """
        Fold `combined_score` into the session's EMA / streak / tier.
        Feed the RAW combined score here (classifier + gated semantic
        boost), NOT the final score with the tier penalty added -
        otherwise the penalty compounds into its own next input.

        Returns a snapshot of the updated state.
        """
        score = float(max(0.0, min(1.0, combined_score)))

        state = self._sessions.setdefault(session_id, self._new_state(session_id))

        # EMA update
        a = self.config.ema_alpha
        state.ema = a * score + (1.0 - a) * state.ema

        # Streak update
        if score >= self.config.suspicion_floor:
            state.consecutive_suspicious += 1
        else:
            state.consecutive_suspicious = 0

        # Derived tiers
        state.ema_tier = _tier_from_ema(state.ema, self.config.tier_thresholds)
        streak_forced = _tier_from_streak(
            state.consecutive_suspicious, self.config.streak_escalations
        )
        state.tier = max_tier(state.ema_tier, streak_forced)
        state.tier_penalty = self.config.tier_penalties[state.tier]

        # Bookkeeping
        state.message_count += 1
        state.last_score = score
        state.updated_at = _utc_now()

        return dataclasses.replace(state)

    def reset(self, session_id: str) -> None:
        """Drop a session's state entirely."""
        self._sessions.pop(session_id, None)

    def reset_all(self) -> None:
        self._sessions.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _new_state(self, session_id: str) -> SessionState:
        now = _utc_now()
        return SessionState(
            session_id=session_id,
            message_count=0,
            ema=0.0,
            ema_tier=TIER_NORMAL,
            tier=TIER_NORMAL,
            tier_penalty=self.config.tier_penalties[TIER_NORMAL],
            consecutive_suspicious=0,
            started_at=now,
            updated_at=now,
            last_score=None,
        )
