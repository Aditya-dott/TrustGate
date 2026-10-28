"""
Step 3 comprehensive validation for SessionTracker.

Sections:
    1. Fresh session defaults
    2. EMA arithmetic correctness (exact values)
    3. Tier boundary transitions on rising EMA
    4. Tier decay on clean messages
    5. Streak detection: increments, reset, forced tier
    6. Streak-forced tier when EMA alone would stay Normal
    7. Tier penalty scales with tier
    8. Session isolation across multiple session_ids
    9. Score clamping (negative / > 1 inputs)
    10. Reset / reset_all
    11. Config validation
    12. State snapshot immutability (caller can't mutate tracker state)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supervisor.session_tracker import (  # noqa: E402
    SessionTracker, SessionTrackerConfig, SessionState,
    TIER_NORMAL, TIER_ELEVATED, TIER_HIGH, TIER_CRITICAL,
    tier_rank, max_tier,
)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


class TestRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def check(self, cond: bool, description: str, detail: str = "") -> None:
        if cond:
            self.passed += 1
            print(f"  [{PASS}] {description}")
        else:
            self.failed += 1
            print(f"  [{FAIL}] {description}")
            if detail:
                print(f"         {detail}")
            self.failures.append(description)

    def approx(self, a: float, b: float, tol: float = 1e-6) -> bool:
        return math.isclose(a, b, abs_tol=tol)

    def summary(self) -> int:
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print(f"Summary: {self.passed}/{total} passed")
        if self.failures:
            print("Failures:")
            for m in self.failures:
                print(f"  - {m}")
            return 1
        print("All tests passed.")
        return 0


def test_fresh_session(t: TestRunner) -> None:
    print("\n[Section 1] fresh session defaults")
    tracker = SessionTracker()
    state = tracker.get_state("s1")
    t.check(state.session_id == "s1", "session_id matches")
    t.check(state.message_count == 0, "message_count == 0")
    t.check(state.ema == 0.0, "ema == 0")
    t.check(state.ema_tier == TIER_NORMAL, "ema_tier == Normal")
    t.check(state.tier == TIER_NORMAL, "tier == Normal")
    t.check(state.tier_penalty == 0.0, "tier_penalty == 0")
    t.check(state.consecutive_suspicious == 0, "streak == 0")
    t.check(state.last_score is None, "last_score is None")
    t.check(state.started_at == state.updated_at,
            "started_at == updated_at on fresh session")


def test_ema_arithmetic(t: TestRunner) -> None:
    print("\n[Section 2] EMA arithmetic")
    cfg = SessionTrackerConfig(ema_alpha=0.4)
    tracker = SessionTracker(cfg)

    # First message: score 0.5 → EMA = 0.4*0.5 + 0.6*0 = 0.2
    s1 = tracker.update("s", 0.5)
    t.check(t.approx(s1.ema, 0.20), f"after 0.5: ema=0.20, got {s1.ema:.4f}")
    t.check(s1.message_count == 1, "message_count == 1")
    t.check(s1.last_score == 0.5, "last_score == 0.5")

    # Second message: score 0.8 → EMA = 0.4*0.8 + 0.6*0.2 = 0.44
    s2 = tracker.update("s", 0.8)
    t.check(t.approx(s2.ema, 0.44), f"after 0.8: ema=0.44, got {s2.ema:.4f}")
    t.check(s2.message_count == 2, "message_count == 2")

    # Third message: score 0.0 → EMA = 0.4*0.0 + 0.6*0.44 = 0.264
    s3 = tracker.update("s", 0.0)
    t.check(t.approx(s3.ema, 0.264), f"after 0.0: ema=0.264, got {s3.ema:.4f}")
    t.check(s3.message_count == 3, "message_count == 3")

    # Alternative alpha
    tracker2 = SessionTracker(SessionTrackerConfig(ema_alpha=0.2))
    s = tracker2.update("s", 1.0)  # 0.2*1 + 0.8*0 = 0.2
    t.check(t.approx(s.ema, 0.2), f"alpha=0.2 after 1.0: ema=0.2, got {s.ema:.4f}")


def test_tier_rise(t: TestRunner) -> None:
    print("\n[Section 3] tier transitions on rising EMA")
    tracker = SessionTracker()
    # Feed a run of 0.8 scores and watch tier climb.
    # Threshold defaults: N<0.30 <=E<0.50 <=H<0.70 <=C
    # With alpha=0.4 and scores of 0.8:
    #   ema_1 = 0.32   -> Elevated
    #   ema_2 = 0.512  -> High
    #   ema_3 = 0.6272 -> High
    #   ema_4 = 0.7  approx -> Critical (0.4*0.8 + 0.6*0.6272 = 0.6963 - still High)
    #   ema_5 = 0.7178 -> Critical
    seq = [0.8] * 6
    tiers = []
    emas = []
    for x in seq:
        s = tracker.update("s", x)
        tiers.append(s.tier)
        emas.append(s.ema)

    t.check(tiers[0] == TIER_ELEVATED,
            f"after 1x0.8: Elevated (ema={emas[0]:.3f}, tier={tiers[0]})")
    t.check(tiers[1] == TIER_HIGH,
            f"after 2x0.8: High (ema={emas[1]:.3f}, tier={tiers[1]})")
    # Critical arrives once EMA crosses 0.70. With alpha=0.4 starting from 0
    # and a stream of 0.8: EMA converges to 0.8, so eventually Critical.
    t.check(any(tier == TIER_CRITICAL for tier in tiers),
            f"eventually Critical (emas={[round(x, 3) for x in emas]})")


def test_tier_decay(t: TestRunner) -> None:
    print("\n[Section 4] tier decay on clean messages")
    tracker = SessionTracker()
    # Push to Critical
    for _ in range(6):
        tracker.update("s", 1.0)
    state = tracker.get_state("s")
    t.check(state.tier == TIER_CRITICAL,
            f"pushed to Critical (ema={state.ema:.3f}, streak={state.consecutive_suspicious})")

    # Now feed clean messages and watch tier fall
    tiers_over_time = []
    for i in range(10):
        s = tracker.update("s", 0.0)
        tiers_over_time.append((round(s.ema, 3), s.tier, s.consecutive_suspicious))

    # Streak resets to 0 on the first clean message
    t.check(tiers_over_time[0][2] == 0, "streak resets to 0 on first clean message")

    # EMA decays monotonically toward 0 when input is 0
    emas = [x[0] for x in tiers_over_time]
    t.check(all(emas[i] > emas[i+1] for i in range(len(emas)-1)),
            f"EMA strictly decays on clean messages ({emas})")

    # Tier eventually returns to Normal
    final_tier = tiers_over_time[-1][1]
    t.check(final_tier == TIER_NORMAL,
            f"tier returns to Normal after enough clean messages (final={final_tier})")


def test_streak_detection(t: TestRunner) -> None:
    print("\n[Section 5] streak detection")
    tracker = SessionTracker()
    cfg = tracker.config
    floor = cfg.suspicion_floor  # default 0.4

    # Below floor: no streak
    for _ in range(5):
        s = tracker.update("s", floor - 0.01)
    t.check(s.consecutive_suspicious == 0, "scores below floor don't build streak")

    tracker.reset("s")

    # At or above floor: streak grows
    scores = [floor, floor, floor + 0.01, floor + 0.02]
    for score in scores:
        s = tracker.update("s", score)
    t.check(s.consecutive_suspicious == 4,
            f"streak grows to 4 (got {s.consecutive_suspicious})")

    # One clean message resets the streak
    s = tracker.update("s", 0.0)
    t.check(s.consecutive_suspicious == 0,
            f"streak resets on clean message (got {s.consecutive_suspicious})")


def test_streak_forces_tier(t: TestRunner) -> None:
    print("\n[Section 6] streak forces tier above what EMA alone would give")
    # Use scores at the suspicion floor so the streak builds, and use a
    # small alpha so the EMA rises slowly enough to stay under the
    # Elevated threshold (0.30) for the whole run. This isolates the
    # streak mechanism from the EMA-tier mechanism.
    # alpha=0.2, score=0.4 each message:
    #   ema_1 = 0.080  ema_tier=Normal  streak=1  streak_tier=Normal  tier=Normal
    #   ema_2 = 0.144  ema_tier=Normal  streak=2  streak_tier=Normal  tier=Normal
    #   ema_3 = 0.195  ema_tier=Normal  streak=3  streak_tier=Elevated tier=Elevated
    #   ema_4 = 0.236  ema_tier=Normal  streak=4  streak_tier=High     tier=High
    #   ema_5 = 0.269  ema_tier=Normal  streak=5  streak_tier=Critical tier=Critical
    tracker = SessionTracker(SessionTrackerConfig(ema_alpha=0.2))
    scores = [0.4] * 6
    results = []
    for x in scores:
        s = tracker.update("s", x)
        results.append((round(s.ema, 3), s.ema_tier, s.tier,
                        s.consecutive_suspicious))

    # First two: both Normal
    t.check(results[0][1] == TIER_NORMAL and results[0][2] == TIER_NORMAL,
            f"msg1: ema_tier=Normal, tier=Normal (got {results[0]})")
    t.check(results[1][1] == TIER_NORMAL and results[1][2] == TIER_NORMAL,
            f"msg2: ema_tier=Normal, tier=Normal (got {results[1]})")

    # Third: EMA still under 0.3 but streak=3 forces Elevated
    t.check(results[2][1] == TIER_NORMAL,
            f"msg3: ema_tier still Normal (got {results[2][1]}, ema={results[2][0]})")
    t.check(results[2][2] == TIER_ELEVATED,
            f"msg3: streak forces tier to Elevated (got {results[2][2]})")

    # Fourth: streak=4 forces High
    t.check(results[3][1] == TIER_NORMAL,
            f"msg4: ema_tier still Normal (got {results[3][1]}, ema={results[3][0]})")
    t.check(results[3][2] == TIER_HIGH,
            f"msg4: streak forces tier to High (got {results[3][2]})")

    # Fifth: streak=5 forces Critical
    t.check(results[4][1] == TIER_NORMAL,
            f"msg5: ema_tier still Normal (got {results[4][1]}, ema={results[4][0]})")
    t.check(results[4][2] == TIER_CRITICAL,
            f"msg5: streak forces tier to Critical (got {results[4][2]})")

    # One clean message resets streak AND drops tier back toward what EMA supports
    s_clean = tracker.update("s", 0.0)
    t.check(s_clean.consecutive_suspicious == 0, "clean message resets streak")
    t.check(tier_rank(s_clean.tier) <= tier_rank(TIER_ELEVATED),
            f"clean message drops tier back to ema_tier (got {s_clean.tier}, "
            f"ema={s_clean.ema:.3f})")


def test_tier_penalty_scaling(t: TestRunner) -> None:
    print("\n[Section 7] tier penalty scales with tier")
    cfg = SessionTrackerConfig()
    t.check(cfg.tier_penalties[TIER_NORMAL] == 0.0, "Normal penalty = 0")
    t.check(cfg.tier_penalties[TIER_ELEVATED] > cfg.tier_penalties[TIER_NORMAL],
            "Elevated penalty > Normal")
    t.check(cfg.tier_penalties[TIER_HIGH] > cfg.tier_penalties[TIER_ELEVATED],
            "High penalty > Elevated")
    t.check(cfg.tier_penalties[TIER_CRITICAL] > cfg.tier_penalties[TIER_HIGH],
            "Critical penalty > High")


def test_session_isolation(t: TestRunner) -> None:
    print("\n[Section 8] session isolation")
    tracker = SessionTracker()
    # Push s1 to Critical
    for _ in range(6):
        tracker.update("s1", 1.0)
    # s2 stays fresh
    s2 = tracker.get_state("s2")
    t.check(s2.tier == TIER_NORMAL, "s2 stays Normal despite s1 being Critical")
    t.check(s2.ema == 0.0, "s2 EMA still 0")
    t.check(s2.message_count == 0, "s2 has no messages")

    # Feeding s2 doesn't touch s1
    tracker.update("s2", 0.1)
    s1_after = tracker.get_state("s1")
    t.check(s1_after.tier == TIER_CRITICAL, "s1 still Critical")
    t.check(s1_after.message_count == 6, "s1 message_count preserved")

    # list_sessions returns both
    all_ids = set(tracker.list_sessions())
    t.check(all_ids == {"s1", "s2"}, f"list_sessions = {all_ids}")


def test_score_clamping(t: TestRunner) -> None:
    print("\n[Section 9] score clamping")
    tracker = SessionTracker()
    s = tracker.update("s", -0.5)  # clamp to 0
    t.check(s.last_score == 0.0, f"negative score clamped to 0 (got {s.last_score})")
    t.check(s.ema == 0.0, "EMA reflects clamped 0")

    tracker.reset("s")
    s = tracker.update("s", 2.5)  # clamp to 1
    t.check(s.last_score == 1.0, f"score>1 clamped to 1 (got {s.last_score})")
    t.check(t.approx(s.ema, 0.4), f"EMA reflects clamped 1 (got {s.ema:.4f})")


def test_reset(t: TestRunner) -> None:
    print("\n[Section 10] reset / reset_all")
    tracker = SessionTracker()
    for _ in range(5):
        tracker.update("s1", 1.0)
        tracker.update("s2", 1.0)
    t.check(tracker.has_session("s1"), "s1 exists before reset")

    tracker.reset("s1")
    t.check(not tracker.has_session("s1"), "s1 gone after reset")
    t.check(tracker.has_session("s2"), "s2 unaffected by s1 reset")
    # get_state re-creates a fresh session
    s = tracker.get_state("s1")
    t.check(s.ema == 0.0 and s.message_count == 0,
            "get_state after reset gives fresh Normal state")

    tracker.reset_all()
    t.check(tracker.list_sessions() == [], "reset_all clears everything")


def test_config_validation(t: TestRunner) -> None:
    print("\n[Section 11] config validation")
    # Bad alpha
    try:
        SessionTrackerConfig(ema_alpha=0.0)
        t.check(False, "alpha=0 should raise")
    except ValueError:
        t.check(True, "alpha=0 raises ValueError")
    try:
        SessionTrackerConfig(ema_alpha=1.5)
        t.check(False, "alpha>1 should raise")
    except ValueError:
        t.check(True, "alpha>1 raises ValueError")
    # Bad thresholds (not increasing)
    try:
        SessionTrackerConfig(tier_thresholds=(
            (0.5, TIER_NORMAL), (0.3, TIER_ELEVATED),
            (float("inf"), TIER_CRITICAL),
        ))
        t.check(False, "non-increasing thresholds should raise")
    except ValueError:
        t.check(True, "non-increasing thresholds raises ValueError")
    # Missing penalty
    try:
        SessionTrackerConfig(tier_penalties={TIER_NORMAL: 0.0})
        t.check(False, "missing tier penalty should raise")
    except ValueError:
        t.check(True, "missing tier penalty raises ValueError")


def test_snapshot_immutability(t: TestRunner) -> None:
    print("\n[Section 12] snapshot immutability")
    tracker = SessionTracker()
    tracker.update("s", 0.6)
    snap = tracker.get_state("s")
    original_ema = snap.ema
    original_count = snap.message_count

    # Mutate the returned snapshot; internal state should not change
    snap.ema = 999.0
    snap.message_count = -1

    fresh = tracker.get_state("s")
    t.check(fresh.ema == original_ema,
            f"mutating snapshot doesn't corrupt tracker EMA "
            f"(orig={original_ema}, fresh={fresh.ema})")
    t.check(fresh.message_count == original_count,
            f"mutating snapshot doesn't corrupt tracker message_count "
            f"(orig={original_count}, fresh={fresh.message_count})")


def test_max_tier(t: TestRunner) -> None:
    print("\n[Bonus] max_tier / tier_rank helpers")
    t.check(max_tier(TIER_NORMAL, TIER_CRITICAL) == TIER_CRITICAL, "max includes Critical")
    t.check(max_tier(TIER_ELEVATED, TIER_HIGH) == TIER_HIGH, "max Elevated vs High")
    t.check(max_tier() == TIER_NORMAL, "empty max defaults to Normal")
    t.check(tier_rank(TIER_NORMAL) < tier_rank(TIER_CRITICAL), "Normal ranks below Critical")


def main() -> int:
    t = TestRunner()
    test_fresh_session(t)
    test_ema_arithmetic(t)
    test_tier_rise(t)
    test_tier_decay(t)
    test_streak_detection(t)
    test_streak_forces_tier(t)
    test_tier_penalty_scaling(t)
    test_session_isolation(t)
    test_score_clamping(t)
    test_reset(t)
    test_config_validation(t)
    test_snapshot_immutability(t)
    test_max_tier(t)
    return t.summary()


if __name__ == "__main__":
    sys.exit(main())
