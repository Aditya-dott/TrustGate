"""
SQLite persistence for TrustGate.

One record per processed prompt. Every field the spec calls out is stored:
    - original prompt
    - full Prompt Guard report (JSON)
    - full Supervisor decision (JSON)
    - full Privacy Sieve report (JSON, nullable for BLOCKed prompts)
    - LLM response before masking (nullable for BLOCK or LLM error)
    - final output shown to the user
    - session ID, message number, timestamp
    - denormalized decision/score/tier columns so the analytics queries
      that back the monitor page don't need to parse JSON on every row

Concurrency: the FastAPI backend is single-process; sqlite3 with
`check_same_thread=False` plus a threading lock is enough. All writes go
through the same connection, and every method that touches the DB grabs
the lock first, so nothing races.

Schema is created idempotently on first connect. The database file lives
next to `main.py` by default; tests pass `:memory:` and get a scratch DB
that vanishes when the connection closes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm.client import LLMCallInfo
from privacy_sieve.pii_detector import PrivacySieveReport
from prompt_guard.scorer import PromptGuardReport
from supervisor.policy_engine import SupervisorDecision

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "trustgate.db"

# Schema idempotent so init_schema can be called on every start.
SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_records (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                TEXT    NOT NULL,
    message_number            INTEGER NOT NULL,
    timestamp                 TEXT    NOT NULL,

    original_prompt           TEXT    NOT NULL,

    prompt_guard_report_json  TEXT    NOT NULL,
    supervisor_decision_json  TEXT    NOT NULL,

    decision                  TEXT    NOT NULL,
    detected_attack_type      TEXT,
    classifier_score          REAL,
    attack_similarity         REAL,
    deviation_score           REAL,
    raw_combined_score        REAL,
    final_combined_score      REAL,
    session_tier              TEXT,
    session_ema               REAL,
    consecutive_suspicious    INTEGER,
    tier_penalty_applied      REAL,

    llm_response              TEXT,
    llm_backend               TEXT,
    llm_model                 TEXT,
    llm_latency_ms            INTEGER,
    llm_error                 TEXT,

    privacy_sieve_report_json TEXT,
    pii_entities_found        INTEGER NOT NULL DEFAULT 0,

    final_output              TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pipeline_session
    ON pipeline_records(session_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_session_msg
    ON pipeline_records(session_id, message_number);
"""


class Database:
    """
    Thin wrapper over a single sqlite3 connection. Reused across
    requests. The FastAPI startup event creates one Database, the
    shutdown event closes it.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path: str = str(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_schema(self) -> None:
        conn = self.connect()
        with self._lock:
            conn.executescript(SCHEMA)
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert_record(
        self,
        *,
        session_id: str,
        original_prompt: str,
        pg_report: PromptGuardReport,
        decision: SupervisorDecision,
        llm_response: str | None,
        llm_info: LLMCallInfo | None,
        llm_error: str | None,
        ps_report: PrivacySieveReport | None,
        final_output: str,
    ) -> int:
        """
        Store one full pipeline pass. Returns the new row id.
        message_number and session state fields are read from `decision`
        (the SupervisorDecision knows both the pre- and post-update
        session state).
        """
        conn = self.connect()
        with self._lock:
            cur = conn.execute(
                """
                INSERT INTO pipeline_records (
                    session_id, message_number, timestamp,
                    original_prompt,
                    prompt_guard_report_json,
                    supervisor_decision_json,
                    decision, detected_attack_type,
                    classifier_score, attack_similarity, deviation_score,
                    raw_combined_score, final_combined_score,
                    session_tier, session_ema,
                    consecutive_suspicious, tier_penalty_applied,
                    llm_response, llm_backend, llm_model, llm_latency_ms, llm_error,
                    privacy_sieve_report_json, pii_entities_found,
                    final_output
                ) VALUES (
                    ?, ?, ?,
                    ?,
                    ?,
                    ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?,
                    ?
                )
                """,
                (
                    session_id, decision.message_number,
                    datetime.now(timezone.utc).isoformat(),

                    original_prompt,
                    json.dumps(pg_report.to_dict()),
                    json.dumps(decision.to_dict()),

                    decision.decision, decision.detected_attack_type,
                    decision.classifier_score, decision.attack_similarity,
                    decision.deviation_score,
                    decision.raw_combined_score, decision.final_combined_score,
                    decision.session_tier_after, decision.session_ema_after,
                    decision.consecutive_suspicious_after,
                    decision.tier_penalty_applied,

                    llm_response,
                    llm_info.backend if llm_info else None,
                    llm_info.model if llm_info else None,
                    llm_info.latency_ms if llm_info else None,
                    llm_error,

                    json.dumps(ps_report.to_dict()) if ps_report else None,
                    len(ps_report.entities) if ps_report else 0,

                    final_output,
                ),
            )
            conn.commit()
            row_id = cur.lastrowid
        assert row_id is not None
        return int(row_id)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_record(self, record_id: int) -> dict[str, Any] | None:
        conn = self.connect()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM pipeline_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_records_by_session(
        self,
        session_id: str,
        since_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Records for a session, ordered by message_number ASC. `since_id`
        is used by the monitor page's polling loop to fetch only new
        rows added since the last poll.
        """
        conn = self.connect()
        with self._lock:
            rows = conn.execute(
                """
                SELECT * FROM pipeline_records
                WHERE session_id = ? AND id > ?
                ORDER BY message_number ASC
                LIMIT ?
                """,
                (session_id, since_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_all_sessions(self) -> list[str]:
        conn = self.connect()
        with self._lock:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM pipeline_records"
            ).fetchall()
        return [r["session_id"] for r in rows]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def get_session_analytics(self, session_id: str) -> dict[str, Any]:
        """
        Aggregated stats for the monitor page's charts:
            - total messages
            - decision counts (ALLOW / FLAG / BLOCK)
            - attack type distribution
            - per-message score & tier timeline
            - PII type counts across all Privacy Sieve reports
        """
        conn = self.connect()
        with self._lock:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM pipeline_records WHERE session_id = ?",
                (session_id,),
            ).fetchone()["c"]

            decision_counts = {
                row["decision"]: row["c"]
                for row in conn.execute(
                    """SELECT decision, COUNT(*) AS c FROM pipeline_records
                       WHERE session_id = ? GROUP BY decision""",
                    (session_id,),
                ).fetchall()
            }

            attack_type_counts = {
                row["detected_attack_type"]: row["c"]
                for row in conn.execute(
                    """SELECT detected_attack_type, COUNT(*) AS c
                       FROM pipeline_records
                       WHERE session_id = ? AND detected_attack_type IS NOT NULL
                       GROUP BY detected_attack_type""",
                    (session_id,),
                ).fetchall()
            }

            timeline = [
                dict(row) for row in conn.execute(
                    """SELECT id, message_number,
                              classifier_score, raw_combined_score,
                              final_combined_score, session_ema,
                              session_tier, decision, detected_attack_type,
                              timestamp
                       FROM pipeline_records WHERE session_id = ?
                       ORDER BY message_number ASC""",
                    (session_id,),
                ).fetchall()
            ]

            # PII type counts across all responses in the session
            pii_rows = conn.execute(
                """SELECT privacy_sieve_report_json FROM pipeline_records
                   WHERE session_id = ?
                     AND privacy_sieve_report_json IS NOT NULL""",
                (session_id,),
            ).fetchall()
        pii_type_counts: dict[str, int] = {}
        for row in pii_rows:
            report = json.loads(row["privacy_sieve_report_json"])
            for etype, count in report.get("entity_type_counts", {}).items():
                pii_type_counts[etype] = pii_type_counts.get(etype, 0) + count

        return {
            "session_id": session_id,
            "total_messages": total,
            "decision_counts": decision_counts,
            "attack_type_counts": attack_type_counts,
            "score_timeline": timeline,
            "pii_type_counts": pii_type_counts,
        }

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------
    def clear_session(self, session_id: str) -> int:
        conn = self.connect()
        with self._lock:
            cur = conn.execute(
                "DELETE FROM pipeline_records WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return cur.rowcount

    def clear_all(self) -> int:
        conn = self.connect()
        with self._lock:
            cur = conn.execute("DELETE FROM pipeline_records")
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """
        Turn a Row into a plain dict. JSON columns are decoded so the
        API returns nested objects, not stringified blobs.
        """
        d: dict[str, Any] = dict(row)
        for json_field, out_field in (
            ("prompt_guard_report_json",  "prompt_guard_report"),
            ("supervisor_decision_json",  "supervisor_decision"),
            ("privacy_sieve_report_json", "privacy_sieve_report"),
        ):
            raw = d.pop(json_field, None)
            d[out_field] = json.loads(raw) if raw else None
        return d
