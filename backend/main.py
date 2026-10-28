"""
TrustGate FastAPI backend.

Wires together every module built in the earlier steps and exposes them
as HTTP endpoints for the React frontend to call.

    POST   /api/session/new              -> new session id
    POST   /api/chat                     -> full pipeline: guard -> supervisor -> LLM -> sieve -> DB
    GET    /api/session/{sid}/state      -> current session tracker state
    GET    /api/session/{sid}/records    -> all pipeline records for the session
    GET    /api/session/{sid}/analytics  -> aggregated stats for the monitor page
    GET    /api/records/{record_id}      -> one full record
    DELETE /api/session/{sid}            -> reset session (tracker + DB rows)
    GET    /api/health                   -> liveness + backend info

Design notes:
    - The heavy modules (embedding backend, classifier, spaCy) load
      exactly once at startup via FastAPI's lifespan hook.
    - The Supervisor holds the SessionTracker; the SessionTracker's
      state is in-memory, so it resets when the process restarts. The
      SQLite records survive across restarts.
    - Env vars used:
        TRUSTGATE_DB_PATH        override sqlite path (":memory:" in tests)
        TRUSTGATE_LLM_BACKEND    force a specific LLM backend
        GROQ_API_KEY / GEMINI_API_KEY / OLLAMA_HOST as documented in llm/client.py
    - The frontend is served separately on port 5173 (Vite), so CORS
      is opened for local dev.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.database import DEFAULT_DB_PATH, Database
from llm.client import LLMClient, LLMClientError
from privacy_sieve.pii_detector import PrivacySieve
from prompt_guard.scorer import PromptGuardScorer
from supervisor.policy_engine import (
    DECISION_ALLOW, DECISION_BLOCK, DECISION_FLAG, Supervisor,
)
from supervisor.session_tracker import SessionTracker


# ---------------------------------------------------------------------------
# App-level state
# ---------------------------------------------------------------------------

class AppState:
    """Container for the heavyweight loaded modules."""
    scorer: PromptGuardScorer
    tracker: SessionTracker
    supervisor: Supervisor
    llm: LLMClient
    sieve: PrivacySieve
    db: Database


state: AppState = AppState()


PRETTY_ATTACK_LABEL: dict[str, str] = {
    "prompt_injection":  "Prompt Injection",
    "jailbreak":         "Jailbreak Attempt",
    "data_exfiltration": "Data Exfiltration Attempt",
    "model_extraction":  "Model Extraction Attempt",
}


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[TrustGate] loading pipeline...")

    state.scorer = PromptGuardScorer()
    state.tracker = SessionTracker()
    state.supervisor = Supervisor(state.scorer, state.tracker)
    state.llm = LLMClient()
    state.sieve = PrivacySieve()

    db_path = os.environ.get("TRUSTGATE_DB_PATH", str(DEFAULT_DB_PATH))
    state.db = Database(db_path)
    state.db.init_schema()

    print(f"[TrustGate] ready. "
          f"LLM backend={state.llm.backend_name} "
          f"model={state.llm.model_name} "
          f"NER={'on' if state.sieve.ner_enabled else 'off (regex-only)'} "
          f"db={db_path}")
    yield
    state.db.close()


app = FastAPI(title="TrustGate", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class ChatResponse(BaseModel):
    id: int
    session_id: str
    message_number: int
    decision: str
    final_output: str
    detected_attack_type: str | None = None
    session_tier_after: str
    session_ema_after: float
    consecutive_suspicious_after: int
    llm_error: str | None = None
    pii_entities_found: int = 0


class NewSessionResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    ok: bool
    llm_backend: str
    llm_model: str
    ner_enabled: bool


# ---------------------------------------------------------------------------
# Dependencies (make it easy for tests to override)
# ---------------------------------------------------------------------------

def get_supervisor() -> Supervisor:
    return state.supervisor


def get_tracker() -> SessionTracker:
    return state.tracker


def get_llm() -> LLMClient:
    return state.llm


def get_sieve() -> PrivacySieve:
    return state.sieve


def get_db() -> Database:
    return state.db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pretty_block_message(attack_type: str | None) -> str:
    """User-facing message shown when a prompt is blocked."""
    pretty = PRETTY_ATTACK_LABEL.get(attack_type or "", "malicious content")
    return f"⚠️ Blocked — {pretty} detected"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health(llm: LLMClient = Depends(get_llm),
           sieve: PrivacySieve = Depends(get_sieve)) -> HealthResponse:
    return HealthResponse(
        ok=True,
        llm_backend=llm.backend_name,
        llm_model=llm.model_name,
        ner_enabled=sieve.ner_enabled,
    )


@app.post("/api/session/new", response_model=NewSessionResponse)
def new_session() -> NewSessionResponse:
    return NewSessionResponse(session_id=f"session-{uuid.uuid4().hex[:12]}")


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    supervisor: Supervisor = Depends(get_supervisor),
    llm: LLMClient = Depends(get_llm),
    sieve: PrivacySieve = Depends(get_sieve),
    db: Database = Depends(get_db),
) -> ChatResponse:
    """
    Full pipeline for one message:
        1. Prompt Guard scores + Supervisor decides (updates session tracker)
        2. If ALLOW/FLAG, LLM completes; if BLOCK, LLM is skipped entirely
        3. If the LLM produced a response, Privacy Sieve scans/masks it
        4. Full record persisted; response fields returned to the chat page
    """
    # 1. Guard + supervisor
    pg_report, decision = supervisor.evaluate(req.prompt, req.session_id)

    # 2. LLM (skipped on BLOCK)
    llm_response: str | None = None
    llm_error: str | None = None
    llm_info = None
    if decision.decision != DECISION_BLOCK:
        try:
            llm_response = llm.complete(req.prompt)
            llm_info = llm.last_call_info
        except LLMClientError as e:
            llm_error = str(e)

    # 3. Privacy Sieve (only if we have a response)
    ps_report = sieve.scan(llm_response) if llm_response else None

    # 4. Final output for the chat page
    if decision.decision == DECISION_BLOCK:
        final_output = _pretty_block_message(decision.detected_attack_type)
    elif llm_error:
        final_output = "⚠️ The AI model is unavailable right now. Please try again."
    elif ps_report:
        final_output = ps_report.cleaned_text
    else:
        final_output = ""

    # 5. Persist
    record_id = db.insert_record(
        session_id=req.session_id,
        original_prompt=req.prompt,
        pg_report=pg_report,
        decision=decision,
        llm_response=llm_response,
        llm_info=llm_info,
        llm_error=llm_error,
        ps_report=ps_report,
        final_output=final_output,
    )

    return ChatResponse(
        id=record_id,
        session_id=req.session_id,
        message_number=decision.message_number,
        decision=decision.decision,
        final_output=final_output,
        detected_attack_type=decision.detected_attack_type,
        session_tier_after=decision.session_tier_after,
        session_ema_after=decision.session_ema_after,
        consecutive_suspicious_after=decision.consecutive_suspicious_after,
        llm_error=llm_error,
        pii_entities_found=len(ps_report.entities) if ps_report else 0,
    )


@app.get("/api/session/{session_id}/state")
def get_session_state(
    session_id: str,
    tracker: SessionTracker = Depends(get_tracker),
) -> dict[str, Any]:
    state = tracker.get_state(session_id)
    return state.to_dict()


@app.get("/api/session/{session_id}/records")
def get_records(
    session_id: str,
    since_id: int = 0,
    limit: int = 500,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    """
    Records for a session, ordered by message_number ASC.
    The monitor page uses `since_id` to fetch only new rows added since
    its last poll, keeping traffic small once a session has many messages.
    """
    records = db.get_records_by_session(session_id, since_id=since_id, limit=limit)
    return {"records": records, "count": len(records)}


@app.get("/api/session/{session_id}/analytics")
def get_analytics(
    session_id: str,
    db: Database = Depends(get_db),
    tracker: SessionTracker = Depends(get_tracker),
) -> dict[str, Any]:
    """
    Aggregated stats for the monitor page's charts. Also includes the
    live SessionTracker state so the session overview panel shows the
    up-to-date tier/EMA even before the next chart update.
    """
    analytics = db.get_session_analytics(session_id)
    analytics["current_state"] = tracker.get_state(session_id).to_dict()
    return analytics


@app.get("/api/records/{record_id}")
def get_record(
    record_id: int,
    db: Database = Depends(get_db),
) -> dict[str, Any]:
    record = db.get_record(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/api/session/{session_id}")
def delete_session(
    session_id: str,
    db: Database = Depends(get_db),
    tracker: SessionTracker = Depends(get_tracker),
) -> dict[str, Any]:
    """
    Clear both the tracker state and all DB rows for this session.
    Useful for demos and for the frontend's 'reset session' button.
    """
    tracker.reset(session_id)
    deleted = db.clear_session(session_id)
    return {"ok": True, "session_id": session_id, "rows_deleted": deleted}


# ---------------------------------------------------------------------------
# Entry point for `python backend/main.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("TRUSTGATE_PORT", 8000)),
        reload=False,
    )
