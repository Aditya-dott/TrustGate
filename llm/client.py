"""
LLM client for TrustGate.

The LLM receives only the clean supervisor-approved prompt. It has no
awareness of the security layer around it: no system prompt about
guardrails, no injected sentinels, no metadata. This module is the
single place any LLM call goes through, so swapping the underlying
model touches nothing else in the pipeline.

Preferred backends, in priority order:

    1. GroqBackend       - free tier, extremely fast (LPU inference).
                           Default model: openai/gpt-oss-20b.
                           Signup: https://console.groq.com/
    2. GeminiBackend     - free tier via Google AI Studio.
                           Default model: gemini-2.0-flash.
                           Signup: https://aistudio.google.com/apikey
    3. OllamaBackend     - fully local, no API key. Requires the user
                           to install Ollama and pull a model, e.g.
                           `ollama pull llama3.2`.
                           Default model: llama3.2.
    4. MockBackend       - deterministic canned response for tests and
                           for demos when no network / API key is set.

Backend auto-selection at import time (see `make_default_backend`) reads
these environment variables:
    GROQ_API_KEY          -> GroqBackend
    GEMINI_API_KEY        -> GeminiBackend
    OLLAMA_HOST           -> OllamaBackend at that host
                             (or http://localhost:11434 if unset when
                             `TRUSTGATE_LLM_BACKEND=ollama`)
    TRUSTGATE_LLM_BACKEND -> force a specific backend by name,
                             overriding auto-detection. Values:
                             "groq" | "gemini" | "ollama" | "mock"

Public API:
    client = LLMClient()                # auto-select
    text = client.complete(prompt)      # returns str
    info = client.health_check()        # returns dict, never raises

All backends raise LLMClientError on any failure so the FastAPI backend
can catch a single exception type and surface a friendly message.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests


DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMClientError(Exception):
    """Any LLM backend failure. Caught by the FastAPI backend."""


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMCallInfo:
    """Metadata about a completed LLM call, for the monitor / DB record."""
    backend: str
    model: str
    latency_ms: int
    prompt_length_chars: int
    response_length_chars: int
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to respond in clean plain text
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Always respond in clean plain text only. "
    "Never use markdown formatting such as **, ##, *, |, ---, or <br> tags. "
    "Do not use tables, bullet symbols, or any special formatting characters. "
    "Use normal sentences and paragraphs with blank lines between sections. "
    "If you need to present structured information, use numbered lists with "
    "plain numbers like 1. 2. 3. and simple line breaks."
)


# ---------------------------------------------------------------------------
# Post-processing: strip any markdown the LLM still sneaks in
# ---------------------------------------------------------------------------

import re as _re

def clean_response(text: str) -> str:
    """
    Strip common markdown / formatting artifacts from the LLM response.
    Applied automatically before the text reaches Privacy Sieve or the
    user. Deterministic, no ML, safe to run on any string.
    """
    if not text:
        return text

    s = text

    # HTML tags: <br>, <br/>, <b>, </b>, <i>, </i>, <strong>, etc.
    s = _re.sub(r'<br\s*/?>',  '\n', s, flags=_re.IGNORECASE)
    s = _re.sub(r'</?(?:b|i|em|strong|u|s|del|mark|code|pre|span|div|p|h[1-6])(?:\s[^>]*)?>',
                '', s, flags=_re.IGNORECASE)

    # Bold / italic markers: **text**, __text__, *text*, _text_
    # Do bold first (** / __) then italic (* / _)
    s = _re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = _re.sub(r'__(.+?)__', r'\1', s)
    s = _re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\1', s)
    s = _re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', s)

    # Headings: ## Heading -> Heading
    s = _re.sub(r'^#{1,6}\s+', '', s, flags=_re.MULTILINE)

    # Horizontal rules: ---, ***, ___
    s = _re.sub(r'^[\s]*[-*_]{3,}\s*$', '', s, flags=_re.MULTILINE)

    # Table pipes: lines starting/ending with | or containing multiple |
    # Convert table rows to comma-separated values
    def _clean_table_line(m):
        line = m.group(0)
        # Separator rows like |---|---|---| -> skip entirely
        if _re.match(r'^[\s|:\-]+$', line):
            return ''
        # Data rows: | A | B | C | -> A, B, C
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]  # drop empty edge cells
        return ', '.join(cells)

    s = _re.sub(r'^.*\|.*\|.*$', _clean_table_line, s, flags=_re.MULTILINE)

    # Inline code backticks: `code` -> code
    s = _re.sub(r'`([^`]+)`', r'\1', s)

    # Code block fences: ```language ... ```
    s = _re.sub(r'```\w*\n?', '', s)

    # Blockquote markers: > text -> text
    s = _re.sub(r'^>\s?', '', s, flags=_re.MULTILINE)

    # Bullet markers: - item, * item -> item (preserve numbered 1. 2. 3.)
    s = _re.sub(r'^[\s]*[-*•]\s+', '', s, flags=_re.MULTILINE)

    # Collapse 3+ consecutive blank lines into 2
    s = _re.sub(r'\n{3,}', '\n\n', s)

    # Strip leading/trailing whitespace
    s = s.strip()

    return s


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Interface every backend implements."""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send `prompt` to the model, return the response text."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """
        Return a dict describing this backend's state. Must never raise;
        report errors inside the dict as `ok: False, error: "..."`.
        """

    @property
    def model(self) -> str:
        """Model identifier for logging / DB records."""
        return getattr(self, "_model", "unknown")


# ---------------------------------------------------------------------------
# Mock backend (always safe, no network)
# ---------------------------------------------------------------------------

MOCK_RESPONSE_TEMPLATE = (
    "Thanks for your message. Here's a brief response: I've read what you "
    "sent and I'm happy to help. If you'd like to reach out to the team "
    "directly, you can email support@trustgate-demo.example.com or call "
    "+91-9876543210. Best regards, Team TrustGate."
)

MOCK_RESPONSE_SHORT = (
    "Here's a helpful mock response. Configure a real LLM backend "
    "(GROQ_API_KEY, GEMINI_API_KEY, or Ollama) to get real answers."
)


class MockBackend(LLMBackend):
    """
    Deterministic canned response. No network. Use for tests, CI, and
    for demoing the UI when no API key is available.

    The default response deliberately contains PII markers (email and
    phone) so Privacy Sieve has something to redact in end-to-end tests.
    """

    name = "mock"

    def __init__(self, response: str | None = None, model: str = "mock-v1"):
        self._response = response if response is not None else MOCK_RESPONSE_TEMPLATE
        self._model = model

    def complete(self, prompt: str) -> str:
        # Simulate a small amount of work so latency metrics are non-zero
        time.sleep(0.005)
        return self._response

    def health_check(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.name, "model": self._model,
                "note": "mock backend - no real inference"}


# ---------------------------------------------------------------------------
# Groq backend (free tier, fast)
# ---------------------------------------------------------------------------

class GroqBackend(LLMBackend):
    """
    Groq's OpenAI-compatible chat completions endpoint.

    Signup: https://console.groq.com/
    Auth: bearer token in Authorization header.
    Default model: llama-3.3-70b-versatile.
    """

    name = "groq"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-oss-20b",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise LLMClientError(
                "GroqBackend needs GROQ_API_KEY (env var or constructor arg)."
            )
        self._model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            resp = requests.post(
                self.ENDPOINT, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise LLMClientError(f"Groq request failed: {e}") from e
        if resp.status_code != 200:
            raise LLMClientError(
                f"Groq returned HTTP {resp.status_code}: {resp.text[:400]}"
            )
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMClientError(f"Groq response parse error: {e}") from e

    def health_check(self) -> dict[str, Any]:
        base = {"ok": False, "backend": self.name, "model": self._model}
        try:
            # Cheap probe: hit the models endpoint
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            base["ok"] = resp.status_code == 200
            if not base["ok"]:
                base["error"] = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            base["error"] = str(e)
        return base


# ---------------------------------------------------------------------------
# Google Gemini backend (free tier via AI Studio)
# ---------------------------------------------------------------------------

class GeminiBackend(LLMBackend):
    """
    Google Gemini via the AI Studio REST endpoint.

    Signup: https://aistudio.google.com/apikey
    Auth: API key as `key=` query parameter (Google's standard).
    Default model: gemini-2.0-flash.
    """

    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMClientError(
                "GeminiBackend needs GEMINI_API_KEY (env var or constructor arg)."
            )
        self._model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        url = f"{self.BASE}/{self._model}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise LLMClientError(f"Gemini request failed: {e}") from e
        if resp.status_code != 200:
            raise LLMClientError(
                f"Gemini returned HTTP {resp.status_code}: {resp.text[:400]}"
            )
        try:
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, ValueError) as e:
            raise LLMClientError(f"Gemini response parse error: {e}") from e

    def health_check(self) -> dict[str, Any]:
        base = {"ok": False, "backend": self.name, "model": self._model}
        try:
            url = f"{self.BASE}?key={self.api_key}"
            resp = requests.get(url, timeout=10)
            base["ok"] = resp.status_code == 200
            if not base["ok"]:
                base["error"] = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            base["error"] = str(e)
        return base


# ---------------------------------------------------------------------------
# Ollama backend (local, no API key)
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """
    Local Ollama daemon.

    Setup:
        1. Install Ollama: https://ollama.com/download
        2. Pull a model: `ollama pull llama3.2`
        3. Ollama serves on http://localhost:11434 by default.

    No API key. Great fallback when no free API is reachable or when
    privacy requires all inference to stay on the user's machine.
    """

    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str = "llama3.2",
        timeout: int = 120,   # local generation can be slower
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.host = (host or os.environ.get("OLLAMA_HOST")
                     or "http://localhost:11434").rstrip("/")
        self._model = model
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self._model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMClientError(f"Ollama request failed: {e}") from e
        if resp.status_code != 200:
            raise LLMClientError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:400]}"
            )
        try:
            data = resp.json()
            return data["response"]
        except (KeyError, ValueError) as e:
            raise LLMClientError(f"Ollama response parse error: {e}") from e

    def health_check(self) -> dict[str, Any]:
        base = {"ok": False, "backend": self.name, "model": self._model,
                "host": self.host}
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            base["ok"] = resp.status_code == 200
            if not base["ok"]:
                base["error"] = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            base["error"] = str(e)
        return base


# ---------------------------------------------------------------------------
# Default backend selection
# ---------------------------------------------------------------------------

def make_default_backend() -> LLMBackend:
    """
    Auto-select a backend from environment variables. Priority:
        1. TRUSTGATE_LLM_BACKEND (explicit override)
        2. GROQ_API_KEY -> Groq
        3. GEMINI_API_KEY -> Gemini
        4. OLLAMA_HOST -> Ollama at that host
        5. Fallback: Mock (never fails, useful for tests / offline demo)
    """
    forced = os.environ.get("TRUSTGATE_LLM_BACKEND", "").strip().lower()
    if forced == "groq":
        return GroqBackend()
    if forced == "gemini":
        return GeminiBackend()
    if forced == "ollama":
        return OllamaBackend()
    if forced == "mock":
        return MockBackend()

    if os.environ.get("GROQ_API_KEY"):
        return GroqBackend()
    if os.environ.get("GEMINI_API_KEY"):
        return GeminiBackend()
    if os.environ.get("OLLAMA_HOST"):
        return OllamaBackend()
    return MockBackend()


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper the rest of TrustGate imports. Uses whichever backend
    was selected by `make_default_backend()`, or a backend passed in
    explicitly. Timing and metadata are recorded so the FastAPI backend
    can log them into SQLite for the monitor page.
    """

    def __init__(self, backend: LLMBackend | None = None):
        self.backend: LLMBackend = backend or make_default_backend()
        self.last_call_info: LLMCallInfo | None = None

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @property
    def model_name(self) -> str:
        return self.backend.model

    def complete(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise LLMClientError("prompt must be a string.")
        if not prompt.strip():
            raise LLMClientError("prompt is empty.")
        start = time.time()
        try:
            text = self.backend.complete(prompt)
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
        if not isinstance(text, str):
            raise LLMClientError(
                f"Backend {self.backend_name} returned non-string response."
            )
        self.last_call_info = LLMCallInfo(
            backend=self.backend_name,
            model=self.model_name,
            latency_ms=elapsed_ms,
            prompt_length_chars=len(prompt),
            response_length_chars=len(text),
        )
        # Strip any markdown the LLM sneaked past the system prompt
        text = clean_response(text)
        return text

    def health_check(self) -> dict[str, Any]:
        try:
            return self.backend.health_check()
        except Exception as e:  # noqa: BLE001 - health_check must not raise
            return {"ok": False, "backend": self.backend_name,
                    "error": f"{type(e).__name__}: {e}"}
