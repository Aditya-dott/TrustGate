"""
Step 5 comprehensive validation for the LLM client.

Sections:
    1. LLMClientError type and helpers
    2. MockBackend: deterministic response, health check
    3. LLMClient wrapper: uses provided backend, records LLMCallInfo,
       rejects empty / non-string prompts
    4. GroqBackend: init requires API key, success path with mocked
       requests.post, HTTP-error path, network-error path, malformed
       JSON path, health check path
    5. GeminiBackend: same coverage as Groq
    6. OllamaBackend: same coverage
    7. make_default_backend(): env-var priority, explicit override,
       fallback to MockBackend
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.client import (  # noqa: E402
    DEFAULT_MAX_TOKENS, GeminiBackend, GroqBackend, LLMBackend,
    LLMCallInfo, LLMClient, LLMClientError, MockBackend, OllamaBackend,
    make_default_backend,
)

import requests  # noqa: E402


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


def clean_env(prefixes: tuple[str, ...] = ("GROQ_", "GEMINI_", "OLLAMA_", "TRUSTGATE_")) -> dict:
    """Return an env dict with all matching prefixes removed."""
    env = dict(os.environ)
    for key in list(env.keys()):
        for pref in prefixes:
            if key.startswith(pref):
                del env[key]
                break
    return env


# ---------------------------------------------------------------------------
# Section 1: error type
# ---------------------------------------------------------------------------

def test_error_type(t: TestRunner) -> None:
    print("\n[Section 1] error type")
    t.check(issubclass(LLMClientError, Exception), "LLMClientError is an Exception")
    err = LLMClientError("test")
    t.check(str(err) == "test", "LLMClientError carries message")


# ---------------------------------------------------------------------------
# Section 2: MockBackend
# ---------------------------------------------------------------------------

def test_mock_backend(t: TestRunner) -> None:
    print("\n[Section 2] MockBackend")
    m = MockBackend()
    r1 = m.complete("hello")
    r2 = m.complete("something completely different")
    t.check(r1 == r2, "MockBackend returns deterministic response")
    t.check(isinstance(r1, str) and len(r1) > 0, "MockBackend returns non-empty string")
    t.check("support@" in r1 and "+91" in r1,
            "default mock includes PII markers for Privacy Sieve to catch")

    m2 = MockBackend(response="Custom response.", model="mock-custom")
    t.check(m2.complete("x") == "Custom response.", "custom response works")
    t.check(m2.model == "mock-custom", "custom model name propagates")

    h = m.health_check()
    t.check(h["ok"] is True, "MockBackend health_check returns ok=True")
    t.check(h["backend"] == "mock", "health_check reports backend name")


# ---------------------------------------------------------------------------
# Section 3: LLMClient wrapper
# ---------------------------------------------------------------------------

def test_client_wrapper(t: TestRunner) -> None:
    print("\n[Section 3] LLMClient wrapper")
    m = MockBackend(response="Hello world.", model="mock-v1")
    client = LLMClient(backend=m)

    t.check(client.backend_name == "mock", "backend_name = mock")
    t.check(client.model_name == "mock-v1", "model_name propagates")
    t.check(client.last_call_info is None, "no call info before first complete")

    text = client.complete("How do I make paneer?")
    t.check(text == "Hello world.", "complete returns backend response")

    info = client.last_call_info
    t.check(info is not None, "last_call_info populated after complete")
    t.check(isinstance(info, LLMCallInfo), "last_call_info is LLMCallInfo")
    t.check(info.backend == "mock", "info.backend = mock")
    t.check(info.model == "mock-v1", "info.model correct")
    t.check(info.prompt_length_chars == len("How do I make paneer?"),
            "prompt length recorded")
    t.check(info.response_length_chars == len("Hello world."),
            "response length recorded")
    t.check(info.latency_ms >= 0, "latency is non-negative")

    # Rejects empty / whitespace prompt
    for bad in ["", "   ", "\n\n"]:
        try:
            client.complete(bad)
            t.check(False, f"reject empty prompt: {bad!r}")
        except LLMClientError:
            t.check(True, f"empty prompt rejected: {bad!r}")

    # Rejects non-string prompt
    try:
        client.complete(123)  # type: ignore[arg-type]
        t.check(False, "reject non-string prompt")
    except LLMClientError:
        t.check(True, "non-string prompt rejected")

    # health_check goes through
    h = client.health_check()
    t.check(h["ok"] is True, "client.health_check() passes through to backend")


# ---------------------------------------------------------------------------
# Section 4: GroqBackend
# ---------------------------------------------------------------------------

def test_groq_backend(t: TestRunner) -> None:
    print("\n[Section 4] GroqBackend")

    # Missing API key -> LLMClientError
    with patch.dict(os.environ, clean_env(), clear=True):
        try:
            GroqBackend()
            t.check(False, "missing GROQ_API_KEY should raise")
        except LLMClientError:
            t.check(True, "missing GROQ_API_KEY raises LLMClientError")

    # Explicit api_key overrides env
    g = GroqBackend(api_key="test-key")
    t.check(g.api_key == "test-key", "explicit api_key wins")
    t.check(g.name == "groq", "name = groq")
    t.check(g.model == "openai/gpt-oss-20b", "default model")

    # Success path with mocked HTTP
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": "Hi from Groq!"}}]
    }
    with patch("llm.client.requests.post", return_value=fake_resp) as mock_post:
        out = g.complete("Say hi.")
        t.check(out == "Hi from Groq!", f"success path returns content (got {out!r})")
        # Verify request shape
        call = mock_post.call_args
        t.check(call.args[0] == GroqBackend.ENDPOINT, "hits the right endpoint")
        headers = call.kwargs["headers"]
        t.check(headers["Authorization"] == "Bearer test-key",
                "sends bearer token")
        payload = call.kwargs["json"]
        t.check(payload["model"] == "openai/gpt-oss-20b", "model in payload")
        t.check(payload["messages"][0]["role"] == "system",
                "system prompt included in messages")
        t.check(payload["messages"][1] == {"role": "user", "content": "Say hi."},
                "prompt in messages array")

    # Non-200 status
    bad_resp = MagicMock()
    bad_resp.status_code = 429
    bad_resp.text = "rate limit"
    with patch("llm.client.requests.post", return_value=bad_resp):
        try:
            g.complete("hi")
            t.check(False, "HTTP 429 should raise")
        except LLMClientError as e:
            t.check("429" in str(e), f"error mentions HTTP 429 (got {e})")

    # Network exception
    with patch("llm.client.requests.post",
               side_effect=requests.ConnectionError("network down")):
        try:
            g.complete("hi")
            t.check(False, "network error should raise")
        except LLMClientError as e:
            t.check("network down" in str(e), "error wraps requests exception")

    # Malformed JSON (missing choices)
    weird_resp = MagicMock()
    weird_resp.status_code = 200
    weird_resp.json.return_value = {"unexpected": "shape"}
    with patch("llm.client.requests.post", return_value=weird_resp):
        try:
            g.complete("hi")
            t.check(False, "malformed response should raise")
        except LLMClientError:
            t.check(True, "malformed response raises LLMClientError")

    # Health check success
    ok_resp = MagicMock(); ok_resp.status_code = 200
    with patch("llm.client.requests.get", return_value=ok_resp):
        h = g.health_check()
        t.check(h["ok"] is True, "health_check returns ok=True on 200")
        t.check(h["backend"] == "groq", "health_check reports backend")

    # Health check failure
    with patch("llm.client.requests.get",
               side_effect=requests.ConnectionError("nope")):
        h = g.health_check()
        t.check(h["ok"] is False, "health_check returns ok=False on network error")
        t.check("nope" in h["error"], "health_check reports error message")


# ---------------------------------------------------------------------------
# Section 5: GeminiBackend
# ---------------------------------------------------------------------------

def test_gemini_backend(t: TestRunner) -> None:
    print("\n[Section 5] GeminiBackend")

    with patch.dict(os.environ, clean_env(), clear=True):
        try:
            GeminiBackend()
            t.check(False, "missing GEMINI_API_KEY should raise")
        except LLMClientError:
            t.check(True, "missing GEMINI_API_KEY raises LLMClientError")

    g = GeminiBackend(api_key="test-key")
    t.check(g.name == "gemini", "name = gemini")
    t.check(g.model == "gemini-2.0-flash", "default model")

    # Success
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "Hi from "}, {"text": "Gemini!"}]}}
        ]
    }
    with patch("llm.client.requests.post", return_value=fake) as mock_post:
        out = g.complete("Say hi.")
        t.check(out == "Hi from Gemini!", "concatenates text parts")
        url = mock_post.call_args.args[0]
        t.check("key=test-key" in url, "sends API key as query param")
        t.check("gemini-2.0-flash" in url, "model in URL")
        payload = mock_post.call_args.kwargs["json"]
        t.check(payload["contents"][0]["parts"][0]["text"] == "Say hi.",
                "prompt in contents payload")

    # HTTP error
    bad = MagicMock(); bad.status_code = 400; bad.text = "bad"
    with patch("llm.client.requests.post", return_value=bad):
        try:
            g.complete("hi")
            t.check(False, "HTTP 400 should raise")
        except LLMClientError as e:
            t.check("400" in str(e), "error mentions HTTP 400")

    # Malformed
    weird = MagicMock(); weird.status_code = 200
    weird.json.return_value = {}
    with patch("llm.client.requests.post", return_value=weird):
        try:
            g.complete("hi")
            t.check(False, "malformed should raise")
        except LLMClientError:
            t.check(True, "malformed response raises")


# ---------------------------------------------------------------------------
# Section 6: OllamaBackend
# ---------------------------------------------------------------------------

def test_ollama_backend(t: TestRunner) -> None:
    print("\n[Section 6] OllamaBackend")

    o = OllamaBackend()
    t.check(o.name == "ollama", "name = ollama")
    t.check(o.host == "http://localhost:11434", "default host")

    o2 = OllamaBackend(host="http://otherhost:9999", model="mistral")
    t.check(o2.host == "http://otherhost:9999", "custom host")
    t.check(o2.model == "mistral", "custom model")

    # Success
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {"response": "Hi from Ollama!"}
    with patch("llm.client.requests.post", return_value=fake) as mock_post:
        out = o.complete("Say hi.")
        t.check(out == "Hi from Ollama!", "returns response text")
        payload = mock_post.call_args.kwargs["json"]
        t.check(payload["model"] == "llama3.2", "model in payload")
        t.check(payload["prompt"] == "Say hi.", "prompt in payload")
        t.check(payload["stream"] is False, "stream=False")

    # HTTP error
    bad = MagicMock(); bad.status_code = 500; bad.text = "boom"
    with patch("llm.client.requests.post", return_value=bad):
        try:
            o.complete("hi")
            t.check(False, "HTTP 500 should raise")
        except LLMClientError as e:
            t.check("500" in str(e), "error mentions HTTP 500")

    # Not running
    with patch("llm.client.requests.post",
               side_effect=requests.ConnectionError("connection refused")):
        try:
            o.complete("hi")
            t.check(False, "connection error should raise")
        except LLMClientError as e:
            t.check("connection refused" in str(e), "wraps connection error")


# ---------------------------------------------------------------------------
# Section 7: make_default_backend
# ---------------------------------------------------------------------------

def test_factory(t: TestRunner) -> None:
    print("\n[Section 7] make_default_backend factory")

    # No env vars -> Mock
    with patch.dict(os.environ, clean_env(), clear=True):
        b = make_default_backend()
        t.check(isinstance(b, MockBackend),
                f"no env vars -> MockBackend (got {type(b).__name__})")

    # GROQ_API_KEY set -> Groq
    with patch.dict(os.environ, {**clean_env(), "GROQ_API_KEY": "x"}, clear=True):
        b = make_default_backend()
        t.check(isinstance(b, GroqBackend), "GROQ_API_KEY -> GroqBackend")

    # GEMINI_API_KEY set -> Gemini
    with patch.dict(os.environ, {**clean_env(), "GEMINI_API_KEY": "x"}, clear=True):
        b = make_default_backend()
        t.check(isinstance(b, GeminiBackend), "GEMINI_API_KEY -> GeminiBackend")

    # OLLAMA_HOST set -> Ollama
    with patch.dict(os.environ,
                    {**clean_env(), "OLLAMA_HOST": "http://custom:1234"},
                    clear=True):
        b = make_default_backend()
        t.check(isinstance(b, OllamaBackend), "OLLAMA_HOST -> OllamaBackend")
        t.check(b.host == "http://custom:1234", "host propagates from env")

    # Priority: TRUSTGATE_LLM_BACKEND wins over API keys
    with patch.dict(os.environ,
                    {**clean_env(), "GROQ_API_KEY": "x",
                     "TRUSTGATE_LLM_BACKEND": "mock"},
                    clear=True):
        b = make_default_backend()
        t.check(isinstance(b, MockBackend),
                "TRUSTGATE_LLM_BACKEND=mock overrides GROQ_API_KEY")

    # Groq preferred over Gemini when both set
    with patch.dict(os.environ,
                    {**clean_env(), "GROQ_API_KEY": "x", "GEMINI_API_KEY": "y"},
                    clear=True):
        b = make_default_backend()
        t.check(isinstance(b, GroqBackend),
                "Groq wins when both GROQ_API_KEY and GEMINI_API_KEY set")


# ---------------------------------------------------------------------------
def main() -> int:
    t = TestRunner()
    test_error_type(t)
    test_mock_backend(t)
    test_client_wrapper(t)
    test_groq_backend(t)
    test_gemini_backend(t)
    test_ollama_backend(t)
    test_factory(t)
    return t.summary()


if __name__ == "__main__":
    sys.exit(main())
