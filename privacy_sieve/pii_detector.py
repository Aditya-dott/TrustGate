"""
Privacy Sieve - PII detection and masking.

Runs on every LLM response after generation, before it reaches the user.
Detects personally identifiable information and replaces each occurrence
with a labeled placeholder ([EMAIL REDACTED], [PHONE REDACTED], etc.)
that keeps the surrounding sentence coherent.

Two-track detection:

    1. Regex patterns for STRUCTURED PII whose formats are well-defined:
         - EMAIL          RFC-flavored address
         - PHONE          international / Indian / US formats (post-
                          filtered to require 10-15 digits)
         - CREDIT_CARD    13-19 digit sequences, Luhn-validated to cut
                          false positives
         - AADHAAR        12 Indian digits, first digit 2-9 per the UIDAI
                          rule (so "1234 5678 9012" doesn't match)
         - PASSPORT       Indian passport: 1 letter + 7 digits
         - IP_ADDRESS     IPv4 with 0-255 octet validation
         - DATE_OF_BIRTH  any date whose year falls in 1900-2020
         - SENSITIVE_URL  URLs whose query string carries a token /
                          api_key / password / secret / etc. parameter

    2. spaCy NER for UNSTRUCTURED PII whose form varies too much for
       regex to be reliable:
         - PERSON         detected via spaCy's PERSON entity, filtered
                          to multi-token spans to reduce false positives
                          on common words
         - ADDRESS        detected via spaCy's GPE / LOC / FAC entities

Overlap resolution:
    Regex matches and NER matches routinely overlap ("john@x.com"
    contains "john" which NER may tag as PERSON). Every match is
    tagged with a priority and the higher-priority match wins, so the
    email stays intact rather than being nested inside a name redaction.

Replacements are applied from the end of the string back to the
start so the character offsets we recorded stay valid. The report
gives every entity's offset in the ORIGINAL text (not the cleaned
one) so the monitor can visualise where each redaction happened.

If spaCy or its model is not installed, the module falls back to
regex-only detection and prints a one-line warning. That keeps the
pipeline running for the structured categories even in stripped-down
environments.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Categories, priorities, replacement labels
# ---------------------------------------------------------------------------

CATEGORIES: tuple[str, ...] = (
    "CREDIT_CARD", "AADHAAR", "PASSPORT",
    "EMAIL", "PHONE", "IP_ADDRESS", "SENSITIVE_URL",
    "DATE_OF_BIRTH", "PERSON", "ADDRESS",
)

# Higher wins when spans overlap. Structured / high-precision detectors
# outrank NER because regex matches are exact character spans while NER
# spans are approximations.
PRIORITY: dict[str, int] = {
    "CREDIT_CARD": 100,
    "AADHAAR": 95,
    "PASSPORT": 90,
    "SENSITIVE_URL": 85,
    "EMAIL": 80,
    "PHONE": 75,
    "IP_ADDRESS": 70,
    "DATE_OF_BIRTH": 55,
    "PERSON": 40,
    "ADDRESS": 30,
}

REPLACEMENTS: dict[str, str] = {
    "CREDIT_CARD":   "[CREDIT CARD REDACTED]",
    "AADHAAR":       "[AADHAAR REDACTED]",
    "PASSPORT":      "[PASSPORT REDACTED]",
    "EMAIL":         "[EMAIL REDACTED]",
    "PHONE":         "[PHONE REDACTED]",
    "IP_ADDRESS":    "[IP REDACTED]",
    "SENSITIVE_URL": "[URL REDACTED]",
    "DATE_OF_BIRTH": "[DATE OF BIRTH REDACTED]",
    "PERSON":        "[NAME REDACTED]",
    "ADDRESS":       "[ADDRESS REDACTED]",
}


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Broad phone matcher; results are post-filtered to require 10-15 digits.
PHONE_RE = re.compile(
    r"(?:(?<![\d.])"                    # not immediately after a digit or dot
    r"(?:\+\d{1,3}[-.\s]?)?"            # optional country code (+91, +1, ...)
    r"(?:\(\d{2,5}\)[-.\s]?)?"          # optional area code (022)
    r"\d{2,5}"                          # first block (2-5 digits: UK 20, Indian 987)
    r"(?:[-.\s]?\d{2,5}){1,4}"          # 1-4 more blocks
    r"(?!\d))"
)

# 13-19 digit sequences with optional 4-digit grouping via space or dash.
CREDIT_CARD_RE = re.compile(r"\b(?:\d[\-\s]?){12,18}\d\b")

# 12-digit Aadhaar; UIDAI reserves first digit 2-9.
AADHAAR_RE = re.compile(r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b")

# Indian passport: one letter (A-Z minus Q, X, Z which aren't used) + 7 digits.
PASSPORT_RE = re.compile(r"\b[A-PR-WY][0-9]{7}\b")

# IPv4 with per-octet 0-255 validation.
IPV4_RE = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|[01]?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3}\b"
)

# Common date shapes: DD/MM/YYYY, MM-DD-YY, YYYY/MM/DD, etc. Post-filtered
# for a plausible birth year (1900-2020).
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/.\s]\d{1,2}[-/.\s]\d{2,4}"
    r"|\d{4}[-/.\s]\d{1,2}[-/.\s]\d{1,2})\b"
)

# URLs whose query string carries a sensitive-looking parameter.
SENSITIVE_URL_RE = re.compile(
    r"https?://\S+?[?&]"
    r"(?:token|apikey|api[-_]?key|password|passwd|pwd|secret|"
    r"access[-_]?token|session|sessionid|session[-_]?id|"
    r"auth|authorization|credential)="
    r"\S+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def luhn_valid(number_str: str) -> bool:
    """Standard Luhn checksum; returns True for a well-formed card number."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


BIRTH_YEAR_RE = re.compile(r"\b(19\d{2}|20[01]\d|2020)\b")


def looks_like_dob(text: str) -> bool:
    """Heuristic: date whose 4-digit year sits in [1900, 2020]."""
    return BIRTH_YEAR_RE.search(text) is not None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIEntity:
    """One detected PII occurrence."""
    entity_type: str
    original_text: str
    start: int          # character offset in the ORIGINAL text
    end: int            # exclusive
    replacement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacySieveReport:
    """
    Structured report the FastAPI backend stores and the monitor page
    displays. Contains everything about what was found and what the
    cleaned text looks like.
    """
    entities: list[PIIEntity]
    cleaned_text: str
    pii_found: bool
    original_length: int
    cleaned_length: int
    entity_type_counts: dict[str, int] = field(default_factory=dict)
    ner_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "cleaned_text": self.cleaned_text,
            "pii_found": self.pii_found,
            "original_length": self.original_length,
            "cleaned_length": self.cleaned_length,
            "entity_type_counts": self.entity_type_counts,
            "ner_enabled": self.ner_enabled,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PrivacySieve:
    """
    Scans a piece of text (typically an LLM response) and produces a
    structured report of PII entities found plus a cleaned version of
    the text with each entity replaced by a labeled placeholder.

    Call `scan(text)` per response. The instance is safe to reuse across
    calls; it holds only the loaded spaCy pipeline as state.
    """

    def __init__(
        self,
        use_ner: bool = True,
        ner_model_name: str = "en_core_web_sm",
        min_person_tokens: int = 2,
    ):
        """
        Args:
            use_ner: if False, skip NER entirely and rely on regex only.
            ner_model_name: spaCy model to load. Default is the small
                English model which is fast and lightweight (~13MB).
            min_person_tokens: PERSON entities with fewer tokens than
                this are dropped, since single capitalized words are
                often common nouns (e.g., "January", "Monday", "God").
        """
        self._nlp = None
        self.ner_enabled = False
        self.min_person_tokens = max(1, min_person_tokens)

        if use_ner:
            try:
                import spacy
                self._nlp = spacy.load(ner_model_name)
                self.ner_enabled = True
            except Exception as e:  # noqa: BLE001
                print(
                    f"[privacy_sieve] NER unavailable "
                    f"({type(e).__name__}: {e}); using regex-only detection. "
                    f"Install spaCy and run "
                    f"`python -m spacy download {ner_model_name}` to enable "
                    f"name and address detection."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scan(self, text: str) -> PrivacySieveReport:
        text = text if isinstance(text, str) else ""
        raw: list[PIIEntity] = []
        raw.extend(self._detect_regex(text))
        if self._nlp is not None:
            raw.extend(self._detect_ner(text))

        entities = self._resolve_overlaps(raw)
        entities.sort(key=lambda e: e.start)

        cleaned = self._apply_replacements(text, entities)

        counts: dict[str, int] = {}
        for e in entities:
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1

        return PrivacySieveReport(
            entities=entities,
            cleaned_text=cleaned,
            pii_found=bool(entities),
            original_length=len(text),
            cleaned_length=len(cleaned),
            entity_type_counts=counts,
            ner_enabled=self.ner_enabled,
        )

    # ------------------------------------------------------------------
    # Regex detectors
    # ------------------------------------------------------------------
    def _detect_regex(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []

        for m in EMAIL_RE.finditer(text):
            found.append(self._make("EMAIL", m.group(), m.start(), m.end()))

        for m in CREDIT_CARD_RE.finditer(text):
            if luhn_valid(m.group()):
                found.append(self._make("CREDIT_CARD", m.group(), m.start(), m.end()))

        for m in AADHAAR_RE.finditer(text):
            found.append(self._make("AADHAAR", m.group(), m.start(), m.end()))

        for m in PASSPORT_RE.finditer(text):
            found.append(self._make("PASSPORT", m.group(), m.start(), m.end()))

        for m in PHONE_RE.finditer(text):
            phone_text = m.group()
            digit_count = sum(1 for c in phone_text if c.isdigit())
            if 10 <= digit_count <= 15:
                found.append(self._make("PHONE", phone_text, m.start(), m.end()))

        for m in IPV4_RE.finditer(text):
            found.append(self._make("IP_ADDRESS", m.group(), m.start(), m.end()))

        for m in DATE_RE.finditer(text):
            if looks_like_dob(m.group()):
                found.append(self._make("DATE_OF_BIRTH", m.group(), m.start(), m.end()))

        for m in SENSITIVE_URL_RE.finditer(text):
            found.append(self._make("SENSITIVE_URL", m.group(), m.start(), m.end()))

        return found

    # ------------------------------------------------------------------
    # NER detector
    # ------------------------------------------------------------------
    def _detect_ner(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []
        doc = self._nlp(text)  # type: ignore[misc]
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                token_count = len(ent.text.split())
                if token_count < self.min_person_tokens:
                    continue
                found.append(self._make("PERSON", ent.text,
                                        ent.start_char, ent.end_char))
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                found.append(self._make("ADDRESS", ent.text,
                                        ent.start_char, ent.end_char))
        return found

    # ------------------------------------------------------------------
    # Overlap resolution
    # ------------------------------------------------------------------
    def _resolve_overlaps(self, matches: list[PIIEntity]) -> list[PIIEntity]:
        """
        Higher-priority match wins over any span it overlaps with.
        Ties are broken by earlier start, then longer span (so we prefer
        the fuller detection over a truncated one).
        """
        if not matches:
            return []
        matches_sorted = sorted(
            matches,
            key=lambda e: (-PRIORITY.get(e.entity_type, 0), e.start,
                           -(e.end - e.start)),
        )
        kept: list[PIIEntity] = []
        for m in matches_sorted:
            if any(self._overlaps(m, k) for k in kept):
                continue
            kept.append(m)
        return kept

    @staticmethod
    def _overlaps(a: PIIEntity, b: PIIEntity) -> bool:
        return not (a.end <= b.start or b.end <= a.start)

    # ------------------------------------------------------------------
    # Replacement
    # ------------------------------------------------------------------
    def _apply_replacements(
        self, text: str, entities: list[PIIEntity]
    ) -> str:
        """
        Replace each entity's span with its labeled placeholder. We
        walk from the END of the string back to the START so earlier
        character offsets stay valid as we splice.
        """
        if not entities:
            return text
        result = text
        for e in sorted(entities, key=lambda x: -x.start):
            result = result[:e.start] + e.replacement + result[e.end:]
        return result

    # ------------------------------------------------------------------
    def _make(
        self, kind: str, original: str, start: int, end: int
    ) -> PIIEntity:
        return PIIEntity(
            entity_type=kind,
            original_text=original,
            start=start,
            end=end,
            replacement=REPLACEMENTS[kind],
        )
