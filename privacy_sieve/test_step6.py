"""
Step 6 comprehensive validation for the Privacy Sieve.

Sections:
    1. Utility helpers: luhn_valid, looks_like_dob
    2. Email detection (single, multiple, edge cases)
    3. Phone detection (Indian +91, plain 10-digit, US, international)
    4. Credit card (valid Luhn, invalid Luhn, formatted with dashes/spaces)
    5. Aadhaar (first-digit rule)
    6. Passport (Indian format)
    7. IP address (valid IPv4, invalid octets)
    8. Sensitive URL (with/without sensitive params)
    9. Date of birth (year range filter)
    10. NER: person names (if spaCy available; else skipped)
    11. NER: addresses
    12. Overlap resolution: high-priority wins
    13. Position accuracy: offsets point to original text
    14. Replacement produces coherent readable text
    15. Empty text and no-PII cases
    16. Multi-PII response with 5+ entity types at once
    17. End-to-end: the Mock LLM response gets scrubbed correctly
    18. Report is JSON-serializable
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_sieve.pii_detector import (  # noqa: E402
    PIIEntity, PrivacySieve, PrivacySieveReport,
    REPLACEMENTS, luhn_valid, looks_like_dob,
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


# Reusable helpers ---------------------------------------------------------

def types_in(report: PrivacySieveReport) -> set[str]:
    return {e.entity_type for e in report.entities}


def entity_of(report: PrivacySieveReport, kind: str) -> PIIEntity | None:
    for e in report.entities:
        if e.entity_type == kind:
            return e
    return None


# ==========================================================================
# Section 1: utilities
# ==========================================================================

def test_utilities(t: TestRunner) -> None:
    print("\n[Section 1] utilities")
    # Luhn - known valid card numbers
    t.check(luhn_valid("4242424242424242"), "Stripe test Visa passes Luhn")
    t.check(luhn_valid("5555555555554444"), "Stripe test Mastercard passes Luhn")
    t.check(luhn_valid("378282246310005"), "AmEx test passes Luhn (15 digits)")
    t.check(not luhn_valid("1234567890123456"), "sequential digits fail Luhn")
    t.check(not luhn_valid("1234"), "too short fails Luhn")
    t.check(not luhn_valid("1" * 25), "too long fails Luhn")
    # Luhn allows separators
    t.check(luhn_valid("4242 4242 4242 4242"), "Luhn accepts spaces")
    t.check(luhn_valid("4242-4242-4242-4242"), "Luhn accepts dashes")

    # DoB heuristic
    t.check(looks_like_dob("15/08/1990"), "1990 date looks like DoB")
    t.check(looks_like_dob("2000-01-01"), "2000 date looks like DoB")
    t.check(not looks_like_dob("15/08/2050"), "2050 date does not")
    t.check(not looks_like_dob("15-Jan"), "no 4-digit year -> no DoB")


# ==========================================================================
# Section 2: email
# ==========================================================================

def test_email(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 2] email")

    r = sieve.scan("Contact us at support@example.com for help.")
    t.check("EMAIL" in types_in(r), "simple email detected")
    e = entity_of(r, "EMAIL")
    t.check(e is not None and e.original_text == "support@example.com",
            f"email text = 'support@example.com' (got {e.original_text if e else None})")
    t.check("[EMAIL REDACTED]" in r.cleaned_text, "cleaned text has redacted marker")
    t.check("support@example.com" not in r.cleaned_text, "original email removed")

    # Multiple emails
    r = sieve.scan("Email a@b.co or c.d@e-f.org for questions.")
    emails = [e for e in r.entities if e.entity_type == "EMAIL"]
    t.check(len(emails) == 2, f"two emails detected (got {len(emails)})")

    # Email with plus sign and dots
    r = sieve.scan("Try alice.bob+tag@example.co.uk please.")
    e = entity_of(r, "EMAIL")
    t.check(e is not None and "alice.bob+tag@example.co.uk" == e.original_text,
            "email with +tag and multi-part domain detected")


# ==========================================================================
# Section 3: phone
# ==========================================================================

def test_phone(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 3] phone")
    # Indian +91
    r = sieve.scan("Call us at +91-9876543210 anytime.")
    t.check("PHONE" in types_in(r), "+91-9876543210 detected")

    # Plain 10-digit Indian mobile
    r = sieve.scan("My number is 9876543210, save it.")
    t.check("PHONE" in types_in(r), "plain 10-digit Indian mobile detected")

    # US format
    r = sieve.scan("Reach him at (415) 555-0132 for details.")
    t.check("PHONE" in types_in(r), "US (XXX) XXX-XXXX detected")

    # International
    r = sieve.scan("From UK: +44 20 7946 0958 works too.")
    t.check("PHONE" in types_in(r), "UK international number detected")

    # Short number should NOT match (avoids matching "call 100")
    r = sieve.scan("The service is 100.")
    t.check("PHONE" not in types_in(r), "'100' alone is not a phone number")

    # A large round number that isn't a phone
    r = sieve.scan("The 2026 revenue was 12345.")
    t.check("PHONE" not in types_in(r), "short numbers are not phones")


# ==========================================================================
# Section 4: credit card
# ==========================================================================

def test_credit_card(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 4] credit card")
    r = sieve.scan("Card number: 4242 4242 4242 4242 for the demo.")
    t.check("CREDIT_CARD" in types_in(r), "Luhn-valid Visa detected")

    r = sieve.scan("Card 4242-4242-4242-4242 is on file.")
    t.check("CREDIT_CARD" in types_in(r), "dash-formatted card detected")

    r = sieve.scan("The number 1234567890123456 is fake.")
    # Non-Luhn 16 digits should not be flagged as credit card
    t.check("CREDIT_CARD" not in types_in(r),
            "invalid Luhn sequence not flagged as credit card")


# ==========================================================================
# Section 5: Aadhaar
# ==========================================================================

def test_aadhaar(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 5] Aadhaar")
    r = sieve.scan("My Aadhaar is 2345 6789 0123.")
    t.check("AADHAAR" in types_in(r), "Aadhaar with spaces detected")

    r = sieve.scan("Aadhaar: 234567890123.")
    t.check("AADHAAR" in types_in(r), "Aadhaar without separators detected")

    r = sieve.scan("Invalid Aadhaar 1234 5678 9012 starts with 1.")
    t.check("AADHAAR" not in types_in(r),
            "Aadhaar starting with 1 rejected (UIDAI rule)")


# ==========================================================================
# Section 6: passport
# ==========================================================================

def test_passport(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 6] passport")
    r = sieve.scan("Passport K1234567 was issued last year.")
    t.check("PASSPORT" in types_in(r), "Indian passport K1234567 detected")

    r = sieve.scan("The code AB123456 is a product code.")
    t.check("PASSPORT" not in types_in(r), "two-letter prefix is not a passport")


# ==========================================================================
# Section 7: IP address
# ==========================================================================

def test_ip(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 7] IP address")
    r = sieve.scan("Server at 192.168.1.100 is down.")
    t.check("IP_ADDRESS" in types_in(r), "IPv4 192.168.1.100 detected")

    r = sieve.scan("The number 999.999.999.999 is not an IP.")
    t.check("IP_ADDRESS" not in types_in(r), "invalid octets not detected as IP")


# ==========================================================================
# Section 8: sensitive URL
# ==========================================================================

def test_sensitive_url(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 8] sensitive URL")

    r = sieve.scan("Login: https://api.example.com/oauth?access_token=abc123xyz")
    t.check("SENSITIVE_URL" in types_in(r), "URL with access_token= detected")

    r = sieve.scan("Reset: https://x.com/reset?token=deadbeef more text")
    t.check("SENSITIVE_URL" in types_in(r), "URL with token= detected")

    r = sieve.scan("Please read https://example.com/docs/index.html carefully")
    t.check("SENSITIVE_URL" not in types_in(r),
            "URL without sensitive params NOT flagged")


# ==========================================================================
# Section 9: date of birth
# ==========================================================================

def test_dob(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 9] date of birth")
    r = sieve.scan("Born on 15/08/1990 in Mumbai.")
    t.check("DATE_OF_BIRTH" in types_in(r), "1990 date detected as DoB")

    r = sieve.scan("The event is on 15/08/2030.")
    t.check("DATE_OF_BIRTH" not in types_in(r),
            "future date not flagged as DoB")


# ==========================================================================
# Section 10 + 11: NER (skipped gracefully if spaCy unavailable)
# ==========================================================================

def test_ner(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Sections 10-11] NER: names and addresses")
    if not sieve.ner_enabled:
        print("  (spaCy not enabled; NER tests skipped)")
        return

    r = sieve.scan("John Smith visited Delhi last week.")
    t.check("PERSON" in types_in(r), "PERSON 'John Smith' detected via NER")
    t.check("ADDRESS" in types_in(r), "ADDRESS 'Delhi' detected via NER")

    # Single-word "John" (below min_person_tokens=2) shouldn't match by default
    r = sieve.scan("John left yesterday.")
    t.check("PERSON" not in types_in(r),
            "single-token PERSON filtered by min_person_tokens")


# ==========================================================================
# Section 12: overlap resolution
# ==========================================================================

def test_overlap(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 12] overlap resolution")

    # Email vs person name buried inside it
    r = sieve.scan("Contact John Smith at john.smith@example.com today.")
    types = types_in(r)
    t.check("EMAIL" in types, "email still detected in overlap scenario")
    # The email span should not be broken into pieces by NER
    email_ent = entity_of(r, "EMAIL")
    if email_ent is not None:
        # No PERSON entity should overlap the email span
        person_overlaps_email = any(
            e.entity_type == "PERSON"
            and not (e.end <= email_ent.start or email_ent.end <= e.start)
            for e in r.entities
        )
        t.check(not person_overlaps_email,
                "no PERSON entity overlaps the email span (email wins)")
    # John Smith (outside the email) can still be caught as a separate PERSON
    if sieve.ner_enabled:
        t.check("PERSON" in types,
                "PERSON 'John Smith' outside the email still detected")


# ==========================================================================
# Section 13: position accuracy
# ==========================================================================

def test_positions(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 13] position accuracy")
    text = "Email: support@example.com is our address."
    r = sieve.scan(text)
    e = entity_of(r, "EMAIL")
    t.check(e is not None, "email entity produced")
    if e is not None:
        # Extract using recorded offsets on the ORIGINAL text
        extracted = text[e.start:e.end]
        t.check(extracted == e.original_text,
                f"offsets extract the original text "
                f"(text[{e.start}:{e.end}]={extracted!r} vs "
                f"stored={e.original_text!r})")


# ==========================================================================
# Section 14: coherent cleaned output
# ==========================================================================

def test_cleaned_coherence(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 14] cleaned output is coherent")
    text = "Please email us at support@example.com or call +91-9876543210."
    r = sieve.scan(text)
    cleaned = r.cleaned_text
    # Placeholder markers present
    t.check("[EMAIL REDACTED]" in cleaned, "email placeholder in cleaned text")
    t.check("[PHONE REDACTED]" in cleaned, "phone placeholder in cleaned text")
    # Original PII removed
    t.check("support@example.com" not in cleaned, "email fully removed")
    t.check("9876543210" not in cleaned, "phone digits fully removed")
    # Non-PII words survive
    t.check("Please email us at" in cleaned, "non-PII words preserved")
    t.check("or call" in cleaned, "surrounding words intact")


# ==========================================================================
# Section 15: empty / no-PII
# ==========================================================================

def test_edge_cases(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 15] empty / no-PII")
    r = sieve.scan("")
    t.check(r.pii_found is False, "empty text -> pii_found=False")
    t.check(r.entities == [], "empty text -> no entities")
    t.check(r.cleaned_text == "", "empty text -> empty cleaned")

    r = sieve.scan("The weather is nice today and the birds are singing.")
    t.check(r.pii_found is False, "clean text -> pii_found=False")
    t.check(r.cleaned_text.startswith("The weather"),
            "clean text unchanged")


# ==========================================================================
# Section 16: multi-PII
# ==========================================================================

def test_multi_pii(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 16] multi-PII response")
    text = (
        "Hello, my name is Priya Sharma. My Aadhaar is 2345 6789 0123, "
        "passport K1234567, card 4242 4242 4242 4242. "
        "Email me at priya@example.com or call +91-9876543210. "
        "I was born on 15/08/1990. Server IP is 192.168.1.100."
    )
    r = sieve.scan(text)
    types = types_in(r)

    expected_structured = {"AADHAAR", "PASSPORT", "CREDIT_CARD", "EMAIL",
                           "PHONE", "DATE_OF_BIRTH", "IP_ADDRESS"}
    missing = expected_structured - types
    t.check(not missing,
            f"all structured PII categories detected "
            f"(missing: {missing})")

    if sieve.ner_enabled:
        t.check("PERSON" in types, "PERSON detected in multi-PII text")

    # All original PII removed
    for original in ["2345 6789 0123", "K1234567", "4242 4242 4242 4242",
                     "priya@example.com", "+91-9876543210", "15/08/1990",
                     "192.168.1.100"]:
        t.check(original not in r.cleaned_text,
                f"original {original!r} scrubbed from output")


# ==========================================================================
# Section 17: end-to-end on the Mock LLM response
# ==========================================================================

def test_mock_response(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 17] end-to-end on the MockBackend response")
    from llm.client import MOCK_RESPONSE_TEMPLATE

    r = sieve.scan(MOCK_RESPONSE_TEMPLATE)
    t.check(r.pii_found is True, "MockBackend response contains PII")
    t.check("EMAIL" in types_in(r), "MockBackend email detected")
    t.check("PHONE" in types_in(r), "MockBackend phone detected")
    t.check("[EMAIL REDACTED]" in r.cleaned_text, "email placeholder in output")
    t.check("[PHONE REDACTED]" in r.cleaned_text, "phone placeholder in output")
    t.check("Team TrustGate" in r.cleaned_text, "non-PII text preserved")


# ==========================================================================
# Section 18: JSON serialization
# ==========================================================================

def test_json(t: TestRunner, sieve: PrivacySieve) -> None:
    print("\n[Section 18] report is JSON-serializable")
    r = sieve.scan("Email: support@example.com. Call +91-9876543210.")
    try:
        blob = json.dumps(r.to_dict())
        t.check(len(blob) > 0, "PrivacySieveReport.to_dict() serializes")
    except TypeError as e:
        t.check(False, f"report not JSON-serializable: {e}")


# ==========================================================================
def main() -> int:
    t = TestRunner()
    sieve = PrivacySieve()  # tries to load spaCy; falls back gracefully

    test_utilities(t)
    test_email(t, sieve)
    test_phone(t, sieve)
    test_credit_card(t, sieve)
    test_aadhaar(t, sieve)
    test_passport(t, sieve)
    test_ip(t, sieve)
    test_sensitive_url(t, sieve)
    test_dob(t, sieve)
    test_ner(t, sieve)
    test_overlap(t, sieve)
    test_positions(t, sieve)
    test_cleaned_coherence(t, sieve)
    test_edge_cases(t, sieve)
    test_multi_pii(t, sieve)
    test_mock_response(t, sieve)
    test_json(t, sieve)
    return t.summary()


if __name__ == "__main__":
    sys.exit(main())
