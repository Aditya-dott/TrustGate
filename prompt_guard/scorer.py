"""
Prompt Guard scorer.

Public API:
    scorer = PromptGuardScorer()
    report: PromptGuardReport = scorer.score(prompt_text)

Pipeline:
    1. Split the prompt into overlapping word windows so long prompts with
       a malicious clause buried mid-paragraph cannot dilute the score.
       Short prompts pass through as a single chunk.
    2. Embed every chunk once (the embedding backend is shared between the
       classifier and the semantic analyzer, so this is a single call).
    3. Run the calibrated classifier on the chunk embeddings.
    4. Run the semantic analyzer on the same chunks to get attack-similarity
       and deviation signals per chunk.
    5. Aggregate: overall suspicion_score is MAX per-chunk P(malicious).
       Attack similarity and deviation are MAX over chunks (worst case).
    6. Build a structured PromptGuardReport. It contains scores, per-chunk
       details, the detected attack type (if suspicion is high enough),
       and the confidence. It never contains an allow / flag / block
       verdict - those live only in the Supervisor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from prompt_guard.embedding_backend import EmbeddingBackend
from prompt_guard.semantic_analyzer import SemanticAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "models"

DEFAULT_WINDOW_WORDS = 60
DEFAULT_STRIDE_WORDS = 30

MALICIOUS_CLASSES = ("prompt_injection", "jailbreak", "data_exfiltration", "model_extraction")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    window: int = DEFAULT_WINDOW_WORDS,
    stride: int = DEFAULT_STRIDE_WORDS,
) -> list[str]:
    """
    Split `text` into overlapping windows of `window` words with `stride`
    word steps. Short text (<= window words) is returned as one chunk.

    The final window is always tail-aligned to len(words), so no words get
    truncated into a short trailing chunk and every chunk has exactly
    `window` words (except when the whole text is shorter than a window).
    """
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError("Require 0 < stride <= window.")
    words = text.split()
    if len(words) <= window:
        return [text]

    # Regular strided starts
    starts = list(range(0, len(words) - window + 1, stride))
    # Ensure the last window ends exactly at len(words)
    last_full_start = len(words) - window
    if starts[-1] != last_full_start:
        starts.append(last_full_start)

    return [" ".join(words[s:s + window]) for s in starts]


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PromptGuardReport:
    """
    Complete structured report returned by the scorer.

    NO verdict field. The Supervisor is the only component that decides
    block / flag / allow.
    """
    suspicion_score: float                   # max per-chunk (1 - P(benign))
    per_chunk_scores: list[float]            # per-chunk suspicion values
    num_chunks: int
    detected_attack_type: str | None         # top malicious class if suspicion >= 0.5, else None
    top_attack_class: str                    # top malicious class, always populated
    attack_similarity: float                 # max per-chunk normalized attack similarity
    deviation_score: float                   # max per-chunk deviation
    raw_attack_similarity: float             # max per-chunk raw cosine to attack templates
    raw_deviation: float                     # max per-chunk raw (1 - max-anchor-cos)
    noise_baseline: float                    # auto-calibrated noise floor
    confidence: float                        # max class probability on the top chunk
    class_probabilities: dict[str, float]    # on the top chunk
    per_chunk_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PromptGuardScorer:
    """
    Loads the embedding backend, the trained classifier, and the semantic
    analyzer, and produces a full PromptGuardReport for any prompt.
    """

    def __init__(
        self,
        models_dir: Path | str = DEFAULT_MODELS_DIR,
        window: int = DEFAULT_WINDOW_WORDS,
        stride: int = DEFAULT_STRIDE_WORDS,
    ):
        models_dir = Path(models_dir)
        self.window = window
        self.stride = stride
        self.backend: EmbeddingBackend = EmbeddingBackend.load(models_dir / "embedding_model.pkl")
        self.classifier = joblib.load(models_dir / "classifier.pkl")
        self.classes: list[str] = list(self.classifier.classes_)
        if "benign" not in self.classes:
            raise ValueError("Trained classifier does not have a 'benign' class.")
        self.benign_idx: int = self.classes.index("benign")
        self.malicious_indices: list[int] = [
            i for i, c in enumerate(self.classes) if c != "benign"
        ]
        # SemanticAnalyzer precomputes template + anchor vectors in its __init__
        self.analyzer: SemanticAnalyzer = SemanticAnalyzer(self.backend)

    # ------------------------------------------------------------------
    def score(self, prompt: str) -> PromptGuardReport:
        prompt = (prompt or "").strip()
        if not prompt:
            return self._empty_report()

        chunks = chunk_text(prompt, self.window, self.stride)

        # Encode once - shared between classifier and analyzer for efficiency
        chunk_vecs = self.backend.encode(chunks)
        proba = self.classifier.predict_proba(chunk_vecs)   # (n_chunks, n_classes)

        # Semantic analyzer needs raw text (it re-encodes internally). We
        # could pass vectors in for efficiency, but keeping the interface
        # text-in / signals-out makes the analyzer independently usable.
        signals = self.analyzer.analyze_batch(chunks)       # length n_chunks

        # Per-chunk suspicion = 1 - P(benign)
        per_chunk_suspicion = 1.0 - proba[:, self.benign_idx]

        # Aggregations (max across chunks)
        top_chunk = int(np.argmax(per_chunk_suspicion))
        suspicion_score = float(per_chunk_suspicion[top_chunk])
        attack_sim = float(max(s.attack_similarity for s in signals))
        deviation = float(max(s.deviation_score for s in signals))
        raw_attack = float(max(s.raw_attack_similarity for s in signals))
        raw_dev = float(max(s.raw_deviation for s in signals))

        # Top chunk class distribution -> attack type
        top_proba = proba[top_chunk]
        class_probs = {cls: float(top_proba[i]) for i, cls in enumerate(self.classes)}
        confidence = float(top_proba.max())

        mal_probs_on_top = np.array([top_proba[i] for i in self.malicious_indices])
        top_mal_local = int(np.argmax(mal_probs_on_top))
        top_attack_class = self.classes[self.malicious_indices[top_mal_local]]

        # detected_attack_type: only meaningful if the classifier itself
        # thinks the prompt is malicious. Threshold at 0.5 (Supervisor may
        # still block / flag / allow independently).
        detected_attack_type = top_attack_class if suspicion_score >= 0.5 else None

        per_chunk_details: list[dict[str, Any]] = []
        for i, (chunk, sig) in enumerate(zip(chunks, signals)):
            row = proba[i]
            per_chunk_details.append({
                "chunk_index": i,
                "text_preview": chunk if len(chunk) <= 220 else chunk[:220] + "...",
                "p_benign": float(row[self.benign_idx]),
                "class_probabilities": {cls: float(row[j]) for j, cls in enumerate(self.classes)},
                "suspicion": float(per_chunk_suspicion[i]),
                "attack_similarity": sig.attack_similarity,
                "deviation_score": sig.deviation_score,
                "raw_attack_similarity": sig.raw_attack_similarity,
                "raw_deviation": sig.raw_deviation,
                "closest_attack_template": sig.closest_attack_template,
                "closest_normal_anchor": sig.closest_normal_anchor,
            })

        return PromptGuardReport(
            suspicion_score=suspicion_score,
            per_chunk_scores=[float(x) for x in per_chunk_suspicion],
            num_chunks=len(chunks),
            detected_attack_type=detected_attack_type,
            top_attack_class=top_attack_class,
            attack_similarity=attack_sim,
            deviation_score=deviation,
            raw_attack_similarity=raw_attack,
            raw_deviation=raw_dev,
            noise_baseline=self.analyzer.noise_baseline,
            confidence=confidence,
            class_probabilities=class_probs,
            per_chunk_details=per_chunk_details,
        )

    # ------------------------------------------------------------------
    def _empty_report(self) -> PromptGuardReport:
        """Report for empty / whitespace prompts - safe defaults."""
        return PromptGuardReport(
            suspicion_score=0.0,
            per_chunk_scores=[],
            num_chunks=0,
            detected_attack_type=None,
            top_attack_class=self.classes[self.malicious_indices[0]],
            attack_similarity=0.0,
            deviation_score=0.0,
            raw_attack_similarity=0.0,
            raw_deviation=0.0,
            noise_baseline=self.analyzer.noise_baseline,
            confidence=1.0,
            class_probabilities={cls: 0.0 for cls in self.classes} | {"benign": 1.0},
            per_chunk_details=[],
        )
