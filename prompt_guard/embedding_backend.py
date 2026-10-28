"""
Embedding backend for Prompt Guard.

Two interchangeable implementations, both producing L2-normalized float32
dense vectors of a fixed dimensionality. The classifier and semantic
analyzer treat them identically.

Primary (production):
    SentenceTransformerBackend
    Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim) downloaded
    from the Hugging Face Hub the first time it runs.

Fallback (offline / air-gapped / CI):
    LocalHashSvdBackend
    A HashingVectorizer(char_wb + word) -> TruncatedSVD pipeline fitted
    at training time from the training corpus. Produces the same 384-dim
    normalized vectors so nothing downstream changes.

Both backends are pickled and reloaded so inference reuses whatever was
trained.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from typing import Iterable

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

EMBED_DIM = 384


class EmbeddingBackend:
    """Common interface."""

    name: str = "base"
    dim: int = EMBED_DIM

    def fit(self, texts: list[str]) -> "EmbeddingBackend":
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @staticmethod
    def load(path: Path) -> "EmbeddingBackend":
        payload = joblib.load(path)
        kind = payload.get("kind")
        if kind == "sentence_transformer":
            return SentenceTransformerBackend.from_payload(payload)
        if kind == "local_hash_svd":
            return LocalHashSvdBackend.from_payload(payload)
        raise ValueError(f"Unknown embedding backend kind: {kind}")


class SentenceTransformerBackend(EmbeddingBackend):
    """Wrap a sentence-transformers model. Preferred for production."""

    name = "sentence_transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            # Imported lazily so the fallback path doesn't need the library.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> "SentenceTransformerBackend":
        # Pretrained: nothing to fit.
        self._ensure()
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure()
        arr = model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(arr, dtype=np.float32)

    def save(self, path: Path) -> None:
        joblib.dump({"kind": self.name, "model_name": self.model_name}, path)

    @classmethod
    def from_payload(cls, payload: dict) -> "SentenceTransformerBackend":
        return cls(model_name=payload["model_name"])


class LocalHashSvdBackend(EmbeddingBackend):
    """
    Offline dense embedder.

    HashingVectorizer with (a) word tokens and (b) char_wb n-grams, followed
    by TruncatedSVD to bring us down to `EMBED_DIM`, followed by L2
    normalization. Yields normalized dense float32 vectors of dimension
    `EMBED_DIM` — the same contract as SentenceTransformer.
    """

    name = "local_hash_svd"

    def __init__(self, dim: int = EMBED_DIM, random_state: int = 1729):
        self.dim = dim
        self.random_state = random_state
        self._pipe: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        union = FeatureUnion([
            (
                "word",
                HashingVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    n_features=2**15,
                    alternate_sign=False,
                    norm="l2",
                    lowercase=True,
                ),
            ),
            (
                "char_wb",
                HashingVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 4),
                    n_features=2**15,
                    alternate_sign=False,
                    norm="l2",
                    lowercase=True,
                ),
            ),
        ])
        svd = TruncatedSVD(
            n_components=self.dim,
            random_state=self.random_state,
            algorithm="randomized",
            n_iter=5,
        )
        return Pipeline([
            ("features", union),
            ("svd", svd),
            ("l2", Normalizer(norm="l2")),
        ])

    def fit(self, texts: list[str]) -> "LocalHashSvdBackend":
        self._pipe = self._build_pipeline()
        self._pipe.fit(list(texts))
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("LocalHashSvdBackend not fitted / loaded.")
        arr = self._pipe.transform(list(texts))
        return np.asarray(arr, dtype=np.float32)

    def save(self, path: Path) -> None:
        if self._pipe is None:
            raise RuntimeError("Refusing to save an unfitted embedder.")
        joblib.dump(
            {
                "kind": self.name,
                "dim": self.dim,
                "random_state": self.random_state,
                "pipeline": self._pipe,
            },
            path,
        )

    @classmethod
    def from_payload(cls, payload: dict) -> "LocalHashSvdBackend":
        inst = cls(dim=payload["dim"], random_state=payload["random_state"])
        inst._pipe = payload["pipeline"]
        return inst


def make_backend(prefer: str = "sentence_transformer") -> EmbeddingBackend:
    """
    Return the requested backend. Falls back automatically if the
    preferred one can't initialize (e.g. no HF access).
    """
    if prefer == "sentence_transformer":
        try:
            b = SentenceTransformerBackend()
            b.fit([])  # forces model load
            return b
        except Exception as e:  # noqa: BLE001
            print(f"[embedding_backend] SentenceTransformer unavailable ({e.__class__.__name__}); "
                  f"falling back to LocalHashSvdBackend.")
            return LocalHashSvdBackend()
    if prefer == "local":
        return LocalHashSvdBackend()
    raise ValueError(f"Unknown backend preference: {prefer}")
