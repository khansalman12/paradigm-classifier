"""Handles all the embedding + similarity math."""

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Config
from .preprocessor import Paradigm


class EmbeddingEngine:
    """Embeds paradigms once at init, then scores queries against them."""

    def __init__(self, paradigms):
        Config.set_seeds()
        self._paradigms = paradigms

        self._model = SentenceTransformer(Config.MODEL_NAME)
        self._model.eval()

        self._paradigm_vecs = self._embed_paradigms()

    def _embed_paradigms(self):
        """Embed all three paradigms. Returns (3, dim) numpy array."""
        texts = [p.get_enriched_text() for p in self._paradigms]
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors → dot product = cosine sim
            show_progress_bar=False,
        )

    def _embed_query(self, query):
        """Embed a single query string."""
        vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    def compute_similarities(self, query):
        """Return [(name, score), ...] sorted best-first."""
        q_vec = self._embed_query(query)
        scores = np.dot(self._paradigm_vecs, q_vec)

        results = [
            (p.name, float(s))
            for p, s in zip(self._paradigms, scores)
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
