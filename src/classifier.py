"""Main classifier — query in, paradigm out."""

from dataclasses import dataclass

from .config import Config
from .embeddings import EmbeddingEngine
from .preprocessor import parse_document


@dataclass
class ClassificationResult:
    """Holds the classification output."""

    paradigm: str
    confidence: float
    all_scores: list
    is_match: bool

    def __str__(self):
        scores_str = ", ".join(
            f"{name}: {s:.4f}" for name, s in self.all_scores
        )
        if not self.is_match:
            return (
                f"Result: None (no paradigm matched)\n"
                f"Reason: best score ({self.confidence:.4f}) "
                f"< threshold ({Config.CONFIDENCE_THRESHOLD})\n"
                f"Scores: [{scores_str}]"
            )
        return (
            f"Result: {self.paradigm}\n"
            f"Confidence: {self.confidence:.4f}\n"
            f"Scores: [{scores_str}]"
        )


class ParadigmClassifier:
    """Loads model once, then classifies queries against the three paradigms."""

    def __init__(self, document_path=None):
        Config.set_seeds()
        self._paradigms = parse_document(document_path)
        self._engine = EmbeddingEngine(self._paradigms)

    def classify(self, query):
        """Classify a query into one of the three paradigms."""
        if not query or not query.strip():
            return ClassificationResult(
                paradigm=None,
                confidence=0.0,
                all_scores=[(p.name, 0.0) for p in self._paradigms],
                is_match=False,
            )

        scores = self._engine.compute_similarities(query.strip())
        best_name, best_score = scores[0]
        matched = best_score >= Config.CONFIDENCE_THRESHOLD

        return ClassificationResult(
            paradigm=best_name if matched else None,
            confidence=best_score,
            all_scores=scores,
            is_match=matched,
        )
