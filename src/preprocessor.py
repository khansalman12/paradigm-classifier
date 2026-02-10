"""Markdown parser — turns programming_types.md into Paradigm objects."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config


@dataclass
class Paradigm:
    """A single paradigm from the source document."""

    name: str
    description: str
    key_concepts: list = field(default_factory=list)
    synonyms: list = field(default_factory=list)

    def get_enriched_text(self):
        """Build the full text we actually embed (description + extra keywords)."""
        parts = [self.description]
        if self.key_concepts:
            parts.append("Key concepts: " + ", ".join(self.key_concepts))
        if self.synonyms:
            parts.append("Related terms: " + ", ".join(self.synonyms))
        return "\n".join(parts)


# these fill gaps in the source doc — e.g. it never mentions "lambda"
_ENRICHMENTS = {
    "Functional Programming": {
        "key_concepts": [
            "immutability", "pure functions", "first-class functions",
            "declarative", "no side effects", "mathematical functions",
            "composing functions", "higher-order functions",
            "data science", "concurrent systems",
        ],
        "synonyms": [
            "FP", "functional style", "lambda calculus", "lambda functions",
            "lambda expressions", "map", "reduce", "filter",
            "map reduce", "stateless", "referential transparency",
            "Haskell", "Erlang", "Clojure", "Elixir", "immutable data",
            "functional composition", "recursion over loops",
        ],
    },
    "Procedural Programming": {
        "key_concepts": [
            "procedure call", "sequence of instructions", "step by step",
            "top-down", "subroutines", "modular blocks",
            "flow of execution", "global state", "imperative",
        ],
        "synonyms": [
            "procedural style", "structured programming",
            "C programming", "Fortran", "Pascal", "BASIC",
            "scripting", "routines", "sequential execution",
            "step-by-step instructions",
        ],
    },
    "Object-Oriented Programming": {
        "key_concepts": [
            "objects", "classes", "encapsulation", "abstraction",
            "inheritance", "polymorphism", "methods", "attributes",
            "fields", "four pillars", "code reusability",
        ],
        "synonyms": [
            "OOP", "object oriented", "class-based",
            "Java style", "C++ style", "design patterns",
            "interfaces", "data hiding", "real-world modelling",
            "enterprise applications",
        ],
    },
}


def parse_document(filepath=None):
    """Read the markdown, split by headings, return 3 Paradigm objects."""
    filepath = filepath or Config.DOCUMENT_PATH
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {filepath}  — "
            "make sure programming_types.md is in the project root."
        )

    text = path.read_text(encoding="utf-8")
    sections = re.split(r"^#\s+\d+\.\s+", text, flags=re.MULTILINE)

    paradigms = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        matched = None
        for label in Config.PARADIGM_LABELS:
            if label.lower() in section[:80].lower():
                matched = label
                break

        if matched is None:
            continue

        lines = section.split("\n", 1)
        description = lines[1].strip() if len(lines) > 1 else section

        extras = _ENRICHMENTS.get(matched, {})
        paradigms.append(Paradigm(
            name=matched,
            description=description,
            key_concepts=extras.get("key_concepts", []),
            synonyms=extras.get("synonyms", []),
        ))

    if len(paradigms) != 3:
        raise ValueError(
            f"Expected 3 paradigms but found {len(paradigms)}.  "
            "Check the format of programming_types.md."
        )

    return paradigms
