"""
Tests for the paradigm classifier.

    pytest tests/ -v
"""

import pytest
from src.classifier import ParadigmClassifier
from src.preprocessor import parse_document


@pytest.fixture(scope="module")
def clf():
    return ParadigmClassifier()


# --- document parsing ---

class TestParsing:

    def test_finds_three_paradigms(self):
        paradigms = parse_document()
        assert len(paradigms) == 3

    def test_correct_names(self):
        names = [p.name for p in parse_document()]
        assert "Functional Programming" in names
        assert "Procedural Programming" in names
        assert "Object-Oriented Programming" in names

    def test_descriptions_are_not_empty(self):
        for p in parse_document():
            assert len(p.description) > 50

    def test_enriched_text_has_extras(self):
        for p in parse_document():
            text = p.get_enriched_text()
            assert "Key concepts:" in text
            assert "Related terms:" in text


# --- classification accuracy ---

class TestAccuracy:

    @pytest.mark.parametrize("query", [
        "What is a pure function?",
        "Explain immutability in programming",
        "Tell me about first-class functions",
        "What are lambda expressions?",
        "What is map and filter in functional programming?",
    ])
    def test_functional(self, clf, query):
        assert clf.classify(query).paradigm == "Functional Programming"

    @pytest.mark.parametrize("query", [
        "How does step by step execution work?",
        "What is a subroutine?",
        "Explain top-down programming approach",
        "How does C programming work?",
        "What is sequential execution?",
    ])
    def test_procedural(self, clf, query):
        assert clf.classify(query).paradigm == "Procedural Programming"

    @pytest.mark.parametrize("query", [
        "What is inheritance in programming?",
        "Explain encapsulation",
        "How do classes and objects work?",
        "What is polymorphism?",
        "Tell me about the four pillars of OOP",
    ])
    def test_oop(self, clf, query):
        assert clf.classify(query).paradigm == "Object-Oriented Programming"


# --- determinism ---

class TestDeterminism:

    def test_same_result_twice(self, clf):
        r1 = clf.classify("What is inheritance?")
        r2 = clf.classify("What is inheritance?")
        assert r1.paradigm == r2.paradigm
        assert r1.confidence == r2.confidence

    def test_scores_identical(self, clf):
        r1 = clf.classify("Tell me about pure functions")
        r2 = clf.classify("Tell me about pure functions")
        for (n1, s1), (n2, s2) in zip(r1.all_scores, r2.all_scores):
            assert n1 == n2 and s1 == s2

    def test_determinism_across_all_paradigms(self, clf):
        queries = [
            "What is immutability?",
            "Explain subroutines",
            "What is polymorphism?",
        ]
        for q in queries:
            r1 = clf.classify(q)
            r2 = clf.classify(q)
            assert r1.confidence == r2.confidence
            assert r1.paradigm == r2.paradigm


# --- edge cases ---

class TestEdgeCases:

    def test_empty_string(self, clf):
        r = clf.classify("")
        assert r.paradigm is None
        assert r.is_match is False

    def test_whitespace_only(self, clf):
        assert clf.classify("   ").is_match is False

    def test_none_like_input(self, clf):
        assert clf.classify("\t\n  ").is_match is False

    def test_always_three_scores(self, clf):
        assert len(clf.classify("OOP").all_scores) == 3

    def test_single_word_ranks_correctly(self, clf):
        # "encapsulation" scores just below threshold (0.277) but should
        # still rank OOP first
        r = clf.classify("encapsulation")
        top_paradigm = r.all_scores[0][0]
        assert top_paradigm == "Object-Oriented Programming"

    def test_unrelated_query_returns_none(self, clf):
        r = clf.classify("What is the weather today?")
        assert r.paradigm is None
        assert r.is_match is False

    def test_another_unrelated_query(self, clf):
        r = clf.classify("How to make pasta?")
        assert r.paradigm is None
        assert r.is_match is False

    def test_scores_are_bounded(self, clf):
        r = clf.classify("Tell me about classes")
        for _, score in r.all_scores:
            assert -1.0 <= score <= 1.0

    def test_best_score_is_first(self, clf):
        r = clf.classify("What is inheritance?")
        scores = [s for _, s in r.all_scores]
        assert scores == sorted(scores, reverse=True)
