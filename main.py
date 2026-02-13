#!/usr/bin/env python3
"""
Entry point.  Three modes:

    python main.py                        interactive REPL
    python main.py "What is inheritance?" single query
    python main.py --test                 run the built-in test suite
"""

import sys

from src.classifier import ParadigmClassifier


TEST_QUERIES = [
    ("What is a pure function?", "Functional Programming"),
    ("Explain immutability in programming", "Functional Programming"),
    ("What is map and filter in functional programming?", "Functional Programming"),
    ("Tell me about first-class functions", "Functional Programming"),
    ("What are lambda expressions?", "Functional Programming"),
    ("How does step by step execution work?", "Procedural Programming"),
    ("What is a subroutine?", "Procedural Programming"),
    ("Explain top-down programming approach", "Procedural Programming"),
    ("How does C programming work?", "Procedural Programming"),
    ("What is sequential execution?", "Procedural Programming"),
    ("What is inheritance in programming?", "Object-Oriented Programming"),
    ("Explain encapsulation", "Object-Oriented Programming"),
    ("How do classes and objects work?", "Object-Oriented Programming"),
    ("What is polymorphism?", "Object-Oriented Programming"),
    ("Tell me about the four pillars of OOP", "Object-Oriented Programming"),
]


def run_tests(clf):
    """Run all test queries and check determinism."""

    print("=" * 65)
    print("  TEST SUITE  —  15 queries across 3 paradigms")
    print("=" * 65)

    passed = failed = 0

    for query, expected in TEST_QUERIES:
        result = clf.classify(query)
        ok = result.paradigm == expected
        passed += ok
        failed += (not ok)

        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] \"{query}\"")
        print(f"        expected {expected}, got {result.paradigm} "
              f"({result.confidence:.4f})")
        print()

    print("-" * 65)
    print("  DETERMINISM CHECK")
    print("-" * 65)

    all_ok = True
    for query, _ in TEST_QUERIES:
        r1 = clf.classify(query)
        r2 = clf.classify(query)
        if r1.confidence != r2.confidence or r1.paradigm != r2.paradigm:
            print(f"  [FAIL] non-deterministic: \"{query}\"")
            all_ok = False

    if all_ok:
        print("  [PASS] every query returned identical results on re-run")

    print()
    print("=" * 65)
    print(f"  {passed}/{len(TEST_QUERIES)} passed, {failed} failed")
    print(f"  determinism: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 65)


def interactive(clf):
    """Simple REPL for manual testing."""
    print()
    print("  Programming Paradigm Classifier")
    print("  type a query, or 'quit' to exit")
    print()

    while True:
        try:
            query = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = clf.classify(query)
        print()
        for line in str(result).split("\n"):
            print(f"  {line}")
        print()


def main():
    print()
    print("  loading model…")
    clf = ParadigmClassifier()
    print("  ready")
    print()

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_tests(clf)
        else:
            query = " ".join(sys.argv[1:])
            result = clf.classify(query)
            print(result)
    else:
        interactive(clf)


if __name__ == "__main__":
    main()
