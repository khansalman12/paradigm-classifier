# Programming Paradigm Classifier

Given a natural-language query, determines which programming paradigm
(Functional, Procedural, or OOP) the user is asking about — regardless of how
the question is worded. Returns `None` if nothing matches above a confidence
threshold.

## Quick Start

### Docker (recommended)

```bash
docker build -t paradigm-classifier .
docker run paradigm-classifier                                # run test suite
docker run paradigm-classifier python main.py "What is OOP?"  # single query
docker run -it paradigm-classifier python main.py              # interactive
```

### Local

```bash
pip install -r requirements.txt
python main.py --test                        # run test suite
python main.py "What is a pure function?"    # single query
python main.py                               # interactive REPL
```

### Tests

```bash
pytest tests/ -v
```

## How It Works

1. **Parse** `programming_types.md` → three paradigm descriptions
2. **Enrich** each description with related keywords and synonyms so the model
   can match phrasings not present in the source text
3. **Embed** everything using `all-MiniLM-L6-v2` (runs locally, ~80 MB, no API calls)
4. **Classify** by embedding the query, computing cosine similarity against all
   three paradigm vectors, and returning the best match above a confidence
   threshold of `0.28`

## Project Structure

```
├── programming_types.md   # source document with paradigm descriptions
├── main.py                # CLI — test suite, single query, or interactive REPL
├── Dockerfile             # self-contained image, model baked in at build time
├── requirements.txt
├── DESIGN.md              # why I made the choices I made
├── src/
│   ├── config.py          # seeds, model name, threshold — all tunables
│   ├── preprocessor.py    # markdown parser + keyword enrichment
│   ├── embeddings.py      # sentence-transformer wrapper + similarity scoring
│   └── classifier.py      # ties it all together
└── tests/
    └── test_classifier.py # accuracy, determinism, edge cases
```

## Reproducibility

Same input → same output, every run, every machine. Guaranteed by:

- Fixed random seeds — Python, NumPy, PyTorch all pinned to `42`
- Model in `eval()` mode (dropout disabled)
- `PYTHONHASHSEED=42` set in both code and Dockerfile
- Zero sampling or temperature anywhere in the pipeline

The test suite verifies determinism explicitly: every query is run twice and
scores are compared for bitwise equality.

## Design Decisions

See [DESIGN.md](DESIGN.md) for a detailed walkthrough of why I chose this
approach, the problems I ran into, and how I solved them. Summary:

| Decision | Why |
|---|---|
| Sentence embeddings over keyword matching | Handles rephrasing naturally — "what is inheritance" and "how do objects share behavior" both match OOP without manually listing every possible wording |
| `all-MiniLM-L6-v2` | Small (80 MB), fast, trained for semantic similarity, runs fully offline |
| Keyword enrichment | The source document doesn't mention "lambda" or "Haskell" — enrichment bridges that gap without modifying the source |
| Cosine similarity | Standard for comparing normalized sentence embeddings — dot product shortcut when vectors are unit-length |
| Threshold at 0.28 | Empirically determined — low enough to catch valid single-word queries, high enough to reject unrelated noise |
