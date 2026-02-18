# Design Decisions

What I tried, what broke, and what I ended up going with.

## Why sentence embeddings?

The task says "independently from the wording used." So I can't just match
keywords — someone might say "what is inheritance" or "how do child classes
get behavior from parents" and mean the same thing.

Keyword matching needs you to think of every possible phrasing. Miss one and
it fails. TF-IDF is better but still word-level — it can't connect "lambda"
to "anonymous function."

Sentence embeddings handle this. The model maps text to vectors where similar
meaning = nearby vectors. Different words, same concept → still a match.

## Why all-MiniLM-L6-v2?

Three things: it's small (~80 MB, loads fast), it's actually *trained* for
semantic similarity (not just general language modeling), and it runs offline.
No API keys, the Docker image has everything baked in.

I also tried `all-mpnet-base-v2` (~420 MB). Slightly better on edge cases but
not enough to justify 5x the size — especially after I got enrichment working.

## The enrichment problem

This was the tricky part. The source document talks about "mathematical
functions" and "immutability" for Functional Programming but never mentions
"lambda," "Haskell," or "map/reduce." Users will definitely use those words.

Without enrichment, "What are lambda expressions?" matched **Procedural**
because "expressions" was closer to that embedding space. Completely wrong.

I tried a few things before getting it right:

1. **Lowered the threshold** — "lambda" scored 0.24 against Functional, so just
   accept anything above 0.20. But then "what's the weather" scored 0.22 against
   Procedural and got classified. Can't fix a signal problem with threshold.

2. **Added single-word keywords** — "map," "class," "function" etc. Made it
   worse. "Function" appears in all three paradigms, so everything got pulled
   toward everything.

3. **Specific multi-word phrases, balanced count** — "lambda functions,"
   "map and filter operations," "recursion over loops" for Functional, and
   similar specificity for the others. Balanced so no paradigm dominates.
   This worked — all test queries pass.

The takeaway: enrichment terms need to be specific enough to disambiguate and
balanced enough not to bias toward one paradigm.

## How I found the 0.28 threshold

No formula. I logged scores for about 20 queries:

- Real paradigm queries: **0.30–0.65**
- Borderline single-word queries ("encapsulation"): **0.28–0.32**
- Random unrelated stuff ("what's the weather"): **0.18–0.25**

0.28 is the gap. Below that, it's probably noise.

## Determinism

Sounds simple ("just set a seed") but randomness hides in a few places:

- `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`
- `cudnn.deterministic = True` (GPU ops)
- `PYTHONHASHSEED=42` — in code AND the Dockerfile (missed this the first time,
  tests passed locally but not in Docker)
- `model.eval()` to disable dropout

The test suite runs every query twice and checks for exact score matches. If
any source of randomness leaks through, this catches it.
