# Development Log

## Oct 20: Project setup
- Set up Python, PyCharm, Git and a virtual environment (.venv); installed Ollama to run a free local model.
- Problem: packages installed into my main Python instead of .venv. Fixed by activating .venv and checking `sys.prefix`.

## Oct 21: First working gateway
- Built a FastAPI service with `/ask` and `/health`. It forwards questions to llama3.2 and returns the answer, response time and token counts.
- Problem: "model not found" error. The model wasn't downloaded yet. The gateway handled it with a clear 502 message instead of crashing.
- Added .env.example so others know which settings to fill in, and kept .env out of Git.

## Oct 22: Tests and database
- Wrote automated tests with pytest, including a fake model (mocking) so tests run without Ollama.
- Ran Postgres with pgvector in Docker and designed the `requests` table to log every request.
- Problem: a YAML indentation mistake in docker-compose.yml broke every Docker command.

## Oct 24: Logging, cost tracking and baseline report
- The gateway now saves every request, including failures, to Postgres, using parameterized queries to prevent SQL injection.
- Added cost calculation per request using exact Decimal math.
- Built a report script: success rate, average and p95 response time, tokens, cost.
- Finding: the local model took 39–85 seconds per answer on my laptop. Free, but slow.
- Problem: PyCharm's Move tool rewrote my imports incorrectly; I tracked it down by reading the traceback.

## Oct 26: Two model tiers and a test question set
- The gateway now supports a cheap model (local llama) and a strong model (Claude), chosen per request.
- Built a batch runner that sends a set of test questions to one or both models and saves the results.
- Finding: [which questions llama got right and wrong, e.g. "llama got X of 5 math questions right and missed the 'two coins' trick question"]

## Oct 27: Claude added as the strong model
- Built: connected Claude (claude-sonnet-5-5) as the strong tier, confirmed its pricing, and wrote a script that compares both models side by side.
- Problem: I accidentally pasted my API key into a chat → disabled it and created a new one. Lesson: keys only ever go in .env.
- Finding: on 20 sample questions, llama3.2:1b got 13/14 checkable answers right and Claude got 14/14. Claude was ~7x faster (2.5 s vs 17.4 s) and cost $0.045 for all 20 (~$0.0023 per question).
- Llama's only miss: the logic question (reason-01). It answered "Yes" when the correct answer is "No."
- Lesson: my "does the answer contain the reference?" check wrongly marked Claude's correct answer to reason-02 as wrong because of different wording. Simple text matching isn't reliable, so grading needs exact number checks plus an LLM judge.
- Next: the sample questions are mostly too easy. Moving to 200 real questions from GSM8K and Dolly.

## Oct 28: 200-question test set and automatic grading
- Built: a 200-question test set (100 GSM8K math problems, 100 Dolly everyday requests) with a fixed random seed, and a grading script: exact number checks for math, Claude as an LLM judge for everything else.
- Ran both models on all 200 questions and graded every cheap-model answer.
- Finding: the free model (llama3.2:1b) was good enough on 68/200 questions (34%).
  - Math: 41% for llama vs 89% for Claude.
  - Everyday requests: 17–33% depending on category (brainstorming lowest at 17%, open Q&A 31%).
- Judge cost for grading 100 open-ended answers: $0.24.
- What it means: about a third of these questions could go to the free model. That's the realistic ceiling for what a router could save on this question mix, if it routes perfectly.
- Problem: test_grading.py imported scripts that didn't exist yet → limited the import to what exists each day.
- Next: hand-check a sample of the judge's verdicts to see whether it can be trusted.

## Oct 28 (continued): checking the judge
- Reviewed 17 of the judge's verdicts myself, without seeing its decision first.
- Finding: we agreed on 8/17 (47%), but all 9 disagreements went one way: I passed answers the judge failed.
- Reading the judge's reasons, most of those answers contained real factual errors (invented chess players and openings, wrong facts about Charleston and Alaska) or didn't actually answer the question. I had judged them on how fluent they sounded.
- Lesson: small models hallucinate confidently, and fluent answers hide errors from a quick read. A judge with a reference answer catches them.
- The judge seemed too strict in a couple of cases (formatting, missing details from the reference).

## Oct 29: checking the math grading
- Checked 10 math answers marked wrong: all 10 were really wrong, and none were correct answers mis-graded. The last-number check is reliable.
- Llama's math errors are mostly multi-step reasoning mistakes, not arithmetic:
  - gsm8k-0018: reached the correct $66 mid-answer, then did an unnecessary subtraction and ended at $134.
  - gsm8k-0017: divided dollars by the number of kids and called the result "books."
  - gsm8k-0016: misread what the question asked and gave an unrelated number.
- Takeaway: multi-step word problems are exactly what the router should send to the strong model.

## Nov 1: Claude Haiku as the cheap model

What I did
- Switched the cheap tier from local llama3.2:1b to Claude Haiku (only `.env` changed; the gateway code didn't).
- Added `scripts/merge_results.py` so I only re-ran the cheap model and reused the Sonnet answers I'd already paid for.
- Updated `routing_data.py` to count the cheap model's cost (it isn't free anymore).
- Re-graded all 200 questions (judge cost $0.21) and re-ran every router.
- Fixed `train_router.py` to use the same pipeline as `improve_router` (StandardScaler + logistic regression) and saved the new router model.
- Tests: 36 passed.

Results
- Haiku was good enough on 175/199 questions (88%), compared with 34% for llama. On math it matched Sonnet exactly (89/100).
- Always Haiku: 87.9% quality and 95% cheaper than always Sonnet (94.5% quality, $0.72 for 199 questions).
- Hand-written rules: 90.5% quality, 53% savings. They sent all 100 math questions to Sonnet, which wasted money now that Haiku is as good at math.
- Embedding router (scaled embeddings + logistic regression): cross-validated AUC 0.735 (llama: 0.46). It clearly beats the hand-written rules, but with only 24 Haiku failures in 199 questions, the exact savings at a given quality level are noisy (they swing between roughly 30% and 80%).
- Oracle (perfect routing): 98% quality at 78% savings. Perfect routing beats always using Sonnet, because Haiku sometimes gets right what Sonnet gets wrong.

Learned
- Whether routing works depends on the cheap model: with a weak one, nothing works; with a strong one, routing pays off.
- Rules written by hand encode assumptions that go stale when models change; a router trained on labels adapts.
- My two scripts disagreed (AUC 0.675 vs 0.735) because one scaled the embeddings and one didn't. Training and evaluation must use exactly the same pipeline.
- Caveat: Sonnet's everyday answers are assumed correct, so "always Sonnet" quality is slightly flattered.

#Cascade vs router

What I did
- Built `scripts/cascade.py`: Haiku answers, then a check decides whether to keep the answer or escalate to Sonnet.
  - Check A: Haiku rates its own answer from 0 to 100 (it never sees the reference answer).
  - Check B: for math, ask Haiku twice and compare the final numbers; self-check for the rest.
- Paid check calls are cached in `data/cascade/`, so re-runs are free. Added 4 tests (40 passed).

Results (199 questions)
- The self-check scored below 90 on 18/24 of Haiku's wrong answers, but it also doubts many correct ones (at threshold 90, only 57% of answers kept cheap).
- Asking twice caught only 3/11 wrong math answers: Haiku usually repeats the same mistake.
- Savings at a given quality loss (router / cascade A / cascade B):
  - no loss: 0% / 6% / 15%
  - ≤1%: 0% / 43% / 47%
  - ≤3%: 65% / 69% / 68%
  - ≤5%: 82% / 80% / 78%
- When quality must stay within 1% of Sonnet, only the cascade saves money. At 3–5% loss, router and cascade tie (the differences are 1–2 questions).
- The self-check roughly doubled the Haiku spend, and the cascade adds latency (up to 3 calls in a row).

Learned
- Looking at the answer beats guessing from the question, but only when you need high quality.
- The model's errors are systematic, so sampling twice doesn't catch them.
- A "check" has its own cost and latency, and those belong in the comparison.
- With 24 failures, differences of a few percent are noise.


## Nov 2: "tier": "auto" in the gateway

What I did
- Added `app/routing.py`: the gateway now picks the model itself for `"tier": "auto"`.
  - Cascade mode (default): Haiku answers, rates its own answer, and escalates to Sonnet below 85 (offline: 94.0% quality, 43% savings).
  - Router mode: the trained router picks Haiku or Sonnet from the question with one call (threshold 0.95, from train_router's "at most 3% loss" setting).
- Fallbacks: Haiku down → Sonnet; Sonnet down → Haiku's answer; router unavailable → cascade.
- Responses and database rows record the route, self-check score, number of calls and total cost (including the check).
- The self-check prompt now lives in one place, shared by the offline evaluation and the live gateway.
- Tests: 51 passed.

Tried it live (cascade mode)
- "Capital of France": Haiku kept, $0.00002, 2.6 s
- Multi-step pens math problem: escalated to Sonnet, $0.0031, 7.6 s
- Haiku about rain: Haiku kept, $0.00017, 3.0 s
- Easy questions stay almost free; the escalated one cost about 150x more and took 3x longer (three calls in a row).

Problems I hit
- Requests weren't logged: Docker Desktop and the database container were off. The gateway kept answering (by design) but couldn't save rows. Start order: Docker Desktop → `docker compose up -d` → Ollama → uvicorn.
- `.env` changes seemed ignored: four old uvicorn processes were still listening on port 8000 after Ctrl+C (a Windows issue with `--reload`). Worked around it by running on port 8001.

Next: replay the 200 test questions through the live gateway in both modes and check that the savings match the offline evaluation.

## Nov 2 (later): Live replay of 199 questions

What I did
- Added `"mode"` to requests, so cascade or router can be chosen per request without restarting.
- Added `scripts/replay.py`: sends the labeled questions through the running gateway and reports quality, cost, calls and latency.
- Added `scripts/route_report.py`: cost and latency by route, straight from the requests table.
- Tests: 55 passed.

Results (199 questions each, 4 at a time, 0 errors out of 398 requests)
| | Cascade (threshold 85) | Router (threshold 0.95) |
|---|---|---|
| Answered by Haiku | 71% | 87% |
| Quality | 92.0% | 97.0% (not valid, see below) |
| Saves vs always Sonnet | 41% | 79% (not valid) |
| Calls per question | 2.29 | 1.00 |
| Avg / p95 latency | 5.3 s / 15.4 s | 5.2 s / 10.3 s |

- The cascade matched the offline estimate closely (offline: 94.0% quality, 43% savings). Live quality was 2 points lower because the live answers were new samples and math was checked on them exactly.
- The router's live numbers are almost the same as perfect routing (98%, 78%). That's data leakage: the router was trained on these same 199 questions. Its honest estimate is still the cross-validated one (about 65% savings at at most 3% loss).
- Latency: escalated cascade requests averaged 11.5 s (p95 17 s), because they make three calls in a row. Router-to-Haiku (one call) was slower than cascade-kept (two calls), because the router computes embeddings in Ollama on my laptop's CPU, which becomes a bottleneck under load.

Learned
- Never evaluate a model on the data it was trained on. A result that matches the theoretical best is a warning sign.
- The offline evaluation predicted live cascade behavior well, so the method is trustworthy.
- Live testing shows things offline evaluation can't: latency, and bottlenecks like local embeddings.

Next: build a held-out set of new questions to compare router and cascade fairly.

## Nov 3: Held-out test set, a fair router vs cascade comparison

What I did
- Built `scripts/build_holdout.py`: 100 new questions (50 math, 50 everyday) from the same datasets, with zero overlap with the main set. Never used for training the router or choosing thresholds.
- Answered them with Haiku and Sonnet, graded them (judge cost $0.12), and replayed both routing modes through the live gateway (0 errors).
- Named the files `holdout_*` so training scripts never pick them up by accident.

Results (100 held-out questions)
| Strategy | Quality | Saves | Calls | p95 latency |
|---|---|---|---|---|
| Always Sonnet | 94.0% | 0% | 1 | n/a |
| Router (threshold 0.95) | 94.0% | 64% | 1.00 | 13.1 s |
| Cascade (threshold 85) | 94.0% | 46% | 2.24 | 16.4 s |
| Always Haiku | 92.0% | 95% | 1 | n/a |
| Oracle | 97.0% | 84% | n/a | n/a |

- Haiku was good enough on 92/100 and beat Sonnet at math (46/50 vs 44/50).
- The router saved 64% on unseen questions, matching its cross-validated estimate (65%). Yesterday's 79% was leakage; cross-validation was the honest number.
- At equal quality, the router beat the cascade on cost, number of calls and worst-case latency.
- Always Haiku was only 2 points below (2 questions), so on easy questions like these, routing adds little over the cheap model alone.

Learned
- A held-out set is the only fair test. Cross-validation predicted it well; testing on training data didn't.
- The value of routing depends on how often the cheap model fails. When it rarely fails, the cheap model alone is a strong baseline.

Decision: switched the default to router mode (cascade stays as the automatic fallback when the router can't run).

## Nov 3 (later): Semantic cache, part 1

What I did
- Added a `cache` table in Postgres with pgvector: each answered question is saved with its 768-number embedding, plus an HNSW index for fast similarity search.
- Added `app/cache.py`: find the most similar saved question (cosine similarity), save new answers, count hits. If the database is down, the cache just misses.
- For `"tier": "auto"`, the gateway checks the cache first. A similarity of at least 0.95 returns the saved answer instantly (route `cache-hit`, $0, no model call).
- The router reuses the cache's embedding, so each question is embedded only once.
- Requests can skip the cache with `"use_cache": false`. Tests: 64 passed.

Tried it live
- "What is the capital of France?" → miss, answered and saved.
- Same question again → hit.
- "What's the capital city of France?" → hit (reused the France answer).
- "What is the capital of Germany?" → miss (correctly not confused with France), answered and saved.
- Cache hits cost $0 but took 3–4 s, about as long as a Haiku answer: the embedding is computed in Ollama on my laptop's CPU, which dominates the time. The database search itself is fast. A hosted embedding model would make hits near-instant.
- The first request took 24 s (cold start: Ollama loading the embedding model and the gateway loading the router).
