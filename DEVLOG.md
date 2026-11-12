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
- Finding: on 20 sample questions, llama3.2:1b got 13/14 checkable questions right, and Claude got 14/14.
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

## Nov 4: Choosing the cache threshold from data (no threshold is safe)

What I did
- `scripts/make_cache_pairs.py`: Sonnet wrote a paraphrase (same answer) and a look-alike (one detail changed, different answer) for 60 main-set questions; 59 were usable (cost about $0.10).
- Spot-checked 10 pairs by hand: 10/10 correct.
- `scripts/eval_cache.py`: filled a pretend cache with the originals, looked up every paraphrase and look-alike, and swept the threshold from 0.70 to 1.00. Also tested a "same numbers" rule (`app/cache.py`).
- Tests: 68 passed.

Results
- Median similarity to the original: math paraphrases 0.969, math look-alikes 0.999; everyday paraphrases 0.951, everyday look-alikes 0.939.
- Look-alikes are often MORE similar than paraphrases: changing one number keeps every word, while rewording changes many words.
- No threshold works. At 0.98 the cache catches only 15% of paraphrases but still serves 56% of look-alikes. Five look-alikes scored 1.000.
- The same-numbers rule halves wrong hits (at 0.90: 85% → 37%), but blocks paraphrases that write numbers as words ("two hours" vs "2 hours") and misses changes written in words.

Learned
- Embeddings measure topic, not answer. "3 apples" and "4 apples" look identical to an embedding but need different answers.
- A cache that returns an answer to a different question is worse than no cache. Measuring wrong hits matters more than measuring hit rate.

Decision: semantic cache switched off (`CACHE_ENABLED=false`) until it has a verifier.

Tried it live
- Exact repeats (France, Germany): `cache-exact`, 0 calls, 0.2–0.6 s. Yesterday's semantic hits took 3–4 s; the exact layer skips the embedding.
- "Which city is the capital of France?": `cache-verified` (similarity 0.986, Haiku said YES), $0.000057, 6.8 s.
- "Capital of Spain": similarity 0.84, below 0.86, so no verifier; answered by Haiku via the router, $0.000010.
- For very short questions, the verifier check ($0.000057) costs more than answering with Haiku ($0.000010). The verifier pays off when the answer would be long or would go to Sonnet.

## Nov 9: Cache safety

What I did
- Time-sensitive and personal questions are never cached (not reused, not saved): a word rule for "today", "tomorrow", "now", "latest", "weather", "my", "are you"... Responses say why with `cache_skipped`.
- Questions with 2+ numbers count as self-contained (math word problems), so story words like "today" don't block them.
- 7-day expiry (`CACHE_TTL_HOURS=168`): older saved answers are ignored.
- Warm-up at startup: the embedding model and router load in the background when the gateway starts.
- `scripts/check_cacheable.py` measures how many real questions the rule refuses. Tests: 90 passed.

Results
- First version refused 6–7% of real questions, mostly false alarms in math word problems ("Last year there were 50 students...").
- With the 2+ numbers exemption: 2% (main set) and 4% (held-out set). What's left is genuinely personal ("why my laptop won't start?") or time-dependent ("walk today").
- The look-alike the verifier let through last time ("Are you going for a walk tomorrow?") is now refused.
- Known gaps: "Eve... now has only a quarter" is still refused (numbers written as words); a time-dependent question with 2+ numbers ("weather in 10 days at 3 pm") would be cacheable, limited by the 7-day expiry.
- First request after warm-up: 10192 ms (was about 24 s before).

Learned
- A safety rule should be measured like a model: count its false alarms on real data, not only whether it catches the bad case.

## Nov 10 (later): RAG, part 2: hybrid search and cited answers

**What I did**
- Hybrid search: Postgres full-text keyword search (`db/005_rag_keywords.sql`) plus embeddings, merged with Reciprocal Rank Fusion (RRF, k = 60).
- `POST /rag/ask`: retrieves 4 chunks, numbers them [1]–[4], and Haiku answers only from them with citations, or says "I don't know based on the project documents".
- `scripts/eval_hybrid.py` compares retrieval methods; `scripts/eval_rag.py` has Sonnet judge each answer as CORRECT and GROUNDED (about $0.20 a run).
- Made vector search the default after the evaluation. Tests: 109 passed.

**A stale test set nearly fooled me**
- Editing the README and DEVLOG shifted chunk numbers, so saved test questions pointed at the wrong chunks: vector recall@5 looked like 60% (some ranks 25+ of 49), and hybrid looked better.
- After regenerating the questions: vector recall@5 85%, and the hybrid result flipped. A test set has to be regenerated whenever the documents it tests change.

**Results (40 fresh questions)**
| Retrieval | recall@5 | Correct answers | Grounded |
|---|---|---|---|
| Vector | 85% | 85% | 88% |
| Hybrid (two runs) | 80% | 72% / 75% | 90% / 95% |
| Keyword only | 68% | n/a | n/a |

- Answer quality tracks retrieval: when the right chunk is found, the answer is almost always right.
- Hybrid didn't help here. The test questions were written to avoid the docs' exact words, which favors embeddings, and these docs are explanations rather than exact codes or names. With equal weights in RRF, the weaker keyword list pulled wrong chunks up.
- Two hybrid runs differed by 3 points: model answers vary between runs, so small differences are noise.
- Worst failure: borrowing a number from a neighbouring chunk (for example 98% from the oracle line), which sounds right but isn't.
- The README holds both old (llama-era) and new conclusions, so old sections can produce outdated answers.
- Cost: about $0.0003 per answer with Haiku.

**Tried it live**
- Plain `/ask`: "How much did the router save?" → Haiku had no idea what router I meant; "What is the capital of France?" → "Paris".
- `/rag/ask`: "How much did the router save on the held-out test?" → "64% at the same 94.0% quality [1][4], matching the cross-validated 65% [2][3]" ($0.0002, 3.7 s).
- `/rag/ask`: "...on new questions?" → missed the held-out sections (the docs say "held-out", not "new") but stayed honest: it said the sources don't give a valid figure instead of inventing one.
- `/rag/ask`: "What is the capital of France?" → "I don't know based on the project documents", even though three retrieved chunks mention France.

**Problems I hit**
- Deleted the `tier` line by accident while editing `RagRequest`, which broke the endpoint (3 failing tests). Check the lines around every edit.
- Two gateways were running at once, so requests reached the old code (500 error). Stopped all Python processes and started one.

**Learned**
- Retrieval quality decides answer quality.
- A test set must stay in sync with its documents, or it measures the wrong thing.
- Wording matters: query rewriting or adding synonyms to the docs would catch "new" vs "held-out".

**Decision:** vector search is the default for `/rag/ask`; hybrid stays available as an option.

**Next:** reliability (retries, fallbacks, timeouts, rate limiting).

## Nov 12: Reliability

What I did
- Retries in `app/models.py`: temporary failures (429 rate limited, 529 overloaded, 5xx, dropped connections) are retried up to 2 times with exponential backoff and jitter (about 0.5 s, then 1 s), or as long as the provider's Retry-After header says.
- No retries for permanent errors (401 wrong key, 404 unknown model, 400 bad request): they fail at once with a clear message.
- Timeouts: 5 s to connect, 60 s for an answer (was 120 s). Timeouts are not retried.
- Router fallback: if the chosen model fails even after retries, the other model answers (route `router-cheap-failed-strong` or `router-strong-failed-cheap`). Answers from Haiku standing in for Sonnet are never cached.
- Rate limiting (`app/limits.py`): at most 30 requests per minute per client (by X-Client-Id header or IP), sliding window; over the limit returns 429 with Retry-After.
- 13 tests that simulate failures (retries, Retry-After, give-up, timeouts, fallback both ways, the limiter). Tests: 122 passed.

Tried it live
- Rate limit set to 3: the 4th request within a minute got 429 with `retry-after: 31`. Requests spaced about 25 s apart were all allowed, because each one leaves the 60-second sliding window.
- Broken cheap model name: "What is the capital of Italy?" still answered by Sonnet, route `router-cheap-failed-strong`, 2 calls, 5.5 s. The 404 wasn't retried.

Problems I hit
- `NameError: Depends` after missing one import edit.
- The rate limit seemed broken at first: an old uvicorn process without the new code was still answering. Check `netstat` before starting the gateway.

Learned
- Retry only what can fix itself; retrying a wrong key or model name just wastes time.
- A fallback answer is fine to send but not to cache.
- The in-memory rate limiter works for one gateway process; several servers would need a shared store like Redis.

