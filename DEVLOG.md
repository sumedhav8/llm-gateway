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