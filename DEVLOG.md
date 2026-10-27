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