# llm-gateway

# LLM Gateway

A smart gateway that sits between an app and AI models and cuts LLM costs by:

- **Routing** each question to the cheapest model that can answer it well
- **Caching by meaning**, so repeated questions with different wording are answered instantly for free
- **Checking quality** and falling back to a stronger model when a cheap answer looks wrong
- **Tracking** cost, speed and quality on a live dashboard

## Status

Work in progress. Building step by step, with the results measured at each stage.

## Tech stack

Python, FastAPI, Ollama (local model), Postgres + pgvector, scikit-learn, Streamlit, Docker

## Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env
## Baseline (before routing and caching)

Every request goes to a single local model (llama3.2 via Ollama).

| Metric | Value |
|---|---|
| Requests | 14 |
| Success rate | 92.9% |
| Average latency | 49550 ms |
| p95 latency | 91810 ms |
| Cost | $0 (local model) |

Measured on 10/24/2026. Later milestones compare against these numbers.


## Usage

Start the gateway:

    uvicorn app.main:app --reload

Then open http://localhost:8000/docs to send a question to `/ask`.

## Running tests

    pytest -v