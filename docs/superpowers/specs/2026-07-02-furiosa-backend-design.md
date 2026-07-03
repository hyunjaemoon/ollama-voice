# Design: LLM Backend Switching (Ollama / Furiosa LLM)

**Date:** 2026-07-02
**Status:** Implemented (user was AFK during design review; decisions below are open to revision)

## Goal

Let the voice agent use [Furiosa LLM](https://developer.furiosa.ai/v2026.3.0/en/get_started/furiosa_llm.html)
as an alternative LLM backend to Ollama, selected at startup.

## Decisions

### Integration method: OpenAI-compatible HTTP client

`furiosa-llm serve <model>` exposes an OpenAI-compatible API (default
`http://localhost:8000`). The voice agent talks to it over HTTP using the
`/v1/chat/completions` endpoint.

Rejected alternative: importing the `furiosa_llm` Python SDK in-process. That
requires the RNGD NPU and the furiosa-llm package on the same machine as the
voice agent (Linux only), while the HTTP client works from any machine —
including a macOS dev laptop pointing at a remote RNGD host. It also keeps the
dependency footprint to a plain HTTP library (`requests`).

### Switch UX: startup CLI flag

```bash
# Default — unchanged behavior
python main.py

# Furiosa LLM on localhost
python main.py --backend furiosa

# Furiosa LLM on a remote RNGD host, explicit model
python main.py --backend furiosa --llm-url http://rngd-box:8000/v1 --model furiosa-ai/Qwen3-32B-FP8
```

Rejected alternative: runtime switching (voice command or hotkey
mid-conversation). More state to manage for a need that hasn't come up; the
process restarts in seconds. Can be added later without reworking this design.

### Per-backend model default

`--model` keeps working for both backends. Its argparse default becomes `None`
and is resolved per backend:

- `ollama` → `llama3.2` (unchanged)
- `furiosa` → `EMPTY` (the Furiosa server routes `"model": "EMPTY"` to
  whatever model it is serving, per the Furiosa docs)

This avoids silently sending the Ollama default model name to a Furiosa server.

## Components

- **`llm.py`** — adds `process_with_furiosa(text, model, base_url)` (POST
  `{base_url}/chat/completions`, extract `choices[0].message.content`, same
  error-message contract as the Ollama path) and a `generate_response(text,
  backend, model, llm_url)` dispatcher that `main.py` calls.
- **`models.py`** — `initialize_models()` gains `backend` and `llm_url`
  parameters. Verification is backend-aware: Ollama keeps the test generation;
  Furiosa does a GET `{base_url}/models` connectivity check.
- **`main.py`** — adds `--backend {ollama,furiosa}` (default `ollama`) and
  `--llm-url` (default `http://localhost:8000/v1`, used by the furiosa backend
  only), resolves the per-backend model default, and makes banner/status
  prints backend-aware.
- **`config.py`** — adds `DEFAULT_FURIOSA_URL`, `DEFAULT_OLLAMA_MODEL`,
  `DEFAULT_FURIOSA_MODEL`, and `LLM_REQUEST_TIMEOUT`.
- **`requirements.txt`** — adds `requests`.
- **`README.md`** — documents the new flags and a short Furiosa quick-start.

## Data flow

Unchanged except step 3 of the loop: `main.py` calls
`llm.generate_response(text, backend, model, llm_url)` instead of
`llm.process_with_ollama(text, model)`; the dispatcher routes to the right
backend function. Both paths return plain response text or a spoken-friendly
error string, so the TTS side needs no changes.

## Error handling

The Furiosa path mirrors the Ollama path's contract: any request failure,
non-2xx status, or unexpected payload shape logs the detail to the console and
returns "I'm sorry, I encountered an error processing your request." so the
conversation loop keeps running. Startup verification failure exits with a
hint to run `furiosa-llm serve <model>` (parallel to the existing
`ollama serve` hint).

## Testing

No test suite exists in the repo. Verification is manual: a stub
OpenAI-compatible server (scratchpad script implementing `/v1/models` and
`/v1/chat/completions`) exercises the furiosa request path end-to-end, and the
Ollama path is spot-checked against a live local Ollama if available.
