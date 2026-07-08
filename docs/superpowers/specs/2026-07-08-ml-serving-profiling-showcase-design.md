# Design: ML Serving, Optimization & Profiling Showcase

**Date:** 2026-07-08
**Status:** Awaiting user review
**Builds on:** `2026-07-02-furiosa-backend-design.md`

## Goal

Extend the local voice agent so the repository demonstrates, with running code
on Apple Silicon, the skill set: *AI model implementation, optimization, and
deployment using PyTorch/HuggingFace, serving frameworks (vLLM, SGLang), and
performance profiling tools* — plus a web dashboard for monitoring the
inference service and an end-to-end test.

Every JD keyword maps to a real, runnable piece of the repo:

| JD skill | Where it lives | Runs live on Mac? |
|---|---|---|
| Implement (PyTorch/HF) | `backends/hf_backend.py` — in-process `transformers`+`torch` generation on MPS | Yes |
| Optimize | dtype / `torch.compile` / weight-only quant flags, with measured before/after | Yes |
| Deploy / serve (vLLM, SGLang) | `backends/openai_backend.py` client (vllm/sglang/furiosa presets) + `serve.py` self-built OpenAI server | Client + own server: yes. Real vLLM/SGLang: docs only (no GPU) |
| Profile | `benchmark.py` + `profiling/` — TTFT, ITL, tokens/s, p50/p95, concurrency, `torch.profiler` | Yes |
| Monitor | `serve.py` dashboard + `/ws` + `/metrics` | Yes |

## Constraints & honesty policy

- Target hardware for live demo: **Apple Silicon Mac only** (MPS/CPU). No CUDA
  GPU, no Furiosa NPU.
- vLLM has no real Metal/MPS path and SGLang is CUDA-only, so **neither runs
  real GPU inference on this machine**. We ship genuine OpenAI-compatible
  *client* backends for them, validated live against a local OpenAI-compatible
  server we *can* run (our own `serve.py`, or Ollama's `:11434/v1`), with
  documented GPU launch commands. **No fabricated GPU benchmark numbers.** The
  README states this plainly.
- The self-built `serve.py` is described as a minimal reference implementation
  of the OpenAI serving interface; vLLM/SGLang/Furiosa LLM are production
  implementations of the same API.

## Decisions

### D1. LLM layer becomes a `backends/` package; `llm.py` is removed

Replace the flat `llm.py` with a package exposing one small, testable unit per
backend behind a common interface. `main.py` and `models.py` are updated to use
the factory. Rationale: the existing spec anticipated growth; a clean interface
is the clearest way to demonstrate implementation/serving/deployment separation
to a reviewer, and keeps each file small enough to reason about.

### D2. Backend interface (`backends/base.py`)

```python
class LLMBackend(ABC):
    name: str          # "ollama" | "furiosa" | "vllm" | "sglang" | "hf" | "echo"
    model: str
    def load(self) -> None: ...            # init/verify; raises BackendError on failure
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...   # yields text deltas
    def info(self) -> dict: ...            # {backend, model, device, dtype, url}
```

- Default `generate()` = `"".join(self.stream(prompt))`; backends may override
  for a native non-streaming path.
- `stream()` yields incremental text so the profiler can measure TTFT and
  inter-token latency for every backend.
- **Error contract change (intentional):** backends *raise* `BackendError` on
  failure instead of returning a friendly string. Callers decide handling:
  `main.py` catches and speaks `LLM_ERROR_MESSAGE` (today's UX preserved);
  `serve.py` returns an OpenAI-style error JSON; `benchmark.py` records the
  request as failed. This removes error-swallowing from the LLM layer.

### D3. Backends

- **`ollama_backend.py`** — wraps current Ollama logic; adds streaming via
  `ollama.generate(..., stream=True)`. `load()` = test generation (as today).
- **`openai_backend.py`** — uses the official `openai` Python client
  (`OpenAI(base_url=..., api_key="EMPTY")`). Named presets set default URL/model:
  - `vllm` → `http://localhost:8000/v1`
  - `sglang` → `http://localhost:30000/v1`
  - `furiosa` → `http://localhost:8000/v1`, model `EMPTY`
  `generate()`/`stream()` call `chat.completions.create(...)`; `load()` = a
  `client.models.list()` connectivity check. **One client serves all three
  frameworks** — the vLLM/SGLang serving evidence.
- **`hf_backend.py`** — in-process PyTorch + HuggingFace:
  - `AutoTokenizer` + `AutoModelForCausalLM.from_pretrained(model, torch_dtype=…)`.
  - Device auto-detect `mps > cuda > cpu`, override with `--device`.
  - Prompt built with `tokenizer.apply_chat_template([...], add_generation_prompt=True)`.
  - `stream()` via `TextIteratorStreamer` + a generation `Thread` → real TTFT.
  - `generate()` via `model.generate(...)`, decoding only new tokens.
  - `load()` downloads/loads weights, applies optimizations, moves to device,
    runs a short warmup generation (fair first-token timing; primes compile/MPS).
  - Optimization knobs (see D5).
- **`echo_backend.py`** — a zero-dependency backend that streams a canned
  response without loading a model. Used by e2e tests and for instant
  dashboard/serving demos (`serve.py --backend echo`) with no model download.

### D4. Factory (`backends/__init__.py`)

`create_backend(spec) -> LLMBackend` maps a backend name + options
(model, url, device, dtype, compile, quantize, generation params) to a backend
instance. Single source of truth used by `main.py`, `serve.py`, and
`benchmark.py`.

### D5. Optimization (measured on this Mac)

HF-backend flags, each with before/after numbers captured by `benchmark.py`:

- `--dtype {auto,fp32,fp16,bf16}` (auto → bf16 on mps/cuda, fp32 on cpu).
- `--compile` → `torch.compile(model)`; guarded — warn and fall back if the MPS
  backend can't compile a given op.
- `--quantize {none,int8,int4}` → weight-only quant via `optimum-quanto`
  (`quantize` + `freeze`), which supports CPU/MPS; degrades gracefully with a
  warning if unsupported.

`docs/OPTIMIZATION.md` explains each technique, its measured impact, and how it
maps to dedicated inference hardware (Furiosa RNGD, GPUs).

### D6. Serving layer (`serve.py`, FastAPI + uvicorn)

Wraps one loaded backend (default `hf`; `echo` for demos) and exposes:

- `GET  /v1/models` — lists the served model.
- `POST /v1/chat/completions` — `stream` true/false; non-stream returns
  OpenAI-shaped JSON, stream returns SSE chunks ending with `[DONE]`.
- `GET  /` — the monitoring dashboard (serves `web/dashboard.html`).
- `WS   /ws` — pushes a metrics snapshot ~1×/sec for the live dashboard.
- `GET  /api/stats` — JSON metrics snapshot (polling fallback for `/ws`).
- `GET  /metrics` — Prometheus text exposition of the same metrics.
- `GET  /health` — liveness/readiness.

Run: `python serve.py --model … --port 8000 [--backend hf|echo] [--dtype/--compile/--quantize]`.
Every request is timed and recorded into the shared `MetricsCollector` (D7).
This lets `main.py --backend vllm --llm-url http://localhost:8000/v1` (or the
benchmark) drive **our own server live on the Mac**.

### D7. Metrics core (`profiling/metrics.py`) — shared by benchmark + dashboard

- `RequestRecord`: timestamp, prompt preview, ttft, inter-token latencies,
  output tokens, e2e latency, status (ok/error).
- `MetricsCollector`: thread-safe; keeps a rolling window of records and
  in-flight count; computes aggregates — request count, error count, TTFT
  p50/p95, decode tokens/sec, e2e latency p50/p90/p95, requests/sec — plus
  system stats via `psutil` (CPU%, RAM; MPS memory via
  `torch.mps.current_allocated_memory()` when torch is present).
- Token counting: exact from the HF tokenizer or the OpenAI `usage` field when
  available; whitespace estimate as a labelled fallback.

### D8. Profiler (`benchmark.py` + `profiling/report.py`)

CLI over the factory:

```
python benchmark.py --backend hf --model … [--dtype/--compile/--quantize] \
    [--prompts FILE] [--num-requests N] [--concurrency C] [--warmup W] \
    [--compare "cfgA,cfgB,…"] [--profile-torch] [--output docs/benchmarks/NAME]
```

- Metrics per config: TTFT, ITL (mean/p50/p95), decode tokens/sec, e2e latency
  (p50/p90/p95).
- **Concurrency sweep** (HTTP backends): `ThreadPoolExecutor` fires C concurrent
  requests, reports aggregate throughput + latency. In-process HF is
  single-stream — this is stated, not faked.
- `--compare` tabulates multiple configs (e.g. fp32 vs bf16 vs compiled vs int8).
- `--profile-torch` wraps an HF generate in `torch.profiler.profile` (CPU always;
  MPS activity when available), exporting an op-level table + Chrome trace JSON.
- `report.py` prints a console table (`rich`/`tabulate`) and writes CSV + JSON,
  plus an optional matplotlib chart, into `docs/benchmarks/`.

### D9. Monitoring dashboard (`web/dashboard.html`)

Self-contained single page — inline CSS/JS, **no CDN, no build step** (matches
the repo's fully-local ethos; charts hand-rolled on `<canvas>`). Connects to
`/ws` (falls back to polling `/api/stats`). Shows:

- Header: service name, backend/model/device/dtype, uptime, health.
- KPI tiles: total requests, in-flight, TTFT p50/p95, decode tokens/sec, e2e
  latency p50/p95, errors.
- Live rolling charts: tokens/sec, latency, requests/sec.
- System: CPU%, RAM (MPS mem when available).
- Recent-requests table: time, prompt preview, output tokens, TTFT, latency,
  status.

### D10. Dependencies split

- `requirements.txt` (unchanged core, voice agent runs without torch):
  `faster-whisper, ollama, pyttsx3, requests, sounddevice, numpy`.
- `requirements-ml.txt` (new): `torch, transformers, accelerate, openai,
  fastapi, uvicorn, psutil, optimum-quanto, matplotlib, rich` (+ `pytest,
  httpx` for tests). Documented as needed only for the `hf` backend, serving,
  and benchmarking.

## Components (file-by-file)

- **`backends/base.py`** ➕ `LLMBackend` ABC + `BackendError`.
- **`backends/__init__.py`** ➕ `create_backend()` factory + registry.
- **`backends/ollama_backend.py`** ➕ (migrated from `llm.py`, + streaming).
- **`backends/openai_backend.py`** ➕ vllm/sglang/furiosa presets.
- **`backends/hf_backend.py`** ➕ PyTorch/HF + optimization.
- **`backends/echo_backend.py`** ➕ test/demo backend.
- **`serve.py`** ➕ FastAPI server + endpoints + dashboard wiring.
- **`web/dashboard.html`** ➕ monitoring UI.
- **`profiling/metrics.py`** ➕ `MetricsCollector`, `RequestRecord`.
- **`profiling/report.py`** ➕ table/CSV/JSON/chart output.
- **`benchmark.py`** ➕ profiling CLI.
- **`main.py`** ✎ `--backend {ollama,furiosa,vllm,sglang,hf}` + `--device/
  --dtype/--compile/--quantize/--max-new-tokens`; builds backend via factory;
  loop calls `backend.generate()`; banner from `backend.info()`.
- **`models.py`** ✎ keeps Whisper + TTS init; LLM verification removed (now
  `backend.load()`).
- **`config.py`** ✎ backend presets/URLs, `DEFAULT_HF_MODEL=
  "Qwen/Qwen2.5-0.5B-Instruct"`, dtype/device defaults, `LLM_MAX_NEW_TOKENS`,
  `LLM_TEMPERATURE`, serving host/port, benchmark defaults.
- **`llm.py`** ✖ removed.
- **`requirements-ml.txt`** ➕; **`requirements.txt`** unchanged.
- **`tests/`** ➕ `test_backends.py`, `test_metrics.py`, `test_e2e.py`.
- **`README.md`** ✎ Backends / Optimization / Serving / Benchmarking /
  Monitoring sections + honest hardware caveats; updated project structure.
- **`docs/OPTIMIZATION.md`** ➕; **`docs/benchmarks/`** ➕ generated artifacts.

## Data flow

- **Voice agent:** mic → Whisper → `backend.generate(prompt)` → TTS. (Loop shape
  unchanged; only the call target is now a backend object.)
- **Serving:** client (`openai_backend` / curl / benchmark) → `serve.py`
  `/v1/chat/completions` → `backend.stream|generate` → `MetricsCollector.record`
  → response. `/ws` streams collector snapshots to the dashboard each second.
- **Benchmark:** `create_backend` → warmup → run prompts via `stream()` →
  `metrics.record` → `report` writes artifacts.

## Error handling

- Backends raise `BackendError` on load/generate failure.
- `main.py`: `load()` failure exits with a backend-specific hint (`ollama serve`
  / `furiosa-llm serve` / `vllm serve` / SGLang launch / HF model id); per-turn
  `generate()` failure is caught and spoken as `LLM_ERROR_MESSAGE`, loop
  continues.
- `serve.py`: request errors → OpenAI-style error JSON (HTTP 500) and recorded
  as failed in metrics; server stays up.
- `benchmark.py`: per-request failures counted; the run completes and reports
  the error rate.

## Testing

- **Unit** (`test_metrics.py`): percentile / throughput math on known inputs.
- **Unit** (`test_backends.py`): factory returns the right backend per name;
  `openai_backend` parses stream + non-stream responses (mocked transport);
  `BackendError` raised on failure. HF/torch tests are `skipif` torch absent, so
  CI stays light and no model is downloaded.
- **End-to-end** (`test_e2e.py`): a fixture starts a real `uvicorn` server on an
  ephemeral port with the **echo** backend, waits for `/health`, then drives it
  through the actual `openai_backend` client — asserting non-stream content,
  streamed deltas, `/v1/models`, `/api/stats` (metrics incremented), and
  `/metrics` text. Genuine client→server→response path, no model download.
- **Manual verification (live on Mac):** run `serve.py --backend hf` + open the
  dashboard; run `benchmark.py --compare fp32,bf16,compiled` and commit the real
  artifacts to `docs/benchmarks/`; run `main.py --backend hf`; point
  `main.py --backend vllm --llm-url http://localhost:8000/v1` at the local
  server and hold a voice conversation while watching the dashboard.

## Out of scope (YAGNI)

- Real GPU/NPU execution of vLLM/SGLang/Furiosa (no hardware).
- Runtime backend switching mid-conversation (restart is seconds).
- Multi-model routing, auth, or a production observability stack (Grafana etc.).
- Incremental/streaming TTS in the voice loop (backend streaming exists for the
  profiler; wiring it into TTS is a separate change).
