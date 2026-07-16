#!/usr/bin/env python3
"""
OpenAI-compatible serving layer with a live multi-backend router + monitoring
dashboard.

`serve.py` wraps a `BackendRouter` (a registry of configured backends + a
"current" pointer) behind an OpenAI-compatible HTTP API, and hosts a
self-contained dashboard that shows live inference telemetry and lets you switch
the active backend on the fly.

Run:
    python serve.py --backends echo,ollama --ollama-model gemma4:12b-mlx --port 8000

Then open http://localhost:8000 and drive it from the prompt box (or point the
voice agent / benchmark at http://localhost:8000/v1).
"""

import argparse
import asyncio
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool

import config
import tls
from backends import BackendError, create_backend
from profiling import MetricsCollector, RequestRecord

WEB_DIR = Path(__file__).parent / "web"


# ---------------------------------------------------------------------------
# Backend router
# ---------------------------------------------------------------------------
class BackendRouter:
    """Registry of configured backends with a lazily-loaded current pointer."""

    def __init__(self, specs: List[dict], default: Optional[str] = None):
        self._specs = {s["name"]: s for s in specs}
        self._order = [s["name"] for s in specs]
        self._instances = {}
        self._status = {n: "idle" for n in self._specs}   # idle|loading|ready|error
        self._error = {n: None for n in self._specs}
        self._current = default or (self._order[0] if self._order else None)
        self._lock = threading.RLock()

    def ensure_loaded(self, name: str):
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            if name not in self._specs:
                raise BackendError(f"No such backend: {name}")
            self._status[name] = "loading"
        # Load outside the lock — a large model can take seconds.
        try:
            backend = self._specs[name]["factory"]()
            backend.load()
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._status[name] = "error"
                self._error[name] = str(e)
            raise BackendError(str(e)) from e
        with self._lock:
            self._instances[name] = backend
            self._status[name] = "ready"
            self._error[name] = None
        return backend

    def select(self, name: str):
        if name not in self._specs:
            raise BackendError(f"No such backend: {name}")
        backend = self.ensure_loaded(name)  # may raise BackendError
        with self._lock:
            self._current = name
        return backend

    def current(self):
        with self._lock:
            name = self._current
        return name, self.ensure_loaded(name)

    def current_name(self) -> Optional[str]:
        with self._lock:
            return self._current

    def status(self) -> dict:
        with self._lock:
            return {
                "current": self._current,
                "backends": [
                    {
                        "name": n,
                        "label": self._specs[n].get("label", n),
                        "model": self._specs[n].get("model"),
                        "status": self._status[n],
                        "error": self._error[n],
                        "current": n == self._current,
                    }
                    for n in self._order
                ],
            }


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
app = FastAPI(title="ollama-voice serving layer")
METRICS = MetricsCollector()
ROUTER: Optional[BackendRouter] = None


def configure(backends: List[str], ollama_model: str, default: Optional[str] = None):
    """(Re)build the router from a list of backend names."""
    global ROUTER
    specs = []
    for name in backends:
        name = name.strip().lower()
        if name == "echo":
            specs.append({
                "name": "echo", "label": "Echo (demo)", "model": "echo-1",
                "factory": lambda: create_backend("echo"),
            })
        elif name == "ollama":
            specs.append({
                "name": "ollama", "label": f"Ollama · {ollama_model}",
                "model": ollama_model,
                "factory": (lambda m=ollama_model: create_backend("ollama", model=m)),
            })
        else:
            print(f"⚠️  Skipping unknown backend '{name}'")
    if not specs:
        specs.append({
            "name": "echo", "label": "Echo (demo)", "model": "echo-1",
            "factory": lambda: create_backend("echo"),
        })
    ROUTER = BackendRouter(specs, default=default)
    # Eagerly load the default only if it is cheap (echo) so startup is instant.
    if ROUTER.current_name() == "echo":
        try:
            ROUTER.ensure_loaded("echo")
        except BackendError:
            pass
    return ROUTER


# Import-time default so `uvicorn serve:app` works without args.
configure(["echo", "ollama"], "gemma4:12b-mlx", default="echo")


# ---------------------------------------------------------------------------
# OpenAI-compatible helpers
# ---------------------------------------------------------------------------
def _extract_prompt(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):  # OpenAI content-parts form
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                ).strip()
            return (content or "").strip()
    return "\n".join(
        m.get("content", "") for m in messages if isinstance(m.get("content"), str)
    ).strip()


def _extract_system(messages: List[dict]) -> str:
    """Join any client-provided system messages; default to the voice-agent
    conversational prompt when none are given."""
    parts = [
        m.get("content", "") for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    joined = "\n".join(p for p in parts if p).strip()
    return joined or config.SYSTEM_PROMPT


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _stream_completion(
    name: str, backend, model: str, prompt: str, system: str
) -> Iterator[str]:
    """Sync generator producing OpenAI SSE chunks while recording metrics."""
    cmpl_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    base = {"id": cmpl_id, "object": "chat.completion.chunk",
            "created": created, "model": model}
    METRICS.start_request()
    start = time.time()
    ttft: Optional[float] = None
    ntok = 0
    status = "ok"
    yield _sse({**base, "choices": [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    try:
        for delta in backend.stream(prompt, system=system):
            if ttft is None:
                ttft = time.time() - start
            ntok += 1
            yield _sse({**base, "choices": [
                {"index": 0, "delta": {"content": delta}, "finish_reason": None}]})
        yield _sse({**base, "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]})
    except Exception as e:  # noqa: BLE001
        status = "error"
        yield _sse({**base, "choices": [
            {"index": 0, "delta": {"content": f"\n[error: {e}]"},
             "finish_reason": "stop"}]})
    finally:
        yield "data: [DONE]\n\n"
        METRICS.record(RequestRecord(
            ts=start, backend=name, model=model, prompt=prompt,
            ttft=ttft, e2e=time.time() - start, output_tokens=ntok, status=status))


def _blocking_completion(
    name: str, backend, model: str, prompt: str, system: str
) -> dict:
    """Run a full (non-streaming) completion, recording metrics. Sync/blocking."""
    METRICS.start_request()
    start = time.time()
    ttft: Optional[float] = None
    parts: List[str] = []
    status = "ok"
    try:
        for delta in backend.stream(prompt, system=system):
            if ttft is None:
                ttft = time.time() - start
            parts.append(delta)
    except Exception as e:  # noqa: BLE001
        status = "error"
        METRICS.record(RequestRecord(
            ts=start, backend=name, model=model, prompt=prompt, ttft=ttft,
            e2e=time.time() - start, output_tokens=len(parts), status=status))
        raise BackendError(str(e)) from e
    text = "".join(parts)
    METRICS.record(RequestRecord(
        ts=start, backend=name, model=model, prompt=prompt, ttft=ttft,
        e2e=time.time() - start, output_tokens=len(parts), status=status))
    return {
        "id": f"chatcmpl-{int(start * 1000)}", "object": "chat.completion",
        "created": int(start), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt.split()),
                  "completion_tokens": len(parts),
                  "total_tokens": len(prompt.split()) + len(parts)},
    }


# ---------------------------------------------------------------------------
# Voice: server-side STT (faster-whisper) + TTS (say / pyttsx3)
# ---------------------------------------------------------------------------
VOICE_SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "base"
_whisper = None
_whisper_lock = threading.Lock()
_tts_lock = threading.Lock()


def _get_whisper():
    """Lazily load faster-whisper so serve.py startup stays instant."""
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            print(f"🎙  Loading Whisper '{WHISPER_MODEL_SIZE}' (first voice request)…")
            _whisper = WhisperModel(WHISPER_MODEL_SIZE)
        return _whisper


def _decode_wav(data: bytes) -> np.ndarray:
    """Decode a PCM16 WAV upload to mono float32 at VOICE_SAMPLE_RATE."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
    if rate != VOICE_SAMPLE_RATE and len(audio) > 1:
        n = int(len(audio) * VOICE_SAMPLE_RATE / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


def _transcribe(audio: np.ndarray) -> str:
    model = _get_whisper()
    try:
        segments, _ = model.transcribe(audio, beam_size=5, vad_filter=True)
    except Exception:  # noqa: BLE001 - VAD can fail on some builds
        segments, _ = model.transcribe(audio, beam_size=5, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


def _synthesize_wav(text: str) -> bytes:
    """Render text to a WAV file and return its bytes. Serialized: TTS engines
    are not thread-safe."""
    with _tts_lock:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "speech.wav"
            if sys.platform == "darwin":
                subprocess.run(
                    ["say", "-o", str(path), "--data-format=LEI16@22050", text],
                    check=True,
                )
            else:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, str(path))
                engine.runAndWait()
            return path.read_bytes()


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(...)):
    data = await file.read()
    try:
        audio = _decode_wav(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=400, content={
            "error": {"message": f"could not decode audio: {e}"}})
    if len(audio) < VOICE_SAMPLE_RATE // 10:  # < 0.1 s
        return JSONResponse(status_code=422, content={
            "error": {"message": "audio too short"}})
    start = time.time()
    try:
        text = await run_in_threadpool(_transcribe, audio)
    except ImportError:
        return JSONResponse(status_code=503, content={"error": {"message":
            "faster-whisper is not installed — pip install faster-whisper"}})
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={
            "error": {"message": f"transcription failed: {e}"}})
    if not text:
        return JSONResponse(status_code=422, content={
            "error": {"message": "didn't catch that — no speech detected"}})
    return {"text": text, "stt_ms": round((time.time() - start) * 1000)}


@app.post("/v1/audio/speech")
async def audio_speech(req: Request):
    body = await req.json()
    text = (body.get("input") or "").strip()
    if not text:
        return JSONResponse(status_code=400,
                            content={"error": {"message": "no input text"}})
    start = time.time()
    try:
        wav = await run_in_threadpool(_synthesize_wav, text)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={
            "error": {"message": f"speech synthesis failed: {e}"}})
    return Response(content=wav, media_type="audio/wav", headers={
        "X-TTS-Ms": str(round((time.time() - start) * 1000))})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "current": ROUTER.current_name()}


@app.get("/v1/models")
async def list_models():
    st = ROUTER.status()
    return {"object": "list", "data": [
        {"id": b["model"] or b["name"], "object": "model", "owned_by": b["name"]}
        for b in st["backends"]
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    prompt = _extract_prompt(messages)
    system = _extract_system(messages)
    stream = bool(body.get("stream", False))
    if not prompt:
        return JSONResponse(status_code=400,
                            content={"error": {"message": "no user message"}})
    try:
        name, backend = await run_in_threadpool(ROUTER.current)
    except BackendError as e:
        return JSONResponse(status_code=503, content={
            "error": {"message": str(e), "type": "backend_unavailable"}})
    model = getattr(backend, "model", name)
    if stream:
        return StreamingResponse(
            _stream_completion(name, backend, model, prompt, system),
            media_type="text/event-stream")
    try:
        payload = await run_in_threadpool(
            _blocking_completion, name, backend, model, prompt, system)
    except BackendError as e:
        return JSONResponse(status_code=500, content={"error": {"message": str(e)}})
    return payload


@app.get("/api/backends")
async def api_backends():
    return ROUTER.status()


@app.post("/api/backends/select")
async def api_select(req: Request):
    body = await req.json()
    name = (body.get("name") or "").strip().lower()
    try:
        await run_in_threadpool(ROUTER.select, name)
    except BackendError as e:
        return JSONResponse(status_code=503,
                            content={"ok": False, "error": str(e), **ROUTER.status()})
    return {"ok": True, **ROUTER.status()}


def _current_info() -> dict:
    name = ROUTER.current_name()
    inst = ROUTER._instances.get(name)  # read-only peek
    if inst is not None:
        return inst.info()
    st = {b["name"]: b for b in ROUTER.status()["backends"]}.get(name, {})
    return {"backend": name, "model": st.get("model")}


def _stats_payload() -> dict:
    return {
        "metrics": METRICS.snapshot(),
        "router": ROUTER.status(),
        "current_info": _current_info(),
        "server_time": time.time(),
    }


@app.get("/api/stats")
async def api_stats():
    return _stats_payload()


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_stats_payload())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - client went away
        return


@app.get("/metrics")
async def prometheus():
    snap = METRICS.snapshot()
    lines = []

    def gauge(metric, value, help_text):
        lines.append(f"# HELP {metric} {help_text}")
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric} {value if value is not None else 'NaN'}")

    gauge("llm_requests_total", snap["total_requests"], "Total requests served")
    gauge("llm_errors_total", snap["errors"], "Total failed requests")
    gauge("llm_in_flight", snap["in_flight"], "In-flight requests")
    gauge("llm_requests_per_second", snap["requests_per_s"], "Requests/sec (10s)")
    gauge("llm_ttft_p50_seconds", snap["ttft_p50"], "Time to first token p50")
    gauge("llm_ttft_p95_seconds", snap["ttft_p95"], "Time to first token p95")
    gauge("llm_e2e_p50_seconds", snap["e2e_p50"], "End-to-end latency p50")
    gauge("llm_e2e_p95_seconds", snap["e2e_p95"], "End-to-end latency p95")
    gauge("llm_decode_tokens_per_second", snap["tokens_per_s"], "Mean decode tok/s")
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/")
async def dashboard():
    return FileResponse(WEB_DIR / "dashboard.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible serving layer + dashboard")
    parser.add_argument("--backends", default="echo,ollama",
                        help="Comma list of backends to register (default: echo,ollama)")
    parser.add_argument("--ollama-model", default="gemma4:12b-mlx",
                        help="Model for the ollama backend (default: gemma4:12b-mlx)")
    parser.add_argument("--default", default=None,
                        help="Backend to start current (default: first registered)")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model size or local model directory for "
                             "voice chat (default: base)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl", action="store_true",
                        help="Serve HTTPS with a cached self-signed cert "
                             "(browsers require HTTPS for mic access away "
                             "from localhost)")
    parser.add_argument("--lan", action="store_true",
                        help="Shortcut for --host 0.0.0.0 --ssl: reach the "
                             "dashboard from a phone on the same Wi-Fi")
    args = parser.parse_args()

    if args.lan:
        args.host = "0.0.0.0"
        args.ssl = True

    global WHISPER_MODEL_SIZE
    WHISPER_MODEL_SIZE = args.whisper_model
    configure([b for b in args.backends.split(",") if b.strip()],
              args.ollama_model, default=args.default)

    import uvicorn
    ssl_kwargs = {}
    scheme = "http"
    if args.ssl:
        cert, key = tls.ensure_self_signed_cert()
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"

    if args.host == "0.0.0.0":
        urls = [f"{scheme}://127.0.0.1:{args.port}",
                f"{scheme}://{tls.get_lan_ip()}:{args.port}"]
    else:
        urls = [f"{scheme}://{args.host}:{args.port}"]
    print(f"\n🚀 Serving dashboard at {'  and  '.join(urls)}")
    print(f"   Backends: {args.backends}  (current: {ROUTER.current_name()})")
    print(f"   OpenAI endpoint: {urls[-1]}/v1")
    if args.ssl:
        print("   📱 On your phone (same Wi-Fi): open the last URL above and "
              "accept the certificate\n      warning (Advanced → Proceed) — "
              "HTTPS is what enables microphone access.")
    print()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                **ssl_kwargs)


if __name__ == "__main__":
    main()
