# Serving Monitor voice chat — design

Date: 2026-07-16

## Goal

Add voice chat to the Serving Monitor dashboard (`serve.py` + `web/dashboard.html`):
push-to-talk in the browser, fully local speech processing on the server,
replies spoken back through the browser tab.

## Decisions

- **Server-side speech processing** — reuse the project's stack: faster-whisper
  for STT, system TTS (macOS `say` / pyttsx3) for synthesis. Fully local, and
  voice-driven LLM calls flow through the existing metered completion path.
- **Push-to-talk + browser playback** — toggle mic button; server returns a WAV
  the browser plays. Works with a headless/remote server, no echo handling needed.
- **OpenAI-compatible endpoints** — the dashboard chains three calls per turn:
  1. `POST /v1/audio/transcriptions` (multipart WAV) → `{"text": ...}`
  2. `POST /v1/chat/completions` (existing, streaming — records metrics)
  3. `POST /v1/audio/speech` (`{"input": text}`) → `audio/wav`

## Server

- Whisper loads lazily on first transcription (keeps startup instant);
  `--whisper-model` CLI flag (default `base`). 503 with a clear message if
  `faster-whisper` isn't installed.
- Browser records 16 kHz mono PCM16 WAV in JS (no ffmpeg dependency); server
  decodes with the stdlib `wave` module, resamples via `np.interp` if needed.
  Too-short audio → 422; empty transcription → 422 ("didn't catch that").
- TTS: on macOS shell out to `say -o out.wav --data-format=LEI16@22050`;
  elsewhere pyttsx3 `save_to_file`. Serialized behind a lock (engines aren't
  thread-safe).
- STT/TTS timing returned in responses; the LLM leg is already in the metrics.

## Dashboard

New "Voice chat" card: mic toggle button (record → send), conversation log
showing the transcript and the streaming reply, `<audio>` playback of the
synthesized answer, and a per-turn latency line (STT · TTFT · E2E · TTS).
Graceful errors for mic-permission denial and unavailable STT.

## Dependencies / docs

- `requirements-serve.txt` += `faster-whisper`, `numpy`.
- README Serving Monitor section documents voice chat and the audio endpoints.

## Testing

- curl the two audio endpoints with a synthetic WAV against the echo backend.
- Browser verification of the full voice turn.
