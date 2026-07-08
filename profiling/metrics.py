"""
Metrics collection shared by the serving dashboard and the benchmark harness.

``MetricsCollector`` is thread-safe: request handlers (run in a threadpool by
Starlette) record into it while the WebSocket/stats endpoints read snapshots.
Each record is tagged with the serving backend so the dashboard can compare
backends side by side.
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

try:
    import psutil

    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover - psutil optional
    _HAVE_PSUTIL = False


@dataclass
class RequestRecord:
    ts: float
    backend: str
    model: str
    prompt: str
    ttft: Optional[float]  # time to first token, seconds
    e2e: Optional[float]  # end-to-end latency, seconds
    output_tokens: int
    status: str  # "ok" | "error"

    def preview(self, n: int = 64) -> str:
        p = " ".join(self.prompt.split())
        return p[:n] + ("…" if len(p) > n else "")


def _percentile(values, p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _round(x, digits: int = 4):
    return round(x, digits) if isinstance(x, (int, float)) else None


class MetricsCollector:
    def __init__(self, window: int = 200):
        self._lock = threading.Lock()
        self._records: "deque[RequestRecord]" = deque(maxlen=window)
        self._in_flight = 0
        self._total = 0
        self._errors = 0
        self._start = time.time()
        if _HAVE_PSUTIL:
            psutil.cpu_percent(interval=None)  # prime the rolling CPU counter

    def start_request(self) -> None:
        with self._lock:
            self._in_flight += 1

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._total += 1
            if rec.status != "ok":
                self._errors += 1
            self._in_flight = max(0, self._in_flight - 1)

    def snapshot(self) -> dict:
        with self._lock:
            records = list(self._records)
            total, errors, in_flight = self._total, self._errors, self._in_flight
            uptime = time.time() - self._start

        ok = [r for r in records if r.status == "ok"]
        ttfts = [r.ttft for r in ok if r.ttft is not None]
        e2es = [r.e2e for r in ok if r.e2e is not None]

        decode_rates = []
        for r in ok:
            if r.ttft is not None and r.e2e is not None and r.output_tokens > 0:
                decode_t = r.e2e - r.ttft
                if decode_t > 0:
                    decode_rates.append(r.output_tokens / decode_t)

        now = time.time()
        rps = len([r for r in records if now - r.ts <= 10]) / 10.0

        return {
            "uptime_s": round(uptime, 1),
            "total_requests": total,
            "errors": errors,
            "in_flight": in_flight,
            "requests_per_s": round(rps, 3),
            "ttft_p50": _round(_percentile(ttfts, 0.50)),
            "ttft_p95": _round(_percentile(ttfts, 0.95)),
            "e2e_p50": _round(_percentile(e2es, 0.50)),
            "e2e_p95": _round(_percentile(e2es, 0.95)),
            "tokens_per_s": _round(
                sum(decode_rates) / len(decode_rates) if decode_rates else None, 1
            ),
            "system": self._system(),
            "recent": [
                {
                    "ts": r.ts,
                    "backend": r.backend,
                    "model": r.model,
                    "prompt": r.preview(),
                    "ttft": _round(r.ttft),
                    "e2e": _round(r.e2e),
                    "tokens": r.output_tokens,
                    "status": r.status,
                }
                for r in reversed(records[-12:])
            ],
        }

    def _system(self) -> dict:
        if not _HAVE_PSUTIL:
            return {"cpu_percent": None, "mem_percent": None}
        vm = psutil.virtual_memory()
        return {
            "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
            "mem_percent": round(vm.percent, 1),
            "mem_used_gb": round(vm.used / 1e9, 2),
            "mem_total_gb": round(vm.total / 1e9, 2),
        }
