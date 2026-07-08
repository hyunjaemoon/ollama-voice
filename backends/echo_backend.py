"""
Echo backend: a zero-dependency backend that streams back a canned response.

Used for instant dashboard/serving demos and end-to-end tests — it needs no
model download and no external server, so the serving layer can start and be
exercised immediately.
"""

import time
from typing import Iterator

from .base import LLMBackend


class EchoBackend(LLMBackend):
    name = "echo"

    def __init__(self, model: str = "echo-1", token_delay: float = 0.035):
        self.model = model
        self._token_delay = token_delay

    def load(self) -> None:
        # Nothing to load.
        return None

    def stream(self, prompt: str) -> Iterator[str]:
        reply = f"Echo backend received {len(prompt.split())} words. You said: {prompt}"
        for word in reply.split(" "):
            time.sleep(self._token_delay)  # simulate per-token decode latency
            yield word + " "

    def info(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model,
            "device": "cpu",
            "dtype": "n/a",
        }
