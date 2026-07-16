"""
Common interface for LLM backends.

Every backend (Ollama, an OpenAI-compatible server, an in-process HuggingFace
model, ...) implements the same small contract so callers — the voice agent,
the serving router, and the benchmark harness — can treat them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional


class BackendError(Exception):
    """Raised when a backend fails to load or to generate a response."""


class LLMBackend(ABC):
    """Abstract base class for a text-generation backend.

    Subclasses set ``name`` and ``model`` and implement ``load`` and ``stream``.
    ``generate`` is derived from ``stream`` by default so every backend supports
    both a one-shot and an incremental (token-level) interface; the incremental
    one is what lets the profiler measure time-to-first-token.
    """

    name: str = "base"
    model: str = ""

    @abstractmethod
    def load(self) -> None:
        """Initialize / verify the backend. Raise ``BackendError`` on failure."""

    @abstractmethod
    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Yield incremental text deltas for ``prompt``.

        ``system`` is an optional system prompt; backends that support one
        should apply it, others may ignore it.
        """

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Return the full response text (default: join the stream)."""
        return "".join(self.stream(prompt, system=system))

    def info(self) -> dict:
        """Return backend metadata for banners, reports, and the dashboard."""
        return {"backend": self.name, "model": self.model}
