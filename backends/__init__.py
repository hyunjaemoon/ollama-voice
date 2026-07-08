"""
LLM backend package.

Exposes the common interface (``LLMBackend``, ``BackendError``) and a
``create_backend`` factory that maps a backend name + options to an instance.
This is the single source of truth used by the voice agent, the serving router,
and the benchmark harness.

Note: heavier backends (HuggingFace/PyTorch, OpenAI-compatible clients) are
imported lazily inside the factory so importing this package does not require
torch/openai to be installed.
"""

from .base import BackendError, LLMBackend
from .echo_backend import EchoBackend
from .ollama_backend import OllamaBackend

__all__ = ["LLMBackend", "BackendError", "create_backend", "AVAILABLE_BACKENDS"]

# Backends registered in this slice. More (hf, vllm, sglang, furiosa) are added
# as they land; the factory raises a clear error for anything unregistered.
AVAILABLE_BACKENDS = ("echo", "ollama")


def create_backend(name: str, **opts) -> LLMBackend:
    """Create a backend instance by name.

    Args:
        name: Backend name (e.g. "echo", "ollama").
        **opts: Backend-specific options (e.g. model="gemma4:12b-mlx").

    Raises:
        BackendError: If the backend name is not recognized.
    """
    key = name.lower()
    if key == "echo":
        return EchoBackend(**{k: v for k, v in opts.items() if k in ("model", "token_delay")})
    if key == "ollama":
        return OllamaBackend(**{k: v for k, v in opts.items() if k in ("model",)})
    raise BackendError(
        f"Unknown backend '{name}'. Available: {', '.join(AVAILABLE_BACKENDS)}."
    )
