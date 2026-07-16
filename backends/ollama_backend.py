"""
Ollama backend.

Talks to the local Ollama service via the ``ollama`` Python package. Model
agnostic, so it runs quantized / MLX tags such as ``gemma4:12b-mlx`` (NVFP4 +
Apple MLX engine + multi-token prediction) with no code change.

Reasoning models emit chain-of-thought in a separate ``thinking`` field; this
backend disables thinking where supported and only ever yields the final
``response`` text, so downstream TTS never speaks the reasoning.
"""

from typing import Iterator, Optional

import ollama

from .base import BackendError, LLMBackend


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def load(self) -> None:
        try:
            self._ping()
        except Exception as e:  # noqa: BLE001 - surface any failure as BackendError
            raise BackendError(
                f"Ollama model '{self.model}' unavailable: {e}. "
                f"Is Ollama running (ollama serve) and pulled (ollama pull {self.model})?"
            ) from e

    def _ping(self) -> None:
        """Warm the model and verify availability with a 1-token generation."""
        try:
            ollama.generate(
                model=self.model, prompt="ping", think=False,
                options={"num_predict": 1},
            )
        except TypeError:
            # Older ollama client without the `think` kwarg.
            ollama.generate(
                model=self.model, prompt="ping", options={"num_predict": 1},
            )

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        # Try with thinking disabled first, then fall back for older clients.
        for think in (False, None):
            try:
                yield from self._stream_once(prompt, think, system)
                return
            except TypeError:
                continue  # `think` kwarg unsupported → retry without it
            except Exception as e:  # noqa: BLE001
                raise BackendError(f"Ollama generation failed: {e}") from e

    def _stream_once(
        self, prompt: str, think: Optional[bool], system: Optional[str]
    ) -> Iterator[str]:
        kwargs = dict(model=self.model, prompt=prompt, stream=True)
        if system:
            kwargs["system"] = system
        if think is not None:
            kwargs["think"] = think
        for chunk in ollama.generate(**kwargs):
            text = (
                chunk.get("response")
                if isinstance(chunk, dict)
                else getattr(chunk, "response", None)
            )
            if text:
                yield text

    def info(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model,
            "device": "ollama (MLX/Metal on Apple Silicon)",
            "dtype": "per-model",
        }
