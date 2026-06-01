"""Factory: turn a ``BackendConfig`` into the right ``Backend`` subclass.

Adding a new backend type:
  1. Implement the class in this package.
  2. Register it in ``_BUILTINS`` here.
  3. Add the literal to ``BackendConfig.type``.
"""

from __future__ import annotations

from typing import Dict, Type

from ..config.schema import BackendConfig
from .base import Backend
from .openai_compat import (
    LlamaCppBackend,
    OllamaOpenAIBackend,
    OpenAICloudBackend,
    SGLangBackend,
    TGIBackend,
    VLLMBackend,
)


# Registered concrete classes per backend type. R3 will register
# ``OllamaNativeBackend`` for ``type: ollama``; we leave it out for now so
# import order is acyclic.
_BUILTINS: Dict[str, Type[Backend]] = {
    "vllm":          VLLMBackend,
    "tgi":           TGIBackend,
    "llamacpp":      LlamaCppBackend,
    "sglang":        SGLangBackend,
    "ollama-openai": OllamaOpenAIBackend,
    "openai":        OpenAICloudBackend,
    # "ollama" registered by backends.ollama_native (R3) via register_backend()
}


def register_backend(type_name: str, cls: Type[Backend]) -> None:
    """Register a Backend implementation. Used by R3 for the native Ollama
    class to avoid a circular import at package load."""
    _BUILTINS[type_name] = cls


def make_backend(cfg: BackendConfig) -> Backend:
    cls = _BUILTINS.get(cfg.type)
    if cls is None:
        raise ValueError(
            f"unknown backend type: {cfg.type!r} "
            f"(registered: {sorted(_BUILTINS)})"
        )
    return cls(cfg)
