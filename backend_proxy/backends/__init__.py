"""Backends package."""
from .base import Backend, PassthroughResponse
from .factory import make_backend, register_backend
from .openai_compat import (
    LlamaCppBackend,
    OllamaOpenAIBackend,
    OpenAICloudBackend,
    OpenAICompatBackend,
    SGLangBackend,
    TGIBackend,
    VLLMBackend,
)
# Side-effect imports register additional backends with the factory.
from . import ollama_native as _ollama_native  # noqa: F401
from . import anthropic as _anthropic_be  # noqa: F401
from . import vertex as _vertex_be  # noqa: F401
from .ollama_native import OllamaNativeBackend
from .anthropic import AnthropicBackend
from .vertex import VertexClaudeBackend, VertexGeminiBackend
from .pool import BackendPool

__all__ = [
    "Backend",
    "PassthroughResponse",
    "BackendPool",
    "make_backend",
    "register_backend",
    "OpenAICompatBackend",
    "VLLMBackend",
    "TGIBackend",
    "LlamaCppBackend",
    "SGLangBackend",
    "OllamaOpenAIBackend",
    "OpenAICloudBackend",
    "OllamaNativeBackend",
    "AnthropicBackend",
    "VertexClaudeBackend",
    "VertexGeminiBackend",
]
