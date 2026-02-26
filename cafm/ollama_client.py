"""Ollama client — wraps the ollama Python library for model interaction."""

from __future__ import annotations

import sys
from typing import Generator

import ollama


def list_local_models() -> list[str]:
    """Return sorted list of model names available in the local Ollama instance."""
    try:
        response = ollama.list()
        return sorted(m.model for m in response.models)
    except Exception as exc:
        print(f"[ERROR] Cannot reach Ollama server: {exc}")
        return []


def validate_models(required: list[str]) -> tuple[list[str], list[str]]:
    """Check which of *required* models are available locally.

    Returns (available, missing).
    """
    local = list_local_models()
    # Normalise: ollama.list() may return tags like "llama3.2:latest"
    local_base = {m.split(":")[0]: m for m in local}
    available: list[str] = []
    missing: list[str] = []
    for model in required:
        base = model.split(":")[0]
        if model in local or base in local_base:
            available.append(model)
        else:
            missing.append(model)
    return available, missing


def pull_model(model: str) -> bool:
    """Attempt to pull a model via Ollama. Returns True on success."""
    try:
        print(f"  Pulling {model} …")
        ollama.pull(model)
        return True
    except Exception as exc:
        print(f"  [ERROR] Failed to pull {model}: {exc}")
        return False


def chat_stream(
    model: str,
    messages: list[dict[str, str]],
    context_limit: int = 4096,
) -> Generator[str, None, None]:
    """Stream a chat completion token-by-token.

    Yields content chunks as they arrive.
    """
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            options={"num_ctx": context_limit},
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as exc:
        yield f"\n[ERROR] Model {model} failed: {exc}\n"


def chat_sync(
    model: str,
    messages: list[dict[str, str]],
    context_limit: int = 4096,
) -> str:
    """Non-streaming chat completion. Returns full content string."""
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            stream=False,
            options={"num_ctx": context_limit},
        )
        return resp.get("message", {}).get("content", "")
    except Exception as exc:
        return f"[ERROR] Model {model} failed: {exc}"
