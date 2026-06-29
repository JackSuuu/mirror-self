"""
Ollama LLM client with streaming support.
Uses the OpenAI-compatible /v1/chat/completions endpoint.
"""
from __future__ import annotations

import json
from typing import Generator, Iterator, Optional

import httpx


def _chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


def stream_chat(
    messages: list[dict],
    conf: dict,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """
    Stream chat completion tokens from Ollama.
    Yields string tokens as they arrive.
    """
    url = _chat_url(conf.get("ollama_base_url", "http://localhost:11434"))
    model = conf.get("llm_model", "qwen2.5:7b")

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "options": {
            "num_ctx": 8192,
        },
    }

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def complete_chat(
    messages: list[dict],
    conf: dict,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Non-streaming version — returns full response string."""
    return "".join(stream_chat(messages, conf, temperature, max_tokens))


def check_ollama(conf: dict) -> tuple[bool, str]:
    """Check if Ollama is reachable and the configured model is available."""
    base_url = conf.get("ollama_base_url", "http://localhost:11434")
    try:
        resp = httpx.get(base_url + "/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        llm_model = conf.get("llm_model", "qwen2.5:7b")
        embed_model = conf.get("embed_model", "nomic-embed-text")
        missing = []
        # Ollama model names can have :latest suffix
        def model_present(name: str) -> bool:
            return any(m.split(":")[0] == name.split(":")[0] for m in models)
        if not model_present(llm_model):
            missing.append(llm_model)
        if not model_present(embed_model):
            missing.append(embed_model)
        if missing:
            return False, f"Models not found: {', '.join(missing)}. Run: ollama pull {' '.join(missing)}"
        return True, "OK"
    except Exception as e:
        return False, f"Cannot reach Ollama at {base_url}: {e}"
