"""
Configuration management for mirror-self.
Config:    ~/.config/mirror-self/config.json
ChromaDB:  ~/.local/share/mirror-self/chroma
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_CONFIG_DIR  = Path.home() / ".config"  / "mirror-self"
_DATA_DIR    = Path.home() / ".local" / "share" / "mirror-self"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS: dict = {
    # ── Journal ──────────────────────────────────────
    "journal_path":        "",
    # ── Identity (shown to the LLM as context) ───────
    "user_name":           "User",
    "language_hint":       "auto",   # "auto" | "Chinese" | "English" | "mixed" | …
    "journal_description": "",       # e.g. "personal diary written in Chinese and English"
    # ── Ollama ───────────────────────────────────────
    "ollama_base_url":     "http://localhost:11434",
    "llm_model":           "qwen2.5:latest",
    "embed_model":         "nomic-embed-text",
    # ── Retrieval ─────────────────────────────────────
    "chroma_path":         str(_DATA_DIR / "chroma"),
    "top_k":               6,
    "min_year_spread":     2,
}


def config_dir() -> Path:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _CONFIG_DIR


def data_dir() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _DATA_DIR


def load() -> dict:
    if not _CONFIG_FILE.exists():
        return dict(_DEFAULTS)
    with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
        saved = json.load(f)
    cfg = dict(_DEFAULTS)
    cfg.update(saved)
    return cfg


def save(cfg: dict) -> None:
    config_dir()
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def set_value(key: str, value) -> None:
    cfg = load()
    cfg[key] = value
    save(cfg)


def get_value(key: str, default=None):
    return load().get(key, default)


def chroma_path() -> Path:
    p = Path(get_value("chroma_path", str(_DATA_DIR / "chroma")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_journal_path() -> Path:
    path_str = get_value("journal_path", "")
    if not path_str:
        raise RuntimeError(
            "Journal path not set. Run: mirror-self init --journal /path/to/journal"
        )
    p = Path(path_str).expanduser()
    if not p.exists():
        raise RuntimeError(f"Journal path does not exist: {p}")
    return p
