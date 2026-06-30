"""
Semantic retrieval with temporal diversity.

Core idea: don't just return the top-K most similar entries.
Enforce that results span multiple years so the "observer" can make
cross-time comparisons ("in 2023 you also felt this way...").
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from mirrorself import config as cfg
from mirrorself.core.indexer import get_collection, embed_texts


@dataclass
class RetrievedEntry:
    chunk_id: str
    text: str
    year: int
    month: int
    day: int
    chapter: int
    date_label: str
    distance: float  # cosine distance (lower = more similar)

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


def _embed_query(query: str, conf: dict) -> list[float]:
    result = embed_texts(
        [query],
        model=conf.get("embed_model", "nomic-embed-text"),
        base_url=conf.get("ollama_base_url", "http://localhost:11434"),
    )
    return result[0]


def _chroma_to_entries(results: dict) -> list[RetrievedEntry]:
    entries = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        entries.append(
            RetrievedEntry(
                chunk_id=chunk_id,
                text=doc,
                year=meta.get("year", 0),
                month=meta.get("month", 0),
                day=meta.get("day", 0),
                chapter=meta.get("chapter", 0),
                date_label=meta.get("date_label", ""),
                distance=dist,
            )
        )
    return entries


def _temporal_diversify(
    entries: list[RetrievedEntry],
    top_k: int,
    min_year_spread: int,
) -> list[RetrievedEntry]:
    """
    Pick top_k entries while ensuring at least min_year_spread distinct years.

    Algorithm:
    1. Take entries sorted by similarity (best first).
    2. Keep adding entries, but once we've added top_k // 2 from the same year,
       prefer entries from less-represented years.
    """
    if not entries:
        return []

    # Sort by similarity descending
    sorted_entries = sorted(entries, key=lambda e: e.distance)

    year_counts: dict[int, int] = {}
    selected: list[RetrievedEntry] = []
    overflow: list[RetrievedEntry] = []  # entries held back for year-spread

    max_per_year = max(1, top_k // min_year_spread)

    for entry in sorted_entries:
        y = entry.year
        if year_counts.get(y, 0) < max_per_year:
            selected.append(entry)
            year_counts[y] = year_counts.get(y, 0) + 1
        else:
            overflow.append(entry)
        if len(selected) >= top_k:
            break

    # If we still have slots, fill with overflow
    for entry in overflow:
        if len(selected) >= top_k:
            break
        selected.append(entry)

    # Final sort by date for readability
    selected.sort(key=lambda e: (e.year, e.month, e.day))
    return selected


# ── Year hint extraction ───────────────────────────────────────────────────────

def extract_year_filter(query: str) -> Optional[int]:
    """
    If the query clearly refers to a single specific year, return that year so
    the caller can pass it as year_filter to search().  Returns None when the
    query spans multiple years or is not year-specific.

    Handles:
    - "2025年干了什么" → 2025
    - "今年"           → current year
    - "去年"           → current year - 1
    - "前年"           → current year - 2
    - "2023年和2024年" → None  (multiple years — don't filter)
    """
    today = datetime.date.today()
    current_year = today.year

    # Explicit 4-digit years written as "XXXXYear" (most reliable)
    year_mentions = re.findall(r"(20\d{2})年", query)
    unique_years = list({int(y) for y in year_mentions})

    if len(unique_years) == 1:
        return unique_years[0]
    if len(unique_years) > 1:
        return None  # cross-year query — let temporal diversify handle it

    # No explicit year digits — check relative expressions
    # Avoid matching if multiple relative words co-exist (e.g. "去年和今年")
    relative_hits = sum([
        "今年" in query,
        "去年" in query,
        "前年" in query,
    ])
    if relative_hits == 1:
        if "今年" in query:
            return current_year
        if "去年" in query:
            return current_year - 1
        if "前年" in query:
            return current_year - 2

    return None


# ── Public retrieval functions ─────────────────────────────────────────────────

def search(
    query: str,
    conf: dict,
    top_k: Optional[int] = None,
    year_filter: Optional[int] = None,
    month_filter: Optional[int] = None,
) -> list[RetrievedEntry]:
    """
    Semantic search with optional temporal filter and year diversity.
    """
    top_k = top_k or conf.get("top_k", 6)
    min_year_spread = conf.get("min_year_spread", 2)
    collection = get_collection(cfg.chroma_path())

    if collection.count() == 0:
        return []

    query_embedding = _embed_query(query, conf)

    where = {}
    if year_filter:
        where["year"] = {"$eq": year_filter}
    if month_filter:
        where["month"] = {"$eq": month_filter}

    n_results = min(top_k * 4, collection.count())  # over-fetch for re-ranking

    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)
    candidates = _chroma_to_entries(raw)
    return _temporal_diversify(candidates, top_k, min_year_spread)


def search_by_year(
    query: str,
    conf: dict,
    years: list[int],
    per_year: int = 2,
) -> list[RetrievedEntry]:
    """
    For timeline comparison: retrieve `per_year` entries from each requested year.
    """
    collection = get_collection(cfg.chroma_path())
    if collection.count() == 0:
        return []

    query_embedding = _embed_query(query, conf)
    all_entries: list[RetrievedEntry] = []

    for year in years:
        n_results = min(per_year * 3, collection.count())
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where={"year": {"$eq": year}},
        )
        candidates = _chroma_to_entries(raw)
        # Take top per_year from this year
        all_entries.extend(candidates[:per_year])

    all_entries.sort(key=lambda e: (e.year, e.month, e.day))
    return all_entries


def search_emotional_low(
    current_text: str,
    conf: dict,
    top_k: int = 4,
) -> list[RetrievedEntry]:
    """
    For pattern warning: find past entries similar to current emotional state.
    Emphasises low-mood/struggle signals by enriching the query.
    """
    enriched = f"焦虑 困惑 迷失 挣扎 痛苦 不确定 {current_text}"
    return search(enriched, conf, top_k=top_k)


def recent_months(
    conf: dict,
    n_months: int = 3,
) -> list[RetrievedEntry]:
    """
    Return all entries from the last n_months (by year/month metadata).
    Used by the daily reflection mode to understand recent context.
    """
    collection = get_collection(cfg.chroma_path())
    if collection.count() == 0:
        return []

    import datetime
    today = datetime.date.today()
    cutoffs: list[tuple[int, int]] = []
    y, mo = today.year, today.month
    for _ in range(n_months):
        cutoffs.append((y, mo))
        mo -= 1
        if mo == 0:
            mo = 12
            y -= 1

    # Build OR filter across recent (year, month) pairs
    # ChromaDB supports $or across conditions
    conditions = [
        {"$and": [{"year": {"$eq": y}}, {"month": {"$eq": m}}]}
        for y, m in cutoffs
    ]
    where = {"$or": conditions} if len(conditions) > 1 else {"year": {"$eq": cutoffs[0][0]}}

    result = collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    entries = []
    for chunk_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        entries.append(
            RetrievedEntry(
                chunk_id=chunk_id,
                text=doc,
                year=meta.get("year", 0),
                month=meta.get("month", 0),
                day=meta.get("day", 0),
                chapter=meta.get("chapter", 0),
                date_label=meta.get("date_label", ""),
                distance=0.0,
            )
        )
    entries.sort(key=lambda e: (e.year, e.month, e.day))
    return entries


def format_entries_as_context(entries: list[RetrievedEntry]) -> str:
    """Format retrieved entries into a readable context block for the LLM."""
    if not entries:
        return ""
    parts = []
    for e in entries:
        parts.append(f"[{e.date_label}]\n{e.text}")
    return "\n\n---\n\n".join(parts)
