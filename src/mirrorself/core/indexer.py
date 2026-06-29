"""
Journal markdown parser and ChromaDB indexer.

Supports all date format variants found in the journal:
  ## 2022/2/1 - 周三
  ## 2023/6/2 - 新冠
  ## January 5th
  ## ==January 3rd - 顺利的一天==
  **六月三日**
  ****一月二十日****
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx

from mirrorself import config as cfg

# ── Chinese numeral converter ──────────────────────────────────────────────────

_CN_BASE = {
    "〇": 0, "零": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_CN_MONTH_MAP = {
    "一月": 1, "二月": 2, "三月": 3, "四月": 4,
    "五月": 5, "六月": 6, "七月": 7, "八月": 8,
    "九月": 9, "十月": 10, "十一月": 11, "十二月": 12,
}
_EN_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _cn_num_to_int(s: str) -> int:
    """Convert Chinese number string like '二十三' to int 23."""
    s = s.strip()
    if not s:
        return 0
    # Single char
    if len(s) == 1:
        return _CN_BASE.get(s, 0)
    # Starts with 十: 十一 = 11
    if s[0] == "十":
        rest = _cn_num_to_int(s[1:]) if len(s) > 1 else 0
        return 10 + rest
    # Two chars: 二十, 三十
    if len(s) == 2 and s[1] == "十":
        return _CN_BASE.get(s[0], 0) * 10
    # Three chars: 二十一 ~ 三十一
    if len(s) == 3 and s[1] == "十":
        return _CN_BASE.get(s[0], 0) * 10 + _CN_BASE.get(s[2], 0)
    return _CN_BASE.get(s, 0)


# ── Filename metadata ──────────────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r"CHAPTER\s+(\d+)\s*[-—]\s*(\d{4})\s*[年]?\s*([^\s.]+)",
    re.IGNORECASE,
)

def parse_filename(path: Path) -> tuple[int, int, int]:
    """Return (chapter, year, month) from filename. month=0 if unparseable."""
    m = _FNAME_RE.search(path.stem)
    if not m:
        return 0, 0, 0
    chapter = int(m.group(1))
    year = int(m.group(2))
    month_str = m.group(3).strip()
    month = _CN_MONTH_MAP.get(month_str, 0)
    if not month:
        month = _EN_MONTH_MAP.get(month_str.lower(), 0)
    return chapter, year, month


def is_standard_chapter(path: Path) -> bool:
    """Return True only for CHAPTER N - YYYY ... files."""
    return bool(_FNAME_RE.search(path.stem))


# ── Date header detection ──────────────────────────────────────────────────────

# Pattern 1: ## 2022/2/1 - any title
_P_ISO = re.compile(
    r"^#{1,3}\s+(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*[-—]\s*.*)?\s*$",
    re.MULTILINE,
)
# Pattern 2: ## January 3rd / ## ==January 3rd - title==
_P_EN = re.compile(
    r"^#{1,3}\s*(?:==)?\s*(January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:\s*[-—]\s*.*)?\s*(?:==)?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
# Pattern 3: **六月三日** / ****一月二十日****  (bold Chinese date)
_P_CN_BOLD = re.compile(
    r"^\*{2,4}([〇一二三四五六七八九十]+月[〇一二三四五六七八九十]+日)\*{2,4}\s*$",
    re.MULTILINE,
)
# Pattern 4: ## section headers that are NOT dates — used to skip them
_P_SECTION = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)


def _find_date_splits(text: str, year: int, month: int) -> list[tuple[int, int, int, int]]:
    """
    Return list of (start_pos, year, month, day) for every date header found.
    start_pos is the position in `text` where the header line begins.
    """
    hits: list[tuple[int, int, int, int]] = []

    for m in _P_ISO.finditer(text):
        hits.append((m.start(), int(m.group(1)), int(m.group(2)), int(m.group(3))))

    for m in _P_EN.finditer(text):
        mo = _EN_MONTH_MAP.get(m.group(1).lower(), month)
        hits.append((m.start(), year, mo, int(m.group(2))))

    for m in _P_CN_BOLD.finditer(text):
        date_str = m.group(1)  # e.g. 六月三日
        parts = re.match(r"([一二三四五六七八九十]+月)([一二三四五六七八九十]+日)", date_str)
        if parts:
            mo_str = parts.group(1).replace("月", "")
            day_str = parts.group(2).replace("日", "")
            mo = _cn_num_to_int(mo_str) or month
            day = _cn_num_to_int(day_str)
            if day:
                hits.append((m.start(), year, mo, day))

    hits.sort(key=lambda x: x[0])
    return hits


# ── Text cleaning ──────────────────────────────────────────────────────────────

_IMG_RE = re.compile(r"!\[.*?\]\(.*?\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MD_BOLD = re.compile(r"\*{1,4}([^*]+)\*{1,4}")
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_HIGHLIGHT_RE = re.compile(r"==([^=]+)==")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip markdown formatting and image links, keep readable prose."""
    text = _IMG_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HIGHLIGHT_RE.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ── Entry dataclass ────────────────────────────────────────────────────────────

@dataclass
class Entry:
    chunk_id: str
    chapter: int
    year: int
    month: int
    day: int          # 0 = unknown / whole-month chunk
    text: str         # cleaned prose
    raw_date: str = ""

    @property
    def date_label(self) -> str:
        if self.day:
            return f"{self.year}/{self.month:02d}/{self.day:02d}"
        return f"{self.year}/{self.month:02d}"

    def to_chroma_doc(self) -> dict:
        return {
            "id": self.chunk_id,
            "document": self.text,
            "metadata": {
                "chapter": self.chapter,
                "year": self.year,
                "month": self.month,
                "day": self.day,
                "date_label": self.date_label,
            },
        }


def _make_id(chapter: int, year: int, month: int, day: int, char_pos: int, idx: int) -> str:
    """
    Use char_pos (byte offset of the date header in the raw file) + sub-chunk idx
    so that duplicate same-day headers within one file get distinct IDs.
    """
    key = f"{chapter}_{year}_{month}_{day}_{char_pos}_{idx}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Parser ─────────────────────────────────────────────────────────────────────

MAX_CHUNK_CHARS = 3000   # ~750 tokens — split long daily entries at paragraphs


def _split_long(text: str) -> list[str]:
    """Split text at paragraph boundaries if it exceeds MAX_CHUNK_CHARS."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    paras = re.split(r"\n{2,}", text)
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 2 > MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = p
        else:
            current = (current + "\n\n" + p) if current else p
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def parse_file(path: Path) -> list[Entry]:
    """Parse a single journal file into a list of Entry objects."""
    chapter, year, month = parse_filename(path)
    if not year:
        return []

    raw = path.read_text(encoding="utf-8")

    # Strip YAML front matter if any
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            raw = raw[end + 3:]

    splits = _find_date_splits(raw, year, month)

    entries: list[Entry] = []

    if not splits:
        # No date headers → treat whole file as one chunk
        cleaned = clean_text(raw)
        if len(cleaned) < 30:
            return []
        for idx, chunk in enumerate(_split_long(cleaned)):
            entries.append(
                Entry(
                    chunk_id=_make_id(chapter, year, month, 0, 0, idx),
                    chapter=chapter,
                    year=year,
                    month=month,
                    day=0,
                    text=chunk,
                )
            )
        return entries

    # Split raw text by date headers
    boundaries = [(pos, y, mo, d) for pos, y, mo, d in splits]
    boundaries.append((len(raw), year, month, 0))  # sentinel

    for i, (start, y, mo, d) in enumerate(boundaries[:-1]):
        end = boundaries[i + 1][0]
        chunk_raw = raw[start:end]
        cleaned = clean_text(chunk_raw)
        if len(cleaned) < 30:
            continue
        for idx, part in enumerate(_split_long(cleaned)):
            entries.append(
                Entry(
                    chunk_id=_make_id(chapter, y, mo, d, start, idx),
                    chapter=chapter,
                    year=y,
                    month=mo,
                    day=d,
                    text=part,
                )
            )

    return entries


def parse_all(journal_path: Path) -> list[Entry]:
    """Parse all standard CHAPTER files in the journal directory."""
    all_entries: list[Entry] = []
    for md_file in sorted(journal_path.glob("CHAPTER *.md")):
        if not is_standard_chapter(md_file):
            continue
        all_entries.extend(parse_file(md_file))
    return all_entries


# ── Ollama embedding ───────────────────────────────────────────────────────────

def _embed_batch(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """Embed a batch of texts via Ollama /api/embed (batch endpoint)."""
    url = base_url.rstrip("/") + "/api/embed"
    resp = httpx.post(url, json={"model": model, "input": texts}, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_texts(
    texts: list[str],
    model: str,
    base_url: str,
    batch_size: int = 32,
    on_progress=None,
) -> list[list[float]]:
    """Embed all texts, calling on_progress(done, total) after each batch."""
    results: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        results.extend(_embed_batch(batch, model, base_url))
        if on_progress:
            on_progress(min(start + batch_size, total), total)
    return results


# ── ChromaDB indexing ──────────────────────────────────────────────────────────

def get_collection(chroma_path: Path):
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        "journal",
        metadata={"hnsw:space": "cosine"},
    )


def _existing_ids(collection) -> set[str]:
    result = collection.get(include=[])
    return set(result["ids"])


def index_journal(
    journal_path: Path,
    conf: dict,
    on_progress=None,
    force: bool = False,
) -> tuple[int, int]:
    """
    Index all journal entries into ChromaDB.
    Returns (new_entries_added, total_entries).
    """
    entries = parse_all(journal_path)
    if not entries:
        return 0, 0

    collection = get_collection(cfg.chroma_path())
    existing = _existing_ids(collection) if not force else set()

    new_entries = [e for e in entries if e.chunk_id not in existing]
    if not new_entries:
        return 0, len(entries)

    texts = [e.text for e in new_entries]
    embeddings = embed_texts(
        texts,
        model=conf.get("embed_model", "nomic-embed-text"),
        base_url=conf.get("ollama_base_url", "http://localhost:11434"),
        on_progress=on_progress,
    )

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(new_entries), batch_size):
        batch = new_entries[start : start + batch_size]
        batch_emb = embeddings[start : start + batch_size]
        collection.upsert(
            ids=[e.chunk_id for e in batch],
            embeddings=batch_emb,
            documents=[e.text for e in batch],
            metadatas=[e.to_chroma_doc()["metadata"] for e in batch],
        )

    return len(new_entries), len(entries)


def collection_stats(conf: dict) -> dict:
    """Return basic statistics about the indexed collection."""
    try:
        col = get_collection(cfg.chroma_path())
        count = col.count()
        if count == 0:
            return {"count": 0, "years": []}
        result = col.get(include=["metadatas"])
        years = sorted(set(m["year"] for m in result["metadatas"] if m.get("year")))
        return {"count": count, "years": years}
    except Exception:
        return {"count": 0, "years": []}
