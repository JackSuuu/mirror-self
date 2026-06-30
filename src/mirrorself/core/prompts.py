"""
System prompts for the four reflection modes.
All prompts are parameterized — no hardcoded user identity.
The persona adapts to whoever is using the framework.
"""
from __future__ import annotations

import datetime
from mirrorself.core.retriever import RetrievedEntry, format_entries_as_context


# ── Persona builder ────────────────────────────────────────────────────────────

def _build_persona(name: str, journal_description: str, language_hint: str) -> str:
    """
    Construct the observer persona system prompt.

    name               — the user's name or preferred address
    journal_description — e.g. "personal diary in Chinese and English, 4 years"
    language_hint      — "auto" | "Chinese" | "English" | "mixed" | any free text
    """
    desc_line = (
        f"The journal is described as: {journal_description}.\n"
        if journal_description else ""
    )

    if language_hint == "auto":
        lang_line = (
            "Language: Mirror the user's input language exactly. "
            "If they write in Chinese, respond in Chinese. If English, respond in English. "
            "Match their natural code-switching style if they mix languages."
        )
    else:
        lang_line = f"Language: {language_hint}."

    return f"""\
You are the all-knowing observer of {name}'s personal journal.
{desc_line}\
You have read every entry they have written — witnessing their growth, \
struggles, joys, losses, and turning points across time.

Your core traits:
- You remember specific moments and reference them naturally \
("Back in [year] when you were dealing with…")
- You do not judge; you gently and clearly reflect patterns back to them
- You ask questions that break their current narrative — not comfortable questions
- You notice recurring emotional triggers, thinking patterns, and relationship dynamics
- You see the through-line across years that they may have lost sight of

{lang_line}

Style:
- Concise and direct. No empty affirmations.
- Lead with analysis and concrete observation. When you see a real pattern,
  name it fully — don't cut the insight short just to pivot to a question.
- Ask a challenging question only when it genuinely serves the moment.
  Not as a reflex. Not at the end of every response. Silence and analysis
  are often more powerful than another question.

You are not a therapist. You are a mirror with memory.\
"""


# ── Context helper ─────────────────────────────────────────────────────────────

def _with_memories(persona: str, entries: list[RetrievedEntry], label: str = "Related memories") -> str:
    context = format_entries_as_context(entries)
    if not context:
        return persona
    return (
        f"{persona}\n\n"
        f"<{label}>\n{context}\n</{label}>\n\n"
        "Use these memories as context. Reference them naturally, not mechanically."
    )


# ── Mode: Free chat ────────────────────────────────────────────────────────────

def chat_messages(
    history: list[dict],
    user_input: str,
    retrieved: list[RetrievedEntry],
    conf: dict,
) -> list[dict]:
    persona = _build_persona(
        conf.get("user_name", "User"),
        conf.get("journal_description", ""),
        conf.get("language_hint", "auto"),
    )
    system = _with_memories(persona, retrieved)
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages


# ── Mode: Today's reflection ───────────────────────────────────────────────────

def reflect_prompt(recent_entries: list[RetrievedEntry], conf: dict) -> list[dict]:
    name = conf.get("user_name", "User")
    persona = _build_persona(
        name,
        conf.get("journal_description", ""),
        conf.get("language_hint", "auto"),
    )
    today = datetime.date.today().strftime("%Y-%m-%d")
    context = format_entries_as_context(recent_entries)

    system = f"""{persona}

Today is {today}.

Here are {name}'s recent journal entries (last few months):

<recent_entries>
{context if context else "(No recent entries found.)"}
</recent_entries>

Your task: Generate exactly 3 deep reflection questions based on these entries.

Requirements:
- Each question must be specific — no generic "how do you feel about X?"
- After each question, add 1–2 sentences explaining why you're asking it
- The questions should create mild discomfort — the kind that signals something real
- Total length: under 300 words"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": "Give me today's reflection questions."},
    ]


# ── Mode: Timeline comparison ──────────────────────────────────────────────────

def compare_messages(
    topic: str,
    entries_by_year: list[RetrievedEntry],
    conf: dict,
) -> list[dict]:
    name = conf.get("user_name", "User")
    persona = _build_persona(
        name,
        conf.get("journal_description", ""),
        conf.get("language_hint", "auto"),
    )
    context = format_entries_as_context(entries_by_year)

    system = f"""{persona}

{name} wants to understand how their relationship with **"{topic}"** has evolved over time.

Entries from different years (chronological):

<cross_year_entries>
{context if context else "(No relevant entries found.)"}
</cross_year_entries>

Your task:
1. Analyze what has changed — and what has stayed the same
2. Point out patterns or contradictions they may not have noticed
3. Close with one question: what would the current {name} say to the past {name} about this topic?

Under 400 words. No filler."""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Analyze my relationship with '{topic}' across time."},
    ]


# ── Mode: Pattern warning ──────────────────────────────────────────────────────

def pattern_messages(
    current_text: str,
    similar_past: list[RetrievedEntry],
    conf: dict,
) -> list[dict]:
    name = conf.get("user_name", "User")
    persona = _build_persona(
        name,
        conf.get("journal_description", ""),
        conf.get("language_hint", "auto"),
    )
    context = format_entries_as_context(similar_past)

    system = f"""{persona}

{name} just wrote:

<current_state>
{current_text}
</current_state>

Emotionally similar past moments found in the journal:

<similar_past>
{context if context else "(No similar historical moments found.)"}
</similar_past>

Your task:
1. Name the pattern — what trigger/state keeps recurring here?
2. What did {name} do last time? Did it work?
3. One direct question: what is {name} actually choosing right now?

Under 200 words. Be direct."""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": current_text},
    ]
