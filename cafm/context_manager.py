"""Context manager — handles token-budget aware message construction."""

from __future__ import annotations

from typing import Any


# Rough estimator: 1 token ≈ 4 characters (English). Conservative enough for safety.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Cheap token estimate based on character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        # Role overhead ≈ 4 tokens
        total += 4 + estimate_tokens(msg.get("content", ""))
    return total


def build_transcript(entries: list[dict[str, Any]]) -> str:
    """Build a human-readable transcript string from debate entries.

    Each entry: {"model": str, "round": int, "content": str}
    """
    lines: list[str] = []
    for e in entries:
        header = f"[{e['model']} - Round {e['round']}]"
        lines.append(f"{header}:\n{e['content']}\n")
    return "\n".join(lines)


def sliding_window(
    messages: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """Return a trimmed message list that fits within *limit* tokens.

    Strategy: always keep the system message (index 0) and the user query
    (index 1), then include as many recent messages as possible.
    """
    if not messages:
        return messages

    # Reserve first two messages (system + user query)
    reserved = messages[:2]
    rest = messages[2:]

    budget = limit - estimate_messages_tokens(reserved) - 64  # safety margin
    if budget <= 0:
        return reserved

    # Walk backwards through the rest, adding as many as fit
    included: list[dict[str, str]] = []
    for msg in reversed(rest):
        cost = estimate_messages_tokens([msg])
        if budget - cost < 0:
            break
        included.insert(0, msg)
        budget -= cost

    return reserved + included


def summarize_transcript(
    entries: list[dict[str, Any]],
    summary_func,
    limit: int,
) -> str:
    """Produce a condensed summary of the debate so far.

    *summary_func* is a callable(model, messages, ctx) -> str (e.g. chat_sync).
    Falls back to a naive truncation if the summary call fails.
    """
    full = build_transcript(entries)
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a concise summarizer. Compress the following debate "
                "transcript into a brief summary that preserves every key argument, "
                "point of agreement, and point of contention. Use bullet points."
            ),
        },
        {"role": "user", "content": full},
    ]
    try:
        summary = summary_func(prompt_messages)
        return summary
    except Exception:
        # Fallback: naive tail truncation
        max_chars = limit * CHARS_PER_TOKEN
        if len(full) > max_chars:
            return "…[earlier transcript truncated]…\n" + full[-max_chars:]
        return full


def prepare_messages(
    *,
    system_prompt: str,
    user_query: str,
    transcript_entries: list[dict[str, Any]],
    context_limit: int,
    strategy: str = "sliding_window",
    summary_func=None,
) -> list[dict[str, str]]:
    """Build the message list for a model call, respecting the context budget.

    Returns a list of {role, content} dicts ready for ollama.chat().
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {user_query}"},
    ]

    if transcript_entries:
        transcript_text = build_transcript(transcript_entries)
        messages.append(
            {"role": "user", "content": f"Debate transcript so far:\n\n{transcript_text}"}
        )

    total = estimate_messages_tokens(messages)

    if total <= context_limit:
        return messages

    # Strategy: sliding_window or summary
    if strategy == "summary" and summary_func is not None:
        summary = summarize_transcript(
            transcript_entries, summary_func, context_limit // 2
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Query: {user_query}"},
            {
                "role": "user",
                "content": f"Condensed debate summary:\n\n{summary}",
            },
        ]
        # If still too large, fall through to sliding window
        if estimate_messages_tokens(messages) <= context_limit:
            return messages

    return sliding_window(messages, context_limit)
