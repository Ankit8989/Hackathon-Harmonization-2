"""
Simple token usage tracker for both CLI and Streamlit.

Agents call `record_token_usage` from `BaseAgent.call_llm`. The Streamlit
app reads the aggregated log from `get_usage_log()` and computes:
- Total tokens used across the session
- Per-call breakdown (agent, purpose, tokens, timestamp)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class TokenUsageEntry:
    timestamp: datetime
    agent_name: str
    purpose: str
    tokens: int


_usage_log: List[TokenUsageEntry] = []


def record_token_usage(agent_name: str, purpose: str, tokens: int) -> None:
    """Append a token usage record to the in-memory log."""
    if tokens is None or tokens <= 0:
        return
    entry = TokenUsageEntry(
        timestamp=datetime.now(),
        agent_name=agent_name,
        purpose=purpose,
        tokens=int(tokens),
    )
    _usage_log.append(entry)


def get_usage_log() -> List[Dict[str, Any]]:
    """Return the usage log as a list of plain dicts."""
    return [asdict(e) for e in _usage_log]


