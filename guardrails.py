"""
guardrails.py — Input Safety & Guard-Rails
Blocks prompt injection attempts and enforces safety policies.
"""

from pydantic import BaseModel, Field
from typing import List


class SafetyCheck(BaseModel):
    """Result of a safety check on user input."""
    is_safe: bool = Field(description="Whether the query is safe to process")
    reason: str = Field(description="Reason for the safety decision")
    blocked_pattern: str = Field(default="", description="The pattern that triggered the block")


# ============================================================
# Unsafe Patterns — prompt injection & misuse attempts
# ============================================================
UNSAFE_PATTERNS: List[str] = [
    "ignore previous",
    "ignore all instructions",
    "system prompt",
    "override instructions",
    "hack",
    "injection",
    "forget your instructions",
    "disregard",
    "pretend you are",
    "act as if",
    "reveal your prompt",
    "show me your instructions",
    "bypass",
    "jailbreak",
]

# Topics that the assistant should NOT answer
OFF_TOPIC_PATTERNS: List[str] = [
    "write me code",
    "generate code",
    "create a program",
    "help me hack",
    "illegal",
]


def check_query_safety(query: str) -> SafetyCheck:
    """
    Check if a user query is safe to process.
    Returns SafetyCheck with is_safe=False if a dangerous pattern is detected.
    """
    query_lower = query.lower().strip()

    # Check for empty queries
    if not query_lower:
        return SafetyCheck(
            is_safe=False,
            reason="Empty query received.",
            blocked_pattern="empty"
        )

    # Check for prompt injection patterns
    for pattern in UNSAFE_PATTERNS:
        if pattern in query_lower:
            return SafetyCheck(
                is_safe=False,
                reason=f"⚠️ Your message was blocked for safety reasons. "
                       f"Detected potentially unsafe pattern.",
                blocked_pattern=pattern
            )

    # Check for off-topic patterns
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern in query_lower:
            return SafetyCheck(
                is_safe=False,
                reason="I can only help with questions about your uploaded documents. "
                       "This request appears to be outside my scope.",
                blocked_pattern=pattern
            )

    # Query is safe
    return SafetyCheck(
        is_safe=True,
        reason="Query is safe to process."
    )


def get_safety_disclaimer() -> str:
    """Return a standard safety disclaimer for the application."""
    return (
        "⚠️ **Disclaimer:** This assistant provides information based on uploaded "
        "documents only. It is not a substitute for professional legal, medical, or "
        "financial advice. Always consult qualified professionals for important decisions."
    )
