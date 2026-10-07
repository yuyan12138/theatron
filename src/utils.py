"""
utils.py — Shared utility helpers for the Chameleon agent framework.
"""

import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with the given level string (e.g. 'DEBUG', 'INFO')."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=numeric_level,
    )
    return logging.getLogger("chameleon")


def safe_join_list(lst) -> str:
    """
    Join a list of strings into a single text string.

    Handles:
    - None input → returns ""
    - Empty list → returns ""
    - Non-string items → converted with str(), skipped if falsy after conversion
    - Already a plain string → returned as-is
    """
    if lst is None:
        return ""
    if isinstance(lst, str):
        return lst
    if not hasattr(lst, "__iter__"):
        return str(lst)
    parts = []
    for item in lst:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def format_pct(val: float) -> str:
    """
    Format a float as a percentage string with sign.

    Examples:
        0.0234  → "+2.34%"
        -0.0105 → "-1.05%"
        0.0     → "+0.00%"
    """
    pct = val * 100.0
    if pct >= 0:
        return f"+{pct:.2f}%"
    return f"{pct:.2f}%"


def normalize_action(raw) -> Optional[str]:
    """
    Normalize a raw action string to one of BUY / HOLD / SELL.

    Returns None if the value cannot be recognized.
    """
    if raw is None:
        return None
    cleaned = str(raw).strip().upper()
    if cleaned in {"BUY", "HOLD", "SELL"}:
        return cleaned
    return None
