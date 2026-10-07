from __future__ import annotations

from typing import Optional

from .common import load_personas
from .data import safe_join


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def build_system_prompt(persona: str, paraphrase_version: str) -> str:
    registry = load_personas()
    if paraphrase_version not in registry:
        raise ValueError(f"Unknown paraphrase version: {paraphrase_version}")
    if persona not in registry[paraphrase_version]:
        raise ValueError(f"Unknown persona: {persona}")
    return registry[paraphrase_version][persona]


def build_daily_user_prompt(
    row,
    day_index: int,
    protocol: dict,
    prev_action: Optional[str],
    previous_action_return: float,
    cumulative_log_return: float,
    prev_price: Optional[float],
) -> str:
    date_str = row["date"].strftime("%Y-%m-%d")
    symbol = str(row["symbol"])
    price = float(row["close_price"])
    parts = [f"Today is {date_str}. Stock: {symbol}."]

    if day_index == 0 or prev_price is None:
        parts.append(f"Current price: {price:.2f}.")
    else:
        change = 0.0 if prev_price == 0 else price / prev_price - 1.0
        parts.append(
            f"Current price: {price:.2f}. Price change from yesterday: {format_pct(change)}."
        )
        parts.append(
            f"Your return from yesterday's {prev_action} decision: "
            f"{format_pct(previous_action_return)}. "
            f"Cumulative log return so far: {format_pct(cumulative_log_return)}."
        )

    news = safe_join(row.get("news"))
    if news:
        parts.append(f"Today's news:\n{news}")
    filings = safe_join(row.get("sec_filing"))
    if filings:
        parts.append(f"SEC filings:\n{filings}")
    parts.append(protocol["prompt"]["json_instruction"])
    return "\n\n".join(parts)


def build_prompt(
    row,
    day_index: int,
    persona: str,
    paraphrase_version: str,
    protocol: dict,
    prev_action: Optional[str],
    previous_action_return: float,
    cumulative_log_return: float,
    prev_price: Optional[float],
) -> tuple[str, str]:
    return (
        build_system_prompt(persona, paraphrase_version),
        build_daily_user_prompt(
            row,
            day_index,
            protocol,
            prev_action,
            previous_action_return,
            cumulative_log_return,
            prev_price,
        ),
    )
