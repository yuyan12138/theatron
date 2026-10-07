from __future__ import annotations

from typing import Optional

from .data import safe_join


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def system_message_for(
    condition: str,
    protocol: dict,
    personas: dict[str, str],
) -> str:
    if condition == "task_only":
        return protocol["prompt"]["task_only_system_message"]
    return personas[condition]


def build_prompt(
    row,
    day_index: int,
    condition: str,
    protocol: dict,
    personas: dict[str, str],
    prev_action: Optional[str],
    previous_action_return: float,
    cumulative_log_return: float,
    prev_price: Optional[float],
) -> tuple[str, str]:
    """Build the matched Zero-Shot prompt.

    Only the system message differs between task-only and persona conditions.
    The user-side market/state template is shared exactly.
    """
    system_message = system_message_for(condition, protocol, personas)
    date_str = row["date"].strftime("%Y-%m-%d")
    symbol = str(row["symbol"])
    price = float(row["close_price"])
    parts = [f"Today is {date_str}. Stock: {symbol}."]

    if day_index == 0 or prev_price is None:
        parts.append(f"Current price: {price:.2f}.")
    else:
        change = 0.0 if prev_price == 0 else price / prev_price - 1.0
        parts.append(
            f"Current price: {price:.2f}. "
            f"Price change from yesterday: {format_pct(change)}."
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
    return system_message, "\n\n".join(parts)
