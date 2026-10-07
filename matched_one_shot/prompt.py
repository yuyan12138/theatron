from __future__ import annotations

import json
from typing import Optional

from .common import load_examples, load_personas
from .data import safe_join


def format_pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def build_system_prompt(persona: str, scenario: str, protocol: dict) -> str:
    personas = load_personas()
    examples = load_examples()
    if persona not in personas:
        raise ValueError(f"Unknown persona: {persona}")
    if scenario not in examples:
        raise ValueError(f"Unknown scenario: {scenario}")
    prompt_cfg = protocol["prompt"]
    example = examples[scenario]
    decision_record = example["decisions"][persona]
    decision = (
        "{"
        f"\"date\": {json.dumps(decision_record['date'], ensure_ascii=False)}, "
        f"\"symbol\": {json.dumps(decision_record['symbol'], ensure_ascii=False)}, "
        f"\"action\": {json.dumps(decision_record['action'], ensure_ascii=False)}, "
        f"\"rationale\": {json.dumps(decision_record['rationale'], ensure_ascii=False)}, "
        f"\"confidence\": {float(decision_record['confidence']):.2f}"
        "}"
    )
    return "\n\n".join(
        [
            personas[persona],
            prompt_cfg["introduction"],
            prompt_cfg["example_header"],
            example["market_block"],
            f"Decision:\n{decision}",
            prompt_cfg["guardrail"],
        ]
    )


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
    scenario: str,
    protocol: dict,
    prev_action: Optional[str],
    previous_action_return: float,
    cumulative_log_return: float,
    prev_price: Optional[float],
) -> tuple[str, str]:
    return (
        build_system_prompt(persona, scenario, protocol),
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
