from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .model_clients import BaseClient, Generation


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class CallAttempt:
    kind: str
    system_message: str
    user_message: str
    generation: Generation


@dataclass
class ParsedDecision:
    record: dict
    status: str
    attempts: list[CallAttempt]


def _objects_in_text(text: str):
    cleaned = THINK_RE.sub("", text).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _validate(value: dict, date: str, symbol: str) -> Optional[dict]:
    action = str(value.get("action", "")).strip().upper()
    if action not in {"BUY", "HOLD", "SELL"}:
        return None
    confidence = value.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "date": date,
        "symbol": symbol,
        "action": action,
        "rationale": str(value.get("rationale", "")),
        "confidence": confidence,
    }


def parse_or_repair(
    client: BaseClient,
    first: Generation,
    date: str,
    symbol: str,
    seed: int,
    protocol: dict,
    first_system_message: str = "",
    first_user_message: str = "",
) -> ParsedDecision:
    attempts = [CallAttempt("decision", first_system_message, first_user_message, first)]
    for value in _objects_in_text(first.text):
        valid = _validate(value, date, symbol)
        if valid:
            return ParsedDecision(valid, "direct", attempts)

    repair_system = (
        "You are a JSON repair assistant. Return only one valid JSON object with keys "
        "date, symbol, action, rationale, and confidence. Action must be BUY, HOLD, or SELL."
    )
    repair_user = (
        f"Rewrite the response below as valid JSON for date={date}, symbol={symbol}.\n\n"
        f"{first.text}"
    )
    repaired = client.generate(repair_system, repair_user, seed)
    attempts.append(CallAttempt("repair", repair_system, repair_user, repaired))
    for value in _objects_in_text(repaired.text):
        valid = _validate(value, date, symbol)
        if valid:
            return ParsedDecision(valid, "repaired", attempts)

    fallback = {
        "date": date,
        "symbol": symbol,
        "action": protocol["generation"]["fallback_action"],
        "rationale": "Invalid model output after one repair attempt; protocol fallback applied.",
        "confidence": 0.0,
    }
    return ParsedDecision(fallback, "fallback", attempts)
