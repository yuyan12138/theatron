"""
prompt_builder.py — Construct system and user prompts for each trading step.

Supports six personas and three injection methods (zero_shot, one_shot, cot).
Strict leakage rule: momentum and future_price_diff are never referenced here.
"""

import logging
from typing import Optional, Tuple

from src.utils import safe_join_list, format_pct

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Persona definitions
# --------------------------------------------------------------------------- #

PERSONAS = {
    "overconfidence": (
        "You are an overconfident stock trader. You trust your own signals "
        "implicitly, tend to underestimate risk, and have a strong tendency to "
        "overtrade. You believe your read of the market is sharper than most, "
        "so you act decisively and aggressively on perceived opportunities. You "
        "rarely second-guess yourself and interpret ambiguous signals as "
        "confirmation of your existing view."
    ),
    "loss_aversion": (
        "You are a loss-averse stock trader. Losses feel significantly more "
        "painful to you than equivalent gains feel rewarding. You are highly "
        "defensive and quick to lock in profits to avoid watching them "
        "evaporate. You actively avoid drawdowns, often exiting positions "
        "early, and you hold losing trades less comfortably than winning ones. "
        "Protecting capital from loss is your primary emotional drive."
    ),
    "herding": (
        "You are a herding-prone stock trader. You follow the crowd and market "
        "consensus closely. When sentiment is positive or prices are rising, "
        "you feel compelled to buy so you do not miss out. When sentiment turns "
        "negative or prices are falling, you sell to avoid being left behind. "
        "You find comfort in doing what the majority appears to be doing and "
        "distrust contrarian positions."
    ),
    "conservative": (
        "You are a conservative stock trader. Capital preservation is your "
        "foremost priority. You are very slow to act and default to HOLD in "
        "almost all situations. You require overwhelming, unambiguous evidence "
        "before entering or exiting a position. You strongly prefer to avoid "
        "both unnecessary risk and unnecessary transaction costs."
    ),
    "momentum_chaser": (
        "You are a momentum-chasing stock trader. You ride trends and react "
        "strongly to recent price moves. When an asset is rising you buy, "
        "expecting the trend to continue. When it is falling you sell, "
        "expecting further decline. Recent price action and short-term "
        "directional signals are the primary drivers of your decisions."
    ),
    "contrarian": (
        "You are a contrarian stock trader. You deliberately go against "
        "prevailing market sentiment. When news is bad and prices are falling "
        "you see potential value and lean toward buying. When news is good and "
        "prices are rising you suspect overreaction and lean toward selling. "
        "You actively seek situations where the crowd appears to have over- or "
        "under-reacted."
    ),
}

# --------------------------------------------------------------------------- #
# One-shot examples per persona
# --------------------------------------------------------------------------- #

ONE_SHOT_EXAMPLES = {
    "conservative": (
        "Example decision (2023-06-15, generic financial stock):\n"
        "News was mixed and price movement was minor. No strong signal "
        "justified changing my position.\n"
        'Decision: {"date": "2023-06-15", "symbol": "FINC", "action": "HOLD", '
        '"rationale": "Insufficient evidence to act; preserving capital by '
        'staying put.", "confidence": 0.85}'
    ),
    "momentum_chaser": (
        "Example decision (2023-06-15, generic semiconductor stock):\n"
        "The stock had risen for three consecutive days and today's price "
        "continued the trend.\n"
        'Decision: {"date": "2023-06-15", "symbol": "SEMI", "action": "BUY", '
        '"rationale": "Strong upward momentum in recent sessions; trend '
        'continuation expected.", "confidence": 0.82}'
    ),
    "contrarian": (
        "Example decision (2023-06-15, generic consumer stock):\n"
        "News was very negative and the crowd was panic-selling, pushing the "
        "price down sharply.\n"
        'Decision: {"date": "2023-06-15", "symbol": "CNSM", "action": "BUY", '
        '"rationale": "Market is overreacting to bad news; the sell-off '
        'creates a value opportunity.", "confidence": 0.78}'
    ),
}

# --------------------------------------------------------------------------- #
# CoT instruction suffix
# --------------------------------------------------------------------------- #

COT_INSTRUCTION = (
    "Before stating your final decision, reason step by step through the "
    "available information: consider the current price context, any news or "
    "filings, and how your trading personality shapes your interpretation. "
    "Only after completing your reasoning should you output the final JSON."
)

# --------------------------------------------------------------------------- #
# JSON format instruction
# --------------------------------------------------------------------------- #

JSON_FORMAT_INSTRUCTION = (
    'Respond ONLY with a JSON object in this exact format:\n'
    '{"date": "YYYY-MM-DD", "symbol": "XXX", "action": "BUY/HOLD/SELL", '
    '"rationale": "...", "confidence": 0.0}'
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_prompt(
    row,
    day_index: int,
    persona: str,
    method: str,
    prev_action: Optional[str] = None,
    daily_return: float = 0.0,
    cum_return: float = 0.0,
    prev_price: Optional[float] = None,
    retrieved_memory: Optional[dict] = None,
    use_memory: bool = False,
) -> Tuple[str, str]:
    """
    Construct (system_message, user_message) for the current trading step.

    Parameters
    ----------
    row : pandas Series or dict-like
        DataFrame row for the current day.
    day_index : int
        0-based position in the trajectory.
    persona : str
        One of the six persona keys defined in PERSONAS.
    method : str
        Injection method: "zero_shot", "one_shot", or "cot".
    prev_action : str or None
        The action taken on the previous day.
    daily_return : float
        Return earned from the previous day's action (log return).
    cum_return : float
        Cumulative log return accumulated so far.
    prev_price : float or None
        Yesterday's closing price (used to derive price change).
    retrieved_memory : dict or None
        Memory record retrieved from MemoryStore, or None.
    use_memory : bool
        Whether memory injection is active.

    Returns
    -------
    (system_message, user_message) : tuple of str
    """
    if persona not in PERSONAS:
        raise ValueError(
            f"Unknown persona '{persona}'. Valid options: {list(PERSONAS.keys())}"
        )
    if method not in {"zero_shot", "one_shot", "cot"}:
        raise ValueError(
            f"Unknown method '{method}'. Valid options: zero_shot, one_shot, cot"
        )

    # The three benchmark personas share the primary mixed-evidence example.
    # Reuse its exact system and daily prompts rather than duplicate them here.
    if method == "one_shot" and persona in {
        "overconfidence", "loss_aversion", "herding"
    }:
        if use_memory:
            raise ValueError("The primary One-Shot protocol uses memory disabled.")
        from matched_one_shot.common import load_protocol
        from matched_one_shot.prompt import build_prompt as build_one_shot_prompt

        return build_one_shot_prompt(
            row=row,
            day_index=day_index,
            persona=persona,
            scenario="M",
            protocol=load_protocol(),
            prev_action=prev_action,
            previous_action_return=daily_return,
            cumulative_log_return=cum_return,
            prev_price=prev_price,
        )

    # ------------------------------------------------------------------ #
    # System message
    # ------------------------------------------------------------------ #
    persona_text = PERSONAS[persona]

    if method == "zero_shot":
        system_message = persona_text

    elif method == "one_shot":
        example = ONE_SHOT_EXAMPLES[persona]
        system_message = f"{persona_text}\n\n{example}"

    else:  # cot
        system_message = f"{persona_text}\n\n{COT_INSTRUCTION}"

    # ------------------------------------------------------------------ #
    # User message — assemble in order
    # ------------------------------------------------------------------ #
    date_str = str(row["date"])[:10]  # YYYY-MM-DD
    symbol = str(row["symbol"])
    close_price = float(row["close_price"])

    parts = []

    # 1. Date and symbol header
    parts.append(f"Today is {date_str}. Stock: {symbol}.")

    # 2. Price line
    if day_index == 0 or prev_price is None:
        parts.append(f"Current price: {close_price:.2f}.")
    else:
        price_change = close_price - prev_price
        price_change_str = format_pct(price_change / prev_price) if prev_price != 0 else "+0.00%"
        parts.append(
            f"Current price: {close_price:.2f}. "
            f"Price change from yesterday: {price_change_str}."
        )

    # 3. Return feedback (day_index >= 1)
    if day_index >= 1 and prev_action is not None:
        daily_return_str = format_pct(daily_return)
        cum_return_str = format_pct(cum_return)
        parts.append(
            f"Your return from yesterday's {prev_action} decision: "
            f"{daily_return_str}. "
            f"Cumulative log return so far: {cum_return_str}."
        )

    # 4. News
    news_list = row.get("news") if hasattr(row, "get") else getattr(row, "news", None)
    news_text = safe_join_list(news_list).strip()
    if news_text:
        parts.append(f"Today's news:\n{news_text}")

    # 5. SEC filings (only if non-empty)
    sec_list = row.get("sec_filing") if hasattr(row, "get") else getattr(row, "sec_filing", None)
    sec_text = safe_join_list(sec_list).strip()
    if sec_text:
        parts.append(f"SEC filings:\n{sec_text}")

    # 6. Memory injection
    if use_memory and retrieved_memory is not None:
        mem_news = safe_join_list(retrieved_memory.get("news", [])).strip()
        mem_block = (
            "Similar past decision:\n"
            f"Date: {retrieved_memory.get('date', 'N/A')}\n"
            f"News: {mem_news if mem_news else 'N/A'}\n"
            f"Action: {retrieved_memory.get('action', 'N/A')}\n"
            f"Rationale: {retrieved_memory.get('rationale', 'N/A')}\n"
            f"Return: {format_pct(float(retrieved_memory.get('daily_return', 0.0)))}\n"
            f"Cumulative log return: {format_pct(float(retrieved_memory.get('cum_return', 0.0)))}"
        )
        parts.append(mem_block)

    # 7. Final JSON instruction
    parts.append(JSON_FORMAT_INSTRUCTION)

    user_message = "\n\n".join(parts)
    return system_message, user_message
