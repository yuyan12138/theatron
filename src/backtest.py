"""
backtest.py — Single-trajectory execution engine for the Chameleon framework.

Runs one (symbol, persona, method, model, seed) experimental condition
end-to-end, writing a CSV trajectory and a JSON summary to the outputs/
directories.
"""

import json
import logging
import math
import os
from typing import Optional

import pandas as pd

from src.data_loader import load_symbol_data
from src.memory_store import MemoryStore
from src.prompt_builder import build_prompt
from src.trading_agent import TradingAgent
from src.utils import format_pct, safe_join_list

logger = logging.getLogger(__name__)

# Base directory relative to this file's location
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAJECTORIES_DIR = os.path.join(_REPO_ROOT, "outputs", "trajectories")
_SUMMARIES_DIR = os.path.join(_REPO_ROOT, "outputs", "summaries")


def _build_filename_base(config: dict) -> str:
    """
    Construct the base filename (without extension) shared by the CSV and JSON.

    Pattern: {model}_{symbol}_{persona}_{method}_run{run_id}_seed{random_seed}_memory-{true/false}
    """
    model_tag = config.get("model", "unknown").replace("/", "_").replace(":", "_")
    symbol = config.get("symbol", "UNKNOWN")
    persona = config.get("persona", "unknown")
    method = config.get("method", "zero_shot")
    run_id = config.get("run_id", 1)
    random_seed = config.get("random_seed", 42)
    use_memory = config.get("use_memory", False)
    memory_tag = "true" if use_memory else "false"

    return (
        f"{model_tag}_{symbol}_{persona}_{method}"
        f"_run{run_id}_seed{random_seed}_memory-{memory_tag}"
    )


def run_trajectory(config: dict) -> dict:
    """
    Execute a full trading trajectory for the condition specified in *config*.

    Steps:
        1. Load symbol data
        2. Initialise MemoryStore
        3. Initialise TradingAgent
        4. Step through each trading day:
           - Compute returns from previous action
           - Retrieve memory (if enabled)
           - Build prompt
           - Agent decides
           - Append to trajectory list
           - Store memory record
           - Advance state
        5. Write trajectory CSV
        6. Write summary JSON
        7. Return summary dict

    Parameters
    ----------
    config : dict
        Must contain at minimum: symbol, persona, method, model, backend.
        See run_experiment.py for the full argument list.

    Returns
    -------
    dict
        Summary record matching the fields documented in the spec.
    """
    symbol = config["symbol"]
    persona = config["persona"]
    method = config["method"]
    model_name = config.get("model", "unknown")
    backend = config.get("backend", "api")
    api_provider = config.get("api_provider", "openai")
    local_backend = config.get("local_backend", "")
    run_id = config.get("run_id", 1)
    random_seed = config.get("random_seed", 42)
    use_memory = bool(config.get("use_memory", False))

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    logger.info(
        "Loading data for symbol='%s'", symbol
    )
    df = load_symbol_data(symbol)

    # Optional step cap for testing
    max_steps = config.get("max_steps")
    if max_steps is not None and max_steps > 0:
        df = df.head(int(max_steps)).reset_index(drop=True)
        logger.info("max_steps=%d: truncating trajectory to %d days", max_steps, len(df))

    # ------------------------------------------------------------------ #
    # 2. Initialise MemoryStore
    # ------------------------------------------------------------------ #
    memory_config = dict(config)
    memory_config["memory_db_path"] = config.get(
        "memory_db_path",
        os.path.join(_REPO_ROOT, "outputs", "memory"),
    )
    memory = MemoryStore(memory_config)

    # ------------------------------------------------------------------ #
    # 3. Initialise TradingAgent
    # ------------------------------------------------------------------ #
    logger.info(
        "Initialising TradingAgent (backend=%s, model=%s)", backend, model_name
    )
    agent = TradingAgent(config)

    # ------------------------------------------------------------------ #
    # 4. Trajectory loop
    # ------------------------------------------------------------------ #
    trajectory_records = []

    prev_action: Optional[str] = None
    prev_price: Optional[float] = None
    daily_return: float = 0.0
    cum_return: float = 0.0

    for day_index, row in df.iterrows():
        current_price = float(row["close_price"])
        date_str = str(row["date"])[:10]

        # Compute return from previous step
        if day_index == 0:
            daily_return = 0.0
            cum_return = 0.0
        else:
            if prev_price is not None and prev_price > 0:
                stock_return = math.log(current_price / prev_price)
            else:
                stock_return = 0.0

            if prev_action == "BUY":
                daily_return = stock_return
            elif prev_action == "SELL":
                daily_return = -stock_return
            else:  # HOLD or None
                daily_return = 0.0

            cum_return += daily_return

        # Compute price_change for CSV
        if prev_price is not None and prev_price > 0:
            price_change = current_price - prev_price
        else:
            price_change = float("nan")

        # Retrieve memory
        retrieved_memory = None
        memory_used = False
        if memory.is_enabled():
            query_news = safe_join_list(row.get("news"))
            candidates = memory.retrieve(query_news, date_str)
            if candidates:
                retrieved_memory = candidates[0]
                memory_used = True

        # Build prompt
        system_msg, user_msg = build_prompt(
            row=row,
            day_index=day_index,
            persona=persona,
            method=method,
            prev_action=prev_action,
            daily_return=daily_return,
            cum_return=cum_return,
            prev_price=prev_price,
            retrieved_memory=retrieved_memory,
            use_memory=use_memory,
        )

        # Agent decides
        decision = agent.decide(
            system_msg=system_msg,
            user_msg=user_msg,
            date_str=date_str,
            symbol=symbol,
        )
        action = decision.get("action", "HOLD")
        rationale = decision.get("rationale", "")
        confidence = float(decision.get("confidence", 0.0))

        # Build the trajectory record
        news_list = row.get("news") if hasattr(row, "get") else getattr(row, "news", [])
        news_str = safe_join_list(news_list)
        if len(news_str) > 200:
            news_str = news_str[:200]

        record = {
            "date": date_str,
            "symbol": symbol,
            "price": current_price,
            "price_change": price_change,
            "news": news_str,
            "action": action,
            "rationale": rationale,
            "confidence": confidence,
            "daily_return": daily_return,
            "cumulative_log_return": cum_return,
            "memory_used": memory_used,
            "retrieved_memory_date": retrieved_memory.get("date", "") if retrieved_memory else "",
            "retrieved_memory_news": retrieved_memory.get("news", "") if retrieved_memory else "",
            "retrieved_memory_action": retrieved_memory.get("action", "") if retrieved_memory else "",
        }
        trajectory_records.append(record)

        # Store in memory
        if memory.is_enabled():
            memory.add({
                "date": date_str,
                "symbol": symbol,
                "news": news_list,
                "action": action,
                "rationale": rationale,
                "daily_return": daily_return,
                "cum_return": cum_return,
            })

        # Advance state
        prev_action = action
        prev_price = current_price

        logger.info(
            "Day %d | %s | price=%.2f | action=%s | daily_ret=%s | cum_ret=%s",
            day_index,
            date_str,
            current_price,
            action,
            format_pct(daily_return),
            format_pct(cum_return),
        )

    # ------------------------------------------------------------------ #
    # 5. Save trajectory CSV
    # ------------------------------------------------------------------ #
    os.makedirs(_TRAJECTORIES_DIR, exist_ok=True)
    filename_base = _build_filename_base(config)
    csv_path = os.path.join(_TRAJECTORIES_DIR, f"{filename_base}.csv")

    traj_df = pd.DataFrame(trajectory_records)
    traj_df.to_csv(csv_path, index=False)
    logger.info("Trajectory saved to %s", csv_path)

    # ------------------------------------------------------------------ #
    # 6. Build and save summary JSON
    # ------------------------------------------------------------------ #
    os.makedirs(_SUMMARIES_DIR, exist_ok=True)
    json_path = os.path.join(_SUMMARIES_DIR, f"{filename_base}.json")

    action_list = [r["action"] for r in trajectory_records]
    buy_count = action_list.count("BUY")
    hold_count = action_list.count("HOLD")
    sell_count = action_list.count("SELL")
    final_cum_return = trajectory_records[-1]["cumulative_log_return"] if trajectory_records else 0.0

    summary = {
        "symbol": symbol,
        "persona": persona,
        "method": method,
        "model": model_name,
        "backend": backend,
        "api_provider": api_provider,
        "local_backend": local_backend,
        "run_id": run_id,
        "random_seed": random_seed,
        "use_memory": use_memory,
        "CR": final_cum_return,
        "action_list": action_list,
        "buy_count": buy_count,
        "hold_count": hold_count,
        "sell_count": sell_count,
        "trajectory_length": len(trajectory_records),
        "trajectory_csv_path": csv_path,
        "summary_json_path": json_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", json_path)

    return summary
