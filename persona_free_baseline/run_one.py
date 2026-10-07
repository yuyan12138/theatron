from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "persona_free_baseline"

from .common import (
    code_hashes,
    load_models,
    load_personas,
    load_protocol,
    output_paths,
    runtime_versions,
    sha256_json,
    sha256_text,
    write_json,
)
from .data import load_frozen_symbol
from .model_clients import create_client
from .parsing import parse_or_repair
from .prompt import build_persona_free_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one task-only or Zero-Shot persona trajectory")
    parser.add_argument("--model", required=True, help="Model key from common.py")
    parser.add_argument("--symbol", required=True, choices=["AAPL", "TSLA"])
    parser.add_argument("--run-id", required=True, type=int, choices=range(1, 6))
    parser.add_argument(
        "--persona", choices=["overconfidence", "loss_aversion", "herding"],
        help="Add a Zero-Shot persona; omit for the task-only baseline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/settings and print prompt hashes without loading/calling a model",
    )
    return parser.parse_args()


def validate_condition(model_key: str, symbol: str, run_id: int, protocol: dict, models: dict):
    if model_key not in models:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {sorted(models)}")
    if symbol not in protocol["symbols"]:
        raise ValueError(f"Symbol {symbol} is not frozen in the protocol")
    if run_id not in protocol["run_ids"]:
        raise ValueError(f"run_id {run_id} is not frozen in the protocol")
    model_cfg = models[model_key]
    generation = protocol["generation"]
    for field in ["temperature", "top_p"]:
        if float(model_cfg[field]) != float(generation[field]):
            raise ValueError(f"{model_key}: {field} differs from protocol")
    message = protocol["prompt"]["system_message"].lower()
    present = [term for term in protocol["prompt"]["persona_terms_forbidden_in_system_message"] if term.lower() in message]
    if present:
        raise ValueError(f"Persona-bearing terms found in system prompt: {present}")
    return model_cfg


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    models = load_models()
    personas = load_personas()
    model_cfg = validate_condition(args.model, args.symbol, args.run_id, protocol, models)
    seed = int(protocol["seeds"][str(args.run_id)])
    df, data_provenance = load_frozen_symbol(args.symbol, protocol)

    first_system, first_user = build_persona_free_prompt(
        df.iloc[0], 0, protocol, None, 0.0, 0.0, None
    )
    if args.persona is not None:
        first_system = personas[args.persona]
    if args.dry_run:
        last_system, last_user = build_persona_free_prompt(
            df.iloc[-1], len(df) - 1, protocol, "HOLD", 0.0, 0.0, float(df.iloc[-2]["close_price"])
        )
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "condition": {"model": args.model, "symbol": args.symbol, "persona": args.persona, "method": "zero_shot" if args.persona else "task_only", "run_id": args.run_id, "seed": seed},
                    "data": data_provenance,
                    "system_message": first_system,
                    "first_user_prompt_sha256": sha256_text(first_user),
                    "last_user_prompt_sha256_for_neutral_state": sha256_text(last_user),
                    "generation": {
                        key: model_cfg[key]
                        for key in ["temperature", "top_p", "max_new_tokens"]
                    },
                },
                indent=2,
            )
        )
        return 0

    paths = output_paths(args.model, args.symbol, args.run_id, seed, args.persona)
    collisions = [str(path) for key, path in paths.items() if key != "base" and path.exists()]
    if collisions:
        raise RuntimeError("Refusing to overwrite existing outputs:\n" + "\n".join(collisions))
    paths["base"].mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    client = create_client(model_cfg)
    records: list[dict] = []
    calls: list[dict] = []
    previous_action = None
    previous_price = None
    cumulative_return = 0.0

    for day_index, row in df.iterrows():
        price = float(row["close_price"])
        previous_action_return = 0.0
        if previous_price is not None:
            stock_log_return = math.log(price / previous_price)
            exposure = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}[previous_action]
            previous_action_return = exposure * stock_log_return
            cumulative_return += previous_action_return

        system_message, user_message = build_persona_free_prompt(
            row=row,
            day_index=day_index,
            protocol=protocol,
            prev_action=previous_action,
            previous_action_return=previous_action_return,
            cumulative_log_return=cumulative_return,
            prev_price=previous_price,
        )
        if args.persona is not None:
            system_message = personas[args.persona]
        generation = client.generate(system_message, user_message, seed)
        date = row["date"].strftime("%Y-%m-%d")
        parsed = parse_or_repair(
            client,
            generation,
            date,
            args.symbol,
            seed,
            protocol,
            first_system_message=system_message,
            first_user_message=user_message,
        )

        for attempt in parsed.attempts:
            call_generation = attempt.generation
            calls.append(
                {
                    "decision_index": int(day_index),
                    "date": date,
                    "call_kind": attempt.kind,
                    "request_seed": seed,
                    "system_message": attempt.system_message,
                    "user_message": attempt.user_message,
                    "system_message_sha256": sha256_text(attempt.system_message),
                    "user_message_sha256": sha256_text(attempt.user_message),
                    "raw_response": call_generation.text,
                    "raw_response_sha256": sha256_text(call_generation.text),
                    "provider_metadata": call_generation.metadata,
                }
            )

        decision = parsed.record
        price_change = None if previous_price is None else price / previous_price - 1.0
        records.append(
            {
                "decision_index": int(day_index),
                "date": date,
                "symbol": args.symbol,
                "price": price,
                "price_change_fraction": price_change,
                "previous_action_return": previous_action_return,
                "cumulative_log_return_before_decision": cumulative_return,
                "action": decision["action"],
                "rationale": decision["rationale"],
                "confidence": decision["confidence"],
                "parse_status": parsed.status,
                "system_message_sha256": sha256_text(system_message),
                "user_message_sha256": sha256_text(user_message),
                "realized_next_log_return": None,
            }
        )
        previous_action = decision["action"]
        previous_price = price

    # Attach the return caused by decision i at observation i+1. The final action
    # has no subsequent observed price and is intentionally marked missing.
    for index in range(len(records) - 1):
        records[index]["realized_next_log_return"] = records[index + 1]["previous_action_return"]

    _write_csv(paths["trajectory"], records)
    _write_jsonl(paths["calls"], calls)
    action_list = [record["action"] for record in records]
    parse_counts = {status: sum(record["parse_status"] == status for record in records) for status in ["direct", "repaired", "fallback"]}
    summary = {
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["protocol_version"],
        "condition": {
            "persona": args.persona,
            "method": "zero_shot" if args.persona else "task_only",
            "model_key": args.model,
            "reported_model": model_cfg["reported_name"],
            "request_model": model_cfg["request_model"],
            "backend": model_cfg["backend"],
            "symbol": args.symbol,
            "run_id": args.run_id,
            "seed": seed,
        },
        "trajectory_length": len(records),
        "consequential_actions": len(records) - 1,
        "final_cumulative_log_return": cumulative_return,
        "action_list": action_list,
        "action_counts_all_102": {action: action_list.count(action) for action in ["BUY", "HOLD", "SELL"]},
        "parse_counts": parse_counts,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {key: str(path) for key, path in paths.items() if key != "base"},
    }
    write_json(paths["summary"], summary)
    manifest = {
        "protocol": protocol,
        "model_config": model_cfg,
        "persona_system_message": personas[args.persona] if args.persona else None,
        "data_provenance": data_provenance,
        "settings_hashes": {
            "protocol": sha256_json(protocol),
            "models": sha256_json(models),
        },
        "code_hashes": code_hashes(),
        "runtime": runtime_versions(),
        "credential_values_stored": False,
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps(summary, indent=2))
    return 0


def _write_csv(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
