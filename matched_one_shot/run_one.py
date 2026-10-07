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
    __package__ = "matched_one_shot"

from .common import (
    atomic_temp_path,
    all_conditions,
    code_hashes,
    load_examples,
    load_models,
    load_personas,
    load_protocol,
    output_paths,
    runtime_versions,
    sha256_text,
    write_json,
)
from .data import load_frozen_symbol
from .model_clients import create_client
from .parsing import parse_or_repair
from .prompt import build_prompt, build_system_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen matched One-Shot trajectory")
    parser.add_argument("--model", required=True, help="Model key from common.py")
    parser.add_argument("--condition-index", type=int, choices=range(90))
    parser.add_argument("--symbol", choices=["AAPL", "TSLA"])
    parser.add_argument(
        "--persona", choices=["overconfidence", "loss_aversion", "herding"]
    )
    parser.add_argument("--scenario", choices=["M", "P", "N"])
    parser.add_argument("--run-id", type=int, choices=range(1, 6))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print prompt/config hashes without loading or calling a model.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Write to smoke_outputs instead of formal outputs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Limit steps for smoke tests only; forbidden for formal runs.",
    )
    return parser.parse_args()


def resolve_condition(args: argparse.Namespace, protocol: dict) -> dict:
    direct = [args.symbol, args.persona, args.scenario, args.run_id]
    if args.condition_index is not None:
        if any(value is not None for value in direct):
            raise ValueError(
                "--condition-index cannot be combined with --symbol/--persona/--scenario/--run-id"
            )
        return all_conditions(protocol)[args.condition_index]
    if any(value is None for value in direct):
        raise ValueError(
            "Provide --condition-index, or all of --symbol --persona --scenario --run-id"
        )
    return {
        "symbol": args.symbol,
        "persona": args.persona,
        "scenario": args.scenario,
        "run_id": int(args.run_id),
        "seed": int(protocol["seeds"][str(args.run_id)]),
        "condition_index": next(
            item["condition_index"]
            for item in all_conditions(protocol)
            if all(
                item[key] == value
                for key, value in {
                    "symbol": args.symbol,
                    "persona": args.persona,
                    "scenario": args.scenario,
                    "run_id": args.run_id,
                }.items()
            )
        ),
    }


def validate_condition(model_key: str, condition: dict, protocol: dict, models: dict) -> dict:
    if model_key not in models:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {sorted(models)}")
    for key in ["symbol", "persona", "scenario"]:
        plural = {"symbol": "symbols", "persona": "personas", "scenario": "scenarios"}[key]
        if condition[key] not in protocol[plural]:
            raise ValueError(f"Invalid {key}: {condition[key]}")
    if condition["run_id"] not in protocol["run_ids"]:
        raise ValueError(f"Invalid run_id: {condition['run_id']}")
    if condition["seed"] != int(protocol["seeds"][str(condition["run_id"])]):
        raise ValueError("Condition seed does not match frozen run-to-seed mapping")
    model_cfg = models[model_key]
    for field in ["temperature", "top_p"]:
        if float(model_cfg[field]) != float(protocol["generation"][field]):
            raise ValueError(f"{model_key}: {field} differs from protocol")
    return model_cfg


def main() -> int:
    args = parse_args()
    if args.max_steps is not None and not args.smoke:
        raise ValueError("--max-steps is permitted only together with --smoke")
    protocol = load_protocol()
    models = load_models()
    condition = resolve_condition(args, protocol)
    model_cfg = validate_condition(args.model, condition, protocol, models)
    df, data_provenance = load_frozen_symbol(condition["symbol"], protocol)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        df = df.head(args.max_steps).reset_index(drop=True)

    system_message = build_system_prompt(
        condition["persona"], condition["scenario"], protocol
    )
    first_system, first_user = build_prompt(
        df.iloc[0],
        0,
        condition["persona"],
        condition["scenario"],
        protocol,
        None,
        0.0,
        0.0,
        None,
    )
    if first_system != system_message:
        raise RuntimeError("Prompt builder returned inconsistent system messages")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "condition": condition,
                    "model": args.model,
                    "data": data_provenance,
                    "system_message": system_message,
                    "system_message_sha256": sha256_text(system_message),
                    "first_user_message_sha256": sha256_text(first_user),
                    "generation": {
                        key: model_cfg[key]
                        for key in ["temperature", "top_p", "max_new_tokens"]
                    },
                    "formal_output_written": False,
                },
                indent=2,
            )
        )
        return 0

    paths = output_paths(
        args.model,
        condition["symbol"],
        condition["persona"],
        condition["scenario"],
        condition["run_id"],
        condition["seed"],
        smoke=args.smoke,
    )
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

        system_message, user_message = build_prompt(
            row,
            int(day_index),
            condition["persona"],
            condition["scenario"],
            protocol,
            previous_action,
            previous_action_return,
            cumulative_return,
            previous_price,
        )
        generation = client.generate(system_message, user_message, condition["seed"])
        date = row["date"].strftime("%Y-%m-%d")
        parsed = parse_or_repair(
            client,
            generation,
            date,
            condition["symbol"],
            condition["seed"],
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
                    "request_seed": condition["seed"],
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
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
                "symbol": condition["symbol"],
                "scenario": condition["scenario"],
                "persona": condition["persona"],
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

    for index in range(len(records) - 1):
        records[index]["realized_next_log_return"] = records[index + 1][
            "previous_action_return"
        ]

    _write_csv(paths["trajectory"], records)
    _write_jsonl(paths["calls"], calls)
    actions = [record["action"] for record in records]
    parse_counts = {
        status: sum(record["parse_status"] == status for record in records)
        for status in ["direct", "repaired", "fallback"]
    }
    examples = load_examples()
    summary = {
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["protocol_version"],
        "condition": {
            **condition,
            "method": "one_shot_matched",
            "model_key": args.model,
            "reported_model": model_cfg["reported_name"],
            "request_model": model_cfg["request_model"],
            "backend": model_cfg["backend"],
        },
        "scenario_role": examples[condition["scenario"]]["role"],
        "demonstration_action": examples[condition["scenario"]]["decisions"][
            condition["persona"]
        ]["action"],
        "trajectory_length": len(records),
        "consequential_actions": max(len(records) - 1, 0),
        "final_cumulative_log_return": cumulative_return,
        "action_list": actions,
        "action_counts_all_observations": {
            action: actions.count(action) for action in ["BUY", "HOLD", "SELL"]
        },
        "parse_counts": parse_counts,
        "system_message_sha256": sha256_text(system_message),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": args.smoke,
        "paths": {key: str(path) for key, path in paths.items() if key != "base"},
    }
    write_json(paths["summary"], summary)
    manifest = {
        "protocol": protocol,
        "model_config": model_cfg,
        "condition": condition,
        "data_provenance": data_provenance,
        "prompt": {
            "system_message": system_message,
            "system_message_sha256": sha256_text(system_message),
            "example_market_block_sha256": sha256_text(
                examples[condition["scenario"]]["market_block"]
            ),
        },
        "settings_hashes": {
            "protocol": sha256_text(json.dumps(protocol, sort_keys=True)),
            "models": sha256_text(json.dumps(models, sort_keys=True)),
            "personas": sha256_text(json.dumps(load_personas(), sort_keys=True)),
            "examples": sha256_text(json.dumps(examples, sort_keys=True)),
        },
        "code_hashes": code_hashes(),
        "runtime": runtime_versions(),
        "credential_values_stored": False,
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps(summary, indent=2))
    return 0


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError("Cannot write an empty trajectory")
    temporary = atomic_temp_path(path)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = atomic_temp_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
