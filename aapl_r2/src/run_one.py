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
    __package__ = "src"

from .common import (
    all_conditions,
    atomic_temp_path,
    code_hashes,
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
from .prompt import build_prompt, system_message_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one frozen AAPL R2 matched Zero-Shot trajectory"
    )
    parser.add_argument("--model", required=True, help="Model key defined in src/common.py")
    parser.add_argument("--condition-index", type=int)
    parser.add_argument("--symbol")
    parser.add_argument("--condition")
    parser.add_argument("--run-id", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print hashes without loading/calling a model.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Write to smoke_outputs rather than formal outputs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Limit observations; permitted only with --smoke.",
    )
    return parser.parse_args()


def resolve_condition(args: argparse.Namespace, protocol: dict) -> dict:
    matrix = all_conditions(protocol)
    direct = [args.symbol, args.condition, args.run_id]
    if args.condition_index is not None:
        if any(value is not None for value in direct):
            raise ValueError(
                "--condition-index cannot be combined with direct condition arguments"
            )
        if not 0 <= args.condition_index < len(matrix):
            raise ValueError(
                f"condition-index must be in [0, {len(matrix) - 1}]"
            )
        return matrix[args.condition_index]
    if any(value is None for value in direct):
        raise ValueError(
            "Provide --condition-index, or all of "
            "--symbol --condition --run-id"
        )
    requested = {
        "symbol": str(args.symbol),
        "condition": str(args.condition),
        "run_id": int(args.run_id),
        "seed": int(protocol["seeds"][str(int(args.run_id))]),
    }
    if requested not in matrix:
        raise ValueError(f"Condition is outside the frozen matrix: {requested}")
    return requested


def validate_condition(
    model_key: str,
    condition: dict,
    protocol: dict,
    models: dict,
    personas: dict,
) -> dict:
    if model_key not in models:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {sorted(models)}")
    if model_key not in protocol["matrix"]["models"] and model_key != "mock-deterministic":
        raise ValueError(f"Model is outside the formal matrix: {model_key}")
    if condition["symbol"] not in protocol["symbols"]:
        raise ValueError(f"Invalid symbol: {condition['symbol']}")
    if condition["condition"] not in protocol["conditions"]:
        raise ValueError(f"Invalid condition: {condition['condition']}")
    if condition["run_id"] not in protocol["run_ids"]:
        raise ValueError(f"Invalid run_id: {condition['run_id']}")
    if condition["seed"] != int(protocol["seeds"][str(condition["run_id"])]):
        raise ValueError("Condition seed differs from the frozen mapping")
    if sorted(personas) != sorted(protocol["personas"]):
        raise ValueError("Persona definitions do not match the protocol")
    model_cfg = models[model_key]
    for field in ["temperature", "top_p"]:
        if float(model_cfg[field]) != float(protocol["generation"][field]):
            raise ValueError(f"{model_key}: {field} differs from protocol")
    return model_cfg


def main() -> int:
    args = parse_args()
    if args.max_steps is not None and not args.smoke:
        raise ValueError("--max-steps is allowed only together with --smoke")
    if args.smoke and args.dry_run:
        raise ValueError("--smoke and --dry-run are mutually exclusive")

    protocol = load_protocol()
    models = load_models()
    personas = load_personas()
    condition = resolve_condition(args, protocol)
    model_cfg = validate_condition(
        args.model, condition, protocol, models, personas
    )
    frame, data_provenance = load_frozen_symbol(
        condition["symbol"], protocol
    )
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        frame = frame.head(args.max_steps).reset_index(drop=True)

    system_message = system_message_for(
        condition["condition"], protocol, personas
    )
    first_system, first_user = build_prompt(
        frame.iloc[0],
        0,
        condition["condition"],
        protocol,
        personas,
        None,
        0.0,
        0.0,
        None,
    )
    if first_system != system_message:
        raise RuntimeError("Prompt builder returned an inconsistent system message")
    for forbidden in protocol["prompt"]["forbidden_model_visible_terms"]:
        if forbidden.lower() in (first_system + "\n" + first_user).lower():
            raise RuntimeError(f"Protocol label leaked into prompt: {forbidden}")

    if args.dry_run:
        last_system, last_user = build_prompt(
            frame.iloc[-1],
            len(frame) - 1,
            condition["condition"],
            protocol,
            personas,
            "HOLD",
            0.0,
            0.0,
            float(frame.iloc[-2]["close_price"]),
        )
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
                    "last_user_message_sha256_for_neutral_state": sha256_text(
                        last_user
                    ),
                    "same_system_at_last_step": last_system == system_message,
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
        condition["condition"],
        condition["run_id"],
        condition["seed"],
        smoke=args.smoke,
    )
    collisions = [
        str(path)
        for key, path in paths.items()
        if key != "base" and path.exists()
    ]
    if collisions:
        raise RuntimeError(
            "Refusing to overwrite existing outputs:\n" + "\n".join(collisions)
        )
    paths["base"].mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    client = create_client(model_cfg)
    records: list[dict] = []
    calls: list[dict] = []
    previous_action: str | None = None
    previous_price: float | None = None
    cumulative_return = 0.0

    for day_index, row in frame.iterrows():
        price = float(row["close_price"])
        previous_action_return = 0.0
        if previous_price is not None:
            stock_log_return = math.log(price / previous_price)
            exposure = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}[
                previous_action
            ]
            previous_action_return = exposure * stock_log_return
            cumulative_return += previous_action_return

        step_system, user_message = build_prompt(
            row,
            int(day_index),
            condition["condition"],
            protocol,
            personas,
            previous_action,
            previous_action_return,
            cumulative_return,
            previous_price,
        )
        if step_system != system_message:
            raise RuntimeError("System message changed within trajectory")
        generation = client.generate(
            system_message, user_message, condition["seed"]
        )
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
            calls.append(
                {
                    "decision_index": int(day_index),
                    "date": date,
                    "call_kind": attempt.kind,
                    "request_seed": condition["seed"],
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "system_message": attempt.system_message,
                    "user_message": attempt.user_message,
                    "system_message_sha256": sha256_text(
                        attempt.system_message
                    ),
                    "user_message_sha256": sha256_text(attempt.user_message),
                    "raw_response": attempt.generation.text,
                    "raw_response_sha256": sha256_text(
                        attempt.generation.text
                    ),
                    "provider_metadata": attempt.generation.metadata,
                }
            )

        decision = parsed.record
        records.append(
            {
                "decision_index": int(day_index),
                "date": date,
                "symbol": condition["symbol"],
                "condition": condition["condition"],
                "price": price,
                "price_change_fraction": (
                    None
                    if previous_price is None
                    else price / previous_price - 1.0
                ),
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
    summary = {
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["protocol_version"],
        "formal_run": not args.smoke,
        "condition": {
            **condition,
            "method": "task_only"
            if condition["condition"] == "task_only"
            else "zero_shot_persona",
            "persona": None
            if condition["condition"] == "task_only"
            else condition["condition"],
            "model_key": args.model,
            "reported_model": model_cfg["reported_name"],
            "request_model": model_cfg["request_model"],
            "backend": model_cfg["backend"],
        },
        "trajectory_length": len(records),
        "consequential_actions": len(records) - 1,
        "final_cumulative_log_return": cumulative_return,
        "action_list": actions,
        "action_counts_all_recorded": {
            action: actions.count(action) for action in ["BUY", "HOLD", "SELL"]
        },
        "parse_counts": parse_counts,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            key: str(path) for key, path in paths.items() if key != "base"
        },
    }
    write_json(paths["summary"], summary)
    manifest = {
        "protocol": protocol,
        "model_config": model_cfg,
        "condition": condition,
        "system_message": system_message,
        "data_provenance": data_provenance,
        "code_hashes": code_hashes(),
        "runtime": runtime_versions(),
        "credential_values_stored": False,
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps(summary, indent=2))
    return 0


def _write_csv(path: Path, records: list[dict]) -> None:
    temp = atomic_temp_path(path)
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temp, path)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    temp = atomic_temp_path(path)
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp, path)


if __name__ == "__main__":
    raise SystemExit(main())
