from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Fixed experimental settings are embedded; no external config files are needed.
PROTOCOL = {
    "protocol_name": "theatron-aapl-r2-official-zero-shot-v1",
    "protocol_version": 1,
    "purpose": "A matched, non-overlapping AAPL Zero-Shot extension over the official-source R2 evidence stream. It tests whether persona-conditioned separation is detectable on a second AAPL price path.",
    "method": "zero_shot",
    "symbols": [
        "AAPL"
    ],
    "conditions": [
        "task_only",
        "overconfidence",
        "loss_aversion",
        "herding"
    ],
    "personas": [
        "overconfidence",
        "loss_aversion",
        "herding"
    ],
    "run_ids": [
        1,
        2,
        3,
        4,
        5
    ],
    "seeds": {
        "1": 41,
        "2": 42,
        "3": 43,
        "4": 44,
        "5": 45
    },
    "matrix": {
        "models": [
            "qwen3-8b-nothink",
            "qwen3-14b-thinking",
            "gpt-5.4-mini",
            "gemini-2.5-flash"
        ],
        "conditions_per_model": 20,
        "expected_trajectories": 80,
        "expected_decisions": 8160,
        "expected_consequential_actions": 8080
    },
    "data": {
        "source_dir": "data",
        "source_file": "AAPL-R2-2025-06-01_2025-10-25.parquet",
        "start_date": "2025-06-02",
        "end_date": "2025-10-24",
        "expected_raw_rows": 147,
        "expected_trading_days": 102,
        "calendar": "NYSE",
        "roll_nontrading_news_forward": True,
        "forbidden_input_columns": [
            "momentum",
            "future_price_diff"
        ],
        "raw_sha256": {
            "AAPL": "473a002fc8e7d6403538d0f32b420d64ba8b9700c039c83eb635c7ed15f3cfda"
        },
        "processed_date_price_sha256": {
            "AAPL": "d1515026368e3547c9641e46da2b9083558aa934d11d064e60c20c1d51a46be7"
        },
        "text_evidence_policy": "official_apple_newsroom_and_sec_point_in_time",
        "news_cells_rebuilt": 147,
        "empty_raw_news_rows": 0
    },
    "prompt": {
        "version": "aapl-r2-matched-zero-shot-v1",
        "task_only_system_message": "You are a stock trading agent. Make each decision using only the information provided.",
        "json_instruction": "Respond ONLY with a JSON object in this exact format:\n{\"date\": \"YYYY-MM-DD\", \"symbol\": \"XXX\", \"action\": \"BUY/HOLD/SELL\", \"rationale\": \"...\", \"confidence\": 0.0}",
        "forbidden_model_visible_terms": [
            "task_only",
            "experimental condition",
            "AAPL R2",
            "regime",
            "window comparison",
            "baseline comparison"
        ]
    },
    "environment": {
        "actions": [
            "BUY",
            "HOLD",
            "SELL"
        ],
        "buy_exposure": 1,
        "hold_exposure": 0,
        "sell_exposure": -1,
        "return_type": "log_return",
        "feedback": [
            "previous_action_return",
            "cumulative_log_return"
        ],
        "memory": False,
        "transaction_cost": 0,
        "position_inventory": False,
        "observations": 102,
        "consequential_actions": 101
    },
    "generation": {
        "temperature": 1,
        "top_p": 0.9,
        "do_sample_local": True,
        "seed_policy": "rollout_seed_reused_for_each_request",
        "api_seed_semantics": "best_effort_when_supported",
        "repair_attempts": 1,
        "repair_uses_same_seed": True,
        "fallback_action": "HOLD"
    },
    "analysis": {
        "analysis_seed": 20260915,
        "bootstrap_repetitions": 10000,
        "global_bw_permutations": 10000,
        "baseline_persona_test": "exact_all_C_10_5_labelings",
        "return_distance": "mean_absolute_cumulative_log_return_distance_over_101_realized_points",
        "action_distance_primary": "normalized_hamming_over_101_consequential_actions",
        "action_distance_secondary": "normalized_hamming_over_102_recorded_decisions",
        "epsilon": 1e-9,
        "bh_fdr_families": [
            "AAPL_R2_baseline_vs_persona_return_12_tests",
            "AAPL_R2_baseline_vs_persona_action_12_tests"
        ]
    },
    "output": {
        "write_exact_prompts": True,
        "write_raw_responses": True,
        "refuse_overwrite": True,
        "formal_output_root": "outputs",
        "smoke_output_root": "smoke_outputs",
        "run_label": "official-news-aapl-r2-v1"
    }
}

MODELS = {
    "qwen3-8b-nothink": {
        "reported_name": "Qwen3-8B",
        "request_model": "Qwen3-8B-nothink",
        "backend": "local",
        "local_model_path": "/your/path/Qwen3-8B",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "enable_thinking": False,
        "max_new_tokens": 512,
        "temperature": 1,
        "top_p": 0.9
    },
    "qwen3-14b-thinking": {
        "reported_name": "Qwen3-14B",
        "request_model": "Qwen3-14B-thinking",
        "backend": "local",
        "local_model_path": "/your/path/Qwen3-14B",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "enable_thinking": True,
        "max_new_tokens": 8192,
        "temperature": 1,
        "top_p": 0.9
    },
    "gpt-5.4-mini": {
        "reported_name": "GPT-5.4-mini",
        "request_model": "gpt-5.4-mini-2026-03-17",
        "backend": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "official_endpoint_only": True,
        "max_new_tokens": 512,
        "reasoning_effort": "none",
        "inter_request_sleep_seconds": 7,
        "temperature": 1,
        "top_p": 0.9,
        "seed_support": "best_effort"
    },
    "gemini-2.5-flash": {
        "reported_name": "Gemini-2.5-Flash",
        "request_model": "gemini-2.5-flash",
        "backend": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "official_endpoint_only": True,
        "thinking_budget": 0,
        "inter_request_sleep_seconds": 7,
        "transient_503_max_retries": 5,
        "transient_503_initial_backoff_seconds": 30,
        "transient_503_max_backoff_seconds": 300,
        "max_new_tokens": 2048,
        "temperature": 1,
        "top_p": 0.9,
        "seed_support": "best_effort"
    },
    "mock-deterministic": {
        "reported_name": "Mock deterministic backend",
        "request_model": "mock-deterministic",
        "backend": "mock",
        "max_new_tokens": 32,
        "temperature": 1,
        "top_p": 0.9
    }
}

PERSONAS = {
    "overconfidence": "You are an overconfident stock trader. You trust your own signals implicitly, tend to underestimate risk, and have a strong tendency to overtrade. You believe your read of the market is sharper than most, so you act decisively and aggressively on perceived opportunities. You rarely second-guess yourself and interpret ambiguous signals as confirmation of your existing view.",
    "loss_aversion": "You are a loss-averse stock trader. Losses feel significantly more painful to you than equivalent gains feel rewarding. You are highly defensive and quick to lock in profits to avoid watching them evaporate. You actively avoid drawdowns, often exiting positions early, and you hold losing trades less comfortably than winning ones. Protecting capital from loss is your primary emotional drive.",
    "herding": "You are a herding-prone stock trader. You follow the crowd and market consensus closely. When sentiment is positive or prices are rising, you feel compelled to buy so you do not miss out. When sentiment turns negative or prices are falling, you sell to avoid being left behind. You find comfort in doing what the majority appears to be doing and distrust contrarian positions."
}


def load_protocol() -> dict[str, Any]:
    return deepcopy(PROTOCOL)


def load_models() -> dict[str, Any]:
    models = deepcopy(MODELS)
    for key, env_name in [
        ("qwen3-8b-nothink", "THEATRON_QWEN3_8B_PATH"),
        ("qwen3-14b-thinking", "THEATRON_QWEN3_14B_PATH"),
    ]:
        models[key]["local_model_path"] = os.environ.get(
            env_name, models[key]["local_model_path"]
        )
    return models


def load_personas() -> dict[str, str]:
    return deepcopy(PERSONAS)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def model_tag(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_key)


def all_conditions(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "condition": condition,
            "run_id": int(run_id),
            "seed": int(protocol["seeds"][str(run_id)]),
        }
        for symbol in protocol["symbols"]
        for condition in protocol["conditions"]
        for run_id in protocol["run_ids"]
    ]


def run_stem(
    model_key: str,
    symbol: str,
    condition: str,
    run_id: int,
    seed: int,
) -> str:
    return (
        f"{model_tag(model_key)}_{symbol}_{condition}_zero-shot_"
        f"run{run_id}_seed{seed}_official-news-aapl-r2-v1"
    )


def output_paths(
    model_key: str,
    symbol: str,
    condition: str,
    run_id: int,
    seed: int,
    *,
    smoke: bool = False,
) -> dict[str, Path]:
    stem = run_stem(model_key, symbol, condition, run_id, seed)
    root = ROOT / ("smoke_outputs" if smoke else "outputs")
    base = root / model_tag(model_key) / symbol / condition
    return {
        "base": base,
        "trajectory": base / f"{stem}.csv",
        "summary": base / f"{stem}.summary.json",
        "calls": base / f"{stem}.calls.jsonl",
        "manifest": base / f"{stem}.manifest.json",
    }


def atomic_temp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = atomic_temp_path(path)
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp, path)


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def runtime_versions() -> dict[str, Any]:
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "pandas_market_calendars",
        "scipy",
        "torch",
        "transformers",
        "openai",
        "google-genai",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in packages},
    }


def code_hashes() -> dict[str, str]:
    files = sorted((ROOT / "src").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
