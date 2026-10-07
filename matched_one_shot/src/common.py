from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

# Fixed experimental settings are embedded to keep the release self-contained.
PROTOCOL = {'protocol_name': 'theatron-matched-one-shot-v1',
 'protocol_version': 1,
 'primary_scenario': 'M',
 'scenarios': ['M', 'P', 'N'],
 'personas': ['overconfidence', 'loss_aversion', 'herding'],
 'symbols': ['AAPL', 'TSLA'],
 'run_ids': [1, 2, 3, 4, 5],
 'seeds': {'1': 41, '2': 42, '3': 43, '4': 44, '5': 45},
 'condition_shuffle_seed': 20260729,
 'data': {'source_dir': '../data',
          'start_date': '2025-01-02',
          'end_date': '2025-05-30',
          'expected_trading_days': 102,
          'calendar': 'NYSE',
          'roll_nontrading_news_forward': True,
          'forbidden_input_columns': ['momentum', 'future_price_diff'],
          'raw_sha256': {'AAPL': '4e8061e12062d6a17c144b8064b562fa13e07437e01d2bf27f3d5a70fed05957',
                         'TSLA': '75be8b97fb2096bff07b2bb468e36f3d7c16d379078d9cfd4b072c7aa16cf1d5'},
          'processed_date_price_sha256': {'AAPL': '0eb6a714048ebb7c64ff65021c889edb910125f98946994a57f380d77cd8108a',
                                          'TSLA': '7a20847fe62cd704659db4c98deead6aab0c8cae73fb5cfe1746e07e06e6d364'}},
 'prompt': {'version': 'matched-one-shot-v1',
            'example_date': '2024-02-15',
            'example_symbol': 'DEMO',
            'introduction': 'The following example illustrates one decision consistent with this '
                            'trading persona.',
            'example_header': 'Example (2024-02-15; generic stock DEMO):',
            'guardrail': 'When making each new decision, apply this trading persona to the current '
                         'market evidence and account feedback. Treat the example as an '
                         'illustration, not as a default action.',
            'forbidden_runtime_terms': ['synthetic',
                                        'matched scenario',
                                        'scenario m',
                                        'scenario p',
                                        'scenario n',
                                        'primary scenario',
                                        'ablation scenario'],
            'json_instruction': 'Respond ONLY with a JSON object in this exact format:\n'
                                '{"date": "YYYY-MM-DD", "symbol": "XXX", "action": '
                                '"BUY/HOLD/SELL", "rationale": "...", "confidence": 0.0}'},
 'environment': {'actions': ['BUY', 'HOLD', 'SELL'],
                 'buy_exposure': 1.0,
                 'hold_exposure': 0.0,
                 'sell_exposure': -1.0,
                 'return_type': 'log_return',
                 'feedback': ['previous_action_return', 'cumulative_log_return'],
                 'memory': False,
                 'transaction_cost': 0.0,
                 'position_inventory': False,
                 'observations': 102,
                 'consequential_actions': 101},
 'generation': {'temperature': 1.0,
                'top_p': 0.9,
                'do_sample_local': True,
                'seed_policy': 'rollout_seed_reused_for_each_request',
                'api_seed_semantics': 'best_effort_when_supported',
                'repair_attempts': 1,
                'repair_uses_same_seed': True,
                'fallback_action': 'HOLD'},
 'output': {'write_exact_prompts': True, 'write_raw_responses': True, 'refuse_overwrite': True}}

MODELS = {'qwen3-8b-nothink': {'reported_name': 'Qwen3-8B',
                      'request_model': 'Qwen3-8B-nothink',
                      'backend': 'local',
                      'local_model_path': 'your_path/Qwen3-8B',
                      'torch_dtype': 'bfloat16',
                      'device_map': 'auto',
                      'enable_thinking': False,
                      'max_new_tokens': 512,
                      'temperature': 1.0,
                      'top_p': 0.9},
 'qwen3-14b-thinking': {'reported_name': 'Qwen3-14B',
                        'request_model': 'Qwen3-14B-thinking',
                        'backend': 'local',
                        'local_model_path': 'your_path/Qwen3-14B',
                        'torch_dtype': 'bfloat16',
                        'device_map': 'auto',
                        'enable_thinking': True,
                        'max_new_tokens': 8192,
                        'temperature': 1.0,
                        'top_p': 0.9},
 'gpt-5.4-mini': {'reported_name': 'GPT-5.4-mini',
                  'request_model': 'gpt-5.4-mini-2026-03-17',
                  'backend': 'openai',
                  'api_key_env': 'OPENAI_API_KEY',
                  'official_endpoint_only': True,
                  'max_new_tokens': 512,
                  'reasoning_effort': 'none',
                  'inter_request_sleep_seconds': 7.0,
                  'temperature': 1.0,
                  'top_p': 0.9,
                  'seed_support': 'best_effort'},
 'gemini-2.5-flash': {'reported_name': 'Gemini-2.5-Flash',
                      'request_model': 'gemini-2.5-flash',
                      'backend': 'gemini',
                      'api_key_env': 'GEMINI_API_KEY',
                      'official_endpoint_only': True,
                      'thinking_budget': 0,
                      'inter_request_sleep_seconds': 7.0,
                      'transient_503_max_retries': 5,
                      'transient_503_initial_backoff_seconds': 30.0,
                      'transient_503_max_backoff_seconds': 300.0,
                      'max_new_tokens': 2048,
                      'temperature': 1.0,
                      'top_p': 0.9,
                      'seed_support': 'best_effort'},
 'mock-deterministic': {'reported_name': 'Mock deterministic backend',
                        'request_model': 'mock-deterministic',
                        'backend': 'mock',
                        'max_new_tokens': 32,
                        'temperature': 1.0,
                        'top_p': 0.9}}

PERSONAS = {'overconfidence': 'You are an overconfident stock trader. You trust your own signals implicitly, '
                   'tend to underestimate risk, and have a strong tendency to overtrade. You '
                   'believe your read of the market is sharper than most, so you act decisively '
                   'and aggressively on perceived opportunities. You rarely second-guess yourself '
                   'and interpret ambiguous signals as confirmation of your existing view.',
 'loss_aversion': 'You are a loss-averse stock trader. Losses feel significantly more painful to '
                  'you than equivalent gains feel rewarding. You are highly defensive and quick to '
                  'lock in profits to avoid watching them evaporate. You actively avoid drawdowns, '
                  'often exiting positions early, and you hold losing trades less comfortably than '
                  'winning ones. Protecting capital from loss is your primary emotional drive.',
 'herding': 'You are a herding-prone stock trader. You follow the crowd and market consensus '
            'closely. When sentiment is positive or prices are rising, you feel compelled to buy '
            'so you do not miss out. When sentiment turns negative or prices are falling, you sell '
            'to avoid being left behind. You find comfort in doing what the majority appears to be '
            'doing and distrust contrarian positions.'}

EXAMPLES = {'M': {'role': 'primary_mixed',
       'market_block': 'Current price: 98.80. Price change from yesterday: -1.20%.\n'
                       '\n'
                       "Your return from yesterday's BUY decision: -1.21%. Cumulative log return "
                       'so far: -0.20%.\n'
                       '\n'
                       "Today's news:\n"
                       'The company reported slightly better-than-expected revenue and maintained '
                       'its full-year guidance. Despite this, the stock fell 1.20% on '
                       'above-average trading volume. Short-term commentary is cautious, analyst '
                       'views are split, and public market sentiment is mildly bearish. No '
                       'material new filing was reported.',
       'decisions': {'overconfidence': {'date': '2024-02-15',
                                        'symbol': 'DEMO',
                                        'action': 'BUY',
                                        'rationale': 'The fundamental update supports an upside '
                                                     'opportunity despite the short-term sell-off. '
                                                     'I trust this reading and will act decisively '
                                                     'with a one-period positive exposure.',
                                        'confidence': 0.8},
                     'loss_aversion': {'date': '2024-02-15',
                                       'symbol': 'DEMO',
                                       'action': 'HOLD',
                                       'rationale': 'The recent loss and conflicting signals make '
                                                    'another directional position too risky. I '
                                                    'will stay neutral for one period to avoid '
                                                    'compounding an avoidable drawdown.',
                                       'confidence': 0.8},
                     'herding': {'date': '2024-02-15',
                                 'symbol': 'DEMO',
                                 'action': 'SELL',
                                 'rationale': 'The falling price, above-average volume, cautious '
                                              'commentary, and bearish sentiment show the crowd '
                                              'moving away. I will follow the prevailing direction '
                                              'with a one-period negative exposure.',
                                 'confidence': 0.8}}},
 'P': {'role': 'ablation_positive_aligned',
       'market_block': 'Current price: 101.80. Price change from yesterday: +1.80%.\n'
                       '\n'
                       "Your return from yesterday's HOLD decision: +0.00%. Cumulative log return "
                       'so far: +0.00%.\n'
                       '\n'
                       "Today's news:\n"
                       'The company reported stronger-than-expected revenue and earnings and '
                       'raised its full-year guidance. The stock rose 1.80% on above-average '
                       'trading volume. Analyst commentary and public market sentiment are broadly '
                       'bullish, and no material negative development was reported.',
       'decisions': {'overconfidence': {'date': '2024-02-15',
                                        'symbol': 'DEMO',
                                        'action': 'BUY',
                                        'rationale': 'The operating results and price response '
                                                     'both confirm a strong opportunity. I trust '
                                                     'this reading and will act decisively with a '
                                                     'one-period positive exposure.',
                                        'confidence': 0.8},
                     'loss_aversion': {'date': '2024-02-15',
                                       'symbol': 'DEMO',
                                       'action': 'BUY',
                                       'rationale': 'The strong results and broad confirmation '
                                                    'reduce near-term downside risk. I will take a '
                                                    'measured one-period positive exposure while '
                                                    'remaining alert to a reversal.',
                                       'confidence': 0.8},
                     'herding': {'date': '2024-02-15',
                                 'symbol': 'DEMO',
                                 'action': 'BUY',
                                 'rationale': 'The price rise, strong volume, bullish analyst '
                                              'commentary, and positive public sentiment all point '
                                              'in the same direction. I will follow the clear '
                                              'consensus with a one-period positive exposure.',
                                 'confidence': 0.8}}},
 'N': {'role': 'ablation_negative_aligned',
       'market_block': 'Current price: 97.60. Price change from yesterday: -2.40%.\n'
                       '\n'
                       "Your return from yesterday's HOLD decision: +0.00%. Cumulative log return "
                       'so far: +0.00%.\n'
                       '\n'
                       "Today's news:\n"
                       'The company reported weaker-than-expected revenue and earnings and lowered '
                       'its full-year guidance. The stock fell 2.40% on above-average trading '
                       'volume. Analyst commentary and public market sentiment are broadly '
                       'bearish, and no material positive development was reported.',
       'decisions': {'overconfidence': {'date': '2024-02-15',
                                        'symbol': 'DEMO',
                                        'action': 'SELL',
                                        'rationale': 'The earnings miss, weaker guidance, and '
                                                     'price breakdown form a clear downside '
                                                     'signal. I trust this reading and will act '
                                                     'decisively with a one-period negative '
                                                     'exposure.',
                                        'confidence': 0.8},
                     'loss_aversion': {'date': '2024-02-15',
                                       'symbol': 'DEMO',
                                       'action': 'HOLD',
                                       'rationale': 'The outlook is negative, but taking a new '
                                                    'directional position after a sharp move could '
                                                    'add avoidable loss. I will stay neutral for '
                                                    'one period to protect capital.',
                                       'confidence': 0.8},
                     'herding': {'date': '2024-02-15',
                                 'symbol': 'DEMO',
                                 'action': 'SELL',
                                 'rationale': 'The falling price, bearish analyst commentary, and '
                                              'negative public sentiment show a clear market '
                                              'consensus. I will follow the crowd with a '
                                              'one-period negative exposure.',
                                 'confidence': 0.8}}}}

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


def load_personas() -> dict:
    return deepcopy(PERSONAS)


def load_examples() -> dict[str, Any]:
    return deepcopy(EXAMPLES)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_tag(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_key)


def all_conditions(protocol: dict | None = None) -> list[dict[str, Any]]:
    """Return the frozen, deterministically interleaved 90-condition model matrix."""
    protocol = protocol or load_protocol()
    conditions = [
        {
            "scenario": scenario,
            "symbol": symbol,
            "persona": persona,
            "run_id": int(run_id),
            "seed": int(protocol["seeds"][str(run_id)]),
        }
        for scenario in protocol["scenarios"]
        for symbol in protocol["symbols"]
        for persona in protocol["personas"]
        for run_id in protocol["run_ids"]
    ]
    random.Random(int(protocol["condition_shuffle_seed"])).shuffle(conditions)
    for index, condition in enumerate(conditions):
        condition["condition_index"] = index
    return conditions


def run_stem(
    model_key: str,
    symbol: str,
    persona: str,
    scenario: str,
    run_id: int,
    seed: int,
) -> str:
    return (
        f"{model_tag(model_key)}_{symbol}_{persona}_one-shot-matched_{scenario}"
        f"_run{run_id}_seed{seed}"
    )


def output_paths(
    model_key: str,
    symbol: str,
    persona: str,
    scenario: str,
    run_id: int,
    seed: int,
    smoke: bool = False,
) -> dict[str, Path]:
    stem = run_stem(model_key, symbol, persona, scenario, run_id, seed)
    root_name = "smoke_outputs" if smoke else "outputs"
    base = ROOT / root_name / scenario / model_tag(model_key) / symbol / persona
    return {
        "base": base,
        "trajectory": base / f"{stem}.csv",
        "summary": base / f"{stem}.summary.json",
        "calls": base / f"{stem}.calls.jsonl",
        "manifest": base / f"{stem}.manifest.json",
    }


def atomic_temp_path(path: Path) -> Path:
    """Return a process-unique sibling path for an atomic write."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = atomic_temp_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


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
