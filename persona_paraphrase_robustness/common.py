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


ROOT = Path(__file__).resolve().parent

# Fixed experimental settings are embedded to keep the release self-contained.
PROTOCOL = {'protocol_name': 'theatron-zero-shot-paraphrase-robustness-v1',
 'protocol_version': 1,
 'run_versions': ['v1_controlled_lexical_substitution', 'v2_sentential_reformulation'],
 'analysis_versions': ['v0_original',
                       'v1_controlled_lexical_substitution',
                       'v2_sentential_reformulation'],
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
 'prompt': {'version': 'zero-shot-persona-paraphrase-v1',
            'json_instruction': 'Respond ONLY with a JSON object in this exact format:\n'
                                '{"date": "YYYY-MM-DD", "symbol": "XXX", "action": '
                                '"BUY/HOLD/SELL", "rationale": "...", "confidence": 0.0}',
            'forbidden_model_visible_terms': ['v0_original',
                                              'v1_controlled_lexical_substitution',
                                              'v2_sentential_reformulation',
                                              'paraphrase',
                                              'lexical substitution',
                                              'sentential reformulation']},
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

PERSONAS = {'v0_original': {'overconfidence': 'You are an overconfident stock trader. You trust your own '
                                   'signals implicitly, tend to underestimate risk, and have a '
                                   'strong tendency to overtrade. You believe your read of the '
                                   'market is sharper than most, so you act decisively and '
                                   'aggressively on perceived opportunities. You rarely '
                                   'second-guess yourself and interpret ambiguous signals as '
                                   'confirmation of your existing view.',
                 'loss_aversion': 'You are a loss-averse stock trader. Losses feel significantly '
                                  'more painful to you than equivalent gains feel rewarding. You '
                                  'are highly defensive and quick to lock in profits to avoid '
                                  'watching them evaporate. You actively avoid drawdowns, often '
                                  'exiting positions early, and you hold losing trades less '
                                  'comfortably than winning ones. Protecting capital from loss is '
                                  'your primary emotional drive.',
                 'herding': 'You are a herding-prone stock trader. You follow the crowd and market '
                            'consensus closely. When sentiment is positive or prices are rising, '
                            'you feel compelled to buy so you do not miss out. When sentiment '
                            'turns negative or prices are falling, you sell to avoid being left '
                            'behind. You find comfort in doing what the majority appears to be '
                            'doing and distrust contrarian positions.'},
 'v1_controlled_lexical_substitution': {'overconfidence': 'You are an overly self-confident stock '
                                                          'trader. You rely on your own signals '
                                                          'unquestioningly, tend to downplay risk, '
                                                          'and have a strong inclination to trade '
                                                          'excessively. You believe your '
                                                          'interpretation of the market is better '
                                                          'than most, so you act resolutely and '
                                                          'aggressively on apparent opportunities. '
                                                          'You seldom reconsider your judgments '
                                                          'and interpret uncertain signals as '
                                                          'validation of your existing view.',
                                        'loss_aversion': 'You are a stock trader strongly averse '
                                                         'to losses. Losses feel considerably more '
                                                         'distressing to you than equivalent gains '
                                                         'feel satisfying. You are highly '
                                                         'defensive and quick to secure profits to '
                                                         'avoid watching them disappear. You '
                                                         'actively limit drawdowns, often closing '
                                                         'positions early, and you hold losing '
                                                         'trades less comfortably than profitable '
                                                         'ones. Preserving capital from loss is '
                                                         'your primary emotional motivation.',
                                        'herding': 'You are a crowd-following stock trader. You '
                                                   'track other investors and prevailing market '
                                                   'consensus closely. When sentiment is bullish '
                                                   'or prices are increasing, you feel driven to '
                                                   'buy so you do not miss the move. When '
                                                   'sentiment becomes bearish or prices are '
                                                   'decreasing, you sell to avoid falling behind. '
                                                   'You feel reassured by doing what the majority '
                                                   'appears to be doing and have little trust in '
                                                   'contrarian positions.'},
 'v2_sentential_reformulation': {'overconfidence': 'As a stock trader, you have excessive '
                                                   'confidence in your judgment. You give great '
                                                   'weight to your own signals, discount risk, and '
                                                   'trade too often. Believing your market '
                                                   'assessment exceeds that of other traders, you '
                                                   'pursue perceived opportunities with decisive, '
                                                   'aggressive action. Doubt seldom changes your '
                                                   'conclusions; when evidence is mixed, you '
                                                   'usually read it as validating the view you '
                                                   'already hold.',
                                 'loss_aversion': 'As a stock trader, you are highly sensitive to '
                                                  'losses. Losing an amount causes more distress '
                                                  'than gaining the same amount provides '
                                                  'satisfaction. This asymmetry makes you '
                                                  'defensive and eager to realize profits before '
                                                  'they vanish. You seek to contain drawdowns, '
                                                  'commonly leave positions early, and feel less '
                                                  'at ease with losing trades than winning ones. '
                                                  'Your dominant emotional motive is protecting '
                                                  'capital from loss.',
                                 'herding': 'As a stock trader, your decisions are strongly shaped '
                                            'by other market participants. You use prevailing '
                                            'opinion and crowd behavior as guides. Improving '
                                            'sentiment or upward price movement pushes you to buy '
                                            'because you do not want to miss the move. '
                                            'Deteriorating sentiment or downward price movement '
                                            'leads you to sell rather than be left behind. Acting '
                                            'with the majority feels reassuring, whereas taking a '
                                            'contrarian position inspires little confidence.'}}

FORMAL_MODELS = [
    "qwen3-8b-nothink",
    "qwen3-14b-thinking",
    "gpt-5.4-mini",
    "gemini-2.5-flash",
]

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


def version_tag(version: str) -> str:
    return {
        "v0_original": "v0-original",
        "v1_controlled_lexical_substitution": "v1-lexical",
        "v2_sentential_reformulation": "v2-sentential",
    }[version]


def all_conditions(protocol: dict | None = None) -> list[dict[str, Any]]:
    """Return the frozen, deterministically interleaved 60-condition matrix."""
    protocol = protocol or load_protocol()
    conditions = [
        {
            "paraphrase_version": version,
            "symbol": symbol,
            "persona": persona,
            "run_id": int(run_id),
            "seed": int(protocol["seeds"][str(run_id)]),
        }
        for version in protocol["run_versions"]
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
    paraphrase_version: str,
    run_id: int,
    seed: int,
) -> str:
    return (
        f"{model_tag(model_key)}_{symbol}_{persona}_zero-shot-paraphrase_"
        f"{version_tag(paraphrase_version)}_run{run_id}_seed{seed}"
    )


def output_paths(
    model_key: str,
    symbol: str,
    persona: str,
    paraphrase_version: str,
    run_id: int,
    seed: int,
    smoke: bool = False,
) -> dict[str, Path]:
    stem = run_stem(
        model_key, symbol, persona, paraphrase_version, run_id, seed
    )
    root_name = "smoke_outputs" if smoke else "outputs"
    base = (
        ROOT
        / root_name
        / paraphrase_version
        / model_tag(model_key)
        / symbol
        / persona
    )
    return {
        "base": base,
        "trajectory": base / f"{stem}.csv",
        "summary": base / f"{stem}.summary.json",
        "calls": base / f"{stem}.calls.jsonl",
        "manifest": base / f"{stem}.manifest.json",
    }


def atomic_temp_path(path: Path) -> Path:
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
    files = sorted(ROOT.glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}
