"""
run_experiment.py — CLI entry point for a single Chameleon experiment run.

Usage example:
    python -m src.run_experiment \\
        --symbol AAPL \\
        --persona overconfidence \\
        --method zero_shot \\
        --model gpt-5.4-mini \\
        --backend api \\
        --api_provider openai \\
        --api_key_env OPENAI_API_KEY \\
        --top_p 0.9 \\
        --run_id 1 \\
        --random_seed 41
"""

import argparse
import json
import logging
import os
import sys

# Allow running as `python src/run_experiment.py` or `python -m src.run_experiment`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest import run_trajectory
from src.utils import setup_logging


def _parse_bool(value: str) -> bool:
    """Accept 'true'/'false' strings or standard truthy values."""
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single Chameleon agent experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Required experiment identifiers
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock ticker symbol to trade (e.g. AAPL).",
    )
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        choices=[
            "overconfidence",
            "loss_aversion",
            "herding",
            "conservative",
            "momentum_chaser",
            "contrarian",
        ],
        help="Cognitive bias persona to inject into the agent.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["zero_shot", "one_shot", "cot"],
        help="Persona injection method.",
    )

    # ------------------------------------------------------------------ #
    # Model / backend selection
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g. 'gpt-5.4-mini', 'gemini-2.5-flash', 'Qwen3-8B-nothink').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="api",
        choices=["api", "local"],
        help="Backend type: 'api' for cloud providers, 'local' for HuggingFace models.",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        choices=["openai", "gemini", "deepseek"],
        help="Cloud API provider (only used when --backend=api).",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Name of the environment variable holding the API key.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="",
        help="Optional base URL for an OpenAI-compatible API.",
    )

    # ------------------------------------------------------------------ #
    # Local model settings
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--local_backend",
        type=str,
        default="",
        help="HuggingFace backend label (e.g. 'transformers'). Used only when --backend=local.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="",
        help="File-system path to the local HuggingFace model directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for local inference (e.g. 'cuda', 'cpu', 'mps', 'auto').",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for local model weights.",
    )

    # ------------------------------------------------------------------ #
    # Generation hyperparameters
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per step.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold.",
    )

    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
        help="Integer run identifier for repeated trials.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # ------------------------------------------------------------------ #
    # Memory settings
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--use_memory",
        type=_parse_bool,
        default=False,
        metavar="BOOL",
        help="Enable episodic memory (true/false).",
    )
    parser.add_argument(
        "--memory_top_k",
        type=int,
        default=1,
        help="Number of past memories to retrieve per step.",
    )
    parser.add_argument(
        "--memory_db_path",
        type=str,
        default="./outputs/memory",
        help="Directory path for the persistent ChromaDB memory store.",
    )

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--inter_request_sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep after each successful Gemini API call.",
    )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Limit trajectory to this many trading days (useful for testing).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)
    logger = logging.getLogger("chameleon.run_experiment")

    # Build config dict from parsed args
    config = {
        # Experiment identifiers
        "symbol": args.symbol,
        "persona": args.persona,
        "method": args.method,
        # Model / backend
        "model": args.model,
        "backend": args.backend,
        "api_provider": args.api_provider,
        "api_key_env": args.api_key_env,
        "api_base_url": args.api_base_url,
        "local_backend": args.local_backend,
        "local_model_path": args.local_model_path,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        # Generation
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        # Reproducibility
        "run_id": args.run_id,
        "random_seed": args.random_seed,
        # Memory
        "use_memory": args.use_memory,
        "memory_top_k": args.memory_top_k,
        "memory_db_path": args.memory_db_path,
        # Rate limiting
        "inter_request_sleep": args.inter_request_sleep,
        # Step cap
        "max_steps": args.max_steps,
        # Logging
        "log_level": args.log_level,
    }

    logger.info("Starting experiment with config:\n%s", json.dumps(config, indent=2))

    try:
        summary = run_trajectory(config)
    except Exception as exc:
        logger.error("Experiment failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Print summary to stdout as JSON
    print(json.dumps(summary, indent=2))
    logger.info("Experiment complete. Final cumulative log return: %s", summary.get("CR"))


if __name__ == "__main__":
    main()
