from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    metadata: dict[str, Any]


class BaseClient:
    def generate(self, system_message: str, user_message: str, seed: int) -> Generation:
        raise NotImplementedError


class OpenAIClient(BaseClient):
    def __init__(self, cfg: dict):
        from openai import OpenAI

        env_name = cfg["api_key_env"]
        api_key = os.environ.get(env_name)
        if not api_key:
            raise RuntimeError(f"Required API key environment variable is unset: {env_name}")
        if cfg.get("official_endpoint_only", False) and os.environ.get("OPENAI_BASE_URL"):
            raise RuntimeError("OPENAI_BASE_URL is set, but this protocol requires the official endpoint")
        self.cfg = cfg
        self.client = OpenAI(api_key=api_key)

    def generate(self, system_message: str, user_message: str, seed: int) -> Generation:
        response = self.client.chat.completions.create(
            model=self.cfg["request_model"],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=float(self.cfg["temperature"]),
            top_p=float(self.cfg["top_p"]),
            seed=int(seed),
            max_completion_tokens=int(self.cfg["max_new_tokens"]),
            reasoning_effort=self.cfg.get("reasoning_effort", "none"),
        )
        sleep_seconds = float(self.cfg.get("inter_request_sleep_seconds", 0.0))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        choice = response.choices[0]
        return Generation(
            text=choice.message.content or "",
            metadata={
                "provider": "openai",
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "usage": _safe_dump(getattr(response, "usage", None)),
                "seed_semantics": self.cfg.get("seed_support", "best_effort"),
            },
        )


class GeminiClient(BaseClient):
    def __init__(self, cfg: dict):
        from google import genai

        env_name = cfg["api_key_env"]
        api_key = os.environ.get(env_name)
        if not api_key:
            raise RuntimeError(f"Required API key environment variable is unset: {env_name}")
        self.cfg = cfg
        self.client = genai.Client(api_key=api_key)

    def generate(self, system_message: str, user_message: str, seed: int) -> Generation:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_message,
            temperature=float(self.cfg["temperature"]),
            top_p=float(self.cfg["top_p"]),
            max_output_tokens=int(self.cfg["max_new_tokens"]),
            seed=int(seed),
            thinking_config=types.ThinkingConfig(
                thinking_budget=int(self.cfg.get("thinking_budget", 0))
            ),
        )
        max_retries = int(self.cfg.get("transient_503_max_retries", 0))
        initial_backoff = float(self.cfg.get("transient_503_initial_backoff_seconds", 30.0))
        max_backoff = float(self.cfg.get("transient_503_max_backoff_seconds", 300.0))
        retry_count = 0
        retry_wait_seconds = 0.0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.cfg["request_model"],
                    contents=user_message,
                    config=config,
                )
                break
            except Exception as exc:
                if not _is_gemini_503(exc) or retry_count >= max_retries:
                    raise
                wait_seconds = min(initial_backoff * (2**retry_count), max_backoff)
                retry_count += 1
                retry_wait_seconds += wait_seconds
                logger.warning(
                    "Gemini 503 unavailable; retry %d/%d after %.0fs.",
                    retry_count,
                    max_retries,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
        sleep_seconds = float(self.cfg.get("inter_request_sleep_seconds", 0.0))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        return Generation(
            text=response.text or "",
            metadata={
                "provider": "gemini",
                "response_id": getattr(response, "response_id", None),
                "response_model": getattr(response, "model_version", None),
                "finish_reason": str(getattr(candidate, "finish_reason", None)),
                "usage": _safe_dump(getattr(response, "usage_metadata", None)),
                "seed_semantics": self.cfg.get("seed_support", "best_effort"),
                "transient_503_retry_count": retry_count,
                "transient_503_retry_wait_seconds": retry_wait_seconds,
            },
        )


class LocalTransformersClient(BaseClient):
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        path = cfg["local_model_path"]
        if not os.path.isdir(path):
            raise RuntimeError(f"Local model directory does not exist: {path}")
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[cfg["torch_dtype"]]
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=cfg.get("device_map", "auto"),
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    def generate(self, system_message: str, user_message: str, seed: int) -> Generation:
        import torch

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": bool(self.cfg.get("enable_thinking", False)),
        }
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=float(self.cfg["temperature"]),
                top_p=float(self.cfg["top_p"]),
                max_new_tokens=int(self.cfg["max_new_tokens"]),
            )
        new_tokens = output[0][input_length:]
        return Generation(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            metadata={
                "provider": "local_transformers",
                "response_model": self.cfg["request_model"],
                "seed_semantics": "explicit_torch_seed",
                "input_tokens": int(input_length),
                "output_tokens": int(len(new_tokens)),
            },
        )


class MockClient(BaseClient):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate(self, system_message: str, user_message: str, seed: int) -> Generation:
        date = "1970-01-01"
        symbol = "MOCK"
        first_line = user_message.splitlines()[0] if user_message else ""
        if first_line.startswith("Today is ") and ". Stock: " in first_line:
            date, symbol = first_line.removeprefix("Today is ").removesuffix(".").split(
                ". Stock: ", 1
            )
        text = json.dumps(
            {
                "date": date,
                "symbol": symbol,
                "action": "HOLD",
                "rationale": "Deterministic offline smoke test.",
                "confidence": 0.5,
            }
        )
        return Generation(text=text, metadata={"provider": "mock", "seed": seed})


def create_client(cfg: dict) -> BaseClient:
    backend = cfg["backend"]
    if backend == "openai":
        return OpenAIClient(cfg)
    if backend == "gemini":
        return GeminiClient(cfg)
    if backend == "local":
        return LocalTransformersClient(cfg)
    if backend == "mock":
        return MockClient(cfg)
    raise ValueError(f"Unsupported backend: {backend}")


def _safe_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def _is_gemini_503(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 503:
        return True
    return "503" in str(exc) and "UNAVAILABLE" in str(exc).upper()
