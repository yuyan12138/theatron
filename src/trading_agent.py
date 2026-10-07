"""
trading_agent.py — Unified trading agent supporting multiple LLM backends.

Supported backends:
  - openai      : OpenAI Chat Completions API
  - deepseek    : OpenAI-compatible client pointed at DeepSeek endpoint
  - gemini      : Google Gen AI SDK (google.genai)
  - local       : HuggingFace Transformers (AutoModelForCausalLM)

JSON parsing with one automatic repair retry and a HOLD fallback.
"""

import json
import logging
import os
import re
from typing import Optional

from src.utils import normalize_action

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
_BARE_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
# Qwen3 thinking models wrap reasoning in <think>...</think> before the answer
_THINKING_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by thinking models."""
    return _THINKING_BLOCK_RE.sub("", text).strip()


def _extract_json_str(text: str) -> Optional[str]:
    """Try to pull the first JSON object out of *text*.

    Iterates all {…} candidates so that preamble text (e.g. from Gemini's
    inline reasoning) containing stray braces does not fool the regex into
    returning a non-JSON fragment instead of the actual decision object.
    """
    # Strip Qwen3-style <think>...</think> blocks
    text = _strip_thinking(text)
    # Prefer fenced code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    # Scan every {…} candidate; return the first one that is valid JSON
    # containing an "action" key (the required field for a decision object).
    for m in _BARE_JSON_RE.finditer(text):
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if "action" in obj:
                return candidate
        except (json.JSONDecodeError, ValueError):
            continue
    # Last resort: return whatever the first match is and let the caller fail
    m = _BARE_JSON_RE.search(text)
    return m.group(0) if m else None


def _make_fallback(date_str: str, symbol: str) -> dict:
    return {
        "date": date_str,
        "symbol": symbol,
        "action": "HOLD",
        "rationale": "Invalid model output; defaulted to HOLD.",
        "confidence": 0.0,
    }


def _validate_and_fix(parsed: dict, date_str: str, symbol: str) -> Optional[dict]:
    """Return a valid record or None if action is unrecognizable."""
    action = normalize_action(parsed.get("action"))
    if action is None:
        return None
    parsed["action"] = action
    # Ensure required fields have sensible defaults
    parsed.setdefault("date", date_str)
    parsed.setdefault("symbol", symbol)
    parsed.setdefault("rationale", "")
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    return parsed


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class TradingAgent:
    """
    Unified interface over multiple LLM backends.

    Parameters (via config dict)
    ----------------------------
    backend : str
        "api" or "local"
    api_provider : str
        "openai", "gemini", or "deepseek"  (only relevant when backend=="api")
    api_key_env : str
        Name of the environment variable holding the API key.
    model : str
        Model identifier string (e.g. "gpt-4o", "gemini-1.5-pro").
    local_backend : str
        HuggingFace backend type for local models.
    local_model_path : str
        File-system path to a local HuggingFace model.
    device : str
        Device string such as "cuda", "cpu", "mps" (local only).
    torch_dtype : str
        "float16", "bfloat16", or "float32" (local only).
    top_p : float
    max_new_tokens : int
    random_seed : int
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get("backend", "api").lower()
        self.api_provider = config.get("api_provider", "openai").lower()
        self.model = config.get("model", "gpt-4o-mini")
        self.top_p = float(config.get("top_p", 0.9))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.random_seed = int(config.get("random_seed", 42))

        self._client = None
        self._genai_client = None
        self._tokenizer = None
        self._model = None

        if self.backend == "api":
            self._init_api_client()
        elif self.backend == "local":
            self._init_local_model()
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'api' or 'local'.")

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #

    def _get_api_key(self) -> str:
        env_var = self.config.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            logger.warning(
                "API key environment variable '%s' is not set or is empty.", env_var
            )
        return key

    def _init_api_client(self):
        provider = self.api_provider

        if provider in {"openai", "deepseek"}:
            import openai

            api_key = self._get_api_key()
            if provider == "deepseek":
                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
            else:
                base_url = self.config.get("api_base_url") or None
                self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
            logger.info("Initialized %s client (model=%s)", provider, self.model)

        elif provider == "gemini":
            import google.genai as genai

            api_key = self._get_api_key()
            self._genai_client = genai.Client(api_key=api_key)
            logger.info("Initialized Gemini client (model=%s)", self.model)

        else:
            raise ValueError(
                f"Unknown api_provider '{provider}'. Use 'openai', 'gemini', or 'deepseek'."
            )

    def _init_local_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.config.get("local_model_path", "")
        if not model_path:
            raise ValueError(
                "local_model_path must be set when backend='local'."
            )

        dtype_str = self.config.get("torch_dtype", "float16").lower()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.float16)

        logger.info("Loading local model from '%s' (dtype=%s)…", model_path, dtype_str)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        logger.info("Local model loaded successfully.")

    # ------------------------------------------------------------------ #
    # Raw generation
    # ------------------------------------------------------------------ #

    def generate(self, system_msg: str, user_msg: str) -> str:
        """
        Call the underlying model and return raw text output.
        """
        if self.backend == "api":
            return self._generate_api(system_msg, user_msg)
        return self._generate_local(system_msg, user_msg)

    def _generate_api(self, system_msg: str, user_msg: str) -> str:
        provider = self.api_provider

        if provider in {"openai", "deepseek"}:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "top_p": self.top_p,
                "seed": self.random_seed,
            }
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        elif provider == "gemini":
            import re as _re
            import time
            from google.genai import types as _gtypes

            combined_prompt = f"{system_msg}\n\n{user_msg}"

            gen_cfg = _gtypes.GenerateContentConfig(
                top_p=self.top_p,
                max_output_tokens=max(self.max_new_tokens, 2048),
                thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
            )

            max_retries = 8
            for attempt in range(max_retries):
                try:
                    response = self._genai_client.models.generate_content(
                        model=self.model,
                        contents=combined_prompt,
                        config=gen_cfg,
                    )
                    text = response.text or ""
                    sleep_sec = float(self.config.get("inter_request_sleep", 0))
                    if sleep_sec > 0:
                        time.sleep(sleep_sec)
                    return text
                except Exception as exc:
                    exc_str = str(exc)
                    is_rate_limit = (
                        "429" in exc_str
                        or "quota" in exc_str.lower()
                        or "rate limit" in exc_str.lower()
                        or "resource_exhausted" in exc_str.lower()
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        m = _re.search(
                            r"retry[_ ]in[_\s:]+(\d+)|retry_delay\s*\{\s*seconds:\s*(\d+)",
                            exc_str, _re.IGNORECASE,
                        )
                        wait_sec = int(m.group(1) or m.group(2)) + 2 if m else min(15 * (2 ** attempt), 300)
                        logger.warning(
                            "Gemini rate limit, waiting %ds (attempt %d/%d)…",
                            wait_sec, attempt + 1, max_retries,
                        )
                        time.sleep(wait_sec)
                    else:
                        raise
            raise RuntimeError(f"Gemini generation failed after {max_retries} retries.")

        raise ValueError(f"Unknown provider '{provider}'.")

    def _generate_local(self, system_msg: str, user_msg: str) -> str:
        import torch

        tokenizer = self._tokenizer
        model = self._model

        # Qwen3 "nothink" models disable the internal chain-of-thought block
        enable_thinking = "nothink" not in self.model.lower()

        # Build prompt with chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = f"{system_msg}\n\n{user_msg}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        torch.manual_seed(self.random_seed)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                do_sample=True,
            )

        # Decode only newly generated tokens
        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Repair prompt helper
    # ------------------------------------------------------------------ #

    def _repair_generate(self, bad_output: str, date_str: str, symbol: str) -> str:
        """Ask the model to re-emit only a valid JSON object."""
        repair_system = (
            "You are a JSON repair assistant. Output ONLY a valid JSON object "
            "with keys: date, symbol, action (BUY/HOLD/SELL), rationale, confidence."
        )
        repair_user = (
            f"The following response could not be parsed as valid JSON:\n\n"
            f"{bad_output}\n\n"
            f"Please rewrite it as a single valid JSON object for date={date_str} "
            f"and symbol={symbol}. Output ONLY the JSON, nothing else."
        )
        try:
            return self.generate(repair_system, repair_user)
        except Exception as exc:
            logger.error("Repair generation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # Parse + validate + fallback
    # ------------------------------------------------------------------ #

    def parse_model_output(self, raw_text: str, date_str: str, symbol: str) -> dict:
        """
        Extract and validate a JSON decision from *raw_text*.

        Retry once with a repair prompt on failure; return a HOLD fallback
        if the repair also fails.
        """
        # --- First attempt ---
        json_str = _extract_json_str(raw_text)
        if json_str:
            try:
                parsed = json.loads(json_str)
                result = _validate_and_fix(parsed, date_str, symbol)
                if result is not None:
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Could not parse model output on first attempt for %s %s. "
            "Attempting repair.",
            date_str,
            symbol,
        )

        # --- Repair retry ---
        repaired_text = self._repair_generate(raw_text, date_str, symbol)
        json_str2 = _extract_json_str(repaired_text)
        if json_str2:
            try:
                parsed2 = json.loads(json_str2)
                result2 = _validate_and_fix(parsed2, date_str, symbol)
                if result2 is not None:
                    logger.info("Repair successful for %s %s.", date_str, symbol)
                    return result2
            except json.JSONDecodeError:
                pass

        logger.error(
            "Repair also failed for %s %s. Returning HOLD fallback.", date_str, symbol
        )
        return _make_fallback(date_str, symbol)

    # ------------------------------------------------------------------ #
    # High-level decision method
    # ------------------------------------------------------------------ #

    def decide(
        self,
        system_msg: str,
        user_msg: str,
        date_str: str,
        symbol: str,
    ) -> dict:
        """
        Generate a response and return a validated decision dict.

        The dict always contains: date, symbol, action, rationale, confidence.
        """
        try:
            raw_text = self.generate(system_msg, user_msg)
        except Exception as exc:
            logger.error(
                "Model generation failed for %s %s: %s", date_str, symbol, exc
            )
            return _make_fallback(date_str, symbol)

        return self.parse_model_output(raw_text, date_str, symbol)
