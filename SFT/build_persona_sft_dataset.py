"""
Persona-conditioned SFT dataset builder.

Reads daily stock parquet files, aligns calendar days to NYSE trading days,
remaps weekend/holiday news to the next trading day, extracts FinBERT
sentiment from news + 10-K/10-Q text, then emits one supervised sample per
(stock, trading-day, persona) tuple with a Long/Short/Flat label.

Spec: SFT/persona_sft_labeling_instruction.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SFT_ROOT = Path(__file__).resolve().parent
DATA_DIR = SFT_ROOT / "data"
OUT_JSON = SFT_ROOT / "persona_sft_training_data.json"
OUT_CSV = SFT_ROOT / "persona_sft_training_data.csv"

SEED = 20250523

# Default backbone for weak sentiment labeling.
FINBERT_MODEL = "ProsusAI/finbert"

# Default API provider + model. API labeling is OFF by default; enabled via
# CLI flag --label-provider openai.
LABEL_PROVIDER = "finbert"
OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-5"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

PERSONAS = ("Overconfidence", "Loss Aversion", "Herding")
ALLOWED_LABELS = ("Long", "Short", "Flat")

# NYSE 2025 holidays covering the Jan–May span of the source data. Hard-coded
# to avoid a runtime dependency on pandas_market_calendars / exchange_calendars.
NYSE_2025_HOLIDAYS = pd.to_datetime([
    "2025-01-01",  # New Year's Day
    "2025-01-09",  # National Day of Mourning (Jimmy Carter)
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
])

# FinBERT max input length (BERT-base).
FINBERT_MAX_TOKENS = 510

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def _as_list(cell) -> list[str]:
    """Normalize parquet cell (numpy array / list / None) to list[str]."""
    if cell is None:
        return []
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, (list, tuple)):
        return [str(x) for x in cell if x is not None and str(x).strip()]
    s = str(cell).strip()
    return [s] if s else []


def load_stock_frame(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Drop the future-information column to honor the no-leakage rule.
    if "future_price_diff" in df.columns:
        df = df.drop(columns=["future_price_diff"])
    for col in ("news", "10k", "10q"):
        df[col] = df[col].apply(_as_list)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def next_trading_day(d: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the first trading day >= d, or None if past the calendar."""
    pos = trading_days.searchsorted(d, side="left")
    if pos >= len(trading_days):
        return None
    return trading_days[pos]


def align_to_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to NYSE trading days, remap weekend/holiday content forward.

    Each calendar row's news / 10-K / 10-Q text is attached to the *next*
    trading day (the day that information would actually be tradable on).
    """
    cal_start, cal_end = df["date"].min(), df["date"].max()
    all_business = pd.bdate_range(cal_start, cal_end)
    trading_days = all_business.difference(NYSE_2025_HOLIDAYS)

    target = df["date"].apply(lambda d: next_trading_day(d, trading_days))
    df = df.assign(_target=target)
    df = df[df["_target"].notna()].copy()

    # Aggregate news / filings across all source rows that point at the same
    # trading day. Price + momentum come from the actual trading-day row.
    groups: dict[pd.Timestamp, dict] = {}
    for _, row in df.iterrows():
        td = row["_target"]
        bucket = groups.setdefault(
            td,
            {
                "asset": row["asset"],
                "news": [],
                "10k": [],
                "10q": [],
                "prices": None,
                "momentum": None,
            },
        )
        bucket["news"].extend(row["news"])
        bucket["10k"].extend(row["10k"])
        bucket["10q"].extend(row["10q"])
        if row["date"] == td:
            bucket["prices"] = float(row["prices"])
            bucket["momentum"] = str(row["momentum"]).strip().lower()

    aligned = []
    for td in sorted(groups.keys()):
        b = groups[td]
        if td not in trading_days:
            continue
        if b["prices"] is None or b["momentum"] is None:
            # Filing/news exists for a trading day not in source frame; skip.
            continue
        aligned.append(
            {
                "date": td,
                "asset": b["asset"],
                "prices": b["prices"],
                "momentum": b["momentum"],
                "news": b["news"],
                "10k": b["10k"],
                "10q": b["10q"],
            }
        )
    out = pd.DataFrame(aligned)
    return out


# --------------------------------------------------------------------------- #
# FinBERT sentiment
# --------------------------------------------------------------------------- #


@dataclass
class SentimentScores:
    positive: float
    neutral: float
    negative: float
    label: str  # "positive" | "neutral" | "negative"

    @property
    def net(self) -> float:
        return self.positive - self.negative

    def to_dict(self) -> dict:
        return {
            "positive": round(self.positive, 4),
            "neutral": round(self.neutral, 4),
            "negative": round(self.negative, 4),
            "net": round(self.net, 4),
            "label": self.label,
        }


class FinBERTScorer:
    """Thin wrapper that loads ProsusAI/finbert lazily and scores text."""

    def __init__(self, model_name: str = FINBERT_MODEL):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._labels: list[str] | None = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        # FinBERT id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
        id2label = self._model.config.id2label
        self._labels = [id2label[i].lower() for i in range(len(id2label))]
        self._device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._model.to(self._device)
        self._torch = torch

    def _score_chunk(self, text: str) -> dict:
        self._ensure_loaded()
        torch = self._torch
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=FINBERT_MAX_TOKENS,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {lbl: float(p) for lbl, p in zip(self._labels, probs)}

    def _chunk_text(self, text: str) -> list[str]:
        """Greedy whitespace chunking so long news pieces still get scored."""
        self._ensure_loaded()
        tokens = self._tokenizer.tokenize(text)
        if len(tokens) <= FINBERT_MAX_TOKENS:
            return [text]
        # Approximate: split on tokens, decode chunks back to text.
        chunks = []
        for i in range(0, len(tokens), FINBERT_MAX_TOKENS):
            piece = tokens[i : i + FINBERT_MAX_TOKENS]
            chunks.append(self._tokenizer.convert_tokens_to_string(piece))
        return chunks

    def score_texts(self, texts: Iterable[str]) -> SentimentScores:
        """Average FinBERT probabilities across all input texts/chunks."""
        agg = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        n = 0
        for t in texts:
            t = (t or "").strip()
            if not t:
                continue
            for chunk in self._chunk_text(t):
                probs = self._score_chunk(chunk)
                for k in agg:
                    agg[k] += probs.get(k, 0.0)
                n += 1
        if n == 0:
            return SentimentScores(0.0, 1.0, 0.0, "neutral")
        for k in agg:
            agg[k] /= n
        label = max(agg, key=agg.get)
        return SentimentScores(agg["positive"], agg["neutral"], agg["negative"], label)


# --------------------------------------------------------------------------- #
# Persona-conditioned label generation
# --------------------------------------------------------------------------- #


PERSONA_PROMPT_TEMPLATES = {
    "Overconfidence": (
        "You are a trader with an OVERCONFIDENCE bias. You trust your own "
        "analysis strongly, take decisive positions when signals point in one "
        "direction, and are slow to admit ambiguity. You almost always pick a "
        "side rather than stand aside."
    ),
    "Loss Aversion": (
        "You are a trader with a LOSS AVERSION bias. Losses hurt much more than "
        "equivalent gains feel good. You demand strong positive evidence before "
        "taking risk and prefer staying Flat whenever signals are mixed or "
        "downside risk is meaningful."
    ),
    "Herding": (
        "You are a trader with a HERDING bias. You move with the crowd: recent "
        "price momentum and prevailing market sentiment drive your decisions "
        "more than independent fundamental analysis."
    ),
}


def persona_label_rule(
    sentiment: SentimentScores,
    momentum: str,
    persona: str,
    has_filing: bool,
) -> str:
    """Deterministic rule-based label that operationalises each persona."""
    net = sentiment.net
    pos = sentiment.positive
    neg = sentiment.negative
    mom = (momentum or "neutral").lower()

    if persona == "Overconfidence":
        # Strong conviction either way; rarely Flat.
        agree_bull = (mom == "bullish") and (net >= -0.05)
        agree_bear = (mom == "bearish") and (net <= 0.05)
        if agree_bull or net >= 0.08:
            return "Long"
        if agree_bear or net <= -0.08:
            return "Short"
        # Pick the side that any single signal hints at.
        if mom == "bullish" or pos > neg:
            return "Long"
        if mom == "bearish" or neg > pos:
            return "Short"
        return "Flat"

    if persona == "Loss Aversion":
        # Demands strong positive evidence; otherwise Flat.
        strong_pos = (net >= 0.20) and (mom != "bearish") and (neg < 0.20)
        strong_neg = (net <= -0.20) and (mom != "bullish") and (pos < 0.20)
        if strong_pos:
            return "Long"
        if strong_neg:
            return "Short"
        return "Flat"

    if persona == "Herding":
        # Momentum is the primary cue; news sentiment is a tiebreaker.
        if mom == "bullish":
            return "Long" if net >= -0.15 else "Flat"
        if mom == "bearish":
            return "Short" if net <= 0.15 else "Flat"
        # Neutral momentum -> follow news sentiment if strong enough.
        if net >= 0.10:
            return "Long"
        if net <= -0.10:
            return "Short"
        return "Flat"

    raise ValueError(f"Unknown persona: {persona}")


def _strict_parse_label(raw: str) -> str:
    """Parse the unified provider response into one of Long/Short/Flat."""
    if raw is None:
        raise ValueError("empty response")
    text = raw.strip()
    # Try strict JSON first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "position" in obj:
            cand = str(obj["position"]).strip().capitalize()
            if cand in ALLOWED_LABELS:
                return cand
    except json.JSONDecodeError:
        pass
    # Fallback: scan tokens.
    upper = text.upper()
    for lbl in ALLOWED_LABELS:
        if lbl.upper() in upper:
            return lbl
    raise ValueError(f"Could not parse label from: {raw!r}")


def _build_user_prompt(persona: str, payload: dict) -> str:
    return (
        f"Persona: {persona}\n"
        f"Stock: {payload['asset']}\n"
        f"Date: {payload['date']}\n"
        f"Close price (as-of, no future info): {payload['prices']}\n"
        f"Momentum signal: {payload['momentum']}\n"
        f"FinBERT sentiment (positive/neutral/negative/net): "
        f"{payload['sentiment']['positive']:.3f} / "
        f"{payload['sentiment']['neutral']:.3f} / "
        f"{payload['sentiment']['negative']:.3f} / "
        f"{payload['sentiment']['net']:+.3f}\n"
        f"News digest: {payload['news_excerpt']}\n"
        f"Filing digest: {payload['filing_excerpt'] or 'none'}\n\n"
        f"Decide a single position for the next trading session given this "
        f"persona. Respond with strict JSON only: "
        f'{{"position": "Long" | "Short" | "Flat"}}'
    )


def generate_persona_label(
    provider: str,
    persona_prompt: str,
    input_text: str,
    *,
    model: str | None = None,
) -> dict:
    """Unified label interface across providers.

    Returns ``{"position": "Long" | "Short" | "Flat"}`` only. Providers other
    than ``finbert`` are optional and require the corresponding API key.
    """
    provider = provider.lower()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model or OPENAI_DEFAULT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": input_text},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        return {"position": _strict_parse_label(raw)}

    if provider == "gemini":
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        m = genai.GenerativeModel(model or GEMINI_DEFAULT_MODEL)
        resp = m.generate_content(
            [persona_prompt, input_text],
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        return {"position": _strict_parse_label(resp.text)}

    if provider == "claude":
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model or CLAUDE_DEFAULT_MODEL,
            max_tokens=64,
            temperature=0,
            system=persona_prompt,
            messages=[{"role": "user", "content": input_text}],
        )
        raw = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return {"position": _strict_parse_label(raw)}

    if provider == "deepseek":
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        resp = client.chat.completions.create(
            model=model or DEEPSEEK_DEFAULT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": input_text},
            ],
        )
        raw = resp.choices[0].message.content
        return {"position": _strict_parse_label(raw)}

    raise ValueError(f"Unsupported provider: {provider}")


# --------------------------------------------------------------------------- #
# Sample assembly
# --------------------------------------------------------------------------- #


def _excerpt(texts: list[str], max_chars: int = 1200) -> str:
    """Concatenate text snippets with a hard char cap for prompt safety."""
    joined = " | ".join(t.strip() for t in texts if t and t.strip())
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 3] + "..."


def build_sample(
    *,
    asset: str,
    date: pd.Timestamp,
    prices: float,
    momentum: str,
    news: list[str],
    filings: list[str],
    sentiment: SentimentScores,
    persona: str,
    label: str,
    label_source: str,
) -> dict:
    news_excerpt = _excerpt(news, max_chars=1500)
    filing_excerpt = _excerpt(filings, max_chars=1500)
    payload = {
        "asset": asset,
        "date": date.strftime("%Y-%m-%d"),
        "prices": round(float(prices), 4),
        "momentum": momentum,
        "sentiment": sentiment.to_dict(),
        "news_excerpt": news_excerpt,
        "filing_excerpt": filing_excerpt,
    }
    persona_prompt = PERSONA_PROMPT_TEMPLATES[persona]
    user_prompt = _build_user_prompt(persona, payload)
    assistant = json.dumps({"position": label})

    return {
        "id": f"{asset}-{payload['date']}-{persona.replace(' ', '_')}",
        "asset": asset,
        "date": payload["date"],
        "persona": persona,
        "label": label,
        "label_source": label_source,
        "momentum": momentum,
        "prices": payload["prices"],
        "sentiment": payload["sentiment"],
        "messages": [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "news_excerpt": news_excerpt,
            "filing_excerpt": filing_excerpt,
            "news_count": len(news),
            "filing_count": len(filings),
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def discover_parquets(data_dir: Path) -> list[Path]:
    files = sorted(p for p in data_dir.iterdir() if p.suffix == ".parquet")
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    return files


def run_pipeline(
    *,
    label_provider: str = LABEL_PROVIDER,
    openai_model: str = OPENAI_DEFAULT_MODEL,
    limit_stocks: int | None = None,
    api_workers: int = 8,
    verbose: bool = True,
) -> list[dict]:
    random.seed(SEED)
    np.random.seed(SEED)

    parquet_files = discover_parquets(DATA_DIR)
    if limit_stocks:
        parquet_files = parquet_files[:limit_stocks]

    scorer = FinBERTScorer(FINBERT_MODEL)

    # Pass 1 — load, align, run FinBERT, compute rule labels. This stays
    # serial so FinBERT (single-process model) isn't contested.
    tasks: list[dict] = []
    rows_processed = 0
    for pq in parquet_files:
        df = load_stock_frame(pq)
        aligned = align_to_trading_days(df)
        if verbose:
            print(
                f"[{pq.stem}] raw_rows={len(df)} trading_day_rows={len(aligned)}",
                flush=True,
            )
        for _, row in aligned.iterrows():
            sentiment_inputs = list(row["news"]) + list(row["10k"]) + list(row["10q"])
            sentiment = scorer.score_texts(sentiment_inputs)
            has_filing = bool(row["10k"] or row["10q"])
            for persona in PERSONAS:
                rule_label = persona_label_rule(
                    sentiment=sentiment,
                    momentum=row["momentum"],
                    persona=persona,
                    has_filing=has_filing,
                )
                tasks.append(
                    {
                        "asset": row["asset"],
                        "date": row["date"],
                        "prices": float(row["prices"]),
                        "momentum": row["momentum"],
                        "news": list(row["news"]),
                        "filings": list(row["10k"]) + list(row["10q"]),
                        "sentiment": sentiment,
                        "persona": persona,
                        "rule_label": rule_label,
                    }
                )
            rows_processed += 1
            if verbose and rows_processed % 25 == 0:
                print(f"  ... {rows_processed} trading-day rows scored", flush=True)

    # Pass 2 — optional teacher labels. Parallelised via a thread pool because
    # OpenAI / Anthropic / Gemini SDKs are I/O-bound. Default 8 workers.
    api_results: dict[int, tuple[str, str]] = {}
    if label_provider != "finbert":
        if verbose:
            print(
                f"[api] dispatching {len(tasks)} {label_provider} calls "
                f"({openai_model if label_provider == 'openai' else 'default-model'}, "
                f"{api_workers} workers)",
                flush=True,
            )

        def _one_call(idx: int, t: dict) -> tuple[int, str, str]:
            payload = {
                "asset": t["asset"],
                "date": t["date"].strftime("%Y-%m-%d"),
                "prices": t["prices"],
                "momentum": t["momentum"],
                "sentiment": t["sentiment"].to_dict(),
                "news_excerpt": _excerpt(t["news"], 1200),
                "filing_excerpt": _excerpt(t["filings"], 1200),
            }
            input_text = _build_user_prompt(t["persona"], payload)
            persona_prompt = PERSONA_PROMPT_TEMPLATES[t["persona"]]
            try:
                api_resp = generate_persona_label(
                    provider=label_provider,
                    persona_prompt=persona_prompt,
                    input_text=input_text,
                    model=openai_model if label_provider == "openai" else None,
                )
                return idx, api_resp["position"], f"{label_provider}+finbert"
            except Exception as exc:
                print(
                    f"  [warn] {label_provider} failed for {t['asset']} "
                    f"{t['date'].date()} / {t['persona']}: {exc} -- "
                    f"falling back to FinBERT rule",
                    file=sys.stderr,
                    flush=True,
                )
                return idx, t["rule_label"], "finbert+rule(fallback)"

        completed = 0
        with ThreadPoolExecutor(max_workers=api_workers) as pool:
            futs = [pool.submit(_one_call, i, t) for i, t in enumerate(tasks)]
            for fut in as_completed(futs):
                idx, label, source = fut.result()
                api_results[idx] = (label, source)
                completed += 1
                if verbose and completed % 100 == 0:
                    print(
                        f"  [api] {completed}/{len(tasks)} done", flush=True
                    )
        if verbose:
            print(f"  [api] {completed}/{len(tasks)} done", flush=True)

    # Pass 3 — assemble final samples in deterministic task order, then shuffle.
    samples: list[dict] = []
    for i, t in enumerate(tasks):
        if label_provider == "finbert":
            label, label_source = t["rule_label"], "finbert+rule"
        else:
            label, label_source = api_results[i]
        if label not in ALLOWED_LABELS:
            raise RuntimeError(
                f"Disallowed label {label!r} for {t['asset']} "
                f"{t['date']} / {t['persona']}"
            )
        samples.append(
            build_sample(
                asset=t["asset"],
                date=t["date"],
                prices=t["prices"],
                momentum=t["momentum"],
                news=t["news"],
                filings=t["filings"],
                sentiment=t["sentiment"],
                persona=t["persona"],
                label=label,
                label_source=label_source,
            )
        )

    random.Random(SEED).shuffle(samples)
    return samples


def write_outputs(samples: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=2)

    # Flatten for CSV. Keep messages as JSON strings.
    flat = []
    for s in samples:
        flat.append(
            {
                "id": s["id"],
                "asset": s["asset"],
                "date": s["date"],
                "persona": s["persona"],
                "label": s["label"],
                "label_source": s["label_source"],
                "momentum": s["momentum"],
                "prices": s["prices"],
                "sent_positive": s["sentiment"]["positive"],
                "sent_neutral": s["sentiment"]["neutral"],
                "sent_negative": s["sentiment"]["negative"],
                "sent_net": s["sentiment"]["net"],
                "sent_label": s["sentiment"]["label"],
                "news_count": s["meta"]["news_count"],
                "filing_count": s["meta"]["filing_count"],
                "system": s["messages"][0]["content"],
                "user": s["messages"][1]["content"],
                "assistant": s["messages"][2]["content"],
            }
        )
    pd.DataFrame(flat).to_csv(OUT_CSV, index=False)


def print_stats(samples: list[dict]) -> None:
    total = len(samples)
    by_label = Counter(s["label"] for s in samples)
    by_persona = Counter(s["persona"] for s in samples)
    by_stock = Counter(s["asset"] for s in samples)
    by_source = Counter(s["label_source"] for s in samples)
    by_persona_label = Counter((s["persona"], s["label"]) for s in samples)

    unique_days = {(s["asset"], s["date"]) for s in samples}
    unique_dates = sorted({s["date"] for s in samples})
    leakage_keys = {"future_price_diff", "future_return", "future_close"}
    leakage_hits = sum(
        1 for s in samples if any(k in s["meta"]["news_excerpt"].lower() for k in leakage_keys)
    )

    print()
    print("=" * 64)
    print("PERSONA SFT DATASET STATISTICS")
    print("=" * 64)
    print(f"Total samples:                {total}")
    print(f"Unique (stock, trading-day):  {len(unique_days)}")
    print(f"Unique trading days covered:  {len(unique_dates)}")
    print(f"Date range:                   {unique_dates[0]} -> {unique_dates[-1]}")
    print()
    print("Label distribution:")
    for lbl in ALLOWED_LABELS:
        c = by_label.get(lbl, 0)
        print(f"  {lbl:<6}  {c:>5}  ({c / total:.1%})")
    print()
    print("Persona distribution:")
    for p in PERSONAS:
        c = by_persona.get(p, 0)
        print(f"  {p:<16}  {c:>5}")
    print()
    print("Persona x label:")
    for p in PERSONAS:
        row = " ".join(f"{lbl}={by_persona_label.get((p, lbl), 0):>4}" for lbl in ALLOWED_LABELS)
        print(f"  {p:<16}  {row}")
    print()
    print("Stock distribution:")
    for stock, c in sorted(by_stock.items()):
        print(f"  {stock:<6}  {c:>5}")
    print()
    print("Label source distribution:")
    for src, c in sorted(by_source.items()):
        print(f"  {src:<22}  {c:>5}")
    print()
    print("Sanity checks:")
    print(f"  All labels in {ALLOWED_LABELS}: {set(by_label).issubset(set(ALLOWED_LABELS))}")
    print(f"  Exactly {len(PERSONAS)} samples per (stock, day): "
          f"{all(v == len(PERSONAS) for v in Counter((s['asset'], s['date']) for s in samples).values())}")
    print(f"  Future-leakage keyword hits in prompts: {leakage_hits} (expect 0)")
    print(f"  Output JSON: {OUT_JSON}")
    print(f"  Output CSV : {OUT_CSV}")
    print("=" * 64)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-provider",
        default=LABEL_PROVIDER,
        choices=["finbert", "openai", "gemini", "claude", "deepseek"],
        help="Label backbone. Default 'finbert' uses local weak labeling only.",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_DEFAULT_MODEL,
        help="OpenAI model id when --label-provider=openai.",
    )
    parser.add_argument(
        "--limit-stocks",
        type=int,
        default=None,
        help="Optional cap on number of parquet files (smoke testing).",
    )
    parser.add_argument(
        "--api-workers",
        type=int,
        default=8,
        help="Thread-pool size for API teacher calls (no effect for finbert).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress logs.",
    )
    args = parser.parse_args()

    samples = run_pipeline(
        label_provider=args.label_provider,
        openai_model=args.openai_model,
        limit_stocks=args.limit_stocks,
        api_workers=args.api_workers,
        verbose=not args.quiet,
    )
    write_outputs(samples)
    print_stats(samples)


if __name__ == "__main__":
    main()
