from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from .common import ROOT, sha256_file, sha256_text


def _normalise_list(value) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "to_pylist"):
        value = value.to_pylist()
    if isinstance(value, str):
        value = [value]
    elif not hasattr(value, "__iter__"):
        value = [value]
    result = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _nyse_days(start: date, end: date) -> set[date]:
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    return set(schedule.index.date)


def _roll_news(df: pd.DataFrame, trading_days: set[date]) -> pd.DataFrame:
    pending: list[str] = []
    prepend: dict[date, list[str]] = {}
    for _, row in df.sort_values("date").iterrows():
        current_date = row["date"].date()
        current_news = list(row["news"])
        if current_date not in trading_days:
            pending.extend(current_news)
        elif pending:
            prepend[current_date] = pending[:]
            pending.clear()

    result = df[df["date"].map(lambda value: value.date() in trading_days)].copy()
    result = result.sort_values("date").reset_index(drop=True)

    def merge(row) -> list[str]:
        combined = prepend.get(row["date"].date(), []) + list(row["news"])
        if not combined:
            return []
        return ["\n\n---\n\n".join(combined)]

    result["news"] = result.apply(merge, axis=1)
    return result


def canonical_date_price_hash(df: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{value.date():%Y-%m-%d},{float(price):.10f}"
        for value, price in zip(df["date"], df["close_price"])
    )
    return sha256_text(payload)


def load_frozen_symbol(
    symbol: str, protocol: dict
) -> tuple[pd.DataFrame, dict]:
    data_cfg = protocol["data"]
    if symbol != "AAPL":
        raise RuntimeError(f"AAPL R2 protocol does not define symbol {symbol}")
    path = (ROOT / data_cfg["source_dir"] / data_cfg["source_file"]).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing frozen AAPL R2 parquet: {path}")
    raw_hash = sha256_file(path)
    expected_raw_hash = data_cfg["raw_sha256"][symbol]
    if raw_hash != expected_raw_hash:
        raise RuntimeError(
            f"Raw data hash mismatch for {symbol}: "
            f"{raw_hash} != {expected_raw_hash}"
        )

    df = pd.read_parquet(path)
    if len(df) != int(data_cfg["expected_raw_rows"]):
        raise RuntimeError(
            f"{symbol}: expected {data_cfg['expected_raw_rows']} raw rows, "
            f"found {len(df)}"
        )
    df = df.rename(columns={"asset": "symbol", "prices": "close_price"})
    forbidden_present = sorted(
        set(data_cfg["forbidden_input_columns"]) & set(df.columns)
    )
    df = df.drop(columns=forbidden_present)
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    else:
        df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for column in ["news", "10k", "10q"]:
        if column not in df:
            df[column] = [[] for _ in range(len(df))]
        else:
            df[column] = df[column].map(_normalise_list)
    if not df["news"].map(bool).all():
        missing = df.loc[~df["news"].map(bool), "date"].dt.strftime("%Y-%m-%d")
        raise RuntimeError(f"{symbol}: empty repaired news rows: {missing.tolist()}")

    trading_days = _nyse_days(df["date"].min().date(), df["date"].max().date())
    df = _roll_news(df, trading_days)
    df["sec_filing"] = df.apply(
        lambda row: row["10k"] + row["10q"], axis=1
    )
    df = df[
        ["date", "symbol", "close_price", "news", "sec_filing"]
    ].reset_index(drop=True)

    expected_n = int(data_cfg["expected_trading_days"])
    if len(df) != expected_n:
        raise RuntimeError(f"{symbol}: expected {expected_n} observations, got {len(df)}")
    if df["date"].duplicated().any():
        raise RuntimeError(f"{symbol}: duplicate dates after loading")
    if df["date"].iloc[0].strftime("%Y-%m-%d") != data_cfg["start_date"]:
        raise RuntimeError(f"{symbol}: unexpected start date")
    if df["date"].iloc[-1].strftime("%Y-%m-%d") != data_cfg["end_date"]:
        raise RuntimeError(f"{symbol}: unexpected end date")
    if not df["date"].map(lambda value: value.date() in trading_days).all():
        raise RuntimeError(f"{symbol}: non-NYSE day survived filtering")

    processed_hash = canonical_date_price_hash(df)
    expected_processed = data_cfg["processed_date_price_sha256"][symbol]
    if processed_hash != expected_processed:
        raise RuntimeError(
            f"Processed date/price hash mismatch for {symbol}: "
            f"{processed_hash} != {expected_processed}"
        )

    provenance = {
        "path": str(path),
        "raw_sha256": raw_hash,
        "processed_date_price_sha256": processed_hash,
        "rows": len(df),
        "start_date": data_cfg["start_date"],
        "end_date": data_cfg["end_date"],
        "forbidden_columns_removed": forbidden_present,
        "text_evidence_policy": data_cfg["text_evidence_policy"],
    }
    return df, provenance


def safe_join(values: Iterable[str] | str | None) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    return " ".join(
        str(value).strip() for value in values if str(value).strip()
    )
