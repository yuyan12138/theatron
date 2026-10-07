"""
data_loader.py — Load and pre-process per-symbol parquet files.

Raw parquet schema:
    date, asset, prices, news, 10k, 10q, momentum, future_price_diff

Output DataFrame columns:
    date, symbol, close_price, news, sec_filing

Processing pipeline:
    1. Rename fields (asset→symbol, prices→close_price)
    2. Drop leakage columns (momentum, future_price_diff)
    3. Normalize list fields (news, 10k, 10q)
    4. Identify NYSE trading days (excludes weekends + US market holidays)
    5. Forward-roll non-trading-day news onto the next trading day,
       splicing all accumulated articles into one combined article
    6. Keep only trading-day rows
"""

import glob
import logging
import os
from datetime import date
from typing import List, Set

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# NYSE calendar helper
# --------------------------------------------------------------------------- #

def _get_nyse_trading_days(start: date, end: date) -> Set[date]:
    """Return the set of NYSE trading days between *start* and *end* inclusive."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    return set(schedule.index.date)


# --------------------------------------------------------------------------- #
# List field normalization
# --------------------------------------------------------------------------- #

def _normalize_list_field(value) -> List[str]:
    """
    Coerce a raw field value into a clean list of non-empty strings.

    Handles:
    - None → []
    - Plain string → [string] (if non-empty)
    - pyarrow ChunkedArray / Array → converted via .to_pylist()
    - list / other iterables → iterate, convert each item
    """
    if value is None:
        return []
    if hasattr(value, "to_pylist"):
        value = value.to_pylist()
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not hasattr(value, "__iter__"):
        text = str(value).strip()
        return [text] if text else []
    result = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result


# --------------------------------------------------------------------------- #
# News forward-rolling
# --------------------------------------------------------------------------- #

def _roll_nontrading_news(df: pd.DataFrame, trading_days: Set[date]) -> pd.DataFrame:
    """
    For every non-trading day (weekend or US market holiday), accumulate its
    news items and prepend them to the next trading day's news, splicing
    everything into one combined article.

    Returns a DataFrame containing only trading-day rows, with the merged
    news stored as a single-element list per row.
    """
    df = df.sort_values("date").reset_index(drop=True)

    pending_news: List[str] = []   # news accumulated from non-trading days
    extra: dict = {}               # trading_date -> list[str] to prepend

    for _, row in df.iterrows():
        d = row["date"].date()
        day_news = list(row["news"])   # already normalized list[str]

        if d not in trading_days:
            # Non-trading day: buffer the news
            pending_news.extend(day_news)
        else:
            # Trading day: collect any buffered news and clear
            if pending_news:
                extra[d] = pending_news[:]
                pending_news = []

    # Filter to trading days
    trading_mask = df["date"].apply(lambda dt: dt.date() in trading_days)
    result = df[trading_mask].copy().reset_index(drop=True)

    # Merge extra news into each trading day and splice into one article
    def _merge(row):
        d = row["date"].date()
        prior = extra.get(d, [])
        combined = prior + list(row["news"])
        if not combined:
            return []
        if len(combined) == 1:
            return combined
        # Splice multiple articles into one with a clear separator
        return ["\n\n---\n\n".join(combined)]

    result["news"] = result.apply(_merge, axis=1)
    return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_symbol_data(symbol: str) -> pd.DataFrame:
    """
    Find and load the parquet file for *symbol*, apply normalization,
    forward-roll non-trading-day news, and return a clean DataFrame
    containing only NYSE trading days.

    Raises:
        FileNotFoundError: if no matching parquet file is found.
        ValueError: if the file contains no rows after filtering.
    """
    pattern = os.path.join(DATA_DIR, f"{symbol}-*.parquet")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No parquet file found for symbol '{symbol}' in {DATA_DIR} "
            f"(pattern: {pattern})"
        )

    filepath = sorted(matches)[0]
    if len(matches) > 1:
        logger.warning(
            "Multiple parquet files match symbol '%s': %s — using %s",
            symbol, matches, filepath,
        )

    logger.info("Loading data from %s", filepath)
    raw_df = pd.read_parquet(filepath)

    # ------------------------------------------------------------------ #
    # 1. Column renaming
    # ------------------------------------------------------------------ #
    rename_map = {}
    if "asset" in raw_df.columns:
        rename_map["asset"] = "symbol"
    if "prices" in raw_df.columns:
        rename_map["prices"] = "close_price"
    if rename_map:
        raw_df = raw_df.rename(columns=rename_map)

    # ------------------------------------------------------------------ #
    # 2. Leakage prevention
    # ------------------------------------------------------------------ #
    forbidden = {"momentum", "future_price_diff"}
    cols_to_drop = [c for c in raw_df.columns if c in forbidden]
    if cols_to_drop:
        raw_df = raw_df.drop(columns=cols_to_drop)
        logger.debug("Dropped leakage columns: %s", cols_to_drop)

    # ------------------------------------------------------------------ #
    # 3. Filter to the requested symbol
    # ------------------------------------------------------------------ #
    if "symbol" in raw_df.columns:
        df = raw_df[raw_df["symbol"] == symbol].copy()
    else:
        df = raw_df.copy()
        df["symbol"] = symbol

    if df.empty:
        raise ValueError(f"No rows found for symbol '{symbol}' after filtering.")

    # ------------------------------------------------------------------ #
    # 4. Date parsing and sorting
    # ------------------------------------------------------------------ #
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 5. Normalize list fields (before rolling so news is ready to merge)
    # ------------------------------------------------------------------ #
    for col in ["news", "10k", "10q"]:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_list_field)
        else:
            df[col] = [[] for _ in range(len(df))]

    # ------------------------------------------------------------------ #
    # 6. Forward-roll non-trading-day news onto next trading day
    # ------------------------------------------------------------------ #
    start_date = df["date"].min().date()
    end_date   = df["date"].max().date()
    trading_days = _get_nyse_trading_days(start_date, end_date)

    total_before = len(df)
    df = _roll_nontrading_news(df, trading_days)
    dropped = total_before - len(df)
    logger.info(
        "Removed %d non-trading days (weekends + holidays); "
        "%d NYSE trading days remain.",
        dropped, len(df),
    )

    # ------------------------------------------------------------------ #
    # 7. Build sec_filing and select final columns
    # ------------------------------------------------------------------ #
    df["sec_filing"] = df.apply(lambda r: r["10k"] + r["10q"], axis=1)

    if "close_price" not in df.columns:
        raise ValueError(
            f"Expected column 'close_price' (originally 'prices') not found "
            f"in {filepath}. Available columns: {list(df.columns)}"
        )

    final_cols = ["date", "symbol", "close_price", "news", "sec_filing"]
    df = df[final_cols].copy().reset_index(drop=True)

    logger.info(
        "Loaded %d trading days for '%s' (%s to %s).",
        len(df), symbol,
        df["date"].iloc[0].date(),
        df["date"].iloc[-1].date(),
    )
    return df
