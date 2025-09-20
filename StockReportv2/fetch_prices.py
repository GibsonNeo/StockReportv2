
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import time
import random
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Robust Yahoo download with retries, chunking, backoff, jitter, and audit-friendly logs
# --------------------------------------------------------------------------------------
#
# This module exposes a single public entry point:
#     fetch_5y_daily(tickers: List[str]) -> Dict[str, DataFrame]
#
# It implements the retry/round behavior controlled by the following knobs. They are
# loaded from config.yml's "download:" section if present; otherwise sane defaults.
#
# download:
#   chunk_size: 24
#   chunk_attempts: 3
#   rounds: 2
#   round_pause_seconds: 2.0
#   backoff_base_seconds: 1.5
#   jitter_seconds: 0.25
#   shuffle_each_round: true
#   prefer_adj_close: true
#   drop_partial_last_row_if_today: true
#   require_full: false
#   throttle_between_batches_seconds: 0.2
#   auto_adjust: false
#
# NOTES
# - We deliberately DO NOT depend on load_config.load_config here to keep this module
#   standalone and avoid any chance of circular imports. We parse config.yml directly.
# - We keep all comments; feel free to update but please do not remove per project rules.
# - We standardize output columns: ['Open','High','Low','Close','Adj Close','Volume']
# - If prefer_adj_close: we set Close <- Adj Close when Adj Close is present.
# - If drop_partial_last_row_if_today: drop the last row if it's today's date and has
#   any NaNs in O/H/L/C/Adj Close/Volume (intra-day partial row).
# - If require_full: treat tickers that returned an empty frame as "failed" and retry
#   in subsequent rounds.
# - We always return a mapping for every requested ticker; any still-missing tickers
#   are returned as empty DataFrames with the standard columns (so downstream code
#   never KeyErrors on dict lookups).
# --------------------------------------------------------------------------------------


_DEFAULTS = {
    "chunk_size": 24,
    "chunk_attempts": 3,
    "rounds": 2,
    "round_pause_seconds": 2.0,
    "backoff_base_seconds": 1.5,
    "jitter_seconds": 0.25,
    "shuffle_each_round": True,
    "prefer_adj_close": True,
    "drop_partial_last_row_if_today": True,
    "require_full": False,
    "throttle_between_batches_seconds": 0.2,
    "auto_adjust": False,
}

_STD_COLS = ['Open','High','Low','Close','Adj Close','Volume']


def _read_download_cfg(cfg_path: Optional[Path] = None) -> dict:
    """Read download knobs from config.yml if present; else return defaults."""
    try:
        import yaml  # local dependency only; failure just yields defaults
    except Exception:
        return dict(_DEFAULTS)
    p = Path(cfg_path) if cfg_path else Path("config.yml")
    if not p.exists():
        return dict(_DEFAULTS)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        dl = data.get("download") or {}
        out = dict(_DEFAULTS)
        for k in out.keys():
            if k in dl:
                out[k] = dl[k]
        # normalize types
        out["chunk_size"] = int(out.get("chunk_size", _DEFAULTS["chunk_size"]))
        out["chunk_attempts"] = int(out.get("chunk_attempts", _DEFAULTS["chunk_attempts"]))
        out["rounds"] = int(out.get("rounds", _DEFAULTS["rounds"]))
        out["round_pause_seconds"] = float(out.get("round_pause_seconds", _DEFAULTS["round_pause_seconds"]))
        out["backoff_base_seconds"] = float(out.get("backoff_base_seconds", _DEFAULTS["backoff_base_seconds"]))
        out["jitter_seconds"] = float(out.get("jitter_seconds", _DEFAULTS["jitter_seconds"]))
        out["shuffle_each_round"] = bool(out.get("shuffle_each_round", _DEFAULTS["shuffle_each_round"]))
        out["prefer_adj_close"] = bool(out.get("prefer_adj_close", _DEFAULTS["prefer_adj_close"]))
        out["drop_partial_last_row_if_today"] = bool(out.get("drop_partial_last_row_if_today", _DEFAULTS["drop_partial_last_row_if_today"]))
        out["require_full"] = bool(out.get("require_full", _DEFAULTS["require_full"]))
        out["throttle_between_batches_seconds"] = float(out.get("throttle_between_batches_seconds", _DEFAULTS["throttle_between_batches_seconds"]))
        out["auto_adjust"] = bool(out.get("auto_adjust", _DEFAULTS["auto_adjust"]))
        return out
    except Exception:
        return dict(_DEFAULTS)


def _standardize_frame(df: pd.DataFrame, prefer_adj_close: bool, drop_partial_today: bool) -> pd.DataFrame:
    """Ensure our output has _STD_COLS and clean the last row if it's intraday partial."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_STD_COLS)

    # Ensure DatetimeIndex
    idx = pd.to_datetime(df.index)
    df = df.copy()
    df.index = idx

    # Ensure columns exist
    for c in _STD_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[_STD_COLS]

    # Optionally overwrite Close with Adj Close
    if prefer_adj_close and "Adj Close" in df:
        # When Adj Close exists but is all NaN for some older data, leave as-is
        if df["Adj Close"].notna().any():
            df["Close"] = df["Adj Close"]

    # Optionally drop today's partial last row
    if drop_partial_today and len(df.index) > 0:
        last_idx = df.index[-1]
        # consider "today" in exchange tz unknown; use date() compare in local tz
        if pd.Timestamp(last_idx).date() >= date.today():
            # drop if any NaNs among std cols (common during session)
            if df[_STD_COLS].iloc[-1].isna().any():
                df = df.iloc[:-1]

    # Drop rows that are fully NaN across core price columns
    df = df.dropna(how="all")

    return df


def _download_chunk_yf(tickers: List[str], auto_adjust: bool) -> Dict[str, pd.DataFrame]:
    """Download one chunk via yfinance. Returns {TICKER: DataFrame} best-effort.

    We use yf.download for batch efficiency; it returns:
      - MultiIndex columns when multiple tickers
      - Single DataFrame when 1 ticker
    """
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf  # local import so users without it still import module
    except Exception as e:
        # If yfinance is absent, return empty frames (caller will treat as failure)
        for t in tickers:
            out[t] = pd.DataFrame(columns=_STD_COLS)
        return out

    try:
        data = yf.download(
            tickers=tickers,
            period="5y",
            interval="1d",
            auto_adjust=bool(auto_adjust),
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        # Hard failure for the whole batch
        for t in tickers:
            out[t] = pd.DataFrame(columns=_STD_COLS)
        return out

    # Parse output
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker batch
        for t in tickers:
            try:
                df_t = data[t].copy()
            except Exception:
                df_t = pd.DataFrame(columns=_STD_COLS)
            out[t] = df_t
    elif isinstance(data, pd.DataFrame):
        # Could be single-ticker (no multiindex)
        t = tickers[0]
        out[t] = data.copy()
        for other in tickers[1:]:
            out[other] = pd.DataFrame(columns=_STD_COLS)
    else:
        # Unknown shape; mark all empty
        for t in tickers:
            out[t] = pd.DataFrame(columns=_STD_COLS)

    return out


def _retry_sleep(attempt: int, backoff_base: float, jitter: float) -> None:
    # Exponential-ish backoff with bounded jitter
    base = (backoff_base ** max(0, attempt - 1))
    wait = base + random.uniform(-jitter, jitter)
    if wait < 0:
        wait = 0.0
    time.sleep(wait)


def fetch_5y_daily(tickers: List[str], config_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Download ~5 years of daily OHLCV for tickers using a robust, round-based retry loop.

    Returns: dict {TICKER: DataFrame with columns _STD_COLS}, one key per requested ticker.
    """
    # Load knobs
    knobs = _read_download_cfg(config_path)
    chunk_size = int(knobs["chunk_size"])
    chunk_attempts = int(knobs["chunk_attempts"])
    rounds = int(knobs["rounds"])
    round_pause = float(knobs["round_pause_seconds"])
    backoff_base = float(knobs["backoff_base_seconds"])
    jitter = float(knobs["jitter_seconds"])
    shuffle_each_round = bool(knobs["shuffle_each_round"])
    prefer_adj_close = bool(knobs["prefer_adj_close"])
    drop_partial_today = bool(knobs["drop_partial_last_row_if_today"])
    require_full = bool(knobs["require_full"])
    throttle = float(knobs["throttle_between_batches_seconds"])
    auto_adjust = bool(knobs["auto_adjust"])

    tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    if not tickers:
        return {}

    # Prepare output buffers
    results: Dict[str, pd.DataFrame] = {}
    failed: set = set(tickers)

    # Rounds loop
    for r in range(1, rounds + 1):
        batch_list = [list(failed)[i:i + chunk_size] for i in range(0, len(failed), chunk_size)]
        if shuffle_each_round:
            random.shuffle(batch_list)

        print(f"[yf] round {r}/{rounds} - {sum(len(b) for b in batch_list)} tickers across {len(batch_list)} batches")

        for bi, batch in enumerate(batch_list, start=1):
            # Attempt loop for this batch
            success_this_batch: Dict[str, pd.DataFrame] = {}
            for attempt in range(1, chunk_attempts + 1):
                out = _download_chunk_yf(batch, auto_adjust=auto_adjust)
                # Standardize and filter
                cleaned: Dict[str, pd.DataFrame] = {}
                for t, df in out.items():
                    df_std = _standardize_frame(df, prefer_adj_close=prefer_adj_close, drop_partial_today=drop_partial_today)
                    cleaned[t] = df_std

                # Collect successes from this attempt
                newly_ok = []
                for t, df in cleaned.items():
                    if df is not None and not df.empty:
                        newly_ok.append(t)
                        success_this_batch[t] = df

                # Remove newly_ok from remaining batch; retry only failures
                remaining = [t for t in batch if t not in newly_ok]

                # If everything succeeded or (not require_full and we got at least 1), break
                if not remaining:
                    break

                if attempt < chunk_attempts and remaining:
                    print(f"[yf] batch {bi}/{len(batch_list)} attempt {attempt}/{chunk_attempts} - retrying {len(remaining)} tickers")
                    _retry_sleep(attempt=attempt+1, backoff_base=backoff_base, jitter=jitter)
                    batch = remaining  # retry only remaining
                else:
                    # Last attempt done; keep what we got
                    batch = remaining

            # Commit successes from this batch to results
            for t, df in success_this_batch.items():
                results[t] = df
                if t in failed:
                    failed.remove(t)

            # Throttle between batches to be nice to vendor
            if throttle > 0:
                time.sleep(throttle)

        # Round pause if there are still failures
        if failed and r < rounds and round_pause > 0:
            print(f"[yf] round {r} complete; {len(failed)} still missing. Sleeping {round_pause}s before next round.")
            time.sleep(round_pause)

        # Early exit if all done
        if not failed:
            break

    # Any still-missing tickers → return empty frames (standard columns)
    if failed:
        for t in list(failed):
            results[t] = pd.DataFrame(columns=_STD_COLS)

    # Ensure every requested ticker has an entry
    for t in tickers:
        if t not in results:
            results[t] = pd.DataFrame(columns=_STD_COLS)

    # Final info
    missing = [t for t, df in results.items() if df.empty]
    if missing:
        print(f"[yf] Completed with {len(missing)} empty downloads: {sorted(missing)[:8]}{'...' if len(missing)>8 else ''}")
    else:
        print("[yf] Completed all downloads successfully.")

    return results
