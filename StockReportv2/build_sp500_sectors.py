#!/usr/bin/env python3
"""
Build a YAML sector mapping for current S&P 500 constituents.

Run with no flags to write `sp500sectors.yml` next to this file:
    python build_sp500_sectors.py

Optional (kept for flexibility):
    python build_sp500_sectors.py --csv data/sp500_sectors.csv
    python build_sp500_sectors.py --merge config.yml

Notes:
- Source: Wikipedia "List of S&P 500 companies" (Symbol + GICS Sector).
- YAML output shape matches your config style:
    tickers:
      sectors:
        technology: ["AAPL","MSFT",...]
        communication_services: ["GOOGL","META",...]
        ...
"""
from __future__ import annotations

import sys
import argparse
from typing import Dict, List, Iterable
from io import StringIO
from pathlib import Path

import requests
import pandas as pd
import yaml
from datetime import datetime

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_MOBILE_URL = "https://en.m.wikipedia.org/wiki/List_of_S%26P_500_companies"

HEADERS = {
    # Spoof a normal browser UA to avoid 403
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://en.wikipedia.org/",
}

SECTOR_KEY_MAP = {
    "Information Technology": "technology",
    "Communication Services": "communication_services",
    "Consumer Discretionary": "consumer_discretionary",
    "Consumer Staples": "consumer_staples",
    "Energy": "energy",
    "Financials": "financials",
    "Health Care": "health_care",
    "Industrials": "industrials",
    "Materials": "materials",
    "Real Estate": "real_estate",
    "Utilities": "utilities",
}

DEFAULT_YML = Path("sp500sectors.yml")  # ← hard-wired default output


def _fetch_html_with_fallback(urls: Iterable[str]) -> str:
    """Fetch HTML with a browsery UA; try multiple URLs to dodge 403s."""
    s = requests.Session()
    for url in urls:
        try:
            r = s.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and r.text:
                return r.text
        except requests.RequestException:
            pass
    raise RuntimeError("Failed to fetch Wikipedia page (403 or network error).")


def _normalize_symbol(sym: str) -> str:
    """Uppercase; replace '.' with '-' (e.g., BRK.B -> BRK-B) for yfinance compat."""
    s = (sym or "").strip().upper()
    if "." in s and not s.startswith("^"):
        s = s.replace(".", "-")
    return s


def fetch_sp500_table() -> pd.DataFrame:
    """Download and parse the S&P 500 constituents table."""
    html = _fetch_html_with_fallback([WIKI_URL, WIKI_MOBILE_URL])
    tables = pd.read_html(StringIO(html), flavor="lxml")
    target = None
    for df in tables:
        cols = [str(c).strip().lower() for c in df.columns]
        has_symbol = any(c in ("symbol", "ticker") for c in cols)
        has_sector  = any(c in ("gics sector", "sector") for c in cols)
        if has_symbol and has_sector:
            target = df
            break
    if target is None:
        raise RuntimeError("Could not find S&P 500 table (Symbol + GICS Sector) on the page.")
    symbol_col = next(c for c in target.columns if str(c).strip().lower() in ("symbol", "ticker"))
    sector_col = next(c for c in target.columns if str(c).strip().lower() in ("gics sector", "sector"))
    df = target[[symbol_col, sector_col]].rename(columns={symbol_col: "Symbol", sector_col: "GICS Sector"})
    df["Symbol"] = df["Symbol"].astype(str).map(_normalize_symbol)
    df["GICS Sector"] = df["GICS Sector"].astype(str).str.strip()
    return df


def build_mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {v: [] for v in SECTOR_KEY_MAP.values()}
    for sector, sub in df.groupby("GICS Sector"):
        key = SECTOR_KEY_MAP.get(sector) or sector.lower().replace(" ", "_")
        mapping.setdefault(key, [])
        tickers = sorted(
            sub["Symbol"].dropna().astype(str).map(_normalize_symbol).unique().tolist()
        )
        mapping[key].extend(tickers)
    # Deduplicate & sort
    for k, vals in mapping.items():
        mapping[k] = sorted(dict.fromkeys(vals))
    return mapping


def to_yaml(mapping: Dict[str, List[str]]) -> str:
    payload = {"tickers": {"sectors": mapping}}

    # Flow-style (compact) lists just for readability
    class FlowSeq(list):
        pass

    def flow_representer(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    # IMPORTANT FIX: register on SafeDumper (used by yaml.safe_dump)
    yaml.SafeDumper.add_representer(FlowSeq, flow_representer)

    payload_compact = {"tickers": {"sectors": {k: FlowSeq(v) for k, v in mapping.items()}}}
    return yaml.safe_dump(payload_compact, sort_keys=True, width=200, allow_unicode=True)


def merge_into_config(mapping: Dict[str, List[str]], cfg_path: Path) -> None:
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("tickers", {})
    cfg["tickers"]["sectors"] = {k: list(v) for k, v in mapping.items()}
    # backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = cfg_path.with_suffix(cfg_path.suffix + f".bak_{ts}")
    with open(bak, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=200, allow_unicode=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=200, allow_unicode=True)
    total = sum(len(v) for v in mapping.values())
    print(f"[ok] Updated {cfg_path.name} with {total} tickers across {len(mapping)} sectors (backup: {bak.name}).")


def main():
    ap = argparse.ArgumentParser(add_help=False)  # flags optional; defaults hard-wired
    ap.add_argument("--csv", metavar="CSV", help="Write the raw Symbol/Sector CSV")
    ap.add_argument("--merge", metavar="CONFIG_YML", help="Update config.yml in-place (with backup)")
    ap.add_argument("--yml", metavar="YML", default=str(DEFAULT_YML), help="Write a standalone YAML sector file")
    args, _ = ap.parse_known_args()

    df = fetch_sp500_table()
    mapping = build_mapping(df)

    # Always write YAML to DEFAULT_YML unless overridden
    yml_path = Path(args.yml)
    yml_path.write_text(to_yaml(mapping), encoding="utf-8")
    total = sum(len(v) for v in mapping.values())
    print(f"[ok] Wrote YAML: {yml_path} with {total} tickers across {len(mapping)} sectors.")

    if args.csv:
        p = Path(args.csv); p.parent.mkdir(parents=True, exist_ok=True)
        df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector"}).to_csv(p, index=False)
        print(f"[ok] Wrote CSV: {p}")

    if args.merge:
        merge_into_config(mapping, Path(args.merge))


if __name__ == "__main__":
    main()
