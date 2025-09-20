from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import math
import copy
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

from paths import REPORTS_DIR, AUDIT_DIR
from fetch_prices import fetch_5y_daily
from sector_lookup import resolve_sector
from columns import compute_mom_12_1_spdji

# =========================
# Config loading
# =========================

def _deep_update(base: dict, override: dict) -> dict:
    """Deep-merge override into base (non-destructive on lists)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

# Defaults are safe, you will override via allocator_config.yml (and per-portfolio overrides)
_DEFAULTS = {
    "enabled": True,
    "price_source": "last_close",
    "score_column": "mom_12_1_spdji",
    "parking_ticker": "SGOV",
    "sma_gate": {
        "enabled": True,
        "apply_to": ["stocks","etfs","commodities"],
        "lot_size_multiple": 3
    },
    # Sleeve weights (percent of total portfolio).
    # Sector targets below are INSIDE the stocks sleeve and should sum to 1.0.
    "sleeves": {
        "stocks_weight_pct": 0.70,
        "etfs_weight_pct": 0.20,
        "commodities_weight_pct": 0.10
    },
    "stocks": {
        "enabled": True,
        "per_ticker_cap_pct": 0.05,   # 5% of TOTAL portfolio per single stock
        "max_names_per_sector": 5,
        "sector_targets": {
            "Technology": 0.25,
            "Health Care": 0.15,
            "Financials": 0.12,
            "Industrials": 0.10,
            "Consumer Discretionary": 0.10,
            "Consumer Staples": 0.08,
            "Energy": 0.07,
            "Materials": 0.05,
            "Communication Services": 0.05,
            "Utilities": 0.02,
            "Real Estate": 0.01,
        }
    },
    "etfs": {
        "enabled": True,
        "per_ticker_cap_pct": 0.10,   # 10% of TOTAL portfolio per ETF
        "tickers": ["SPY","RSP","QQQ","XLK"]
    },
    "commodities": {
        "enabled": True,
        "per_ticker_cap_pct": 0.10,   # 10% of TOTAL portfolio per commodity ETF
        "tickers": ["GLD","SLV","USO"]
    },
    "score_filter": {
        "mode": "percentile",      # percentile | absolute | both
        "scope": "per_bucket",     # per_bucket | global
        "percentile_keep_top": 0.30,
        "absolute_floor": 0.0
    },
    "portfolios": [
        {
            "name": "retirement",
            "portfolio_value": 120000,
            "whole_shares_only": True,
            "cash_buffer_pct": 0.02,
            "overrides": {
                "sleeves": {
                    "stocks_weight_pct": 0.70,
                    "etfs_weight_pct": 0.20,
                    "commodities_weight_pct": 0.10
                },
                "score_filter": {
                    "mode": "both",
                    "scope": "per_bucket",
                    "percentile_keep_top": 0.30,
                    "absolute_floor": 0.0
                }
            }
        },
        {
            "name": "personal",
            "portfolio_value": 35000,
            "whole_shares_only": True,
            "cash_buffer_pct": 0.02,
            "overrides": {
                "sleeves": {
                    "stocks_weight_pct": 0.00,   # ETFs-only
                    "etfs_weight_pct": 0.98,
                    "commodities_weight_pct": 0.00
                },
                "stocks": { "enabled": False },
                "commodities": { "enabled": False },
                "etfs": {
                    "enabled": True,
                    "per_ticker_cap_pct": 0.10
                    # NOTE: If you set tickers: [] here, that means "explicitly none";
                    #       comment the key out to inherit from root config injection.
                },
                "score_filter": {
                    "mode": "percentile",
                    "scope": "per_bucket",
                    "percentile_keep_top": 0.30,
                    "absolute_floor": 0.0
                }
            }
        }
    ]
}

def load_allocator_config(path: Path) -> dict:
    """Load allocator YAML (if present) and deep-merge onto defaults."""
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
        root = y.get("portfolio_builder") or y
        cfg = _deep_update(_DEFAULTS, root)
        cfg = _normalize_portfolios(cfg)
        return cfg
    return copy.deepcopy(_DEFAULTS)

def _normalize_portfolios(cfg: dict) -> dict:
    """
    Accept both legacy list-of-portfolios with `overrides` and the new mapping form:
      portfolios:
        personal: { portfolio_value: ..., sleeves: {...}, stocks: {...}, ... }
        retirement: { ... }

    - If mapping, convert to list[{name, portfolio_value, whole_shares_only, cash_buffer_pct, planner, overrides:{...}}]
      where overrides pulls recognized knobs (sleeves/stocks/etfs/commodities/sma_gate/score_filter/parking_ticker).
    - Leaves planner dict attached to each portfolio entry for planner workbook use.
    """
    ports = cfg.get("portfolios")
    if isinstance(ports, dict):
        out_list = []
        for name, body in ports.items():
            body = dict(body or {})
            # extract known meta
            pval = body.pop("portfolio_value", 0.0)
            whole = body.pop("whole_shares_only", True)
            cashb = body.pop("cash_buffer_pct", 0.0)
            planner = body.pop("planner", {})
            # Anything else is considered overrides
            overrides = {}
            for k in ("sleeves","stocks","etfs","commodities","sma_gate","score_filter","parking_ticker"):
                if k in body:
                    overrides[k] = body.pop(k)
            # Whatever remains, keep inside overrides too (future-proof)
            overrides.update(body)
            out_list.append({
                "name": name,
                "portfolio_value": pval,
                "whole_shares_only": whole,
                "cash_buffer_pct": cashb,
                "planner": planner,
                "overrides": overrides
            })
        cfg = dict(cfg)
        cfg["portfolios"] = out_list
    # Ensure each portfolio has name field
    lst = cfg.get("portfolios") or []
    out = []
    for i,p in enumerate(lst):
        if isinstance(p, dict):
            q = dict(p)
            if "name" not in q:
                q["name"] = f"portfolio_{i+1}"
            out.append(q)
    cfg["portfolios"] = out
    return cfg


# =========================
# Math helpers
# =========================

def _last_adj_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or 'Adj Close' not in df:
        return None
    s = df['Adj Close'].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])

def _sma(series: pd.Series, length: int) -> Optional[float]:
    s = series.dropna()
    if len(s) < length:
        return None
    return float(s.rolling(length).mean().iloc[-1])

def _sma_gate_fraction(price: float, sma20: Optional[float], sma50: Optional[float], sma100: Optional[float]) -> float:
    """3/3 if >= all; 2/3 if <20 but >=50 & >=100; 1/3 if <50 but >=100; else 0."""
    if any(v is None for v in (price, sma20, sma50, sma100)):
        # conservative if SMAs unavailable
        return 0.0
    above20 = price >= sma20
    above50 = price >= sma50
    above100 = price >= sma100
    if above20 and above50 and above100:
        return 1.0
    if (not above20) and above50 and above100:
        return 2/3
    if (not above50) and above100:
        return 1/3
    return 0.0

def _canonical_sector(s: str) -> str:
    """Normalize sector names to match config keys."""
    if not s:
        return "unknown"
    x = s.strip().lower().replace("&", "and")
    for ch in ",./-":
        x = x.replace(ch, " ")
    x = "_".join(x.split())
    aliases = {
        "communication_services":"communication_services",
        "communications":"communication_services",
        "comm_services":"communication_services",
        "health_care":"health_care",
        "healthcare":"health_care",
        "consumer_discretionary":"consumer_discretionary",
        "consumer_staples":"consumer_staples",
        "information_technology":"technology",
        "tech":"technology",
        "realestate":"real_estate",
    }
    return aliases.get(x, x)

def _rank_series_desc(s: pd.Series) -> pd.Index:
    return s.sort_values(ascending=False).index

def _weight_pct(dollars: float, V: float) -> float:
    return (dollars / V) if (V and dollars) else 0.0

# =========================
# Allocation
# =========================

@dataclass
class SliceDiagnostics:
    slice: str
    candidates_total: int
    candidates_after_filter: int
    gate_full: int
    gate_2_3: int
    gate_1_3: int
    gate_0: int
    buyable_ge1lot: int
    spent: float
    reserved_to_sgov: float
    unused_budget: float

def _apply_score_filter(
    scores: pd.Series,
    mode: str = "percentile",
    scope: str = "per_bucket",
    percentile_keep_top: float = 0.30,
    absolute_floor: Optional[float] = None
) -> pd.Series:
    """Return a boolean mask (scores.index-aligned) of rows to keep."""
    s = pd.to_numeric(scores, errors="coerce")
    keep = pd.Series(False, index=s.index)
    if s.dropna().empty:
        return keep

    if mode in ("percentile","both"):
        thresh = float(s.quantile(1.0 - float(percentile_keep_top)))
        keep_percentile = s >= thresh
    else:
        keep_percentile = pd.Series(True, index=s.index)

    if mode in ("absolute","both"):
        if absolute_floor is None:
            keep_abs = pd.Series(True, index=s.index)
        else:
            keep_abs = s >= float(absolute_floor)
    else:
        keep_abs = pd.Series(True, index=s.index)

    return keep_percentile & keep_abs

def _ensure_prices(prices: Dict[str, pd.DataFrame], need: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch missing tickers and merge into the prices mapping."""
    missing = [t for t in need if t not in prices or prices[t] is None or prices[t].empty]
    if missing:
        fetched = fetch_5y_daily(missing)
        prices = dict(prices)
        prices.update(fetched)
    return prices

def _alloc_bucket(
    *,
    slice_name: str,
    portfolio_value: float,
    candidates_filtered: List[str],
    candidates_pre_filter_count: int,
    prices: Dict[str, pd.DataFrame],
    scores: pd.Series,
    sectors_map: Dict[str, str],
    per_ticker_cap_pct: float,
    bucket_budget: float,
    lot_size: int,
    whole_shares: bool,
    sma_gate_enabled: bool,
    max_names_limit: Optional[int] = None
) -> Tuple[pd.DataFrame, float, float, SliceDiagnostics]:
    """
    Allocate a single slice (sector / ETFs / commodities).
    - per_ticker_cap is now evaluated vs TOTAL portfolio_value (V), not the slice.
    Returns: (holds_df, spent_dollars, reserved_to_sgov_dollars, diagnostics)
    """
    rows = []
    remaining = float(bucket_budget)
    reserved_to_sgov = 0.0
    spent = 0.0

    # Precompute prices and SMAs
    last_prices: Dict[str, Optional[float]] = {}
    sma20: Dict[str, Optional[float]] = {}
    sma50: Dict[str, Optional[float]] = {}
    sma100: Dict[str, Optional[float]] = {}

    for t in candidates_filtered:
        df = prices.get(t)
        p = _last_adj_close(df)
        last_prices[t] = p
        if df is None or df.empty or p is None:
            sma20[t] = sma50[t] = sma100[t] = None
        else:
            s = df['Adj Close']
            sma20[t] = _sma(s, 20)
            sma50[t] = _sma(s, 50)
            sma100[t] = _sma(s, 100)

    ranked = [t for t in _rank_series_desc(scores.loc[candidates_filtered]) if pd.notna(scores.get(t))]

    # Diagnostics counters
    gate_full = gate_2_3 = gate_1_3 = gate_0 = 0
    buyable_ge1lot = 0
    used_names = 0

    for t in ranked:
        if remaining <= 0:
            break

        price = last_prices.get(t)
        # Cap is based on TOTAL portfolio value (V), not slice budget.
        cap_i_total = portfolio_value * float(per_ticker_cap_pct)
        alloc_base = min(cap_i_total, remaining)

        if price is None or not np.isfinite(price) or price <= 0:
            # no price → reserve alloc_base to SGOV
            reserved_to_sgov += alloc_base
            remaining -= alloc_base
            gate_0 += 1
            rows.append({
                "ticker": t, "bucket": slice_name, "sector": sectors_map.get(t,"Unknown"),
                "score": float(scores.get(t, np.nan)), "price": price, "f": 0.0,
                "shares": 0, "dollars": 0.0, "note": "No price → reserved to SGOV"
            })
            continue

        # SMA gate fraction
        if sma_gate_enabled:
            f = _sma_gate_fraction(price, sma20.get(t), sma50.get(t), sma100.get(t))
        else:
            f = 1.0

        if f >= 1.0:
            gate_full += 1
        elif f > 0.5:  # 2/3
            gate_2_3 += 1
        elif f > 0.0:  # 1/3
            gate_1_3 += 1
        else:
            gate_0 += 1

        if f <= 0.0:
            reserved_to_sgov += alloc_base
            remaining -= alloc_base
            rows.append({
                "ticker": t, "bucket": slice_name, "sector": sectors_map.get(t,"Unknown"),
                "score": float(scores.get(t, np.nan)), "price": price, "f": f,
                "shares": 0, "dollars": 0.0, "note": "SMA gate=0 → reserved to SGOV"
            })
            continue

        alloc_planned = f * alloc_base

        # lot rounding
        unit = lot_size if whole_shares else 1
        lot_cost = price * unit
        if lot_cost <= 0:
            lot_cost = price

        if whole_shares:
            lots = math.floor(alloc_planned / lot_cost)
            shares = lots * unit
        else:
            shares = alloc_planned / price

        if shares <= 0:
            # not enough to buy even one lot at current f/cap → reserve full alloc_base to SGOV
            reserved_to_sgov += alloc_base
            remaining -= alloc_base
            rows.append({
                "ticker": t, "bucket": slice_name, "sector": sectors_map.get(t,"Unknown"),
                "score": float(scores.get(t, np.nan)), "price": price, "f": f,
                "shares": 0, "dollars": 0.0, "note": "Insufficient for one lot → reserved to SGOV"
            })
            continue

        buyable_ge1lot += 1

        dollars = shares * price
        remaining_after_spend = remaining - dollars

        # If fully above SMAs, try to add more lots up to TOTAL-cap and remaining
        if f >= 1.0 and whole_shares:
            add_budget = max(0.0, min(cap_i_total, remaining) - dollars)
            if add_budget >= lot_cost:
                extra_lots = math.floor(add_budget / lot_cost)
                extra_shares = extra_lots * unit
                extra_dollars = extra_shares * price
                shares += extra_shares
                dollars += extra_dollars
                remaining_after_spend -= extra_dollars

        rows.append({
            "ticker": t, "bucket": slice_name, "sector": sectors_map.get(t,"Unknown"),
            "score": float(scores.get(t, np.nan)), "price": price, "f": f,
            "shares": int(shares) if whole_shares else shares,
            "dollars": float(dollars),
            "note": "" if f>=1.0 else "Partial (f<1), remainder reserved to SGOV"
        })

        if f >= 1.0:
            # reallocation allowed under f=1 → only reduce remaining by spent
            spent += dollars
            remaining = remaining_after_spend
        else:
            # remainder of alloc_base reserved to SGOV; consume full alloc_base from slice
            spent += dollars
            reserved_to_sgov += (alloc_base - dollars)
            remaining -= alloc_base

        if max_names_limit and slice_name not in ("ETFs","Commodities"):
            used_names += 1
            if used_names >= int(max_names_limit):
                break

    unused_budget = max(0.0, remaining)
    holds = pd.DataFrame(rows)

    diag = SliceDiagnostics(
        slice=slice_name,
        candidates_total=candidates_pre_filter_count,
        candidates_after_filter=len(candidates_filtered),
        gate_full=gate_full,
        gate_2_3=gate_2_3,
        gate_1_3=gate_1_3,
        gate_0=gate_0,
        buyable_ge1lot=buyable_ge1lot,
        spent=float(spent),
        reserved_to_sgov=float(reserved_to_sgov),
        unused_budget=float(unused_budget),
    )
    return holds, spent, reserved_to_sgov, diag

# =========================
# Public entry
# =========================

def build_allocator_workbook(
    base_prices: Dict[str, pd.DataFrame],
    base_universe: List[str],
    types_map: Dict[str, str],
    sectors_map: Dict[str, str],
    alloc_cfg: dict,
    out_xlsx: Optional[Path] = None,
) -> Path:
    """
    Build a separate allocator spreadsheet with one sheet per portfolio,
    plus per-portfolio Summary & Diagnostics sheets and CSV audits.
    """
    cfg = copy.deepcopy(alloc_cfg or {})
    if not cfg.get("enabled", True):
        raise SystemExit("[allocator] disabled by config")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = out_xlsx or (REPORTS_DIR / f"allocator_{ts}.xlsx")

    # Which universes are enabled globally
    stocks_enabled = (cfg.get("stocks") or {}).get("enabled", True)
    etfs_cfg = (cfg.get("etfs") or {})
    comd_cfg = (cfg.get("commodities") or {})
    etf_enabled = etfs_cfg.get("enabled", True)
    comd_enabled = comd_cfg.get("enabled", True)

    stock_tickers = list(dict.fromkeys([t for t in (base_universe or [])])) if stocks_enabled else []
    etf_tickers = list(map(str, (etfs_cfg.get("tickers", []) or []))) if etf_enabled else []
    comd_tickers = list(map(str, (comd_cfg.get("tickers", []) or []))) if comd_enabled else []
    parking_ticker = str(cfg.get("parking_ticker", "SGOV")).upper()

    need = sorted(set(stock_tickers + etf_tickers + comd_tickers + [parking_ticker]))
    prices = dict(base_prices or {})
    prices = _ensure_prices(prices, need)

    # Score metric
    if (cfg.get("score_column") or "mom_12_1_spdji").lower() != "mom_12_1_spdji":
        raise SystemExit("[allocator] only mom_12_1_spdji is supported for now")
    scores_all = compute_mom_12_1_spdji(prices, need)

    # Ensure sectors for all tickers used by allocator
    missing = [t for t in need if t not in sectors_map or not sectors_map.get(t)]
    if missing:
        # >>> NEW: try to use sp500sectors.yml in the repo root to resolve sectors
        sp500_guess = Path("sp500sectors.yml")
        sp500_path = sp500_guess if sp500_guess.exists() else None
        try:
            res = resolve_sector(missing, sp500_yml=sp500_path)
            sectors_map = dict(sectors_map)
            sectors_map.update(res)
        except Exception:
            # fall back silently
            pass
        # <<< NEW

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        for p in (cfg.get("portfolios") or []):
            name = str(p.get("name","portfolio"))
            V = float(p.get("portfolio_value", 0.0))
            whole_shares = bool(p.get("whole_shares_only", True))
            cash_buffer_pct = float(p.get("cash_buffer_pct", 0.0))
            cash_buffer = V * cash_buffer_pct

            # Effective config for this portfolio (deep-merge)
            pcfg = _deep_update(cfg, p.get("overrides") or {})

            # Lot/SMA settings
            lot_size = int(((pcfg.get("sma_gate") or {}).get("lot_size_multiple") or 3))
            sma_apply_to = set(((pcfg.get("sma_gate") or {}).get("apply_to") or ["stocks","etfs","commodities"]))
            sma_enabled = bool((pcfg.get("sma_gate") or {}).get("enabled", True))

            # Score filter
            filt_cfg = pcfg.get("score_filter") or {}
            f_mode = str(filt_cfg.get("mode","percentile")).lower()
            f_pct = float(filt_cfg.get("percentile_keep_top", 0.30) or 0.30)
            f_abs = filt_cfg.get("absolute_floor", 0.0)
            if f_abs is not None:
                f_abs = float(f_abs)

            # Sleeve weights (per-portfolio)
            sleeves = pcfg.get("sleeves") or {}
            stocks_w = float(sleeves.get("stocks_weight_pct", np.nan))
            etfs_w = float(sleeves.get("etfs_weight_pct", np.nan))
            comd_w = float(sleeves.get("commodities_weight_pct", np.nan))

            # Backward-compat: if sleeves not provided, fallback to legacy bucket weights
            if not np.isfinite(stocks_w) or not np.isfinite(etfs_w) or not np.isfinite(comd_w):
                etfs_w = etfs_w if np.isfinite(etfs_w) else float((pcfg.get("etfs") or {}).get("bucket_weight_pct", 0.0))
                comd_w = comd_w if np.isfinite(comd_w) else float((pcfg.get("commodities") or {}).get("bucket_weight_pct", 0.0))
                stocks_w = stocks_w if np.isfinite(stocks_w) else max(0.0, 1.0 - (etfs_w + comd_w))

            # Budgets by sleeve
            B_stocks = V * max(0.0, stocks_w)
            B_etfs   = V * max(0.0, etfs_w)
            B_comd   = V * max(0.0, comd_w)

            # Per-portfolio enable flags (after overrides)
            stocks_on = (pcfg.get("stocks") or {}).get("enabled", True) and (B_stocks > 0)
            etfs_on   = (pcfg.get("etfs") or {}).get("enabled", True)   and (B_etfs > 0)
            comd_on   = (pcfg.get("commodities") or {}).get("enabled", True) and (B_comd > 0)

            # Per-sleeve caps
            stock_cap_pct = float((pcfg.get("stocks") or {}).get("per_ticker_cap_pct", 0.05))
            etf_cap_pct   = float((pcfg.get("etfs") or {}).get("per_ticker_cap_pct", 0.10))
            comd_cap_pct  = float((pcfg.get("commodities") or {}).get("per_ticker_cap_pct", 0.10))
            max_names_per_sector = (pcfg.get("stocks") or {}).get("max_names_per_sector", None)

            holdings_rows: List[pd.DataFrame] = []
            diags: List[SliceDiagnostics] = []
            reserved_to_sgov_total = 0.0
            spent_total = 0.0

            # ===== STOCKS by sector (inside B_stocks) =====
            if stocks_on:
                sectors_weights = (pcfg.get("stocks") or {}).get("sector_targets") or {}
                # Normalize targets to sum to 1.0 (avoid user drift)
                cfg_sectors = { _canonical_sector(k): float(v) for k,v in sectors_weights.items() if v is not None }
                total_w = sum(v for v in cfg_sectors.values() if v >= 0)
                if total_w <= 0:
                    sector_targets_norm = {}
                else:
                    sector_targets_norm = {k: (v / total_w) for k,v in cfg_sectors.items()}

                # Candidate map per sector
                by_sector: Dict[str, List[str]] = {}
                for t in (stock_tickers or []):
                    sec = _canonical_sector(sectors_map.get(t,"Unknown"))
                    if sec in sector_targets_norm:
                        by_sector.setdefault(sec, []).append(t)

                for sec_key, rel_w in sector_targets_norm.items():
                    sec_name_display = sec_key.replace("_"," ").title()
                    B_s = B_stocks * float(rel_w)
                    if B_s <= 0:
                        continue

                    cands_all = by_sector.get(sec_key, [])
                    pre_count = len(cands_all)
                    if pre_count == 0:
                        # No candidates in this sector: move B_s to SGOV
                        reserved_to_sgov_total += B_s
                        diags.append(SliceDiagnostics(sec_name_display, 0, 0, 0,0,0,0, 0, 0.0, B_s, 0.0))
                        continue

                    # Score filter within sector
                    s_scores = scores_all.loc[cands_all].copy()
                    s_scores = pd.to_numeric(s_scores, errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
                    mask = _apply_score_filter(s_scores, mode=f_mode, scope="per_bucket",
                                               percentile_keep_top=f_pct, absolute_floor=f_abs)
                    cands = list(s_scores.index[mask])
                    if not cands:
                        # All filtered out → reserve to SGOV
                        reserved_to_sgov_total += B_s
                        diags.append(SliceDiagnostics(sec_name_display, pre_count, 0, 0,0,0,0, 0, 0.0, B_s, 0.0))
                        continue

                    holds, spent, reserved, d = _alloc_bucket(
                        slice_name=sec_name_display,
                        portfolio_value=V,
                        candidates_filtered=cands,
                        candidates_pre_filter_count=pre_count,
                        prices=prices, scores=scores_all, sectors_map=sectors_map,
                        per_ticker_cap_pct=stock_cap_pct,
                        bucket_budget=B_s,
                        lot_size=lot_size,
                        whole_shares=whole_shares,
                        sma_gate_enabled=(sma_enabled and ("stocks" in sma_apply_to)),
                        max_names_limit=max_names_per_sector
                    )
                    holdings_rows.append(holds)
                    spent_total += spent
                    reserved_to_sgov_total += reserved
                    diags.append(d)

            # ===== ETFs slice =====
            if etfs_on:
                etf_list = list(map(str, (pcfg.get("etfs") or {}).get("tickers", [])))
                pre_count = len(etf_list)
                if pre_count == 0:
                    reserved_to_sgov_total += B_etfs
                    diags.append(SliceDiagnostics("ETFs", 0, 0, 0,0,0,0, 0, 0.0, B_etfs, 0.0))
                else:
                    s_scores = scores_all.loc[etf_list].copy()
                    s_scores = pd.to_numeric(s_scores, errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
                    mask = _apply_score_filter(s_scores, mode=f_mode, scope="per_bucket",
                                               percentile_keep_top=f_pct, absolute_floor=f_abs)
                    cands = list(s_scores.index[mask])
                    if not cands:
                        reserved_to_sgov_total += B_etfs
                        diags.append(SliceDiagnostics("ETFs", pre_count, 0, 0,0,0,0, 0, 0.0, B_etfs, 0.0))
                    else:
                        holds, spent, reserved, d = _alloc_bucket(
                            slice_name="ETFs",
                            portfolio_value=V,
                            candidates_filtered=cands,
                            candidates_pre_filter_count=pre_count,
                            prices=prices, scores=scores_all, sectors_map=sectors_map,
                            per_ticker_cap_pct=etf_cap_pct,
                            bucket_budget=B_etfs,
                            lot_size=lot_size,
                            whole_shares=whole_shares,
                            sma_gate_enabled=(sma_enabled and ("etfs" in sma_apply_to)),
                            max_names_limit=None
                        )
                        holdings_rows.append(holds)
                        spent_total += spent
                        reserved_to_sgov_total += reserved
                        diags.append(d)

            # ===== Commodities slice =====
            if comd_on:
                comd_list = list(map(str, (pcfg.get("commodities") or {}).get("tickers", [])))
                pre_count = len(comd_list)
                if pre_count == 0:
                    reserved_to_sgov_total += B_comd
                    diags.append(SliceDiagnostics("Commodities", 0, 0, 0,0,0,0, 0, 0.0, B_comd, 0.0))
                else:
                    s_scores = scores_all.loc[comd_list].copy()
                    s_scores = pd.to_numeric(s_scores, errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
                    mask = _apply_score_filter(s_scores, mode=f_mode, scope="per_bucket",
                                               percentile_keep_top=f_pct, absolute_floor=f_abs)
                    cands = list(s_scores.index[mask])
                    if not cands:
                        reserved_to_sgov_total += B_comd
                        diags.append(SliceDiagnostics("Commodities", pre_count, 0, 0,0,0,0, 0, 0.0, B_comd, 0.0))
                    else:
                        holds, spent, reserved, d = _alloc_bucket(
                            slice_name="Commodities",
                            portfolio_value=V,
                            candidates_filtered=cands,
                            candidates_pre_filter_count=pre_count,
                            prices=prices, scores=scores_all, sectors_map=sectors_map,
                            per_ticker_cap_pct=comd_cap_pct,
                            bucket_budget=B_comd,
                            lot_size=lot_size,
                            whole_shares=whole_shares,
                            sma_gate_enabled=(sma_enabled and ("commodities" in sma_apply_to)),
                            max_names_limit=None
                        )
                        holdings_rows.append(holds)
                        spent_total += spent
                        reserved_to_sgov_total += reserved
                        diags.append(d)

            # ----- SGOV placement (lots of 3, same as before) -----
            unspent_total = V - cash_buffer - spent_total - reserved_to_sgov_total
            if unspent_total < 0:
                unspent_total = 0.0
            sgov_pool = reserved_to_sgov_total + unspent_total + cash_buffer

            parking_t = str(pcfg.get("parking_ticker", cfg.get("parking_ticker","SGOV"))).upper()
            if parking_t not in prices:
                prices = _ensure_prices(prices, [parking_t])

            sgov_price = _last_adj_close(prices.get(parking_t))
            if sgov_price is None or not np.isfinite(sgov_price) or sgov_price <= 0:
                sgov_shares = 0
                sgov_spent = 0.0
                leftover_cash = sgov_pool
                sgov_note = "No price → hold as cash"
            else:
                unit = lot_size if whole_shares else 1
                lot_cost = sgov_price * unit
                if lot_cost <= 0:
                    lot_cost = sgov_price
                if whole_shares:
                    lots = math.floor(sgov_pool / lot_cost)
                    sgov_shares = lots * unit
                else:
                    sgov_shares = sgov_pool / sgov_price
                sgov_spent = sgov_shares * sgov_price
                leftover_cash = sgov_pool - sgov_spent
                sgov_note = ""

            holds_all = pd.concat(holdings_rows, ignore_index=True) if holdings_rows else pd.DataFrame(columns=[
                "ticker","bucket","sector","score","price","f","shares","dollars","note"
            ])
            sgov_row = pd.DataFrame([{
                "ticker": parking_t, "bucket": "Parking", "sector": "Cash & Equivalents",
                "score": float(scores_all.get(parking_t, np.nan)), "price": sgov_price, "f": 1.0,
                "shares": int(sgov_shares) if whole_shares else sgov_shares,
                "dollars": float(sgov_spent),
                "note": sgov_note
            }])
            # Avoid pandas FutureWarning by skipping empty frames
            frames = [df_part for df_part in (holds_all, sgov_row) if df_part is not None and not df_part.empty]
            final_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
                "ticker","bucket","sector","score","price","f","shares","dollars","note"
            ])
            final_df["weight_pct"] = final_df["dollars"].apply(lambda x: _weight_pct(x, V))

            # Sheets
            final_df.to_excel(xw, sheet_name=f"Portfolio-{name}", index=False)

            # Summary sheet (slice-level dollars)
            summary_rows = []
            for d in diags:
                summary_rows.append({
                    "slice": d.slice,
                    "spent": d.spent,
                    "reserved_to_sgov": d.reserved_to_sgov,
                    "unused_budget": d.unused_budget
                })
            summary_rows.append({"slice": "Parking/SGOV", "spent": float(sgov_spent), "reserved_to_sgov": 0.0, "unused_budget": float(leftover_cash)})
            pd.DataFrame(summary_rows).to_excel(xw, sheet_name=f"Summary-{name}", index=False)

            # Diagnostics sheet
            diag_rows = [{
                "slice": d.slice,
                "candidates_total": d.candidates_total,
                "candidates_after_filter": d.candidates_after_filter,
                "gate_full(3/3)": d.gate_full,
                "gate_2/3": d.gate_2_3,
                "gate_1/3": d.gate_1_3,
                "gate_0": d.gate_0,
                "buyable_≥1_lot": d.buyable_ge1lot,
                "spent": d.spent,
                "reserved_to_sgov": d.reserved_to_sgov,
                "unused_budget": d.unused_budget
            } for d in diags]
            pd.DataFrame(diag_rows).to_excel(xw, sheet_name=f"Diagnostics-{name}", index=False)

            # Audit CSV
            audit_path = AUDIT_DIR / f"allocator_{name}_{ts}.csv"
            aud = final_df.copy()
            aud["portfolio_value"] = V
            aud.to_csv(audit_path, index=False)

    

# =========================
# Planner Workbook (no auto-allocation)
# =========================

def build_allocator_planner_workbook(
    *,
    out_xlsx: Path | None = None,
    prices: Dict[str, pd.DataFrame],
    base_universe: List[str],
    types_map: Dict[str,str],
    sectors_map: Dict[str,str],
    alloc_cfg: dict,
) -> Path:
    """
    Build a planning workbook with three sheets per portfolio:
      • Planner-% (portfolio): share ladder as % of portfolio value
      • Planner-$ (portfolio): share ladder as $ cost
      • Caps (portfolio): sleeves + per-ticker caps + sector dollars inside stocks

    Writes audit CSVs to /audit as well.
    """
    import pandas as pd
    from datetime import datetime
    from columns import compute_mom_12_1_spdji, compute_sma20_slope_3d, compute_sma_stack_hits_20_50_100

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = out_xlsx or (REPORTS_DIR / f"planner_{ts}.xlsx")

    # Ensure config normalized
    cfg = _normalize_portfolios(dict(alloc_cfg or {}))

    # Prepare universes
    tickers = list(dict.fromkeys([t for t in (base_universe or [])]))
    # Some type/sector maps may miss entries; fill with Unknown
    def _tmap(t): return types_map.get(t, "Unknown")
    def _smap(t): return sectors_map.get(t, "Unknown")

    # Pre-compute columns once
    mom = compute_mom_12_1_spdji(prices, tickers)
    slope = compute_sma20_slope_3d(prices, tickers)
    sma_stack_num = compute_sma_stack_hits_20_50_100(prices, tickers, as_fraction=False)
    sma_stack_frac= compute_sma_stack_hits_20_50_100(prices, tickers, as_fraction=True)

    # Last price helper
    def _last_price(t):
        df = prices.get(t)
        if df is None or df.empty or 'Adj Close' not in df: return float('nan')
        v = df['Adj Close'].iloc[-1]
        try:
            return float(v)
        except Exception:
            return float('nan')

    # Build sheets
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
        for p in (cfg.get("portfolios") or []):
            name = str(p.get("name","portfolio"))
            planner = p.get("planner") or {}
            if not bool(planner.get("enabled", True)):
                continue

            V = float(p.get("portfolio_value", 0.0))
            ladder = planner.get("share_ladder") or cfg.get("planner",{}).get("share_ladder") or [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,102,153,204,255,306]
            show_frac = bool(planner.get("show_sma_fraction", True))

            rows = []
            for t in tickers:
                price = _last_price(t)
                rows.append({
                    "ticker": t,
                    "price": price,
                    "type": _tmap(t),
                    "sector": _smap(t),
                    "mom_12_1_spdji": float(mom.get(t, float('nan'))),
                    "sma20_slope_3d": str(slope.get(t, "Flat")),
                    "sma_stack_20_50_100": (str(sma_stack_frac.get(t, "0/3")) if show_frac else int(sma_stack_num.get(t, 0))),
                })
            base_df = pd.DataFrame(rows)

            # Percentage sheet
            pct_df = base_df.copy()
            for n in ladder:
                pct_df[f"{n}"] = (pct_df["price"] * float(n)) / V if V else float('nan')
            pct_df.to_excel(xw, sheet_name=f"Planner-% ({name})", index=False)

            # Dollar sheet
            usd_df = base_df.copy()
            for n in ladder:
                usd_df[f"{n}"] = (usd_df["price"] * float(n))
            usd_df.to_excel(xw, sheet_name=f"Planner-$ ({name})", index=False)

            # Caps sheet
            sleeves = (p.get("overrides") or {}).get("sleeves") or (cfg.get("sleeves") or {})
            stocks_w = float(sleeves.get("stocks_weight_pct", float('nan')))
            etfs_w   = float(sleeves.get("etfs_weight_pct", float('nan')))
            comd_w   = float(sleeves.get("commodities_weight_pct", float('nan')))
            # back-compat: normalize if missing
            if not (np.isfinite(stocks_w) and np.isfinite(etfs_w) and np.isfinite(comd_w)):
                etfs_w = etfs_w if np.isfinite(etfs_w) else float((cfg.get("etfs") or {}).get("bucket_weight_pct", 0.0))
                comd_w = comd_w if np.isfinite(comd_w) else float((cfg.get("commodities") or {}).get("bucket_weight_pct", 0.0))
                stocks_w = stocks_w if np.isfinite(stocks_w) else max(0.0, 1.0 - (etfs_w + comd_w))

            B_stocks = V * max(0.0, stocks_w)
            B_etfs   = V * max(0.0, etfs_w)
            B_comd   = V * max(0.0, comd_w)

            stock_cap_pct = float(((p.get("overrides") or {}).get("stocks") or {}).get("per_ticker_cap_pct", (cfg.get("stocks") or {}).get("per_ticker_cap_pct", 0.05)))
            etf_cap_pct   = float(((p.get("overrides") or {}).get("etfs") or {}).get("per_ticker_cap_pct", (cfg.get("etfs") or {}).get("per_ticker_cap_pct", 0.10)))
            comd_cap_pct  = float(((p.get("overrides") or {}).get("commodities") or {}).get("per_ticker_cap_pct", (cfg.get("commodities") or {}).get("per_ticker_cap_pct", 0.10)))

            caps_rows = [
                {"item":"Portfolio Value","percent":1.0,"dollars":V},
                {"item":"Stocks sleeve","percent":stocks_w,"dollars":B_stocks},
                {"item":"ETFs sleeve","percent":etfs_w,"dollars":B_etfs},
                {"item":"Commodities sleeve","percent":comd_w,"dollars":B_comd},
                {"item":"Per-stock cap","percent":stock_cap_pct,"dollars":V*stock_cap_pct},
                {"item":"Per-ETF cap","percent":etf_cap_pct,"dollars":V*etf_cap_pct},
                {"item":"Per-commodity cap","percent":comd_cap_pct,"dollars":V*comd_cap_pct},
            ]

            # Sector dollars inside stocks
            sector_targets = (((p.get("overrides") or {}).get("stocks") or {}).get("sector_targets")) or ((cfg.get("stocks") or {}).get("sector_targets") or {})
            # Normalize to 1.0 if totals drift
            st = { (k if isinstance(k,str) else str(k)) : float(v) for k,v in sector_targets.items() if v is not None }
            total_w = sum(v for v in st.values() if v >= 0)
            if total_w > 0:
                st_norm = {k: (v/total_w) for k,v in st.items()}
                for sec,w in st_norm.items():
                    caps_rows.append({"item": f"Stock sector: {sec}", "percent": w, "dollars": B_stocks * w})

            caps_df = pd.DataFrame(caps_rows)
            caps_df.to_excel(xw, sheet_name=f"Caps ({name})", index=False)

            # Audits
            pct_df.assign(portfolio_value=V).to_csv(AUDIT_DIR / f"planner_percent_{name}_{ts}.csv", index=False)
            usd_df.assign(portfolio_value=V).to_csv(AUDIT_DIR / f"planner_dollars_{name}_{ts}.csv", index=False)
            caps_df.to_csv(AUDIT_DIR / f"planner_caps_{name}_{ts}.csv", index=False)

            # Save effective cfg snapshot for this portfolio
            import json
            eff = {
                "name": name,
                "portfolio_value": V,
                "planner": planner,
                "effective_sleeves": {"stocks_weight_pct":stocks_w, "etfs_weight_pct":etfs_w, "commodities_weight_pct":comd_w},
                "per_ticker_caps": {"stock": stock_cap_pct, "etf": etf_cap_pct, "commodity": comd_cap_pct},
                "sector_targets": st,
            }
            (AUDIT_DIR / f"planner_effective_config_{name}_{ts}.json").write_text(json.dumps(eff, indent=2))

    return out_xlsx