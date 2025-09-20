from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import requests

from load_config import load_config
from paths import DATA_DIR, AUDIT_DIR, REPORTS_DIR
from fetch_prices import fetch_5y_daily
from sector_lookup import resolve_sector
from build_sp500_sectors import fetch_sp500_table, build_mapping, to_yaml
from human_report_writer import write_formatted_excel

# === NEW: allocator support ===
# Separate workbook, separate YAML, no changes to main report columns.
# Keep original import idea, but add a robust fallback and silence Pylance on the fallback.
try:
    from allocator.sheets import build_allocator_workbook, load_allocator_config, build_allocator_planner_workbook
except Exception:
    from sheets import build_allocator_workbook, load_allocator_config, build_allocator_planner_workbook  # type: ignore[reportMissingImports]


def _norm_ticker_csv(x: object) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    if "." in s and not s.startswith("^"):
        s = s.replace(".", "-")
    return s

def _read_csv_tickers(csv_path: Path) -> list[str]:
    try:
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path, header=None)
        if df.empty:
            return []
        first_col = df.iloc[:, 0]
        out: list[str] = []
        seen: set[str] = set()
        for v in first_col:
            t = _norm_ticker_csv(v)
            if t and t not in seen:
                out.append(t)
                seen.add(t)
        return out
    except Exception:
        return []

def _get_finnhub_api_key(cfg: dict) -> str | None:
    # Try a few common nesting patterns
    paths = [
        ['api_keys','finnhub'],
        ['vendors','finnhub','api_key'],
        ['finnhub','api_key'],
        ['finnhub','token'],
    ]
    for p in paths:
        cur = cfg
        ok = True
        for k in p:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False; break
        if ok and isinstance(cur, (str, int)):
            s = str(cur).strip()
            if s:
                return s
    return None

def _finnhub_lookup_sector(symbols: list[str], api_key: str) -> dict[str,str]:
    out = {}
    base = "https://finnhub.io/api/v1/stock/profile2"
    for t in symbols:
        try:
            resp = requests.get(base, params={'symbol': t, 'token': api_key}, timeout=5)
            if resp.status_code == 200:
                data = resp.json() or {}
                # Try multiple fields
                sec = data.get('sector') or data.get('finnhubIndustry') or ''
                if sec:
                    out[t] = str(sec)
        except Exception:
            pass
    return out

def _label_sector_alignment(diff_series: pd.Series) -> pd.Series:
    # diff = ticker_value - sector_median; we want Diverging+/-, Converging
    s = pd.to_numeric(diff_series, errors='coerce')
    s_valid = s.dropna()
    if s_valid.empty:
        return pd.Series(['Converging']*len(diff_series), index=diff_series.index)
    q1 = s_valid.quantile(0.25); q3 = s_valid.quantile(0.75)
    def lab(v: float) -> str:
        if pd.isna(v):
            return 'Converging'
        if v >= q3:
            return 'Diverging+'
        if v <= q1:
            return 'Diverging-'
        return 'Converging'
    return s.apply(lab)

from columns import (
    compute_rf_ann_from_sgov,
    compute_ret_12_1_vs_spy,
    compute_ulcer_63d,
    compute_max_dd_63d,
    compute_rsi14,
    compute_rsi14_trend_1w,
    compute_sma20_slope_3d,
    compute_vol_trend_20d,
    compute_sharpe_12_1,
    compute_sortino_12_1,
    compute_trend_consistency_weekly,
    compute_beta_and_idio_vol,
    compute_ulcer_12_1,
    compute_max_dd_12_1,
    compute_bull_stack_days_12_1_20_50_100,
    # NEW: sector-aware relatives & internals
    compute_ticker_vs_sector_z_12_1,
    label_ticker_vs_sector,
    compute_sector_vs_market_z_12_1,
    label_sector_vs_market,
    compute_internals_snapshot_per_ticker,
    # ADDED: S&P/DJI-style daily 12-1 momentum
    compute_mom_12_1_spdji,
    compute_mom_12_1_spdji_vs_spy,
)

DEFAULT_CONFIG = Path("config.yml")          # hard-wired default
DEFAULT_SP500_YML = Path("sp500sectors.yml") # hard-wired default


def _ensure_sp500_yaml_fresh(yml_path: Path, max_age_days: int = 7) -> None:
    """Rebuild sp500sectors.yml if missing or older than max_age_days."""
    try:
        need = True
        if yml_path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(yml_path.stat().st_mtime)).days
            need = age_days > max_age_days
        if need:
            df = fetch_sp500_table()
            mapping = build_mapping(df)
            payload = to_yaml(mapping)
            yml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yml_path, 'w', encoding='utf-8') as f:
                f.write(payload)
            print(f"[ok] Refreshed {yml_path.name} with {sum(len(v) for v in mapping.values())} tickers across {len(mapping)} sectors.")
    except Exception as e:
        print(f"[warn] Could not refresh SP500 sectors YAML: {e}")

def _is_disabled_benchmark(val: str) -> bool:
    v = (val or "").strip().upper()
    return v in {"", "NONE", "N/A", "NULL"}

# === NEW: helpers to collect tickers from root config.yml ====================

def _flatten_grouped_tickers(obj) -> list[str]:
    """
    Recursively flatten any nested dict/list structure into a unique, ordered
    list of uppercased tickers.
    Accepts shapes like:
      {"GroupA": ["AAPL","MSFT"], "GroupB": {"bench": ["SPY"]}}
      ["AAPL","MSFT"]
    """
    out: list[str] = []
    def _walk(x):
        if x is None:
            return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)
        else:
            s = str(x).strip().upper()
            if s:
                out.append(s)
    _walk(obj)
    # de-dup preserving first-seen order
    return list(dict.fromkeys(out))

def _collect_root_universes(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """
    From the root config.yml, collect three universes:
      - stock_tickers (grouped)
      - etf_tickers (grouped)
      - commodities_tickers (grouped)
    """
    stocks_raw = (cfg.get("stock_tickers") or {})
    etfs_raw = (cfg.get("etf_tickers") or {})
    coms_raw = (cfg.get("commodities_tickers") or {})
    stock_list = _flatten_grouped_tickers(stocks_raw)
    etf_list = _flatten_grouped_tickers(etfs_raw)
    com_list = _flatten_grouped_tickers(coms_raw)
    return stock_list, etf_list, com_list

def main():
    # Flags are optional; defaults are hard-wired to files in CWD.
    ap = argparse.ArgumentParser(description="PortfolioScreener root-level runner", add_help=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config.yml")
    ap.add_argument("--sp500-sectors", type=Path, default=DEFAULT_SP500_YML, help="Path to sp500sectors.yml")
    # NEW: allocator YAML path (separate file)
    ap.add_argument("--allocator-config", type=Path, default=Path("allocator/allocator_config.yml"), help="Path to allocator/allocator_config.yml")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"config.yml not found at {args.config.resolve()}")

    cfg = load_config(args.config)

    # Load the RAW root YAML too for allocator universes (grouped trees).
    # The normalized cfg returned by load_config() drops those trees.
    try:
        with open(args.config, "r", encoding="utf-8") as _f:
            raw_cfg_for_allocator = yaml.safe_load(_f) or {}
    except Exception:
        raw_cfg_for_allocator = {}

    # ADDED: momentum knob (defaults keep original behavior)
    mom_cfg = (cfg.get('momentum') or {})
    momentum_metric = str(mom_cfg.get('metric', 'sortino_like')).strip().lower()

    grade_cfg = (cfg.get('grade') or {})
    grade_version = str(grade_cfg.get('version', 'v1')).strip().lower()
    grade_weights = grade_cfg.get('weights', {}) if isinstance(grade_cfg.get('weights', {}), dict) else {}

    tickers = list(dict.fromkeys(cfg.get('tickers', [])))
    types_map = cfg.get('types', {})
    # Optional CSV override for stock tickers only
    csv_path = Path("ticker.csv")
    csv_tickers = _read_csv_tickers(csv_path)
    if csv_tickers:
        print(f"[info] Using {csv_path.name} for STOCK universe with {len(csv_tickers)} symbols.")
        tickers = csv_tickers[:]
        types_map = {t: "Stock" for t in tickers}
    else:
        print("[info] ticker.csv not found or empty, using config.yml tickers.")

    # === NEW: collect allocator universes from ROOT YAML (grouped sections)
    alloc_stock_universe, alloc_etf_universe, alloc_com_universe = _collect_root_universes(raw_cfg_for_allocator)
    if csv_tickers:
        alloc_stock_universe = tickers[:]

    # augment types_map without overwriting existing labels
    for t in alloc_stock_universe:
        types_map.setdefault(t, "Stock")
    for t in alloc_etf_universe:
        types_map.setdefault(t, "ETF")
    for t in alloc_com_universe:
        types_map.setdefault(t, "Commodity")
    # also tag SGOV as ETF if not already present
    types_map.setdefault("SGOV", "ETF")

    # Handle benchmark: skip if NONE/disabled
    raw_bench = str(cfg.get("benchmark", "SPY"))
    use_benchmark = not _is_disabled_benchmark(raw_bench)
    benchmark = raw_bench.upper() if use_benchmark else None

    if len(tickers) == 0:
        print("[warn] No tickers configured in config.yml. Add groups or set universe.include and re-run. Exiting.")
        return

    if use_benchmark and benchmark not in tickers:
        tickers_all = tickers + [benchmark]
    else:
        tickers_all = tickers

    # Add SGOV for risk-free estimation
    if 'SGOV' not in tickers_all:
        tickers_all = tickers_all + ['SGOV']

    # Load/refresh SP500 sectors for sector/market relatives and internals
    _ensure_sp500_yaml_fresh(args.sp500_sectors)
    sp500_map_path = args.sp500_sectors
    sp500_map = {}
    try:
        with open(args.sp500_sectors, 'r', encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
            sp500_map = ((y.get('tickers') or {}).get('sectors')) or (y.get('sectors') or {})
    except Exception:
        sp500_map = {}

    sp500_all = []
    for _, lst in (sp500_map or {}).items():
        if isinstance(lst, list):
            sp500_all.extend([str(x).strip().upper() for x in lst])
    sp500_all = list(dict.fromkeys([t for t in sp500_all if t]))
    # Avoid polluting universe, add for fetching only
    extra_for_medians = [t for t in sp500_all if t not in tickers_all]
    if extra_for_medians:
        tickers_all = tickers_all + extra_for_medians

    print(f"[info] universe size: {len(tickers)} (benchmark={'none' if not use_benchmark else benchmark})")
    prices = fetch_5y_daily(tickers_all)

    # Output paths and timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = REPORTS_DIR / f"tickers_report_{ts}.xlsx"
    out_csv = REPORTS_DIR / f"tickers_report_{ts}.csv"

    # Risk-free from SGOV
    rf_ann = compute_rf_ann_from_sgov(prices)

    # Compute columns
    if use_benchmark:
        rel_vs_spy = compute_ret_12_1_vs_spy(prices, tickers, benchmark=benchmark)
    else:
        rel_vs_spy = pd.Series(index=tickers, dtype=float)

    # S and P DJI style momentum
    mom_spdji = compute_mom_12_1_spdji(prices, tickers)
    rel_spdji_vs_spy = compute_mom_12_1_spdji_vs_spy(prices, tickers, benchmark=benchmark) if use_benchmark else pd.Series(index=tickers, dtype=float)

    ulcer63 = compute_ulcer_63d(prices, tickers)
    maxdd63 = compute_max_dd_63d(prices, tickers)
    ulcer12 = compute_ulcer_12_1(prices, tickers)
    maxdd12 = compute_max_dd_12_1(prices, tickers)
    bullstack_12_1 = compute_bull_stack_days_12_1_20_50_100(prices, tickers)
    rsi14_s = compute_rsi14(prices, tickers)
    rsi_tr = compute_rsi14_trend_1w(prices, tickers)
    sma20_slope = compute_sma20_slope_3d(prices, tickers)
    vol_trend = compute_vol_trend_20d(prices, tickers)

    sharpe12 = compute_sharpe_12_1(prices, tickers, rf_ann=rf_ann)
    sortino12 = compute_sortino_12_1(prices, tickers, rf_ann=rf_ann)
    tcs_df = compute_trend_consistency_weekly(prices, tickers)
    beta_df = compute_beta_and_idio_vol(prices, tickers, benchmark=benchmark if use_benchmark else 'SPY')
    # alias new column to legacy name so downstream code stays stable
    if 'beta_252d' in beta_df.columns and 'beta_252d_vs_spy' not in beta_df.columns:
        beta_df = beta_df.rename(columns={'beta_252d': 'beta_252d_vs_spy'})

    # Sector alignment vs SP500 sector medians
    sector_medians = {}
    for sec, lst in (sp500_map or {}).items():
        vals = pd.to_numeric(pd.Series({t: mom_spdji.get(t, np.nan) for t in (lst or [])}), errors='coerce')
        v = vals.dropna()
        sector_medians[sec] = float(v.median()) if not v.empty else np.nan

    # Sector mapping
    sectors = resolve_sector(tickers, sp500_yml=sp500_map_path)

    # Fill unknown sectors via Finnhub if configured
    finnhub_key = _get_finnhub_api_key(cfg)
    unknowns = [t for t in tickers if sectors.get(t, 'Unknown') == 'Unknown']
    if finnhub_key and unknowns:
        fill = _finnhub_lookup_sector(unknowns, finnhub_key)
        for t, sec in fill.items():
            if sec and sectors.get(t, 'Unknown') == 'Unknown':
                sectors[t] = sec

    # New relative columns
    z_ticker_vs_sector = compute_ticker_vs_sector_z_12_1(prices, tickers, sectors, sp500_map)
    ticker_vs_sector = label_ticker_vs_sector(z_ticker_vs_sector, threshold=1.0)

    z_sector_vs_mkt = compute_sector_vs_market_z_12_1(prices, tickers, sectors, sp500_map, benchmark=benchmark if use_benchmark else 'SPY')
    sector_vs_market = label_sector_vs_market(z_sector_vs_mkt, threshold=0.5)

    internals = compute_internals_snapshot_per_ticker(prices, tickers, sectors, sp500_map)

    sector_alignment_score = z_ticker_vs_sector.astype(float)

    # Assemble DataFrame
    rows = []
    for t in tickers:
        rows.append({
            "ticker": t,
            "type": str(types_map.get(t, "Unknown")),
            "sector": sectors.get(t, "Unknown"),
            "ret_12_1_vs_spy": float(rel_vs_spy.get(t, np.nan)) if pd.notna(rel_vs_spy.get(t, np.nan)) else np.nan,
            "mom_12_1_spdji": float(mom_spdji.get(t, np.nan)) if pd.notna(mom_spdji.get(t, np.nan)) else np.nan,
            "mom_12_1_spdji_vs_spy": float(rel_spdji_vs_spy.get(t, np.nan)) if pd.notna(rel_spdji_vs_spy.get(t, np.nan)) else np.nan,
            "ulcer_12_1": float(ulcer12.get(t, np.nan)) if pd.notna(ulcer12.get(t, np.nan)) else np.nan,
            "max_dd_12_1": float(maxdd12.get(t, np.nan)) if pd.notna(maxdd12.get(t, np.nan)) else np.nan,
            "bull_stack_days_12_1_20_50_100": int(bullstack_12_1.get(t)) if pd.notna(bullstack_12_1.get(t, np.nan)) else np.nan,
            "trend_consistency": float(tcs_df.loc[t, 'trend_consistency']) if t in tcs_df.index else np.nan,
            "trend_bucket": str(tcs_df.loc[t, 'trend_bucket']) if t in tcs_df.index else "Choppy",
            "sharpe_12_1": float(sharpe12.get(t, np.nan)) if pd.notna(sharpe12.get(t, np.nan)) else np.nan,
            "sortino_12_1": float(sortino12.get(t, np.nan)) if pd.notna(sortino12.get(t, np.nan)) else np.nan,
            "rsi14": float(rsi14_s.get(t, np.nan)) if pd.notna(rsi14_s.get(t, np.nan)) else np.nan,
            "rsi14_trend_1w": str(rsi_tr.get(t, 'Flat')),
            "sma20_slope_3d": str(sma20_slope.get(t, 'Flat')),
            "vol_trend_20d": str(vol_trend.get(t, 'Flat')),
            "ulcer_63d": float(ulcer63.get(t, np.nan)) if pd.notna(ulcer63.get(t, np.nan)) else np.nan,
            "max_dd_63d": float(maxdd63.get(t, np.nan)) if pd.notna(maxdd63.get(t, np.nan)) else np.nan,
            "beta_252d_vs_spy": float(beta_df.loc[t, 'beta_252d_vs_spy']) if t in beta_df.index else np.nan,
        })

    df = pd.DataFrame(rows)

    # New sector-aware columns
    df['ticker_vs_sector_z_12_1'] = df['ticker'].map(z_ticker_vs_sector)
    df['ticker_vs_sector'] = df['ticker'].map(ticker_vs_sector).fillna('')
    df['sector_vs_market_z_12_1'] = df['ticker'].map(z_sector_vs_mkt)
    df['sector_vs_market'] = df['ticker'].map(sector_vs_market).fillna('')

    # Internals snapshot duplicated per-row
    for col in ['sector_dispersion_12_1_pct','sector_corr_21d_z','market_dispersion_12_1_pct','market_corr_21d_z']:
        df[col] = internals[col].values if col in internals.columns else np.nan
    # Exclude ETFs or Commodities from sector aware columns
    excl = df['type'].str.upper().isin(['ETF','COMMODITY','COMMODITIES'])
    for col in ['ticker_vs_sector_z_12_1','ticker_vs_sector','sector_vs_market_z_12_1','sector_vs_market',
                'sector_dispersion_12_1_pct','sector_corr_21d_z','market_dispersion_12_1_pct','market_corr_21d_z']:
        if col in df.columns:
            df.loc[excl, col] = np.nan if df[col].dtype.kind in 'fci' else ''

    df['sector_alignment_score'] = df['ticker_vs_sector_z_12_1']

    if df.empty:
        print("[warn] Nothing to compute: your ticker list is empty. Add symbols and re-run.")
        return

    thresholds = write_formatted_excel(df, out_xlsx)
    df.to_csv(out_csv, index=False)

    if thresholds:
        thr_rows = [{'column': k, 'q1': v[0], 'q3': v[1]} for k, v in thresholds.items()]
        pd.DataFrame(thr_rows).to_csv(AUDIT_DIR / f"format_quartiles_{ts}.csv", index=False)

    # ----------------- Allocator (separate workbook and YAML) -----------------
    try:
        alloc_cfg_path = args.allocator_config
        if not alloc_cfg_path.exists():
            alt = Path("allocator_config.yml")
            if alt.exists():
                alloc_cfg_path = alt

        alloc_cfg = load_allocator_config(alloc_cfg_path)
        if alloc_cfg.get("enabled", True):
            etf_section = alloc_cfg.setdefault("etfs", {})
            etf_section["tickers"] = alloc_etf_universe
            com_section = alloc_cfg.setdefault("commodities", {})
            com_section["tickers"] = alloc_com_universe

            portfolios = (alloc_cfg.get("portfolios") or [])
            planner_enabled_any = any(bool((p.get("planner") or {}).get("enabled", False)) for p in portfolios)
            if planner_enabled_any:
                plan_out = build_allocator_planner_workbook(
                    out_xlsx=None,
                    prices=prices,
                    base_universe=tickers,
                    types_map=types_map,
                    sectors_map=sectors,
                    alloc_cfg=alloc_cfg,
                )
                print(f"[ok] Planner workbook: {plan_out.name}")

            planner_only_all = (len(portfolios) > 0) and all(bool((p.get("planner") or {}).get("planner_only", False)) for p in portfolios)

            if not planner_only_all:
                alloc_out = build_allocator_workbook(
                    prices=prices,
                    base_universe=alloc_stock_universe,
                    types_map=types_map,
                    sectors_map=sectors,
                    alloc_cfg=alloc_cfg,
                )
                print(f"[ok] Allocator workbook: {alloc_out.name}")
    except Exception as e:
        print(f"[warn] Allocator build failed: {e}")

    print(f"[ok] Wrote Excel: {out_xlsx.name} | CSV: {out_csv.name}")
    print(f"[ok] Audit CSVs written to {AUDIT_DIR}")

if __name__ == "__main__":
    main()
