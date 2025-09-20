from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from math import sqrt
from dataclasses import dataclass

# =========================
# Utilities
# =========================

TRADING_DAYS = 252
ULCER_WINDOW = 63

def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)

def daily_log_returns(adj_close: pd.Series) -> pd.Series:
    return np.log(adj_close).diff()

# Simple returns helper for S and P style momentum
def daily_simple_returns(adj_close: pd.Series) -> pd.Series:
    s = pd.to_numeric(adj_close, errors="coerce")
    return s.pct_change()

def rolling_annualized_vol(log_returns: pd.Series, window: int = 20) -> pd.Series:
    return log_returns.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)

def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ema = pd.Series(up, index=series.index).ewm(alpha=1 / length, adjust=False).mean()
    down_ema = pd.Series(down, index=series.index).ewm(alpha=1 / length, adjust=False).mean()
    rs = up_ema / (down_ema.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()

# =========================
# Helpers required by sector and internals code
# =========================

def _zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(index=x.index, data=0.0, dtype=float)
    return (x - mu) / sd

def _rolling_percentile_last(x: pd.Series, window: int = 252) -> float:
    arr = pd.to_numeric(x, errors="coerce").dropna()
    if len(arr) < max(5, window // 4):
        return float("nan")
    arr = arr.tail(window)
    last = arr.iloc[-1]
    rank = (arr <= last).mean()
    return float(rank)

# =========================
# 12 to 1 window helpers, exclude most recent month
# =========================

def _slice_12_1_window(series: pd.Series, lookback_days: int = TRADING_DAYS, exclude_recent: int = 22) -> pd.Series:
    s = series.dropna()
    if len(s) < (lookback_days + exclude_recent):
        return pd.Series(dtype=float, index=s.index[:0])
    end = len(s) - exclude_recent
    start = end - lookback_days
    return s.iloc[start:end]

def _slice_12_1_window_indices(n: int, lookback_days: int = TRADING_DAYS, exclude_recent: int = 22) -> tuple[int, int]:
    if n < (lookback_days + exclude_recent + 1):
        return (-1, -1)
    end = n - exclude_recent
    start = end - lookback_days
    if start < 0 or end <= start:
        return (-1, -1)
    return (start, end)

# =========================
# 12 to 1 drawdowns
# =========================

def ulcer_index_pct_from_series(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    rolling_max = s.cummax()
    dd = 100.0 * (s - rolling_max) / rolling_max
    return float(np.sqrt(np.mean(dd.values ** 2)))

def max_drawdown_pct_from_series(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    peak = s.expanding(min_periods=1).max()
    dd = (s / peak - 1.0) * 100.0
    return float(dd.min())

def compute_ulcer_12_1(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or "Adj Close" not in df:
            out[t] = np.nan
            continue
        path = _slice_12_1_window(df["Adj Close"])
        out[t] = ulcer_index_pct_from_series(path)
    return pd.Series(out, dtype=float)

def compute_max_dd_12_1(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or "Adj Close" not in df:
            out[t] = np.nan
            continue
        path = _slice_12_1_window(df["Adj Close"])
        out[t] = max_drawdown_pct_from_series(path)
    return pd.Series(out, dtype=float)

# =========================
# 12 to 1 bullish SMA stack days
# =========================

def compute_bull_stack_days_12_1_20_50_100(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or "Adj Close" not in df:
            out[t] = np.nan
            continue
        s = df["Adj Close"].astype(float)
        sma20 = s.rolling(20, min_periods=20).mean()
        sma50 = s.rolling(50, min_periods=50).mean()
        sma100 = s.rolling(100, min_periods=100).mean()
        idx = _slice_12_1_window(s).index
        stacked = (sma20 > sma50) & (sma50 > sma100)
        cnt = int(stacked.loc[idx].sum()) if len(idx) else np.nan
        out[t] = cnt
    return pd.Series(out, dtype=float)

# =========================
# Momentum and risk adjusted returns
# =========================

def downside_deviation_annualized_3y(adj_close: pd.Series) -> float:
    r = daily_log_returns(adj_close).dropna().tail(TRADING_DAYS * 3)
    if len(r) == 0:
        return np.nan
    neg = np.minimum(0.0, r.values)
    dd = np.sqrt(np.mean(neg ** 2))
    return float(dd * np.sqrt(TRADING_DAYS))

def ret_12_1_log(adj_close: pd.Series) -> float:
    s = adj_close.dropna()
    if len(s) < TRADING_DAYS + 22:
        return np.nan
    return float(np.log(s.iloc[-22]) - np.log(s.iloc[-TRADING_DAYS - 1]))

def ret_20d_log(adj_close: pd.Series) -> float:
    s = adj_close.dropna()
    if len(s) < 21:
        return np.nan
    return float(np.log(s.iloc[-1]) - np.log(s.iloc[-21]))

def compute_ret_12_1_vs_spy(prices: Dict[str, pd.DataFrame], tickers: List[str], benchmark: str = "SPY") -> pd.Series:
    out = {}
    spy = prices.get(benchmark)
    spy_rat = np.nan
    if spy is not None and not spy.empty:
        sn = ret_12_1_log(spy["Adj Close"])
        sd = downside_deviation_annualized_3y(spy["Adj Close"])
        spy_rat = sn / sd if np.isfinite(sn) and np.isfinite(sd) and sd != 0 else np.nan
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            out[t] = np.nan
            continue
        n = ret_12_1_log(df["Adj Close"])
        d = downside_deviation_annualized_3y(df["Adj Close"])
        rat = n / d if np.isfinite(n) and np.isfinite(d) and d != 0 else np.nan
        out[t] = rat - spy_rat if np.isfinite(rat) and np.isfinite(spy_rat) else np.nan
    return pd.Series(out)

# S and P DJI style 12 to 1 momentum, daily approximation
def compute_mom_12_1_spdji(prices: Dict[str, pd.DataFrame], tickers: List[str], lookback_days: int = TRADING_DAYS, exclude_recent: int = 22) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or "Adj Close" not in df:
            out[t] = np.nan
            continue
        s = pd.to_numeric(df["Adj Close"], errors="coerce").dropna()
        start, end = _slice_12_1_window_indices(len(s), lookback_days, exclude_recent)
        if start == -1:
            out[t] = np.nan
            continue
        window = s.iloc[start:end]
        if len(window) < 2:
            out[t] = np.nan
            continue
        mom_ret = float(window.iloc[-1] / window.iloc[0] - 1.0)
        r = window.pct_change().dropna()
        vol = float(r.std(ddof=0)) if len(r) else np.nan
        out[t] = mom_ret / vol if np.isfinite(mom_ret) and np.isfinite(vol) and vol != 0 else np.nan
    return pd.Series(out, dtype=float)

def compute_mom_12_1_spdji_vs_spy(prices: Dict[str, pd.DataFrame], tickers: List[str], benchmark: str = "SPY", lookback_days: int = TRADING_DAYS, exclude_recent: int = 22) -> pd.Series:
    spy_val = compute_mom_12_1_spdji(prices, [benchmark], lookback_days, exclude_recent).get(benchmark, np.nan)
    out = {}
    for t in tickers:
        v = compute_mom_12_1_spdji(prices, [t], lookback_days, exclude_recent).get(t, np.nan)
        out[t] = float(v - spy_val) if np.isfinite(v) and np.isfinite(spy_val) else np.nan
    return pd.Series(out, dtype=float)

# =========================
# Drawdowns
# =========================

def ulcer_index_pct(close: pd.Series, window: int = ULCER_WINDOW) -> float:
    s = close.dropna().tail(window)
    if s.empty:
        return np.nan
    rolling_max = s.cummax()
    dd = 100.0 * (s - rolling_max) / rolling_max
    ulcer = float(np.sqrt(np.mean(dd.values ** 2)))
    return ulcer

def max_drawdown_pct(close: pd.Series, window: int = ULCER_WINDOW) -> float:
    s = close.dropna().tail(window)
    if s.empty:
        return np.nan
    peak = s.expanding(min_periods=1).max()
    dd = (s / peak - 1.0) * 100.0
    return float(dd.min())

def compute_ulcer_63d(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    return pd.Series({t: ulcer_index_pct(prices.get(t)["Adj Close"]) if prices.get(t) is not None and not prices.get(t).empty else np.nan for t in tickers})

def compute_max_dd_63d(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    return pd.Series({t: max_drawdown_pct(prices.get(t)["Adj Close"]) if prices.get(t) is not None and not prices.get(t).empty else np.nan for t in tickers})

# =========================
# RSI, slope, vol trend
# =========================

def compute_rsi14(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            out[t] = np.nan
            continue
        out[t] = float(rsi_wilder(df["Adj Close"]).iloc[-1])
    return pd.Series(out)

def compute_rsi14_trend_1w(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or len(df) < 6:
            out[t] = "Flat"
            continue
        r = rsi_wilder(df["Adj Close"])
        delta = float(r.iloc[-1] - r.iloc[-5])
        out[t] = "Up" if delta > 1.0 else ("Down" if delta < -1.0 else "Flat")
    return pd.Series(out)

def compute_sma20_slope_3d(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or len(df) < 25:
            out[t] = "Flat"
            continue
        sma20 = df["Adj Close"].rolling(20).mean()
        slope = float(sma20.iloc[-1] - sma20.iloc[-4])
        out[t] = "Up" if slope > 0 else ("Down" if slope < 0 else "Flat")
    return pd.Series(out)

def compute_vol_trend_20d(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or len(df) < 45:
            out[t] = "Flat"
            continue
        logret = daily_log_returns(df["Adj Close"])
        vol = rolling_annualized_vol(logret, 20)
        slope = float(vol.iloc[-1] - vol.iloc[-5])
        out[t] = "Rising" if slope > 0 else ("Falling" if slope < 0 else "Flat")
    return pd.Series(out)

# =========================
# Sharpe and Sortino
# =========================

def compute_sharpe_12_1(prices: Dict[str, pd.DataFrame], tickers: List[str], rf_ann: float | None = None) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            out[t] = np.nan
            continue
        r = daily_log_returns(df["Adj Close"]).dropna()
        if len(r) < TRADING_DAYS + 22:
            out[t] = np.nan
            continue
        r = r.iloc[-(TRADING_DAYS + 22) : -22]
        rf = 0.0 if (rf_ann is None or not np.isfinite(rf_ann)) else float(rf_ann)
        mu = r.mean() * TRADING_DAYS
        sig = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
        ex = mu - rf
        out[t] = (ex / sig) if sig and np.isfinite(sig) else np.nan
    return pd.Series(out)

def compute_sortino_12_1(prices: Dict[str, pd.DataFrame], tickers: List[str], rf_ann: float | None = None) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            out[t] = np.nan
            continue
        r = daily_log_returns(df["Adj Close"]).dropna()
        if len(r) < TRADING_DAYS + 22:
            out[t] = np.nan
            continue
        r = r.iloc[-(TRADING_DAYS + 22) : -22]
        neg = np.minimum(0.0, r.values)
        dd = np.sqrt(np.mean(neg ** 2)) * np.sqrt(TRADING_DAYS)
        rf = 0.0 if (rf_ann is None or not np.isfinite(rf_ann)) else float(rf_ann)
        mu = r.mean() * TRADING_DAYS
        ex = mu - rf
        out[t] = ex / dd if dd and np.isfinite(dd) else np.nan
    return pd.Series(out)

# =========================
# Trend consistency weekly
# =========================

def compute_trend_consistency_weekly(prices: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DataFrame:
    out = {}
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty or len(df) < 100:
            out[t] = {"trend_consistency": np.nan, "trend_bucket": "Choppy"}
            continue
        wk = df["Adj Close"].resample("W-FRI").last().dropna()
        y = np.log(wk)
        x = np.arange(len(y))
        if len(y) < 26:
            out[t] = {"trend_consistency": np.nan, "trend_bucket": "Choppy"}
            continue
        x_mat = np.vstack([np.ones_like(x), x]).T
        beta = np.linalg.lstsq(x_mat, y.values, rcond=None)[0]
        y_hat = x_mat @ beta
        resid = y.values - y_hat
        ss_tot = ((y.values - y.values.mean()) ** 2).sum()
        ss_res = (resid ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        recent = wk.tail(26)
        rollmax = recent.cummax()
        dd = 100.0 * (recent - rollmax) / rollmax
        ulcer = float(np.sqrt(np.mean(dd.values ** 2))) if len(dd) > 0 else np.nan
        rng = float(100.0 * (recent.max() - recent.min()) / recent.max()) if len(recent) > 0 else np.nan
        smooth = float(np.clip(1.0 - (ulcer / max(rng, 1e-9)), 0.0, 1.0)) if np.isfinite(ulcer) and np.isfinite(rng) else 0.0
        up_weeks = (wk.diff() > 0).tail(26).mean()
        comp = 100.0 * (0.45 * r2 + 0.35 * smooth + 0.20 * float(up_weeks))
        bucket = "Clean" if comp >= 66 else ("Acceptable" if comp >= 40 else "Choppy")
        out[t] = {"trend_consistency": comp, "trend_bucket": bucket}
    return pd.DataFrame.from_dict(out, orient="index")

# =========================
# Beta, idio removed
# =========================

def compute_beta_and_idio_vol(prices: Dict[str, pd.DataFrame], tickers: List[str], benchmark: str = "SPY") -> pd.DataFrame:
    out = {}
    spy = prices.get(benchmark)
    if spy is None or spy.empty:
        return pd.DataFrame(index=tickers, data={"beta_252d": np.nan})
    mkt = daily_log_returns(spy["Adj Close"]).dropna().tail(TRADING_DAYS)
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            out[t] = {"beta_252d": np.nan}
            continue
        r = daily_log_returns(df["Adj Close"]).dropna().tail(TRADING_DAYS)
        j = r.index.intersection(mkt.index)
        if len(j) < 40:
            out[t] = {"beta_252d": np.nan}
            continue
        X = mkt.loc[j].values.reshape(-1, 1)
        y = r.loc[j].values.reshape(-1, 1)
        x_mean = X.mean()
        y_mean = y.mean()
        cov = ((X - x_mean) * (y - y_mean)).sum()
        var = ((X - x_mean) ** 2).sum()
        beta = float(cov / var) if var != 0 else np.nan
        out[t] = {"beta_252d": beta}
    return pd.DataFrame.from_dict(out, orient="index")

# =========================
# Internals support
# =========================

def _avg_pairwise_corr_window(returns_window: pd.DataFrame) -> float:
    X = returns_window.dropna(axis=1, how="any").values
    if X.shape[1] < 3:
        return np.nan
    C = np.corrcoef(X.T)
    iu = np.triu_indices_from(C, k=1)
    return float(np.nanmean(C[iu]))

def _rolling_avg_corr(series_df: pd.DataFrame, window: int = 21) -> pd.Series:
    out = []
    idx = series_df.index
    for i in range(len(idx)):
        if i + 1 < window:
            out.append(np.nan)
            continue
        w = series_df.iloc[i + 1 - window : i + 1]
        out.append(_avg_pairwise_corr_window(w))
    return pd.Series(out, index=idx, dtype=float)

def _build_returns_matrix(prices: Dict[str, pd.DataFrame], names: List[str]) -> pd.DataFrame:
    cols = {}
    for t in names:
        df = prices.get(t)
        if df is None or df.empty or "Adj Close" not in df:
            continue
        cols[t] = daily_log_returns(df["Adj Close"])
    if not cols:
        return pd.DataFrame()
    R = pd.concat(cols, axis=1).dropna(how="all")
    return R

# =========================
# Sector and market comparisons
# =========================

def compute_ticker_vs_sector_z_12_1(
    prices: Dict[str, pd.DataFrame],
    tickers: List[str],
    sectors: Dict[str, str],
    sp500_map: Dict[str, List[str]],
) -> pd.Series:
    t_scores = compute_mom_12_1_spdji(prices, tickers)

    sp_all = []
    for _, lst in (sp500_map or {}).items():
        if isinstance(lst, list):
            sp_all.extend([str(x).strip().upper() for x in lst])
    sp_all = list(dict.fromkeys([t for t in sp_all if t]))
    sp_scores = compute_mom_12_1_spdji(prices, sp_all) if sp_all else pd.Series(dtype=float)

    out = {}
    sector_dists: Dict[str, pd.Series] = {}
    for sec, lst in (sp500_map or {}).items():
        vals = pd.to_numeric(pd.Series({t: sp_scores.get(t, np.nan) for t in (lst or [])}), errors="coerce")
        v = winsorize(vals.dropna())
        sector_dists[sec] = v if not v.empty else pd.Series(dtype=float)

    for t in tickers:
        sec = sectors.get(t, "Unknown")
        if sec not in sector_dists or sector_dists[sec].empty:
            out[t] = np.nan
            continue
        val = pd.to_numeric(pd.Series([t_scores.get(t, np.nan)])).iloc[0]
        dist = sector_dists[sec]
        mu, sd = float(dist.mean()), float(dist.std(ddof=0))
        if not np.isfinite(val) or not np.isfinite(sd) or sd == 0.0:
            out[t] = np.nan
        else:
            out[t] = float((val - mu) / sd)
    return pd.Series(out, dtype=float)

def label_ticker_vs_sector(z: pd.Series, threshold: float = 1.0) -> pd.Series:
    def lab(v: float) -> str:
        if pd.isna(v):
            return ""
        if v >= threshold:
            return "Best in Breed"
        if v <= -threshold:
            return "Underperforming"
        return "In Line"
    return pd.to_numeric(z, errors="coerce").apply(lab)

def compute_sector_vs_market_z_12_1(
    prices: Dict[str, pd.DataFrame],
    tickers: List[str],
    sectors: Dict[str, str],
    sp500_map: Dict[str, List[str]],
    benchmark: str = "SPY",
) -> pd.Series:
    sp_all = []
    for _, lst in (sp500_map or {}).items():
        if isinstance(lst, list):
            sp_all.extend([str(x).strip().upper() for x in lst])
    sp_all = list(dict.fromkeys([t for t in sp_all if t]))
    if not sp_all:
        return pd.Series({t: np.nan for t in tickers}, dtype=float)

    sp_scores = compute_mom_12_1_spdji(prices, sp_all)

    spy_df = prices.get(benchmark)
    spy_val = np.nan
    if spy_df is not None and not spy_df.empty:
        num = ret_12_1_log(spy_df["Adj Close"])
        den = downside_deviation_annualized_3y(spy_df["Adj Close"])
        spy_val = num / den if np.isfinite(num) and np.isfinite(den) and den != 0 else np.nan

    sector_meds = {}
    for sec, lst in (sp500_map or {}).items():
        vals = pd.to_numeric(pd.Series({t: sp_scores.get(t, np.nan) for t in (lst or [])}), errors="coerce")
        v = vals.dropna()
        sector_meds[sec] = float(v.median()) if not v.empty else np.nan

    sec_vec = pd.Series({sec: (sector_meds.get(sec, np.nan) - spy_val) for sec in sp500_map.keys()}, dtype=float)
    sec_z = _zscore_series(sec_vec)

    out = {}
    for t in tickers:
        sec = sectors.get(t, "Unknown")
        out[t] = float(sec_z.get(sec, np.nan))
    return pd.Series(out, dtype=float)

def label_sector_vs_market(z: pd.Series, threshold: float = 0.5) -> pd.Series:
    def lab(v: float) -> str:
        if pd.isna(v):
            return ""
        if v >= threshold:
            return "Sector Outperform"
        if v <= -threshold:
            return "Sector Underperform"
        return "Sector Neutral"
    return pd.to_numeric(z, errors="coerce").apply(lab)

# =========================
# Internals snapshot
# =========================

def compute_internals_snapshot_per_ticker(
    prices: Dict[str, pd.DataFrame],
    tickers: List[str],
    sectors: Dict[str, str],
    sp500_map: Dict[str, List[str]],
    corr_window: int = 21,
    ref_window: int = TRADING_DAYS,
) -> pd.DataFrame:
    sp_all = []
    for _, lst in (sp500_map or {}).items():
        if isinstance(lst, list):
            sp_all.extend([str(x).strip().upper() for x in lst])
    sp_all = list(dict.fromkeys([t for t in sp_all if t]))

    out = pd.DataFrame(index=tickers)
    if sp_all:
        Rm = _build_returns_matrix(prices, sp_all)
        if not Rm.empty:
            disp_mkt = Rm.std(axis=1, ddof=0)
            mkt_disp_pct = _rolling_percentile_last(disp_mkt, window=ref_window)
            avgc_mkt_series = _rolling_avg_corr(Rm, window=corr_window)
            tail = avgc_mkt_series.tail(ref_window).dropna()
            if tail.empty:
                mkt_corr_z = np.nan
            else:
                sd = tail.std(ddof=0)
                mkt_corr_z = float((tail.iloc[-1] - tail.mean()) / (sd if sd != 0 else np.nan))
        else:
            mkt_disp_pct, mkt_corr_z = np.nan, np.nan
    else:
        mkt_disp_pct, mkt_corr_z = np.nan, np.nan

    sector_disp = {}
    sector_corrz = {}
    for sec, lst in (sp500_map or {}).items():
        names = [t for t in (lst or []) if isinstance(t, str)]
        Rs = _build_returns_matrix(prices, names)
        if Rs.empty:
            sector_disp[sec] = np.nan
            sector_corrz[sec] = np.nan
            continue
        d = Rs.std(axis=1, ddof=0)
        sector_disp[sec] = _rolling_percentile_last(d, window=ref_window)
        c_series = _rolling_avg_corr(Rs, window=corr_window)
        tail = c_series.tail(ref_window).dropna()
        if tail.empty:
            sector_corrz[sec] = np.nan
        else:
            sd = tail.std(ddof=0)
            sector_corrz[sec] = float((tail.iloc[-1] - tail.mean()) / (sd if sd != 0 else np.nan))

    out["sector_dispersion_12_1_pct"] = [float(sector_disp.get(sectors.get(t, "Unknown"), np.nan)) for t in tickers]
    out["sector_corr_21d_z"] = [float(sector_corrz.get(sectors.get(t, "Unknown"), np.nan)) for t in tickers]
    out["market_dispersion_12_1_pct"] = float(mkt_disp_pct)
    out["market_corr_21d_z"] = float(mkt_corr_z)
    return out

# =========================
# Misc small features that remain
# =========================

def robust_z(s: pd.Series, lower: float = 0.01, upper: float = 0.99, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.dropna().empty:
        return pd.Series(index=s.index, dtype=float).fillna(0.0)
    sw = winsorize(s, lower=lower, upper=upper)
    mu = float(sw.mean())
    sigma = float(sw.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0.0:
        return pd.Series(0.0, index=s.index, dtype=float)
    z = (sw - mu) / sigma
    return z.fillna(0.0).clip(-clip, clip)

def compute_sma_stack_hits_20_50_100(prices: Dict[str, pd.DataFrame], tickers: List[str], as_fraction: bool = False) -> pd.Series:
    out = {}
    for t in tickers:
        df = prices.get(t)
        cnt = 0
        if df is not None and not df.empty and "Adj Close" in df and len(df) >= 100:
            s = df["Adj Close"]
            last = float(s.iloc[-1])
            sma20 = float(s.rolling(20).mean().iloc[-1])
            sma50 = float(s.rolling(50).mean().iloc[-1])
            sma100 = float(s.rolling(100).mean().iloc[-1])
            cnt = int((last >= sma20) + (last >= sma50) + (last >= sma100))
        out[t] = (f"{cnt}/3" if as_fraction else cnt)
    return pd.Series(out)

def compute_rf_ann_from_sgov(prices: Dict[str, pd.DataFrame]) -> float:
    df = prices.get("SGOV")
    if df is None or df.empty or "Adj Close" not in df:
        return 0.0
    r = daily_log_returns(df["Adj Close"]).dropna()
    if len(r) < 63:
        return 0.0
    return float(r.tail(TRADING_DAYS).mean() * TRADING_DAYS)
