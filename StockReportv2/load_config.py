from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

# -------------------------------
# Helpers
# -------------------------------

def _norm_ticker(x: Any) -> Optional[str]:
    """Uppercase, strip, and map '.' -> '-' for yfinance compatibility."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    if "." in s and not s.startswith("^"):
        s = s.replace(".", "-")
    return s

def _dedup(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for t in seq:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out

def _get_path(node: Any, path: str) -> Any:
    """Fetch nested value by a dot path (e.g., 'etf_tickers.us_core.bench')."""
    cur = node
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def _flatten_container(obj: Any, sink: List[str]) -> None:
    """Recursively walk lists/tuples/sets/dicts and collect leaf strings as tickers."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for _, v in obj.items():
            _flatten_container(v, sink)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _flatten_container(v, sink)
    else:
        t = _norm_ticker(obj)
        if t:
            sink.append(t)

def _infer_benchmark(cfg: Dict[str, Any]) -> str:
    """If config.benchmark set, use it; else try etf_tickers.us_core.bench[0]; else 'SPY'."""
    raw = cfg.get("benchmark", None)
    if isinstance(raw, str):
        raw = raw.strip()
    if raw:
        t = _norm_ticker(raw)
        return t or "SPY"
    try:
        us_core = (cfg.get("etf_tickers") or {}).get("us_core") or {}
        bench_list = us_core.get("bench") or []
        if bench_list:
            t = _norm_ticker(bench_list[0])
            if t:
                return t
    except Exception:
        pass
    return "SPY"

def _collect_from_paths(cfg: Dict[str, Any], include_paths: List[str]) -> Tuple[List[str], Dict[str,str]]:
    """Collect tickers only from the specified dot-paths; return (tickers, types map)."""
    tickers: List[str] = []
    types: Dict[str,str] = {}
    for path in include_paths:
        node = _get_path(cfg, path)
        if node is None:
            continue
        buf: List[str] = []
        _flatten_container(node, buf)
        # determine type by top-level section
        top = path.split('.')[0]
        if top == "stock_tickers":
            label = "Stock"
        elif top == "commodities_tickers":
            label = "Commodity"
        elif top == "etf_tickers":
            label = "ETF"
        else:
            label = "Unknown"
        for t in buf:
            tickers.append(t)
            types.setdefault(t, label)
    return _dedup(tickers), types

def _collect_all_grouped(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str,str]]:
    """Collect tickers from all grouped sections (backward-compatible default)."""
    tickers: List[str] = []
    types: Dict[str,str] = {}
    mapping = [
        ("stock_tickers", "Stock"),
        ("commodities_tickers", "Commodity"),
        ("etf_tickers", "ETF"),
    ]
    for key, label in mapping:
        node = cfg.get(key)
        if node is None:
            continue
        buf: List[str] = []
        _flatten_container(node, buf)
        for t in buf:
            tickers.append(t)
            types.setdefault(t, label)
    return _dedup(tickers), types

# -------------------------------
# Public API
# -------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML and return:
      {
        "tickers": [ ...deduped, normalized... ],
        "benchmark": "SPY" | <symbol>,
        "types": {TICKER: "Stock|ETF|Commodity|Unknown"}
      }

    By default (backward compatible), every list under these grouped sections is included:
      stock_tickers, commodities_tickers, etf_tickers

    If you want to limit the universe, add:
      universe:
        include:
          - stock_tickers.added
          - etf_tickers.us_core.bench
    Only those dot-paths are used to build the report universe.

    IMPORTANT: If the config has a legacy top-level `tickers:` dict with `sectors:` inside
    (from a previous S&P-500 merge), we DO NOT include those sector members in the universe.
    A plain list under `tickers:` is still honored.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Universe selection
    include_paths = []
    try:
        inc = ((cfg.get("universe") or {}).get("include")) or []
        if isinstance(inc, list) and inc:
            include_paths = [str(x) for x in inc]
    except Exception:
        include_paths = []

    if include_paths:
        tickers, types = _collect_from_paths(cfg, include_paths)
    else:
        tickers, types = _collect_all_grouped(cfg)

    # Backward compatibility: also honor a flat 'tickers' list (but not a 'sectors' mapping).
    if "tickers" in cfg:
        node = cfg["tickers"]
        if isinstance(node, (list, tuple, set)):
            buf: List[str] = []
            _flatten_container(node, buf)
            for t in buf:
                tickers.append(t)
                types.setdefault(t, "Unknown")
        elif isinstance(node, dict):
            # If it's a dict *without* 'sectors', allow flatten (legacy custom shape).
            if "sectors" not in node:
                buf: List[str] = []
                _flatten_container(node, buf)
                for t in buf:
                    tickers.append(t)
                    types.setdefault(t, "Unknown")
            # else: ignore, because it's an S&P500 sectors mapping, not a universe list.

    tickers = _dedup(tickers)
    benchmark = _infer_benchmark(cfg)

    
    return {
        "tickers": tickers,
        "benchmark": benchmark,
        "types": types,
        # ADDED: momentum knob (safe defaults; ignored by code unless used)
        "momentum": {
            "metric": str((cfg.get("momentum") or {}).get("metric", "sortino_like")).strip().lower(),
            "window_mode": str((cfg.get("momentum") or {}).get("window_mode", "daily")).strip().lower(),
            "days_per_year": int((cfg.get("momentum") or {}).get("days_per_year", 252)),
            "days_per_month": int((cfg.get("momentum") or {}).get("days_per_month", 22)),
        },
        # NEW: grade knob (version + weights); defaults keep original behavior
        
    }
