from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import re
import os
import yaml

# Optional runtime HTTP libs (loaded lazily where needed)
# We intentionally avoid top-level import failures if requests/yfinance aren't installed.
# Finnhub and yfinance lookups are wrapped in try/except and skipped gracefully.
# NOTE: We never hard-fail sector resolution due to a vendor outage.
try:
    from .paths import DATA_DIR  # project layout
except Exception:
    from pathlib import Path as _P
    DATA_DIR = _P(__file__).resolve().parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = DATA_DIR / "sector_cache.yml"
OVERRIDES_DEFAULT = DATA_DIR / "sector_overrides.yml"

# -----------------------------------------------------------------------------
# Canonical sector keys (must match sp500sectors.yml keys used elsewhere)
# -----------------------------------------------------------------------------
_CANON_SECTOR_KEYS = {
    "communication_services",
    "consumer_discretionary",
    "consumer_staples",
    "energy",
    "financials",
    "health_care",
    "industrials",
    "materials",
    "real_estate",
    "technology",
    "utilities",
}

# Map common variations (from APIs or files) to canonical keys.
_SECTOR_SYNONYMS = {
    # Communications
    "communication services": "communication_services",
    "communications": "communication_services",
    "telecommunication services": "communication_services",
    "telecommunications": "communication_services",
    "telecom": "communication_services",

    # Consumer
    "consumer discretionary": "consumer_discretionary",
    "consumer cyclical": "consumer_discretionary",
    "consumer staples": "consumer_staples",
    "consumer defensive": "consumer_staples",

    # Health
    "health care": "health_care",
    "healthcare": "health_care",

    # Technology
    "information technology": "technology",
    "technology": "technology",
    "tech": "technology",

    # Financials
    "financial": "financials",
    "financials": "financials",
    "banks": "financials",
    "banking": "financials",

    # Industrials
    "industrial": "industrials",
    "industrials": "industrials",

    # Materials
    "materials": "materials",
    "material": "materials",
    "basic materials": "materials",
    "chemicals": "materials",
    "metals & mining": "materials",
    "metals and mining": "materials",

    # Real estate
    "real estate": "real_estate",
    "reit": "real_estate",
    "reits": "real_estate",
    "equity real estate investment trusts (reits)": "real_estate",

    # Utilities
    "utilities": "utilities",
    "utility": "utilities",

    # Energy
    "energy": "energy",
    "oil & gas": "energy",
    "oil and gas": "energy",
}

# Some common Yahoo "industry" strings to sectors
_INDUSTRY_TO_SECTOR = {
    # Technology
    "semiconductors": "technology",
    "semiconductor equipment & materials": "technology",
    "semiconductor materials & equipment": "technology",
    "computer hardware": "technology",
    "consumer electronics": "technology",
    "electronic components": "technology",
    "software—application": "technology",
    "software—infrastructure": "technology",
    "information technology services": "technology",
    "communication equipment": "technology",
    "scientific & technical instruments": "technology",

    # Financials
    "banks—regional": "financials",
    "banks—diversified": "financials",
    "asset management": "financials",
    "capital markets": "financials",
    "credit services": "financials",
    "insurance—life": "financials",
    "insurance—property & casualty": "financials",
    "insurance—specialty": "financials",
    "financial data & stock exchanges": "financials",

    # Health Care
    "biotechnology": "health_care",
    "drug manufacturers—general": "health_care",
    "drug manufacturers—specialty & generic": "health_care",
    "healthcare plans": "health_care",
    "medical devices": "health_care",
    "medical instruments & supplies": "health_care",
    "diagnostics & research": "health_care",
    "health information services": "health_care",

    # Industrials
    "aerospace & defense": "industrials",
    "airlines": "industrials",
    "specialty industrial machinery": "industrials",
    "engineering & construction": "industrials",
    "waste management": "industrials",
    "electrical equipment & parts": "industrials",
    "conglomerates": "industrials",
    "farm & heavy construction machinery": "industrials",
    "marine shipping": "industrials",
    "trucking": "industrials",
    "railroads": "industrials",
    "tools & accessories": "industrials",

    # Consumer Discretionary
    "automobiles & auto parts": "consumer_discretionary",
    "auto manufacturers": "consumer_discretionary",
    "auto parts": "consumer_discretionary",
    "leisure": "consumer_discretionary",
    "resorts & casinos": "consumer_discretionary",
    "travel services": "consumer_discretionary",
    "apparel manufacturing": "consumer_discretionary",
    "footwear & accessories": "consumer_discretionary",
    "furnishings, fixtures & appliances": "consumer_discretionary",
    "specialty retail": "consumer_discretionary",
    "internet retail": "consumer_discretionary",
    "restaurants": "consumer_discretionary",

    # Consumer Staples
    "beverages—non-alcoholic": "consumer_staples",
    "beverages—brewers": "consumer_staples",
    "household & personal products": "consumer_staples",
    "packaged foods": "consumer_staples",
    "confectioners": "consumer_staples",
    "grocery stores": "consumer_staples",
    "tobacco": "consumer_staples",

    # Materials
    "chemicals": "materials",
    "agricultural inputs": "materials",
    "building materials": "materials",
    "lumber & wood production": "materials",
    "paper & paper products": "materials",
    "steel": "materials",
    "aluminum": "materials",
    "gold": "materials",
    "silver": "materials",
    "copper": "materials",
    "other industrial metals & mining": "materials",
    "coal": "materials",

    # Energy
    "oil & gas e&p": "energy",
    "oil & gas integrated": "energy",
    "oil & gas midstream": "energy",
    "oil & gas refining & marketing": "energy",
    "uranium": "energy",
    "thermal coal": "energy",
    "oil & gas equipment & services": "energy",

    # Real Estate
    "reit—specialty": "real_estate",
    "reit—industrial": "real_estate",
    "reit—office": "real_estate",
    "reit—retail": "real_estate",
    "reit—healthcare facilities": "real_estate",
    "reit—residential": "real_estate",
    "reit—mortgage": "real_estate",
    "real estate services": "real_estate",
    "real estate—development": "real_estate",
    "real estate—diversified": "real_estate",

    # Utilities
    "utilities—regulated electric": "utilities",
    "utilities—regulated gas": "utilities",
    "utilities—regulated water": "utilities",
    "utilities—independent power producers": "utilities",
    "utilities—renewable": "utilities",
}

# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def _normalize_sector_key(name: str) -> str:
    """Convert arbitrary sector labels to canonical SP500 keys.
    Returns 'Unknown' for empty/unmapped.
    """
    if not name:
        return "Unknown"
    s = str(name).strip()
    if not s:
        return "Unknown"
    s_lc = s.lower().replace("-", " ").replace("_", " ")
    s_lc = s_lc.replace("&", "and").replace("/", " ")
    s_lc = re.sub(r"\s+", " ", s_lc).strip()
    s_key = s_lc.replace(" ", "_")

    if s_key in _CANON_SECTOR_KEYS:
        return s_key
    if s_lc in _SECTOR_SYNONYMS:
        return _SECTOR_SYNONYMS[s_lc]
    if s_key in _SECTOR_SYNONYMS:
        return _SECTOR_SYNONYMS[s_key]
    if s_key.endswith("s") and s_key[:-1] in _SECTOR_SYNONYMS:
        return _SECTOR_SYNONYMS[s_key[:-1]]
    return "Unknown"


def _guess_sector_from_industry(industry: Optional[str]) -> str:
    if not industry:
        return "Unknown"
    s = str(industry).strip().lower()
    s = s.replace("&", "and").replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in _INDUSTRY_TO_SECTOR:
        return _INDUSTRY_TO_SECTOR[s]
    # broader contains checks
    if any(k in s for k in ["semiconductor", "electronic", "software", "information technology"]):
        return "technology"
    if any(k in s for k in ["bank", "insur", "capital market", "asset", "credit", "exchange"]):
        return "financials"
    if any(k in s for k in ["biotech", "drug", "medical", "health", "diagnostic"]):
        return "health_care"
    if any(k in s for k in ["aerospace", "defense", "industrial", "machinery", "airline", "shipping", "railroad", "trucking"]):
        return "industrials"
    if any(k in s for k in ["grocery", "tobacco", "beverage", "household", "packaged food"]):
        return "consumer_staples"
    if any(k in s for k in ["apparel", "retail", "restaurant", "leisure", "casino", "internet retail", "travel", "auto"]):
        return "consumer_discretionary"
    if any(k in s for k in ["reit", "real estate", "property"]):
        return "real_estate"
    if any(k in s for k in ["utilit", "power", "renewable"]):
        return "utilities"
    if any(k in s for k in ["chemical", "metal", "mining", "steel", "aluminum", "paper", "lumber", "building material"]):
        return "materials"
    if any(k in s for k in ["oil", "gas", "coal", "uranium", "refining"]):
        return "energy"
    return "Unknown"


def _guess_sector_from_name(name: Optional[str]) -> str:
    if not name:
        return "Unknown"
    s = str(name).lower()
    if any(k in s for k in ["bank", "banc", "financial", "capital", "insurance", "holdings"]):
        return "financials"
    if any(k in s for k in ["pharma", "biotech", "therapeutics", "health", "medical"]):
        return "health_care"
    if any(k in s for k in ["semiconductor", "semi", "micro", "electronics", "software", "systems", "networks", "telecom", "communications"]):
        return "technology"
    if any(k in s for k in ["oil", "gas", "petro", "energy", "midstream"]):
        return "energy"
    if any(k in s for k in ["mining", "steel", "aluminum", "gold", "copper", "chem", "materials"]):
        return "materials"
    if "reit" in s or "realty" in s or "properties" in s:
        return "real_estate"
    if any(k in s for k in ["utility", "utilities", "power", "electric"]):
        return "utilities"
    if any(k in s for k in ["airlines", "aerospace", "industrial", "machinery", "construction", "shipping", "rail"]):
        return "industrials"
    if any(k in s for k in ["grocery", "beverage", "tobacco", "staples"]):
        return "consumer_staples"
    if any(k in s for k in ["retail", "restaurant", "casino", "leisure", "travel", "apparel", "auto"]):
        return "consumer_discretionary"
    return "Unknown"

# -----------------------------------------------------------------------------
# YAML I/O
# -----------------------------------------------------------------------------

def _invert_mapping(sector_map: Dict[str, list]) -> Dict[str, str]:
    """Invert {sector_key: [tickers]} -> {TICKER: sector_key} with TICKER uppercased.

    NOTE: We keep sector_key canonical, but still pass it through _normalize_sector_key
    to be robust if a file uses synonyms.
    """
    t2s: Dict[str, str] = {}
    for sector_key, tickers in (sector_map or {}).items():
        if not isinstance(tickers, (list, tuple)):
            continue
        for t in tickers:
            tu = str(t or "").strip().upper()
            if tu:
                t2s[tu] = _normalize_sector_key(sector_key)
    return t2s


def load_sp500_sector_yml(path: Optional[Path]) -> Dict[str, str]:
    """Load a sp500sectors.yml-like file and return {TICKER: sector_key}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sectors_node = ((data.get("tickers") or {}).get("sectors")) or {}
    return _invert_mapping(sectors_node)


def load_overrides_yml(path: Optional[Path]) -> Dict[str, str]:
    """Load a user overrides YAML with shape {TICKER: sector_label_or_key}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: Dict[str, str] = {}
    for k, v in (data or {}).items():
        tu = str(k or "").strip().upper()
        if not tu:
            continue
        out[tu] = _normalize_sector_key(v)
    return out


def load_cache_yml(path: Optional[Path] = None) -> Dict[str, str]:
    """Load the sector cache YAML with shape {TICKER: sector_key}."""
    p = Path(path) if path else CACHE_PATH
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: Dict[str, str] = {}
    for k, v in (data or {}).items():
        tu = str(k or "").strip().upper()
        if not tu:
            continue
        out[tu] = _normalize_sector_key(v)
    return out


def write_cache_yml(mapping: Dict[str, str], path: Optional[Path] = None) -> None:
    """Write (or update) the sector cache YAML with shape {TICKER: sector_key}."""
    p = Path(path) if path else CACHE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cache_yml(p)
    existing.update({k.upper(): _normalize_sector_key(v) for k, v in mapping.items()})
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing, f, sort_keys=True)

# -----------------------------------------------------------------------------
# Vendor lookups (lazy imports)
# -----------------------------------------------------------------------------

def _finnhub_lookup(tickers: List[str], api_key: Optional[str]) -> Dict[str, str]:
    """Query Finnhub profile2 for sector/industry. Returns {TICKER: sector_label}."""
    if not api_key:
        return {}
    # Avoid hard dependency at import time
    try:
        import requests  # type: ignore
    except Exception:
        return {}
    out: Dict[str, str] = {}
    base = "https://finnhub.io/api/v1/stock/profile2"
    for t in tickers:
        try:
            resp = requests.get(base, params={"symbol": t, "token": api_key}, timeout=6)
            if resp.status_code != 200:
                continue
            data = resp.json() or {}
            sec = data.get("sector") or data.get("finnhubIndustry") or ""
            if sec:
                out[t] = str(sec)
        except Exception:
            # best-effort only
            pass
    return out


def _yfinance_lookup(tickers: List[str]) -> Dict[str, Dict[str, str]]:
    """Return {TICKER: {"sector": ..., "industry": ..., "name": ...}} (best effort)."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for t in tickers:
        sec_label = ""
        industry = ""
        long_name = ""
        short_name = ""
        # fast_info first
        try:
            fi = yf.Ticker(t).fast_info
            if hasattr(fi, "sector") and getattr(fi, "sector"):
                sec_label = str(getattr(fi, "sector"))
        except Exception:
            pass
        # fallback info
        try:
            info = yf.Ticker(t).info or {}
            sec_label = sec_label or str(info.get("sector") or "")
            industry = str(info.get("industry") or "")
            long_name = str(info.get("longName") or "")
            short_name = str(info.get("shortName") or "")
        except Exception:
            pass
        out[t] = {"sector": sec_label, "industry": industry, "name": long_name or short_name}
    return out

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def resolve_sector(
    tickers: List[str],
    sp500_yml: Optional[Path] = None,
    overrides_yml: Optional[Path] = None,
    use_cache: bool = True,
    finnhub_api_key: Optional[str] = None,
    finnhub_enabled: Optional[bool] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Resolve sector names for tickers (keys returned in UPPERCASE).

    Precedence:
      1) User overrides YAML (if provided or present at data/sector_overrides.yml)
      2) sp500sectors.yml mapping (if provided)
      3) Local cache (data/sector_cache.yml) if enabled
      4) Finnhub (profile2) if enabled / api key available
      5) yfinance fast_info.sector
      6) yfinance info.sector
      7) yfinance info.industry  -> mapped via _INDUSTRY_TO_SECTOR / heuristics
      8) Name heuristics (Ticker longName/shortName)
      9) 'Unknown'

    All outputs are normalized to canonical keys.
    """
    # Normalize input
    norm_tickers: List[str] = []
    for t in tickers or []:
        tu = str(t or "").strip().upper()
        if tu:
            norm_tickers.append(tu)

    # 1) Overrides
    if overrides_yml is None:
        overrides_yml = OVERRIDES_DEFAULT
    overrides = load_overrides_yml(overrides_yml)

    # 2) SP500 mapping
    mapping = load_sp500_sector_yml(sp500_yml)

    # 3) Cache
    cache = load_cache_yml(CACHE_PATH) if use_cache else {}

    sectors: Dict[str, str] = {}

    # First pass: overrides > sp500 > cache
    for tu in norm_tickers:
        if tu in overrides:
            sectors[tu] = _normalize_sector_key(overrides[tu])
        elif tu in mapping:
            sectors[tu] = _normalize_sector_key(mapping[tu])
        elif tu in cache:
            sectors[tu] = _normalize_sector_key(cache[tu])
        else:
            sectors[tu] = "Unknown"

    unresolved: List[str] = [tu for tu in norm_tickers if sectors.get(tu) == "Unknown"]
    if unresolved:
        # Discover finnhub settings if not provided
        if finnhub_api_key is None or finnhub_enabled is None:
            # Try environment first
            env_key = os.environ.get("FINNHUB_API_KEY") or os.environ.get("FINNHUB_TOKEN")
            if finnhub_api_key is None and env_key:
                finnhub_api_key = env_key
            # Optionally parse a config.yml if present
            if finnhub_enabled is None or (finnhub_api_key is None):
                try:
                    cfg_path = config_path or Path("config.yml")
                    if cfg_path.exists():
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f) or {}
                        fn = (cfg.get("finnhub") or {})
                        if finnhub_enabled is None:
                            finnhub_enabled = bool(fn.get("enabled", False))
                        if finnhub_api_key is None:
                            key = fn.get("api_key") or ""
                            finnhub_api_key = str(key).strip() or finnhub_api_key
                except Exception:
                    pass
        # default enabled behavior if key present
        if finnhub_enabled is None:
            finnhub_enabled = bool(finnhub_api_key)

        # 4) Finnhub first (more robust for sector mapping)
        if finnhub_enabled and finnhub_api_key:
            fh = _finnhub_lookup(unresolved, finnhub_api_key)
            newly_resolved: Dict[str, str] = {}
            for tu in unresolved:
                raw = fh.get(tu, "")
                cand = _normalize_sector_key(raw)
                if cand != "Unknown":
                    sectors[tu] = cand
                    newly_resolved[tu] = cand
            if newly_resolved and use_cache:
                try:
                    write_cache_yml(newly_resolved, CACHE_PATH)
                except Exception:
                    pass

        # 5/6/7/8) yfinance fallbacks for any still-Unknown
        still = [tu for tu in norm_tickers if sectors.get(tu) == "Unknown"]
        if still:
            yfm = _yfinance_lookup(still)
            newly_resolved: Dict[str, str] = {}
            for tu in still:
                meta = yfm.get(tu, {}) or {}
                sec_label = meta.get("sector") or ""
                industry = meta.get("industry") or ""
                name = meta.get("name") or ""

                cand = _normalize_sector_key(sec_label)
                if cand == "Unknown":
                    cand = _normalize_sector_key(_guess_sector_from_industry(industry))
                if cand == "Unknown":
                    cand = _normalize_sector_key(_guess_sector_from_name(name))

                sectors[tu] = cand
                if cand != "Unknown":
                    newly_resolved[tu] = cand

            if newly_resolved and use_cache:
                try:
                    write_cache_yml(newly_resolved, CACHE_PATH)
                except Exception:
                    pass

    return sectors
