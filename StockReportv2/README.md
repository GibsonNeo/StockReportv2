# PortfolioScreener (root-level version)

Produce a human‑readable Excel report from a YAML list of tickers—no packages, **all `.py` files live at repo root**.
The pipeline fetches ~5y daily prices with yfinance, computes momentum/risk/mean‑reversion/sector/volatility metrics,
writes a formatted Excel workbook, a CSV, **/audit** artifacts per run, and **/history** snapshots per ticker.

## Structure (root)
```
/PortfolioScreener
  ├── config.yml               # your universes + benchmark
  ├── cli.py                   # entry point
  ├── columns.py               # ALL column math lives here
  ├── fetch_prices.py          # yfinance downloader
  ├── sector_lookup.py         # sector resolver (yfinance fallback)
  ├── human_report_writer.py   # Excel + conditional formatting
  ├── writer.py                # append per‑ticker history rows
  ├── load_config.py           # parse config.yml
  ├── paths.py                 # ROOT + (data|history|audit|reports) dirs
  ├── build_sp500_sectors.py   # optional helper
  ├── requirements.txt         # dependencies
  └── README.md
  # Generated at runtime:
  /data      /history     /audit     /reports
```

## Install & Run
```bash
python -m pip install -r requirements.txt
python cli.py --config config.yml  # writes reports + audit + history
```

### Notes / Rules this repo honors
- **All column calculations live inside `columns.py`** (single consolidated module).
- **No packages or subfolders for code**—per your request, everything stays at repo root.
- **/history**: one CSV per ticker capturing daily snapshots of report columns.
- **/audit**: this run’s computed values are emitted as CSVs (`audit_<timestamp>.csv`) for transparency.
- **Excel conditional formatting** fixes applied:
  - *Ulcer Index (63d)* is **not** formatted as a percent (it’s in percentage points already).
  - *Max Drawdown (63d)*: **higher (less negative) is better** in coloring.
  - Categorical columns (e.g., `sma20_slope_3d`, `trend_bucket`, `rsi14_trend_1w`, `vol_trend_20d`, `sector_alignment`)
    use text‑based rules with colors.
- **`best_buys_today`** is now fully implemented with component subscores:
  `overextended_risk`, `entry_grade_total`, `overall_quality`, `turnaround_bonus`,
  `grade_overall_dispersion`, and `atrp_14` (ATR as % of price).

## Outputs
- `reports/tickers_report_<timestamp>.xlsx` – formatted Excel
- `reports/tickers_report_<timestamp>.csv` – same data as CSV
- `audit/audit_<timestamp>.csv`        – per‑ticker values of all numeric outputs
- `audit/format_quartiles_<timestamp>.csv` – thresholds used for Excel quartile CF
- `history/<TICKER>.csv` – per-ticker append‑only snapshots (date‑stamped)

## FAQ
**Q: What if yfinance misses a ticker or returns an empty frame?**  
A: The pipeline skips missing tickers, fills their row with NaN, and continues.

**Q: Can I change the Ulcer window from 63 days?**  
A: Yes—edit `ULCER_WINDOW` in `columns.py` or call `compute_ulcer(window=...)` variants.

**Q: Where do I set my benchmark (for relative/β/α calculations)?**  
A: In `config.yml` under `benchmark:` (defaults to `SPY` if omitted).

## (Optional) Use a standalone S&P 500 sector file
Generate once, then reuse:
```bash
python build_sp500_sectors.py --yml sp500sectors.yml
```
Run the report and point to it:
```bash
python cli.py --config config.yml --sp500-sectors sp500sectors.yml
```
This uses the YAML mapping for any tickers present; other tickers fall back to yfinance sectors or 'Unknown'.
