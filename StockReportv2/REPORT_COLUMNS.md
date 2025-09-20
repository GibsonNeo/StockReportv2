# Portfolio Report — How to Read & Use It

This guide explains what each column means, how to interpret the numbers, and how you might use them to make investing decisions. It matches the Excel report produced in `reports/tickers_report_<timestamp>.xlsx`.

> **Color key**
> - **Green** = favorable (for that metric’s “good” direction).
> - **Red** = unfavorable.
> - **Yellow/Blue** = neutral categories for certain text signals.
> - Numeric colors are **quartile-based** (top/bottom 25% within the sheet).

---

## Quick Start (5‑minute read)

1) **Sort by `best_buys_today`** to see actionable ideas first.  
   Check the supporting signals: trend quality, entries (RSI trend / SMA slope), and risk (Ulcer, MaxDD, ATRP).

2) **Confirm quality with `grade_12_1`.**  
   High grade means the *12→1‑month* trend is healthy and risk‑adjusted.

3) **Scan risk**: avoid names with very high `ulcer_63d` or very negative `max_dd_63d`.  
   Large `atrp_14` = bigger daily swings; size positions accordingly.

4) **Gut‑check with the benchmark**: `ret_12_1_vs_spy` > 0 means it outperformed the benchmark on the core signal.

5) **Pick entries**: look for **Up** in `sma20_slope_3d` / `rsi14_trend_1w` (for momentum) or small |`mr_z_12_1`| with high `mr_prob_10d_12_1` (for mean‑reversion entries).

> **Not investment advice.** Use alongside your process, risk rules, and fundamental or macro work.

---

## Column‑by‑Column

### ticker
Your configured symbol (uppercased; `.` → `-` for Yahoo compatibility, e.g., `BRK.B` → `BRK-B`).

### sector
Sector label used for summary / comparisons (from `sp500sectors.yml` when available; otherwise from Yahoo; else `Unknown`).

### grade_12_1 (0–100) — *Quality of the medium‑term trend*
A composite that rewards strong, risk‑adjusted **12→1‑month** momentum and stable behavior, while penalizing drawdown stress.
- **Higher is better.**  
- **Typical use:** filter for sturdy leaders; de‑prioritize names with chronic drawdown/whipsaw.
- **Rules of thumb:**  
  - **80–100**: elite trend quality  
  - **60–80**: solid  
  - **40–60**: mixed/average  
  - **<40**: weak or unstable

### best_buys_today (0–100) — *Actionability today*
A weighted blend focused on **entry timing + quality**. It favors:
- Reasonable **reversion** (small |`mr_z_12_1`|, high `mr_prob_10d_12_1`)
- Positive **short‑term direction** (`rsi14_trend_1w`, `sma20_slope_3d`) inside a healthy trend (`trend_consistency`, `grade_12_1`)
- Moderate **volatility** (`atrp_14`) and favorable **dispersion** (easier stock‑picking)
- **Higher is better.** Top quartile is a good hunting ground.

### trend_consistency (0–100) — *Cleanliness of the path*
How “straight” and persistent the weekly trend has been (uses regression fit, smoothness, fraction of up weeks).
- **Higher is better.**
- **Use:** favor smooth leaders; avoid choppy names for momentum plays.

### trend_bucket — *Label for quick scan*
Derived from `trend_consistency`:
- **Clean** (≥66) — smooth leadership
- **Acceptable** (≥40) — usable
- **Choppy** (<40) — noisy; treat with caution

### rsi14 (0–100) — *Momentum/overbought*
Wilder RSI(14) on closes.
- **>70** often stretched; **<30** often washed‑out  
- Use direction (`rsi14_trend_1w`) and context (trend/mean‑reversion) rather than absolute levels alone.

### rsi14_trend_1w — *RSI direction*
**Up / Flat / Down** over ~1 week.  
- **Use:** align entries with **Up** in momentum setups; **Down** can favor reversion shorts or patience on longs.

### sma20_slope_3d — *Short‑term price slope*
Slope of the 20‑day SMA over last ~3 trading days: **Up / Flat / Down**.
- **Use:** confirmation for entries; “Up” into a strong trend is ideal for momentum adds.

### vol_trend_20d — *Short‑term volatility slope*
**Rising / Flat / Falling** annualized vol over ~1 week.
- **Use:**  
  - **Falling**: smoother, often better for breakouts/adds.  
  - **Rising**: watch risk; breakouts may fail more often.

### ret_12_1_sortino_like — *Core: risk‑adjusted medium‑term return*
12→1‑month **log return** divided by long‑run **downside deviation** (annualized).  
- **Higher is better.**
- **Use:** core ranking signal (momentum quality).

### ret_20d_sortino_like — *Short‑term risk‑adjusted return*
**20‑day** log return / downside deviation.
- **Use:** complements the 12→1 measure for near‑term acceleration/deceleration.

### ret_12_1_vs_spy — *Relative to benchmark*
Your `ret_12_1_sortino_like` **minus** the benchmark’s (default `SPY`).
- **> 0** = outperformance on the core signal
- **Use:** sanity check leadership vs market.

### ulcer_63d — *Drawdown stress (lower is better)*
Ulcer Index over ~3 months (63 trading days). Expressed in **percentage points**.
- **Use:** high values = grinding drawdowns; avoid for clean momentum; acceptable in turnaround/reversion plays if position‑sized smaller.

### max_dd_63d — *Worst peak‑to‑trough (higher is better)*
Max drawdown over ~3 months, in **percent** (usually negative).  
- **Higher/less negative is better** (e.g., **−5%** is better than **−20%**).

### mr_z_12_1 — *How stretched vs its own mean*
Z‑score of today’s log price vs the 12→1 mean.
- **Closer to 0** means “near typical”; **large |Z|** = extended (up or down).  
- **Use:** reversion entries: prefer small |Z| or opposite‑extreme turning toward 0.

### mr_prob_10d_12_1 — *Odds of mean‑reversion soon*
Historical fraction of times the price moved **toward** the 12→1 mean within ~10 days.
- **Higher is better** for reversion setups (context matters).

### mr_target_mean_price_12_1 — *Fair‑value anchor*
Implied price from the 12→1 log mean (≈ “gravity” for reversion).
- **Use:** compare to current price for potential move back toward the mean.

### sharpe_12_1 / sortino_12_1 — *Classic risk‑adjusted returns*
Annualized Sharpe/Sortino over the 12→1 window.
- **Higher is better.**  
- **Use:** cross‑check with the custom Sortino‑like metrics.

### beta_252d_vs_spy — *Market sensitivity*
CAPM beta over ~1 year (252 trading days).
- **≈1**: market‑like; **>1** higher sensitivity; **<1** defensive.  
- **Use:** position sizing & portfolio risk mix.

### idio_vol_252d_ann — *Stock‑specific noise*
Annualized volatility of regression residuals (what’s not explained by the market).
- **Lower is better** for steady trends; higher for story/volatile names.

### atrp_14 (% of price) — *Typical daily move*
ATR(14) as a percent of price.
- **Use:** position sizing; higher ATRP = wider stops, smaller size.

### sector_alignment (if present) — *Convergence / Divergence*
Label that compares a stock’s recent path to its sector’s median path:  
- **Converging**: moving with sector winds  
- **Diverging+**: outperforming (but watch for over‑extension)  
- **Diverging−**: lagging (could be value or trouble)

---

## Practical Playbooks

### 1) Momentum continuation (trend following)
- Look for: **High `grade_12_1`**, **Clean** `trend_bucket`, **Up** `sma20_slope_3d`, **Up** `rsi14_trend_1w`, **Falling** `vol_trend_20d`, **ret_12_1_vs_spy > 0`**.  
- Avoid: very high `ulcer_63d` / very negative `max_dd_63d`.
- Size down if `atrp_14` is large.

### 2) Pullback‑and‑go (buy the dip within an uptrend)
- Look for: **High `grade_12_1`**, `mr_z_12_1` near 0 or slightly negative, **rsi14_trend_1w: Up**, `sma20_slope_3d: Up` flipping from Flat/Down.  
- Risk checks: `ulcer_63d` not extreme; `vol_trend_20d` not spiking.

### 3) Mean‑reversion (short‑term swing)
- Look for: Large |`mr_z_12_1`| **starting to shrink**, **high `mr_prob_10d_12_1`**, ideally with **Falling** `vol_trend_20d`.  
- Best on liquid tickers with moderate `atrp_14` and manageable `ulcer_63d`.

### 4) Turnaround / value‑with‑catalyst (advanced)
- Look for: **Improving** short‑term signals (`sma20_slope_3d: Up`, `rsi14_trend_1w: Up`) while **grade_12_1** is below average but rising; supportive `sector_alignment`.  
- Watch risk: `ulcer_63d`, `max_dd_63d`, and `atrp_14`. Size appropriately.

---

## Tips, Thresholds & Pitfalls

- **Quartile colors** are *relative* to the current sheet; a “green” might still be mediocre vs an external universe.
- **Data gaps** (new IPOs, corporate actions) can produce NaNs; such cells won’t be colored.
- **Small caps / news events**: metrics can whipsaw; rely more on risk columns and position sizing.
- **Benchmark**: `ret_12_1_vs_spy` depends on the configured benchmark (defaults to `SPY` if not set).

---

## FAQ

**Q: Why doesn’t a column have colors?**  
A: It’s either a text field with no rule, a numeric column with no data (NaN), or it’s marked “neutral.”

**Q: Why does a “green” stock still trade poorly?**  
A: Colors are **relative ranks** and can’t see news, liquidity, or execution. Use risk controls and confirm entries.

**Q: Should I use RSI thresholds like 70/30?**  
A: Better to watch **direction** and **context**: an RSI >70 in a clean trend can keep trending; an RSI rising from 40→55 with “Up” slope may be a better momentum add than a blind “overbought” sell.

---

*This guide describes the report’s quantitative signals. It is not investment advice.*
