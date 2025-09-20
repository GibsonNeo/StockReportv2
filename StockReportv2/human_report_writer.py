from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd

def _num_to_excel_col(n: int) -> str:
    """0-based column index -> Excel column label (A, B, ..., Z, AA, AB, ...)."""
    if n < 0:
        return 'A'
    out = []
    n0 = n
    while n0 >= 0:
        n0, rem = divmod(n0, 26)
        out.append(chr(ord('A') + rem))
        n0 -= 1
    return ''.join(reversed(out))


def _autofit_columns(ws, df: pd.DataFrame) -> None:
    """Auto-fit column widths to header + sample of data (approximate Excel double-click).
    Caps widths for readability.
    """
    max_width = 48
    min_width = 8
    sample_rows = min(len(df), 500)
    for colx, colname in enumerate(df.columns):
        # measure header
        header_len = len(str(colname))
        # measure sample of data (as displayed)
        data_lens = []
        if sample_rows > 0:
            # convert to string for length estimate
            sample = df.iloc[:sample_rows, colx]
            for val in sample:
                s = '' if pd.isna(val) else str(val)
                data_lens.append(len(s))
        best = max([header_len] + data_lens + [min_width])
        # add a little padding
        width = min(max(best + 5, min_width), max_width)
        ws.set_column(colx, colx, width)

def _apply_quartile_cf(wb, ws, df: pd.DataFrame, direction: Dict[str, str]) -> Dict[str, Tuple[float,float]]:
    """Apply quartile-based highlighting for numeric columns.
    direction: 'high' (green for top quartile), 'low' (green for bottom quartile),
               'low_magnitude' (green for smallest |value|), 'neutral' (no CF)

    NEW (expanded): 4-band scheme
      - Green  = best quartile
      - Blue   = 'good' near-best quartile
      - Yellow = below-average quartile
      - Red    = worst quartile

    Returns thresholds used per-column for auditing.
    """
    thresholds = {}
    green  = wb.add_format({'bg_color':'#C6EFCE'})  # best
    blue   = wb.add_format({'bg_color':'#DDEBF7'})  # good
    yellow = wb.add_format({'bg_color':'#FFF2CC'})  # caution
    red    = wb.add_format({'bg_color':'#FFC7CE'})  # worst

    for colx, colname in enumerate(df.columns):
        if colname not in direction: 
            continue
        dirn = direction[colname]
        if dirn == 'neutral':
            continue
        # numeric only
        if not pd.api.types.is_numeric_dtype(df[colname]):
            continue
        series = pd.to_numeric(df[colname], errors='coerce')
        s_valid = series.dropna()
        if s_valid.empty:
            continue

        q1 = float(s_valid.quantile(0.25))
        q2 = float(s_valid.quantile(0.50))
        q3 = float(s_valid.quantile(0.75))
        thresholds[colname] = (q1, q3)
        first_row = 2  # 1-based in Excel; row 1 is header
        last_row  = len(df) + 1
        col_letter = _num_to_excel_col(colx)
        rng = f"${col_letter}${first_row}:${col_letter}${last_row}"

        if dirn in ('high','low'):
            if dirn == 'high':
                # Apply mid-bands first; extremes last to override edges cleanly
                ws.conditional_format(rng, {'type': 'cell', 'criteria': 'between', 'minimum': q1, 'maximum': q2, 'format': yellow})
                ws.conditional_format(rng, {'type': 'cell', 'criteria': 'between', 'minimum': q2, 'maximum': q3, 'format': blue})
                ws.conditional_format(rng, {'type': 'cell', 'criteria': '<=', 'value': q1, 'format': red})
                ws.conditional_format(rng, {'type': 'cell', 'criteria': '>=', 'value': q3, 'format': green})
            else:  # 'low' orientation
                ws.conditional_format(rng, {'type': 'cell', 'criteria': 'between', 'minimum': q1, 'maximum': q2, 'format': blue})
                ws.conditional_format(rng, {'type': 'cell', 'criteria': 'between', 'minimum': q2, 'maximum': q3, 'format': yellow})
                ws.conditional_format(rng, {'type': 'cell', 'criteria': '<=', 'value': q1, 'format': green})
                ws.conditional_format(rng, {'type': 'cell', 'criteria': '>=', 'value': q3, 'format': red})

        elif dirn == 'low_magnitude':
            # Band on |x|
            abs_vals = s_valid.abs()
            aq1 = float(abs_vals.quantile(0.25)); aq2 = float(abs_vals.quantile(0.50)); aq3 = float(abs_vals.quantile(0.75))
            # Use formulas so we can reference ABS(cell). Relative row is handled by Excel CF.
            cell_ref = f'${col_letter}{first_row}'
            ws.conditional_format(rng, {'type': 'formula', 'criteria': f'=AND(ABS({cell_ref})>={aq1},ABS({cell_ref})<={aq2})', 'format': blue})
            ws.conditional_format(rng, {'type': 'formula', 'criteria': f'=AND(ABS({cell_ref})>={aq2},ABS({cell_ref})<={aq3})', 'format': yellow})
            ws.conditional_format(rng, {'type': 'formula', 'criteria': f'=ABS({cell_ref})<={aq1}', 'format': green})
            ws.conditional_format(rng, {'type': 'formula', 'criteria': f'=ABS({cell_ref})>={aq3}', 'format': red})
    return thresholds

def _apply_text_cf(wb, ws, df: pd.DataFrame) -> None:
    colormap = {
        'green':  wb.add_format({'bg_color':'#C6EFCE'}),
        'blue':   wb.add_format({'bg_color':'#DDEBF7'}),
        'yellow': wb.add_format({'bg_color':'#FFF2CC'}),
        'red':    wb.add_format({'bg_color':'#FFC7CE'}),
    }
    # Column -> {text -> color_key}
    rules = {
        'sma20_slope_3d': {'Up':'green','Down':'red','Flat':'yellow'},
        'trend_bucket': {'Clean':'green','Acceptable':'blue','Choppy':'red'},
        'rsi14_trend_1w': {'Up':'green','Down':'red','Flat':'yellow'},
        'vol_trend_20d': {'Rising':'red','Falling':'green','Flat':'yellow'},
        # legacy label (kept): sector_alignment
        'sector_alignment': {'Diverging+':'red','Diverging-':'green','Converging':'blue'},
        # new labels
        'ticker_vs_sector': {'Best in Breed':'green','Underperforming':'red','In-Line':'blue'},
        'sector_vs_market': {'Sector Outperform':'green','Sector Underperform':'red','Sector Neutral':'blue'},
    }
    first_row = 2; last_row = len(df)+1
    for colx, colname in enumerate(df.columns):
        mapping = rules.get(colname)
        if not mapping: 
            continue
        col_letter = _num_to_excel_col(colx)
        for txt, color in mapping.items():
            ws.conditional_format(f'${col_letter}${first_row}:${col_letter}${last_row}', {
                'type':'text', 'criteria':'containing', 'value': txt, 'format': colormap[color]
            })

    # --- CHANGE: prevent any generic text CF on 'ticker' and 'sector' ---
    exclude_cf_cols = {'ticker', 'sector'}

    # Generic YES/NO and UP/DOWN mapping for any textual columns not in explicit rules
    generic_map = {'Yes':'green','No':'red','UP':'green','Down':'red','up':'green','down':'red','YES':'green','NO':'red'}
    for colx, colname in enumerate(df.columns):
        if colname in rules:
            continue
        if colname in exclude_cf_cols:  # do not touch ticker/sector with generic text rules
            continue
        if not pd.api.types.is_object_dtype(df[colname]):
            continue
        col_letter = _num_to_excel_col(colx)
        for txt, color in generic_map.items():
            ws.conditional_format(f'${col_letter}${first_row}:${col_letter}${last_row}', {
                'type':'text', 'criteria':'containing', 'value': txt, 'format': colormap[color]
            })
    
    # Shade the 'sector' column to mirror each row's sector_vs_market status
    if 'sector' in df.columns and 'sector_vs_market' in df.columns:
        first_row = 2
        last_row  = len(df) + 1
        sector_colx = df.columns.get_loc('sector')
        svm_colx    = df.columns.get_loc('sector_vs_market')
        sector_col  = _num_to_excel_col(sector_colx)
        svm_col     = _num_to_excel_col(svm_colx)

        rng_sector = f'${sector_col}${first_row}:${sector_col}${last_row}'

        # Use mixed refs: lock the column ($), let the row float
        ws.conditional_format(rng_sector, {
            'type': 'formula',
            'criteria': f'=${svm_col}{first_row}="Sector Outperform"',
            'format': colormap['green']
        })
        ws.conditional_format(rng_sector, {
            'type': 'formula',
            'criteria': f'=${svm_col}{first_row}="Sector Neutral"',
            'format': colormap['blue']
        })
        ws.conditional_format(rng_sector, {
            'type': 'formula',
            'criteria': f'=${svm_col}{first_row}="Sector Underperform"',
            'format': colormap['red']
        })


def write_formatted_excel(df: pd.DataFrame, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_used = {}
    with pd.ExcelWriter(out_path, engine='xlsxwriter', datetime_format='yyyy-mm-dd', date_format='yyyy-mm-dd') as xw:
        sheet = 'Report'
        df.to_excel(xw, sheet_name=sheet, index=False)
        wb = xw.book
        ws = xw.sheets[sheet]

        # Freeze only the header row + Column A (release B & C)
        ws.freeze_panes(1, 1)

        # Add an autofilter on the header row across all columns
        ws.autofilter(0, 0, len(df), len(df.columns)-1)

        # Number formats: make a reasonable default for numbers; special-case some known columns
        num_fmt = wb.add_format({'num_format': '0.00'})
        pct_fmt = wb.add_format({'num_format': '0.00%'})
        int_fmt = wb.add_format({'num_format': '0'})
        for colx, colname in enumerate(df.columns):
            # choose a sensible default
            if colname.lower() in {'ticker','name','sector'}:
                continue  # leave as text
            fmt = num_fmt
            if colname in {'rsi14'}:
                fmt = num_fmt
                fmt = pct_fmt
            if colname in {'bull_stack_days_12_1_20_50_100'}:
                fmt = int_fmt
            ws.set_column(colx, colx, None, fmt)

        # Auto-fit columns to show header words fully (run last so widths stick)
        _autofit_columns(ws, df)

        # Conditional formatting directions
        direction = {
            'ret_12_1_vs_spy': 'high',
            # ADDED: include new momentum fields
            'mom_12_1_spdji': 'high',
            'mom_12_1_spdji_vs_spy': 'high',
            'ticker_vs_sector_z_12_1': 'high',
            'trend_consistency': 'high',  # Higher consistency score = better

            'ulcer_63d': 'low',             # smaller is better (NOT a percent format)
            'max_dd_63d': 'high',           # less negative (higher) is better
            'ulcer_12_1': 'low',             # smaller is better (NOT a percent)
            'max_dd_12_1': 'high',           # less negative (higher) is better
            'bull_stack_days_12_1_20_50_100': 'high',  # more days is better
            'sharpe_12_1': 'high',
            'sortino_12_1': 'high',
            'rsi14': 'high',
            'beta_252d_vs_spy': 'low_magnitude',
        
            'sector_vs_market_z_12_1': 'neutral',
            'sector_dispersion_12_1_pct': 'neutral',
            'sector_corr_21d_z': 'neutral',
            'market_dispersion_12_1_pct': 'neutral',
            'market_corr_21d_z': 'neutral',
        }
        thresholds_used = _apply_quartile_cf(wb, ws, df, direction)
        _apply_text_cf(wb, ws, df)
    return thresholds_used
