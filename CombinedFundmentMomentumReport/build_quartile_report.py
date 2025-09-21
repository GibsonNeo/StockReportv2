
#!/usr/bin/env python3
"""
build_quartile_report.py

Read two CSV inputs, join on a key column (default Ticker), compute quartile grades for selected columns
with small decile nudges, then produce a combined weighted score.

Usage examples:
  python build_quartile_report.py --input-a path/to/tickers_report.csv --input-b path/to/momentum_report.csv --config grading.yml --out Combined-Quartile-Report.xlsx

Config file notes:
  See grading.yml alongside this script for a template.
  The config controls the join key, the join type, the list of columns to grade, the direction
  for each column, the weights, and the nudge rules.
"""

import argparse
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import yaml

def read_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def _coerce_numeric(s: pd.Series) -> pd.Series:
    # Convert strings like "12.3%", " 5 ", or "NaN" to floats
    # Leave NaN as is
    s2 = pd.to_numeric(s.astype(str).str.replace("%","", regex=False).str.strip(), errors="coerce")
    return s2

def _upper_strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def _qcut_safe(values: pd.Series, q: int) -> Tuple[pd.Series, bool]:
    """
    Try qcut on the non NaN values.
    If it fails due to insufficient unique bins, fallback to rank based bins.
    Returns a series of bin labels 1..q on the same index as values, and a flag indicating fallback used.
    """
    idx = values.index
    x = values.dropna()
    used_fallback = False
    labels = None

    if len(x) == 0:
        return pd.Series(np.nan, index=idx), used_fallback

    try:
        labels = pd.qcut(x, q=q, labels=list(range(1, q+1)), duplicates="drop")
        # If qcut dropped bins, adjust to full 1..q using rank fallback
        if len(pd.unique(labels)) < q:
            raise ValueError("qcut dropped bins")
    except Exception:
        used_fallback = True
        # Rank 1..n, then cut into q equal sized buckets
        r = x.rank(method="average", ascending=True)
        # Create edges for q buckets
        edges = np.linspace(r.min(), r.max(), num=q+1)
        # Include rightmost edge
        bins = pd.cut(r, bins=edges, include_lowest=True, labels=list(range(1, q+1)))
        labels = bins

    # Reindex to full original index
    out = pd.Series(np.nan, index=idx, dtype="float")
    out.loc[labels.index] = labels.astype(float)
    return out, used_fallback

def _decile_mask(values: pd.Series, decile: float, higher_is_better: bool) -> Tuple[pd.Series, pd.Series]:
    """
    Return two boolean masks, top_decile and bottom_decile, aligned to values.index.
    If higher is better, top_decile corresponds to values in the highest decile.
    If lower is better, top_decile corresponds to values in the lowest decile.
    """
    x = values.dropna()
    if len(x) == 0:
        idx = values.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)

    # Compute quantile thresholds
    lo_q = x.quantile(0.10)
    hi_q = x.quantile(0.90)

    # Higher better: top is x >= hi_q, bottom is x <= lo_q
    # Lower better: invert
    if higher_is_better:
        top = values >= hi_q
        bottom = values <= lo_q
    else:
        top = values <= lo_q
        bottom = values >= hi_q

    top = top.fillna(False)
    bottom = bottom.fillna(False)
    return top, bottom

def grade_one_column(values: pd.Series,
                     higher_is_better: bool,
                     n_quartiles: int,
                     decile_nudge: float) -> pd.DataFrame:
    """
    Compute quartile grade in {1..4} plus a small decile nudge in {-d, 0, +d},
    then combine as gc = clip(quartile + nudge, 1, 4). Return a DataFrame with
    quartile, nudge, and grade columns.
    """
    # Build quartiles
    q_labels, used_fallback = _qcut_safe(values, q=n_quartiles)

    # Map quartiles to grades, 4 is best
    # If higher is better, quartile 4 is best, so grade = quartile
    # If lower is better, invert, so grade = 5 - quartile
    if higher_is_better:
        quartile_grade = q_labels
    else:
        quartile_grade = q_labels.map(lambda q: np.nan if pd.isna(q) else (n_quartiles + 1 - int(q)))

    quartile_grade = quartile_grade.astype(float)

    # Decile nudges
    top_mask, bottom_mask = _decile_mask(values, decile=0.10, higher_is_better=higher_is_better)
    nudge = pd.Series(0.0, index=values.index, dtype="float")
    nudge[top_mask] = decile_nudge
    nudge[bottom_mask] = -decile_nudge

    # Final column grade
    gc = quartile_grade + nudge
    gc = gc.clip(lower=1.0, upper=float(n_quartiles))
    gc = gc.round(2)

    out = pd.DataFrame({
        "quartile": q_labels.astype("float"),
        "quartile_grade": quartile_grade.astype("float"),
        "decile_nudge": nudge,
        "column_grade": gc
    }, index=values.index)

    # For transparency include a flag when qcut fallback was used
    out["quartile_fallback_used"] = used_fallback
    return out

def compute_all_grades(df: pd.DataFrame,
                       spec: List[Dict[str, Any]],
                       decile_nudge: float,
                       n_quartiles: int) -> pd.DataFrame:
    """
    For each spec item with keys, name, source_column, direction, weight, compute the column_grade.
    Returns a wide DataFrame with new columns for each graded metric, plus a combined score.
    """
    # Prepare accumulators for the weighted sum
    weighted_sum = pd.Series(0.0, index=df.index, dtype="float")
    weight_present = pd.Series(0.0, index=df.index, dtype="float")

    all_parts = []

    for item in spec:
        name = item["name"]
        src = item["source_column"]
        direction = item.get("direction", "higher_better")
        weight = float(item.get("weight", 1.0))

        higher_is_better = direction == "higher_better"

        # Coerce numeric values
        vals = _coerce_numeric(df[src]) if src in df.columns else pd.Series(np.nan, index=df.index, dtype="float")

        gdf = grade_one_column(vals, higher_is_better=higher_is_better,
                               n_quartiles=n_quartiles, decile_nudge=decile_nudge)

        # Rename columns to include the metric name
        gdf = gdf.rename(columns={
            "quartile": f"{name}_quartile_raw",
            "quartile_grade": f"{name}_quartile_grade",
            "decile_nudge": f"{name}_decile_nudge",
            "column_grade": f"{name}_grade",
            "quartile_fallback_used": f"{name}_quartile_fallback_used",
        })

        all_parts.append(gdf)

        # Update weighted accumulators where grade is present
        mask = gdf[f"{name}_grade"].notna()
        weighted_sum.loc[mask] += gdf.loc[mask, f"{name}_grade"] * weight
        weight_present.loc[mask] += weight

    # Combine parts
    out = pd.concat([df] + all_parts, axis=1)

    # Final combined scores
    combined_1_to_4 = weighted_sum / weight_present.replace(0.0, np.nan)
    combined_percent = ((combined_1_to_4 - 1.0) / 3.0) * 100.0

    out["combined_weighted_1_to_4"] = combined_1_to_4.round(3)
    out["combined_percent_0_to_100"] = combined_percent.round(2)
    out["combined_weights_covered"] = weight_present

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-a", required=True, help="Path to first CSV, for example fundamentals or risk")
    ap.add_argument("--input-b", required=True, help="Path to second CSV, for example momentum")
    ap.add_argument("--config", required=True, help="Path to grading .yml")
    ap.add_argument("--out", required=True, help="Output Excel or CSV file path, extension determines format")
    args = ap.parse_args()

    cfg = read_config(Path(args.config))

    key = cfg.get("join_key", "Ticker")
    join_type = cfg.get("join_type", "inner")  # inner, left_a, left_b, outer
    to_grade = cfg.get("grade_columns", [])
    n_quartiles = int(cfg.get("n_quartiles", 4))
    decile_nudge = float(cfg.get("decile_nudge", 0.25))

    # Load inputs
    df_a = pd.read_csv(args.input_a)
    df_b = pd.read_csv(args.input_b)

    # Standardize key case
    if key not in df_a.columns or key not in df_b.columns:
        sys.stderr.write(f"Join key '{key}' must exist in both CSVs\n")
        sys.exit(2)

    df_a[key] = _upper_strip(df_a[key])
    df_b[key] = _upper_strip(df_b[key])

    # Join
    if join_type == "inner":
        merged = pd.merge(df_a, df_b, on=key, how="inner", suffixes=("_A", "_B"))
    elif join_type == "left_a":
        merged = pd.merge(df_a, df_b, on=key, how="left", suffixes=("_A", "_B"))
    elif join_type == "left_b":
        merged = pd.merge(df_b, df_a, on=key, how="left", suffixes=("_B", "_A"))
    elif join_type == "outer":
        merged = pd.merge(df_a, df_b, on=key, how="outer", suffixes=("_A", "_B"))
    else:
        sys.stderr.write(f"Unsupported join_type '{join_type}', use inner, left_a, left_b, or outer\n")
        sys.exit(2)

    # Compute grades
    out = compute_all_grades(merged, spec=to_grade,
                             decile_nudge=decile_nudge,
                             n_quartiles=n_quartiles)

    # Save
    out_path = Path(args.out)
    if out_path.suffix.lower() in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        try:
            import openpyxl  # noqa
        except Exception:
            sys.stderr.write("openpyxl is required to write Excel files, please pip install openpyxl\n")
            sys.exit(2)
        out.to_excel(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows")

if __name__ == "__main__":
    main()
