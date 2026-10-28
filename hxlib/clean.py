"""Row-level cleaning applied before any analysis (original dataframe is never modified)."""
from __future__ import annotations
import pandas as pd


def clean_dataset(df, cols, target, clean_val, fouled_val, dedup, drop_noise, mp):
    """Remove duplicates, rows with missing values and physically inconsistent (noisy) rows.

    Returns (cleaned_df, report). The cleaned frame keeps the ORIGINAL row index so that
    row numbers always refer to the uploaded CSV.
    """
    rep = {"Original rows": len(df)}
    d = df
    n = len(d)
    if dedup:
        d = d.drop_duplicates()
    rep["Duplicate rows removed"] = n - len(d)
    n = len(d)
    d = d.dropna(subset=[c for c in dict.fromkeys(cols) if c in d.columns])
    rep["Rows removed (missing values)"] = n - len(d)
    n = len(d)
    if target:
        d = d[d[target].astype(str).isin([str(clean_val), str(fouled_val)])]
    rep["Rows removed (other labels)"] = n - len(d)
    n = len(d)
    if drop_noise and len(d):
        bad = pd.Series(False, index=d.index)
        t = [mp.get(k) for k in ("hot_in", "hot_out", "cold_in", "cold_out")]
        if all(t):
            hi, ho, ci, co = (d[c] for c in t)
            bad |= (hi <= ho) | (co <= ci) | (co > hi)
        if mp.get("u"):
            bad |= d[mp["u"]] <= 0
        d = d[~bad]
    rep["Rows removed (noise: inconsistent temperatures / U ≤ 0)"] = n - len(d)
    if mp.get("time") and len(d):
        t = pd.to_datetime(d[mp["time"]], errors="coerce")
        d = d.assign(_t=t if t.notna().any() else d[mp["time"]]).sort_values("_t", kind="stable").drop(columns="_t")
    rep["Total rows removed"] = len(df) - len(d)
    rep["Rows available for analysis"] = len(d)
    return d, rep
