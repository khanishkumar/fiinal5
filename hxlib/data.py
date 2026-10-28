"""Dataset loading, validation and automatic column-role detection."""
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd

# role -> (label, default unit)
ROLES = {
    "target": ("Fouling label (target)", ""),
    "u": ("Overall heat-transfer coefficient U", ""),
    "hot_in": ("Hot-side inlet temperature", "°C"),
    "hot_out": ("Hot-side outlet temperature", "°C"),
    "cold_in": ("Cold-side inlet temperature", "°C"),
    "cold_out": ("Cold-side outlet temperature", "°C"),
    "hot_flow": ("Hot-side flow rate", "m³/h"),
    "cold_flow": ("Cold-side flow rate", "m³/h"),
    "hot_dp": ("Hot-side pressure drop", "bar"),
    "cold_dp": ("Cold-side pressure drop", "bar"),
    "q": ("Heat-transfer rate", "kW"),
    "effectiveness": ("Thermal effectiveness", "%"),
    "time": ("Timestamp / time index", ""),
}
NON_NUMERIC_ROLES = {"target", "time"}


def _tok(name: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}


def _has(t: set, *opts) -> bool:
    return bool(t & set(opts))


_RULES = {
    "target": lambda t: _has(t, "fouling", "fouled", "label", "target", "class", "status", "condition")
    and not _has(t, "resistance", "rf", "factor"),
    "u": lambda t: _has(t, "htc", "u") or (_has(t, "coefficient") and _has(t, "heat", "overall", "transfer")),
    "hot_in": lambda t: _has(t, "hot") and _has(t, "in", "inlet") and _has(t, "temp", "temperature", "t"),
    "hot_out": lambda t: _has(t, "hot") and _has(t, "out", "outlet") and _has(t, "temp", "temperature", "t"),
    "cold_in": lambda t: _has(t, "cold", "coolant") and _has(t, "in", "inlet") and _has(t, "temp", "temperature", "t"),
    "cold_out": lambda t: _has(t, "cold", "coolant") and _has(t, "out", "outlet") and _has(t, "temp", "temperature", "t"),
    "hot_flow": lambda t: _has(t, "hot") and _has(t, "flow", "flowrate"),
    "cold_flow": lambda t: _has(t, "cold", "coolant") and _has(t, "flow", "flowrate"),
    "hot_dp": lambda t: _has(t, "hot") and _has(t, "drop", "dp", "pressure"),
    "cold_dp": lambda t: _has(t, "cold", "coolant") and _has(t, "drop", "dp", "pressure"),
    "q": lambda t: (_has(t, "q", "duty") or (_has(t, "heat") and _has(t, "rate", "duty")))
    and not _has(t, "coefficient", "htc", "u"),
    "effectiveness": lambda t: _has(t, "effectiveness", "efficiency"),
    "time": lambda t: _has(t, "time", "timestamp", "date", "datetime"),
}


def load_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV. Raises ValueError with a readable message."""
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if not raw or not raw.strip():
        raise ValueError("The uploaded file is empty.")
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded file contains no data.")
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise ValueError(f"Could not parse the CSV: {last}")
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The CSV has no data rows or no columns.")
    if df.columns.duplicated().any():
        raise ValueError("The CSV has duplicate column names: "
                         + ", ".join(df.columns[df.columns.duplicated()]))
    return df.reset_index(drop=True)


def detect_columns(df: pd.DataFrame) -> dict:
    """Heuristic candidate for every role (user must confirm)."""
    found, used = {}, set()
    for role in ROLES:
        found[role] = None
        rule = _RULES.get(role)
        if rule is None:
            continue
        for c in df.columns:
            if c not in used and rule(_tok(c)):
                found[role] = c
                used.add(c)
                break
    return found


def guess_unit(col: str | None, role: str) -> str:
    default = ROLES[role][1]
    if not col:
        return default
    t = _tok(col)
    if role in ("hot_in", "hot_out", "cold_in", "cold_out"):
        return "K" if "k" in t else "°F" if "f" in t else "°C"
    for key, unit in [("m3h", "m³/h"), ("lpm", "L/min"), ("kpa", "kPa"), ("bar", "bar"),
                      ("psi", "psi"), ("kw", "kW"), ("pct", "%")]:
        if key in t:
            return unit
    return default


def guess_u_unit(col: str | None) -> str:
    c = (col or "").lower()
    return "kW/(m²·K)" if ("kw" in c and "m2" in c) else "W/(m²·K)"


def guess_rf_unit(col: str | None) -> str:
    c = (col or "").lower()
    return "m²·K/kW" if "kw" in c else "m²·K/W"


def guess_labels(values: list[str]):
    clean_k = {"0", "clean", "no", "false", "normal", "healthy", "n"}
    foul_k = {"1", "fouled", "fouling", "yes", "true", "dirty", "faulty", "y"}
    clean = next((v for v in values if v.strip().lower() in clean_k), values[0] if values else None)
    fouled = next((v for v in values if v.strip().lower() in foul_k and v != clean),
                  next((v for v in values if v != clean), clean))
    return clean, fouled


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non-null": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "missing %": (df.isna().mean() * 100).round(2).values,
        "unique": df.nunique().values,
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def validate_mapping(df: pd.DataFrame, mp: dict):
    """Return (errors, warnings) for a role->column mapping."""
    errors, warnings = [], []
    used = {}
    for role, col in mp.items():
        if col is None:
            continue
        if col not in df.columns:
            errors.append(f"Mapped column '{col}' for {role} does not exist.")
            continue
        if col in used:
            warnings.append(f"Column '{col}' is mapped to both '{used[col]}' and '{role}'.")
        used[col] = role
        if role not in NON_NUMERIC_ROLES and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"'{col}' ({ROLES[role][0]}) is not numeric (dtype {df[col].dtype}).")
    if mp.get("target") and df[mp["target"]].nunique(dropna=True) > 20:
        errors.append(f"Target column '{mp['target']}' has >20 distinct values; it looks continuous, not a class label.")
    if mp.get("target") and df[mp["target"]].nunique(dropna=True) < 2:
        errors.append(f"Target column '{mp['target']}' has fewer than 2 classes.")
    return errors, warnings


def boundary_pinned(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Count values sitting exactly at a column's min or max (possible clipping/imputation artefacts)."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        lo, hi = s.min(), s.max()
        rows.append({"column": c, "min": lo, "rows at min": int((s == lo).sum()),
                     "max": hi, "rows at max": int((s == hi).sum()), "n": len(s)})
    return pd.DataFrame(rows)
