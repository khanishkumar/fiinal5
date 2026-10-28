"""Fouling-resistance engineering calculations (SI internally)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

U_UNITS = {"W/(m²·K)": 1.0, "kW/(m²·K)": 1000.0}          # -> W/(m²·K)
RF_UNITS = {"m²·K/W": 1.0, "m²·K/kW": 1e-3}                # -> m²·K/W
DEFAULT_THRESHOLD_M2K_PER_KW = 0.043                      # provisional, NOT verified


def _ok(x) -> bool:
    return x is not None and np.isfinite(x)


def to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


@dataclass
class RfResult:
    available: bool
    rf_si: float | None = None            # m²·K/W
    uf_si: float | None = None            # W/(m²·K)
    uclean_si: float | None = None        # W/(m²·K)
    negative: bool = False
    message: str = ""


def compute_rf(uf_si, uclean_si) -> RfResult:
    """Rf = 1/Uf - 1/Uclean  (both in W/(m²·K), same area basis)."""
    if not _ok(uf_si):
        return RfResult(False, message="Uf (operating-row U) is missing or non-numeric.")
    if not _ok(uclean_si):
        return RfResult(False, uf_si=uf_si, message="No valid clean reference Uclean is available.")
    if uf_si <= 0 or uclean_si <= 0:
        return RfResult(False, uf_si=uf_si, uclean_si=uclean_si,
                        message="U values must be > 0 W/(m²·K) to compute Rf.")
    rf = 1.0 / uf_si - 1.0 / uclean_si
    return RfResult(True, rf, uf_si, uclean_si, rf < 0,
                    "Rf is negative: Uf exceeds Uclean. Inspect the baseline, operating conditions, "
                    "measurement uncertainty and data quality." if rf < 0 else "")


def rf_from_column(rf_value, unit: str) -> RfResult:
    v = to_float(rf_value)
    if v is None:
        return RfResult(False, message="Rf value in the selected row is missing or non-numeric.")
    si = v * RF_UNITS[unit]
    return RfResult(True, si, negative=si < 0,
                    message="Rf from the dataset is negative; check its definition and baseline." if si < 0 else "")


def rf_series(u_si: pd.Series, uclean_si: float) -> pd.Series:
    """Vectorised Rf (m²·K/W); NaN where U<=0/NaN or Uclean invalid."""
    out = pd.Series(np.nan, index=u_si.index, dtype=float)
    if not _ok(uclean_si) or uclean_si <= 0:
        return out
    m = u_si.notna() & (u_si > 0)
    out[m] = 1.0 / u_si[m] - 1.0 / uclean_si
    return out


def threshold_to_si(value: float, unit: str) -> float:
    return float(value) * RF_UNITS[unit]


def compare_threshold(rf_si, thr_si) -> dict:
    if not _ok(rf_si) or not _ok(thr_si) or thr_si <= 0:
        return {"ratio": None, "at_or_above": None, "status": "Not assessable"}
    ratio = rf_si / thr_si
    at = rf_si >= thr_si
    return {"ratio": ratio, "at_or_above": at,
            "status": "At/above threshold" if at else "Below threshold"}


def severity_from_rf(rf_si, thr_si, med_mult: float, high_mult: float) -> dict:
    """Low < med_mult·thr <= Medium < high_mult·thr <= High (all configurable)."""
    b_med, b_high = med_mult * thr_si, high_mult * thr_si
    base = {"low_max": b_med, "med_max": b_high}
    if not _ok(rf_si) or thr_si <= 0 or high_mult <= med_mult:
        return {**base, "level": None, "reason": "Rf unavailable or boundaries invalid."}
    if rf_si < 0:
        return {**base, "level": None, "reason": "Rf is negative; severity is not assessable."}
    level = "Low" if rf_si < b_med else "Medium" if rf_si < b_high else "High"
    return {**base, "level": level, "reason": ""}
