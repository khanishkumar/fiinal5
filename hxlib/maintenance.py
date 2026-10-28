"""Transparent rule-based maintenance decision support (applied to every analysed row)."""
from __future__ import annotations
import numpy as np

RECOMMEND = {
    "1": "Verify measurements, inspect the exchanger for deposits, investigate possible CaCO₃ scaling and consider a compatible cleaning method.",
    "2": "Investigate the influential parameters, check the clean baseline and review the data for this row.",
    "3": "ML and Rf disagree: verify the clean baseline and inspect the system when appropriate.",
    "4": "Routine monitoring; record performance trends (U, Rf, pressure drop).",
    "5": "Rf is not valid for this row: confirm the clean baseline; the ML result is reported separately.",
}
CASE_NAMES = {
    "1": "1 · ML Fouled & Rf ≥ threshold",
    "2": "2 · ML Fouled & Rf < threshold",
    "3": "3 · ML Clean & Rf ≥ threshold",
    "4": "4 · ML Clean & Rf < threshold",
    "5": "5 · Rf unavailable",
}


def case_key(ml_label, rf_value_si, at_or_above) -> str:
    if rf_value_si is None or not np.isfinite(rf_value_si) or rf_value_si < 0:
        return "5"
    if ml_label == "Fouled":
        return "1" if at_or_above else "2"
    return "3" if at_or_above else "4"


def recommend(key: str, severity: str | None) -> str:
    txt = RECOMMEND[key]
    return f"{txt} Rf severity: {severity}." if severity in ("Low", "Medium", "High") and key in "123" else txt
