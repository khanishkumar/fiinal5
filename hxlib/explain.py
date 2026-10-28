"""SHAP explanations computed from the trained pipelines (never hard-coded)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .models import feature_names


def _norm(sv):
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def make_explainer(pipe, name, bg_raw: pd.DataFrame):
    """Return an explainer object, or the string 'xgb-native' for the XGBoost fallback."""
    import shap
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    if name == "Logistic Regression":
        return shap.LinearExplainer(clf, pre.transform(bg_raw))
    try:
        return shap.TreeExplainer(clf)
    except Exception:
        if name == "XGBoost":
            return "xgb-native"   # XGBoost's own exact TreeSHAP (pred_contribs)
        raise


def shap_values(pipe, name, explainer, X_raw: pd.DataFrame, num, cat):
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pre.transform(X_raw)
    names = feature_names(pre, num, cat)
    if isinstance(explainer, str):
        import xgboost as xgb
        contrib = clf.get_booster().predict(xgb.DMatrix(Xt, feature_names=None), pred_contribs=True)
        return contrib[:, :-1], Xt, names
    return _norm(explainer.shap_values(Xt)), Xt, names


def local_table(sv_row, Xt_row, names, raw_row: pd.Series, units: dict) -> pd.DataFrame:
    rows = []
    for n, s, xt in zip(names, sv_row, Xt_row):
        in_raw = n in raw_row.index
        rows.append({"Feature": n,
                     "Actual value": raw_row[n] if in_raw else float(xt),
                     "Unit": units.get(n, ""),
                     "SHAP value": float(s),
                     "Pushes toward": "Fouled" if s > 0 else "Clean" if s < 0 else "Neutral"})
    return pd.DataFrame(rows).reindex(
        pd.Series([abs(r["SHAP value"]) for r in rows]).sort_values(ascending=False).index).reset_index(drop=True)
