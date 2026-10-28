"""Leakage-safe preprocessing, model training and evaluation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_OPTS = ["Median imputation (fitted on training split)",
                "Mean imputation (fitted on training split)",
                "Drop rows with missing feature values"]
OUTLIER_OPTS = ["Keep all values (flag only)",
                "Clip to IQR fences (fitted on training split)"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise to [Q1-k·IQR, Q3+k·IQR]; fences are learned on the training data only."""
    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def iqr_table(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty or not pd.api.types.is_numeric_dtype(s):
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        below, above = int((s < lo).sum()), int((s > hi).sum())
        rows.append({"column": c, "Q1": q1, "Q3": q3, "lower fence": lo, "upper fence": hi,
                     "below": below, "above": above, "outliers %": round(100 * (below + above) / len(s), 2)})
    return pd.DataFrame(rows)


def split_feature_types(df: pd.DataFrame, features: list[str]):
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return num, cat


def build_preprocessor(num_cols, cat_cols, missing: str, outlier: str, iqr_k: float):
    strat = "mean" if missing.startswith("Mean") else "median"
    steps = []
    if outlier.startswith("Clip"):
        steps.append(("clip", IQRClipper(iqr_k)))
    steps += [("imp", SimpleImputer(strategy=strat, keep_empty_features=True)), ("sc", StandardScaler())]
    parts = []
    if num_cols:
        parts.append(("num", Pipeline(steps), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


def feature_names(pre: ColumnTransformer, num_cols, cat_cols) -> list[str]:
    names = list(num_cols)
    if cat_cols:
        names += list(pre.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols))
    return names


def prepare_dataset(df, features, target, clean_val, fouled_val, dedup, missing):
    """Row-level cleaning only (no fitted statistics -> no leakage). Original df is untouched."""
    rep = {"original rows": len(df)}
    d = df.copy()
    if dedup:
        d = d.drop_duplicates()
    rep["after duplicate handling"] = len(d)
    lab = d[target].astype(str)
    d = d[lab.isin([str(clean_val), str(fouled_val)])]
    rep["after removing missing/other labels"] = len(d)
    if missing.startswith("Drop"):
        d = d.dropna(subset=features)
    rep["after missing-value handling"] = len(d)
    y = (d[target].astype(str) == str(fouled_val)).astype(int)
    return d, y, rep


def build_estimators(seed, n_trees, pos_weight):
    est, errs = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=n_trees, class_weight="balanced",
                                                n_jobs=-1, random_state=seed),
    }, {}
    try:
        from xgboost import XGBClassifier
        est["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                                       colsample_bytree=0.9, eval_metric="logloss", tree_method="hist",
                                       scale_pos_weight=pos_weight, random_state=seed, n_jobs=-1)
    except Exception as e:  # noqa: BLE001
        errs["XGBoost"] = f"XGBoost unavailable: {e}"
    return est, errs


def train_and_evaluate(data, y, features, selected_models, test_size, seed, n_trees,
                       missing, outlier, iqr_k, time_values=None):
    if len(data) < 40:
        raise ValueError(f"Only {len(data)} usable rows; at least 40 are required for a train/test split.")
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 5:
        raise ValueError(f"Each class needs >= 5 rows. Class counts: {counts.to_dict()}")
    X = data[features]
    num, cat = split_feature_types(data, features)
    if time_values is not None:
        t = pd.to_datetime(time_values.loc[X.index], errors="coerce")
        if t.isna().all():
            t = time_values.loc[X.index]
        order = np.argsort(t.to_numpy(), kind="stable")
        n_test = max(1, int(round(len(X) * test_size)))
        tr_idx, te_idx = X.index[order[:-n_test]], X.index[order[-n_test:]]
        split_desc = "time-ordered (latest rows are the test set)"
    else:
        tr_idx, te_idx = train_test_split(X.index, test_size=test_size, random_state=seed, stratify=y)
        split_desc = "stratified random"
    Xtr, Xte, ytr, yte = X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]
    if ytr.nunique() < 2:
        raise ValueError("Training split contains a single class; change the split.")
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    est, errors = build_estimators(seed, n_trees, pos_w)
    models, rows, cms, rocs = {}, [], {}, {}
    for name in selected_models:
        if name not in est:
            continue
        try:
            pipe = Pipeline([("pre", build_preprocessor(num, cat, missing, outlier, iqr_k)), ("clf", est[name])])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            proba = pipe.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, proba) if yte.nunique() == 2 else np.nan
            rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                         "Precision": precision_score(yte, pred, zero_division=0),
                         "Recall": recall_score(yte, pred, zero_division=0),
                         "F1": f1_score(yte, pred, zero_division=0), "ROC-AUC": auc})
            cms[name] = confusion_matrix(yte, pred, labels=[0, 1])
            if yte.nunique() == 2:
                fpr, tpr, _ = roc_curve(yte, proba)
                rocs[name] = (fpr, tpr)
            models[name] = pipe
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{name} failed: {e}"
    if not models:
        raise ValueError("No model could be trained. " + " | ".join(errors.values()))
    return {"models": models, "metrics": pd.DataFrame(rows), "cms": cms, "rocs": rocs, "errors": errors,
            "features": features, "num": num, "cat": cat, "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
            "train_idx": set(tr_idx), "test_idx": set(te_idx), "split": split_desc,
            "class_counts": counts.to_dict(), "n_train": len(Xtr), "n_test": len(Xte)}


def predict_row(models: dict, row: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, pipe in models.items():
        try:
            p = float(pipe.predict_proba(row)[0, 1])
            out.append({"Model": name, "Predicted": "Fouled" if p >= 0.5 else "Clean",
                        "P(Fouled)": p, "Confidence": max(p, 1 - p)})
        except Exception as e:  # noqa: BLE001
            out.append({"Model": name, "Predicted": f"error: {e}", "P(Fouled)": np.nan, "Confidence": np.nan})
    return pd.DataFrame(out)
