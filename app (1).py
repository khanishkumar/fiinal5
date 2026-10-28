"""AI-Based Heat Exchanger Fouling Prediction, Performance Monitoring & Maintenance Support Dashboard.
Prototype: automotive oil-cooler-based, PHE-like heat-exchanger (hot: water, cold: coolant)."""
from __future__ import annotations
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hxlib import (clean as C, data as D, engineering as E, explain as X, forecast as F,
                   maintenance as R, models as M)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HX Fouling Dashboard", page_icon="🔧", layout="wide")
st.markdown("""<style>
.kpi{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:12px 14px;margin-bottom:8px;
background:rgba(128,128,128,.07)}
.kpi .kl{font-size:.78rem;opacity:.75}.kpi .kv{font-size:1.45rem;font-weight:700}
.kpi .ku{font-size:.8rem;font-weight:400;opacity:.75}.kpi .ks{font-size:.75rem;opacity:.7}
.kpi.good{border-left:6px solid #2e9e5b}.kpi.bad{border-left:6px solid #d64545}
.kpi.warn{border-left:6px solid #e0a030}.kpi.na{border-left:6px solid #888}
</style>""", unsafe_allow_html=True)

NONE = "— not available —"
SECTIONS = ["1 · Data Input", "2 · Data Preprocessing", "3 · Engineering Performance",
            "4 · Fouling Prediction & XAI", "5 · Maintenance Support"]
DEFAULTS = {"p_section": SECTIONS[0], "p_confirmed": False, "p_thr_val": 0.043, "p_thr_unit": "m²·K/kW",
            "p_sev_med": 1.0, "p_sev_high": 3.0, "p_dedup": True, "p_noise": True, "p_test_size": 0.25,
            "p_seed": 42, "p_trees": 200, "p_split": "Stratified random", "p_models": M.MODEL_NAMES,
            "p_u_unit": "W/(m²·K)", "p_start": 0, "p_end": 500, "p_shap_n": 500, "p_lstm_win": 10,
            "p_lstm_hidden": 16, "p_lstm_epochs": 40, "p_lstm_horizon": 30, "p_lstm_smooth": 25}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
for _k in [k for k in st.session_state.keys() if k.startswith("p_")]:   # keep widget values across pages
    st.session_state[_k] = st.session_state[_k]


# ------------------------------------------------------------------ helpers
def fmt(x, spec=".4g", unit=""):
    try:
        if x is None or not np.isfinite(float(x)):
            return "n/a"
        return f"{float(x):{spec}}" + (f" {unit}" if unit else "")
    except (TypeError, ValueError):
        return "n/a"


def kpi(label, value, unit="", sub="", tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="kl">{label}</div><div class="kv">{value}'
                f'<span class="ku"> {unit}</span></div><div class="ks">{sub}</div></div>', unsafe_allow_html=True)


def mapping() -> dict:
    out = {}
    for r in D.ROLES:
        v = st.session_state.get(f"p_map_{r}")
        out[r] = None if v in (None, NONE) else v
    return out


def role_unit(role):
    return st.session_state.get(f"p_unit_{role}") or D.ROLES[role][1]


def feature_units(mp) -> dict:
    u = {c: role_unit(r) for r, c in mp.items() if c and r not in ("target", "time", "u")}
    if mp.get("u"):
        u[mp["u"]] = st.session_state["p_u_unit"]
    return u


def ingest(up):
    sig = (up.name, up.size)
    if st.session_state.get("file_sig") == sig:
        return
    try:
        df = D.load_csv(up)
    except ValueError as e:
        st.error(f"Upload failed: {e}")
        return
    for k in [k for k in st.session_state.keys() if k.startswith("p_") and k != "p_section"]:
        del st.session_state[k]
    for k in ("ml", "shap_cache", "lstm"):
        st.session_state.pop(k, None)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.update(df=df, file_sig=sig, file_name=up.name)
    det = D.detect_columns(df)
    for r, c in det.items():
        st.session_state[f"p_map_{r}"] = c or NONE
        st.session_state[f"p_unit_{r}"] = D.guess_unit(c, r)
    st.session_state["p_u_unit"] = D.guess_u_unit(det["u"])
    if det["target"]:
        vals = sorted(df[det["target"]].dropna().astype(str).unique().tolist())
        st.session_state["p_clean_val"], st.session_state["p_fouled_val"] = D.guess_labels(vals)


def require_mapping():
    df = st.session_state.get("df")
    if df is None:
        st.write("Upload a CSV in **Section 1 · Data Input** to begin.")
        st.stop()
    mp = mapping()
    errs, _ = D.validate_mapping(df, mp)
    if errs or not st.session_state.get("p_confirmed"):
        st.write("Confirm the column mappings in Section 1 first.")
        for e in errs:
            st.error(e)
        st.stop()
    return df, mp


def need_u_and_label(mp):
    if not (mp["u"] and mp["target"]):
        st.error("Map the U column and the label column in Section 1 (needed for the clean baseline and ML).")
        st.stop()


def label_values(df, mp):
    if not mp["target"]:
        return None, None
    opts = sorted(df[mp["target"]].dropna().astype(str).unique().tolist())
    c, f = st.session_state.get("p_clean_val"), st.session_state.get("p_fouled_val")
    if c not in opts or f not in opts:
        c, f = D.guess_labels(opts)
        st.session_state["p_clean_val"], st.session_state["p_fouled_val"] = c, f
    return c, f


def default_features(mp):
    return [mp[r] for r in D.ROLES if r not in ("target", "time") and mp[r]]


def feature_list(df, mp):
    cur = [c for c in st.session_state.get("p_features", []) if c in df.columns and c != mp["target"]]
    new = cur or default_features(mp)
    if st.session_state.get("p_features") != new:
        st.session_state["p_features"] = new
    return new


def get_clean(df, mp):
    c, f = label_values(df, mp)
    feats = feature_list(df, mp)
    cols = [x for x in mp.values() if x] + feats
    return C.clean_dataset(df, cols, mp["target"], c, f, st.session_state["p_dedup"], st.session_state["p_noise"], mp)


def baseline_u(clean, mp, df):
    """Stable clean baseline: median U (W/(m²·K)) of Clean-labelled rows among the cleaned data."""
    c, _ = label_values(df, mp)
    s = pd.to_numeric(clean.loc[clean[mp["target"]].astype(str) == str(c), mp["u"]], errors="coerce")
    s = s[s > 0]
    return float(s.median()) * E.U_UNITS[st.session_state["p_u_unit"]] if len(s) else None


def thr_si():
    return E.threshold_to_si(st.session_state["p_thr_val"], st.session_state["p_thr_unit"])


def analysis_context(df, mp):
    clean, rep = get_clean(df, mp)
    n = len(clean)
    if n == 0:
        st.error("No rows remain after cleaning.")
        st.stop()
    s = st.session_state
    for k in ("p_start", "p_end"):
        v = int(min(max(s[k], 0), n - 1))
        if s[k] != v:
            s[k] = v
    a, b = sorted((s["p_start"], s["p_end"]))
    return {"clean": clean, "rep": rep, "n": n, "a": a, "b": b}


def ml_signature(df, mp, n_clean):
    s = st.session_state
    return (s.get("file_sig"), tuple(feature_list(df, mp)), mp["target"], s.get("p_clean_val"), s.get("p_fouled_val"),
            s["p_dedup"], s["p_noise"], n_clean, s["p_test_size"], s["p_seed"], s["p_split"], s["p_trees"],
            tuple(s["p_models"]))


def run_training(df, mp, clean):
    s = st.session_state
    _, f = label_values(df, mp)
    y = (clean[mp["target"]].astype(str) == str(f)).astype(int)
    tv = clean[mp["time"]] if (s["p_split"].startswith("Time") and mp["time"]) else None
    out = M.train_and_evaluate(clean, y, feature_list(df, mp), s["p_models"], s["p_test_size"], s["p_seed"],
                               s["p_trees"], M.MISSING_OPTS[2], M.OUTLIER_OPTS[0], 1.5, tv)
    out["sig"] = ml_signature(df, mp, len(clean))
    return out


def ensure_models(df, mp, clean, force=False):
    ml = st.session_state.get("ml")
    if ml and not force and ml["sig"] == ml_signature(df, mp, len(clean)):
        return ml
    try:
        with st.spinner("Training models on the cleaned dataset…"):
            st.session_state["ml"] = run_training(df, mp, clean)
            st.session_state["shap_cache"] = {}
        return st.session_state["ml"]
    except Exception as e:  # noqa: BLE001
        st.session_state.pop("ml", None)
        st.error(f"Training failed: {e}")
        return None


def primary_model(ml):
    best = ml["metrics"].sort_values("F1").iloc[-1]["Model"]
    if st.session_state.get("p_primary") not in ml["models"]:
        st.session_state["p_primary"] = best
    return st.session_state["p_primary"]


def label_name(series, df, mp):
    c, f = label_values(df, mp)
    s = series.astype(str)
    return np.where(s == str(c), "Clean", np.where(s == str(f), "Fouled", "Other"))


def build_range(df, mp, ctx, ml, model, uclean, thr):
    """Analyse EVERY row in the selected range: Rf, threshold status, ML status, severity, decision case."""
    clean, a, b = ctx["clean"], ctx["a"], ctx["b"]
    sub = clean.iloc[a:b + 1]
    u_col = f"U ({st.session_state['p_u_unit']})"
    rf = E.rf_series(pd.to_numeric(sub[mp["u"]], errors="coerce") * E.U_UNITS[st.session_state["p_u_unit"]], uclean).to_numpy()
    p = ml["models"][model].predict_proba(sub[ml["features"]])[:, 1]
    ml_status = np.where(p >= 0.5, "Fouled", "Clean")
    at = rf >= thr
    ok = np.isfinite(rf)
    low, med = st.session_state["p_sev_med"] * thr, st.session_state["p_sev_high"] * thr
    sev = np.where(~ok | (rf < 0), "n/a", np.where(rf < low, "Low", np.where(rf < med, "Medium", "High")))
    keys = [R.case_key(m, r if np.isfinite(r) else None, bool(x)) for m, r, x in zip(ml_status, rf, at)]
    fr = pd.DataFrame({"Row": sub.index.astype(int), "Position": np.arange(a, a + len(sub)),
                       u_col: sub[mp["u"]].to_numpy(), "Rf (m²·K/kW)": rf * 1e3, "Rf / threshold": rf / thr,
                       "At/above": at & ok, "Rf status": np.where(~ok, "n/a", np.where(at, "At/above threshold", "Below threshold")),
                       "ML status": ml_status, "P(Fouled)": p, "Actual label": label_name(sub[mp["target"]], df, mp),
                       "Severity": sev, "Decision case": [R.CASE_NAMES[k] for k in keys],
                       "Recommended action": [R.recommend(k, v) for k, v in zip(keys, sev)]})
    for name, pipe in ml["models"].items():
        fr[f"Pred · {name}"] = np.where(pipe.predict_proba(sub[ml["features"]])[:, 1] >= 0.5, "Fouled", "Clean")
    return fr


def get_explainer(ml, name):
    import shap  # noqa: F401
    cache = st.session_state.setdefault("shap_cache", {})
    if name not in cache:
        bg = ml["Xtr"].sample(min(100, len(ml["Xtr"])), random_state=0)
        cache[name] = {"explainer": X.make_explainer(ml["models"][name], name, bg)}
    return cache[name]


def range_shap(ml, name, clean, a, b, cap=1000):
    """SHAP values for the analysed rows (sampled if more than `cap`). Cached per model and range."""
    c = get_explainer(ml, name)
    key = (a, b, cap, ml["sig"])
    if c.get("rkey") != key:
        sub = clean.iloc[a:b + 1]
        if len(sub) > cap:
            sub = sub.sample(cap, random_state=0)
        c["rres"] = X.shap_values(ml["models"][name], name, c["explainer"], sub[ml["features"]], ml["num"], ml["cat"])
        c["rsub"], c["rkey"] = sub, key
    return c["rres"], c["rsub"]


# ------------------------------------------------------------------ Section 1
def page_input():
    st.header("Section 1 · Data Input")
    up = st.file_uploader("Upload CSV dataset", type=["csv"])
    if up is not None:
        with st.spinner("Reading file…"):
            ingest(up)
    df = st.session_state.get("df")
    if df is None:
        st.write("Upload a CSV file to begin.")
        return
    st.markdown(f"Loaded **{st.session_state['file_name']}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), width="stretch")

    st.subheader("Column mapping (auto-detected — please verify)")
    cols = [NONE] + list(df.columns)
    grid = st.columns(3)
    for i, (role, (label, _)) in enumerate(D.ROLES.items()):
        with grid[i % 3]:
            st.selectbox(label, cols, key=f"p_map_{role}")
            if role == "u":
                st.selectbox("Unit of U", list(E.U_UNITS), key="p_u_unit")
            elif role not in D.NON_NUMERIC_ROLES:
                st.text_input("Unit (display only)", key=f"p_unit_{role}")
    mp = mapping()
    errs, warns = D.validate_mapping(df, mp)
    for e in errs:
        st.error(e)
    if mp["target"] and not errs:
        st.markdown("**Target label mapping**")
        opts = sorted(df[mp["target"]].dropna().astype(str).unique().tolist())
        label_values(df, mp)
        a, b, c = st.columns(3)
        a.dataframe(df[mp["target"]].astype(str).value_counts(dropna=False).rename("rows").to_frame())
        b.selectbox("Value meaning **Clean**", opts, key="p_clean_val")
        c.selectbox("Value meaning **Fouled**", opts, key="p_fouled_val")
        if st.session_state["p_clean_val"] == st.session_state["p_fouled_val"]:
            st.error("Clean and Fouled must be different values.")
    st.checkbox("I confirm these column mappings, units and label meanings", key="p_confirmed", disabled=bool(errs))


# ------------------------------------------------------------------ Section 2
def page_preprocess():
    st.header("Section 2 · Data Preprocessing")
    df, mp = require_mapping()
    st.subheader("Missing values and duplicates")
    miss = pd.DataFrame({"missing": df.isna().sum(), "missing %": (df.isna().mean() * 100).round(2)})
    a, b = st.columns(2)
    a.dataframe(miss, width="stretch")
    fig = px.bar(miss.reset_index(), x="index", y="missing %", title="Missing values per column (%)")
    fig.update_layout(xaxis_title="", height=330)
    b.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.metric("Duplicate rows", int(df.duplicated().sum()))
    c2.metric("Rows with ≥1 missing value", int(df.isna().any(axis=1).sum()))

    st.subheader("Cleaning configuration")
    x1, x2 = st.columns(2)
    x1.checkbox("Remove duplicate rows", key="p_dedup")
    x2.checkbox("Remove noisy rows (inconsistent temperatures, U ≤ 0)", key="p_noise")
    cand = [c for c in df.columns if c != mp["target"]]
    feature_list(df, mp)
    st.multiselect("Model features", cand, key="p_features")
    if not mp["target"]:
        st.error("Map the label column in Section 1 to build the cleaned dataset.")
        return
    clean, rep = get_clean(df, mp)
    st.subheader("Rows removed and rows available for analysis")
    k = st.columns(3)
    with k[0]:
        kpi("Original rows", f"{rep['Original rows']:,}")
    with k[1]:
        kpi("Rows removed (total)", f"{rep['Total rows removed']:,}", "", f"{100 * rep['Total rows removed'] / max(rep['Original rows'], 1):.1f}% of dataset", "warn")
    with k[2]:
        kpi("Rows available for analysis", f"{rep['Rows available for analysis']:,}", "", "used in Sections 3–5", "good")
    st.dataframe(pd.DataFrame({"Step": list(rep), "Rows": list(rep.values())}), hide_index=True, width="stretch")
    st.subheader("Cleaned dataset preview")
    st.dataframe(clean.head(100), width="stretch")
    if len(clean):
        _, f = label_values(df, mp)
        yy = (clean[mp["target"]].astype(str) == str(f)).astype(int).map({0: "Clean", 1: "Fouled"}).value_counts()
        st.write("Class counts after cleaning:", yy.to_dict())


# ------------------------------------------------------------------ Section 3
def page_engineering():
    st.header("Section 3 · Engineering Performance Analysis")
    df, mp = require_mapping()
    need_u_and_label(mp)
    ctx = analysis_context(df, mp)
    clean, n = ctx["clean"], ctx["n"]
    with st.expander("Threshold & severity settings (configurable)"):
        a, b, c, d = st.columns(4)
        a.number_input("Fouling-resistance threshold", min_value=0.0, step=0.001, format="%.5f", key="p_thr_val")
        b.selectbox("Threshold unit", list(E.RF_UNITS), key="p_thr_unit")
        c.number_input("Severity: Low→Medium at × threshold", min_value=0.01, step=0.5, key="p_sev_med")
        d.number_input("Severity: Medium→High at × threshold", min_value=0.02, step=0.5, key="p_sev_high")
    thr = thr_si()
    st.subheader("Select the range of rows to analyse (positions in the cleaned dataset)")
    r1, r2, r3 = st.columns([1, 1, 2])
    r1.number_input("From position", min_value=0, max_value=n - 1, step=1, key="p_start")
    r2.number_input("To position", min_value=0, max_value=n - 1, step=1, key="p_end")
    ctx = analysis_context(df, mp)
    r3.metric("Rows available after cleaning", f"{n:,}")
    ml = ensure_models(df, mp, clean)
    if ml is None:
        st.stop()
    model = primary_model(ml)
    uclean = baseline_u(clean, mp, df)
    if uclean is None:
        st.error("No Clean-labelled rows with valid U: the clean baseline cannot be computed.")
        st.stop()
    fr = build_range(df, mp, ctx, ml, model, uclean, thr)
    flagged = fr[fr["At/above"]]
    thr_kw = thr * 1e3
    k = st.columns(5)
    with k[0]:
        kpi("Rows analysed", f"{len(fr):,}", "", f"positions {ctx['a']}–{ctx['b']}")
    with k[1]:
        kpi("Uclean (median clean baseline)", f"{uclean:.1f}", "W/(m²·K)")
    with k[2]:
        kpi("Rows with Rf ≥ threshold", f"{len(flagged):,}", "", f"threshold {thr_kw:.4g} m²·K/kW", "bad" if len(flagged) else "good")
    with k[3]:
        kpi("Fouled rows (ML)", f"{int((fr['ML status'] == 'Fouled').sum()):,}", "", f"status by {model}", "bad")
    with k[4]:
        kpi("Clean rows (ML)", f"{int((fr['ML status'] == 'Clean').sum()):,}", "", "", "good")

    st.subheader("Row-by-row status")
    flt = st.radio("Show", ["Rows with Rf ≥ threshold", "All analysed rows", "Fouled (ML)", "Clean (ML)"], horizontal=True)
    show = {"Rows with Rf ≥ threshold": flagged, "All analysed rows": fr,
            "Fouled (ML)": fr[fr["ML status"] == "Fouled"], "Clean (ML)": fr[fr["ML status"] == "Clean"]}[flt]
    cols = ["Row", fr.columns[2], "Rf (m²·K/kW)", "Rf / threshold", "Rf status", "ML status", "P(Fouled)", "Actual label"]
    show = show[cols]
    st.dataframe(show.style.format({"Rf (m²·K/kW)": "{:.4f}", "Rf / threshold": "{:.2f}", "P(Fouled)": "{:.3f}", fr.columns[2]: "{:.1f}"}),
                 hide_index=True, width="stretch", height=380)
    st.download_button("Download this table (CSV)", show.to_csv(index=False).encode(), "row_status.csv", "text/csv")
    fig = px.scatter(fr, x="Row", y="Rf (m²·K/kW)", color="ML status", height=380,
                     color_discrete_map={"Fouled": "#d64545", "Clean": "#2e7bd6"}, title="Rf of every analysed row vs the threshold")
    fig.add_hline(y=thr_kw, line_dash="dash", annotation_text="threshold")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Engineering parameters over the analysed rows")
    rows = []
    sub = clean.iloc[ctx["a"]:ctx["b"] + 1]
    for role in ("hot_in", "hot_out", "cold_in", "cold_out", "hot_flow", "cold_flow", "hot_dp", "cold_dp", "q", "effectiveness", "u"):
        col = mp[role]
        if col:
            sr = pd.to_numeric(sub[col], errors="coerce")
            rows.append({"Parameter": D.ROLES[role][0], "Column": col, "Mean": sr.mean(), "Min": sr.min(), "Max": sr.max(),
                         "Unit": st.session_state["p_u_unit"] if role == "u" else role_unit(role)})
    rows.append({"Parameter": "Fouling threshold", "Column": "(setting)", "Mean": thr_kw, "Min": thr_kw, "Max": thr_kw, "Unit": "m²·K/kW"})
    st.dataframe(pd.DataFrame(rows).style.format({"Mean": "{:.4g}", "Min": "{:.4g}", "Max": "{:.4g}"}), hide_index=True, width="stretch")
    forecast_block(df, mp, ctx, uclean, thr)


def forecast_block(df, mp, ctx, uclean, thr):
    st.subheader("Rf forecast (LSTM)")
    clean, n, b = ctx["clean"], ctx["n"], ctx["b"]
    c = st.columns(5)
    c[0].slider("Smoothing window (rows)", 1, 100, key="p_lstm_smooth")
    c[1].slider("LSTM input window", 5, 30, key="p_lstm_win")
    c[2].slider("Hidden units", 8, 64, step=8, key="p_lstm_hidden")
    c[3].slider("Epochs", 10, 100, step=10, key="p_lstm_epochs")
    c[4].slider("Forecast horizon (steps)", 5, 100, key="p_lstm_horizon")
    if st.button("Train LSTM and forecast Rf"):
        lo, hi = max(0, b + 1 - 1500), b + 1
        if hi - lo < 60:
            lo, hi = 0, min(n, 300)
        sub = clean.iloc[lo:hi]
        rf = E.rf_series(pd.to_numeric(sub[mp["u"]], errors="coerce") * E.U_UNITS[st.session_state["p_u_unit"]], uclean) * 1e3
        ser = rf.rolling(st.session_state["p_lstm_smooth"], min_periods=1).mean().to_numpy()
        try:
            with st.spinner("Training LSTM…"):
                out = F.train_and_forecast(ser, st.session_state["p_lstm_win"], st.session_state["p_lstm_hidden"],
                                           st.session_state["p_lstm_epochs"], st.session_state["p_lstm_horizon"])
            out["lo"] = lo
            st.session_state["lstm"] = out
        except Exception as e:  # noqa: BLE001
            st.error(f"LSTM failed: {e}")
    o = st.session_state.get("lstm")
    if not o:
        return
    thr_kw = thr * 1e3
    h = len(o["history"])
    x_hist = np.arange(o["lo"], o["lo"] + h)
    x_fc = np.arange(o["lo"] + h, o["lo"] + h + len(o["forecast"]))
    fig = go.Figure()
    fig.add_scatter(x=x_hist, y=o["history"], name="Rf (smoothed)")
    fig.add_scatter(x=o["lo"] + o["val_start"] + np.arange(len(o["val_pred"])), y=o["val_pred"],
                    name="LSTM validation prediction")
    fig.add_scatter(x=x_fc, y=o["forecast"], name="LSTM forecast", line=dict(dash="dash", color="#d64545"))
    fig.add_hline(y=thr_kw, line_dash="dot", annotation_text="threshold")
    fig.update_layout(height=400, xaxis_title="Assumed time index (cleaned-row order)", yaxis_title="Rf (m²·K/kW)")
    st.plotly_chart(fig, width="stretch")
    cross = np.where(o["forecast"] >= thr_kw)[0]
    k = st.columns(4)
    with k[0]:
        kpi("Validation RMSE (LSTM)", fmt(o["rmse"], ".4g"), "m²·K/kW")
    with k[1]:
        kpi("Validation RMSE (naive)", fmt(o["naive_rmse"], ".4g"), "m²·K/kW")
    with k[2]:
        kpi("Forecast at last step", fmt(o["forecast"][-1], ".4g"), "m²·K/kW")
    with k[3]:
        kpi("Forecast vs threshold", "Above from step 1" if len(cross) and cross[0] == 0 else
            f"Crosses at step {cross[0] + 1}" if len(cross) else "Stays below", "", "", "bad" if len(cross) else "good")


# ------------------------------------------------------------------ Section 4
def page_ml():
    st.header("Section 4 · Fouling Prediction & Explainable AI")
    df, mp = require_mapping()
    need_u_and_label(mp)
    ctx = analysis_context(df, mp)
    clean = ctx["clean"]
    st.subheader("A. Training configuration")
    a, b, c3, d = st.columns(4)
    a.multiselect("Models", M.MODEL_NAMES, key="p_models")
    b.slider("Test fraction", 0.1, 0.5, step=0.05, key="p_test_size")
    c3.number_input("Random seed", min_value=0, step=1, key="p_seed")
    d.number_input("RF trees", min_value=50, max_value=1000, step=50, key="p_trees")
    split_opts = ["Stratified random"] + (["Time-ordered (latest rows = test)"] if mp["time"] else [])
    if st.session_state["p_split"] not in split_opts:
        st.session_state["p_split"] = split_opts[0]
    st.selectbox("Split strategy", split_opts, key="p_split")
    force = st.button("🚀 Train / re-train models", type="primary")
    ml = ensure_models(df, mp, clean, force=force)
    if ml is None:
        st.stop()
    st.caption(f"Trained on the {ctx['n']:,} cleaned rows · train {ml['n_train']:,} / test {ml['n_test']:,}")

    st.subheader("B. Model evaluation (held-out test set)")
    met = ml["metrics"].set_index("Model")
    st.dataframe(met.style.format("{:.3f}"), width="stretch")
    long = met.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(long, x="Metric", y="Score", color="Model", barmode="group", height=360), width="stretch")
    cols = st.columns(len(ml["cms"]))
    for col, (name, cm) in zip(cols, ml["cms"].items()):
        fig = px.imshow(cm, text_auto=True, x=["Pred Clean", "Pred Fouled"], y=["Actual Clean", "Actual Fouled"],
                        color_continuous_scale="Blues", title=name, height=300)
        fig.update_coloraxes(showscale=False)
        col.plotly_chart(fig, width="stretch")
    fr = go.Figure()
    for name, (fpr, tpr) in ml["rocs"].items():
        fr.add_scatter(x=fpr, y=tpr, name=name)
    fr.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="chance")
    fr.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate", height=360)
    st.plotly_chart(fr, width="stretch")

    model = primary_model(ml)
    uclean = baseline_u(clean, mp, df)
    rng = build_range(df, mp, ctx, ml, model, uclean, thr_si())
    st.subheader(f"C. Predictions for the analysed rows (positions {ctx['a']}–{ctx['b']})")
    pcols = [c for c in rng.columns if c.startswith("Pred · ")]
    rng["Models agree"] = rng[pcols].nunique(axis=1).eq(1).map({True: "Yes", False: "No"})
    cnt = pd.DataFrame({m[7:]: rng[m].value_counts() for m in pcols}).fillna(0).astype(int).rename_axis("Predicted")
    st.dataframe(cnt.reset_index(), hide_index=True, width="stretch")
    cl = cnt.reset_index().melt(id_vars="Predicted", var_name="Model", value_name="Rows")
    st.plotly_chart(px.bar(cl, x="Model", y="Rows", color="Predicted", barmode="group", height=320,
                           color_discrete_map={"Fouled": "#d64545", "Clean": "#2e7bd6"}), width="stretch")
    tbl = rng[["Row", "Actual label"] + pcols + ["P(Fouled)", "Models agree"]]
    st.dataframe(tbl.style.format({"P(Fouled)": "{:.3f}"}), hide_index=True, width="stretch", height=360)
    st.download_button("Download predictions (CSV)", tbl.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    st.subheader("D. Fouling severity from Rf (analysed rows)")
    thr = thr_si()
    st.markdown(f"Boundaries: **Low** < {st.session_state['p_sev_med'] * thr * 1e3:.4g} · **Medium** up to "
                f"{st.session_state['p_sev_high'] * thr * 1e3:.4g} · **High** above, in m²·K/kW")
    sv = rng["Severity"].value_counts().reindex(["Low", "Medium", "High", "n/a"]).fillna(0).astype(int).rename("Rows").reset_index()
    a1, b1 = st.columns(2)
    a1.dataframe(sv, hide_index=True, width="stretch")
    b1.plotly_chart(px.bar(sv, x="Severity", y="Rows", height=260), width="stretch")

    st.subheader("E. SHAP explainability (analysed rows)")
    names_ok = list(ml["models"])
    if st.session_state.get("p_shap_model") not in names_ok:
        st.session_state["p_shap_model"] = names_ok[0]
    st.selectbox("Model to explain", names_ok, key="p_shap_model")
    name = st.session_state["p_shap_model"]
    try:
        (gsv, gxt, gnames), rsub = range_shap(ml, name, clean, ctx["a"], ctx["b"])
    except Exception as e:  # noqa: BLE001
        st.error(f"SHAP failed for {name}: {e}")
        return
    imp = pd.DataFrame({"Feature": gnames, "mean |SHAP|": np.abs(gsv).mean(axis=0),
                        "mean SHAP (→ Fouled)": gsv.mean(axis=0)}).sort_values("mean |SHAP|")
    st.plotly_chart(px.bar(imp, x="mean |SHAP|", y="Feature", orientation="h", height=380,
                           title=f"Global feature importance — {name} ({len(rsub)} rows)"), width="stretch")
    try:
        import shap
        plt.figure()
        shap.summary_plot(gsv, gxt, feature_names=gnames, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"SHAP summary plot failed: {e}")


# ------------------------------------------------------------------ Section 5
def page_maintenance():
    st.header("Section 5 · Maintenance Support")
    df, mp = require_mapping()
    need_u_and_label(mp)
    ctx = analysis_context(df, mp)
    clean = ctx["clean"]
    ml = ensure_models(df, mp, clean)
    if ml is None:
        st.stop()
    primary_model(ml)
    st.selectbox("Primary model for the decision logic", list(ml["models"]), key="p_primary")
    model, thr = st.session_state["p_primary"], thr_si()
    uclean = baseline_u(clean, mp, df)
    fr = build_range(df, mp, ctx, ml, model, uclean, thr)
    st.subheader(f"Decisions for the analysed rows (positions {ctx['a']}–{ctx['b']}, {len(fr):,} rows)")
    cnt = fr["Decision case"].value_counts().rename_axis("Decision case").reset_index(name="Rows").sort_values("Decision case")
    k = st.columns(4)
    for col, key in zip(k, ["1", "3", "2", "4"]):
        with col:
            nm = R.CASE_NAMES[key]
            kpi(nm, f"{int((fr['Decision case'] == nm).sum()):,}", "rows", "", "bad" if key == "1" else "warn" if key in "23" else "good")
    a, b = st.columns(2)
    a.dataframe(cnt, hide_index=True, width="stretch")
    b.plotly_chart(px.bar(cnt, x="Decision case", y="Rows", height=300), width="stretch")
    flt = st.radio("Show rows", ["Needs attention (cases 1–3)", "All analysed rows", "Case 1 only"], horizontal=True)
    t = fr if flt == "All analysed rows" else fr[fr["Decision case"].str.startswith("1")] if flt == "Case 1 only" \
        else fr[fr["Decision case"].str[0].isin(["1", "2", "3"])]
    t = t[["Row", "Actual label", "ML status", "Rf (m²·K/kW)", "Rf status", "Severity", "Decision case", "Recommended action"]]
    st.dataframe(t.style.format({"Rf (m²·K/kW)": "{:.4f}"}), hide_index=True, width="stretch", height=420)
    st.download_button("Download decisions (CSV)", t.to_csv(index=False).encode(), "maintenance_decisions.csv", "text/csv")

    st.subheader(f"SHAP drivers for rows predicted Fouled ({model})")
    try:
        (gsv, gxt, gnames), rsub = range_shap(ml, model, clean, ctx["a"], ctx["b"])
        pf = ml["models"][model].predict_proba(rsub[ml["features"]])[:, 1] >= 0.5
        if pf.any():
            u = feature_units(mp)
            d = pd.DataFrame({"Feature": gnames, "Mean SHAP (→ Fouled)": gsv[pf].mean(axis=0),
                              "Mean actual value": [rsub.loc[pf, g].mean() if g in rsub.columns else np.nan for g in gnames],
                              "Unit": [u.get(g, "") for g in gnames]}).sort_values("Mean SHAP (→ Fouled)", ascending=False)
            st.dataframe(d.head(6).style.format({"Mean SHAP (→ Fouled)": "{:+.4f}", "Mean actual value": "{:.4g}"}),
                         hide_index=True, width="stretch")
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ main
st.sidebar.title("🔧 HX Fouling Dashboard")
st.sidebar.caption("Automotive oil-cooler-based, PHE-like prototype · hot: water · cold: coolant · CaCO₃ scaling focus")
st.sidebar.radio("Section", SECTIONS, key="p_section")
if st.session_state.get("df") is not None:
    st.sidebar.markdown(f"Dataset: {st.session_state['file_name']} ({len(st.session_state['df']):,} rows)")
    if st.session_state.get("p_confirmed"):
        try:
            _mp = mapping()
            if _mp["target"]:
                _, _rep = get_clean(st.session_state["df"], _mp)
                st.sidebar.markdown(f"Rows available after cleaning: {_rep['Rows available for analysis']:,}")
        except Exception:  # noqa: BLE001
            pass
PAGES = dict(zip(SECTIONS, [page_input, page_preprocess, page_engineering, page_ml, page_maintenance]))
try:
    PAGES[st.session_state["p_section"]]()
except Exception as _e:  # noqa: BLE001
    if type(_e).__name__ in ("StopException", "RerunException"):
        raise
    st.error(f"Unexpected error: {_e}")
    st.exception(_e)
