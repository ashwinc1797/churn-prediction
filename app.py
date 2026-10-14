"""Customer Churn Prediction app.  Run:  streamlit run app.py"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from schema import CATEGORICAL_FEATURES, CATEGORY_OPTIONS, FEATURE_LABELS, FEATURES
from schema import NUMERIC_RANGES, POSITIVE_CLASS
from utils.charts import (confusion_matrix_fig, effects_fig, importance_fig, pr_fig,
                          roc_fig)
from utils.explain import perturbation_effects
from utils.model_io import ModelLoadError, load_trusted_model
from utils.validation import validate_dataframe, validate_record

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
META_PATH = ROOT / "models" / "churn_metadata.json"
TEST_PRED_PATH = ROOT / "models" / "churn_test_predictions.csv"

st.set_page_config(page_title="Churn Predictor", page_icon=":material/insights:",
                   layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Loaded once per server process and shared by every user session."""
    return load_trusted_model(MODEL_PATH, META_PATH)


@st.cache_data
def get_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PRED_PATH)


try:
    model, meta, load_warnings = get_model()
except ModelLoadError as exc:
    st.error(f"The model could not be loaded. {exc}")
    st.stop()

POS_IDX = meta["classes"].index(POSITIVE_CLASS)
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    st.caption(f"{meta['model_type']} | trained {meta['trained_at'][:10]}")
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, float(meta["default_threshold"]), 0.05,
        help="Customers with churn probability at or above this value are flagged.",
    )
    st.metric("Test ROC-AUC", f"{meta['metrics']['roc_auc']:.3f}")
    for w in load_warnings:
        st.warning(w)

st.title("Customer Churn Prediction")
st.caption("Estimate the probability that a customer cancels their subscription.")

tab_single, tab_batch, tab_perf, tab_about = st.tabs(
    ["Single prediction", "Batch scoring", "Model performance", "About the model"]
)

# ---------------------------------------------------------------- single
with tab_single:
    with st.form("customer_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.number_input(FEATURE_LABELS["tenure_months"],
                                     *NUMERIC_RANGES["tenure_months"], value=12, step=1)
            monthly = st.number_input(FEATURE_LABELS["monthly_charges"],
                                      *NUMERIC_RANGES["monthly_charges"], value=70.0,
                                      step=0.5)
            total = st.number_input(FEATURE_LABELS["total_charges"],
                                    *NUMERIC_RANGES["total_charges"], value=840.0,
                                    step=10.0)
            calls = st.slider(FEATURE_LABELS["support_calls"],
                              *NUMERIC_RANGES["support_calls"], value=1)
        with c2:
            cats = {c: st.selectbox(FEATURE_LABELS[c], CATEGORY_OPTIONS[c])
                    for c in CATEGORICAL_FEATURES}
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        record = {"tenure_months": int(tenure), "monthly_charges": float(monthly),
                  "total_charges": float(total), "support_calls": int(calls), **cats}
        problems = validate_record(record)
        if problems:
            for p in problems:
                st.error(p)
        else:
            X = pd.DataFrame([record])[FEATURES]
            proba = float(model.predict_proba(X)[0, POS_IDX])
            label = "Likely to churn" if proba >= threshold else "Likely to stay"
            m1, m2, m3 = st.columns(3)
            m1.metric("Churn probability", f"{proba:.1%}")
            m2.metric("Prediction", label)
            m3.metric("Threshold", f"{threshold:.2f}")
            st.progress(proba, text=f"Risk score {proba:.2f}")
            if proba >= threshold:
                st.warning("High risk: consider a retention offer or proactive call.")
            else:
                st.success("Low risk at the current threshold.")

            effects = perturbation_effects(model, record, meta["baseline"], FEATURES,
                                           POS_IDX)
            with st.expander("Why this prediction? (approximate explanation)"):
                if effects.empty:
                    st.info("This customer matches the typical profile on every feature.")
                else:
                    st.plotly_chart(effects_fig(effects), width="stretch")
                    st.caption("Bars show how the probability changes when one feature "
                               "is reset to its typical training value. Red = raises "
                               "risk, green = lowers risk.")
                    st.dataframe(effects, hide_index=True, width="stretch")

            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"), **record,
                "churn_probability": round(proba, 4), "prediction": label,
            })

    if st.session_state.history:
        st.subheader("Predictions in this session")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, hide_index=True, width="stretch")
        st.download_button("Download history (CSV)", hist.to_csv(index=False),
                           "prediction_history.csv", "text/csv")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with these columns:", ", ".join(f"`{c}`" for c in FEATURES))
    upload = st.file_uploader("Customer file", type=["csv"], key="batch_file")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            errors, warns = validate_dataframe(df)
        except Exception as exc:  # malformed CSV, wrong encoding, etc.
            df, errors, warns = None, [f"Could not read the CSV file: {exc}"], []
        for w in warns:
            st.warning(w)
        if errors:
            for e in errors:
                st.error(e)
        else:
            X = df[FEATURES].copy()
            for col in ("tenure_months", "monthly_charges", "total_charges",
                        "support_calls"):
                X[col] = pd.to_numeric(X[col], errors="coerce")
            with st.spinner(f"Scoring {len(X):,} customers..."):
                probs = model.predict_proba(X)[:, POS_IDX]
            result = df.copy()
            result["churn_probability"] = probs.round(4)
            result["prediction"] = (probs >= threshold).astype(int)
            result["prediction"] = result["prediction"].map({1: "Yes", 0: "No"})
            k1, k2, k3 = st.columns(3)
            k1.metric("Customers scored", f"{len(result):,}")
            k2.metric("Flagged as churn", int((result["prediction"] == "Yes").sum()))
            k3.metric("Average probability", f"{probs.mean():.1%}")
            st.dataframe(
                result.sort_values("churn_probability", ascending=False),
                hide_index=True, width="stretch",
                column_config={"churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.2f")},
            )
            st.download_button("Download scored file", result.to_csv(index=False),
                               "churn_predictions.csv", "text/csv", type="primary")

# ---------------------------------------------------------------- performance
with tab_perf:
    st.caption(f"Metrics on a held-out test set of {meta['n_test']:,} customers.")
    cols = st.columns(5)
    for col, (name, value) in zip(cols, meta["metrics"].items()):
        col.metric(name.replace("_", " ").upper(), f"{value:.3f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.plotly_chart(confusion_matrix_fig(meta["confusion_matrix"], meta["classes"]),
                        width="stretch")
    with right:
        st.subheader("Feature importance")
        st.plotly_chart(importance_fig(meta["feature_importance"]), width="stretch")
    preds = get_test_predictions()
    y_bin = (preds["y_true"] == POSITIVE_CLASS).astype(int)
    left, right = st.columns(2)
    with left:
        st.subheader("ROC curve")
        st.plotly_chart(roc_fig(y_bin, preds["y_proba"], meta["metrics"]["roc_auc"]),
                        width="stretch")
    with right:
        st.subheader("Precision-recall curve")
        st.plotly_chart(pr_fig(y_bin, preds["y_proba"]), width="stretch")
    st.subheader("Classification report")
    report = pd.DataFrame(meta["classification_report"]).T.round(3)
    st.dataframe(report, width="stretch")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        f"""
**Model:** {meta['model_type']} inside a scikit-learn `Pipeline`
(median/most-frequent imputation, scaling, one-hot encoding).

**Training rows:** {meta['n_train']:,} | **scikit-learn:** {meta['sklearn_version']}
| **Fingerprint:** `{meta['sha256'][:16]}...`

**Intended use:** prioritise customers for retention outreach. A prediction is a
risk estimate, not a fact about a customer; review flagged accounts before acting.
"""
    )
    st.json({"features": meta["features"], "classes": meta["classes"]}, expanded=False)
