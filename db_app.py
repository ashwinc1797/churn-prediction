"""Churn predictions stored in a database.  Run: streamlit run db_app.py

Database URL resolution order (never hard-code credentials):
  1. st.secrets["database"]["url"]   (.streamlit/secrets.toml or cloud secrets UI)
  2. DATABASE_URL environment variable
  3. local SQLite file for development
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from schema import CATEGORICAL_FEATURES, CATEGORY_OPTIONS, FEATURE_LABELS, FEATURES
from schema import POSITIVE_CLASS
from utils.db import load_history, make_engine, save_feedback, save_prediction
from utils.model_io import load_trusted_model
from utils.validation import validate_record

ROOT = Path(__file__).parent
st.set_page_config(page_title="Churn + Database", layout="wide")


def database_url() -> str:
    try:
        return st.secrets["database"]["url"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'predictions.db'}")


@st.cache_resource
def get_engine():
    return make_engine(database_url())   # one connection pool per server process


@st.cache_resource
def get_model():
    return load_trusted_model(ROOT / "models" / "churn_pipeline.joblib",
                              ROOT / "models" / "churn_metadata.json")


model, meta, _ = get_model()
try:
    engine = get_engine()
except SQLAlchemyError as exc:
    st.error(f"Database unavailable: {exc.__class__.__name__}. Check the URL/secrets.")
    st.stop()
version = f"{meta['model_type']}-{meta['sha256'][:8]}"

st.title("Churn Predictions with History")
predict_tab, history_tab = st.tabs(["Predict", "History & feedback"])

with predict_tab:
    with st.form("predict"):
        c1, c2 = st.columns(2)
        rec = {
            "tenure_months": c1.number_input("Tenure (months)", 0, 72, 6),
            "monthly_charges": c1.number_input("Monthly charges", 18.0, 120.0, 89.0),
            "total_charges": c1.number_input("Total charges", 0.0, 9000.0, 530.0),
            "support_calls": c1.number_input("Support calls", 0, 10, 3),
        }
        for col in CATEGORICAL_FEATURES:
            rec[col] = c2.selectbox(FEATURE_LABELS[col], CATEGORY_OPTIONS[col])
        go = st.form_submit_button("Predict and save", type="primary")
    if go:
        problems = validate_record(rec)
        if problems:
            st.error(" ".join(problems))
        else:
            pos = meta["classes"].index(POSITIVE_CLASS)
            p = float(model.predict_proba(pd.DataFrame([rec])[FEATURES])[0, pos])
            label = "churn" if p >= 0.5 else "stay"
            try:
                pid = save_prediction(engine, rec, p, label, version)
                st.session_state.last_id = pid
                st.success(f"Prediction #{pid} saved: {label} ({p:.1%})")
            except SQLAlchemyError:
                st.warning("Prediction made but could not be saved to the database.")
                st.metric("Churn probability", f"{p:.1%}")
    if "last_id" in st.session_state:
        st.write(f"Was prediction #{st.session_state.last_id} useful?")
        rating = st.feedback("thumbs", key=f"fb_{st.session_state.last_id}")
        if rating is not None:
            save_feedback(engine, st.session_state.last_id, helpful=bool(rating))
            st.toast("Thanks for the feedback!")
            del st.session_state["last_id"]

with history_tab:
    f1, f2 = st.columns(2)
    label_filter = f1.selectbox("Filter by result", ["All", "churn", "stay"])
    limit = f2.number_input("Rows", 10, 1000, 100, step=10)
    try:
        hist = load_history(engine, None if label_filter == "All" else label_filter,
                            limit)
    except SQLAlchemyError as exc:
        st.error(f"Could not read history: {exc.__class__.__name__}")
        hist = pd.DataFrame()
    if hist.empty:
        st.info("No predictions stored yet.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Stored predictions", len(hist))
        k2.metric("Predicted churn", int((hist["label"] == "churn").sum()))
        rated = hist["helpful"].dropna()
        k3.metric("Helpful rating", f"{rated.mean():.0%}" if len(rated) else "n/a")
        st.dataframe(hist, hide_index=True, width="stretch")
        st.download_button("Export history", hist.to_csv(index=False), "history.csv")
