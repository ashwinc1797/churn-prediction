"""Batch scoring app for the churn model.  Run: streamlit run batch_app.py

Training data  (data/churn.csv)          -> used ONLY by train_model.py
Inference data (e.g. new_customers.csv)  -> uploaded here; no target column needed
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from schema import FEATURES, ID_COLUMN, NUMERIC_FEATURES, POSITIVE_CLASS
from utils.model_io import ModelLoadError, load_trusted_model
from utils.validation import validate_dataframe

ROOT = Path(__file__).parent
ALLOWED = ["csv", "xlsx"]
MAX_ROWS = 200_000
CHUNK = 5_000

st.set_page_config(page_title="Batch Churn Scoring", layout="wide")


@st.cache_resource
def get_model():
    return load_trusted_model(ROOT / "models" / "churn_pipeline.joblib",
                              ROOT / "models" / "churn_metadata.json")


@st.cache_data(show_spinner="Reading file...")
def read_upload(name: str, data: bytes) -> pd.DataFrame:
    """Cached on (name, bytes) so re-runs do not re-parse the same file."""
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data))           # needs openpyxl
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file. Save it as UTF-8 CSV and retry.")


def score_in_chunks(model, X: pd.DataFrame, pos_idx: int) -> np.ndarray:
    """Predict in chunks so we can show progress and keep memory use flat."""
    bar = st.progress(0.0, text="Scoring...")
    out = np.empty(len(X))
    for start in range(0, len(X), CHUNK):
        stop = min(start + CHUNK, len(X))
        out[start:stop] = model.predict_proba(X.iloc[start:stop])[:, pos_idx]
        bar.progress(stop / len(X), text=f"Scored {stop:,} of {len(X):,} rows")
    bar.empty()
    return out


try:
    model, meta, _ = get_model()
except ModelLoadError as exc:
    st.error(str(exc))
    st.stop()
pos_idx = meta["classes"].index(POSITIVE_CLASS)

st.title("Batch Churn Scoring")
st.markdown("**Step 1.** Upload new customers (no `churn` column required). "
            "Required columns:")
st.code(", ".join(FEATURES), language=None)
upload = st.file_uploader("Customer file", type=ALLOWED)
if upload is None:
    st.stop()

# ---- step 2: read + validate
try:
    df = read_upload(upload.name, upload.getvalue())
except Exception as exc:
    st.error(f"Could not read {upload.name}: {exc}")
    st.stop()
if len(df) > MAX_ROWS:
    st.error(f"File has {len(df):,} rows; the limit is {MAX_ROWS:,}. Split it first.")
    st.stop()

st.markdown("**Step 2.** Check the data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))
c4.metric("Duplicate rows", int(df.duplicated().sum()))
st.dataframe(df.head(10), width="stretch")

with st.expander("Column-level quality report"):
    report = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                           "unique": df.nunique()})
    report["required"] = report.index.isin(FEATURES)
    st.dataframe(report, width="stretch")

errors, warns = validate_dataframe(df)
if "churn" in df.columns:
    warns.append("A 'churn' column is present. It is ignored here; to measure accuracy "
                 "use the evaluation dashboard instead.")
if ID_COLUMN in df.columns and df[ID_COLUMN].duplicated().any():
    warns.append(f"Duplicate {ID_COLUMN} values found; check for repeated customers.")
for w in warns:
    st.warning(w)
for e in errors:
    st.error(e)
if errors:
    st.stop()

# ---- step 3: predict
st.markdown("**Step 3.** Score")
threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
if not st.button("Run batch prediction", type="primary"):
    st.stop()

X = df[FEATURES].copy()
for col in NUMERIC_FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")       # pipeline imputes NaNs
proba = score_in_chunks(model, X, pos_idx)
result = df.copy()
result["churn_probability"] = proba.round(4)
result["churn_prediction"] = np.where(proba >= threshold, "Yes", "No")
result["risk_band"] = pd.cut(proba, [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"],
                             include_lowest=True)

# ---- step 4: summarise + download
st.markdown("**Step 4.** Results")
k1, k2, k3 = st.columns(3)
k1.metric("Predicted churners", int((result["churn_prediction"] == "Yes").sum()))
k2.metric("Churn rate (predicted)", f"{(proba >= threshold).mean():.1%}")
k3.metric("Mean probability", f"{proba.mean():.3f}")
left, right = st.columns(2)
left.bar_chart(result["risk_band"].value_counts().sort_index())
right.dataframe(result.groupby("contract", observed=True)["churn_probability"]
                .describe()[["count", "mean", "min", "max"]].round(3), width="stretch")
st.dataframe(result.head(500), width="stretch")
st.download_button("Download predictions (CSV)", result.to_csv(index=False).encode("utf-8"),
                   file_name=f"scored_{Path(upload.name).stem}.csv", mime="text/csv",
                   type="primary", on_click="ignore")
