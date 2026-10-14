"""Generate CSV, Excel, PDF and JSON reports from batch predictions.

Run: streamlit run report_app.py   (extra deps: openpyxl reportlab matplotlib)
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from schema import FEATURES, ID_COLUMN, POSITIVE_CLASS
from utils.model_io import load_trusted_model
from utils.reports import (risk_chart_png, to_csv_bytes, to_excel_bytes, to_json_bytes,
                           to_pdf_bytes)
from utils.validation import validate_dataframe

ROOT = Path(__file__).parent
st.set_page_config(page_title="Churn Reports", layout="wide")


@st.cache_resource
def get_model():
    return load_trusted_model(ROOT / "models" / "churn_pipeline.joblib",
                              ROOT / "models" / "churn_metadata.json")


@st.cache_data(show_spinner="Building reports...")
def build_reports(scored: pd.DataFrame, metrics: dict) -> dict:
    """Cached: reruns caused by clicking a download button do not rebuild files."""
    summary = (scored.groupby("contract")["churn_probability"]
               .agg(customers="count", mean_probability="mean")
               .assign(predicted_churners=scored.groupby("contract")["prediction"]
                       .apply(lambda s: (s == "Yes").sum()))
               .round(3))
    overall = pd.DataFrame({"value": {
        "customers": len(scored),
        "predicted_churners": int((scored["prediction"] == "Yes").sum()),
        "mean_probability": round(float(scored["churn_probability"].mean()), 4)}})
    top = (scored.nlargest(10, "churn_probability")
           [[ID_COLUMN, "contract", "tenure_months", "churn_probability"]])
    pdf_summary = summary.reset_index()
    chart = risk_chart_png(scored["churn_probability"])
    return {
        "csv": to_csv_bytes(scored),
        "xlsx": to_excel_bytes({"Overview": overall, "By contract": summary,
                                "Top risk": top.set_index(ID_COLUMN)}),
        "pdf": to_pdf_bytes("Churn Prediction Report", pdf_summary, metrics, chart, top),
        "json": to_json_bytes({"overview": overall["value"].to_dict(),
                               "by_contract": summary.reset_index().to_dict("records"),
                               "model_metrics": metrics}),
        "chart": chart,
    }


model, meta, _ = get_model()
pos = meta["classes"].index(POSITIVE_CLASS)
st.title("Churn Prediction Reports")
up = st.file_uploader("Customers to score (CSV)", type=["csv"])
df = pd.read_csv(up) if up else pd.read_csv(ROOT / "data" / "new_customers.csv")
errors, _ = validate_dataframe(df)
if errors:
    st.error("; ".join(errors))
    st.stop()

scored = df.copy()
scored["churn_probability"] = model.predict_proba(df[FEATURES])[:, pos].round(4)
scored["prediction"] = (scored["churn_probability"] >= 0.5).map({True: "Yes", False: "No"})
files = build_reports(scored, meta["metrics"])

c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(scored))
c2.metric("Predicted churners", int((scored["prediction"] == "Yes").sum()))
c3.metric("Model ROC-AUC (test)", f"{meta['metrics']['roc_auc']:.3f}")
st.image(files["chart"], caption="Risk distribution (also embedded in the PDF)")

st.subheader("Downloads")
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Predictions (CSV)", files["csv"], "churn_predictions.csv", "text/csv",
                   on_click="ignore", icon=":material/table:")
d2.download_button("Summary (Excel)", files["xlsx"], "churn_summary.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                   on_click="ignore", icon=":material/grid_on:")
d3.download_button("Report (PDF)", files["pdf"], "churn_report.pdf", "application/pdf",
                   on_click="ignore", icon=":material/picture_as_pdf:", type="primary")
d4.download_button("Results (JSON)", files["json"], "churn_results.json",
                   "application/json", on_click="ignore", icon=":material/data_object:")
