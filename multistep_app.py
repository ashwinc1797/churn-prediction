"""Five-step prediction wizard with callbacks.  Run: streamlit run multistep_app.py

Step 1 enter -> Step 2 validate -> Step 3 predict -> Step 4 result -> Step 5 report
The current step and all answers live in st.session_state, so they survive reruns.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from schema import CATEGORY_OPTIONS, FEATURE_LABELS, FEATURES, NUMERIC_RANGES
from schema import POSITIVE_CLASS
from utils.model_io import load_trusted_model
from utils.validation import validate_record

ROOT = Path(__file__).parent
STEPS = ["Customer info", "Validate", "Predict", "Result", "Report"]
DEFAULTS = {"tenure_months": 12, "monthly_charges": 70.0, "total_charges": 840.0,
            "support_calls": 0, "contract": "Month-to-month",
            "internet_service": "Fiber optic", "payment_method": "Electronic check",
            "paperless_billing": "Yes", "senior_citizen": "No"}

st.set_page_config(page_title="Churn Wizard")


@st.cache_resource
def get_model():
    return load_trusted_model(ROOT / "models" / "churn_pipeline.joblib",
                              ROOT / "models" / "churn_metadata.json")


# ------------------------------------------------------------ state + callbacks
def init_state():
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("record", dict(DEFAULTS))
    st.session_state.setdefault("result", None)


def go(step: int):
    st.session_state.step = step


def save_step1():
    """on_click callback: copy widget values into the record BEFORE the rerun."""
    rec = st.session_state.record
    for key in FEATURES:
        rec[key] = st.session_state[f"w_{key}"]
    rec["call_reason"] = st.session_state.get("w_call_reason")
    st.session_state.result = None          # inputs changed -> old result is stale
    go(1)


def run_prediction():
    model, meta, _ = get_model()
    X = pd.DataFrame([st.session_state.record])[FEATURES]
    pos = meta["classes"].index(POSITIVE_CLASS)
    proba = float(model.predict_proba(X)[0, pos])
    st.session_state.result = {"probability": proba,
                               "label": "Likely to churn" if proba >= 0.5 else
                               "Likely to stay",
                               "model": meta["model_type"],
                               "timestamp": datetime.now().isoformat(timespec="seconds")}
    go(3)


def reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def sync_total():
    """Dependent widget: suggest total charges = tenure x monthly charge."""
    t = st.session_state.w_tenure_months
    m = st.session_state.w_monthly_charges
    st.session_state.w_total_charges = round(min(t * m, 9000.0), 2)


init_state()
rec = st.session_state.record
st.title("Churn Prediction Wizard")
st.progress((st.session_state.step + 1) / len(STEPS),
            text=f"Step {st.session_state.step + 1} of {len(STEPS)}: "
                 f"{STEPS[st.session_state.step]}")
st.button("Start over", on_click=reset, icon=":material/restart_alt:")

step = st.session_state.step
if step == 0:
    # Seed widget keys from the saved record. Widget state is discarded when a widget
    # is not drawn (steps 2-5), so we restore it here. Do NOT also pass value=/index=:
    # setting a widget both ways triggers a Streamlit warning.
    for key in FEATURES:
        st.session_state.setdefault(f"w_{key}", rec[key])
    c1, c2 = st.columns(2)
    c1.number_input(FEATURE_LABELS["tenure_months"], *NUMERIC_RANGES["tenure_months"],
                    key="w_tenure_months", on_change=sync_total)
    c1.number_input(FEATURE_LABELS["monthly_charges"], *NUMERIC_RANGES["monthly_charges"],
                    key="w_monthly_charges", on_change=sync_total)
    c1.number_input(FEATURE_LABELS["total_charges"], *NUMERIC_RANGES["total_charges"],
                    key="w_total_charges",
                    help="Auto-filled from tenure x monthly charge; you can edit it.")
    c1.number_input(FEATURE_LABELS["support_calls"], *NUMERIC_RANGES["support_calls"],
                    key="w_support_calls")
    for col in ("contract", "internet_service", "payment_method", "paperless_billing",
                "senior_citizen"):
        c2.selectbox(FEATURE_LABELS[col], CATEGORY_OPTIONS[col], key=f"w_{col}")
    # conditional UI: only ask for a reason when the customer called support
    if st.session_state.w_support_calls > 0:
        st.selectbox("Main reason for the last support call",
                     ["Billing", "Network quality", "Device", "Other"],
                     key="w_call_reason")
    st.button("Next: validate", type="primary", on_click=save_step1)

elif step == 1:
    problems = validate_record(rec)
    st.subheader("Check the details")
    st.table(pd.DataFrame({"value": {FEATURE_LABELS[k]: str(rec[k]) for k in FEATURES}}))
    if problems:
        for p in problems:
            st.error(p)
        st.button("Back to edit", on_click=go, args=(0,))
    else:
        st.success("All inputs are valid.")
        b1, b2 = st.columns(2)
        b1.button("Back", on_click=go, args=(0,))
        b2.button("Next: predict", type="primary", on_click=go, args=(2,))

elif step == 2:
    st.write("Ready to score this customer with the trained model.")
    b1, b2 = st.columns(2)
    b1.button("Back", on_click=go, args=(1,))
    b2.button("Generate prediction", type="primary", on_click=run_prediction)

elif step == 3:
    res = st.session_state.result
    st.metric("Churn probability", f"{res['probability']:.1%}", res["label"],
              delta_color="inverse" if res["probability"] >= 0.5 else "normal")
    st.caption(f"Model: {res['model']} | scored at {res['timestamp']}")
    b1, b2 = st.columns(2)
    b1.button("Change inputs", on_click=go, args=(0,))
    b2.button("Next: report", type="primary", on_click=go, args=(4,))

else:
    res = st.session_state.result
    report = {"inputs": {k: rec[k] for k in FEATURES}, "result": res,
              "notes": "Risk estimate for retention prioritisation; not a decision."}
    st.json(report)
    lines = [f"# Churn prediction report ({res['timestamp']})", "",
             f"**Probability:** {res['probability']:.1%} | **Result:** {res['label']}", "",
             "| Feature | Value |", "|---|---|"]
    lines += [f"| {FEATURE_LABELS[k]} | {rec[k]} |" for k in FEATURES]
    c1, c2 = st.columns(2)
    c1.download_button("Download JSON", json.dumps(report, indent=2),
                       "churn_report.json", "application/json", on_click="ignore")
    c2.download_button("Download Markdown", "\n".join(lines), "churn_report.md",
                       "text/markdown", on_click="ignore")
    st.button("Score another customer", on_click=reset, type="primary")
