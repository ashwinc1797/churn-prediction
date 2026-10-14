"""Streamlit front-end that calls the FastAPI service.  Run: streamlit run api_client_app.py

.streamlit/secrets.toml (never commit this file):
    [api]
    url = "http://localhost:8000"
    key = "change-me"
"""
import requests
import streamlit as st

from schema import CATEGORY_OPTIONS

st.set_page_config(page_title="Churn (API client)")
API_URL = st.secrets["api"]["url"].rstrip("/")
HEADERS = {"X-API-Key": st.secrets["api"]["key"]}


@st.cache_data(ttl=30)
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


st.title("Churn Prediction via REST API")
health = api_health()
if health is None:
    st.error(f"Model service at {API_URL} is not reachable.")
    st.stop()
st.caption(f"Connected to {API_URL} | model {health['model']}")

with st.form("api_form"):
    payload = {
        "tenure_months": st.number_input("Tenure (months)", 0, 72, 6),
        "monthly_charges": st.number_input("Monthly charges", 18.0, 120.0, 89.0),
        "total_charges": st.number_input("Total charges", 0.0, 9000.0, 530.0),
        "support_calls": st.number_input("Support calls", 0, 10, 3),
        **{c: st.selectbox(c, opts) for c, opts in CATEGORY_OPTIONS.items()},
    }
    send = st.form_submit_button("Predict", type="primary")

if send:
    try:
        r = requests.post(f"{API_URL}/predict", json=[payload], headers=HEADERS,
                          timeout=10)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
    else:
        if r.status_code == 200:
            res = r.json()[0]
            st.metric("Churn probability", f"{res['churn_probability']:.1%}", res["label"])
            st.caption(f"Model version {res['model_version']}")
        elif r.status_code == 401:
            st.error("The API rejected the key. Check st.secrets['api']['key'].")
        elif r.status_code == 422:
            st.error(f"Validation error from API: {r.json()['detail']}")
        else:
            st.error(f"API error {r.status_code}")
