"""Smoke tests for the churn app. Run from the project root:  pytest -q

AppTest.from_file() resolves paths relative to THIS test file, hence "../app.py".
"""
from streamlit.testing.v1 import AppTest


def test_app_loads_without_errors():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Customer Churn Prediction"


def test_single_prediction_shows_probability():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.number_input[0].set_value(2)            # short tenure
    at.number_input[2].set_value(150.0)        # total charges
    at.button[0].click().run()                 # form submit button
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Churn probability" in labels
    assert len(at.session_state.history) == 1
