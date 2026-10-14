"""Single source of truth for the churn model's input schema.

Both train_model.py and app.py import from this file, so the columns used
at training time are guaranteed to match the columns sent at inference time.
"""

TARGET = "churn"
POSITIVE_CLASS = "Yes"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "senior_citizen",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_OPTIONS = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["Fiber optic", "DSL", "No"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "paperless_billing": ["Yes", "No"],
    "senior_citizen": ["No", "Yes"],
}

# (min, max) accepted by the UI and by batch validation
NUMERIC_RANGES = {
    "tenure_months": (0, 72),
    "monthly_charges": (18.0, 120.0),
    "total_charges": (0.0, 9000.0),
    "support_calls": (0, 10),
}

FEATURE_LABELS = {
    "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges (USD)",
    "total_charges": "Total charges (USD)",
    "support_calls": "Support calls (last 6 months)",
    "contract": "Contract type",
    "internet_service": "Internet service",
    "payment_method": "Payment method",
    "paperless_billing": "Paperless billing",
    "senior_citizen": "Senior citizen",
}
