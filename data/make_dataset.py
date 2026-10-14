"""Generate a synthetic telecom churn dataset with a realistic schema.

Run from the project root:  python data/make_dataset.py
The output data/churn.csv has the same column names as schema.py, so you can
replace it with a real dataset later as long as you keep those columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000


def make_churn_data(n: int = N) -> pd.DataFrame:
    contract = RNG.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = RNG.choice(["Fiber optic", "DSL", "No"], n, p=[0.45, 0.35, 0.20])
    payment = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n,
        p=[0.35, 0.2, 0.22, 0.23],
    )
    paperless = RNG.choice(["Yes", "No"], n, p=[0.6, 0.4])
    senior = RNG.choice(["No", "Yes"], n, p=[0.84, 0.16])
    tenure = np.clip(RNG.gamma(2.0, 14.0, n).round(), 0, 72).astype(int)
    base = np.where(internet == "Fiber optic", 85, np.where(internet == "DSL", 55, 25))
    monthly = np.clip(base + RNG.normal(0, 12, n), 18, 120).round(2)
    total = np.clip(monthly * tenure * RNG.uniform(0.9, 1.05, n), 0, 9000).round(2)
    calls = np.clip(RNG.poisson(1.6, n), 0, 10)

    logit = (
        -1.4
        + 1.3 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic")
        + 0.6 * (payment == "Electronic check")
        + 0.3 * (paperless == "Yes")
        + 0.4 * (senior == "Yes")
        - 0.045 * tenure
        + 0.012 * (monthly - 60)
        + 0.35 * calls
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(RNG.random(n) < prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": calls,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "senior_citizen": senior,
            "churn": churn,
        }
    )
    # a few realistic missing values for the imputers to handle
    missing_idx = RNG.choice(n, 40, replace=False)
    df.loc[missing_idx, "total_charges"] = np.nan
    return df


if __name__ == "__main__":
    out = Path(__file__).parent / "churn.csv"
    data = make_churn_data()
    data.to_csv(out, index=False)
    # an unlabeled file that mimics new customers arriving for scoring
    new = data.drop(columns=["churn"]).sample(200, random_state=7)
    new.to_csv(Path(__file__).parent / "new_customers.csv", index=False)
    print(f"Saved {len(data):,} rows to {out}")
    print(data["churn"].value_counts(normalize=True).round(3))
