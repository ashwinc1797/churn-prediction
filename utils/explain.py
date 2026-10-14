"""A simple, model-agnostic local explanation.

For each feature we replace the customer's value with a 'typical' baseline value
(median for numbers, most frequent category for text) and measure how much the
predicted churn probability changes. Large positive numbers mean the customer's
actual value pushes the risk UP. This is an approximation, not a SHAP value.
"""
import pandas as pd


def perturbation_effects(model, record: dict, baseline: dict, features: list,
                         pos_idx: int) -> pd.DataFrame:
    row = pd.DataFrame([record])[features]
    p_actual = model.predict_proba(row)[0, pos_idx]
    rows = []
    for col in features:
        if record[col] == baseline[col]:
            continue
        modified = row.copy()
        modified[col] = baseline[col]
        p_base = model.predict_proba(modified)[0, pos_idx]
        rows.append({
            "feature": col,
            "customer_value": record[col],
            "typical_value": baseline[col],
            "effect": round(float(p_actual - p_base), 4),
        })
    effects = pd.DataFrame(rows, columns=["feature", "customer_value",
                                          "typical_value", "effect"])
    return effects.sort_values("effect", key=abs, ascending=False)
