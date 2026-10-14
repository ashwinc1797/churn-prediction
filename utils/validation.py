"""Input validation shared by the single-record form and batch scoring."""
import pandas as pd

from schema import CATEGORICAL_FEATURES, CATEGORY_OPTIONS, FEATURES, NUMERIC_FEATURES
from schema import NUMERIC_RANGES


def validate_record(record: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors = []
    for col in FEATURES:
        if col not in record or record[col] is None:
            errors.append(f"'{col}' is required.")
    for col in NUMERIC_FEATURES:
        lo, hi = NUMERIC_RANGES[col]
        val = record.get(col)
        if val is not None and not lo <= val <= hi:
            errors.append(f"'{col}' must be between {lo} and {hi} (got {val}).")
    for col in CATEGORICAL_FEATURES:
        val = record.get(col)
        if val is not None and val not in CATEGORY_OPTIONS[col]:
            errors.append(f"'{col}' has unknown value '{val}'.")
    # business rule: total charges cannot be far below one month of charges
    if record.get("tenure_months", 0) >= 1 and record.get("total_charges") is not None:
        if record["total_charges"] < 0.5 * record.get("monthly_charges", 0):
            errors.append("Total charges look too low for the given tenure.")
    return errors


def validate_dataframe(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for an uploaded batch file."""
    errors, warnings = [], []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors, warnings
    if df.empty:
        errors.append("The file has no rows.")
        return errors, warnings
    for col in NUMERIC_FEATURES:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna() & df[col].notna()
        if bad.any():
            errors.append(f"Column '{col}' has {int(bad.sum())} non-numeric value(s).")
        lo, hi = NUMERIC_RANGES[col]
        out = ((converted < lo) | (converted > hi)).sum()
        if out:
            warnings.append(f"{int(out)} row(s) have '{col}' outside [{lo}, {hi}].")
    for col in CATEGORICAL_FEATURES:
        unknown = set(df[col].dropna().astype(str)) - set(CATEGORY_OPTIONS[col])
        if unknown:
            warnings.append(
                f"'{col}' contains unseen categories {sorted(unknown)[:5]}; the "
                "encoder will treat them as all-zero."
            )
    n_missing = int(df[FEATURES].isna().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing value(s) will be imputed by the pipeline.")
    extra = [c for c in df.columns if c not in FEATURES]
    if extra:
        warnings.append(f"Ignoring extra columns: {', '.join(extra[:8])}")
    return errors, warnings
