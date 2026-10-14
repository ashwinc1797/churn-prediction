import io

import pandas as pd

from schema import FEATURES


def score_large_csv(file_bytes: bytes, model, pos_idx: int, chunksize: int = 50_000):
    """Stream a large CSV through the model without loading it all at once."""
    out = io.StringIO()
    reader = pd.read_csv(io.BytesIO(file_bytes), chunksize=chunksize)
    for i, chunk in enumerate(reader):
        chunk["churn_probability"] = model.predict_proba(chunk[FEATURES])[:, pos_idx]
        chunk.to_csv(out, index=False, header=(i == 0))
    return out.getvalue().encode("utf-8")
