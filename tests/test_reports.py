import io

import pandas as pd
from pypdf import PdfReader

from utils.reports import risk_chart_png, to_excel_bytes, to_pdf_bytes


def test_excel_roundtrip():
    data = to_excel_bytes({"Summary": pd.DataFrame({"value": [1, 2]})})
    assert pd.read_excel(io.BytesIO(data)).shape == (2, 2)


def test_pdf_is_valid():
    pdf = to_pdf_bytes("T", pd.DataFrame({"a": [1]}), {"auc": 0.8},
                       risk_chart_png(pd.Series([0.1, 0.6, 0.9])),
                       pd.DataFrame({"b": [2]}))
    assert pdf.startswith(b"%PDF") and len(PdfReader(io.BytesIO(pdf)).pages) >= 1
