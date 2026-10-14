"""Build downloadable CSV / Excel / PDF / JSON reports entirely in memory.

Using io.BytesIO instead of temporary files means nothing is left on the server's
disk, and concurrent users never overwrite each other's files.
Requires: openpyxl (Excel), reportlab (PDF), matplotlib (chart image).
"""
import io
import json
from datetime import datetime

import matplotlib
import pandas as pd

matplotlib.use("Agg")                      # headless backend for servers
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """One worksheet per DataFrame; sheet names are truncated to Excel's 31 chars."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=True)
            ws = writer.sheets[name[:31]]
            for col_cells in ws.columns:          # simple auto-width
                width = max(len(str(c.value or "")) for c in col_cells) + 2
                ws.column_dimensions[col_cells[0].column_letter].width = min(width, 40)
    return buffer.getvalue()


def risk_chart_png(probabilities: pd.Series) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=150)
    ax.hist(probabilities, bins=20, color="#1f4e8c", edgecolor="white")
    ax.axvline(0.5, color="#f28c28", linestyle="--", label="threshold 0.5")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Customers")
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)                            # free memory on long-running servers
    return buf.getvalue()


def _table(df: pd.DataFrame) -> Table:
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1f3a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
    ]))
    return t


def to_pdf_bytes(title: str, summary: pd.DataFrame, metrics: dict,
                 chart_png: bytes, top_risk: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            title=title)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Generated {datetime.now():%d %b %Y %H:%M}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Prediction summary", styles["Heading2"]), _table(summary),
        Spacer(1, 10),
        Paragraph("Model evaluation (held-out test set)", styles["Heading2"]),
        _table(pd.DataFrame({"metric": list(metrics),
                             "value": [f"{v:.3f}" for v in metrics.values()]})),
        Spacer(1, 10),
        Paragraph("Risk distribution", styles["Heading2"]),
        Image(io.BytesIO(chart_png), width=16 * cm, height=6.9 * cm),
        Paragraph("Highest-risk customers", styles["Heading2"]), _table(top_risk),
        Spacer(1, 10),
        Paragraph("Predictions are risk estimates to prioritise outreach; they are not "
                  "decisions about individual customers.", styles["Italic"]),
    ]
    doc.build(story)
    return buffer.getvalue()
