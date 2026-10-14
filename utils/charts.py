"""Plotly figures used by the churn app."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve


def confusion_matrix_fig(cm: list, labels: list) -> go.Figure:
    fig = px.imshow(cm, x=[f"Pred {c}" for c in labels], y=[f"True {c}" for c in labels],
                    text_auto=True, color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_showscale=False)
    return fig


def roc_fig(y_true: pd.Series, y_score: pd.Series, auc: float) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC={auc:.3f})")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                    line=dict(dash="dash", color="grey"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                      height=340, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def pr_fig(y_true: pd.Series, y_score: pd.Series) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", name="PR curve"))
    fig.add_hline(y=float(y_true.mean()), line_dash="dash", line_color="grey",
                  annotation_text="baseline (positive rate)")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=340,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def importance_fig(importance: dict, top_n: int = 10) -> go.Figure:
    s = pd.Series(importance).head(top_n).sort_values()
    s.index = [i.replace("num__", "").replace("cat__", "") for i in s.index]
    fig = px.bar(s, orientation="h", labels={"value": "Importance", "index": ""})
    fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def effects_fig(effects: pd.DataFrame) -> go.Figure:
    df = effects.head(8).iloc[::-1]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in df["effect"]]
    fig = go.Figure(go.Bar(x=df["effect"], y=df["feature"], orientation="h",
                           marker_color=colors))
    fig.update_layout(xaxis_title="Change in churn probability vs. typical value",
                      height=320, margin=dict(l=10, r=10, t=30, b=10))
    return fig
