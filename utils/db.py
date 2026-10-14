"""Prediction-history storage with SQLAlchemy Core.

The same code works with SQLite, PostgreSQL and MySQL; only the URL changes:
    sqlite:///predictions.db
    postgresql+psycopg://user:password@host:5432/mlapp      (pip install psycopg[binary])
    mysql+pymysql://user:password@host:3306/mlapp            (pip install pymysql)
Every query uses bound parameters (:name), never string formatting, which prevents
SQL injection.
"""
import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at VARCHAR(32) NOT NULL,
        model_version VARCHAR(64) NOT NULL,
        inputs TEXT NOT NULL,
        probability FLOAT NOT NULL,
        label VARCHAR(32) NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL REFERENCES predictions(id),
        created_at VARCHAR(32) NOT NULL,
        helpful INTEGER NOT NULL,
        comment TEXT
    )""",
]


def make_engine(url: str) -> Engine:
    # pool_pre_ping re-checks pooled connections so dropped DB connections recover
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        for ddl in SCHEMA:
            if not url.startswith("sqlite"):
                ddl = ddl.replace("INTEGER PRIMARY KEY AUTOINCREMENT",
                                  "SERIAL PRIMARY KEY" if url.startswith("postgres")
                                  else "INTEGER PRIMARY KEY AUTO_INCREMENT")
            conn.execute(text(ddl))
    return engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_prediction(engine: Engine, inputs: dict, probability: float, label: str,
                    model_version: str) -> int:
    with engine.begin() as conn:   # begin() commits on success, rolls back on error
        result = conn.execute(
            text("INSERT INTO predictions (created_at, model_version, inputs, "
                 "probability, label) VALUES (:ts, :ver, :inputs, :p, :label)"),
            {"ts": _now(), "ver": model_version, "inputs": json.dumps(inputs),
             "p": probability, "label": label},
        )
        return int(result.lastrowid) if result.lastrowid else int(conn.execute(
            text("SELECT MAX(id) FROM predictions")).scalar())


def save_feedback(engine: Engine, prediction_id: int, helpful: bool,
                  comment: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO feedback (prediction_id, created_at, helpful, comment) "
                 "VALUES (:pid, :ts, :h, :c)"),
            {"pid": prediction_id, "ts": _now(), "h": int(helpful), "c": comment[:500]},
        )


def load_history(engine: Engine, label: str | None = None, limit: int = 200) -> pd.DataFrame:
    sql = ("SELECT p.id, p.created_at, p.model_version, p.probability, p.label, "
           "p.inputs, f.helpful FROM predictions p "
           "LEFT JOIN feedback f ON f.prediction_id = p.id ")
    params = {"limit": int(limit)}
    if label:
        sql += "WHERE p.label = :label "
        params["label"] = label
    sql += "ORDER BY p.id DESC LIMIT :limit"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if not df.empty:
        inputs = pd.json_normalize(df.pop("inputs").map(json.loads))
        df = pd.concat([df, inputs], axis=1)
    return df
