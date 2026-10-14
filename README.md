# Customer Churn Prediction (Streamlit)

Predicts the probability that a telecom customer cancels, with single and batch
scoring, explanations and a model performance dashboard.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/make_dataset.py      # creates data/churn.csv and data/new_customers.csv
python train_model.py            # creates models/
streamlit run app.py
```

## Other apps in this folder
| File | Purpose |
|---|---|
| `batch_app.py` | Batch scoring of CSV/Excel files (Chapter 8) |
| `multistep_app.py` | Five-step prediction wizard (Chapter 10) |
| `db_app.py` | Predictions stored in a database (Chapter 11) |
| `api/main.py` + `api_client_app.py` | FastAPI model service and client (Chapter 11) |
| `report_app.py` | CSV / Excel / PDF / JSON reports (Chapter 12) |

## Tests
```bash
pytest -q
```

## Limitations
Trained on synthetic data for teaching. Predictions are risk estimates to prioritise
retention outreach, not decisions about individual customers.
