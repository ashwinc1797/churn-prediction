# ---- Streamlit ML platform image ----------------------------------------
FROM python:3.13-slim

# Python writes no .pyc files and logs straight to the console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install dependencies first: this layer is cached until requirements change
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy the application code and model artifacts
COPY . .

# run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
