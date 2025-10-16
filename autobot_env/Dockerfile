FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# (pandas funziona con le wheel precompilate; build-essential è una rete di sicurezza)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Dipendenze
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Codice e dati necessari a run-time
COPY app.py /app/app.py
COPY main.py /app/main.py
COPY auto_dati.csv /app/auto_dati.csv

EXPOSE 8000
ENV PORT=8000

# Avvio API FastAPI (app:app dentro app.py) — porta fissata a 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]