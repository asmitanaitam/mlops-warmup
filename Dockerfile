FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code plus the exported champion model and its metadata
COPY app.py .
COPY model.joblib .
COPY reference_stats.json .
COPY model_info.json .

# Prediction logs are written here; we mount a volume over this folder
ENV LOG_PATH=/app/logs/predictions.jsonl
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]