FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# Артефакт модели монтируется через volume (см. docker-compose.yml)
# Если хочешь запечь его в образ — раскомментируй строку ниже:
# COPY scoring_artifact_v5.joblib .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]