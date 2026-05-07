FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "energy_forecasting_anomaly.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
