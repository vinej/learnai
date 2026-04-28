FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY forecaster ./forecaster
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "forecaster.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
