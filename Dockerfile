# ─────────────────────────────────────────────────────────────────────────────
#  Stock Engine — Dockerfile
#
#  Multi-stage build:
#   1. deps  — install Python packages
#   2. final — lean runtime image
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS deps

WORKDIR /app

# System deps for psycopg2 + pandas compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Final Stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS final

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY . .

RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "pipeline.run_pipeline"]
