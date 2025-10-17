FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy credentials file - ENSURE THIS FILE EXISTS
COPY credentials.json .

# Set credentials path AFTER copying
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5001
CMD ["gunicorn", "wsgi:app", "-b", "0.0.0.0:5001", "--workers", "2", "--threads", "4", "--timeout", "120"]
