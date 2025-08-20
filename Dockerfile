# syntax=docker/dockerfile:1.7-labs

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default environment
ENV PORT=7860 \
    SERVER_NAME=0.0.0.0 \
    DEBUG=0 \
    GRADIO_SHARE=0

EXPOSE 7860

ENTRYPOINT ["python", "serve.py"]

