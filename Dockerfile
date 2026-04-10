FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py .
COPY static/ static/

# Pipeline source (importable via PYTHONPATH)
COPY src/ src/

# Default Word template
COPY docs/ docs/

# Runtime directories
RUN mkdir -p output cache logs

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health', timeout=8)" || exit 1

CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
