FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[server]" 2>/dev/null || \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers pyyaml opentelemetry-api opentelemetry-sdk

# Copy source
COPY prometheus/ prometheus/
COPY prometheus.yaml .

# Expose MCP server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the MCP server
ENTRYPOINT ["python3", "-m", "prometheus.cli"]
CMD ["serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
