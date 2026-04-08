FROM python:3.11-slim

LABEL org.opencontainers.image.title="911 City-Wide Emergency Dispatch Supervisor"
LABEL org.opencontainers.image.description="City-wide 911 dispatch supervisor RL environment"

WORKDIR /app

# Install curl for the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy and install only the packages that exist on PyPI
# (openenv / openenv-core are not on PyPI — server runs fine without them)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pydantic>=2.7 \
    fastapi>=0.110 \
    "uvicorn[standard]>=0.29" \
    openai>=1.12 \
    httpx>=0.27 \
    matplotlib>=3.7 \
    numpy \
    groq \
    pyyaml>=6.0.1

# Copy source and data
COPY src/ /app/src/
COPY data/ /app/data/
COPY openenv.yaml /app/openenv.yaml
COPY live_dashboard.html /app/live_dashboard.html

# HuggingFace Spaces always routes to port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn src.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
