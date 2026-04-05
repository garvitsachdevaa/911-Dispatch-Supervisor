FROM python:3.11-slim
LABEL org.opencontainers.image.title="911 City-Wide Emergency Dispatch Supervisor"
LABEL org.opencontainers.image.description="City-wide 911 dispatch supervisor RL environment"
WORKDIR /app
COPY . /app
RUN pip install uv && uv sync --frozen
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
