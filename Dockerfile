FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install uv && uv sync --frozen
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "src.server.app"]
