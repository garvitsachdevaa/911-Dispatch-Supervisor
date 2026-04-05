FROM python:3.11-slim

WORKDIR /app

COPY src/server/requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/

EXPOSE 8000

CMD ["python", "-m", "src.server.app"]
