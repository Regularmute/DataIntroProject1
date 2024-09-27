

FROM python:3.11.9-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \ 
    PYTHONDONTWRITEBYTECODE=1 

RUN pip install poetry && poetry config virtualenvs.in-project true

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install

FROM python:3.11.9-slim-bookworm

WORKDIR /app

COPY --from=builder /app .
COPY src/ ./src

EXPOSE 8080

CMD ["/app/.venv/bin/python", "src/app.py"]

