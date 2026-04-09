FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:/app/.venv/bin:${PATH}"

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY config ./config
COPY streamcamserver ./streamcamserver
COPY templates ./templates
COPY static ./static
COPY certificates ./certificates

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "streamcamserver.app"]
