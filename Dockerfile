FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client curl build-essential \
        ffmpeg libffi-dev libsodium-dev libopus0 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ansiblex

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY config.yml .

RUN mkdir -p data/context data/sessions data/logs data/usage data/skills data/chromadb && \
    chown -R ansiblex:ansiblex /app

USER ansiblex

HEALTHCHECK --interval=3m --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["python", "-m", "src"]
