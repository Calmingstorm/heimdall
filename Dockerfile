FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-client curl build-essential \
        ffmpeg libffi-dev libsodium-dev libopus0 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash heimdall

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[all]"

COPY src/ src/
COPY ui/ ui/
COPY config.yml .

RUN mkdir -p data/context data/sessions data/logs data/usage data/skills data/search && \
    chown -R heimdall:heimdall /app

USER heimdall

HEALTHCHECK --interval=3m --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["python", "-m", "src"]
