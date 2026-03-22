FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 + system deps
# Using 3.11 instead of 3.12 because tflite-runtime (openwakeword dep)
# has no Python 3.12 wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3.11-distutils \
        ffmpeg \
        curl \
        && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash voicesvc

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Download default openwakeword models (must run as root before USER switch)
RUN python3 -c "from openwakeword.utils import download_models; download_models()" && \
    chmod -R a+r /usr/local/lib/python3.11/dist-packages/openwakeword/resources/

# Download default Piper voice
RUN piper --model en_US-lessac-medium --output-raw --output_file /dev/null < /dev/null 2>/dev/null || true

# Copy application
COPY src/ src/
COPY config.yml .

# Create directories
RUN mkdir -p voices models && \
    chown -R voicesvc:voicesvc /app

USER voicesvc

EXPOSE 3940

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:3940/health || exit 1

CMD ["python3", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "3940"]
