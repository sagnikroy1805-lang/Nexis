# One reproducible environment for NEXIS (CLAUDE.md: "one reproducible
# Dockerfile is the requirement"; no production compose stack).
#
# Build:   docker build -t nexis .
# GPU run: docker run --gpus all -v %cd%/data:/app/data -v %cd%/results:/app/results nexis make ladder
# API:     docker run -p 8000:8000 -v %cd%/data:/app/data nexis make api
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dev,gnn,api]"

COPY . .
EXPOSE 8000
CMD ["make", "test"]
