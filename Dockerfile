# Reproducible container for the JJUSHYCSH no-show data pipeline (Module 3)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPELINE_ROOT=/opt/pipeline \
    PIPELINE_ACTOR=container

WORKDIR /opt/pipeline

# System packages kept minimal; no compilers needed for the pinned wheels
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

COPY src/ src/
COPY dags/ dags/
COPY tests/ tests/
COPY scripts/ scripts/
COPY run_pipeline.py .

# Data is mounted at run time, never baked into the image (privacy + image size)
RUN useradd --create-home pipeline \
    && mkdir -p data/external data/raw data/interim data/processed reports logs gx \
    && chown -R pipeline:pipeline /opt/pipeline
USER pipeline

VOLUME ["/opt/pipeline/data", "/opt/pipeline/reports", "/opt/pipeline/logs", "/opt/pipeline/gx"]

ENTRYPOINT ["tini", "--"]
# Unit tests run first; the pipeline only starts if they pass
CMD ["sh", "-c", "pytest -q tests && python run_pipeline.py"]
