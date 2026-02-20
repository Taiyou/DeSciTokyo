FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "scripts/run_experiment.py"]
CMD ["--output-dir", "/app/results", "--n-seeds", "500", "--n-jobs", "-1"]
