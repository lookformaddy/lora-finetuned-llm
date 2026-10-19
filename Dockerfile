FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY examples ./examples
COPY tests ./tests
RUN pip install --no-cache-dir -e .

# Default: run the full pipeline (dataset -> pretrain -> LoRA -> evaluate).
CMD ["python", "scripts/train.py"]
