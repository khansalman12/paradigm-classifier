# Dockerfile — builds a self-contained image with the model baked in

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=42

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY programming_types.md .
COPY main.py .
COPY src/ src/
COPY tests/ tests/

# download the model at build time so `docker run` works offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["python", "main.py", "--test"]
