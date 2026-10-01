FROM python:3.14-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# ---------- treino ----------
FROM base AS train
COPY requirements/train.txt requirements/
RUN pip install --no-cache-dir -r requirements/train.txt
COPY src/ ./src/
CMD ["python", "-m", "src.train"]

# ---------- API ----------
FROM base AS api
COPY requirements/api.txt requirements/
RUN pip install --no-cache-dir -r requirements/api.txt
RUN useradd --create-home appuser
COPY src/ ./src/
COPY models/ ./models/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.services.api:app", "--host", "0.0.0.0", "--port", "8000"]