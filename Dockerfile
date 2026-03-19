FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir pip setuptools wheel

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || true
RUN pip install --no-cache-dir \
    langgraph langchain langchain-openai langchain-anthropic \
    pinecone-client rank-bm25 fastapi "uvicorn[standard]" \
    pydantic pydantic-settings boto3 httpx python-dotenv

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
