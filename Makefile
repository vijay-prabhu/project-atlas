.PHONY: install install-dev dev test test-cov lint format docker-up docker-down seed generate loadtest clean

# ─── Setup ────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install langgraph langchain langchain-openai langchain-anthropic langchain-aws \
		pinecone-client rank-bm25 fastapi "uvicorn[standard]" pydantic pydantic-settings \
		boto3 httpx faker python-dotenv sentence-transformers

# ─── Development ──────────────────────────────────────────

dev:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

seed:
	python -m src.data.generator --count 100 --output data/generated_emails.json

generate-1k:
	python -m src.data.generator --count 1000 --output data/generated_1k.json

generate-10k:
	python -m src.data.generator --count 10000 --output data/generated_10k.json

generate-100k:
	python -m src.data.generator --count 100000 --output data/generated_100k.json

generate-dynamo:
	python -m src.data.generator --count 10000 --dynamodb --endpoint http://localhost:8001

# ─── Testing ──────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# ─── Code Quality ────────────────────────────────────────

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

# ─── Docker ───────────────────────────────────────────────

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ─── Load Testing ────────────────────────────────────────

loadtest:
	locust -f loadtest/locustfile.py --host http://localhost:8000

loadtest-headless:
	locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 2m

# ─── AWS ──────────────────────────────────────────────────

sam-validate:
	cd infra && sam validate

sam-build:
	cd infra && sam build

sam-deploy:
	cd infra && sam deploy --guided

# ─── Cleanup ─────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage dist build *.egg-info
