.PHONY: help install test test-leakage demo api config check host-info verify clean docker-build docker-run

VENV := .venv
PY   := $(VENV)/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	python3 -m venv --system-site-packages $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -e .
	@echo "installed. run 'make test'"

test: ## Run the full test suite
	$(PY) -m pytest tests/ -q

test-leakage: ## Run only the causality and leakage tests
	$(PY) -m pytest tests/ -m leakage -q

demo: ## Run the end-to-end pipeline on synthetic data
	$(PY) scripts/demo.py

api: ## Serve the API on 127.0.0.1:8000
	$(VENV)/bin/uvicorn quantforge.api.app:app --host 127.0.0.1 --port 8000

config: ## Print the loaded configuration and its fingerprint
	$(PY) -c "from quantforge.core.config import get_config; \
c=get_config(); print(c.fingerprint()); print('live_trading_enabled:', c.execution.live_trading_enabled)"

check: ## Verify the live-trading gate is closed (expected: not permitted)
	$(PY) -c "from quantforge.execution.live_gate import LiveTradingGate; \
ok,why=LiveTradingGate().check(); print('permitted:', ok); print('reason:', why)"

host-info: ## Report this host's platform capabilities
	$(PY) scripts/bootstrap.py --host-info

verify: ## Install and verify everything for this host
	$(VENV)/bin/python scripts/bootstrap.py

clean: ## Remove caches and generated output
	rm -rf .pytest_cache **/__pycache__ __pycache__ .quantforge
	rm -f quantforge.db
	find . -name '*.pyc' -delete

docker-build: ## Build the container image
	docker build -t quantforge-ai .

docker-run: ## Run the container (API on 127.0.0.1:8000)
	docker run -p 127.0.0.1:8000:8000 quantforge-ai
