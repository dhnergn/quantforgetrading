.PHONY: help install test run-api run-cli generate-data clean docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run all tests
	pytest tests/ -v --tb=short

run-api: ## Start the API server
	uvicorn quantforge.api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

run-cli: ## Run the CLI
	python -m quantforge

generate-data: ## Generate sample data
	python scripts/generate_sample_data.py --symbol AAPL --output data/AAPL.csv
	python scripts/generate_sample_data.py --symbol MSFT --output data/MSFT.csv
	python scripts/generate_sample_data.py --symbol GOOGL --output data/GOOGL.csv

backtest: ## Run a sample backtest
	python -m quantforge backtest --data data/AAPL.csv --strategy crossover --instruments AAPL --start-date 2023-02-01 --end-date 2023-12-31

signals: ## Generate sample signals
	python -m quantforge signals --data data/AAPL.csv --strategy crossover --instruments AAPL

validate: ## Validate sample data
	python -m quantforge validate --data data/AAPL.csv --instrument AAPL

clean: ## Clean generated files
	rm -rf data/*.csv
	rm -rf logs/*.log
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.db

docker-build: ## Build Docker image
	docker build -t quantforge-ai .

docker-run: ## Run Docker container
	docker run -p 8000:8000 quantforge-ai
