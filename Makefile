# ============================================================================
# Makefile — Common Project Commands
# ============================================================================
# Usage: make <target>
# Run `make help` to see all available commands.
#
# CONCEPT: Makefile
# A Makefile defines shortcuts for common commands. Instead of remembering
# long command strings, you just type `make train` or `make deploy`.
# This is standard practice in ML projects and impresses reviewers who see
# that the project is well-organized and easy to use.

.PHONY: help setup install train evaluate test deploy clean lint format

# Default target
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

install: ## Install Python dependencies
	pip install -r requirements.txt

setup: install ## Full AWS environment setup (buckets, tables, roles)
	@echo "Setting up AWS infrastructure..."
	bash infrastructure/scripts/setup_environment.sh
	@echo "Setup complete!"

# ============================================================================
# ML OPERATIONS
# ============================================================================

train: ## Launch SageMaker training job
	@echo "Launching training job..."
	python -c "from src.utils.aws_helpers import launch_training_job; launch_training_job()"

train-local: ## Train locally (for development/debugging)
	python src/training/train.py \
		--train-dir ./data/train \
		--val-dir ./data/val \
		--model-dir ./model \
		--epochs 5 \
		--batch-size 16

evaluate: ## Evaluate trained model on test set
	python -c "from src.training.evaluate import evaluate_from_checkpoint; \
		evaluate_from_checkpoint('./model', './data/test')"

predict: ## Classify a single image (usage: make predict IMG=path/to/image.jpg)
	@if [ -z "$(IMG)" ]; then echo "Usage: make predict IMG=path/to/image.jpg"; exit 1; fi
	python src/inference/predictor.py --image $(IMG)

# ============================================================================
# TESTING
# ============================================================================

test: ## Run all tests
	pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-unit: ## Run only unit tests (fast)
	pytest tests/ -v -m "not integration" --tb=short

test-integration: ## Run integration tests
	pytest tests/ -v -m "integration" --tb=short

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: ## Check code style
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format: ## Auto-format code
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile=black

# ============================================================================
# DEPLOYMENT
# ============================================================================

deploy: ## Deploy full stack to AWS
	@echo "Deploying to AWS..."
	bash infrastructure/scripts/deploy.sh
	@echo "Deployment complete!"

deploy-model: ## Deploy only the ML model endpoint
	python -c "from src.utils.aws_helpers import deploy_endpoint; deploy_endpoint()"

teardown: ## Remove all AWS resources (DESTRUCTIVE!)
	@echo "WARNING: This will delete ALL resources. Press Ctrl+C to cancel."
	@sleep 5
	bash infrastructure/scripts/teardown.sh

# ============================================================================
# MONITORING
# ============================================================================

monitoring-setup: ## Create CloudWatch dashboards and alarms
	python src/monitoring/dashboard.py

# ============================================================================
# CLEANUP
# ============================================================================

clean: ## Remove generated files
	rm -rf __pycache__ .pytest_cache htmlcov .mypy_cache
	rm -rf model/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned!"
