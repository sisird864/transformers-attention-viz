.PHONY: help install dev test lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  make install    Install the package"
	@echo "  make dev        Install in development mode with all extras"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linting checks"
	@echo "  make format     Format code with black and isort"
	@echo "  make clean      Clean build artifacts"
	@echo "  make build      Build distribution packages"
	@echo "  make docs       Build documentation"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	python examples/download_examples.py

test:
	pytest tests/ -v --cov=attention_viz --cov-report=html --cov-report=term

lint:
	flake8 attention_viz tests
	mypy attention_viz
	black --check attention_viz tests
	isort --check-only attention_viz tests

format:
	black attention_viz tests
	isort attention_viz tests

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

docs:
	cd docs && sphinx-build -b html . _build/html

dashboard:
	attention-viz-dashboard