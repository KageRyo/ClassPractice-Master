# Makefile for Image Enhancement Project

.PHONY: test test-verbose test-coverage clean install help

# Default target
help:
	@echo "Available commands:"
	@echo "  make test         - Run all tests"
	@echo "  make test-verbose - Run tests with verbose output"
	@echo "  make test-coverage- Run tests with coverage report"
	@echo "  make install      - Install dependencies"
	@echo "  make clean        - Clean cache files"
	@echo "  make help         - Show this help"

# Run tests
test:
	python -m pytest tests/ -q

# Run tests with verbose output
test-verbose:
	python -m pytest tests/ -v --tb=short

# Run tests with coverage (requires pytest-cov)
test-coverage:
	python -m pytest tests/ --cov=src --cov-report=term --cov-report=html

# Install dependencies
install:
	pip install -r requirements.txt

# Clean cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "htmlcov" -delete

# Run main program
run:
	python main.py