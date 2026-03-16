# Makefile for DSGP-Brain-Tumor_Detection

# Python interpreter
PYTHON = python3
PIP = pip

# Default target
.DEFAULT_GOAL := help

# ===============================
# Install the package in editable mode
# ===============================
install:
	$(PIP) install -e .

# ===============================
# Run everything
# ===============================
all: data preprocess train evaluate

data:
	python src/data/download_data.py

preprocess:
	python src/data/preprocess.py

train:
	python src/models/train.py --config configs/train_config.yml

evaluate:
	python src/models/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf results/outputs/*


# ===============================
# Run all tests
# ===============================
test:
	$(PYTHON) -m pytest tests/ --maxfail=1 --disable-warnings -q

# ===============================
# Lint the code
# ===============================
lint:
	flake8 scripts/ src/ tests/

# ===============================
# Format code
# ===============================
format:
	black scripts/ src/ tests/

# ===============================
# Clean temporary files
# ===============================
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache


# ===============================
# Help
# ===============================
help:
	@echo "Usage:"
	@echo "  make install   - Install the package"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Check code style"
	@echo "  make format    - Auto-format code"
	@echo "  make clean     - Remove temporary files"
	@echo "  make docs      - Build documentation"
