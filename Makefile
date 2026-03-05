# Makefile
# Run: make <target>

.PHONY: install setup lint test clean run-app run-notebook

## Install all Python dependencies
install:
	pip install -r requirements.txt
	pip install -e .

## Prepare processed dataset (runs feature engineering pipeline)
setup:
	python -c "from src.data import load_raw; from src.features import build_all_features; import config; df=load_raw(); df=build_all_features(df); df.to_csv(config.PROCESSED_FILE); print('Processed data saved.')"

## Train and persist the best model
train:
	python scripts/train_pipeline.py

## Launch the Streamlit dashboard
run-app:
	streamlit run app/app.py

## Open the EDA notebook
run-notebook:
	jupyter lab notebooks/

## Lint
lint:
	flake8 src/ app/ tests/ --max-line-length=99 --ignore=E203,W503
	isort --check src/ app/ tests/

## Format
format:
	black src/ app/ tests/ --line-length=99
	isort src/ app/ tests/

## Test with coverage
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

## Clean compiled artefacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/
