PYTHON ?= python3
DOCKER ?= docker
COMPOSE ?= docker-compose
SERVICE ?= api
IMAGE ?= simula-ad:latest
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: build train evaluate api test batch-predict drift benchmark shell

build:
	$(COMPOSE) build --no-cache

train:
	$(COMPOSE) run --rm $(SERVICE) sh -c "PYTHONPATH=src $(PYTHON) scripts/train_model.py"

evaluate:
	$(COMPOSE) run --rm $(SERVICE) sh -c "PYTHONPATH=src $(PYTHON) scripts/evaluate_model.py"

api:
	$(COMPOSE) up $(SERVICE)

test:
	$(COMPOSE) run --rm $(SERVICE) sh -c "PYTHONPATH=src $(PYTHON) -m pytest tests"

batch-predict:
	$(COMPOSE) run --rm $(SERVICE) sh -c "PYTHONPATH=src $(PYTHON) scripts/batch_predict.py"

drift:
	$(COMPOSE) run --rm $(SERVICE) sh -c "PYTHONPATH=src $(PYTHON) scripts/drift_report.py"

benchmark:
	$(COMPOSE) run --rm $(SERVICE) sh -c "PYTHONPATH=src $(PYTHON) scripts/latency_benchmark.py"

shell:
	$(COMPOSE) run --rm $(SERVICE) sh