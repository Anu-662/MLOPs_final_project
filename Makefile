.PHONY: help
help:
	@echo "  CESAR commands:"
	@echo "    make install        Install dependencies"
	@echo "    make train          Train model"
	@echo "    make clean-data     Clean raw DVF data"
	@echo "    make enrich         Enrich data and generate department stats"
	@echo "    make evaluate       Compare model versions"
	@echo "    make pipeline       Run full pipeline: clean → enrich → train"
	@echo "    make serve          Start API (port 8000)"
	@echo "    make ui             Start UI (port 8501)"
	@echo "    make monitoring     Start monitoring dashboard"
	@echo "    make test           Run acceptance tests"
	@echo "    make docker-up      Build and run with Docker"
	@echo "    make docker-down    Stop Docker containers"
	@echo "    make all            Install + train"

.PHONY: install
install:
	pip install -e .
	pip install streamlit requests httpx

.PHONY: clean-data
clean-data:
	python -m data_validation.clean_dvf --input data/ --output data_cleaned/

.PHONY: enrich
enrich:
	python -m data_enrichment.enrich_dvf --input data/ --output data_enriched/

.PHONY: train
train:
	python -m training.scripts.train_from_minimal_csv

.PHONY: evaluate
evaluate:
	python -m evaluation.compare_v1_v2

.PHONY: pipeline
pipeline: clean-data enrich train
	@echo "Full pipeline complete: clean → enrich → train"
	@echo "Start the API with: make serve"

MODEL_PATH := $(shell ls -t artifact_storage/model_*.joblib 2>/dev/null | head -1)
CONTRACT_PATH := $(shell ls -t artifact_storage/contract_*.json 2>/dev/null | head -1)

.PHONY: serve
serve:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: No model found. Run 'make train' first."; \
		exit 1; \
	fi
	CESAR_MODEL_PATH=$(MODEL_PATH) CESAR_CONTRACT_PATH=$(CONTRACT_PATH) \
		uvicorn runtime.prediction_api.app:app --host 0.0.0.0 --port 8000

.PHONY: ui
ui:
	CESAR_API_URL=http://localhost:8000 \
		streamlit run runtime/streamlit_ui/app.py

.PHONY: monitoring
monitoring:
	CESAR_API_URL=http://localhost:8000 \
		streamlit run runtime/streamlit_ui/monitoring.py --server.port 8502

.PHONY: test
test:
	cesar acceptance-tests run --base-url http://localhost:8000

.PHONY: docker-up
docker-up:
	docker compose -f deployment/docker-compose.yml up --build

.PHONY: docker-down
docker-down:
	docker compose -f deployment/docker-compose.yml down

.PHONY: all
all: install train
	@echo "Ready! Start the API with: make serve"
	@echo "Then in another terminal:  make ui"
