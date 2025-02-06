SHELL := /bin/bash

# -----------------------------------------------------------------
#  Variables
# -----------------------------------------------------------------
BACKEND_DIR := ./backend
FRONTEND_DIR := ./nextjs
PYTHON      := python3
VENV        := venv
TORCH_INDEX := https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------
#  Default target
# -----------------------------------------------------------------
default: help

# -----------------------------------------------------------------
#  Backend Targets
# -----------------------------------------------------------------
.PHONY: backend-venv
backend-venv:
	@cd $(BACKEND_DIR) && \
	if [ ! -d "$(VENV)" ]; then \
		echo ">>> Creating fresh venv in $(BACKEND_DIR)/$(VENV)..."; \
		$(PYTHON) -m venv $(VENV); \
		echo ">>> Activating venv and installing dependencies..."; \
		source $(VENV)/bin/activate && pip install --upgrade pip; \
		source $(VENV)/bin/activate && pip install --index-url $(TORCH_INDEX) torch torchvision torchaudio; \
		source $(VENV)/bin/activate && pip install -r requirements.txt; \
	else \
		echo ">>> Using existing venv in $(BACKEND_DIR)/$(VENV)..."; \
	fi

.PHONY: backend-run
backend-run: backend-venv
	@echo ">>> Starting FastAPI server in LOCAL mode (DeepSeek via Ollama)..."
	cd $(BACKEND_DIR) && source $(VENV)/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000

.PHONY: backend-run-api
backend-run-api: backend-venv
	@echo ">>> Starting FastAPI server in API mode (DeepSeek API)..."
	cd $(BACKEND_DIR) && source $(VENV)/bin/activate && MODEL_MODE=api uvicorn app:app --host 0.0.0.0 --port 8000

.PHONY: backend-clean
backend-clean:
	@echo ">>> Removing backend venv..."
	rm -rf $(BACKEND_DIR)/$(VENV)

# -----------------------------------------------------------------
#  Frontend Targets
# -----------------------------------------------------------------
.PHONY: frontend-install
frontend-install:
	@echo ">>> Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && npm install

.PHONY: frontend-dev
frontend-dev: frontend-install
	@echo ">>> Running Next.js dev server..."
	cd $(FRONTEND_DIR) && npm run dev

.PHONY: frontend-build
frontend-build: frontend-install
	@echo ">>> Building Next.js app..."
	cd $(FRONTEND_DIR) && npm run build

# -----------------------------------------------------------------
#  MongoDB Target (Docker)
# -----------------------------------------------------------------
.PHONY: mongodb
mongodb:
	@echo ">>> Starting MongoDB Docker container..."
	@if docker ps -a --format '{{.Names}}' | grep -q "^mongodb$$"; then \
	  docker start mongodb; \
	else \
	  docker run -d --name mongodb -p 27017:27017 mongo; \
	fi

# -----------------------------------------------------------------
#  Ollama Target
# -----------------------------------------------------------------
.PHONY: ollama
ollama:
	@echo ">>> Checking Ollama service status..."
	@service_status=$$(brew services list | awk '/^ollama[[:space:]]/ {print $$2}'); \
	if [ "$$service_status" != "started" ]; then \
	  echo ">>> Ollama is not running. Restarting service..."; \
	  brew services stop ollama; \
	  brew services start ollama; \
	else \
	  echo ">>> Ollama is already running."; \
	fi; \
	echo ">>> Checking if port 11434 is in use..."; \
	if lsof -i :11434 >/dev/null 2>&1; then \
	  echo ">>> Port 11434 is already in use. Skipping 'ollama serve' launch."; \
	else \
	  echo ">>> Running ollama serve..."; \
	  ollama serve; \
	fi

# -----------------------------------------------------------------
#  Combined Target (MongoDB, Ollama, Backend, Frontend)
# -----------------------------------------------------------------
.PHONY: all
all: backend-run

# -----------------------------------------------------------------
#  Combined target for dev (backend & frontend concurrently)
# -----------------------------------------------------------------
.PHONY: dev
dev: backend-venv frontend-install
	@echo ">>> Running BOTH backend & frontend concurrently in LOCAL mode. Press Ctrl + C to kill both."
	( \
	  trap 'kill $(jobs -p); kill $(lsof -t -i:8000) || true' SIGINT SIGTERM; \
	  cd $(BACKEND_DIR) && source $(VENV)/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000 & \
	  cd $(FRONTEND_DIR) && npm run dev \
	)

.PHONY: dev-api
dev-api: backend-venv frontend-install
	@echo ">>> Running BOTH backend & frontend concurrently in API mode. Press Ctrl + C to kill both."
	( \
	  trap 'kill $(jobs -p); kill $(lsof -t -i:8000) || true' SIGINT SIGTERM; \
	  cd $(BACKEND_DIR) && source $(VENV)/bin/activate && MODEL_MODE=api uvicorn app:app --host 0.0.0.0 --port 8000 & \
	  cd $(FRONTEND_DIR) && npm run dev \
	)

# -----------------------------------------------------------------
#  Clean and Help
# -----------------------------------------------------------------
.PHONY: clean
clean: backend-clean
	@echo ">>> (Optional) Clean up other stuff if needed..."

.PHONY: help
help:
	@echo "Available make targets:"
	@echo "  backend-venv       : Create fresh backend venv and install dependencies"
	@echo "  backend-run        : Create venv and run uvicorn"
	@echo "  backend-run-api     : Create venv and run uvicorn in API mode"
	@echo "  backend-clean      : Remove the backend venv"
	@echo "  frontend-install   : npm install for Next.js"
	@echo "  frontend-dev       : npm run dev for Next.js"
	@echo "  frontend-build     : npm run build for Next.js"
	@echo "  mongodb            : Start MongoDB Docker container (or start if exists)"
	@echo "  ollama             : Check (and if needed restart) Ollama and run 'ollama serve' unless port 11434 is in use"
	@echo "  all                : Start backend in LOCAL mode"
	@echo "  dev                : Start backend and frontend concurrently in LOCAL mode"
	@echo "  dev-api            : Start backend and frontend concurrently in API mode"
	@echo "  clean              : Remove venv and any other build artifacts"
