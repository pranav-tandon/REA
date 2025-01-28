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
	@echo ">>> Removing any old venv in $(BACKEND_DIR)..."
	rm -rf $(BACKEND_DIR)/$(VENV)
	@echo ">>> Creating fresh venv in $(BACKEND_DIR)/$(VENV)..."
	cd $(BACKEND_DIR) && $(PYTHON) -m venv $(VENV)
	@echo ">>> Activating venv and installing dependencies..."
	@cd $(BACKEND_DIR) && source $(VENV)/bin/activate && pip install --upgrade pip
	@cd $(BACKEND_DIR) && source $(VENV)/bin/activate && pip install --index-url $(TORCH_INDEX) torch torchvision torchaudio
	@cd $(BACKEND_DIR) && source $(VENV)/bin/activate && pip install -r requirements.txt

.PHONY: backend-run
backend-run: backend-venv
	@echo ">>> Starting FastAPI server..."
	cd $(BACKEND_DIR) && source $(VENV)/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000

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
#  Combined Target
# -----------------------------------------------------------------
.PHONY: dev
dev: backend-venv frontend-install
	@echo ">>> Running BOTH backend & frontend concurrently. Press Ctrl + C to kill both."
	( \
	  trap 'kill $(jobs -p); kill $(lsof -t -i:8000) || true' SIGINT SIGTERM EXIT; \
	  cd $(BACKEND_DIR) && source $(VENV)/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000 & \
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
	@echo "  backend-venv       : Create fresh backend venv and install deps"
	@echo "  backend-run        : Create venv and run uvicorn"
	@echo "  backend-clean      : Remove the backend venv"
	@echo "  frontend-install   : npm install for Next.js"
	@echo "  frontend-dev       : npm run dev for Next.js"
	@echo "  frontend-build     : npm run build for Next.js"
	@echo "  dev                : Start both backend & frontend in parallel, kill both on Ctrl+C"
	@echo "  clean              : Remove venv and any other build artifacts"
