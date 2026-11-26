# =============================================================================
# Proyecto Final Secuencial — automatización de tareas
# =============================================================================

SHELL := /bin/bash

.DEFAULT_GOAL := help
.PHONY: help install dev pre-commit-install pre-commit lint format format-check typecheck test check build clean app

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Sincroniza el entorno virtual e instala dependencias (CPU)
	uv sync --extra cpu --link-mode=copy

dev: install pre-commit-install ## Prepara el entorno de desarrollo (CPU + hooks)

dev-cuda: ## Sincroniza el entorno con soporte PyTorch CUDA (NVIDIA GPU)
	uv sync --all-groups --extra cuda --link-mode=copy
	uv run --no-sync python -m pre_commit install

pre-commit-install: ## Instala los hooks de pre-commit en el repositorio local
	uv run --no-sync python -m pre_commit install

pre-commit: ## Ejecuta todos los hooks de pre-commit sobre el repositorio
	uv run --no-sync python -m pre_commit run --all-files

lint: ## Ejecuta Ruff (lint)
	uv run --no-sync ruff check .

format: ## Formatea el código con Ruff
	uv run --no-sync ruff format .

format-check: ## Verifica el formato sin modificarlo
	uv run --no-sync ruff format --check .

typecheck: ## Verifica los tipos con ty (Astral)
	uv run --no-sync ty check

test: ## Ejecuta las pruebas
	uv run --no-sync pytest

check: lint format-check typecheck test ## Verificación completa (lint + formato + tipos + pruebas)

app: ## Ejecuta la interfaz interactiva en Streamlit
	uv run streamlit run src/seq_transformer/app.py

build: ## Construye el paquete (sdist + wheel)
	uv build

clean: ## Limpia cachés y artefactos
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
