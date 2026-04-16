SHELL := /bin/bash

.PHONY: check-env check-prereqs check-all check-ollama-models check-ollama-llm check-ollama-embed check-pg-connect check-pg-search-path check-pg-table

check-env:
	bash scripts/check_connections.sh env

check-prereqs:
	bash scripts/check_connections.sh prereqs

check-all:
	bash scripts/check_connections.sh all

check-ollama-models:
	bash scripts/check_connections.sh ollama-models

check-ollama-llm:
	bash scripts/check_connections.sh ollama-llm

check-ollama-embed:
	bash scripts/check_connections.sh ollama-embed

check-pg-connect:
	bash scripts/check_connections.sh pg-connect

check-pg-search-path:
	bash scripts/check_connections.sh pg-search-path

check-pg-table:
	bash scripts/check_connections.sh pg-table
