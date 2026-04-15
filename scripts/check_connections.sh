#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_SCHEMA="${POSTGRES_SCHEMA:-stack}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-qwen3-coder:30b}"
OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-qwen3-embedding:8b}"

print_section() {
  printf '\n== %s ==\n' "$1"
}

check_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1"
    return 1
  fi
}

print_section "Environment"
printf 'PROJECT_ROOT=%s\n' "${PROJECT_ROOT}"
printf 'ENV_FILE=%s\n' "${ENV_FILE}"
printf 'POSTGRES_HOST=%s\n' "${POSTGRES_HOST:-<missing>}"
printf 'POSTGRES_PORT=%s\n' "${POSTGRES_PORT}"
printf 'POSTGRES_DB=%s\n' "${POSTGRES_DB:-<missing>}"
printf 'POSTGRES_USER=%s\n' "${POSTGRES_USER:-<missing>}"
printf 'POSTGRES_SCHEMA=%s\n' "${POSTGRES_SCHEMA}"
printf 'OLLAMA_BASE_URL=%s\n' "${OLLAMA_BASE_URL}"
printf 'OLLAMA_LLM_MODEL=%s\n' "${OLLAMA_LLM_MODEL}"
printf 'OLLAMA_EMBED_MODEL=%s\n' "${OLLAMA_EMBED_MODEL}"

print_section "Prerequisites"
check_command curl || exit 1
check_command psql || exit 1

print_section "Ollama models"
curl -sS "${OLLAMA_BASE_URL}/api/tags"
printf '\n'

print_section "LLM test"
curl -sS "${OLLAMA_BASE_URL}/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${OLLAMA_LLM_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}],
    \"stream\": false
  }"
printf '\n'

print_section "Embedding test"
curl -sS "${OLLAMA_BASE_URL}/api/embed" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${OLLAMA_EMBED_MODEL}\",
    \"input\": \"ping\"
  }"
printf '\n'

if [[ -z "${POSTGRES_HOST:-}" || -z "${POSTGRES_DB:-}" || -z "${POSTGRES_USER:-}" || -z "${POSTGRES_PASSWORD:-}" ]]; then
  print_section "PostgreSQL"
  printf 'Missing one of required variables: POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD\n'
  exit 1
fi

print_section "PostgreSQL SELECT 1"
PGPASSWORD="${POSTGRES_PASSWORD}" \
psql \
  -h "${POSTGRES_HOST}" \
  -p "${POSTGRES_PORT}" \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  -c "SELECT 1;"

print_section "PostgreSQL search_path"
PGPASSWORD="${POSTGRES_PASSWORD}" \
PGOPTIONS="-c search_path=${POSTGRES_SCHEMA},public" \
psql \
  -h "${POSTGRES_HOST}" \
  -p "${POSTGRES_PORT}" \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  -c "SHOW search_path;"

print_section "PostgreSQL table check"
PGPASSWORD="${POSTGRES_PASSWORD}" \
PGOPTIONS="-c search_path=${POSTGRES_SCHEMA},public" \
psql \
  -h "${POSTGRES_HOST}" \
  -p "${POSTGRES_PORT}" \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  -c 'SELECT COUNT(*) FROM "Лицевые счета";'
