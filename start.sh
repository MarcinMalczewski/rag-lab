#!/usr/bin/env bash
# Start Qdrant and PostgreSQL in the background.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

docker compose --env-file .env -f docker-compose.yml up -d --wait
docker compose --env-file .env -f docker-compose.yml ps
