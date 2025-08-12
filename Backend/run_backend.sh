#!/usr/bin/env bash
export $(grep -v '^#' .env | xargs)
alembic upgrade head
uvicorn main:app --host 0.0.0.0 --port 8000
