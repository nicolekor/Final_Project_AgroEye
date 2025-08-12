#!/bin/bash
DB_USER="root"
DB_PASS="root"
DB_NAME="yolo_project"
echo ">>> Creating database if not exists"
mysql -u$DB_USER -p$DB_PASS -e "CREATE DATABASE IF NOT EXISTS $DB_NAME CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
echo ">>> Applying migrations"
cd "$(dirname "$0")"
alembic upgrade head
echo ">>> Starting server"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
