#!/bin/bash

CURRENT_DIR="$(pwd)"
docker run --restart=unless-stopped -v "${CURRENT_DIR}/data:/app/data" -p 8501:8501 --name festival-finder-container -d --env-file ./.env festival-finder:latest