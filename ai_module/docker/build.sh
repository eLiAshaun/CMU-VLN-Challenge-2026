#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Build context is ai_module/ so COPY src/ works correctly
cd $SCRIPT_DIR/..
docker build -t docker_ai_module:mast3r-live -f docker/Dockerfile .
