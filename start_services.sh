#!/bin/bash
 
set -e

# Load environment variables from .env file if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Starting FastAPI Server
echo "Starting FastAPI server on port $FASTAPI_PORT..."
nohup uvicorn api-service.main:app --host 0.0.0.0 --port "$FASTAPI_PORT" --reload > uvicorn.log 2>&1 &
echo $! > uvicorn.pid
echo "FastAPI server started (PID: $(cat uvicorn.pid))"