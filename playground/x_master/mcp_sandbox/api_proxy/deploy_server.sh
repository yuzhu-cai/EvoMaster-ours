#!/usr/bin/env bash
set -euo pipefail

# Start FastAPI service
echo "deploy FastAPI service"
python api_server.py &
API_PID=$!

# Wait for API service to start
sleep 4

# Start MCP adapter

python mcp_search_adapter.py &
MCP_PID=$!

# Wait for termination signal
trap "kill $API_PID $MCP_PID; exit" INT TERM
wait