#!/usr/bin/env bash
set -euo pipefail

# ============================================
# X-Master MCP Services (consistent with deploy_server.sh)
# ============================================
#
# Only starts api_proxy + mcp-sandbox (execute, reset_session, web_search, web_parse).
# Does not start search-tools; mcp-sandbox already includes search and parsing capabilities.
#
# Usage:
#   ./start_all.sh              # Start services (background)
#   ./start_all.sh stop         # Stop all services
#   ./start_all.sh status       # Check service status
#   ./start_all.sh restart      # Restart all services
#
# Optional environment variables:
#   SKIP_API_PROXY=1            Do not start api_proxy (must be started elsewhere)
#   START_LEGACY_EXECUTE_SERVER=1  Also start legacy /execute service (default 30008)
#
# Default ports:
#   - api_proxy:  1234
#   - mcp-sandbox: 8001

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${SCRIPT_DIR}/.pids"

SANDBOX_PORT="${SANDBOX_PORT:-8001}"
API_PORT="${API_PORT:-1234}"
LEGACY_PORT="${LEGACY_PORT:-30008}"
HOST="${HOST:-0.0.0.0}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "${PID_DIR}"

start_api_proxy() {
    if [[ "${SKIP_API_PROXY:-0}" == "1" ]]; then
        log_info "Skipping api_proxy (SKIP_API_PROXY=1)"
        return 0
    fi
    CODE="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 1 "http://127.0.0.1:${API_PORT}/search" 2>/dev/null || true)"
    if [[ "${CODE}" == "405" || "${CODE}" == "422" ]]; then
        log_info "api_proxy is already running (:${API_PORT})"
        return 0
    fi
    log_info "Starting api_proxy (port: ${API_PORT})..."
    cd "${SCRIPT_DIR}/api_proxy"
    PORT="${API_PORT}" nohup python api_server.py > "${PID_DIR}/api.log" 2>&1 &
    echo $! > "${PID_DIR}/api.pid"
    log_info "api_proxy started (PID: $(cat ${PID_DIR}/api.pid))"
    sleep 2
}

start_sandbox() {
    log_info "Starting mcp-sandbox (port: ${SANDBOX_PORT})..."
    cd "${SCRIPT_DIR}/MCP"
    PORT="${SANDBOX_PORT}" HOST="${HOST}" nohup python evomaster_mcp_server.py > "${PID_DIR}/sandbox.log" 2>&1 &
    echo $! > "${PID_DIR}/sandbox.pid"
    log_info "mcp-sandbox started (PID: $(cat ${PID_DIR}/sandbox.pid))"
}

start_legacy() {
    if [[ "${START_LEGACY_EXECUTE_SERVER:-0}" != "1" ]]; then
        return 0
    fi
    log_info "Starting legacy /execute service (port: ${LEGACY_PORT})..."
    cd "${SCRIPT_DIR}/MCP"
    nohup uvicorn tool_server:app --host "${HOST}" --port "${LEGACY_PORT}" --lifespan on --workers 1 > "${PID_DIR}/legacy.log" 2>&1 &
    echo $! > "${PID_DIR}/legacy.pid"
    log_info "legacy started (PID: $(cat ${PID_DIR}/legacy.pid))"
}

stop_service() {
    local name=$1
    local pid_file="${PID_DIR}/${name}.pid"

    if [[ -f "${pid_file}" ]]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            log_info "Stopped ${name} (PID: ${pid})"
        else
            log_warn "${name} process does not exist"
        fi
        rm -f "${pid_file}"
    else
        log_warn "${name} PID file does not exist"
    fi
}

stop_all() {
    log_info "Stopping all services..."
    stop_service "sandbox"
    stop_service "api"
    stop_service "legacy"
    log_info "All services stopped"
}

check_service() {
    local name=$1
    local port=$2
    local pid_file="${PID_DIR}/${name}.pid"

    if [[ -f "${pid_file}" ]]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} ${name}: 运行中 (PID: ${pid}, 端口: ${port})"
            return 0
        fi
    fi
    echo -e "  ${RED}○${NC} ${name}: 未运行"
    return 1
}

status() {
    echo "============================================"
    echo "X-Master MCP Services 状态"
    echo "============================================"
    check_service "api" "${API_PORT}" || true
    check_service "sandbox" "${SANDBOX_PORT}" || true
    if [[ "${START_LEGACY_EXECUTE_SERVER:-0}" == "1" ]]; then
        check_service "legacy" "${LEGACY_PORT}" || true
    fi
    echo "============================================"
    echo ""
    echo "MCP 端点:"
    echo "  - mcp-sandbox: http://127.0.0.1:${SANDBOX_PORT}/mcp"
    echo ""
    echo "日志文件: ${PID_DIR}/*.log"
}

start_all() {
    log_info "启动 X-Master MCP 服务 (api_proxy + mcp-sandbox)..."
    echo ""
    start_api_proxy
    start_sandbox
    start_legacy
    echo ""
    log_info "启动完成！"
    echo ""
    status
}

# Main logic
case "${1:-start}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        status
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
