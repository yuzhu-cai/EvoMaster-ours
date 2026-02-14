#!/bin/bash
# 数据采集任务进度监控脚本
# 每 10 秒自动刷新显示所有任务的进度

NAMESPACE="robotwin"
REFRESH_INTERVAL=10

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 清屏函数
clear_screen() {
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  RoboTwin 数据采集任务进度监控${NC}"
    echo -e "${CYAN}  刷新间隔: ${REFRESH_INTERVAL} 秒${NC}"
    echo -e "${CYAN}  时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# 获取任务状态
get_job_status() {
    local job_name=$1
    kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status}' 2>/dev/null
}

# 获取 Pod 名称
get_pod_name() {
    local job_name=$1
    kubectl get pods -n "$NAMESPACE" -l job-name="$job_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

# 获取 Pod 状态
get_pod_status() {
    local pod_name=$1
    if [ -z "$pod_name" ]; then
        echo "NoPod"
    else
        kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown"
    fi
}

# 获取任务完成数
get_completion_count() {
    local job_name=$1
    kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0"
}

# 获取任务失败数
get_failure_count() {
    local job_name=$1
    kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.failed}' 2>/dev/null || echo "0"
}

# 获取任务活跃数
get_active_count() {
    local job_name=$1
    kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.active}' 2>/dev/null || echo "0"
}

# 获取日志最后一行（显示进度信息）
get_latest_log() {
    local pod_name=$1
    if [ -n "$pod_name" ]; then
        kubectl logs "$pod_name" -n "$NAMESPACE" --tail=1 2>/dev/null | head -1 | cut -c1-80
    else
        echo "No logs"
    fi
}

# 显示任务信息
display_job_info() {
    local job_name=$1
    local task_display=$2
    
    local pod_name=$(get_pod_name "$job_name")
    local pod_status=$(get_pod_status "$pod_name")
    local succeeded=$(get_completion_count "$job_name")
    local failed=$(get_failure_count "$job_name")
    local active=$(get_active_count "$job_name")
    local latest_log=$(get_latest_log "$pod_name")
    
    # 状态颜色
    local status_color=""
    local status_text=""
    
    if [ "$succeeded" = "1" ]; then
        status_color="$GREEN"
        status_text="✓ Complete"
    elif [ "$failed" = "1" ]; then
        status_color="$RED"
        status_text="✗ Failed"
    elif [ "$active" = "1" ] || [ "$pod_status" = "Running" ]; then
        status_color="$YELLOW"
        status_text="▶ Running"
    elif [ "$pod_status" = "Pending" ]; then
        status_color="$BLUE"
        status_text="⏳ Pending"
    else
        status_color="$NC"
        status_text="? Unknown"
    fi
    
    # 显示任务信息
    printf "%-35s " "$task_display"
    echo -e "${status_color}${status_text}${NC}"
    
    # 显示 Pod 信息
    if [ -n "$pod_name" ]; then
        printf "  Pod: %-40s " "$pod_name"
        echo -e "Status: ${status_color}${pod_status}${NC}"
    fi
    
    # 显示最新日志
    if [ -n "$latest_log" ] && [ "$latest_log" != "No logs" ]; then
        echo -e "  ${CYAN}Latest:${NC} $latest_log"
    fi
    
    echo ""
}

# 主循环
main() {
    while true; do
        clear_screen
        
        echo -e "${BLUE}=== put_object_cabinet ===${NC}"
        display_job_info "robotwin-collect-put-object-cabinet-1000" "  put_object_cabinet (1000)"
        display_job_info "robotwin-collect-put-object-cabinet-10" "  put_object_cabinet (10)"
        
        echo -e "${BLUE}=== place_phone_stand ===${NC}"
        display_job_info "robotwin-collect-place-phone-stand-1000" "  place_phone_stand (1000)"
        display_job_info "robotwin-collect-place-phone-stand-10" "  place_phone_stand (10)"
        
        echo -e "${BLUE}=== open_laptop ===${NC}"
        display_job_info "robotwin-collect-open-laptop-1000" "  open_laptop (1000)"
        display_job_info "robotwin-collect-open-laptop-10" "  open_laptop (10)"
        
        echo -e "${BLUE}=== hanging_mug ===${NC}"
        display_job_info "robotwin-collect-hanging-mug-1000" "  hanging_mug (1000)"
        display_job_info "robotwin-collect-hanging-mug-10" "  hanging_mug (10)"
        
        echo -e "${CYAN}========================================${NC}"
        echo -e "${CYAN}按 Ctrl+C 退出监控${NC}"
        echo ""
        
        sleep "$REFRESH_INTERVAL"
    done
}

# 捕获 Ctrl+C
trap 'echo -e "\n${YELLOW}监控已停止${NC}"; exit 0' INT

# 运行主循环
main

