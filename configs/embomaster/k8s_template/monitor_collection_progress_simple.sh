#!/bin/bash
# 简化版数据采集任务进度监控脚本
# 每 10 秒自动刷新

NAMESPACE="robotwin"
REFRESH_INTERVAL=10

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 任务列表
declare -A TASKS=(
    ["put-object-cabinet-1000"]="put_object_cabinet (1000)"
    ["put-object-cabinet-10"]="put_object_cabinet (10)"
    ["place-phone-stand-1000"]="place_phone_stand (1000)"
    ["place-phone-stand-10"]="place_phone_stand (10)"
    ["open-laptop-1000"]="open_laptop (1000)"
    ["open-laptop-10"]="open_laptop (10)"
    ["hanging-mug-1000"]="hanging_mug (1000)"
    ["hanging-mug-10"]="hanging_mug (10)"
)

get_status() {
    local job_name="robotwin-collect-$1"
    local status=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
    local failed=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
    
    if [ "$status" = "True" ]; then
        echo -e "${GREEN}✓ Complete${NC}"
    elif [ "$failed" = "True" ]; then
        echo -e "${RED}✗ Failed${NC}"
    else
        local active=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.active}' 2>/dev/null)
        if [ "$active" = "1" ]; then
            echo -e "${YELLOW}▶ Running${NC}"
        else
            echo -e "${BLUE}⏳ Pending${NC}"
        fi
    fi
}

get_progress() {
    local job_name="robotwin-collect-$1"
    local pod=$(kubectl get pods -n "$NAMESPACE" -l job-name="$job_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$pod" ]; then
        kubectl logs "$pod" -n "$NAMESPACE" --tail=1 2>/dev/null | grep -o "episodes collected: [0-9]*" | head -1 || echo ""
    fi
}

while true; do
    clear
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  RoboTwin 数据采集进度监控${NC}"
    echo -e "${CYAN}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    for task_key in "${!TASKS[@]}"; do
        task_name="${TASKS[$task_key]}"
        status=$(get_status "$task_key")
        progress=$(get_progress "$task_key")
        
        printf "%-35s %s\n" "$task_name" "$status"
        if [ -n "$progress" ]; then
            echo -e "  ${CYAN}→${NC} $progress"
        fi
        echo ""
    done
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "按 ${YELLOW}Ctrl+C${NC} 退出 | 每 ${REFRESH_INTERVAL} 秒刷新"
    
    sleep "$REFRESH_INTERVAL"
done

