#!/bin/bash
# 监控数据采集任务的脚本
# 用法: bash monitor_collection_jobs.sh

echo "=========================================="
echo "RoboTwin 数据采集任务监控"
echo "=========================================="
echo ""

# 任务列表
TASKS=(
    "put-object-cabinet-1000"
    "put-object-cabinet-10"
    "place-phone-stand-1000"
    "place-phone-stand-10"
    "open-laptop-1000"
    "open-laptop-10"
    "hanging-mug-1000"
    "hanging-mug-10"
)

echo "=== 任务状态概览 ==="
kubectl get jobs -n robotwin | grep -E "robotwin-collect-(put-object-cabinet|place-phone-stand|open-laptop|hanging-mug)" | \
    awk '{printf "%-50s %-10s %-10s %-10s\n", $1, $2, $3, $4}'
echo ""

echo "=== 详细状态 ==="
for task in "${TASKS[@]}"; do
    job_name="robotwin-collect-${task}"
    echo ""
    echo "--- $job_name ---"
    kubectl get job "$job_name" -n robotwin 2>/dev/null || echo "Job not found"
    
    # 获取 Pod 状态
    pod_name=$(kubectl get pods -n robotwin -l job-name="$job_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$pod_name" ]; then
        echo "Pod: $pod_name"
        kubectl get pod "$pod_name" -n robotwin -o jsonpath='{.status.phase}' 2>/dev/null && echo ""
        
        # 显示日志最后几行
        echo "Latest logs (last 5 lines):"
        kubectl logs "$pod_name" -n robotwin --tail=5 2>/dev/null || echo "No logs available"
    else
        echo "No pod found"
    fi
done

echo ""
echo "=== 快速命令 ==="
echo "查看所有任务: kubectl get jobs -n robotwin | grep robotwin-collect"
echo "查看特定任务日志: kubectl logs -n robotwin -l job-name=robotwin-collect-<task-name> --tail=50"
echo "删除任务: kubectl delete job -n robotwin robotwin-collect-<task-name>"
echo ""

