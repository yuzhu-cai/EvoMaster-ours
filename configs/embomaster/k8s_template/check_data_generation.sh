#!/bin/bash
# 数据生成检查脚本
# 检查所有数据采集任务的完成状态和生成的数据

NAMESPACE="robotwin"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  数据生成状态检查${NC}"
echo -e "${CYAN}  时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 任务列表
declare -A TASKS=(
    ["put-object-cabinet"]="put_object_cabinet"
    ["place-phone-stand"]="place_phone_stand"
    ["open-laptop"]="open_laptop"
    ["hanging-mug"]="hanging_mug"
)

# 检查任务状态
echo -e "${BLUE}=== K8S 任务状态 ===${NC}"
for task_key in "${!TASKS[@]}"; do
    task_name="${TASKS[$task_key]}"
    echo ""
    echo -e "${CYAN}--- $task_name ---${NC}"
    
    for size in 10 1000; do
        job_name="robotwin-collect-${task_key}-${size}"
        
        # 获取任务状态
        status=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)
        failed=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null)
        active=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.active}' 2>/dev/null)
        
        if [ "$status" = "True" ]; then
            status_display="${GREEN}✓ Complete${NC}"
        elif [ "$failed" = "True" ]; then
            status_display="${RED}✗ Failed${NC}"
        elif [ "$active" = "1" ]; then
            status_display="${YELLOW}▶ Running${NC}"
        else
            status_display="${BLUE}⏳ Pending${NC}"
        fi
        
        printf "  %-30s %s\n" "$job_name ($size episodes)" "$status_display"
        
        # 获取 Pod 日志中的进度信息
        pod_name=$(kubectl get pods -n "$NAMESPACE" -l job-name="$job_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [ -n "$pod_name" ]; then
            progress=$(kubectl logs "$pod_name" -n "$NAMESPACE" --tail=5 2>/dev/null | grep -o "episodes collected: [0-9]*" | head -1)
            if [ -n "$progress" ]; then
                echo -e "    ${CYAN}→${NC} $progress"
            fi
        fi
    done
done

echo ""
echo -e "${BLUE}=== 数据文件检查 ===${NC}"

# 检查实际生成的数据文件
for task_key in "${!TASKS[@]}"; do
    task_name="${TASKS[$task_key]}"
    echo ""
    echo -e "${CYAN}--- $task_name ---${NC}"
    
    for config in demo_clean_10 demo_clean_1000; do
        data_dir="/data/yuanshuozhang/RoboTwin/data/${task_name}/${config}/data"
        expected_count=${config##*_}
        
        if [ -d "$data_dir" ]; then
            # 统计 hdf5 文件数量
            count=$(ls -1 "$data_dir"/*.hdf5 2>/dev/null | wc -l)
            
            if [ "$count" -ge "$expected_count" ]; then
                echo -e "  ${GREEN}✓${NC} $config: $count / $expected_count episodes ${GREEN}(Complete)${NC}"
            elif [ "$count" -gt 0 ]; then
                echo -e "  ${YELLOW}⚠${NC} $config: $count / $expected_count episodes ${YELLOW}(In Progress)${NC}"
            else
                echo -e "  ${RED}✗${NC} $config: 0 / $expected_count episodes ${RED}(No data)${NC}"
            fi
            
            # 显示最新的几个文件
            if [ "$count" -gt 0 ]; then
                latest_files=$(ls -1t "$data_dir"/*.hdf5 2>/dev/null | head -3)
                if [ -n "$latest_files" ]; then
                    echo "    Latest files:"
                    echo "$latest_files" | while read file; do
                        size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}')
                        mtime=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
                        echo "      $(basename $file) - $size - $mtime"
                    done
                fi
            fi
        else
            echo -e "  ${RED}✗${NC} $config: Directory not found"
        fi
    done
done

echo ""
echo -e "${BLUE}=== 数据完整性检查 ===${NC}"

# 检查数据完整性
for task_key in "${!TASKS[@]}"; do
    task_name="${TASKS[$task_key]}"
    
    for config in demo_clean_10 demo_clean_1000; do
        data_dir="/data/yuanshuozhang/RoboTwin/data/${task_name}/${config}/data"
        expected_count=${config##*_}
        
        if [ -d "$data_dir" ] && [ "$(ls -1 "$data_dir"/*.hdf5 2>/dev/null | wc -l)" -ge "$expected_count" ]; then
            # 检查第一个和最后一个文件是否存在
            first_file="$data_dir/episode0.hdf5"
            last_idx=$((expected_count - 1))
            last_file="$data_dir/episode${last_idx}.hdf5"
            
            if [ -f "$first_file" ] && [ -f "$last_file" ]; then
                echo -e "  ${GREEN}✓${NC} $task_name/$config: Data files complete (episode0 to episode${last_idx})"
            else
                echo -e "  ${YELLOW}⚠${NC} $task_name/$config: Some episode files missing"
            fi
        fi
    done
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo "检查完成"





