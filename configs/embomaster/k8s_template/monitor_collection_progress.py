#!/usr/bin/env python3
"""
数据采集任务进度监控程序
每 10 秒自动刷新显示所有任务的进度
"""

import subprocess
import time
import os
from datetime import datetime
from collections import OrderedDict

# 配置
NAMESPACE = "robotwin"
REFRESH_INTERVAL = 10

# 任务列表
TASKS = OrderedDict([
    ("put-object-cabinet-1000", "put_object_cabinet (1000)"),
    ("put-object-cabinet-10", "put_object_cabinet (10)"),
    ("place-phone-stand-1000", "place_phone_stand (1000)"),
    ("place-phone-stand-10", "place_phone_stand (10)"),
    ("open-laptop-1000", "open_laptop (1000)"),
    ("open-laptop-10", "open_laptop (10)"),
    ("hanging-mug-1000", "hanging_mug (1000)"),
    ("hanging-mug-10", "hanging_mug (10)"),
])

# ANSI 颜色代码
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


def run_cmd(cmd):
    """执行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except:
        return ""


def get_job_status(task_key):
    """获取任务状态"""
    job_name = f"robotwin-collect-{task_key}"
    
    # 检查任务是否存在
    exists = run_cmd(f"kubectl get job {job_name} -n {NAMESPACE} 2>/dev/null")
    if not exists:
        return "NotFound", "", ""
    
    # 获取完成状态
    succeeded = run_cmd(
        f"kubectl get job {job_name} -n {NAMESPACE} "
        f"-o jsonpath='{{.status.succeeded}}' 2>/dev/null"
    )
    failed = run_cmd(
        f"kubectl get job {job_name} -n {NAMESPACE} "
        f"-o jsonpath='{{.status.failed}}' 2>/dev/null"
    )
    active = run_cmd(
        f"kubectl get job {job_name} -n {NAMESPACE} "
        f"-o jsonpath='{{.status.active}}' 2>/dev/null"
    )
    
    # 获取 Pod 名称
    pod_name = run_cmd(
        f"kubectl get pods -n {NAMESPACE} -l job-name={job_name} "
        f"-o jsonpath='{{.items[0].metadata.name}}' 2>/dev/null"
    )
    
    # 获取 Pod 状态
    pod_status = ""
    if pod_name:
        pod_status = run_cmd(
            f"kubectl get pod {pod_name} -n {NAMESPACE} "
            f"-o jsonpath='{{.status.phase}}' 2>/dev/null"
        )
    
    # 确定状态
    if succeeded == "1":
        return "Complete", pod_name, pod_status
    elif failed == "1":
        return "Failed", pod_name, pod_status
    elif active == "1" or pod_status == "Running":
        return "Running", pod_name, pod_status
    elif pod_status == "Pending":
        return "Pending", pod_name, pod_status
    else:
        return "Unknown", pod_name, pod_status


def get_latest_log(pod_name, lines=1):
    """获取最新日志"""
    if not pod_name:
        return ""
    
    log = run_cmd(
        f"kubectl logs {pod_name} -n {NAMESPACE} --tail={lines} 2>/dev/null"
    )
    
    # 提取进度信息
    if "episodes collected" in log:
        for line in log.split('\n'):
            if "episodes collected" in line:
                return line.strip()
    elif log:
        # 返回最后一行
        return log.split('\n')[-1][:80]
    
    return ""


def format_status(status):
    """格式化状态显示"""
    status_map = {
        "Complete": f"{Colors.GREEN}✓ Complete{Colors.NC}",
        "Failed": f"{Colors.RED}✗ Failed{Colors.NC}",
        "Running": f"{Colors.YELLOW}▶ Running{Colors.NC}",
        "Pending": f"{Colors.BLUE}⏳ Pending{Colors.NC}",
        "Unknown": f"{Colors.MAGENTA}? Unknown{Colors.NC}",
        "NotFound": f"{Colors.RED}✗ Not Found{Colors.NC}",
    }
    return status_map.get(status, status)


def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')


def display_progress():
    """显示进度"""
    clear_screen()
    
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.CYAN}  RoboTwin 数据采集任务进度监控{Colors.NC}")
    print(f"{Colors.CYAN}  刷新间隔: {REFRESH_INTERVAL} 秒{Colors.NC}")
    print(f"{Colors.CYAN}  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
    print()
    
    # 按任务分组显示
    current_group = None
    for task_key, task_name in TASKS.items():
        # 提取任务组名
        group = task_key.split('-')[0] + '-' + task_key.split('-')[1]
        if group != current_group:
            if current_group is not None:
                print()
            print(f"{Colors.BLUE}=== {group.replace('-', '_')} ==={Colors.NC}")
            current_group = group
        
        status, pod_name, pod_status = get_job_status(task_key)
        latest_log = get_latest_log(pod_name)
        
        # 显示任务信息
        print(f"  {task_name:35s} {format_status(status)}")
        
        if pod_name:
            print(f"    Pod: {pod_name:40s} Status: {pod_status}")
        
        if latest_log:
            print(f"    {Colors.CYAN}→{Colors.NC} {latest_log}")
    
    print()
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"按 {Colors.YELLOW}Ctrl+C{Colors.NC} 退出监控")
    print()


def main():
    """主循环"""
    try:
        while True:
            display_progress()
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}监控已停止{Colors.NC}")
        exit(0)


if __name__ == "__main__":
    main()

