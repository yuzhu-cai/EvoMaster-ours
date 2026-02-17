# 数据采集进度监控使用指南

## 📊 监控脚本说明

我创建了三个监控脚本，你可以选择使用：

### 1. monitor_collection_progress.sh (推荐)
功能最全的 Bash 脚本，显示详细信息

### 2. monitor_collection_progress_simple.sh
简化版 Bash 脚本，显示简洁信息

### 3. monitor_collection_progress.py
Python 版本，功能与 Bash 版本类似

## 🚀 使用方法

### 方法1: 使用 Bash 脚本（推荐）

```bash
cd /data/agents/openhands-ml-master/k8s_template
bash monitor_collection_progress.sh
```

### 方法2: 使用简化版

```bash
cd /data/agents/openhands-ml-master/k8s_template
bash monitor_collection_progress_simple.sh
```

### 方法3: 使用 Python 版本

```bash
cd /data/agents/openhands-ml-master/k8s_template
python3 monitor_collection_progress.py
```

## 📋 显示内容

监控脚本会每 10 秒自动刷新，显示：

1. **任务状态**:
   - ✓ Complete (绿色) - 任务完成
   - ▶ Running (黄色) - 任务运行中
   - ✗ Failed (红色) - 任务失败
   - ⏳ Pending (蓝色) - 任务等待中

2. **Pod 信息**:
   - Pod 名称
   - Pod 状态

3. **进度信息**:
   - 最新日志（包含 "episodes collected" 信息）

## 🎯 快速启动

```bash
# 进入目录
cd /data/agents/openhands-ml-master/k8s_template

# 启动监控（选择其中一个）
bash monitor_collection_progress.sh
# 或
python3 monitor_collection_progress.py
```

## ⌨️ 操作说明

- **退出监控**: 按 `Ctrl+C`
- **自动刷新**: 每 10 秒自动刷新一次
- **清屏**: 每次刷新会自动清屏

## 📊 示例输出

```
========================================
  RoboTwin 数据采集任务进度监控
  刷新间隔: 10 秒
  时间: 2026-01-05 18:00:00
========================================

=== put_object_cabinet ===
  put_object_cabinet (1000)      ✓ Complete
    Pod: robotwin-collect-put-object-cabinet-1000-xxxxx
    Status: Succeeded
    → Total episodes collected: 1000 / 1000

  put_object_cabinet (10)        ▶ Running
    Pod: robotwin-collect-put-object-cabinet-10-xxxxx
    Status: Running
    → Processing episode: 5 / 10

...
```

## 🔧 自定义刷新间隔

如果想修改刷新间隔，编辑脚本文件，修改 `REFRESH_INTERVAL` 变量：

```bash
# 在脚本中修改
REFRESH_INTERVAL=5  # 改为 5 秒刷新一次
```

## 💡 提示

1. 确保有 `kubectl` 访问权限
2. 确保可以访问 `robotwin` namespace
3. 如果任务很多，可能需要等待几秒才能获取所有信息

## 🐛 故障排查

如果脚本无法运行：

1. **检查 kubectl 权限**:
   ```bash
   kubectl get jobs -n robotwin
   ```

2. **检查脚本权限**:
   ```bash
   chmod +x monitor_collection_progress.sh
   ```

3. **检查 Python 版本**（如果使用 Python 脚本）:
   ```bash
   python3 --version
   ```

