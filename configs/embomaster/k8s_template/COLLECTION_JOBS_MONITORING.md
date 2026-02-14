# 数据采集任务监控指南

## 📋 已提交的任务

### put_object_cabinet
- ✅ `robotwin-collect-put-object-cabinet-1000` (1000 episodes)
- ✅ `robotwin-collect-put-object-cabinet-10` (10 episodes)

### place_phone_stand
- ✅ `robotwin-collect-place-phone-stand-1000` (1000 episodes)
- ✅ `robotwin-collect-place-phone-stand-10` (10 episodes)

### open_laptop
- ✅ `robotwin-collect-open-laptop-1000` (1000 episodes)
- ✅ `robotwin-collect-open-laptop-10` (10 episodes)

### hanging_mug
- ✅ `robotwin-collect-hanging-mug-1000` (1000 episodes)
- ✅ `robotwin-collect-hanging-mug-10` (10 episodes)

**总计**: 8 个数据采集任务

## 🔍 监控命令

### 1. 查看所有任务状态

```bash
kubectl get jobs -n robotwin | grep -E "robotwin-collect-(put-object-cabinet|place-phone-stand|open-laptop|hanging-mug)"
```

### 2. 使用监控脚本（推荐）

```bash
cd /data/agents/openhands-ml-master/k8s_template
bash monitor_collection_jobs.sh
```

### 3. 查看特定任务状态

```bash
# 查看任务详情
kubectl describe job -n robotwin robotwin-collect-put-object-cabinet-1000

# 查看任务状态
kubectl get job -n robotwin robotwin-collect-put-object-cabinet-1000
```

### 4. 查看 Pod 状态

```bash
# 查看所有相关 Pod
kubectl get pods -n robotwin | grep -E "robotwin-collect-(put-object-cabinet|place-phone-stand|open-laptop|hanging-mug)"

# 查看特定任务的 Pod
kubectl get pods -n robotwin -l job-name=robotwin-collect-put-object-cabinet-1000
```

### 5. 查看实时日志

```bash
# 查看特定任务的日志（最后50行）
kubectl logs -n robotwin -l job-name=robotwin-collect-put-object-cabinet-1000 --tail=50 -f

# 查看所有任务的日志
for job in put-object-cabinet-1000 put-object-cabinet-10 \
           place-phone-stand-1000 place-phone-stand-10 \
           open-laptop-1000 open-laptop-10 \
           hanging-mug-1000 hanging-mug-10; do
    echo "=== robotwin-collect-$job ==="
    kubectl logs -n robotwin -l job-name=robotwin-collect-$job --tail=10
    echo ""
done
```

### 6. 持续监控（watch 模式）

```bash
# 每 5 秒刷新一次任务状态
watch -n 5 'kubectl get jobs -n robotwin | grep robotwin-collect'

# 或者使用 kubectl 的 watch
kubectl get jobs -n robotwin -w | grep robotwin-collect
```

## 📊 任务状态说明

- **Complete**: 任务成功完成
- **Running**: 任务正在运行
- **Failed**: 任务失败
- **Pending**: 任务等待调度

## 🔧 常用操作

### 删除任务

```bash
# 删除单个任务
kubectl delete job -n robotwin robotwin-collect-put-object-cabinet-1000

# 删除所有相关任务
kubectl delete jobs -n robotwin -l job-name=robotwin-collect-put-object-cabinet-1000
```

### 重新提交任务

```bash
# 先删除旧任务
kubectl delete job -n robotwin robotwin-collect-put-object-cabinet-1000

# 重新提交
kubectl apply -f /data/agents/openhands-ml-master/k8s_template/robotwin-collect-put_object_cabinet-1000.yaml
```

### 查看任务资源使用情况

```bash
# 查看 Pod 资源使用
kubectl top pods -n robotwin | grep robotwin-collect
```

## 📈 预计运行时间

- **10 episodes**: 约 10-20 分钟
- **1000 episodes**: 约 4-8 小时（取决于硬件性能）

## ✅ 验证数据采集完成

### 检查数据文件

```bash
# 在 Pod 中检查（需要进入 Pod）
kubectl exec -it -n robotwin <pod-name> -- bash
cd /workspace/RoboTwin
ls -la data/put_object_cabinet/demo_clean_1000/data/ | wc -l
ls -la data/place_phone_stand/demo_clean_1000/data/ | wc -l
ls -la data/open_laptop/demo_clean_1000/data/ | wc -l
ls -la data/hanging_mug/demo_clean_1000/data/ | wc -l
```

### 在主机上检查

```bash
# 检查数据目录
ls -la /data/yuanshuozhang/RoboTwin/data/put_object_cabinet/demo_clean_1000/data/ | wc -l
ls -la /data/yuanshuozhang/RoboTwin/data/place_phone_stand/demo_clean_1000/data/ | wc -l
ls -la /data/yuanshuozhang/RoboTwin/data/open_laptop/demo_clean_1000/data/ | wc -l
ls -la /data/yuanshuozhang/RoboTwin/data/hanging_mug/demo_clean_1000/data/ | wc -l
```

## 🚨 故障排查

### 如果任务失败

1. **查看 Pod 日志**:
   ```bash
   kubectl logs -n robotwin <pod-name> --tail=100
   ```

2. **查看 Pod 事件**:
   ```bash
   kubectl describe pod -n robotwin <pod-name>
   ```

3. **检查资源限制**:
   ```bash
   kubectl describe job -n robotwin <job-name>
   ```

### 常见问题

- **GPU 资源不足**: 检查集群 GPU 资源
- **存储空间不足**: 检查 `/data/yuanshuozhang/RoboTwin` 目录空间
- **配置文件缺失**: 检查 `task_config/demo_clean_1000.yml` 是否存在

## 📝 快速参考

```bash
# 一键查看所有任务状态
kubectl get jobs -n robotwin | grep robotwin-collect | awk '{print $1, $2, $3}'

# 一键查看所有 Pod 状态
kubectl get pods -n robotwin | grep robotwin-collect | awk '{print $1, $3, $4}'

# 一键查看所有任务日志（最后10行）
for job in $(kubectl get jobs -n robotwin | grep robotwin-collect | awk '{print $1}'); do
    echo "=== $job ==="
    kubectl logs -n robotwin -l job-name=$job --tail=10 2>/dev/null || echo "No logs"
    echo ""
done
```

