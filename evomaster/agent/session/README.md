# Session - Agent 与 Env 交互的介质

Session 是 Agent 与远程集群环境交互的中间层，提供统一的接口来执行命令、传输文件等。

## 目录结构

- `base.py` - Session 抽象基类，定义标准接口
- `local.py` - 本地 Session 实现，在本地直接执行命令
- `docker.py` - Docker Session 实现，使用 Docker 容器提供隔离的执行环境

## 核心类

### BaseSession（base.py）
Session 的抽象基类，定义所有 Session 实现必须提供的接口：

- `open()` / `close()` - 会话生命周期管理
- `exec_bash(command)` - 执行 Bash 命令
- `upload(local, remote)` - 上传文件
- `download(remote)` - 下载文件
- `read_file()` / `write_file()` - 文本文件读写
- `path_exists()` / `is_file()` / `is_directory()` - 路径检查

### LocalSession（local.py）
本地 Session 实现，在本地直接执行命令：

- 使用 subprocess 直接执行 bash 命令
- 文件操作为本地复制/读写
- 适合开发和测试
- 无需任何外部依赖（Docker、集群等）

### DockerSession（docker.py）
基于 Docker 的 Session 实现，提供隔离的执行环境：

- 使用 Docker 容器作为执行环境
- 通过 tmux 维持持久化的 bash 会话
- 支持环境变量、工作目录等状态保持
- 支持资源限制（内存、CPU）和卷挂载

## 使用示例

### 本地 Session

```python
from evomaster.agent.session import LocalSession, LocalSessionConfig

# 创建配置
config = LocalSessionConfig(timeout=30)

# 使用 Session
with LocalSession(config) as session:
    # 执行命令
    result = session.exec_bash("python --version")
    print(result["stdout"])

    # 上传文件（本地复制）
    session.upload("/local/path", "/tmp/remote.py")

    # 下载文件（本地读取）
    content = session.download("/tmp/file.txt")
```

### Docker Session

```python
from evomaster.agent.session import DockerSession, DockerSessionConfig

# 创建配置
config = DockerSessionConfig(
    image="python:3.11-slim",
    memory_limit="4g",
    cpu_limit=2.0,
)

# 使用 Session
with DockerSession(config) as session:
    # 执行命令
    result = session.exec_bash("python --version")
    print(result["stdout"])

    # 上传文件
    session.upload("/local/path", "/workspace/remote.py")

    # 下载文件
    content = session.download("/workspace/output.txt")
```

## 设计特点

1. **抽象接口** - BaseSession 定义标准接口，便于多种实现（本地、远程、Kubernetes 等）
2. **多种实现** - 支持本地、Docker、以及未来的其他环境
3. **隔离环境** - Docker 容器提供完整的隔离执行环境
4. **持久化会话** - 使用 tmux 维持 bash 状态，支持长期实验
5. **资源管理** - 支持内存、CPU 等资源限制
6. **上下文管理** - 实现了 Python 上下文管理器接口

## 配置参数

### SessionConfig（基础配置）
- `timeout` - 命令执行超时时间（秒），默认 300
- `workspace_path` - 工作空间路径，默认 `/workspace`

### LocalSessionConfig（本地 Session 配置）
继承 `SessionConfig`，额外参数：
- `encoding` - 文件编码，默认 `utf-8`

### DockerSessionConfig（Docker Session 配置）
继承 `SessionConfig`，额外参数：
- `image` - Docker 镜像名称，默认 `python:3.11-slim`
- `container_name` - 容器名称，自动生成则为 None
- `memory_limit` - 内存限制，默认 `4g`
- `cpu_limit` - CPU 限制，默认 2.0
- `volumes` - 卷挂载 {主机路径: 容器路径}
- `env_vars` - 环境变量
- `auto_remove` - 容器结束后是否自动删除，默认 True

## 后续扩展

可在此基础上实现：
- `RemoteSession` - SSH 连接远程服务器
- `KubernetesSession` - Kubernetes 集群执行
- `RaySession` - Ray 分布式框架

