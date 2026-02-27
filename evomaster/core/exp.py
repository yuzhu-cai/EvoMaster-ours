"""EvoMaster Core - 基础类和通用流程

提供 Exp 和 Playground 的基础实现，供具体的 playground 继承使用。
"""

import json
import logging
from pathlib import Path
from evomaster.utils.types import TaskInstance
from typing import Any


class BaseExp:
    """实验基类

    定义单次实验的通用执行逻辑。
    具体 playground 可以继承并覆盖相关方法。
    """

    def __init__(self, agent, config):
        """初始化实验

        Args:
            agent: Agent 实例
            config: EvoMasterConfig 实例
        """
        self.agent = agent
        self.config = config
        self.results = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.run_dir = None

    @property
    def exp_name(self) -> str:
        """获取 Exp 名称（自动从类名推断）

        例如: SolverExp -> Solver, CriticExp -> Critic
        子类可以覆盖此属性来自定义名称。
        """
        class_name = self.__class__.__name__
        # 移除 "Exp" 后缀
        if class_name.endswith('Exp'):
            return class_name[:-3]
        return class_name

    def set_run_dir(self, run_dir: str | Path) -> None:
        """设置 run 目录

        Args:
            run_dir: Run 目录路径
        """
        self.run_dir = Path(run_dir)

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        """运行一次实验

        Args:
            task_description: 任务描述
            task_id: 任务 ID

        Returns:
            运行结果字典
        """
        # 创建任务实例
        task = TaskInstance(
            task_id=task_id,
            task_type="discovery",
            description=task_description,
        )

        # 运行 Agent
        self.logger.debug(f"Running task: {task_id}")
        trajectory = self.agent.run(task)

        # 保存结果
        result = {
            "task_id": task_id,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
            "trajectory": trajectory,
        }
        self.results.append(result)

        return {
            "trajectory": trajectory,
            "status": trajectory.status,
            "steps": len(trajectory.steps),
        }

    def save_results(self, output_file: str):
        """保存实验结果

        Args:
            output_file: 输出文件路径
        """
        output_data = []
        for result in self.results:
            output_data.append({
                "task_id": result["task_id"],
                "status": result["status"],
                "steps": result["steps"],
                "trajectory": result["trajectory"].model_dump(),
            })

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")


    def _extract_agent_response(self, trajectory: Any) -> str:
        """从轨迹中提取Agent的最终回答

        Args:
            trajectory: 执行轨迹

        Returns:
            Agent的回答文本
        """
        if not trajectory or not trajectory.dialogs:
            return ""

        # 获取最后一个对话
        last_dialog = trajectory.dialogs[-1]
        
        # 查找最后一个助手消息
        for message in reversed(last_dialog.messages):
            if hasattr(message, 'role') and message.role.value == 'assistant':
                if hasattr(message, 'content') and message.content:
                    return message.content
        
        return ""
