"""X-Master Playground Implementation

Implements the complete X-Master workflow:
1. Solver: Generate initial solutions
2. Critic: Critique and correct solutions
3. Rewriter: Rewrite and integrate solutions
4. Selector: Select the best solution
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure evomaster module can be imported
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster import TaskInstance

from .exp import SolveExp, CritiqueExp, RewriteExp, SelectExp


@register_playground("x_master")
class XMasterPlayground(BasePlayground):
    """X-Master Playground

    Coordinates four Exp classes to implement the complete X-Master workflow.
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize X-Master Playground.

        Args:
            config_dir: Path to configuration directory, default is configs/xmaster/
            config_path: Full path to configuration file
        """
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "x_master"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("solver_agent", "critic_agent", "rewriter_agent", "selector_agent")

        # Storage for intermediate results
        self.solver_results = []
        self.critic_results = []
        self.rewriter_results = []
        self.selector_results = []

        self.mcp_manager = None

    def setup(self) -> None:
        """Initialize all components.

        Creates four Agents and corresponding Exp instances.
        """
        self.logger.info("Setting up X-Master playground...")

        # 1. Create Session
        self._setup_session()

        # 2. Create tool registry
        self._setup_tools()

        # 3. Load workflow parameters from configuration
        self._load_workflow_config()

        # 4. Create agents for the four components
        self._setup_agents()

        self.logger.info("X-Master playground setup complete")

    def _load_workflow_config(self) -> None:
        """Load workflow parameters from configuration."""
        xmaster_config = getattr(self.config, 'xmaster', {})
        if not xmaster_config:
            xmaster_config = {}

        self.agent_num = xmaster_config.get('agent_num', 1)
        self.max_workers = xmaster_config.get('max_workers', 1)
        self.parallel = xmaster_config.get('parallel', True)

        self.logger.info(f"Workflow config: agent_num={self.agent_num}, max_workers={self.max_workers}")

    def _create_exp(self, exp_index: int, exp_name: str):
        """Create a multi-agent experiment instance.

        Args:
            exp_index: Index of the experiment.
            exp_name: Name of the experiment type ('solve', 'critique', 'rewrite', 'select').

        Returns:
            An instance of the corresponding Exp class.

        Raises:
            ValueError: If exp_name is unknown.
            RuntimeError: If creation fails.
        """
        exp = None

        if exp_name == "solve":
            solver_agent_copy = self.copy_agent(
                self.agents.solver_agent,
                new_agent_name=f"solve_exp_{exp_index}"
            ) if self.agents.solver_agent else None
            exp = SolveExp(
                solver_agent=solver_agent_copy,
                config=self.config,
                index=exp_index
            )

        elif exp_name == "critique":
            critic_agent_copy = self.copy_agent(
                self.agents.critic_agent,
                new_agent_name=f"critique_exp_{exp_index}"
            ) if self.agents.critic_agent else None
            exp = CritiqueExp(
                critic_agent=critic_agent_copy,
                config=self.config,
                index=exp_index
            )

        elif exp_name == "rewrite":
            rewriter_agent_copy = self.copy_agent(
                self.agents.rewriter_agent,
                new_agent_name=f"rewrite_exp_{exp_index}"
            ) if self.agents.rewriter_agent else None
            exp = RewriteExp(
                rewriter_agent=rewriter_agent_copy,
                config=self.config,
                index=exp_index
            )

        elif exp_name == "select":
            selector_agent_copy = self.copy_agent(
                self.agents.selector_agent,
                new_agent_name=f"select_exp_{exp_index}"
            ) if self.agents.selector_agent else None
            exp = SelectExp(
                selector_agent=selector_agent_copy,
                config=self.config,
                index=exp_index
            )

        else:
            raise ValueError(f"Unknown exp_name: {exp_name}. Expected one of: solve, critique, rewrite, select")

        if exp is None:
            raise RuntimeError(f"Failed to create exp: {exp_name}")

        return exp

    def _extract_solutions_from_results(self, results: List) -> List[str]:
        """Extract list of solutions from Exp results.

        Args:
            results: List of result dictionaries from Solver experiments.

        Returns:
            List of extracted solution strings.
        """
        solutions = []
        # Directly look for "solver_result" key
        for result in results:
            index = result['exp_index']
            key = "solver_result"
            if key in result and result[key] is not None:
                solutions.append(result[key])
                self.logger.info(f"index:{index} found {key}: {result[key][:50]}...")
            elif key in result:
                self.logger.warning(f"index:{index} {key} is None, skipping")
        self.logger.info(f"Finally extracted {len(solutions)} solutions")
        return solutions

    def _extract_corrected_solutions(self, results: List) -> List[str]:
        """Extract corrected solutions from Critic results.

        Args:
            results: List of result dictionaries from CritiqueExp runs.

        Returns:
            List of corrected solution strings.
        """
        solutions = []
        # Directly look for "critic_result" key
        for result in results:
            index = result['exp_index']
            key = "critic_result"
            if key in result and result[key] is not None:
                solutions.append(result[key])
                self.logger.info(f"index:{index} found {key}: {result[key][:50]}...")
            elif key in result:
                self.logger.warning(f"index:{index} {key} is None, skipping")
        self.logger.info(f"Finally extracted {len(solutions)} corrected solutions")
        return solutions

    def _extract_rewritten_solutions(self, results: List) -> List[str]:
        """Extract rewritten solutions from Rewriter results.

        Args:
            results: List of result dictionaries from RewriteExp runs.

        Returns:
            List of rewritten solution strings.
        """
        solutions = []
        # Directly look for "rewriter_result" key
        for result in results:
            index = result['exp_index']
            key = "rewriter_result"
            if key in result and result[key] is not None:
                solutions.append(result[key])
                self.logger.info(f"index:{index} found {key}: {result[key][:50]}...")
            elif key in result:
                self.logger.warning(f"index:{index} {key} is None, skipping")
        self.logger.info(f"Finally extracted {len(solutions)} rewritten solutions")
        return solutions

    def _extract_selected_solution(self, results: Dict[str, Any]) -> str:
        """Extract the selected solution from Selector results.

        Args:
            results: Result dictionary from SelectExp run.

        Returns:
            Selected solution string.
        """
        key = "selector_result"
        solution = results[key]
        self.logger.info(f"Found {key}: {results[key][:50]}...")
        return solution

    def _run_with_parallel(self, task_description: str, task_id: str = None):
        """Run the X-Master workflow in parallel mode.

        Args:
            task_description: Description of the task.
            task_id: Optional task identifier.

        Returns:
            Tuple (original_solutions, corrected_solutions, rewritten_solutions, selected_solution)
        """
        self.logger.info(f"=== Parallel Process ({self.agent_num} agents) ===")
        from functools import partial

        # ---------- Phase 1: Solver (parallel) ----------
        self.logger.info(f"=== Phase 1: Solver (parallel, {self.agent_num} agents) ===")
        solver_tasks = []
        for i in range(self.max_workers):
            exp = self._create_exp(exp_index=i, exp_name="solve")
            task_func = partial(
                exp.run,
                task_description=task_description,
                task_id=f"{task_id}_solver",
            )
            solver_tasks.append(task_func)

        solver_results_list = self.execute_parallel_tasks(solver_tasks, max_workers=self.max_workers)
        self.solver_results = solver_results_list
        original_solutions = self._extract_solutions_from_results(self.solver_results)
        self.logger.info(f"Solver generated {len(original_solutions)} solutions")

        # ---------- Phase 2: Critic (parallel, one-to-one) ----------
        self.logger.info(f"=== Phase 2: Critic (parallel, {self.agent_num} agents) ===")
        critic_tasks = []
        for i in range(self.agent_num):
            exp = self._create_exp(exp_index=i, exp_name="critique")
            task_func = partial(
                exp.run,
                task_description=task_description,
                solution=original_solutions[i],   # pass corresponding solution only
                task_id=f"{task_id}_critic"
            )
            critic_tasks.append(task_func)

        critic_results_list = self.execute_parallel_tasks(critic_tasks, max_workers=self.max_workers)
        self.critic_results = critic_results_list
        corrected_solutions = self._extract_corrected_solutions(self.critic_results)
        self.logger.info(f"Critic generated {len(corrected_solutions)} corrected solutions")

        # ---------- Phase 3: Rewriter (parallel, all solutions) ----------
        self.logger.info(f"=== Phase 3: Rewriter (parallel, {self.agent_num} agents) ===")
        rewriter_tasks = []
        for i in range(self.agent_num):
            exp = self._create_exp(exp_index=i, exp_name="rewrite")
            task_func = partial(
                exp.run,
                task_description=task_description,
                solutions=corrected_solutions,       # pass all corrected solutions
                task_id=f"{task_id}_rewriter"
            )
            rewriter_tasks.append(task_func)

        rewriter_results_list = self.execute_parallel_tasks(rewriter_tasks, max_workers=self.max_workers)
        self.rewriter_results = rewriter_results_list
        rewritten_solutions = self._extract_rewritten_solutions(self.rewriter_results)
        self.logger.info(f"Rewriter generated {len(rewritten_solutions)} rewritten solutions")

        # ---------- Phase 4: Selector (single agent) ----------
        self.logger.info("=== Phase 4: Selector ===")
        selector_exp = self._create_exp(exp_index=0, exp_name="select")
        self.selector_results = selector_exp.run(
            task_description=task_description,
            solutions=rewritten_solutions,
            task_id=f"{task_id}_selector"
        )
        selected_solution = self._extract_selected_solution(self.selector_results)
        self.logger.info("Selector completed, best solution selected")

        return original_solutions, corrected_solutions, rewritten_solutions, selected_solution

    def _run_with_serial(self, task_description: str, task_id: str = None):
        """Run the X-Master workflow in serial mode.

        Args:
            task_description: Description of the task.
            task_id: Optional task identifier.

        Returns:
            Tuple (original_solutions, corrected_solutions, rewritten_solutions, selected_solution)
        """
        self.logger.info(f"=== Serial Process ({self.agent_num} agents) ===")

        # 1. Solver phase: generate initial solutions
        self.logger.info(f"=== Phase 1: Solver (serial, {self.agent_num} agents) ===")
        for i in range(self.agent_num):
            exp = self._create_exp(exp_index=i, exp_name="solve")
            solver_results = exp.run(
                task_description=task_description,
                task_id=f"{task_id}_solver"
            )
            self.solver_results.append(solver_results)

        original_solutions = self._extract_solutions_from_results(self.solver_results)
        self.logger.info(f"Solver generated {len(original_solutions)} solutions")

        # 2. Critic phase: critique and correct solutions
        self.logger.info(f"=== Phase 2: Critic (serial, {self.agent_num} agents) ===")
        for i in range(self.agent_num):
            exp = self._create_exp(exp_index=i, exp_name="critique")
            critic_results = exp.run(
                task_description=task_description,
                solution=original_solutions[i],
                task_id=f"{task_id}_critic"
            )
            self.critic_results.append(critic_results)

        corrected_solutions = self._extract_corrected_solutions(self.critic_results)
        self.logger.info(f"Critic generated {len(corrected_solutions)} corrected solutions")

        # 3. Rewriter phase: rewrite and integrate solutions
        self.logger.info(f"=== Phase 3: Rewriter (serial, {self.agent_num} agents) ===")
        for i in range(self.agent_num):
            exp = self._create_exp(exp_index=i, exp_name="rewrite")
            rewriter_results = exp.run(
                task_description=task_description,
                solutions=corrected_solutions,
                task_id=f"{task_id}_rewriter"
            )
            self.rewriter_results.append(rewriter_results)

        rewritten_solutions = self._extract_rewritten_solutions(self.rewriter_results)
        self.logger.info(f"Rewriter generated {len(rewritten_solutions)} rewritten solutions")

        # 4. Selector phase: select best solution
        self.logger.info("=== Phase 4: Selector ===")
        exp = self._create_exp(exp_index=0, exp_name="select")
        self.selector_results = exp.run(
            task_description=task_description,
            solutions=rewritten_solutions,
            task_id=f"{task_id}_selector"
        )
        selected_solution = self._extract_selected_solution(self.selector_results)
        self.logger.info("Selector completed, best solution selected")

        return original_solutions, corrected_solutions, rewritten_solutions, selected_solution

    def run_xmaster_workflow(self, task_description: str, task_id: str = None) -> Dict[str, Any]:
        """Run the complete X-Master workflow.

        Args:
            task_description: Description of the task.
            task_id: Task identifier (for batch processing).

        Returns:
            Dictionary containing complete X-Master workflow results.
        """
        if not task_id:
            task_id = "xmaster_task_001"

        self.logger.info(f"Starting X-Master workflow for task: {task_id}")
        self.logger.info(f"Task description: {task_description[:100]}...")

        if self.parallel:
            original_solutions, corrected_solutions, rewritten_solutions, selected_solution = self._run_with_parallel(task_description, task_id)
        else:
            original_solutions, corrected_solutions, rewritten_solutions, selected_solution = self._run_with_serial(task_description, task_id)

        final_result = {
            "status": "completed",
            "task_id": task_id,
            "task_description": task_description,
            "final_solution": selected_solution,
            "phase_results": {
                "solver": original_solutions,
                "critic": corrected_solutions,
                "rewriter": rewritten_solutions,
                "selector": selected_solution
            },
            "solutions_summary": {
                "original_count": len(original_solutions),
                "corrected_count": len(corrected_solutions),
                "rewritten_count": len(rewritten_solutions)
            },
            "trajectory": {
                "solver_trajectory": self.solver_results,
                "critic_trajectory": self.critic_results,
                "rewriter_trajectory": self.rewriter_results,
                "selector_trajectory": self.selector_results
            }
        }

        self.logger.info("X-Master workflow completed successfully")
        return final_result

    def run(self, task_description: str, output_file: str | None = None) -> Dict[str, Any]:
        """Run the X-Master workflow (overrides base class method).

        Args:
            task_description: Description of the task.
            output_file: Optional file to save results.

        Returns:
            Dictionary with run results.
        """
        try:
            self.setup()

            # Set trajectory file path (using base class method for unified directory structure)
            self._setup_trajectory_file(output_file)

            # Run the full X-Master workflow
            task_id = getattr(self, 'task_id', None)
            final_result = self.run_xmaster_workflow(task_description, task_id)

            return final_result

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources.

        Overrides base class method, cleans all agents and Exp instances.
        """
        # Clean up base class resources
        super().cleanup()

        # Clear results
        self.solver_results = None
        self.critic_results = None
        self.rewriter_results = None
        self.selector_results = None

        self.logger.debug("X-Master resources cleaned up")