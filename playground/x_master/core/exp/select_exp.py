import logging
import json
import re
from typing import Any
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from evomaster import TaskInstance
from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from .utils import strip_think_and_exec, extract_agent_response


class SelectExp(BaseExp):
    """Select experiment implementation for X-Master.

    Implements the Selecting stage workflow: aggregate all solutions from the previous stage
    and choose the best solution.
    """

    @property
    def exp_name(self) -> str:
        """Return the name of the experiment stage."""
        return "Selecting"

    def __init__(self, selector_agent,  config , index=0):
        """Initialize the SelectExp experiment class.

        Args:
            selector_agent: Selector Agent instance
            config: EvoMasterConfig instance
            index: index assigned when running multiple identical experiments in parallel
        """

        super().__init__(selector_agent, config)
        self.selector = selector_agent
        self.index = index
        self.logger = logging.getLogger(self.__class__.__name__)

        self.selector._current_exp_name = self.exp_name
        self.selector._current_exp_index = self.index

    def run(self, task_description:str,task_id:str = "exp_001", solutions:List[str] = None) -> dict:
            """Run the Selector experiment.

            Workflow: a Selector Agent aggregates all solutions from the previous module and selects the best one.

            Args:
                task_description: the task description
                task_id: the task identifier
                solutions: list of solutions received from the previous module
            Returns:
                A dictionary containing execution results
            """
            results = {
                'task_id':task_id,
                'steps':0,
                'task_description': task_description,
                'exp_index': self.index,
                'status': 'running',
            }
            if solutions is None:
                self.logger.error(f"selector-agent task execution failed: Solutions is None", exc_info=True)
                results['status'] = 'failed'
                results['error'] = "Solutions is None"
                return super().run(task_description, task_id)


            try:
                if self.selector:
                    self.logger.info("="*60)
                    self.logger.info("Selector : Selecting the best solution...")
                    self.logger.info("=" * 60)

                    selector_task = TaskInstance(
                        task_id = f"{task_id}_selector",
                        task_type = "selector",
                        description=task_description,
                        input_data={},
                    )

                    original_format_kwargs = self.selector._prompt_format_kwargs.copy()

                    # Format solutions (clean each with strip_think_and_exec)
                    responses = self._format_solutions_prompt(solutions)

                    try:
                        # Set current experiment info for trajectory recording
                        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=0)
                        self.selector._prompt_format_kwargs.update({
                            'Responses':responses
                        })
                        selector_trajectory = self.selector.run(selector_task)
                        results['selector_trajectory'] = selector_trajectory

                        # Extract raw LLM response
                        selector_response = extract_agent_response(selector_trajectory)
                        results['selector_response'] = selector_response

                        # Parse selection result and return the chosen original solution
                        selected_solution = self._parse_selector_choice(selector_response, solutions)
                        results['selector_result'] = selected_solution
                        results['selected_index'] = self._get_selected_index(selector_response, len(solutions))
                        
                        self.logger.info("Selecting completed")
                    except Exception as e:
                        self.logger.error(f"Selector task failed: {e}", exc_info=True)
                        results['selector_trajectory'] = None
                        results['selector_result'] = None
                        self.logger.info("Selecting failed")

                    self.selector._prompt_format_kwargs = original_format_kwargs


                    results['status'] = 'completed'
                    self.logger.info("Selector-agent task execution completed")

            except Exception as e:
                self.logger.error(f"Selector-agent task execution failed: {e}", exc_info=True)
                results['status'] = 'failed'
                results['error'] = str(e)

            self.results.append(results)
            return results

    def _format_solutions_prompt(self, solutions:List[str]) -> str:
        """Format a list of solutions into a prompt string.

        Args:
            solutions: list of solution strings
        Returns:
            A prompt string in the format:
            ## Response 1
            {solution_1}
            ## Response 2
            {solution_2}
            ## Response 3
            {solution_3}
            ...
        """

        if not solutions:
            return "No solutions"

        prompt_lines = []
        for i, solution in enumerate(solutions,1):
            # Clean each solution using strip_think_and_exec
            clean_solution = strip_think_and_exec(solution)
            if not clean_solution:
                clean_solution = "empty solution"
            prompt_lines.append(f"## Response {i}")
            prompt_lines.append(clean_solution)
            prompt_lines.append("")

        return "\n".join(prompt_lines).strip()

    def _parse_selector_choice(self, selector_response: str, solutions: List[str]) -> str:
        """Parse the chosen solution from the Selector's response.

        Parse <select>Response X</select> tags and return the corresponding original solution.

        Args:
            selector_response: the Selector Agent's response text
            solutions: the original solutions list

        Returns:
            The selected solution text
        """
        if not selector_response or not solutions:
            self.logger.warning("Empty selector_response or solutions, returning first solution")
            return solutions[0] if solutions else ""

        # Regex match for <select>Response X</select>
        match = re.search(r'<select>Response\s*(\d+)</select>', selector_response, re.IGNORECASE)
        if not match:
            self.logger.warning("Could not parse selector's decision. Defaulting to Response 1.")
            return solutions[0]

        idx = int(match.group(1)) - 1  # convert to 0-based index
        # Ensure index is within valid range
        idx = max(0, min(len(solutions) - 1, idx))

        self.logger.info(f"Selector chose Response {idx + 1}")
        return solutions[idx]

    def _get_selected_index(self, selector_response: str, num_solutions: int) -> int:
        """Extract the selected index from the Selector's response.

        Args:
            selector_response: the Selector Agent's response text
            num_solutions: number of solutions

        Returns:
            Selected index (0-based). Returns 0 if parsing fails.
        """
        if not selector_response:
            return 0

        match = re.search(r'<select>Response\s*(\d+)</select>', selector_response, re.IGNORECASE)
        if not match:
            return 0

        idx = int(match.group(1)) - 1
        return max(0, min(num_solutions - 1, idx))



    def save_results(self, output_file: str):
        """Save experiment results.

        Args:
            output_file: output file path
        """
        import json
        from pathlib import Path


        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"SelectExp(index:{self.index}) Results saved to {output_file}")

