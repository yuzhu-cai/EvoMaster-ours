"""Draft node experiment for FrontierScience."""

from __future__ import annotations

from . import NodeExp


class DraftExp(NodeExp):
    def run(
        self,
        *,
        problem: str,
        task_id: str,
        task_input_data: dict,
        reference_rubric: str,
        trajectory_summary: str,
        shared_tool_memory: str,
        worker_workspace: str,
        solution_path: str,
        solution_filename: str,
        images: list[str] | None,
    ) -> dict:
        return self._run_agent(
            prompt_kwargs={
                "problem": problem,
                "reference_rubric": reference_rubric,
                "trajectory_summary": trajectory_summary or "No prior trajectory.",
                "shared_tool_memory": shared_tool_memory,
                "worker_workspace": worker_workspace,
                "solution_path": solution_path,
                "solution_filename": solution_filename,
            },
            problem=problem,
            task_id=f"{task_id}_draft",
            task_type="draft",
            task_input_data=task_input_data,
            previous_answer="",
            reference_rubric=reference_rubric,
            solution_path=solution_path,
            images=images,
        )
