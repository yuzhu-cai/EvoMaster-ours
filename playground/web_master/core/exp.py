"""Flash-Searcher style BrowseComp experiment workflow."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from .state import SearchDAG, SearchNode, SearchNodeResult


BROWSE_TOOL_NAMES = {"google_search", "web_fetch", "think", "finish"}


class FlashSearchExp(BaseExp):
    """DAG-parallel BrowseComp experiment inspired by Flash-Searcher."""

    def __init__(
        self,
        planner,
        searcher,
        finalizer,
        config,
        agent_copier: Callable | None = None,
        max_workers: int = 3,
        max_rounds: int = 4,
    ):
        super().__init__(agent=searcher, config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = planner
        self.searcher = searcher
        self.finalizer = finalizer or planner
        self.agent_copier = agent_copier
        self.max_workers = max(1, int(max_workers))
        self.max_rounds = max(1, int(max_rounds))
        self.ground_truth = None

    @property
    def exp_name(self) -> str:
        return "WebMasterFlash"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images=None,
        on_step=None,
    ) -> dict:
        self.logger.info("=" * 80)
        self.logger.info("WebMaster Flash task start: %s", task_id)
        self.logger.info("Question: %s", task_description)
        self.logger.info("=" * 80)

        result: dict[str, Any] = {
            "task_id": task_id,
            "status": "running",
            "steps": 0,
            "agent_answer": "",
            "ground_truth": self.ground_truth,
        }

        try:
            planner_output = self._run_text_agent(
                self.planner,
                task_id=f"{task_id}_plan",
                task_type="flash_plan",
                description=self._planner_input(task_description),
                on_step=on_step,
            )[1]
            dag = self._build_dag(planner_output, task_description)
            node_results, parallel_rounds = self._execute_dag(
                dag=dag,
                question=task_description,
                task_id=task_id,
                on_step=on_step,
            )

            final_output = self._run_text_agent(
                self.finalizer,
                task_id=f"{task_id}_final",
                task_type="flash_final",
                description=self._finalizer_input(task_description, dag, node_results),
                on_step=on_step,
            )[1]
            final_answer = self._sanitize_final_answer(final_output)

            result.update(
                {
                    "status": "completed" if final_answer else "no_answer",
                    "steps": sum(item.steps for item in node_results.values()),
                    "agent_answer": final_answer,
                    "dag_plan": dag.to_dict(),
                    "node_results": {
                        node_id: self._node_result_to_dict(node_result)
                        for node_id, node_result in node_results.items()
                    },
                    "analysis": self._build_analysis(
                        dag=dag,
                        node_results=node_results,
                        parallel_rounds=parallel_rounds,
                    ),
                }
            )
            self.results.append(result)
        except Exception as exc:
            self.logger.error("WebMaster Flash task crashed: %s", exc, exc_info=True)
            result.update({"status": "failed", "error": str(exc)})
            self.results.append(result)

        self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))
        self.logger.info(
            "WebMaster Flash task end: status=%s steps=%s",
            result.get("status"),
            result.get("steps"),
        )
        return result

    def save_results(self, output_file: str):
        output_data = [self._serialize_result(result) for result in self.results]
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output_data, handle, indent=2, ensure_ascii=False, default=str)
        self.logger.info("Results saved to %s", output_file)

    def _run_text_agent(
        self,
        agent,
        task_id: str,
        task_type: str,
        description: str,
        on_step=None,
    ) -> tuple[object, str]:
        task = TaskInstance(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_data={"benchmark": "browsecomp", "architecture": "flash_searcher"},
        )
        trajectory = agent.run(task, on_step=on_step)
        response = self._extract_agent_response(trajectory).strip()
        return trajectory, response

    def _planner_input(self, question: str) -> str:
        return "\n\n".join(
            [
                "Create a coarse DAG search plan for this BrowseComp question.",
                "Each node should be an independently searchable intent, not a single exact query.",
                "Return JSON only with schema: "
                '{"nodes":[{"id":"n1","goal":"...","depends_on":[]}]}',
                f"Question:\n{question}",
            ]
        )

    def _search_input(self, question: str, node: SearchNode) -> str:
        return "\n\n".join(
            [
                "Solve this one DAG search node for a BrowseComp question.",
                "Use web tools privately. Return a compact JSON object if possible.",
                'Schema: {"answer":"short local answer or UNKNOWN","evidence":["..."],'
                '"queries":["..."],"urls":["..."]}',
                f"Original question:\n{question}",
                f"Node id: {node.node_id}",
                f"Node goal:\n{node.goal}",
            ]
        )

    def _finalizer_input(
        self,
        question: str,
        dag: SearchDAG,
        node_results: dict[str, SearchNodeResult],
    ) -> str:
        compact_results = [
            self._node_result_to_dict(node_result, include_raw=False)
            for node_result in node_results.values()
        ]
        return "\n\n".join(
            [
                "Produce the final BrowseComp answer from the DAG search results.",
                "Return only the exact short answer. No explanation, no citation.",
                f"Original question:\n{question}",
                "DAG:",
                json.dumps(dag.to_dict(), ensure_ascii=False, indent=2),
                "Node results:",
                json.dumps(compact_results, ensure_ascii=False, indent=2),
            ]
        )

    def _build_dag(self, planner_output: str, question: str) -> SearchDAG:
        payload = self._extract_json_object(planner_output)
        raw_nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(raw_nodes, list) or not raw_nodes:
            self.logger.warning("Planner did not return a valid DAG; using fallback nodes")
            raw_nodes = [
                {
                    "id": "n1",
                    "goal": "Find candidate entities or facts that match the strongest clues.",
                    "depends_on": [],
                },
                {
                    "id": "n2",
                    "goal": "Use alternate wording and pivot terms to search for the hidden target.",
                    "depends_on": [],
                },
                {
                    "id": "n3",
                    "goal": "Cross-check the best candidate against decisive hard clues.",
                    "depends_on": ["n1", "n2"],
                },
            ]

        nodes: dict[str, SearchNode] = {}
        for index, raw_node in enumerate(raw_nodes, 1):
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node.get("id") or f"n{index}").strip()
            goal = str(raw_node.get("goal") or raw_node.get("task") or "").strip()
            if not node_id or not goal:
                continue
            depends_on_raw = raw_node.get("depends_on", [])
            if not isinstance(depends_on_raw, list):
                depends_on_raw = []
            nodes[node_id] = SearchNode(
                node_id=node_id,
                goal=goal,
                depends_on=[str(dep).strip() for dep in depends_on_raw if str(dep).strip()],
            )

        if not nodes:
            return SearchDAG(
                nodes={
                    "n1": SearchNode(
                        node_id="n1",
                        goal=f"Search broadly for candidates answering: {question}",
                    )
                }
            )

        for node in nodes.values():
            node.depends_on = [
                dep
                for dep in node.depends_on
                if dep in nodes and dep != node.node_id
            ]

        return SearchDAG(nodes=nodes)

    def _execute_dag(
        self,
        dag: SearchDAG,
        question: str,
        task_id: str,
        on_step=None,
    ) -> tuple[dict[str, SearchNodeResult], int]:
        completed: set[str] = set()
        node_results: dict[str, SearchNodeResult] = {}
        parallel_rounds = 0

        for round_index in range(1, self.max_rounds + 1):
            ready = dag.ready_nodes(completed)
            if not ready:
                break
            parallel_rounds += 1
            self.logger.info(
                "Executing DAG round %s with %s ready nodes",
                round_index,
                len(ready),
            )
            for node in ready:
                node.status = "running"

            round_results = self._run_ready_nodes_parallel(
                ready=ready,
                question=question,
                task_id=task_id,
                round_index=round_index,
                on_step=on_step,
            )
            for node_result in round_results:
                node = dag.nodes[node_result.node_id]
                node.status = node_result.status
                node.error = node_result.error
                node_results[node_result.node_id] = node_result
                if node_result.status == "completed":
                    completed.add(node_result.node_id)

            if len(completed) == len(dag.nodes):
                break

        for node in dag.unfinished_nodes():
            if node.node_id not in node_results:
                node.status = "skipped"
                node.error = "Node was not executed before max rounds or dependencies were unresolved."
                node_results[node.node_id] = SearchNodeResult(
                    node_id=node.node_id,
                    status="skipped",
                    error=node.error,
                )
        return node_results, parallel_rounds

    def _run_ready_nodes_parallel(
        self,
        ready: list[SearchNode],
        question: str,
        task_id: str,
        round_index: int,
        on_step=None,
    ) -> list[SearchNodeResult]:
        results: list[SearchNodeResult] = []
        max_workers = min(self.max_workers, len(ready))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node = {
                executor.submit(
                    self._run_single_node,
                    node,
                    question,
                    task_id,
                    round_index,
                    node_index,
                    on_step,
                ): node
                for node_index, node in enumerate(ready)
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    self.logger.error(
                        "Search node %s failed: %s", node.node_id, exc, exc_info=True
                    )
                    results.append(
                        SearchNodeResult(
                            node_id=node.node_id,
                            status="failed",
                            error=str(exc),
                        )
                    )
        return results

    def _run_single_node(
        self,
        node: SearchNode,
        question: str,
        task_id: str,
        round_index: int,
        node_index: int,
        on_step=None,
    ) -> SearchNodeResult:
        agent = self.searcher
        if self.agent_copier is not None:
            agent = self.agent_copier(
                self.searcher,
                new_agent_name=f"flash_{node.node_id}_r{round_index}",
            )

        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=node_index)
        trajectory, raw_output = self._run_text_agent(
            agent,
            task_id=f"{task_id}_{node.node_id}",
            task_type="flash_search_node",
            description=self._search_input(question, node),
            on_step=on_step,
        )
        parsed = self._extract_json_object(raw_output)
        answer = self._coerce_str(parsed.get("answer") if isinstance(parsed, dict) else "")
        evidence = self._coerce_str_list(parsed.get("evidence") if isinstance(parsed, dict) else [])
        queries = self._coerce_str_list(parsed.get("queries") if isinstance(parsed, dict) else [])
        urls = self._coerce_str_list(parsed.get("urls") if isinstance(parsed, dict) else [])
        tool_usage = self._extract_tool_usage(trajectory)
        queries = queries or tool_usage["queries"]
        urls = urls or tool_usage["urls"]

        if not answer:
            answer = self._sanitize_final_answer(raw_output) or "UNKNOWN"

        return SearchNodeResult(
            node_id=node.node_id,
            status="completed",
            answer=answer,
            evidence=evidence,
            queries=queries,
            urls=urls,
            tool_call_counts=tool_usage["tool_call_counts"],
            raw_output=raw_output,
            steps=len(getattr(trajectory, "steps", []) or []),
        )

    def _extract_tool_usage(self, trajectory) -> dict[str, Any]:
        tool_call_counts: Counter[str] = Counter()
        queries: list[str] = []
        urls: list[str] = []

        for step in getattr(trajectory, "steps", []) or []:
            assistant_message = getattr(step, "assistant_message", None)
            tool_calls = getattr(assistant_message, "tool_calls", None) or []
            for tool_call in tool_calls:
                function = getattr(tool_call, "function", None)
                tool_name = getattr(function, "name", "")
                args_raw = getattr(function, "arguments", "") or "{}"
                tool_call_counts[tool_name] += 1
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
                if tool_name == "google_search":
                    queries.extend(self._coerce_str_list(args.get("query")))
                elif tool_name == "web_fetch":
                    urls.extend(self._coerce_str_list(args.get("url")))

        return {
            "tool_call_counts": dict(tool_call_counts),
            "queries": queries,
            "urls": urls,
        }

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        candidates = []
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
        if fenced:
            candidates.append(fenced.group(1))
        candidates.append(text)

        for candidate in candidates:
            candidate = candidate.strip()
            try:
                payload = json.loads(candidate)
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                pass

            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(candidate[start : end + 1])
                    return payload if isinstance(payload, dict) else {}
                except json.JSONDecodeError:
                    continue
        return {}

    def _sanitize_final_answer(self, text: str) -> str:
        text = (text or "").strip()
        tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
        if tag_match:
            text = tag_match.group(1).strip()
        text = re.sub(r"^Final answer\s*:\s*", "", text, flags=re.I).strip()
        text = re.sub(r"^Answer\s*:\s*", "", text, flags=re.I).strip()
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:200].strip()
        return ""

    def _coerce_str(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _coerce_str_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [self._coerce_str(item) for item in value if self._coerce_str(item)]
        return [self._coerce_str(value)] if self._coerce_str(value) else []

    def _build_analysis(
        self,
        dag: SearchDAG,
        node_results: dict[str, SearchNodeResult],
        parallel_rounds: int,
    ) -> dict[str, Any]:
        return {
            "parallel_rounds": parallel_rounds,
            "completed_nodes": sum(1 for item in node_results.values() if item.status == "completed"),
            "failed_nodes": sum(1 for item in node_results.values() if item.status == "failed"),
            "skipped_nodes": sum(1 for item in node_results.values() if item.status == "skipped"),
            "critical_path_length": dag.critical_path_length(),
            "node_count": len(dag.nodes),
            "tool_call_counts": self._tool_call_counts_from_outputs(node_results),
        }

    def _tool_call_counts_from_outputs(
        self,
        node_results: dict[str, SearchNodeResult],
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for node_result in node_results.values():
            counts.update(node_result.tool_call_counts)
        return dict(counts)

    def _node_result_to_dict(
        self,
        node_result: SearchNodeResult,
        include_raw: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "node_id": node_result.node_id,
            "status": node_result.status,
            "answer": node_result.answer,
            "evidence": node_result.evidence,
            "queries": node_result.queries,
            "urls": node_result.urls,
            "tool_call_counts": node_result.tool_call_counts,
            "steps": node_result.steps,
            "error": node_result.error,
        }
        if include_raw:
            payload["raw_output"] = node_result.raw_output
        return payload

    def _serialize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "steps": result.get("steps", 0),
            "agent_answer": result.get("agent_answer", ""),
            "ground_truth": result.get("ground_truth"),
            "dag_plan": result.get("dag_plan", {}),
            "node_results": result.get("node_results", {}),
            "analysis": result.get("analysis", {}),
            "error": result.get("error"),
        }
