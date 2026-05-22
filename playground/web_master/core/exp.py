"""Plan-once DAG-parallel BrowseComp experiment workflow."""

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

from .state import SchedulerRound, SearchDAG, SearchNode, SearchNodeResult


class FlashSearchExp(BaseExp):
    """DAG-based parallel search workflow inspired by Flash-Searcher."""

    def __init__(
        self,
        planner,
        searcher,
        finalizer,
        config,
        worker_factory: Callable[[str], Any] | None = None,
        max_workers: int = 3,
        max_rounds: int = 4,
    ):
        super().__init__(agent=searcher, config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.planner = planner
        self.searcher = searcher
        self.finalizer = finalizer or planner
        self.worker_factory = worker_factory
        self.max_workers = max(1, int(max_workers))
        self.max_rounds = max(1, int(max_rounds))
        self.ground_truth = None

    @property
    def exp_name(self) -> str:
        return "WebMasterFlashDAG"

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
        images=None,
        on_step=None,
    ) -> dict:
        self.logger.info("=" * 80)
        self.logger.info("WebMaster Flash DAG task start: %s", task_id)
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
            planner_trajectory, planner_output = self._run_text_agent(
                self.planner,
                exp_index=0,
                task_id=f"{task_id}_dag_plan",
                task_type="flash_dag_plan",
                description=self._planner_input(task_description),
                on_step=on_step,
            )
            dag = self._build_dag(planner_output, task_description)
            node_results, rounds = self._execute_dag(
                dag=dag,
                question=task_description,
                task_id=task_id,
                on_step=on_step,
            )
            final_trajectory, final_output = self._run_text_agent(
                self.finalizer,
                exp_index=10_000,
                task_id=f"{task_id}_answer_fusion",
                task_type="flash_answer_fusion",
                description=self._finalizer_input(task_description, dag, node_results, rounds),
                on_step=on_step,
            )
            final_answer = self._sanitize_answer(final_output)
            if not final_answer or final_answer.upper() == "UNKNOWN":
                final_answer = ""

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
                    "scheduler_rounds": [self._round_to_dict(item) for item in rounds],
                    "analysis": self._build_analysis(dag, node_results, rounds),
                    "planner_trajectory": planner_trajectory,
                    "final_trajectory": final_trajectory,
                }
            )
            self.results.append(result)
            self._save_run_result(result)
        except Exception as exc:
            self.logger.error("WebMaster Flash DAG task crashed: %s", exc, exc_info=True)
            result.update({"status": "failed", "error": str(exc)})
            self.results.append(result)
            self._save_run_result(result)

        self.logger.info("Agent final answer: %s", result.get("agent_answer", ""))
        self.logger.info("WebMaster Flash DAG task end: status=%s", result.get("status"))
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
        exp_index: int,
        task_id: str,
        task_type: str,
        description: str,
        on_step=None,
    ) -> tuple[object, str]:
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=exp_index)
        task = TaskInstance(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_data={"benchmark": "browsecomp", "architecture": "flash_searcher_dag"},
        )
        trajectory = agent.run(task, on_step=on_step)
        response = self._extract_agent_response(trajectory).strip()
        return trajectory, response

    def _planner_input(self, question: str) -> str:
        return "\n\n".join(
            [
                "Construct a Flash-Searcher DAG for this BrowseComp question.",
                "Return JSON only. Schema:",
                '{"hard_constraints":["..."],"final_goal":"...",'
                '"nodes":[{"id":"n1","goal":"...","depends_on":[],"priority":0}]}',
                "DAG rules:",
                "- Nodes are coarse searchable/verification subtasks, not single exact queries.",
                "- Independent clues should be separate nodes so they can run in parallel.",
                "- Verification nodes should depend only on the candidate-producing nodes they need.",
                "- The initial DAG must cover the whole problem; there will be no re-planning or dynamic node creation.",
                "- Nodes may produce local intermediate answers; the final answer to the original question is produced only by the finalizer.",
                "- Include enough evidence-gathering and verification nodes for the finalizer to answer the original question.",
                f"Question:\n{question}",
            ]
        )

    def _search_input(
        self,
        question: str,
        dag: SearchDAG,
        node: SearchNode,
        dependency_results: dict[str, SearchNodeResult],
    ) -> str:
        completed_deps = [dep for dep in node.depends_on if dep in dependency_results]
        missing_deps = [dep for dep in node.depends_on if dep not in dependency_results]
        return "\n\n".join(
            [
                "Execute one Flash-Searcher DAG node for a BrowseComp web search task.",
                "Use tools to search/fetch evidence for this node only.",
                "Return JSON only with schema:",
                '{"answer":"local answer or UNKNOWN","confidence":"high|medium|low|unknown",'
                '"evidence":["..."],"missing_constraints":["..."],"queries":["..."],"urls":["..."]}',
                "Rules:",
                "- Prefer exact evidence from primary/entity-adjacent pages.",
                "- Do not use benchmark mirrors, answer dumps, or pages that only repeat the question.",
                "- Dependency results may be incomplete or failed; use them as context, not as the final answer.",
                "- If uncertain, still provide the best local candidate with low confidence; use UNKNOWN only when no useful candidate exists.",
                "- Your node answer is local to this subtask. Do not try to answer the original question unless this node goal explicitly asks for that.",
                f"Original question:\n{question}",
                "Global hard constraints:",
                json.dumps(dag.hard_constraints, ensure_ascii=False, indent=2),
                f"Node id: {node.node_id}",
                f"Node goal:\n{node.goal}",
                f"Completed dependencies: {completed_deps}",
                f"Missing dependencies: {missing_deps}",
                "Dependency results:",
                json.dumps(
                    [self._node_result_to_dict(dependency_results[dep], include_raw=False) for dep in completed_deps],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

    def _finalizer_input(
        self,
        question: str,
        dag: SearchDAG,
        node_results: dict[str, SearchNodeResult],
        rounds: list[SchedulerRound],
    ) -> str:
        return "\n\n".join(
            [
                "Fuse the Flash-Searcher DAG evidence into the final BrowseComp answer.",
                "Return only the exact short answer. No explanation, no citation, no JSON.",
                "Do not return UNKNOWN; choose the best-supported answer from the DAG evidence.",
                "Hard rules:",
                "- Answer the original question, not any single DAG node goal.",
                "- Treat node answers as intermediate local results; never copy a node answer unless it actually answers the original question.",
                "- Check every hard constraint from the original question.",
                "- Prefer high-confidence verified evidence over low-confidence or incomplete node results.",
                "- Reject candidates contradicted by node evidence or missing hard constraints.",
                "- Preserve exact full names/titles, including subtitles after a colon.",
                f"Original question:\n{question}",
                "DAG state:",
                json.dumps(dag.to_dict(), ensure_ascii=False, indent=2),
                "Scheduler rounds:",
                json.dumps([self._round_to_dict(item) for item in rounds], ensure_ascii=False, indent=2),
                "Node results:",
                json.dumps(
                    [self._node_result_to_dict(item, include_raw=False) for item in node_results.values()],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )

    def _build_dag(self, planner_output: str, question: str) -> SearchDAG:
        payload = self._extract_json_object(planner_output)
        raw_nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(raw_nodes, list) or not raw_nodes:
            self.logger.warning("Planner did not return a valid DAG; using fallback DAG")
            raw_nodes = [
                {"id": "n1", "goal": "Search broadly for candidate answers matching the main clues.", "depends_on": [], "priority": 3},
                {"id": "n2", "goal": "Search alternate wording and sources for independent candidates.", "depends_on": [], "priority": 2},
                {"id": "n3", "goal": "Verify the best candidates against all hard constraints.", "depends_on": ["n1", "n2"], "priority": 1},
            ]
            payload = {"hard_constraints": [], "final_goal": f"Answer: {question}", "nodes": raw_nodes}

        nodes: dict[str, SearchNode] = {}
        for index, raw_node in enumerate(raw_nodes, 1):
            if not isinstance(raw_node, dict):
                continue
            node_id = self._coerce_str(raw_node.get("id")) or f"n{index}"
            goal = self._coerce_str(raw_node.get("goal") or raw_node.get("task"))
            if not goal:
                continue
            depends_on_raw = raw_node.get("depends_on") or raw_node.get("dependencies") or []
            if not isinstance(depends_on_raw, list):
                depends_on_raw = []
            nodes[node_id] = SearchNode(
                node_id=node_id,
                goal=goal,
                depends_on=[self._coerce_str(dep) for dep in depends_on_raw if self._coerce_str(dep)],
                priority=self._coerce_int(raw_node.get("priority"), default=0),
            )

        for node in nodes.values():
            node.depends_on = [dep for dep in node.depends_on if dep in nodes and dep != node.node_id]

        if not nodes:
            nodes = {"n1": SearchNode(node_id="n1", goal=f"Search for the answer to: {question}", priority=1)}

        return SearchDAG(
            nodes=nodes,
            hard_constraints=self._coerce_str_list(payload.get("hard_constraints") if isinstance(payload, dict) else []),
            final_goal=self._coerce_str(payload.get("final_goal") if isinstance(payload, dict) else "") or question,
        )

    def _execute_dag(
        self,
        dag: SearchDAG,
        question: str,
        task_id: str,
        on_step=None,
    ) -> tuple[dict[str, SearchNodeResult], list[SchedulerRound]]:
        settled: set[str] = set()
        node_results: dict[str, SearchNodeResult] = {}
        rounds: list[SchedulerRound] = []

        for round_index in range(1, self.max_rounds + 1):
            ready = dag.ready_nodes(settled)
            selected = ready[: self.max_workers]
            if not selected:
                break

            round_info = SchedulerRound(
                round_index=round_index,
                ready_node_ids=[node.node_id for node in selected],
            )
            rounds.append(round_info)
            self.logger.info("DAG round %s: ready=%s", round_index, round_info.ready_node_ids)

            for node in selected:
                node.status = "running"

            round_results = self._run_nodes_parallel(
                nodes=selected,
                dag=dag,
                question=question,
                task_id=task_id,
                round_index=round_index,
                node_results=node_results,
                on_step=on_step,
            )
            for node_result in round_results:
                node = dag.nodes[node_result.node_id]
                node.status = node_result.status
                node.error = node_result.error
                node_results[node_result.node_id] = node_result
                settled.add(node_result.node_id)
                round_info.completed_node_ids.append(node_result.node_id)

            if all(node.status in {"completed", "failed", "skipped"} for node in dag.nodes.values()):
                break

        for node in dag.nodes.values():
            if node.node_id not in node_results:
                node.status = "skipped"
                node.error = "Node was not scheduled before max rounds or dependencies remained unresolved."
                node_results[node.node_id] = SearchNodeResult(
                    node_id=node.node_id,
                    status="skipped",
                    dependency_node_ids=list(node.depends_on),
                    missing_dependency_node_ids=[dep for dep in node.depends_on if dep not in settled],
                    error=node.error,
                )
        return node_results, rounds

    def _run_nodes_parallel(
        self,
        nodes: list[SearchNode],
        dag: SearchDAG,
        question: str,
        task_id: str,
        round_index: int,
        node_results: dict[str, SearchNodeResult],
        on_step=None,
    ) -> list[SearchNodeResult]:
        results: list[SearchNodeResult] = []
        max_workers = min(self.max_workers, len(nodes))
        assignments = []
        for node_index, node in enumerate(nodes):
            worker = self._make_worker_agent(node, round_index)
            assignments.append((node_index, node, worker))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node = {
                executor.submit(
                    self._run_single_node,
                    node_index,
                    node,
                    worker,
                    dag,
                    question,
                    task_id,
                    round_index,
                    dict(node_results),
                    on_step,
                ): node
                for node_index, node, worker in assignments
            }
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    self.logger.error("DAG node %s failed: %s", node.node_id, exc, exc_info=True)
                    results.append(SearchNodeResult(node_id=node.node_id, status="failed", error=str(exc)))
        return results

    def _run_single_node(
        self,
        node_index: int,
        node: SearchNode,
        worker,
        dag: SearchDAG,
        question: str,
        task_id: str,
        round_index: int,
        node_results: dict[str, SearchNodeResult],
        on_step=None,
    ) -> SearchNodeResult:
        trajectory, raw_output = self._run_text_agent(
            worker,
            exp_index=round_index * 100 + node_index,
            task_id=f"{task_id}_{node.node_id}_r{round_index}",
            task_type="flash_dag_node",
            description=self._search_input(question, dag, node, node_results),
            on_step=on_step,
        )
        parsed = self._extract_json_object(raw_output)
        answer = self._coerce_str(parsed.get("answer") if isinstance(parsed, dict) else "")
        if not answer:
            answer = self._sanitize_answer(raw_output) or "UNKNOWN"
        evidence = self._coerce_str_list(parsed.get("evidence") if isinstance(parsed, dict) else [])
        missing_constraints = self._coerce_str_list(parsed.get("missing_constraints") if isinstance(parsed, dict) else [])
        queries = self._coerce_str_list(parsed.get("queries") if isinstance(parsed, dict) else [])
        urls = self._coerce_str_list(parsed.get("urls") if isinstance(parsed, dict) else [])
        confidence = self._coerce_str(parsed.get("confidence") if isinstance(parsed, dict) else "") or "unknown"
        tool_usage = self._extract_tool_usage(trajectory)
        queries = queries or tool_usage["queries"]
        urls = urls or tool_usage["urls"]
        status = "completed" if answer or evidence else "failed"
        return SearchNodeResult(
            node_id=node.node_id,
            status=status,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            missing_constraints=missing_constraints,
            queries=queries,
            urls=urls,
            tool_call_counts=tool_usage["tool_call_counts"],
            dependency_node_ids=list(node.depends_on),
            missing_dependency_node_ids=[dep for dep in node.depends_on if dep not in node_results],
            raw_output=raw_output,
            steps=len(getattr(trajectory, "steps", []) or []),
        )

    def _make_worker_agent(self, node: SearchNode, round_index: int):
        worker_name = f"flash_dag_{node.node_id}_r{round_index}"
        if self.worker_factory is not None:
            return self.worker_factory(worker_name)
        return self.searcher

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
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                if tool_name == "google_search":
                    queries.extend(self._coerce_str_list(args.get("query")))
                elif tool_name == "web_fetch":
                    urls.extend(self._coerce_str_list(args.get("url")))
        return {"tool_call_counts": dict(tool_call_counts), "queries": queries, "urls": urls}

    def _sanitize_answer(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        payload = self._extract_json_object(text)
        if isinstance(payload, dict):
            for key in ("answer", "final_answer", "message"):
                value = self._coerce_str(payload.get(key))
                if value:
                    text = value
                    break
        tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
        if tag_match:
            text = tag_match.group(1).strip()
        text = re.sub(r"^Final answer\s*:\s*", "", text, flags=re.I).strip()
        text = re.sub(r"^Answer\s*:\s*", "", text, flags=re.I).strip()
        text = text.strip().strip('"').strip("'").strip()
        for line in text.splitlines():
            line = line.strip().strip('"').strip("'").strip()
            if line:
                return line[:300].strip()
        return ""

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
        candidates = [fenced.group(1)] if fenced else []
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

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _node_result_to_dict(self, node_result: SearchNodeResult, include_raw: bool = True) -> dict[str, Any]:
        payload = {
            "node_id": node_result.node_id,
            "status": node_result.status,
            "answer": node_result.answer,
            "confidence": node_result.confidence,
            "evidence": node_result.evidence,
            "missing_constraints": node_result.missing_constraints,
            "queries": node_result.queries,
            "urls": node_result.urls,
            "tool_call_counts": node_result.tool_call_counts,
            "dependency_node_ids": node_result.dependency_node_ids,
            "missing_dependency_node_ids": node_result.missing_dependency_node_ids,
            "steps": node_result.steps,
            "error": node_result.error,
        }
        if include_raw:
            payload["raw_output"] = node_result.raw_output
        return payload

    def _round_to_dict(self, round_info: SchedulerRound) -> dict[str, Any]:
        return {
            "round_index": round_info.round_index,
            "ready_node_ids": round_info.ready_node_ids,
            "completed_node_ids": round_info.completed_node_ids,
        }

    def _build_analysis(
        self,
        dag: SearchDAG,
        node_results: dict[str, SearchNodeResult],
        rounds: list[SchedulerRound],
    ) -> dict[str, Any]:
        counts = Counter()
        for item in node_results.values():
            counts.update(item.tool_call_counts)
        return {
            "round_count": len(rounds),
            "node_count": len(dag.nodes),
            "completed_nodes": sum(1 for item in node_results.values() if item.status == "completed"),
            "failed_nodes": sum(1 for item in node_results.values() if item.status == "failed"),
            "skipped_nodes": sum(1 for item in node_results.values() if item.status == "skipped"),
            "critical_path_length": dag.critical_path_length(),
            "tool_call_counts": dict(counts),
        }

    def _serialize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "steps": result.get("steps", 0),
            "agent_answer": result.get("agent_answer", ""),
            "ground_truth": result.get("ground_truth"),
            "dag_plan": result.get("dag_plan", {}),
            "node_results": result.get("node_results", {}),
            "scheduler_rounds": result.get("scheduler_rounds", []),
            "analysis": result.get("analysis", {}),
            "error": result.get("error"),
        }

    def _save_run_result(self, result: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        output_path = Path(self.run_dir) / "result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._serialize_result(result), handle, indent=2, ensure_ascii=False, default=str)
