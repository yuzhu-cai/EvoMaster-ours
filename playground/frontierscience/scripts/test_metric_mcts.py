import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pypdf import PdfWriter

from playground.frontierscience.core.exp import BaseNodeExp
from playground.frontierscience.core.playground import FrontierSciencePlayground
from playground.frontierscience.core.utils import (
    FrontierScienceMCTS,
    FrontierScienceMCTSConfig,
    FrontierScienceMetricEvaluator,
    FrontierScienceSearchNode,
    FrontierScienceSharedContext,
)
from playground.frontierscience.tools.pdf_reader import _resolve_pdf_path
from playground.frontierscience.tools.visit_web import _extract_title_from_content, _slugify_title, local_visit_web


class TestFrontierScienceMetricAndMCTS(unittest.TestCase):
    @staticmethod
    def _build_pdf_bytes() -> bytes:
        import io

        buffer = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(buffer)
        return buffer.getvalue()

    def test_base_node_exp_runs_agent_once_and_extracts_answer(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self._prompt_format_kwargs = {}
                self.run_calls = 0

            def run(self, task):
                del task
                self.run_calls += 1
                return {
                    "dialogs": [
                        {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "finish",
                                                "arguments": (
                                                    '{"message":"Understood. I will now proceed to answer step by step.",'
                                                    '"task_completed":"true"}'
                                                ),
                                            }
                                        }
                                    ],
                                }
                            ]
                        }
                    ],
                    "status": "completed",
                }

        with TemporaryDirectory() as tmpdir:
            solution_path = Path(tmpdir) / "frontierscience_placeholder_retry.md"
            solution_path.write_text("existing", encoding="utf-8")
            exp = BaseNodeExp(FakeAgent())
            result = exp._run_agent(
                prompt_kwargs={},
                problem="Explain weak value amplification.",
                task_id="demo",
                task_type="draft",
                task_input_data={},
                previous_answer="",
                reference_rubric="",
                solution_path=str(solution_path),
                images=None,
            )
            self.assertEqual(exp.agent.run_calls, 1)
            self.assertEqual(
                result["current_answer"],
                "Understood. I will now proceed to answer step by step.",
            )
            self.assertIn("metric_overall=None", result["state_summary"])

    def test_base_node_exp_passes_result_to_metric_evaluator(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self._prompt_format_kwargs = {}
                self.run_calls = 0

            def run(self, task):
                del task
                self.run_calls += 1
                return {
                    "dialogs": [
                        {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "finish",
                                                "arguments": (
                                                    '{"message":"1. Weak value amplification can be improved with entanglement.",'
                                                    '"task_completed":"true"}'
                                                ),
                                            }
                                        }
                                    ],
                                }
                            ]
                        }
                    ],
                    "status": "completed",
                }

        class FakeMetricEvaluator:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def evaluate(self, **kwargs):
                self.calls.append(kwargs)
                return {"overall_score": 0.82, "is_valid": True}

        with TemporaryDirectory() as tmpdir:
            solution_path = Path(tmpdir) / "frontierscience_retrieval_retry.md"
            solution_path.write_text(
                "1. Saved answer from workspace file with concrete evidence citations.",
                encoding="utf-8",
            )
            metric = FakeMetricEvaluator()
            exp = BaseNodeExp(FakeAgent(), metric_evaluator=metric)
            result = exp._run_agent(
                prompt_kwargs={},
                problem="Context: See [1], [2], [3]. Question: explain the cited papers and compare their claims.",
                task_id="demo",
                task_type="draft",
                task_input_data={},
                previous_answer="",
                reference_rubric="Strong evidence grounding required.",
                solution_path=str(solution_path),
                images=None,
            )
            self.assertEqual(exp.agent.run_calls, 1)
            self.assertEqual(len(metric.calls), 1)
            self.assertEqual(metric.calls[0]["previous_answer"], "")
            self.assertEqual(metric.calls[0]["action_type"], "draft")
            self.assertEqual(
                metric.calls[0]["final_answer"],
                "1. Saved answer from workspace file with concrete evidence citations.",
            )
            self.assertEqual(
                result["current_answer"],
                "1. Weak value amplification can be improved with entanglement.",
            )
            self.assertEqual(result["metric_result"]["overall_score"], 0.82)
            self.assertIn("metric_overall=0.82", result["state_summary"])

    def test_metric_outputs_structured_json_like_result(self) -> None:
        trajectory = {
            "dialogs": [
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"function": {"name": "search_web", "arguments": "{\"query\": \"weak value amplification\"}"}},
                                {"function": {"name": "read_paper_pdf", "arguments": "{\"pdf_path\": \"paper.pdf\"}"}},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": "According to the paper, the protocol improves postselection probability while preserving the weak value.",
                        },
                    ]
                }
            ]
        }
        evaluator = FrontierScienceMetricEvaluator()
        result = evaluator.evaluate(
            problem="Explain how entanglement improves weak value amplification efficiency.",
            final_answer=(
                "1. Weak value amplification couples the ancilla and meter weakly.\n"
                "2. According to the cited derivation, entangled ancillas increase the variance of A and improve postselection efficiency.\n"
                "3. The protocol keeps the weak value fixed while increasing the success probability."
            ),
            trajectory=trajectory,
            previous_answer="Weak value amplification exists.",
            action_type="improve",
            is_final=False,
        )
        self.assertIn("overall_score", result)
        self.assertIn("subscores", result)
        self.assertIn("evidence_grounding", result["subscores"])
        self.assertIn("revision_effectiveness", result["subscores"])
        self.assertTrue(result["is_valid"])
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 1.0)

    def test_mcts_runs_draft_improve_evaluate_loop(self) -> None:
        def action_runner(node, action_type):
            if action_type == "draft":
                answer = "Initial answer with limited evidence."
            else:
                answer = node.current_answer + " Improved with evidence and clearer conclusion."
            return {
                "node_id": f"{action_type}_{node.depth + 1}_{len(node.children)}",
                "trajectory": {
                    "dialogs": [
                        {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {"function": {"name": "search_web", "arguments": "{\"query\": \"frontier science\"}"}}
                                    ],
                                },
                                {"role": "assistant", "content": answer},
                            ]
                        }
                    ]
                },
                "current_answer": answer,
                "state_summary": f"{action_type} generated answer length={len(answer)}",
            }

        evaluator = FrontierScienceMetricEvaluator(
            judge_fn=lambda **kwargs: {
                "overall_score": 0.45 if kwargs["action_type"] == "draft" else 0.78,
                "final_answer_score": 0.4 if kwargs["action_type"] == "draft" else 0.8,
                "trajectory_score": 0.5 if kwargs["action_type"] == "draft" else 0.72,
                "subscores": {
                    "task_completion": 2 if kwargs["action_type"] == "draft" else 4,
                    "factual_correctness": 2 if kwargs["action_type"] == "draft" else 4,
                    "evidence_grounding": 2 if kwargs["action_type"] == "draft" else 4,
                    "completeness": 2 if kwargs["action_type"] == "draft" else 4,
                    "scientific_reasonableness": 2 if kwargs["action_type"] == "draft" else 4,
                    "clarity": 2 if kwargs["action_type"] == "draft" else 4,
                    "retrieval_quality": 3,
                    "evidence_usage": 2 if kwargs["action_type"] == "draft" else 4,
                    "reasoning_consistency": 3 if kwargs["action_type"] == "draft" else 4,
                    "revision_effectiveness": 1 if kwargs["action_type"] == "draft" else 4,
                    "exploration_efficiency": 3,
                },
                "strengths": ["uses evidence"],
                "weaknesses": ["needs more detail"] if kwargs["action_type"] == "draft" else [],
                "improvement_suggestions": ["add evidence"],
                "is_valid": True,
                "reason": "mock_judge",
            }
        )
        mcts = FrontierScienceMCTS(
            config=FrontierScienceMCTSConfig(max_depth=2, max_iterations=3, max_children_per_node=2),
            action_runner=action_runner,
            metric_evaluator=lambda node, is_final: evaluator.evaluate(
                problem="Explain the answer.",
                final_answer=node.current_answer,
                trajectory=node.trajectory,
                previous_answer=node.parent.current_answer if node.parent else "",
                action_type=node.action_type,
                is_final=is_final,
            ),
        )
        result = mcts.run()
        self.assertIn("best_answer", result)
        self.assertIn("best_metric_result", result)
        self.assertTrue(result["best_answer"].endswith("clearer conclusion."))
        self.assertGreaterEqual(result["best_metric_result"]["overall_score"], 0.7)
        self.assertGreaterEqual(len(result["tree"]), 2)

    def test_mcts_keeps_expanding_other_frontiers_when_best_branch_is_closed(self) -> None:
        expansions: list[tuple[str, str]] = []

        def action_runner(node, action_type):
            expansions.append((node.node_id, action_type))
            answer = f"{node.node_id}->{action_type}"
            return {
                "node_id": f"{action_type}_{node.node_id}_{len(expansions)}",
                "trajectory": {"dialogs": []},
                "current_answer": answer,
                "state_summary": answer,
            }

        def metric_evaluator(node, is_final):
            del is_final
            score_map = {
                "draft_root_1": 0.2,
                "draft_root_2": 0.9,
                "improve_draft_root_2_3": 0.8,
                "improve_draft_root_1_4": 0.7,
            }
            return {
                "overall_score": score_map.get(node.node_id, 0.1),
                "final_answer_score": score_map.get(node.node_id, 0.1),
                "trajectory_score": score_map.get(node.node_id, 0.1),
                "subscores": {},
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [],
                "is_valid": True,
                "reason": "mock",
            }

        mcts = FrontierScienceMCTS(
            config=FrontierScienceMCTSConfig(
                num_drafts=2,
                max_depth=2,
                max_iterations=4,
                max_children_per_node=2,
                allow_repeat_improve=False,
            ),
            action_runner=action_runner,
            metric_evaluator=metric_evaluator,
        )

        # Mirror the playground execution path instead of mcts.run(),
        # because FrontierSciencePlayground uses _select/_expand helpers directly.
        expansions_done = 0
        while expansions_done < mcts.config.max_iterations:
            selected = mcts._select(mcts.root)
            child = mcts._expand(selected)
            if child is None:
                break
            metric_result = metric_evaluator(child, False)
            child.latest_metric_result = metric_result
            mcts._backup(child, float(metric_result["overall_score"]))
            mcts._maybe_update_best(child)
            expansions_done += 1

        improve_parents = [parent_id for parent_id, action in expansions if action == "improve"]
        self.assertEqual(expansions_done, 4)
        self.assertCountEqual(improve_parents, ["draft_root_1", "draft_root_2"])

    def test_mcts_marks_search_exhausted_after_all_frontiers_are_expanded(self) -> None:
        def action_runner(node, action_type):
            del node
            self.assertEqual(action_type, "draft")
            return {
                "node_id": "draft_root_only",
                "trajectory": {"dialogs": []},
                "current_answer": "single draft",
                "state_summary": "single draft",
            }

        def metric_evaluator(node, is_final):
            del node, is_final
            return {
                "overall_score": 0.5,
                "final_answer_score": 0.5,
                "trajectory_score": 0.5,
                "subscores": {},
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [],
                "is_valid": True,
                "reason": "mock",
            }

        mcts = FrontierScienceMCTS(
            config=FrontierScienceMCTSConfig(
                num_drafts=1,
                max_depth=1,
                max_iterations=5,
                max_children_per_node=1,
            ),
            action_runner=action_runner,
            metric_evaluator=metric_evaluator,
        )

        self.assertFalse(mcts.is_search_exhausted())
        result = mcts.run()

        expansion_count = len([node for node in mcts.all_nodes if node is not mcts.root])
        self.assertEqual(expansion_count, 1)
        self.assertTrue(mcts.is_search_exhausted())
        self.assertEqual(result["best_answer"], "single draft")

    def test_pdf_reader_recovers_pdf_from_parent_workspace(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "paper.pdf"
            expected.write_bytes(self._build_pdf_bytes())
            missing_path = root / "exp_0" / "paper.pdf"
            resolved = _resolve_pdf_path(str(missing_path))
            self.assertEqual(resolved, expected.resolve())

    def test_visit_web_auto_downloads_valid_arxiv_pdf_to_workspace(self) -> None:
        content = "[1305.7154] Colloquium: Understanding Quantum Weak Values: Basics and Applications"
        with TemporaryDirectory() as tmpdir:
            with patch(
                "playground.frontierscience.tools.visit_web._fetch_builtin_content",
                return_value=("direct", content),
            ), patch(
                "playground.frontierscience.tools.visit_web._download_pdf_bytes",
                return_value=self._build_pdf_bytes(),
            ):
                result = local_visit_web(
                    "https://arxiv.org/abs/1305.7154",
                    "Extract key details.",
                    workspace_path=tmpdir,
                )

            self.assertIn("Downloaded PDF Path:", result)
            self.assertIn("PDF Download Status: downloaded_valid_pdf", result)
            self.assertIn("colloquium_understanding_quantum_weak_values_basics_and_applications.pdf", result)

    def test_visit_web_extracts_short_title_from_flattened_arxiv_page(self) -> None:
        content = (
            "[1305.7154] Colloquium: Understanding Quantum Weak Values: Basics and Applications "
            "Authors: Justin Dressel, Mehul Malik, Filippo M. Miatto, Andrew N. Jordan, Robert W. Boyd "
            "Abstract Since its introduction 25 years ago the quantum weak value has gradually transitioned "
            "from a theoretical curiosity to a practical laboratory tool."
        )
        title = _extract_title_from_content(content, "https://arxiv.org/abs/1305.7154")
        self.assertEqual(title, "Colloquium: Understanding Quantum Weak Values: Basics and Applications")

    def test_slugify_title_caps_filename_length(self) -> None:
        slug = _slugify_title("a" * 300, fallback="paper")
        self.assertLessEqual(len(slug), 120)
        self.assertTrue(slug)

    def test_playground_resolves_user_mcts_limits_without_overriding_drafts(self) -> None:
        num_drafts, max_total_nodes = FrontierSciencePlayground._resolve_mcts_limits(
            {"num_drafts": 2, "max_total_nodes": 9}
        )
        self.assertEqual(num_drafts, 2)
        self.assertEqual(max_total_nodes, 9)

    def test_playground_resolves_total_nodes_from_legacy_max_iterations(self) -> None:
        num_drafts, max_total_nodes = FrontierSciencePlayground._resolve_mcts_limits(
            {"num_drafts": 3, "max_iterations": 4}
        )
        self.assertEqual(num_drafts, 3)
        self.assertEqual(max_total_nodes, 4)

    def test_mcts_select_prefers_less_expanded_draft_branch_before_deepening(self) -> None:
        mcts = FrontierScienceMCTS(
            config=FrontierScienceMCTSConfig(
                num_drafts=2,
                max_depth=4,
                max_total_nodes=6,
                max_children_per_node=2,
                allow_repeat_improve=False,
            ),
            action_runner=lambda node, action: {
                "node_id": f"{action}_{node.node_id}_{len(node.children)}",
                "trajectory": {"dialogs": []},
                "current_answer": f"{node.node_id}->{action}",
                "state_summary": f"{node.node_id}->{action}",
            },
            metric_evaluator=lambda node, is_final: {
                "overall_score": 0.5,
                "final_answer_score": 0.5,
                "trajectory_score": 0.5,
                "subscores": {},
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [],
                "is_valid": True,
                "reason": "mock",
            },
        )

        draft_a = mcts.attach_child(
            mcts.root,
            "draft",
            {"node_id": "draft_a", "trajectory": {"dialogs": []}, "current_answer": "a", "state_summary": "a"},
        )
        draft_b = mcts.attach_child(
            mcts.root,
            "draft",
            {"node_id": "draft_b", "trajectory": {"dialogs": []}, "current_answer": "b", "state_summary": "b"},
        )
        mcts.attach_child(
            draft_a,
            "improve",
            {"node_id": "improve_a", "trajectory": {"dialogs": []}, "current_answer": "a+", "state_summary": "a+"},
        )
        mcts.root.visits = 3
        draft_a.visits = 2
        draft_a.value_sum = 1.6
        draft_b.visits = 1
        draft_b.value_sum = 0.7

        selected = mcts._select(mcts.root)
        self.assertEqual(selected.node_id, "draft_b")

    def test_mcts_choose_batch_reserves_other_frontiers_in_parallel_mode(self) -> None:
        mcts = FrontierScienceMCTS(
            config=FrontierScienceMCTSConfig(
                num_drafts=2,
                max_depth=3,
                max_total_nodes=10,
                max_children_per_node=2,
                allow_repeat_improve=False,
            ),
            action_runner=lambda node, action: {
                "node_id": f"{action}_{node.node_id}_{len(node.children)}",
                "trajectory": {"dialogs": []},
                "current_answer": f"{node.node_id}->{action}",
                "state_summary": f"{node.node_id}->{action}",
            },
            metric_evaluator=lambda node, is_final: {
                "overall_score": 0.9 if node.node_id == "draft_a" else 0.2,
                "final_answer_score": 0.5,
                "trajectory_score": 0.5,
                "subscores": {},
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [],
                "is_valid": True,
                "reason": "mock",
            },
        )

        draft_a = mcts.attach_child(
            mcts.root,
            "draft",
            {"node_id": "draft_a", "trajectory": {"dialogs": []}, "current_answer": "a", "state_summary": "a"},
        )
        draft_b = mcts.attach_child(
            mcts.root,
            "draft",
            {"node_id": "draft_b", "trajectory": {"dialogs": []}, "current_answer": "b", "state_summary": "b"},
        )
        mcts.root.visits = 10
        draft_a.visits = 1
        draft_a.value_sum = 1.0
        draft_b.visits = 1
        draft_b.value_sum = 0.2

        batch = mcts._choose_batch(2)
        self.assertEqual([(node.node_id, action) for node, action in batch], [("draft_a", "improve"), ("draft_b", "improve")])

    def test_mcts_final_eval_invalid_best_falls_back_to_next_valid_node(self) -> None:
        final_calls: list[tuple[str, bool]] = []

        def action_runner(node, action_type):
            return {
                "node_id": f"{action_type}_{len(node.children)}_{len(final_calls)}_{node.node_id}",
                "trajectory": {"dialogs": []},
                "current_answer": f"{node.node_id}->{action_type}",
                "state_summary": f"{node.node_id}->{action_type}",
            }

        def metric_evaluator(node, is_final):
            final_calls.append((node.node_id, is_final))
            if is_final and node.node_id == "draft_0_0_root":
                return {
                    "overall_score": 0.0,
                    "final_answer_score": 0.0,
                    "trajectory_score": 0.0,
                    "subscores": {},
                    "strengths": [],
                    "weaknesses": ["invalid"],
                    "improvement_suggestions": [],
                    "is_valid": False,
                    "reason": "final_invalid",
                }
            score = 0.9 if node.node_id == "draft_0_0_root" else 0.7
            return {
                "overall_score": score,
                "final_answer_score": score,
                "trajectory_score": score,
                "subscores": {},
                "strengths": [],
                "weaknesses": [],
                "improvement_suggestions": [],
                "is_valid": True,
                "reason": "mock",
            }

        mcts = FrontierScienceMCTS(
            config=FrontierScienceMCTSConfig(
                num_drafts=2,
                max_depth=1,
                max_total_nodes=2,
                use_full_eval_for_final=True,
            ),
            action_runner=action_runner,
            metric_evaluator=metric_evaluator,
        )

        result = mcts.run()
        self.assertEqual(result["best_node"].node_id, "draft_1_1_root")
        self.assertTrue(result["best_metric_result"]["is_valid"])
        self.assertIn(("draft_0_0_root", True), final_calls)

    def test_shared_context_preserves_fifo_results_for_repeated_same_tool(self) -> None:
        with TemporaryDirectory() as tmpdir:
            context = FrontierScienceSharedContext(Path(tmpdir))
            trajectory = {
                "dialogs": [
                    {
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "read_paper_pdf",
                                            "arguments": '{"query":"weak value amplification"}',
                                        }
                                    },
                                    {
                                        "function": {
                                            "name": "read_paper_pdf",
                                            "arguments": '{"query":"postselection probability"}',
                                        }
                                    },
                                    {
                                        "function": {
                                            "name": "read_paper_pdf",
                                            "arguments": '{"query":"quantum Fisher information"}',
                                        }
                                    },
                                ],
                            },
                            {"role": "tool", "name": "read_paper_pdf", "content": "result-1"},
                            {"role": "tool", "name": "read_paper_pdf", "content": "result-2"},
                            {"role": "tool", "name": "read_paper_pdf", "content": "result-3"},
                        ]
                    }
                ]
            }

            context.add_trajectory(
                node_id="node",
                parent_id="root",
                action_type="draft",
                trajectory=trajectory,
                metric_result={"overall_score": 0.8, "is_valid": True},
            )

            payload = json.loads((Path(tmpdir) / "nodes" / "node.json").read_text(encoding="utf-8"))
            records = payload["tool_records"]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["results"], ["result-1"])
            self.assertEqual(records[1]["results"], ["result-2"])
            self.assertEqual(records[2]["results"], ["result-3"])

    def test_shared_context_appends_jsonl_under_trajectories_root(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            context = FrontierScienceSharedContext(
                base_dir / "tool_trace",
                trajectory_dir=base_dir / "trajectories",
            )
            node = FrontierScienceSearchNode(
                node_id="draft_1",
                parent_id="root",
                depth=1,
                action_type="draft",
                current_answer="Initial answer with evidence.",
                state_summary="draft answer metric_overall=0.71",
            )
            trajectory = {
                "dialogs": [
                    {
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "search_web",
                                            "arguments": '{"query":"weak value amplification review"}',
                                        }
                                    }
                                ],
                            },
                            {"role": "tool", "name": "search_web", "content": "search-result"},
                        ]
                    }
                ]
            }

            context.add_trajectory(
                node_id=node.node_id,
                parent_id=node.parent_id,
                action_type=node.action_type or "draft",
                trajectory=trajectory,
                metric_result={"overall_score": 0.71, "is_valid": True},
                search_node=node,
                exploration_weight=1.414,
            )

            traj_path = base_dir / "trajectories" / "trajectory.jsonl"
            lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["node_id"], "draft_1")
            self.assertEqual(payload["current_answer"], "Initial answer with evidence.")
            self.assertEqual(payload["tool_records"][0]["tool"], "search_web")

    def test_shared_context_persists_search_stats_with_uct(self) -> None:
        with TemporaryDirectory() as tmpdir:
            context = FrontierScienceSharedContext(Path(tmpdir))
            parent = FrontierScienceSearchNode(
                node_id="parent",
                parent_id="root",
                depth=1,
                action_type="draft",
                visits=5,
                value_sum=3.0,
            )
            child = FrontierScienceSearchNode(
                node_id="child",
                parent_id="parent",
                depth=2,
                action_type="improve",
                parent=parent,
                visits=2,
                value_sum=1.6,
            )
            context.add_trajectory(
                node_id=child.node_id,
                parent_id=child.parent_id,
                action_type=child.action_type or "improve",
                trajectory={"dialogs": []},
                metric_result={"overall_score": 0.8, "is_valid": True},
                search_node=child,
                exploration_weight=1.414,
            )
            payload = (Path(tmpdir) / "nodes" / "child.json").read_text(encoding="utf-8")
            self.assertIn('"uct"', payload)
            self.assertIn('"mean_value": 0.8', payload)


if __name__ == "__main__":
    unittest.main()
