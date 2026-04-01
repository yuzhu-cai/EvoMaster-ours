from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from playground.embomaster.core.monitoring import EmboMasterMonitorWriter
from playground.embomaster.dashboard.backend import DashboardManager, TaskRunStore


def build_minimal_trajectory_entry(workspace_path: Path) -> dict:
    return {
        "task_id": "exp_001_coding_r1_coding_step_1",
        "exp_name": "EmboMaster",
        "exp_index": 1,
        "status": "completed",
        "steps": 1,
        "trajectory": {
            "task_id": "exp_001_coding_r1",
            "agent_name": "coding",
            "step": 1,
            "dialogs": [
                {
                    "messages": [
                        {"role": "system", "content": "system"},
                        {
                            "role": "user",
                            "content": (
                                "Task ID: exp_001_coding_r1\n"
                                "Task Type: coding\n"
                                "Round: 1/3\n"
                                "Workspace ID: ws-r1\n"
                                "Parent Workspace ID: None\n"
                                f"Workspace Codebase Path: {workspace_path}\n"
                                "Workspace Source Type: original\n"
                                "Workspace Large Dir Mount Count: 0\n"
                            ),
                        },
                    ]
                }
            ],
            "steps": [
                {
                    "step_id": 1,
                    "assistant_message": {
                        "role": "assistant",
                        "content": "round done",
                        "tool_calls": [
                            {
                                "id": "finish-1",
                                "function": {
                                    "name": "finish",
                                    "arguments": json.dumps({"message": "round done"}),
                                },
                            }
                        ],
                    },
                    "tool_responses": [
                        {
                            "role": "tool",
                            "name": "debug_test",
                            "content": "[debug_test] success (exit_code=0) mode=k8s_debug_pod working_dir=/tmp\nok",
                            "meta": {
                                "info": {
                                    "pod_name": "debug-pod-1",
                                    "namespace": "robotwin",
                                    "mode": "k8s_debug_pod",
                                    "exit_code": 0,
                                }
                            },
                        }
                    ],
                }
            ],
        },
    }


class DashboardBackendTest(unittest.TestCase):
    def test_dashboard_manager_discovers_top_level_and_nested_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dirs = [
                root / "top_run",
                root / "robotwin" / "pick_banana" / "vendor" / "run_nested",
            ]
            for run_dir in run_dirs:
                trajectory_dir = run_dir / "trajectories" / "task_0"
                codebase_dir = run_dir / "workspaces" / "task_0" / "codebase"
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                codebase_dir.mkdir(parents=True, exist_ok=True)
                trajectory_entry = build_minimal_trajectory_entry(codebase_dir)
                (trajectory_dir / "trajectory.jsonl").write_text(
                    json.dumps(trajectory_entry, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            manager = DashboardManager(root, preview_chars=120)
            runs = manager.list_runs()["runs"]
            run_ids = {item["run_id"] for item in runs}

            self.assertIn("top_run:task_0", run_ids)
            self.assertIn("robotwin/pick_banana/vendor/run_nested:task_0", run_ids)

    def test_dashboard_manager_reuses_recent_scan_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "top_run"
            trajectory_dir = run_dir / "trajectories" / "task_0"
            codebase_dir = run_dir / "workspaces" / "task_0" / "codebase"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            codebase_dir.mkdir(parents=True, exist_ok=True)
            trajectory_entry = build_minimal_trajectory_entry(codebase_dir)
            (trajectory_dir / "trajectory.jsonl").write_text(
                json.dumps(trajectory_entry, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            manager = DashboardManager(root, preview_chars=120)
            manager._scan_interval_sec = 3600

            with mock.patch.object(
                manager,
                "_discover_run_candidates",
                side_effect=AssertionError("unexpected rescan"),
            ):
                runs = manager.list_runs()["runs"]

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], "top_run:task_0")

    def test_monitor_writer_creates_structured_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            codebase_dir = run_dir / "workspaces" / "task_0" / "codebase"
            submission_dir = codebase_dir / "submission"
            submission_dir.mkdir(parents=True, exist_ok=True)

            writer = EmboMasterMonitorWriter(run_dir=run_dir, task_id="task_0")
            round_result = {
                "round_index": 1,
                "status": "completed",
                "steps": 3,
                "coding_result": "updated config",
                "feedback": "keep best parent",
                "k8s_status": "succeeded",
                "metric_value": 0.82,
                "metric_source": "pattern:success",
                "metric_valid": True,
                "result_valid": True,
                "validation_errors": [],
                "workspace_id": "ws-r1",
                "parent_workspace_id": "",
                "parent_choice_used": "none",
                "parent_choice_reason": "first round",
                "workspace_codebase_path": str(codebase_dir),
                "workspace_source_type": "original",
                "workspace_large_dirs_count": 0,
                "workspace_large_dirs": [],
                "submission_dir": str(submission_dir),
                "session_dir": str(run_dir / "workspaces" / "task_0"),
                "artifacts_summary": {"has_submission_csv": False},
                "k8s_result": {
                    "manifest_path": "manifest.yaml",
                    "prepared_manifest_path": "prepared.yaml",
                    "job_name": "job-r1",
                    "namespace": "robotwin",
                    "submit": {"status": "applied", "exit_code": 0},
                    "wait": {"status": "succeeded"},
                    "logs": {"output": "success rate: 0.82"},
                    "pods": [{"pod_name": "job-r1-pod", "status": "running", "namespace": "robotwin"}],
                },
            }
            writer.record_round(round_result, best_metric=0.82, best_round_index=1, total_rounds_so_far=1)
            writer.record_final_result(
                {
                    "status": "completed",
                    "total_rounds": 1,
                    "stopped_reason": "completed_all_rounds",
                    "best_metric": 0.82,
                    "best_round_index": 1,
                    "invalid_round_count": 0,
                },
                task_description="demo task",
            )

            latest_path = run_dir / "monitor" / "tasks" / "task_0" / "latest.json"
            final_path = run_dir / "monitor" / "tasks" / "task_0" / "final.json"
            round_path = run_dir / "monitor" / "tasks" / "task_0" / "rounds" / "round_001.json"

            self.assertTrue(latest_path.exists())
            self.assertTrue(final_path.exists())
            self.assertTrue(round_path.exists())

            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(latest_payload["overview"]["total_rounds"], 1)
            self.assertEqual(latest_payload["overview"]["best_round_index"], 1)

    def test_task_run_store_reads_structured_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            trajectory_dir = run_dir / "trajectories" / "task_0"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            codebase_dir = run_dir / "workspaces" / "task_0" / "codebase"
            codebase_dir.mkdir(parents=True, exist_ok=True)

            trajectory_entry = build_minimal_trajectory_entry(codebase_dir)
            (trajectory_dir / "trajectory.jsonl").write_text(
                json.dumps(trajectory_entry, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            writer = EmboMasterMonitorWriter(run_dir=run_dir, task_id="task_0")
            writer.record_round(
                {
                    "round_index": 1,
                    "status": "completed",
                    "steps": 1,
                    "coding_result": "round done",
                    "feedback": "",
                    "k8s_status": "succeeded",
                    "metric_value": 0.91,
                    "metric_source": "structured",
                    "metric_valid": True,
                    "result_valid": True,
                    "validation_errors": [],
                    "workspace_id": "ws-r1",
                    "parent_workspace_id": "",
                    "parent_choice_used": "none",
                    "parent_choice_reason": "first round",
                    "workspace_codebase_path": str(codebase_dir),
                    "workspace_source_type": "original",
                    "workspace_large_dirs_count": 0,
                    "workspace_large_dirs": [],
                    "submission_dir": str(codebase_dir / "submission"),
                    "session_dir": str(run_dir / "workspaces" / "task_0"),
                    "artifacts_summary": {"has_submission_csv": False},
                    "k8s_result": {
                        "job_name": "job-r1",
                        "namespace": "robotwin",
                        "logs": {"output": "success rate: 0.91"},
                    },
                },
                best_metric=0.91,
                best_round_index=1,
                total_rounds_so_far=1,
            )
            writer.record_final_result(
                {
                    "status": "completed",
                    "total_rounds": 1,
                    "stopped_reason": "completed_all_rounds",
                    "best_metric": 0.91,
                    "best_round_index": 1,
                    "invalid_round_count": 0,
                },
                task_description="demo task",
            )

            store = TaskRunStore(
                trajectory_path=trajectory_dir / "trajectory.jsonl",
                run_label="demo",
                run_dir=run_dir,
                task_id="task_0",
                preview_chars=120,
            )
            overview = store.get_overview()
            rounds = store.get_rounds()
            route = store.get_route()
            pods = store.get_pod_items()

            self.assertEqual(overview["monitor_mode"], "structured")
            self.assertEqual(overview["best_metric"], 0.91)
            self.assertEqual(len(rounds), 1)
            self.assertEqual(rounds[0]["k8s_status"], "succeeded")
            self.assertEqual(len(route["nodes"]), 1)
            self.assertTrue(any(item.get("job_name") == "job-r1" for item in pods))

    def test_pod_cache_refresh_and_log_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            trajectory_dir = run_dir / "trajectories" / "task_0"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            codebase_dir = run_dir / "workspaces" / "task_0" / "codebase"
            codebase_dir.mkdir(parents=True, exist_ok=True)

            trajectory_entry = build_minimal_trajectory_entry(codebase_dir)
            (trajectory_dir / "trajectory.jsonl").write_text(
                json.dumps(trajectory_entry, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            writer = EmboMasterMonitorWriter(run_dir=run_dir, task_id="task_0")
            writer.record_round(
                {
                    "round_index": 1,
                    "status": "completed",
                    "steps": 1,
                    "coding_result": "round done",
                    "feedback": "",
                    "k8s_status": "running",
                    "metric_value": 0.5,
                    "metric_source": "structured",
                    "metric_valid": True,
                    "result_valid": True,
                    "validation_errors": [],
                    "workspace_id": "ws-r1",
                    "parent_workspace_id": "",
                    "parent_choice_used": "none",
                    "parent_choice_reason": "first round",
                    "workspace_codebase_path": str(codebase_dir),
                    "workspace_source_type": "original",
                    "workspace_large_dirs_count": 0,
                    "workspace_large_dirs": [],
                    "submission_dir": str(codebase_dir / "submission"),
                    "session_dir": str(run_dir / "workspaces" / "task_0"),
                    "artifacts_summary": {"has_submission_csv": False},
                    "k8s_result": {
                        "job_name": "job-r1",
                        "namespace": "robotwin",
                        "logs": {"output": "stored fallback logs"},
                    },
                },
                best_metric=0.5,
                best_round_index=1,
                total_rounds_so_far=1,
            )

            store = TaskRunStore(
                trajectory_path=trajectory_dir / "trajectory.jsonl",
                run_label="demo",
                run_dir=run_dir,
                task_id="task_0",
                preview_chars=120,
            )
            store._ensure_cluster_poller = lambda: None  # type: ignore[method-assign]

            def fake_run(cmd, capture_output=True, text=True, timeout=0):  # type: ignore[no-untyped-def]
                joined = " ".join(cmd)
                if "jsonpath=" in joined:
                    return subprocess.CompletedProcess(cmd, 0, stdout="job-r1-pod\n", stderr="")
                if "get pod" in joined:
                    payload = {
                        "status": {
                            "phase": "Running",
                            "startTime": "2026-03-19T08:00:00Z",
                            "hostIP": "10.0.0.1",
                            "podIP": "10.0.0.2",
                            "containerStatuses": [
                                {
                                    "name": "main",
                                    "ready": True,
                                    "restartCount": 0,
                                    "state": {"running": {}},
                                    "image": "robotwin:test",
                                }
                            ],
                        },
                        "spec": {"nodeName": "gpu-node001"},
                    }
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=json.dumps(payload, ensure_ascii=False),
                        stderr="",
                    )
                if " logs " in f" {joined} ":
                    return subprocess.CompletedProcess(cmd, 0, stdout="live pod logs\nline2\n", stderr="")
                raise AssertionError(f"unexpected subprocess command: {joined}")

            with mock.patch("playground.embomaster.dashboard.backend.subprocess.run", side_effect=fake_run):
                payload = store.get_pod_payload(refresh=True, tail=400)

            self.assertEqual(payload["cache"]["log_tail"], 400)
            target_item = next(item for item in payload["items"] if item.get("job_name") == "job-r1")
            self.assertEqual(target_item["resolved_pod_name"], "job-r1-pod")
            self.assertEqual(target_item["status"], "running")
            self.assertEqual(target_item["ready_summary"], "1/1")
            self.assertTrue(target_item["has_live_logs"])

            with mock.patch(
                "playground.embomaster.dashboard.backend.subprocess.run",
                side_effect=AssertionError("unexpected subprocess run"),
            ):
                logs_payload = store.get_pod_logs(
                    pod_name="",
                    namespace="robotwin",
                    tail=400,
                    job_name="job-r1",
                    round_index=1,
                    refresh=False,
                )

            self.assertTrue(logs_payload["ok"])
            self.assertEqual(logs_payload["source"], "live_cache")
            self.assertIn("live pod logs", logs_payload["logs"])


if __name__ == "__main__":
    unittest.main()
