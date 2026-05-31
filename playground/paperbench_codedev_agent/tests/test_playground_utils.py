from __future__ import annotations

import tarfile
import json
import time
from types import SimpleNamespace

from playground.paperbench_codedev_agent.core.playground import PaperBenchCodeDevPlayground
from evomaster.agent.tools.builtin.editor import EditorTool
from evomaster.utils.llm import BaseLLM


def make_playground(iteration_cfg: dict) -> PaperBenchCodeDevPlayground:
    playground = object.__new__(PaperBenchCodeDevPlayground)
    playground._iteration_cfg = lambda: iteration_cfg  # type: ignore[method-assign]
    return playground


def test_quality_gate_reports_missing_thresholds():
    playground = make_playground(
        {
            "quality_gate": {
                "enabled": True,
                "min_round": 3,
                "min_nonblank_lines": 1000,
                "min_python_files": 5,
                "require_clean_git": True,
                "require_audit_ok": True,
            }
        }
    )
    audit = {
        "ok": True,
        "git_status": "",
        "counts": {"nonblank_lines": 900, "python_files": 7},
    }

    status = playground._quality_gate_status(audit, elapsed_seconds=120, round_index=2)

    assert not status["passed"]
    assert "min_round=2<3" in status["missing"]
    assert "min_nonblank_lines=900<1000" in status["missing"]
    assert not any(item.startswith("min_python_files") for item in status["missing"])


def test_quality_gate_passes_when_thresholds_met():
    playground = make_playground(
        {
            "quality_gate": {
                "enabled": True,
                "min_round": 2,
                "min_nonblank_lines": 1000,
                "min_python_files": 5,
                "require_clean_git": True,
                "require_audit_ok": True,
            }
        }
    )
    audit = {
        "ok": True,
        "git_status": "",
        "counts": {"nonblank_lines": 1200, "python_files": 5},
    }

    status = playground._quality_gate_status(audit, elapsed_seconds=120, round_index=2)

    assert status["passed"]
    assert status["missing"] == []


def test_extract_grade_score_from_json_output():
    grade_status = {
        "status": "completed",
        "output": 'logs...\\n{"paper_id": "rice", "score": 0.7429}\\nFailure feedback...',
    }

    assert PaperBenchCodeDevPlayground._extract_grade_score(grade_status) == 0.7429


def test_bootstrap_seed_grade_runs_selects_best_clean_room_submission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper_id = "rice"
    low_src = tmp_path / "low_src"
    high_src = tmp_path / "high_src"
    low_src.mkdir()
    high_src.mkdir()
    (low_src / "README.md").write_text("low seed\n", encoding="utf-8")
    (high_src / "README.md").write_text("high seed\n", encoding="utf-8")
    (high_src / "module.py").write_text("VALUE = 5\n", encoding="utf-8")

    low_run = tmp_path / "grade_low"
    high_run = tmp_path / "grade_high"
    for run, src, score in [(low_run, low_src, 0.1), (high_run, high_src, 0.9)]:
        (run / paper_id).mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps([{"paper_id": paper_id, "submission": str(src)}]) + "\n",
            encoding="utf-8",
        )
        (run / paper_id / "grader_output.json").write_text(
            json.dumps({"score": score}) + "\n",
            encoding="utf-8",
        )

    host_submission = tmp_path / "workspace" / "submission"
    host_submission.mkdir(parents=True)
    playground = object.__new__(PaperBenchCodeDevPlayground)

    status = playground._bootstrap_from_seed_grade_runs(
        {
            "seed_grade_runs": [str(low_run), str(high_run)],
            "overwrite_existing": False,
            "commit": False,
        },
        paper_id,
        host_submission,
    )

    assert status["status"] == "applied"
    assert status["seed_score"] == 0.9
    assert (host_submission / "module.py").read_text(encoding="utf-8") == "VALUE = 5\n"
    assert not (host_submission / ".git").exists()


def test_editor_view_range_clamps_end_past_eof():
    class Session:
        def is_directory(self, path: str) -> bool:
            return False

        def read_file(self, path: str) -> str:
            return "a\nb\nc\n"

    output, _ = EditorTool()._view(Session(), "/tmp/example.txt", [2, 100], "file")

    assert "2\tb" in output
    assert "3\tc" in output


def test_editor_create_replaces_existing_file_with_undo_history():
    class Session:
        def __init__(self):
            self.files = {"/tmp/example.py": "OLD = 1\n"}

        def is_directory(self, path: str) -> bool:
            return False

        def is_file(self, path: str) -> bool:
            return path in self.files

        def path_exists(self, path: str) -> bool:
            return path in self.files

        def read_file(self, path: str) -> str:
            return self.files[path]

        def write_file(self, path: str, content: str) -> None:
            self.files[path] = content

    session = Session()
    tool = EditorTool()

    output, info = tool.execute(
        session,
        json.dumps({"command": "create", "path": "/tmp/example.py", "file_text": "NEW = 2\n"}),
    )

    assert "replaced successfully" in output
    assert info["replaced"] is True
    assert session.files["/tmp/example.py"] == "NEW = 2\n"
    assert tool._file_history["/tmp/example.py"][0][0] == "OLD = 1\n"


def test_historical_grade_feedback_extracts_high_deficit_leaves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper_id = "demo-paper"
    grade_run = tmp_path / "grade"
    (grade_run / paper_id).mkdir(parents=True)
    output = {
        "score": 0.25,
        "judge_output": {
            "graded_task_tree": {
                "id": "root",
                "sub_tasks": [
                    {
                        "id": "big",
                        "requirements": "Implement the high-weight algorithm path",
                        "weight": 5,
                        "score": 0,
                        "explanation": "The source repository has no implementation for it.",
                    },
                    {
                        "id": "done",
                        "requirements": "Already satisfied",
                        "weight": 1,
                        "score": 1,
                    },
                ],
            }
        },
    }
    (grade_run / paper_id / "grader_output.json").write_text(json.dumps(output), encoding="utf-8")
    playground = object.__new__(PaperBenchCodeDevPlayground)
    playground._cfg = lambda: {  # type: ignore[method-assign]
        "historical_feedback": {
            "enabled": True,
            "grade_runs": [str(grade_run)],
            "max_failed_leaves": 5,
            "max_chars": 4000,
        }
    }

    feedback = playground._historical_grade_feedback({"paper_id": paper_id})

    assert "Previous score for this paper: 0.25" in feedback
    assert "Implement the high-weight algorithm path" in feedback
    assert "Already satisfied" not in feedback


def test_codex_gap_feedback_lists_codex_passed_evomaster_failed_leaves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paper_id = "demo-paper"
    codex_run = tmp_path / "codex"
    evo_run = tmp_path / "evo"
    for run, score, leaf_score in [(codex_run, 0.8, 1), (evo_run, 0.2, 0)]:
        (run / paper_id).mkdir(parents=True)
        payload = {
            "score": score,
            "judge_output": {
                "graded_task_tree": {
                    "id": "root",
                    "sub_tasks": [
                        {
                            "id": "gap-leaf",
                            "requirements": "Implement the baseline delta that Codex already covers",
                            "weight": 3,
                            "score": leaf_score,
                            "explanation": "Prior EvoMaster missed the concrete implementation.",
                        },
                        {
                            "id": "shared-fail",
                            "requirements": "Both agents fail this leaf",
                            "weight": 1,
                            "score": 0,
                        },
                    ],
                }
            },
        }
        (run / paper_id / "grader_output.json").write_text(json.dumps(payload), encoding="utf-8")

    playground = object.__new__(PaperBenchCodeDevPlayground)
    playground._cfg = lambda: {  # type: ignore[method-assign]
        "codex_gap_feedback": {
            "enabled": True,
            "codex_grade_run": str(codex_run),
            "evomaster_grade_runs": [str(evo_run)],
            "max_gap_leaves": 5,
            "max_chars": 4000,
        }
    }

    feedback = playground._codex_gap_feedback({"paper_id": paper_id})

    assert "Codex=0.8" in feedback
    assert "prior EvoMaster=0.2" in feedback
    assert "Implement the baseline delta that Codex already covers" in feedback
    assert "Both agents fail this leaf" not in feedback


def test_llm_rate_limiter_uses_shared_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOMASTER_LLM_MIN_INTERVAL_SECONDS", "0.03")
    monkeypatch.setenv("EVOMASTER_LLM_RATE_LIMIT_JITTER_SECONDS", "0")
    monkeypatch.setenv("EVOMASTER_LLM_RATE_LIMIT_DIR", str(tmp_path))
    monkeypatch.setenv("EVOMASTER_LLM_RATE_LIMIT_KEY", "unit-test")
    dummy = SimpleNamespace(config=SimpleNamespace(base_url="http://example.test", model="gpt-test"))

    BaseLLM._apply_client_rate_limit(dummy)
    start = time.monotonic()
    BaseLLM._apply_client_rate_limit(dummy)

    assert time.monotonic() - start >= 0.02
    assert (tmp_path / "unit-test.lock").exists()


def test_host_finalize_packages_latest_submission(tmp_path):
    repo = tmp_path / "workspaces" / "paper" / "submission"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("paper reproduction\\n", encoding="utf-8")
    (repo / "module.py").write_text("VALUE = 1\\n", encoding="utf-8")
    pycache = repo / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython-313.pyc").write_bytes(b"cache")

    playground = object.__new__(PaperBenchCodeDevPlayground)
    playground.session = None
    playground._host_submission_path = lambda: repo  # type: ignore[method-assign]
    playground._host_workspace_path = lambda: str(tmp_path / "workspaces" / "paper")  # type: ignore[method-assign]

    status = playground._finalize_submission_artifact(
        {"workspace": "/workspace", "submission_tar_path": "/workspace/artifacts/submission.tar.gz"},
    )

    assert status["ok"]
    tar_path = tmp_path / "workspaces" / "paper" / "artifacts" / "submission.tar.gz"
    assert tar_path.exists()
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    assert "submission/README.md" in names
    assert "submission/module.py" in names
    assert not any("__pycache__" in name for name in names)
