from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


SECRET_ENV_KEYS = {
    "OPENAI_API_KEY",
    "GRADER_OPENAI_API_KEY",
    "GPT_CHAT_MODEL",
    "GPT_BASE_URL",
    "OPENAI_BASE_URL",
    "CODEX4PAPERBENCH_JUDGE_STRUCTURED_MODEL",
}


def _read_proc_env(pid: int) -> dict[str, str]:
    data = Path(f"/proc/{pid}/environ").read_bytes()
    env: dict[str, str] = {}
    for entry in data.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _paper_id(run_dir: Path) -> str:
    try:
        grade = _load_json(run_dir / "grade.json")
        paper_id = (grade.get("paperbench_result") or {}).get("paper_id")
        if paper_id:
            return paper_id
    except Exception:
        pass
    name = run_dir.name
    suffix = name.rsplit("_", 1)[-1]
    if len(suffix) == 36 and suffix.count("-") == 4:
        return name[: -(len(suffix) + 1)]
    return name


def _is_simple_done(run_dir: Path, grade_id: int) -> bool:
    has_grader_output = any(
        run_dir.glob(f"submissions/*/submission_grader_output_{grade_id}.json")
    )
    if not has_grader_output:
        return False

    try:
        grade = _load_json(run_dir / "grade.json")
        judge = ((grade.get("paperbench_result") or {}).get("judge_output") or {})
        if (
            judge.get("judge_type") == "simple"
            and judge.get("num_invalid_leaf_nodes", 0) == 0
        ):
            return True
    except Exception:
        pass
    return False


def _leaf_count(run_dir: Path) -> int:
    for path in run_dir.glob("submissions/*/submission_grader_output_0.json"):
        try:
            obj = _load_json(path)
            count = obj.get("num_leaf_nodes")
            if isinstance(count, int):
                return count
        except Exception:
            continue
    return 10**9


def _try_lock(path: Path) -> int | None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
    return fd


def _run_one(
    run_dir: Path,
    *,
    env: dict[str, str],
    log_dir: Path,
    grade_id: int,
    timeout_seconds: int,
    retry_stop_after: int,
    retry_wait_max: int,
) -> tuple[str, int]:
    paper = _paper_id(run_dir)
    lock = run_dir / f".grade_simple_{grade_id}.lock"
    fd = _try_lock(lock)
    if fd is None:
        return paper, 75
    os.close(fd)

    log_path = log_dir / f"direct_parallel_{paper}.log"
    cmd = [
        "timeout",
        "--preserve-status",
        f"{timeout_seconds}s",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "paperbench",
        "python",
        "playground/codex4paperbench/grade_one_simple.py",
        run_dir.as_posix(),
        "--grade-id",
        str(grade_id),
        "--retry-stop-after",
        str(retry_stop_after),
        "--retry-wait-max",
        str(retry_wait_max),
    ]
    try:
        with log_path.open("ab") as out:
            out.write(f"\n--- START {paper} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ---\n".encode())
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
            out.write(
                f"--- END {paper} rc={proc.returncode} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ---\n".encode()
            )
        return paper, proc.returncode
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_group_dir", type=Path)
    parser.add_argument("--source-env-pid", type=int)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--leaf-concurrency", type=int, default=2)
    parser.add_argument("--grade-id", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--retry-stop-after", type=int, default=600)
    parser.add_argument("--retry-wait-max", type=int, default=15)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--log-dir", type=Path, default=Path("runs/codex4paperbench/launch-logs"))
    args = parser.parse_args()

    env = os.environ.copy()
    if args.source_env_pid:
        source_env = _read_proc_env(args.source_env_pid)
        for key in SECRET_ENV_KEYS:
            if source_env.get(key):
                env[key] = source_env[key]

    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("all_proxy", None)
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    env["no_proxy"] = "localhost,127.0.0.1,::1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "playground/codex4paperbench"
    if env.get("GPT_BASE_URL") and not env.get("OPENAI_BASE_URL"):
        env["OPENAI_BASE_URL"] = env["GPT_BASE_URL"]
    if env.get("OPENAI_API_KEY") and not env.get("GRADER_OPENAI_API_KEY"):
        env["GRADER_OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    if env.get("GPT_CHAT_MODEL"):
        env["CODEX4PAPERBENCH_JUDGE_STRUCTURED_MODEL"] = env["GPT_CHAT_MODEL"]
    env["CODEX4PAPERBENCH_JUDGE_CONTEXT_WINDOW"] = "128000"
    env["CODEX4PAPERBENCH_JUDGE_LEAF_CONCURRENCY"] = str(args.leaf_concurrency)
    env["CODEX4PAPERBENCH_OPENAI_TIMEOUT"] = "240"
    env["CODEX4PAPERBENCH_OPENAI_CLIENT_MAX_RETRIES"] = "0"
    env["CODEX4PAPERBENCH_JUDGE_RETRY_WAIT_MAX"] = str(args.retry_wait_max)
    env["CODEX4PAPERBENCH_JUDGE_RETRY_STOP_AFTER"] = str(args.retry_stop_after)

    missing_required = [key for key in ["OPENAI_API_KEY", "GPT_CHAT_MODEL", "GPT_BASE_URL"] if not env.get(key)]
    if missing_required:
        print(f"Missing required env keys: {', '.join(missing_required)}", file=sys.stderr)
        return 2

    included = set(args.include)
    excluded = set(args.exclude)
    run_dirs = []
    for run_dir in sorted(args.run_group_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        paper = _paper_id(run_dir)
        if included and paper not in included:
            continue
        if paper in excluded:
            continue
        if _is_simple_done(run_dir, args.grade_id):
            continue
        run_dirs.append(run_dir)
    run_dirs.sort(key=lambda p: (_leaf_count(p), _paper_id(p)))

    args.log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"parallel grading start jobs={args.jobs} leaf_concurrency={args.leaf_concurrency} "
        f"tasks={len(run_dirs)}"
    )
    for run_dir in run_dirs:
        print(f"TASK {_paper_id(run_dir)} leaves={_leaf_count(run_dir)}")

    work: queue.Queue[Path] = queue.Queue()
    for run_dir in run_dirs:
        work.put(run_dir)

    results: list[tuple[str, int]] = []
    results_lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                run_dir = work.get_nowait()
            except queue.Empty:
                return
            paper = _paper_id(run_dir)
            if _is_simple_done(run_dir, args.grade_id):
                rc = 0
                print(f"SKIP {paper}", flush=True)
            else:
                print(f"START {paper} {time.strftime('%H:%M:%S', time.gmtime())}", flush=True)
                paper, rc = _run_one(
                    run_dir,
                    env=env,
                    log_dir=args.log_dir,
                    grade_id=args.grade_id,
                    timeout_seconds=args.timeout_seconds,
                    retry_stop_after=args.retry_stop_after,
                    retry_wait_max=args.retry_wait_max,
                )
                print(f"DONE {paper} rc={rc} {time.strftime('%H:%M:%S', time.gmtime())}", flush=True)
            with results_lock:
                results.append((paper, rc))
            work.task_done()

    threads = [threading.Thread(target=worker, daemon=False) for _ in range(max(1, args.jobs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [(paper, rc) for paper, rc in results if rc not in (0,)]
    if failed:
        print("failures: " + ", ".join(f"{paper}:{rc}" for paper, rc in failed), file=sys.stderr)
    print("parallel grading finished")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
