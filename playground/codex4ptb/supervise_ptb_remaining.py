#!/usr/bin/env python3
"""Long-running supervisor for the local PostTrainBench Codex waves.

The script keeps at most one wave active, launches pending PTB combos in
batches of up to 8, and relaunches only dead/invalid run directories.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable


PROJECT_ROOT = Path("/data/yuzhu/Devs/EvoMaster-ours")
RUNS_DIR = PROJECT_ROOT / "runs"
LATEST_WAVE = RUNS_DIR / "codex4ptb_latest_wave.txt"
RUNNER = PROJECT_ROOT / "playground" / "codex4ptb" / "codex4ptb_runner.py"

EVALS = [
    "aime2025",
    "arenahardwriting",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "healthbench",
]

# This order matches the remaining-wave plan: completed combos are skipped.
MODELS = [
    "google/gemma-3-4b-pt",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._-") or "value"


def log(msg: str) -> None:
    line = f"{iso_now()} {msg}"
    print(line, flush=True)
    with (RUNS_DIR / "codex4ptb_supervisor.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pid_alive(pid: int | str) -> bool:
    try:
        pid_int = int(pid)
        os.kill(pid_int, 0)
        stat = Path(f"/proc/{pid_int}/stat")
        if stat.exists():
            parts = stat.read_text(encoding="utf-8", errors="replace").split()
            if len(parts) > 2 and parts[2] == "Z":
                return False
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def load_pids(wave: Path) -> list[dict[str, str]]:
    pids = wave / "pids.tsv"
    if not pids.exists():
        return []
    rows: list[dict[str, str]] = []
    with pids.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def save_pids(wave: Path, rows: list[dict[str, str]]) -> None:
    header = ["pid", "gpu", "eval", "model_to_train", "run_name", "launcher_log"]
    tmp = wave / "pids.tsv.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(k, "")) for k in header) + "\n")
    tmp.replace(wave / "pids.tsv")


def active_run_dirs(wave: Path) -> list[Path]:
    if not wave.is_dir():
        return []
    return sorted(p for p in wave.iterdir() if p.is_dir() and p.name.startswith("gpu") and (p / "run_config.json").exists())


def combo_for_run(run_dir: Path) -> tuple[str, str] | None:
    rc = run_dir / "run_config.json"
    if not rc.exists():
        return None
    try:
        data = read_json(rc)
    except Exception:
        return None
    ev = data.get("eval")
    model = data.get("model_to_train")
    if isinstance(ev, str) and isinstance(model, str):
        return model, ev
    return None


def valid_run(run_dir: Path) -> bool:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return False
    try:
        data = read_json(summary)
    except Exception:
        return False
    if data.get("status") != "completed":
        return False
    if (data.get("eval") or {}).get("status") != "completed":
        return False
    if not (run_dir / "metrics.json").exists():
        return False
    if not (run_dir / "task" / "final_model").is_dir():
        return False
    return True


def all_wave_runs_valid(wave: Path) -> bool:
    runs = active_run_dirs(wave)
    return bool(runs) and all(valid_run(d) for d in runs)


def latest_wave() -> Path | None:
    if not LATEST_WAVE.exists():
        return None
    text = LATEST_WAVE.read_text(encoding="utf-8").strip()
    if not text:
        return None
    p = Path(text)
    return p if p.is_absolute() else PROJECT_ROOT / p


def scan_completed() -> dict[tuple[str, str], Path]:
    completed: dict[tuple[str, str], Path] = {}
    for wave in sorted(RUNS_DIR.glob("codex4ptb_wave_*")):
        if not wave.is_dir():
            continue
        for run_dir in active_run_dirs(wave):
            combo = combo_for_run(run_dir)
            if combo and valid_run(run_dir):
                completed[combo] = run_dir
    return completed


def official_matrix() -> list[tuple[str, str]]:
    return [(model, ev) for model in MODELS for ev in EVALS]


def monitor_alive(wave: Path) -> bool:
    pid_file = wave / "monitor.pid"
    if not pid_file.exists():
        return False
    try:
        return pid_alive(int(pid_file.read_text(encoding="utf-8").strip()))
    except Exception:
        return False


def template_wave() -> Path | None:
    current = latest_wave()
    if current and (current / "monitor_loop.sh").exists() and (current / "status.sh").exists():
        return current
    waves = [p for p in sorted(RUNS_DIR.glob("codex4ptb_wave_*")) if (p / "monitor_loop.sh").exists()]
    return waves[-1] if waves else None


def install_wave_helpers(wave: Path) -> None:
    tmpl = template_wave()
    if tmpl:
        for name in ("monitor_loop.sh", "status.sh"):
            src = tmpl / name
            if src.exists():
                dst = wave / name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                os.chmod(dst, 0o755)
    if not (wave / "monitor_loop.sh").exists():
        (wave / "monitor_loop.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
cd /data/yuzhu/Devs/EvoMaster-ours
WAVE=$(cat runs/codex4ptb_latest_wave.txt)
LOG="$WAVE/monitor.log"
ALERT="$WAVE/needs_attention.tsv"
[ -f "$ALERT" ] || printf 'time\\trun\\tissue\\tdetail\\n' > "$ALERT"
while true; do
  now=$(date -Is)
  echo "===== $now WAVE=$WAVE =====" >> "$LOG"
  awk 'NR>1{print $1}' "$WAVE/pids.tsv" | xargs -r ps -o pid,ppid,sid,stat,etime,cmd -p >> "$LOG" 2>&1 || true
  awk 'NR>1{print $1"\\t"$5}' "$WAVE/pids.tsv" | while IFS=$'\\t' read -r pid run_name; do
    if ! kill -0 "$pid" 2>/dev/null && [ ! -f "$WAVE/$run_name/summary.json" ]; then
      grep -Fq "$run_name" "$ALERT" 2>/dev/null || printf '%s\\t%s\\twrapper_dead\\tno summary.json\\n' "$now" "$run_name" >> "$ALERT"
    fi
  done
  total=$(find "$WAVE" -maxdepth 1 -mindepth 1 -type d -name 'gpu*' | wc -l)
  done_count=$(find "$WAVE" -mindepth 2 -maxdepth 2 -path "$WAVE/gpu*/summary.json" -type f | wc -l)
  alert_count=$(( $(wc -l < "$ALERT") - 1 ))
  if [ "$total" -gt 0 ] && [ "$done_count" -eq "$total" ] && [ "$alert_count" -eq 0 ]; then exit 0; fi
  sleep 300
done
""",
            encoding="utf-8",
        )
        os.chmod(wave / "monitor_loop.sh", 0o755)
    if not (wave / "status.sh").exists():
        (wave / "status.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
cd /data/yuzhu/Devs/EvoMaster-ours
WAVE=$(cat runs/codex4ptb_latest_wave.txt)
echo "WAVE=$WAVE"
awk 'NR>1{print $1}' "$WAVE/pids.tsv" | xargs -r ps -o pid,ppid,sid,stat,etime,cmd -p || true
""",
            encoding="utf-8",
        )
        os.chmod(wave / "status.sh", 0o755)


def start_monitor(wave: Path) -> int:
    if monitor_alive(wave):
        return int((wave / "monitor.pid").read_text(encoding="utf-8").strip())
    log_path = wave / "monitor_launcher.log"
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(wave / "monitor_loop.sh")],
        cwd=str(PROJECT_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (wave / "monitor.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    return proc.pid


def append_incident(wave: Path, run_name: str, action: str, detail: str) -> None:
    path = wave / "supervisor_incidents.tsv"
    if not path.exists():
        path.write_text("time\trun\taction\tdetail\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{iso_now()}\t{run_name}\t{action}\t{detail}\n")


def clear_alerts_for_run(wave: Path, run_name: str) -> None:
    path = wave / "needs_attention.tsv"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    kept = [lines[0]]
    kept.extend(line for line in lines[1:] if f"\t{run_name}\t" not in line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def get_runtime_defaults() -> dict[str, str]:
    defaults = {"CODEX_MODEL": "gpt-5.5", "HOURS": "10", "EVAL_LIMIT": "-1"}
    current = latest_wave()
    if not current:
        return defaults
    for row in load_pids(current):
        pid = row.get("pid")
        if not pid or not pid_alive(pid):
            continue
        try:
            env_data = Path(f"/proc/{int(pid)}/environ").read_bytes().split(b"\0")
        except Exception:
            continue
        for kv in env_data:
            if b"=" not in kv:
                continue
            k, v = kv.split(b"=", 1)
            key = k.decode("utf-8", "replace")
            if key in defaults and v:
                defaults[key] = v.decode("utf-8", "replace")
        break
    return defaults


def launch_wrapper(
    wave: Path,
    *,
    gpu: str,
    eval_name: str,
    model: str,
    run_name: str,
    hours: str,
    eval_limit: str,
    codex_model: str,
    restart: bool = False,
) -> tuple[int, Path]:
    launcher_log = wave / f"launcher_{run_name.replace('/', '_')}.log"
    env = os.environ.copy()
    env.update(
        {
            "GPU": gpu,
            "EVAL_NAME": eval_name,
            "MODEL_TO_TRAIN": model,
            "RUN_NAME": run_name,
            "RUNS_DIR_ABS": str(wave),
            "HOURS": hours,
            "EVAL_LIMIT": eval_limit,
            "CODEX_MODEL": codex_model,
        }
    )
    restart_text = " restart=1" if restart else ""
    script = f"""
cd /data/yuzhu/Devs/EvoMaster-ours
set -a
source .env >/dev/null 2>&1
set +a
export PYTHONUNBUFFERED=1
export CODEX_MODEL="${{CODEX_MODEL:-{codex_model}}}"
export HOURS="${{HOURS:-{hours}}}"
export EVAL_LIMIT="${{EVAL_LIMIT:-{eval_limit}}}"
echo "[$(date -Is)] starting gpu=${{GPU}} eval=${{EVAL_NAME}} model=${{MODEL_TO_TRAIN}} hours=${{HOURS}}{restart_text}"
python playground/codex4ptb/codex4ptb_runner.py run \\
  --eval "$EVAL_NAME" \\
  --model-to-train "$MODEL_TO_TRAIN" \\
  --codex-model "$CODEX_MODEL" \\
  --hours "$HOURS" \\
  --num-gpus 1 \\
  --cuda-visible-devices "$GPU" \\
  --eval-limit "$EVAL_LIMIT" \\
  --runs-dir "$RUNS_DIR_ABS" \\
  --run-name "$RUN_NAME"
status=$?
echo "[$(date -Is)] finished status=${{status}} gpu=${{GPU}} eval=${{EVAL_NAME}} model=${{MODEL_TO_TRAIN}}{restart_text}"
exit "$status"
"""
    handle = launcher_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-c", script],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid, launcher_log


def archive_path(wave: Path, run_name: str) -> Path:
    base = wave / f"failed_{utc_stamp()}_{run_name}"
    candidate = base
    n = 1
    while candidate.exists():
        n += 1
        candidate = wave / f"{base.name}_{n}"
    return candidate


def relaunch_dead_invalid(wave: Path, row: dict[str, str], defaults: dict[str, str]) -> bool:
    run_name = row["run_name"]
    run_dir = wave / run_name
    if valid_run(run_dir):
        return False
    if pid_alive(row["pid"]):
        return False

    archive = archive_path(wave, run_name)
    if run_dir.exists():
        shutil.move(str(run_dir), str(archive))
    old_log = Path(row.get("launcher_log", ""))
    if old_log.exists():
        old_log.rename(old_log.with_suffix(old_log.suffix + f".failed_{utc_stamp()}"))

    pid, launcher = launch_wrapper(
        wave,
        gpu=row["gpu"],
        eval_name=row["eval"],
        model=row["model_to_train"],
        run_name=run_name,
        hours=defaults["HOURS"],
        eval_limit=defaults["EVAL_LIMIT"],
        codex_model=defaults["CODEX_MODEL"],
        restart=True,
    )
    row["pid"] = str(pid)
    row["launcher_log"] = str(launcher)
    clear_alerts_for_run(wave, run_name)
    append_incident(wave, run_name, "restart", f"Archived invalid/dead run to {archive.name}; new pid {pid}")
    log(f"restarted run={run_name} gpu={row['gpu']} pid={pid}")
    return True


def repair_active_wave(wave: Path, defaults: dict[str, str]) -> int:
    rows = load_pids(wave)
    repaired = 0
    for row in rows:
        if relaunch_dead_invalid(wave, row, defaults):
            repaired += 1
    if repaired:
        save_pids(wave, rows)
    install_wave_helpers(wave)
    mon_pid = start_monitor(wave)
    if repaired:
        append_incident(wave, "monitor_loop", "ensure_running", f"monitor pid {mon_pid}")
    return repaired


def write_manifest(wave: Path, batch: list[tuple[str, str]]) -> None:
    with (wave / "manifest.tsv").open("w", encoding="utf-8") as f:
        f.write("gpu\teval\tmodel_to_train\trun_name\n")
        for gpu, (model, ev) in enumerate(batch):
            f.write(f"{gpu}\t{ev}\t{model}\tgpu{gpu}_{ev}_{safe_name(model)}_10h\n")


def start_wave(batch: list[tuple[str, str]], defaults: dict[str, str]) -> Path:
    count = len(list(RUNS_DIR.glob("codex4ptb_wave_*"))) + 1
    wave = RUNS_DIR / f"codex4ptb_wave_{local_stamp()}_batch{count}_{len(batch)}x10h"
    wave.mkdir(parents=True, exist_ok=False)
    install_wave_helpers(wave)
    write_manifest(wave, batch)
    LATEST_WAVE.write_text(str(wave.relative_to(PROJECT_ROOT)) + "\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    for gpu, (model, ev) in enumerate(batch):
        run_name = f"gpu{gpu}_{ev}_{safe_name(model)}_10h"
        pid, launcher = launch_wrapper(
            wave,
            gpu=str(gpu),
            eval_name=ev,
            model=model,
            run_name=run_name,
            hours=defaults["HOURS"],
            eval_limit=defaults["EVAL_LIMIT"],
            codex_model=defaults["CODEX_MODEL"],
        )
        rows.append(
            {
                "pid": str(pid),
                "gpu": str(gpu),
                "eval": ev,
                "model_to_train": model,
                "run_name": run_name,
                "launcher_log": str(launcher),
            }
        )
        log(f"launched wave={wave.name} gpu={gpu} eval={ev} model={model} pid={pid}")
    save_pids(wave, rows)
    (wave / "needs_attention.tsv").write_text("time\trun\tissue\tdetail\n", encoding="utf-8")
    (wave / "supervisor_incidents.tsv").write_text("time\trun\taction\tdetail\n", encoding="utf-8")
    mon_pid = start_monitor(wave)
    append_incident(wave, "wave", "start", f"batch_size={len(batch)} monitor_pid={mon_pid}")
    log(f"started wave={wave.name} monitor_pid={mon_pid}")
    return wave


def wave_state(wave: Path) -> dict:
    rows = load_pids(wave)
    alive = sum(1 for row in rows if pid_alive(row.get("pid", "0")))
    run_dirs = active_run_dirs(wave)
    valid = sum(1 for d in run_dirs if valid_run(d))
    needs = 0
    alert = wave / "needs_attention.tsv"
    if alert.exists():
        needs = max(0, len(alert.read_text(encoding="utf-8").splitlines()) - 1)
    return {
        "wave": str(wave),
        "alive": alive,
        "total": len(run_dirs),
        "valid": valid,
        "needs_attention": needs,
        "monitor_alive": monitor_alive(wave),
    }


def active_or_incomplete(wave: Path | None) -> bool:
    if wave is None or not wave.exists():
        return False
    state = wave_state(wave)
    if state["alive"] > 0:
        return True
    if state["total"] > 0 and state["valid"] < state["total"]:
        return True
    return False


def write_status(status: dict) -> None:
    write_json(RUNS_DIR / "codex4ptb_supervisor_status.json", status)


def pending_combos(completed: dict[tuple[str, str], Path]) -> list[tuple[str, str]]:
    return [combo for combo in official_matrix() if combo not in completed]


def active_combos(wave: Path | None) -> set[tuple[str, str]]:
    combos: set[tuple[str, str]] = set()
    if wave is None or not wave.exists():
        return combos
    for run_dir in active_run_dirs(wave):
        combo = combo_for_run(run_dir)
        if combo and not valid_run(run_dir):
            combos.add(combo)
    return combos


def signal_process(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--once", action="store_true", help="Run one supervisor iteration and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    defaults = get_runtime_defaults()
    log(f"supervisor_start interval={args.interval_sec}s batch_size={args.batch_size} defaults={{'CODEX_MODEL': '{defaults['CODEX_MODEL']}', 'HOURS': '{defaults['HOURS']}', 'EVAL_LIMIT': '{defaults['EVAL_LIMIT']}'}}")

    stop = {"value": False}

    def _stop(signum: int, _frame: object) -> None:
        log(f"received_signal {signum}; stopping after current iteration")
        stop["value"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while True:
        defaults = get_runtime_defaults()
        completed = scan_completed()
        wave = latest_wave()
        state = wave_state(wave) if wave else {}
        pending = pending_combos(completed)
        active = active_combos(wave)
        not_started = [combo for combo in pending if combo not in active]

        status = {
            "time": iso_now(),
            "completed": len(completed),
            "unfinished": len(pending),
            "active": len(active),
            "not_started": len(not_started),
            "latest_wave": str(wave) if wave else None,
            "latest_wave_state": state,
            "next_pending": [{"model": m, "eval": e} for m, e in not_started[: args.batch_size]],
        }
        write_status(status)
        log(
            f"status completed={len(completed)}/28 active={len(active)} "
            f"not_started={len(not_started)} unfinished={len(pending)} wave_state={state}"
        )

        if active_or_incomplete(wave):
            repaired = repair_active_wave(wave, defaults) if wave else 0
            if repaired:
                log(f"repair_count={repaired}")
        elif not pending:
            manifest = {
                "generated_at": iso_now(),
                "completed": [
                    {"model": model, "eval": ev, "run_dir": str(path)}
                    for (model, ev), path in sorted(completed.items())
                ],
            }
            write_json(RUNS_DIR / "codex4ptb_final_manifest.json", manifest)
            log("all PTB combos completed; supervisor exiting")
            return 0
        else:
            batch = pending[: args.batch_size]
            start_wave(batch, defaults)

        if args.once or stop["value"]:
            return 0
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
