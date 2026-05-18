#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import sys
import tarfile
from pathlib import Path


def audit_run(run: Path) -> dict[str, object]:
    rows = []
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        st = d / "status.json"
        md = d / "openclaw_metadata.json"
        gr = d / "grade.json"
        status = exit_code = judge = score = attempts = None
        if st.exists():
            try:
                status = json.loads(st.read_text()).get("status")
            except Exception as e:  # pragma: no cover - audit should keep going
                status = f"bad_status:{e}"
        if md.exists():
            try:
                md_json = json.loads(md.read_text())
                exit_code = md_json.get("openclaw_exit_code")
                attempts = md_json.get("openclaw_attempts")
            except Exception as e:
                exit_code = f"bad_md:{e}"
        if gr.exists():
            try:
                jo = (json.loads(gr.read_text()).get("paperbench_result") or {}).get("judge_output") or {}
                judge = jo.get("judge_type")
                score = jo.get("score")
            except Exception as e:
                judge = f"bad_grade:{e}"
        stdout = (d / "openclaw.stdout.log").read_text(errors="replace") if (d / "openclaw.stdout.log").exists() else ""
        stderr = (d / "openclaw.stderr.log").read_text(errors="replace") if (d / "openclaw.stderr.log").exists() else ""
        text = stdout + "\n" + stderr
        subs = sorted(d.glob("submissions/*/submission.tar.gz"))
        non_git = 0
        cfg_primary = cfg_timeout = cfg_max = cfg_native = None
        if subs:
            try:
                with tarfile.open(subs[-1], "r:gz") as tf:
                    names = tf.getnames()
                    non_git = sum(
                        1
                        for n in names
                        if n.startswith("./submission/")
                        and not n.startswith("./submission/.git/")
                        and n not in ("./submission/", "./submission/.git")
                    )
                    try:
                        cfg = json.load(tf.extractfile("./logs/openclaw-state/openclaw.json"))
                        cfg_primary = cfg.get("agents", {}).get("defaults", {}).get("model", {}).get("primary")
                        pid = cfg_primary.split("/")[0] if cfg_primary else None
                        provider = (cfg.get("models", {}).get("providers", {}).get(pid, {}) or {}) if pid else {}
                        cfg_timeout = provider.get("timeoutSeconds")
                        models = provider.get("models") or []
                        cfg_max = (models[0] or {}).get("maxTokens") if models else None
                        cfg_native = cfg.get("tools", {}).get("web", {}).get("search", {}).get("openaiCodex")
                    except Exception as e:
                        cfg_primary = f"bad_cfg:{e}"
            except Exception:
                non_git = -1
        model_failure = any(
            marker in text
            for marker in [
                "model idle timeout",
                "fetch-timeout",
                "incomplete turn",
                "couldn't generate",
                "400 status code",
            ]
        )
        rows.append(
            (
                d.name,
                status,
                exit_code,
                judge,
                score,
                "400 status code" in text,
                "Who am I" in stdout,
                model_failure,
                non_git,
                cfg_primary,
                cfg_timeout,
                cfg_max,
                json.dumps(cfg_native, sort_keys=True) if cfg_native is not None else None,
                len(subs),
                attempts,
            )
        )
    return {
        "RUN": str(run),
        "n_dirs": len(rows),
        "status": dict(collections.Counter(r[1] for r in rows)),
        "exit_codes": dict(collections.Counter(r[2] for r in rows)),
        "judges": dict(collections.Counter(r[3] for r in rows)),
        "scores": dict(collections.Counter(r[4] for r in rows)),
        "http400_count": sum(r[5] for r in rows),
        "bootstrap_prompt_count": sum(r[6] for r in rows),
        "model_failure_count": sum(r[7] for r in rows),
        "non_git_submission_dirs": sum(1 for r in rows if r[8] > 0),
        "submission_tar_dirs": sum(1 for r in rows if r[13] > 0),
        "cfg_primary": dict(collections.Counter(r[9] for r in rows)),
        "cfg_timeout": dict(collections.Counter(r[10] for r in rows)),
        "cfg_maxTokens": dict(collections.Counter(r[11] for r in rows)),
        "cfg_native": dict(collections.Counter(r[12] for r in rows)),
        "attempts": dict(collections.Counter(r[14] for r in rows)),
        "attention": [r for r in rows if r[1] != "done" or r[2] != 0 or r[5] or r[6] or r[7] or r[8] <= 0],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_openclaw_run.py RUN_DIR", file=sys.stderr)
        return 2
    out = audit_run(Path(sys.argv[1]))
    for k, v in out.items():
        if k == "attention":
            print(k, len(v))
            for row in v[:20]:
                print("ATTN", row)
        else:
            print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
