# playground/ml_master/utils/mlebench_grade.py
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_report_json(output: str) -> dict[str, Any] | None:
    # 从输出里抓最后一个 JSON dict 且含 competition_id
    matches = re.findall(r"(\{[\s\S]*?\})", output)
    for s in reversed(matches):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "competition_id" in obj:
                return obj
        except Exception:
            pass
    return None


def grade_best_submission_and_save(
    *,
    workspace_dir: Path,
    competition_id: str,
    out_name: str = "mlebench_grade.json",
    overwrite: bool = False,
) -> Path:
    """
    在 workspace_dir 下寻找 best_submission/*.csv，执行 mlebench grade-sample，并把结果写到 workspace_dir/out_name
    返回写入的 json 路径。
    """
    workspace_dir = Path(workspace_dir)
    out_path = workspace_dir / out_name

    if out_path.exists() and not overwrite:
        return out_path

    payload: dict[str, Any] = {
        "competition_id": competition_id,
        "workspace_dir": str(workspace_dir),
        "created_at": datetime.now().isoformat(),
        "status": "unknown",
    }

    best_dir = workspace_dir / "best_submission"
    csvs = list(best_dir.glob("*.csv"))
    if not csvs:
        payload.update(
            status="skipped",
            error="best_submission_csv_not_found",
            expected_dir=str(best_dir),
        )
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    # 多个就取最新
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    csv_path = csvs[0]
    payload["submission_csv"] = str(csv_path)

    if shutil.which("mlebench") is None:
        payload.update(status="failed", error="mlebench_not_found_in_PATH")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    cmd = ["mlebench", "grade-sample", str(csv_path), competition_id]
    p = subprocess.run(cmd, capture_output=True, text=True)
    combined = (p.stdout or "") + "\n" + (p.stderr or "")

    report = _parse_report_json(combined)
    payload.update(
        status="completed" if p.returncode == 0 and report is not None else "failed",
        returncode=p.returncode,
        report=report,
        raw_output_tail=combined[-6000:],
    )

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path