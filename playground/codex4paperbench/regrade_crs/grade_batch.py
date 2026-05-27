#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

ROOT = Path('/data/yuzhu/Devs/EvoMaster-ours')
GRADE_SCRIPT = ROOT / 'playground/codex4paperbench/regrade_crs/grade_submission_responses.py'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--grade-run', type=Path, required=True)
    p.add_argument('--model', default='gpt-5.5')
    p.add_argument('--base-url', default='http://139.180.136.5:3000/openai')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY')
    p.add_argument('--paper-workers', type=int, default=10)
    p.add_argument('--leaf-concurrency', type=int, default=4)
    p.add_argument('--leaf-timeout', type=float, default=3600)
    p.add_argument('--retry-stop-after', type=float, default=1800)
    p.add_argument('--reasoning-effort', default='medium')
    p.add_argument('--reasoning-summary', default='')
    p.add_argument('--context-window', type=int, default=272000)
    p.add_argument('--max-output-tokens', type=int, default=4096)
    p.add_argument('--openai-timeout', type=float, default=240)
    return p.parse_args()


async def run_one(sem, args, row):
    paper = row['paper_id']
    sub = Path(row['submission'])
    out = args.grade_run / paper
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'grader_output.json'
    log = out / 'grade.log'
    if done.exists():
        return {'paper_id': paper, 'status': 'skipped', 'out': str(done)}
    cmd = [
        'conda', 'run', '-n', 'paperbench', 'python', str(GRADE_SCRIPT),
        '--submission', str(sub), '--paper-id', paper, '--out-dir', str(out),
        '--model', args.model, '--base-url', args.base_url, '--api-key-env', args.api_key_env,
        '--reasoning-effort', args.reasoning_effort,
        '--context-window', str(args.context_window),
        '--max-output-tokens', str(args.max_output_tokens),
        '--openai-timeout', str(args.openai_timeout),
        '--leaf-concurrency', str(args.leaf_concurrency),
        '--leaf-timeout', str(args.leaf_timeout),
        '--retry-stop-after', str(args.retry_stop_after),
    ]
    if args.reasoning_summary:
        cmd += ['--reasoning-summary', args.reasoning_summary]
    env = os.environ.copy()
    env['GPT_CHAT_MODEL'] = args.model
    env['GPT_BASE_URL'] = args.base_url
    env['OPENAI_BASE_URL'] = args.base_url
    started = time.time()
    async with sem:
        with log.open('w', encoding='utf-8') as f:
            f.write('$ ' + ' '.join(cmd) + '\n')
            f.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT), env=env, stdout=f, stderr=asyncio.subprocess.STDOUT
            )
            rc = await proc.wait()
    status = 'ok' if rc == 0 and done.exists() else 'failed'
    result = {
        'paper_id': paper,
        'status': status,
        'returncode': rc,
        'seconds': round(time.time() - started, 1),
        'out': str(done),
        'log': str(log),
    }
    (out / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


async def main():
    args = parse_args()
    rows = json.loads((args.grade_run / 'manifest.json').read_text())
    sem = asyncio.Semaphore(args.paper_workers)
    summary = args.grade_run / 'status.jsonl'
    tasks = [asyncio.create_task(run_one(sem, args, row)) for row in rows]
    with summary.open('a', encoding='utf-8') as sf:
        for fut in asyncio.as_completed(tasks):
            res = await fut
            print(json.dumps(res, ensure_ascii=False), flush=True)
            sf.write(json.dumps(res, ensure_ascii=False) + '\n')
            sf.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
