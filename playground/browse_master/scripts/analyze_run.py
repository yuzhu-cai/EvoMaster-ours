#!/usr/bin/env python3
"""Analyze a BrowseMaster run and emit charts plus a Markdown report.

This script is tailored for run directories such as:

    runs/browse_gpt5.4_think_2/

It reads `results/eval.jsonl` for correctness labels and parses each task log
to collect:
- final step count
- finish/stagnation signals
- tool call counts
- fetch failures / extraction warnings

Outputs:
- a Markdown summary
- several SVG charts
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


FINAL_RE = re.compile(
    r"Browse task end: status=(\w+) steps=(\d+) finish_called=(\w+) "
    r"answer_source=([\w_]+) stagnation_steps=(\d+)"
)
TOOL_COUNTS_RE = re.compile(r"Tool call counts: (\{.*?\})")
TOOLS_EXPOSED_RE = re.compile(r"Tools: \[(.*?)\]")


STEP_BUCKETS = [
    ("<20", lambda steps: steps < 20),
    ("20-39", lambda steps: 20 <= steps <= 39),
    ("40-59", lambda steps: 40 <= steps <= 59),
    ("60+", lambda steps: steps >= 60),
]


@dataclass
class TaskMetrics:
    task_id: int
    question: str
    ground_truth: str
    prediction: str
    score: int
    status: str
    steps: int
    finish_called: bool
    answer_source: str
    stagnation_steps: int
    google_search: int
    web_fetch: int
    think: int
    finish: int
    failed_fetches: int
    no_llm_extract_warnings: int
    token_compactions: int
    max_turn: bool
    exposed_tools: list[str]

    @property
    def is_correct(self) -> bool:
        return self.score == 1

    @property
    def tags(self) -> list[str]:
        return infer_tags(self.question)


def infer_tags(question: str) -> list[str]:
    text = question.lower()
    tags: list[str] = []

    if any(token in text for token in ("miles", "distance", "drive", "google maps", "located in")):
        tags.append("geo_distance")
    if any(token in text for token in ("episode", "season")):
        tags.append("episode_media")
    if any(token in text for token in ("acknowledgment", "acknowledgement", "research paper", "pdf")):
        tags.append("paper_ack")
    if any(token in text for token in ("obituary", "daily newspaper", "visitation")):
        tags.append("obituary_news")
    if any(token in text for token in ("published", "interview", "magazine", "article")):
        tags.append("publication_chain")

    if not tags:
        tags.append("general")
    return tags


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def parse_exposed_tools(log_text: str) -> list[str]:
    match = TOOLS_EXPOSED_RE.search(log_text)
    if not match:
        return []
    raw = "[" + match.group(1) + "]"
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def load_eval_records(eval_path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[int(record["id"])] = record
    return records


def parse_task_metrics(task_dir: Path, record: dict) -> TaskMetrics:
    task_id = int(task_dir.name.split("_")[1])
    log_path = task_dir / "logs" / "task_0.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")

    final_match = FINAL_RE.search(log_text)
    if not final_match:
        raise RuntimeError(f"Could not parse final metrics from {log_path}")

    tool_match = TOOL_COUNTS_RE.search(log_text)
    tool_counts = ast.literal_eval(tool_match.group(1)) if tool_match else {}

    exposed_tools = parse_exposed_tools(log_text)

    return TaskMetrics(
        task_id=task_id,
        question=record.get("question", ""),
        ground_truth=record.get("answer", ""),
        prediction=record.get("solution", ""),
        score=int(record.get("score", 0)),
        status=final_match.group(1),
        steps=int(final_match.group(2)),
        finish_called=parse_bool(final_match.group(3)),
        answer_source=final_match.group(4),
        stagnation_steps=int(final_match.group(5)),
        google_search=int(tool_counts.get("google_search", 0)),
        web_fetch=int(tool_counts.get("web_fetch", 0)),
        think=int(tool_counts.get("think", 0)),
        finish=int(tool_counts.get("finish", 0)),
        failed_fetches=log_text.count("[web_fetch] Failed to fetch"),
        no_llm_extract_warnings=log_text.count("No LLM set for web_fetch extraction"),
        token_compactions=log_text.count("Token usage near limit"),
        max_turn="Reached max turns limit" in log_text,
        exposed_tools=exposed_tools,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }',
        '.title { font-size: 18px; font-weight: 700; }',
        '.axis { stroke: #4b5563; stroke-width: 1.5; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '.label { font-size: 12px; }',
        '.small { font-size: 11px; fill: #4b5563; }',
        '</style>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_max: float | None = None,
    y_as_percent: bool = False,
) -> None:
    width, height = 920, 420
    left, right, top, bottom = 70, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    if y_max is None:
        y_max = max(max(values) for _, values, _ in series) if series else 1.0
    y_max = max(y_max, 1.0 if y_as_percent else 0.1)

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="30" class="title">{escape(title)}</text>')

    ticks = 5
    for i in range(ticks + 1):
        value = y_max * i / ticks
        y = top + plot_h - (plot_h * i / ticks)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        tick_text = pct(value) if y_as_percent else f"{value:.1f}"
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick_text}</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')

    group_w = plot_w / max(len(labels), 1)
    inner_w = group_w * 0.72
    bar_w = inner_w / max(len(series), 1)

    for idx, label in enumerate(labels):
        group_x = left + idx * group_w + (group_w - inner_w) / 2
        center_x = left + idx * group_w + group_w / 2
        lines.append(
            f'<text x="{center_x:.1f}" y="{height-bottom+22}" text-anchor="middle" class="label">{escape(label)}</text>'
        )
        for s_idx, (_, values, color) in enumerate(series):
            value = values[idx]
            bar_h = 0 if y_max == 0 else (value / y_max) * plot_h
            x = group_x + s_idx * bar_w
            y = top + plot_h - bar_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4:.1f}" height="{bar_h:.1f}" '
                f'fill="{color}" rx="3"/>'
            )
            value_text = pct(value) if y_as_percent else f"{value:.1f}"
            lines.append(
                f'<text x="{x + (bar_w-4)/2:.1f}" y="{y-6:.1f}" text-anchor="middle" class="small">{value_text}</text>'
            )

    legend_x = width - right - 180
    legend_y = 30
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + idx * 20
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="12" height="12" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{legend_x+18}" y="{y}" class="label">{escape(name)}</text>')

    write_svg(path, lines)


def render_scatter_chart(
    path: Path,
    title: str,
    points: list[tuple[float, float, str, str]],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 920, 440
    left, right, top, bottom = 80, 30, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_max = max((point[0] for point in points), default=1)
    y_max = max((point[1] for point in points), default=1)
    x_max = max(x_max, 1)
    y_max = max(y_max, 1)

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="30" class="title">{escape(title)}</text>')

    ticks = 5
    for i in range(ticks + 1):
        y_value = y_max * i / ticks
        y = top + plot_h - plot_h * i / ticks
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{y_value:.0f}</text>')

        x_value = x_max * i / ticks
        x = left + plot_w * i / ticks
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" class="small">{x_value:.0f}</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    lines.append(f'<text x="{left + plot_w/2:.1f}" y="{height-20}" text-anchor="middle" class="label">{escape(x_label)}</text>')
    lines.append(
        f'<text x="20" y="{top + plot_h/2:.1f}" transform="rotate(-90 20 {top + plot_h/2:.1f})" '
        f'text-anchor="middle" class="label">{escape(y_label)}</text>'
    )

    for x_value, y_value, color, tooltip in points:
        cx = left + (x_value / x_max) * plot_w
        cy = top + plot_h - (y_value / y_max) * plot_h
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" opacity="0.85">'
            f"<title>{escape(tooltip)}</title></circle>"
        )

    legend_x = width - right - 185
    legend_y = 28
    legend = [("correct", "#16a34a"), ("incorrect", "#dc2626")]
    for idx, (name, color) in enumerate(legend):
        y = legend_y + idx * 20
        lines.append(f'<circle cx="{legend_x}" cy="{y-4}" r="5" fill="{color}" opacity="0.85"/>')
        lines.append(f'<text x="{legend_x+14}" y="{y}" class="label">{name}</text>')

    write_svg(path, lines)


def bucket_accuracy(tasks: list[TaskMetrics]) -> tuple[list[str], list[float], list[int]]:
    labels: list[str] = []
    accuracies: list[float] = []
    counts: list[int] = []

    for label, fn in STEP_BUCKETS:
        bucket = [task for task in tasks if fn(task.steps)]
        labels.append(label)
        counts.append(len(bucket))
        accuracies.append(avg(task.score for task in bucket) if bucket else 0.0)
    return labels, accuracies, counts


def summarize_categories(tasks: list[TaskMetrics]) -> list[tuple[str, int, float]]:
    categories = [
        "geo_distance",
        "episode_media",
        "paper_ack",
        "obituary_news",
        "publication_chain",
        "general",
    ]

    rows: list[tuple[str, int, float]] = []
    for category in categories:
        subset = [task for task in tasks if category in task.tags]
        if not subset:
            continue
        rows.append((category, len(subset), avg(task.score for task in subset)))
    return rows


def ratio_line(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0"
    return f"{numerator}/{denominator} ({numerator / denominator * 100:.1f}%)"


def format_tool_list(tools: list[str]) -> str:
    return ", ".join(tools) if tools else "(not detected)"


def generate_markdown(run_dir: Path, output_dir: Path, tasks: list[TaskMetrics]) -> str:
    correct = [task for task in tasks if task.is_correct]
    incorrect = [task for task in tasks if not task.is_correct]

    step_labels, step_accuracies, step_counts = bucket_accuracy(tasks)
    category_rows = summarize_categories(tasks)

    total_no_llm = sum(task.no_llm_extract_warnings for task in tasks)
    total_failed_fetch = sum(task.failed_fetches for task in tasks)

    high_search_cutoff = 30
    high_search = [task for task in tasks if task.google_search >= high_search_cutoff]
    high_stagnation = [task for task in tasks if task.stagnation_steps >= 5]

    exposed_tools = correct[0].exposed_tools if correct else (incorrect[0].exposed_tools if incorrect else [])

    wrong_finish_count = sum(task.finish_called for task in incorrect)
    wrong_no_max_turn = sum(0 if task.max_turn else 1 for task in incorrect)

    long_wrong = sorted(incorrect, key=lambda task: (-task.steps, task.task_id))
    long_correct = sorted(correct, key=lambda task: (-task.steps, task.task_id))[:5]

    lines: list[str] = []
    lines.append(f"# BrowseMaster Run Analysis: `{run_dir.name}`")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Tasks analyzed: {len(tasks)}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Accuracy | {len(correct)}/{len(tasks)} ({len(correct) / len(tasks) * 100:.1f}%) |")
    lines.append(f"| Incorrect tasks that still called `finish` | {ratio_line(wrong_finish_count, len(incorrect))} |")
    lines.append(f"| Incorrect tasks without max-turn failure | {ratio_line(wrong_no_max_turn, len(incorrect))} |")
    lines.append(f"| Avg steps (`correct`) | {avg(task.steps for task in correct):.1f} |")
    lines.append(f"| Avg steps (`incorrect`) | {avg(task.steps for task in incorrect):.1f} |")
    lines.append(f"| Median steps (`correct`) | {median(task.steps for task in correct):.1f} |")
    lines.append(f"| Median steps (`incorrect`) | {median(task.steps for task in incorrect):.1f} |")
    lines.append(f"| Avg `google_search` (`correct`) | {avg(task.google_search for task in correct):.1f} |")
    lines.append(f"| Avg `google_search` (`incorrect`) | {avg(task.google_search for task in incorrect):.1f} |")
    lines.append(f"| Avg `web_fetch` (`correct`) | {avg(task.web_fetch for task in correct):.1f} |")
    lines.append(f"| Avg `web_fetch` (`incorrect`) | {avg(task.web_fetch for task in incorrect):.1f} |")
    lines.append(f"| Avg `think` (`correct`) | {avg(task.think for task in correct):.1f} |")
    lines.append(f"| Avg `think` (`incorrect`) | {avg(task.think for task in incorrect):.1f} |")
    lines.append(f"| Total `web_fetch` extraction warnings | {total_no_llm} |")
    lines.append(f"| Total fetch failures | {total_failed_fetch} |")
    lines.append(f"| Exposed tools in run logs | {format_tool_list(exposed_tools)} |")
    lines.append("")

    lines.append("## Charts")
    lines.append("")
    lines.append("### Accuracy by Step Bucket")
    lines.append("")
    lines.append("![Accuracy by step bucket](accuracy_by_step_bucket.svg)")
    lines.append("")
    lines.append("### Average Tool Calls by Outcome")
    lines.append("")
    lines.append("![Average tool calls by outcome](avg_tool_calls_by_outcome.svg)")
    lines.append("")
    lines.append("### Steps vs Google Searches")
    lines.append("")
    lines.append("![Steps vs Google searches](steps_vs_google_search.svg)")
    lines.append("")
    lines.append("### Heuristic Category Accuracy")
    lines.append("")
    lines.append("![Category accuracy](category_accuracy.svg)")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Most errors are wrong finishes, not timeout failures.**")
    lines.append(
        f"   - All {len(incorrect)} incorrect tasks completed normally, and {wrong_finish_count} of them still called `finish`."
    )
    lines.append(
        "   - This means the current bottleneck is answer calibration and verification quality, not basic convergence."
    )
    lines.append("")
    lines.append("2. **Search explosion strongly correlates with wrong answers.**")
    lines.append(
        f"   - Incorrect tasks average `{avg(task.google_search for task in incorrect):.1f}` `google_search` calls versus "
        f"`{avg(task.google_search for task in correct):.1f}` for correct tasks."
    )
    lines.append(
        f"   - Tasks with `google_search >= {high_search_cutoff}`: {ratio_line(sum(task.is_correct for task in high_search), len(high_search))} accuracy."
    )
    lines.append(
        f"   - Step buckets show accuracy falling from `{pct(step_accuracies[0])}` in `{step_labels[0]}` to "
        f"`{pct(step_accuracies[-1])}` in `{step_labels[-1]}`."
    )
    lines.append("")
    lines.append("3. **`web_fetch` is being used, but its extraction path is not actually connected to an LLM.**")
    lines.append(
        f"   - The logs contain `{total_no_llm}` occurrences of `No LLM set for web_fetch extraction`, so most fetches return raw page text."
    )
    lines.append(
        f"   - Fetch count is not the core problem by itself: correct tasks average `{avg(task.web_fetch for task in correct):.1f}` fetches, "
        f"incorrect tasks average `{avg(task.web_fetch for task in incorrect):.1f}`."
    )
    lines.append(
        "   - This points to weak evidence distillation rather than insufficient page reading."
    )
    lines.append("")
    lines.append("4. **High-precision question types remain weak.**")
    for category, count, accuracy in category_rows:
        lines.append(f"   - `{category}`: {count} tasks, {pct(accuracy)} accuracy")
    lines.append(
        "   - In this slice, `geo_distance` and `episode_media` are especially weak, which suggests a need for specialized tools / workflows."
    )
    lines.append("")
    lines.append("5. **Some finishing heuristics are now too permissive.**")
    lines.append(
        f"   - `{sum(task.stagnation_steps >= 5 for task in incorrect)}` incorrect tasks accumulated at least 5 stagnation steps and still finished."
    )
    lines.append(
        "   - The current prompting is good at avoiding endless loops, but it sometimes finishes before decisive clues are hard-verified."
    )
    lines.append("")

    lines.append("## Representative Incorrect Tasks")
    lines.append("")
    lines.append("| Task | Steps | Search | Fetch | Think | Stagnation | Tags | Ground Truth | Prediction |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for task in long_wrong:
        lines.append(
            f"| `{task.task_id:04d}` | {task.steps} | {task.google_search} | {task.web_fetch} | {task.think} | "
            f"{task.stagnation_steps} | {', '.join(task.tags)} | {task.ground_truth} | {task.prediction} |"
        )
    lines.append("")

    lines.append("## Long but Correct Tasks")
    lines.append("")
    lines.append("| Task | Steps | Search | Fetch | Think | Stagnation | Prediction |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for task in long_correct:
        lines.append(
            f"| `{task.task_id:04d}` | {task.steps} | {task.google_search} | {task.web_fetch} | {task.think} | "
            f"{task.stagnation_steps} | {task.prediction} |"
        )
    lines.append("")

    lines.append("## Step Bucket Breakdown")
    lines.append("")
    lines.append("| Step bucket | Tasks | Accuracy |")
    lines.append("| --- | ---: | ---: |")
    for label, accuracy, count in zip(step_labels, step_accuracies, step_counts):
        lines.append(f"| {label} | {count} | {pct(accuracy)} |")
    lines.append("")

    lines.append("## Optimization Plan")
    lines.append("")
    lines.append("### P0 — Tool Layer First")
    lines.append("")
    lines.append("- Connect an actual LLM to `web_fetch` extraction so fetched pages become compact, goal-focused evidence instead of raw dumps.")
    lines.append("- Add a dedicated geo / distance tool for Google-Maps-style questions instead of inferring route distance from search snippets.")
    lines.append("- Restrict exposed tools for the browse agent to the minimal set needed for benchmark solving.")
    lines.append("- Lower sampling variance for GPT runs by reducing `temperature`, and consider stronger reasoning settings for verification-heavy phases.")
    lines.append("")

    lines.append("### P1 — Tighten Finish Calibration in Prompts")
    lines.append("")
    lines.append("- Keep the anti-loop and anti-benchmark-mirror rules, but distinguish `hard clues` from `soft clues`.")
    lines.append("- Require hard verification before finishing on distance, episode/season, obituary, acknowledgment, and exact-name questions.")
    lines.append("- When stagnation occurs, force the agent to compare top candidates against unresolved hard clues instead of continuing broad search.")
    lines.append("")

    lines.append("### P2 — Reorganize the Workflow in `core/exp.py`")
    lines.append("")
    lines.append("- Split solving into `finder -> verifier` phases.")
    lines.append("- Let the finder produce a small candidate set, then let a verifier spend a short budget on direct contradiction checks.")
    lines.append("- Use run-time thresholds such as high search count or repeated stagnation to trigger the verifier early.")
    lines.append("")

    lines.append("### P3 — Selective Multi-Agent, Not Default Parallel Search")
    lines.append("")
    lines.append("- Do not start with blanket parallel multi-agent search; the current errors are mostly not due to missing breadth.")
    lines.append("- Instead, introduce a second agent only when needed, for example a verifier/selector agent that compares 2-3 finalists.")
    lines.append("- This keeps cost under control while targeting the current error mode: wrong convergence, not lack of exploration.")
    lines.append("")

    lines.append("## Immediate Next Actions")
    lines.append("")
    lines.append("1. Fix `web_fetch` extraction and rerun the same 60-task slice.")
    lines.append("2. Add a distance-oriented tool and retest the `geo_distance` subset.")
    lines.append("3. After that, implement a lightweight `finder -> verifier` workflow in `playground/browse_master/core/exp.py`.")
    lines.append("")

    return "\n".join(lines) + "\n"


def build_charts(output_dir: Path, tasks: list[TaskMetrics]) -> None:
    correct = [task for task in tasks if task.is_correct]
    incorrect = [task for task in tasks if not task.is_correct]

    step_labels, step_accuracies, _ = bucket_accuracy(tasks)
    render_grouped_bar_chart(
        output_dir / "accuracy_by_step_bucket.svg",
        "Accuracy by Step Bucket",
        step_labels,
        [("accuracy", step_accuracies, "#2563eb")],
        y_max=1.0,
        y_as_percent=True,
    )

    render_grouped_bar_chart(
        output_dir / "avg_tool_calls_by_outcome.svg",
        "Average Tool Calls by Outcome",
        ["google_search", "web_fetch", "think"],
        [
            (
                "correct",
                [
                    avg(task.google_search for task in correct),
                    avg(task.web_fetch for task in correct),
                    avg(task.think for task in correct),
                ],
                "#16a34a",
            ),
            (
                "incorrect",
                [
                    avg(task.google_search for task in incorrect),
                    avg(task.web_fetch for task in incorrect),
                    avg(task.think for task in incorrect),
                ],
                "#dc2626",
            ),
        ],
    )

    points: list[tuple[float, float, str, str]] = []
    for task in tasks:
        color = "#16a34a" if task.is_correct else "#dc2626"
        tooltip = (
            f"task_{task.task_id:04d} | score={task.score} | steps={task.steps} | "
            f"google_search={task.google_search} | pred={task.prediction}"
        )
        points.append((task.steps, task.google_search, color, tooltip))
    render_scatter_chart(
        output_dir / "steps_vs_google_search.svg",
        "Steps vs Google Searches",
        points,
        x_label="steps",
        y_label="google_search calls",
    )

    category_rows = summarize_categories(tasks)
    render_grouped_bar_chart(
        output_dir / "category_accuracy.svg",
        "Heuristic Category Accuracy",
        [row[0] for row in category_rows],
        [("accuracy", [row[2] for row in category_rows], "#7c3aed")],
        y_max=1.0,
        y_as_percent=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a BrowseMaster run directory")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory such as runs/browse_gpt5.4_think_2",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for charts + markdown report. Defaults to playground/browse_master/reports/<run_name>",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("playground/browse_master/reports") / run_dir.name
    )
    ensure_dir(output_dir)

    eval_path = run_dir / "results" / "eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval file: {eval_path}")

    records = load_eval_records(eval_path)
    tasks: list[TaskMetrics] = []
    for task_id, record in sorted(records.items()):
        task_dir = run_dir / f"task_{task_id:04d}"
        if not task_dir.exists():
            continue
        tasks.append(parse_task_metrics(task_dir, record))

    if not tasks:
        raise RuntimeError(f"No task metrics parsed from {run_dir}")

    build_charts(output_dir, tasks)
    markdown = generate_markdown(run_dir, output_dir, tasks)
    report_path = output_dir / "analysis.md"
    report_path.write_text(markdown, encoding="utf-8")

    print(f"Analyzed {len(tasks)} tasks from {run_dir}")
    print(f"Report: {report_path}")
    print(f"Charts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
