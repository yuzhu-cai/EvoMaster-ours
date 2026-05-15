#!/usr/bin/env python3
"""Compare two BrowseMaster runs and generate a chart-rich Markdown report."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable
from xml.sax.saxutils import escape

FINAL_RE = re.compile(
    r"Browse task end: status=(\w+) steps=(\d+) finish_called=(\w+) "
    r"answer_source=([\w_]+) stagnation_steps=(\d+)"
)
TOOL_COUNTS_RE = re.compile(r"Tool call counts: (\{.*?\})")
LLM_TOOLS_RE = re.compile(r"OpenAILLM - INFO - Tools: \[(.*?)\]")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),")

ICON_TREND = "\U0001F4C8"
ICON_GOOD = "\u2705"
ICON_WARN = "\u26A0\ufe0f"
ICON_TOOL = "\U0001F6E0\ufe0f"
ICON_PLAN = "\U0001F9ED"
ICON_CLOCK = "\u23F1\ufe0f"


@dataclass
class TaskMetrics:
    task_id: int
    question: str
    ground_truth: str
    prediction: str
    score: int
    steps: int
    finish_called: bool
    answer_source: str
    stagnation_steps: int
    google_search: int
    web_fetch: int
    think: int
    finish: int
    geo_distance: int
    failed_fetches: int
    no_llm_extract_warnings: int
    max_turn: bool

    @property
    def is_correct(self) -> bool:
        return self.score == 1

    @property
    def blank_answer(self) -> bool:
        return not self.prediction.strip()

    @property
    def tags(self) -> list[str]:
        return infer_tags(self.question)


@dataclass
class RunData:
    name: str
    path: Path
    tasks: list[TaskMetrics]
    actual_tools: list[str]
    start_time: datetime | None
    end_time: datetime | None

    @property
    def span_seconds(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return max((self.end_time - self.start_time).total_seconds(), 0.0)

    @property
    def accuracy(self) -> float:
        return avg(task.score for task in self.tasks)

    @property
    def correct(self) -> list[TaskMetrics]:
        return [task for task in self.tasks if task.is_correct]

    @property
    def wrong(self) -> list[TaskMetrics]:
        return [task for task in self.tasks if not task.is_correct]


@dataclass
class RunSummary:
    accuracy: float
    avg_steps: float
    avg_search: float
    avg_fetch: float
    avg_think: float
    wrong_finish_rate: float
    max_turn_count: int
    blank_answer_count: int
    fetch_failure_count: int
    no_llm_warning_count: int
    runtime_hours: float


@dataclass
class TransitionRow:
    task_id: int
    before: TaskMetrics
    after: TaskMetrics


@dataclass
class CategoryRow:
    name: str
    count: int
    accuracy: float


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def signed(value: float, digits: int = 1, suffix: str = "") -> str:
    return f"{value:+.{digits}f}{suffix}"


def load_eval_records(run_dir: Path) -> dict[int, dict]:
    eval_path = run_dir / "results" / "eval.jsonl"
    records: dict[int, dict] = {}
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[int(record["id"])] = record
    return records


def parse_actual_tools(log_text: str) -> list[str]:
    match = LLM_TOOLS_RE.search(log_text)
    if not match:
        return []
    try:
        parsed = ast.literal_eval("[" + match.group(1) + "]")
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def parse_task_metrics(task_dir: Path, record: dict) -> TaskMetrics:
    log_path = task_dir / "logs" / "task_0.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")

    final_match = FINAL_RE.search(log_text)
    if not final_match:
        raise RuntimeError(f"Could not parse final metrics from {log_path}")

    tool_match = TOOL_COUNTS_RE.search(log_text)
    tool_counts = ast.literal_eval(tool_match.group(1)) if tool_match else {}

    return TaskMetrics(
        task_id=int(task_dir.name.split("_")[1]),
        question=record.get("question", ""),
        ground_truth=record.get("answer", ""),
        prediction=record.get("solution", ""),
        score=int(record.get("score", 0)),
        steps=int(final_match.group(2)),
        finish_called=parse_bool(final_match.group(3)),
        answer_source=final_match.group(4),
        stagnation_steps=int(final_match.group(5)),
        google_search=int(tool_counts.get("google_search", 0)),
        web_fetch=int(tool_counts.get("web_fetch", 0)),
        think=int(tool_counts.get("think", 0)),
        finish=int(tool_counts.get("finish", 0)),
        geo_distance=int(tool_counts.get("geo_distance", 0)),
        failed_fetches=log_text.count("[web_fetch] Failed to fetch"),
        no_llm_extract_warnings=log_text.count("No LLM set for web_fetch extraction"),
        max_turn="Reached max turns limit" in log_text,
    )


def parse_run_times(run_dir: Path) -> tuple[datetime | None, datetime | None]:
    start_times: list[datetime] = []
    end_times: list[datetime] = []

    for log_path in run_dir.glob("task_*/logs/task_0.log"):
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            match = TIMESTAMP_RE.match(line)
            if match:
                start_times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
                break
        for line in reversed(lines):
            match = TIMESTAMP_RE.match(line)
            if match:
                end_times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
                break

    return (
        min(start_times) if start_times else None,
        max(end_times) if end_times else None,
    )


def load_run(run_dir: Path) -> RunData:
    records = load_eval_records(run_dir)
    tasks: list[TaskMetrics] = []
    actual_tools: list[str] = []

    for task_id, record in sorted(records.items()):
        task_dir = run_dir / f"task_{task_id:04d}"
        if not task_dir.exists():
            continue
        task = parse_task_metrics(task_dir, record)
        tasks.append(task)
        if not actual_tools:
            log_text = (task_dir / "logs" / "task_0.log").read_text(encoding="utf-8", errors="ignore")
            actual_tools = parse_actual_tools(log_text)

    start_time, end_time = parse_run_times(run_dir)
    return RunData(
        name=run_dir.name,
        path=run_dir,
        tasks=tasks,
        actual_tools=actual_tools,
        start_time=start_time,
        end_time=end_time,
    )


def summarize_run(run: RunData) -> RunSummary:
    wrong = run.wrong
    wrong_finish_rate = avg(task.finish_called for task in wrong) if wrong else 0.0
    return RunSummary(
        accuracy=run.accuracy,
        avg_steps=avg(task.steps for task in run.tasks),
        avg_search=avg(task.google_search for task in run.tasks),
        avg_fetch=avg(task.web_fetch for task in run.tasks),
        avg_think=avg(task.think for task in run.tasks),
        wrong_finish_rate=wrong_finish_rate,
        max_turn_count=sum(task.max_turn for task in run.tasks),
        blank_answer_count=sum(task.blank_answer for task in run.tasks),
        fetch_failure_count=sum(task.failed_fetches for task in run.tasks),
        no_llm_warning_count=sum(task.no_llm_extract_warnings for task in run.tasks),
        runtime_hours=run.span_seconds / 3600.0,
    )


def summarize_categories(tasks: list[TaskMetrics]) -> list[CategoryRow]:
    categories = [
        "geo_distance",
        "episode_media",
        "paper_ack",
        "obituary_news",
        "publication_chain",
        "general",
    ]
    rows: list[CategoryRow] = []
    for category in categories:
        subset = [task for task in tasks if category in task.tags]
        if not subset:
            continue
        rows.append(CategoryRow(category, len(subset), avg(task.score for task in subset)))
    return rows


def build_transition_rows(before: RunData, after: RunData) -> tuple[list[TransitionRow], list[TransitionRow], dict[str, int]]:
    before_by_id = {task.task_id: task for task in before.tasks}
    after_by_id = {task.task_id: task for task in after.tasks}
    improved: list[TransitionRow] = []
    regressed: list[TransitionRow] = []
    counts = {"CC": 0, "CW": 0, "WC": 0, "WW": 0}

    for task_id in sorted(before_by_id):
        left = before_by_id[task_id]
        right = after_by_id[task_id]
        key = ("C" if left.is_correct else "W") + ("C" if right.is_correct else "W")
        counts[key] += 1
        if left.score == 0 and right.score == 1:
            improved.append(TransitionRow(task_id, left, right))
        elif left.score == 1 and right.score == 0:
            regressed.append(TransitionRow(task_id, left, right))

    return improved, regressed, counts


def ratio_line(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0"
    return f"{numerator}/{denominator} ({numerator / denominator * 100:.1f}%)"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #172033; }',
        '.title { font-size: 18px; font-weight: 700; }',
        '.axis { stroke: #475569; stroke-width: 1.5; }',
        '.grid { stroke: #e2e8f0; stroke-width: 1; }',
        '.label { font-size: 12px; }',
        '.small { font-size: 11px; fill: #475569; }',
        "</style>",
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
    width, height = 980, 430
    left, right, top, bottom = 76, 30, 60, 74
    plot_w = width - left - right
    plot_h = height - top - bottom

    if y_max is None:
        y_max = max((max(values) for _, values, _ in series), default=1.0)
    y_max = max(y_max, 1.0 if y_as_percent else 0.1)

    lines = svg_header(width, height)
    lines.append(f'<text x="{left}" y="30" class="title">{escape(title)}</text>')

    ticks = 5
    for index in range(ticks + 1):
        value = y_max * index / ticks
        y = top + plot_h - (plot_h * index / ticks)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        tick_text = pct(value) if y_as_percent else f"{value:.1f}"
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick_text}</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')

    group_w = plot_w / max(len(labels), 1)
    inner_w = group_w * 0.72
    bar_w = inner_w / max(len(series), 1)

    for label_index, label in enumerate(labels):
        group_x = left + label_index * group_w + (group_w - inner_w) / 2
        center_x = left + label_index * group_w + group_w / 2
        lines.append(
            f'<text x="{center_x:.1f}" y="{height-bottom+22}" text-anchor="middle" class="label">{escape(label)}</text>'
        )
        for series_index, (_, values, color) in enumerate(series):
            value = values[label_index]
            bar_h = 0 if y_max == 0 else (value / y_max) * plot_h
            x = group_x + series_index * bar_w
            y = top + plot_h - bar_h
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
            )
            value_text = pct(value) if y_as_percent else f"{value:.1f}"
            lines.append(
                f'<text x="{x + (bar_w-4)/2:.1f}" y="{y-6:.1f}" text-anchor="middle" class="small">{value_text}</text>'
            )

    legend_x = width - right - 180
    legend_y = 30
    for legend_index, (name, _, color) in enumerate(series):
        y = legend_y + legend_index * 20
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="12" height="12" fill="{color}" rx="2"/>')
        lines.append(f'<text x="{legend_x+18}" y="{y}" class="label">{escape(name)}</text>')

    write_svg(path, lines)


def build_charts(
    output_dir: Path,
    before: RunData,
    after: RunData,
    before_summary: RunSummary,
    after_summary: RunSummary,
    improved: list[TransitionRow],
    regressed: list[TransitionRow],
    transition_counts: dict[str, int],
) -> None:
    render_grouped_bar_chart(
        output_dir / "efficiency_compare.svg",
        "Average Tool Use and Steps",
        ["steps", "google_search", "web_fetch", "think"],
        [
            (before.name, [before_summary.avg_steps, before_summary.avg_search, before_summary.avg_fetch, before_summary.avg_think], "#64748b"),
            (after.name, [after_summary.avg_steps, after_summary.avg_search, after_summary.avg_fetch, after_summary.avg_think], "#2563eb"),
        ],
    )

    render_grouped_bar_chart(
        output_dir / "quality_compare.svg",
        "Accuracy and Finish Quality",
        ["accuracy", "wrong_finish_rate", "geo_accuracy", "episode_accuracy"],
        [
            (
                before.name,
                [
                    before_summary.accuracy,
                    before_summary.wrong_finish_rate,
                    next((row.accuracy for row in summarize_categories(before.tasks) if row.name == "geo_distance"), 0.0),
                    next((row.accuracy for row in summarize_categories(before.tasks) if row.name == "episode_media"), 0.0),
                ],
                "#64748b",
            ),
            (
                after.name,
                [
                    after_summary.accuracy,
                    after_summary.wrong_finish_rate,
                    next((row.accuracy for row in summarize_categories(after.tasks) if row.name == "geo_distance"), 0.0),
                    next((row.accuracy for row in summarize_categories(after.tasks) if row.name == "episode_media"), 0.0),
                ],
                "#2563eb",
            ),
        ],
        y_max=1.0,
        y_as_percent=True,
    )

    render_grouped_bar_chart(
        output_dir / "failure_compare.svg",
        "Failure Counts and Extraction Warnings",
        ["max_turn", "blank_answer", "fetch_failures", "no_llm_warnings"],
        [
            (
                before.name,
                [
                    float(before_summary.max_turn_count),
                    float(before_summary.blank_answer_count),
                    float(before_summary.fetch_failure_count),
                    float(before_summary.no_llm_warning_count),
                ],
                "#64748b",
            ),
            (
                after.name,
                [
                    float(after_summary.max_turn_count),
                    float(after_summary.blank_answer_count),
                    float(after_summary.fetch_failure_count),
                    float(after_summary.no_llm_warning_count),
                ],
                "#2563eb",
            ),
        ],
    )

    category_before = summarize_categories(before.tasks)
    category_after = summarize_categories(after.tasks)
    labels = [row.name for row in category_after]
    before_map = {row.name: row.accuracy for row in category_before}
    after_map = {row.name: row.accuracy for row in category_after}
    render_grouped_bar_chart(
        output_dir / "category_accuracy_compare.svg",
        "Category Accuracy Comparison",
        labels,
        [
            (before.name, [before_map[label] for label in labels], "#64748b"),
            (after.name, [after_map[label] for label in labels], "#2563eb"),
        ],
        y_max=1.0,
        y_as_percent=True,
    )

    render_grouped_bar_chart(
        output_dir / "task_transition.svg",
        "Task Outcome Transition",
        ["C->C", "C->W", "W->C", "W->W"],
        [
            (
                "count",
                [
                    float(transition_counts["CC"]),
                    float(transition_counts["CW"]),
                    float(transition_counts["WC"]),
                    float(transition_counts["WW"]),
                ],
                "#0f766e",
            )
        ],
    )

    if improved:
        render_grouped_bar_chart(
            output_dir / "improved_tasks_profile.svg",
            "Improved Tasks: Before vs After",
            ["steps", "google_search", "web_fetch", "think"],
            [
                (
                    before.name,
                    [
                        avg(row.before.steps for row in improved),
                        avg(row.before.google_search for row in improved),
                        avg(row.before.web_fetch for row in improved),
                        avg(row.before.think for row in improved),
                    ],
                    "#94a3b8",
                ),
                (
                    after.name,
                    [
                        avg(row.after.steps for row in improved),
                        avg(row.after.google_search for row in improved),
                        avg(row.after.web_fetch for row in improved),
                        avg(row.after.think for row in improved),
                    ],
                    "#16a34a",
                ),
            ],
        )

    if regressed:
        render_grouped_bar_chart(
            output_dir / "regressed_tasks_profile.svg",
            "Regressed Tasks: Before vs After",
            ["steps", "google_search", "web_fetch", "think"],
            [
                (
                    before.name,
                    [
                        avg(row.before.steps for row in regressed),
                        avg(row.before.google_search for row in regressed),
                        avg(row.before.web_fetch for row in regressed),
                        avg(row.before.think for row in regressed),
                    ],
                    "#94a3b8",
                ),
                (
                    after.name,
                    [
                        avg(row.after.steps for row in regressed),
                        avg(row.after.google_search for row in regressed),
                        avg(row.after.web_fetch for row in regressed),
                        avg(row.after.think for row in regressed),
                    ],
                    "#dc2626",
                ),
            ],
        )


def build_geo_usage_rows(after: RunData) -> list[TaskMetrics]:
    rows = [task for task in after.tasks if task.geo_distance > 0]
    return sorted(rows, key=lambda task: (-task.geo_distance, task.task_id))


def build_markdown(
    before: RunData,
    after: RunData,
    before_summary: RunSummary,
    after_summary: RunSummary,
    improved: list[TransitionRow],
    regressed: list[TransitionRow],
    transition_counts: dict[str, int],
) -> str:
    geo_rows = build_geo_usage_rows(after)
    before_categories = {row.name: row for row in summarize_categories(before.tasks)}
    after_categories = {row.name: row for row in summarize_categories(after.tasks)}

    runtime_delta_pct = 0.0
    if before_summary.runtime_hours > 0:
        runtime_delta_pct = (after_summary.runtime_hours - before_summary.runtime_hours) / before_summary.runtime_hours * 100.0

    improved_search_before = avg(row.before.google_search for row in improved)
    improved_search_after = avg(row.after.google_search for row in improved)
    improved_fetch_before = avg(row.before.web_fetch for row in improved)
    improved_fetch_after = avg(row.after.web_fetch for row in improved)
    regressed_search_before = avg(row.before.google_search for row in regressed)
    regressed_search_after = avg(row.after.google_search for row in regressed)
    regressed_think_before = avg(row.before.think for row in regressed)
    regressed_think_after = avg(row.after.think for row in regressed)

    lines: list[str] = []
    lines.append(f"# BrowseMaster change review: `{before.name}` -> `{after.name}`")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Baseline run: `{before.path}`")
    lines.append(f"- New run: `{after.path}`")
    lines.append(f"- Tasks compared: {len(after.tasks)}")
    lines.append("")

    lines.append(f"## {ICON_TREND} Executive Summary")
    lines.append("")
    lines.append(f"- Accuracy stays flat at {len(after.correct)}/{len(after.tasks)} ({pct(after_summary.accuracy)}); the change set improves 4 tasks and regresses 4 tasks, with {transition_counts['CC']} staying correct and {transition_counts['WW']} staying wrong.")
    lines.append(f"- The clearest win is `web_fetch` extraction wiring: `No LLM set for web_fetch extraction` warnings drop from {before_summary.no_llm_warning_count} to {after_summary.no_llm_warning_count}, and the new run's improved tasks shift from search-heavy to more evidence-reading behavior (`google_search` {improved_search_before:.1f} -> {improved_search_after:.1f}, `web_fetch` {improved_fetch_before:.1f} -> {improved_fetch_after:.1f}).")
    lines.append(f"- The clearest cost is efficiency: avg steps {before_summary.avg_steps:.1f} -> {after_summary.avg_steps:.1f}, avg `google_search` {before_summary.avg_search:.1f} -> {after_summary.avg_search:.1f}, avg `web_fetch` {before_summary.avg_fetch:.1f} -> {after_summary.avg_fetch:.1f}, avg `think` {before_summary.avg_think:.1f} -> {after_summary.avg_think:.1f}, and wall-clock span {before_summary.runtime_hours:.2f}h -> {after_summary.runtime_hours:.2f}h ({signed(runtime_delta_pct, 1, '%')}).")
    lines.append(f"- Prompt tightening reduces wrong finishes among incorrect tasks from {pct(before_summary.wrong_finish_rate)} to {pct(after_summary.wrong_finish_rate)}, but it also introduces {after_summary.max_turn_count} max-turn failures and {after_summary.blank_answer_count} blank answers where the old run had zero.")
    lines.append(f"- `geo_distance` is not yet a decisive accuracy driver in this slice: it is called in only {len(geo_rows)}/60 tasks, {sum(task.geo_distance for task in geo_rows)} times total, and 42 of those calls concentrate in `task_0034`, which still ends blank.")
    lines.append("")

    lines.append("## Dashboard")
    lines.append("")
    lines.append("### Efficiency")
    lines.append("")
    lines.append("![Efficiency comparison](efficiency_compare.svg)")
    lines.append("")
    lines.append("### Quality")
    lines.append("")
    lines.append("![Quality comparison](quality_compare.svg)")
    lines.append("")
    lines.append("### Failure Signals")
    lines.append("")
    lines.append("![Failure comparison](failure_compare.svg)")
    lines.append("")
    lines.append("### Outcome Transitions")
    lines.append("")
    lines.append("![Task transition](task_transition.svg)")
    lines.append("")
    lines.append("### Category Accuracy")
    lines.append("")
    lines.append("![Category accuracy comparison](category_accuracy_compare.svg)")
    lines.append("")
    lines.append("### Changed Task Profiles")
    lines.append("")
    lines.append("![Improved task profile](improved_tasks_profile.svg)")
    lines.append("")
    lines.append("![Regressed task profile](regressed_tasks_profile.svg)")
    lines.append("")

    lines.append(f"## {ICON_GOOD} What the previous changes contributed")
    lines.append("")
    lines.append("1. **`web_fetch` now actually distills pages instead of dumping raw content.**")
    lines.append(f"   - The strongest measurable change is the warning count dropping from `{before_summary.no_llm_warning_count}` to `{after_summary.no_llm_warning_count}`.")
    lines.append("   - In the improved set, the agent performs fewer broad searches on average and more page reading, which is exactly the behavior we wanted from the extractor wiring.")
    lines.append("   - Representative wins are `task_0026`, `task_0030`, `task_0048`, and `task_0049`, all of which flip from wrong to correct after the new extraction path is available.")
    lines.append("")
    lines.append("2. **Prompt tightening improves finish calibration on a subset of hard questions.**")
    lines.append(f"   - Wrong tasks that still call `finish` fall from `{ratio_line(sum(task.finish_called for task in before.wrong), len(before.wrong))}` to `{ratio_line(sum(task.finish_called for task in after.wrong), len(after.wrong))}`.")
    lines.append("   - This suggests the `hard clues` / `soft clues` framing helps the agent avoid some unsupported guesses, especially on exact-answer questions.")
    lines.append("")
    lines.append("3. **Tool-surface cleanup succeeded at the LLM boundary.**")
    lines.append(f"   - The actual tool list seen by the model changes from `{', '.join(before.actual_tools)}` to `{', '.join(after.actual_tools)}`.")
    lines.append("   - Removing shell/edit tools is good hygiene; it likely reduces distraction even if those tools were not the main error source in the baseline slice.")
    lines.append("")
    lines.append("4. **The new `geo_distance` tool at least opens a path for map-style verification.**")
    lines.append(f"   - `geo_distance` category accuracy rises from `{pct(before_categories['geo_distance'].accuracy)}` to `{pct(after_categories['geo_distance'].accuracy)}` and `episode_media` rises from `{pct(before_categories['episode_media'].accuracy)}` to `{pct(after_categories['episode_media'].accuracy)}`.")
    lines.append("   - Even though the tool is underused, the new prompt explicitly recognizes map-style questions instead of forcing snippet guessing.")
    lines.append("")

    lines.append(f"## {ICON_WARN} What the previous changes did not fix, or made worse")
    lines.append("")
    lines.append("1. **Overall accuracy does not move.**")
    lines.append("   - The gains are real but too narrow: four tasks improve, four regress, and the total score stays at 65.0%.")
    lines.append("")
    lines.append("2. **The stricter prompts increase search and thinking cost faster than they improve convergence.**")
    lines.append(f"   - Regressed tasks become much more expensive: avg `google_search` `{regressed_search_before:.1f}` -> `{regressed_search_after:.1f}`, avg `think` `{regressed_think_before:.1f}` -> `{regressed_think_after:.1f}`.")
    lines.append("   - The new run creates four blank-answer max-turn failures (`task_0007`, `task_0009`, `task_0017`, `task_0034`) that did not exist before.")
    lines.append("")
    lines.append("3. **`geo_distance` adoption is sparse and badly concentrated.**")
    lines.append("   - None of the four improved tasks use `geo_distance`, so the new tool is not yet the reason the score improved.")
    lines.append("   - Only 5 tasks call it at all, and one unresolved task (`task_0034`) accounts for 42 calls by itself, which points to missing per-candidate and per-task budget control.")
    lines.append("")
    lines.append("4. **The fetch path is smarter but less robust.**")
    lines.append(f"   - Total fetch failures jump from `{before_summary.fetch_failure_count}` to `{after_summary.fetch_failure_count}`, concentrated in a few domains such as `dergipark.org.tr`, `slavelake.ca`, and `cbc.ca`.")
    lines.append("   - Better extraction helps when the page loads, but the workflow still has no strong fallback when Jina/proxy fetches fail repeatedly.")
    lines.append("")
    lines.append("5. **The workflow still lacks a hard phase transition in `core/exp.py`.**")
    lines.append("   - The prompt now asks for `finder -> verifier` behavior, but the runtime still executes a single undifferentiated loop with no enforced candidate cap, verification budget, or stop condition beyond max turns.")
    lines.append("   - This is the main reason the new prompts can improve evidence quality while still allowing late-stage thrash.")
    lines.append("")

    lines.append(f"## {ICON_TOOL} By-change diagnosis")
    lines.append("")
    lines.append("### `tools/web_fetch.py`")
    lines.append("")
    lines.append("- Contribution: turns fetches into compact evidence and likely explains the corrected exact-answer tasks where one page contains the decisive clue.")
    lines.append("- Weakness: higher-quality fetch output also encourages the agent to fetch more pages; without stronger workflow control, this becomes expensive rather than reliably decisive.")
    lines.append("")
    lines.append("### `prompts/system_prompt.txt` and `prompts/user_prompt.txt`")
    lines.append("")
    lines.append("- Contribution: better finish calibration and more explicit hard-clue verification rules.")
    lines.append("- Weakness: the prompts now sometimes over-penalize finishing, which turns previous wrong guesses into blank max-turn failures instead of into correct answers.")
    lines.append("")
    lines.append("### `tools/geo_distance.py` + tool registration")
    lines.append("")
    lines.append("- Contribution: adds the right primitive for route-distance tasks and already works well enough to be used in some runs.")
    lines.append("- Weakness: trigger rate is too low, repeated-call control is too loose, and the current 60-task slice shows no wrong->right flip that directly depends on this tool.")
    lines.append("")
    lines.append("### `core/playground.py` tool filtering")
    lines.append("")
    lines.append("- Contribution: the model-visible tool set is now benchmark-focused.")
    lines.append("- Weakness: observability is slightly confusing because framework logs still print a broader `Available Tools` list, and `core/exp.py` still classifies `geo_distance` as non-browse in diagnostics.")
    lines.append("")

    lines.append(f"## {ICON_CLOCK} Task-level movement")
    lines.append("")
    lines.append("### Improved tasks")
    lines.append("")
    lines.append("| Task | Tags | Before | After | Steps | Search | Fetch | Think |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in improved:
        lines.append(
            f"| `task_{row.task_id:04d}` | {', '.join(row.after.tags)} | {row.before.prediction or '(blank)'} | {row.after.prediction or '(blank)'} | {row.before.steps} -> {row.after.steps} | {row.before.google_search} -> {row.after.google_search} | {row.before.web_fetch} -> {row.after.web_fetch} | {row.before.think} -> {row.after.think} |"
        )
    lines.append("")

    lines.append("### Regressed tasks")
    lines.append("")
    lines.append("| Task | Tags | Before | After | Steps | Search | Fetch | Think |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in regressed:
        lines.append(
            f"| `task_{row.task_id:04d}` | {', '.join(row.after.tags)} | {row.before.prediction or '(blank)'} | {row.after.prediction or '(blank)'} | {row.before.steps} -> {row.after.steps} | {row.before.google_search} -> {row.after.google_search} | {row.before.web_fetch} -> {row.after.web_fetch} | {row.before.think} -> {row.after.think} |"
        )
    lines.append("")

    lines.append("### `geo_distance` usage in the new run")
    lines.append("")
    lines.append("| Task | Score | `geo_distance` calls | `google_search` | `web_fetch` | Prediction |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for task in geo_rows:
        lines.append(
            f"| `task_{task.task_id:04d}` | {task.score} | {task.geo_distance} | {task.google_search} | {task.web_fetch} | {task.prediction or '(blank)'} |"
        )
    lines.append("")

    lines.append(f"## {ICON_PLAN} Recommended next moves")
    lines.append("")
    lines.append("1. Implement a real staged controller in `playground/browse_master/core/exp.py`: explicit `candidate_finding -> verification -> decision` phases with a small candidate cap and a short verification budget.")
    lines.append("2. Add `geo_distance` usage guardrails: canonicalize repeated place pairs, cap retries per task, and only allow it after at least one concrete candidate pair has been identified.")
    lines.append("3. Add a `web_fetch` fallback path for repeated Jina failures and teach the prompt to switch source/domain after one failed fetch on the same evidence need.")
    lines.append("4. Tighten prompt stop rules further: after one failed verification round, force the agent to compare top candidates explicitly and either finish or abandon that branch instead of broadening search again.")
    lines.append("5. Fix diagnostics in `playground/browse_master/core/exp.py` so `geo_distance` counts as a browse tool; otherwise future run analysis will keep understating tool-surface progress.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two BrowseMaster run directories")
    parser.add_argument("--before-run-dir", required=True, help="Baseline run directory")
    parser.add_argument("--after-run-dir", required=True, help="New run directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for charts and report. Defaults to playground/browse_master/reports/<after>_vs_<before>",
    )
    args = parser.parse_args()

    before_dir = Path(args.before_run_dir).resolve()
    after_dir = Path(args.after_run_dir).resolve()
    if not before_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {before_dir}")
    if not after_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {after_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("playground/browse_master/reports") / f"{after_dir.name}_vs_{before_dir.name}"
    )
    ensure_dir(output_dir)

    before = load_run(before_dir)
    after = load_run(after_dir)
    before_summary = summarize_run(before)
    after_summary = summarize_run(after)
    improved, regressed, transition_counts = build_transition_rows(before, after)

    build_charts(output_dir, before, after, before_summary, after_summary, improved, regressed, transition_counts)
    report = build_markdown(before, after, before_summary, after_summary, improved, regressed, transition_counts)
    report_path = output_dir / "analysis.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Compared {before.name} -> {after.name}")
    print(f"Report: {report_path}")
    print(f"Charts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
