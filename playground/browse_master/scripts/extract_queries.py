#!/usr/bin/env python3

import argparse
import ast
import json
from pathlib import Path


DEFAULT_LOG_PATH = Path("runs/browse-dsv4pro_think/task_0000/logs/task_0.log")
QUERY_MARKER = "Info: {'queries'"


def extract_queries(log_path: Path) -> list[dict]:
    results = []

    with log_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if QUERY_MARKER not in raw_line:
                continue

            info_text = raw_line.split("Info: ", 1)[-1].strip()

            try:
                info = ast.literal_eval(info_text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Failed to parse line {line_no}: {raw_line.strip()}") from exc

            queries = info.get("queries")
            if not isinstance(queries, list):
                continue

            results.append(
                {
                    "line_no": line_no,
                    "queries": queries,
                }
            )

    return results


def format_as_text(extracted: list[dict]) -> str:
    lines = []

    for item in extracted:
        line_no = item["line_no"]
        for index, query in enumerate(item["queries"], start=1):
            lines.append(f"{line_no}\t{index}\t{query}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all query strings from log lines containing `Info: {'queries'`."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=DEFAULT_LOG_PATH,
        type=Path,
        help=f"Path to the log file. Defaults to {DEFAULT_LOG_PATH}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save extracted results as JSON. Uses `-o` if provided, otherwise writes next to the log file.",
    )
    args = parser.parse_args()

    extracted = extract_queries(args.log_path)
    if args.json:
        output_text = json.dumps(extracted, ensure_ascii=False, indent=2)
        output_path = args.output or args.log_path.with_name(f"{args.log_path.stem}_queries.json")
        output_path.write_text(output_text + ("\n" if output_text else ""), encoding="utf-8")
        print(f"Saved JSON to {output_path}")
        return

    output_text = format_as_text(extracted)

    if args.output:
        args.output.write_text(output_text + ("\n" if output_text else ""), encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
