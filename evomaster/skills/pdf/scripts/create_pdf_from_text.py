#!/usr/bin/env python
"""
Write a piece of text to a PDF file.

Usage: provide --output_pdf and one of --text, --text_file, or --text_b64.
For long or multi-line content, use --text_file or --text_b64 to avoid shell
quoting/newline errors. Requires: pip install reportlab
"""

import argparse
import base64
import sys
from pathlib import Path


def _decode_b64(s: str) -> str:
    try:
        return base64.b64decode(s.encode("ascii")).decode("utf-8")
    except Exception as e:
        raise SystemExit(f"Invalid base64: {e}") from e


def _load_reportlab():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: reportlab. Install with: pip install reportlab"
        ) from exc
    return letter, SimpleDocTemplate, Paragraph, getSampleStyleSheet


def _get_text(value: str | None, file_path: str | None, b64_value: str | None) -> str:
    if b64_value is not None and b64_value.strip():
        return _decode_b64(b64_value.strip())
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        return path.read_text(encoding="utf-8")
    return value or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write text to a PDF file.")
    parser.add_argument("--output_pdf", required=True, help="Output PDF path.")
    parser.add_argument("--text", help="Text content (for short strings).")
    parser.add_argument("--text_file", help="Path to UTF-8 text file.")
    parser.add_argument(
        "--text_b64",
        help="Base64-encoded UTF-8 text (for long/multi-line to avoid shell issues).",
    )
    args = parser.parse_args(argv)

    text = _get_text(args.text, args.text_file, args.text_b64)
    if not text.strip():
        raise SystemExit("No text provided. Use --text, --text_file, or --text_b64.")

    letter, SimpleDocTemplate, Paragraph, getSampleStyleSheet = _load_reportlab()
    out_path = Path(args.output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    body = Paragraph(text.replace("\n", "<br/>"), styles["Normal"])
    doc.build([body])

    print(f"Saved PDF to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
