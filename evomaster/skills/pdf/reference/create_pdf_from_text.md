## `scripts/create_pdf_from_text.py`: Write Text to PDF

Writes a piece of text to a single PDF file. The script is generic, does not assume any business schema, and only requires an output path and a text source.

### Feature Overview

- **Single purpose**: Given a piece of text, produces one PDF with line breaks preserved in the body.
- **Multiple text inputs**: Short text via CLI, UTF-8 file path, or base64-encoded string for long/multi-line content (avoids shell quoting and newline issues when invoked via skill `run_script`).
- **Dependency**: Uses `reportlab` only; the script prints a clear message to run `pip install reportlab` if it is missing.

### Usage Pattern (Conceptual)

The script is typically invoked by the host via a “skill script runner”, not by typing a full command in the terminal. This section describes the parameters the script expects and how to pass them.

The caller must provide:

- **Output path**: `--output_pdf` (required). Parent directories are created if they do not exist.
- **Text source (one of)**:
  - `--text`: Short string passed directly.
  - `--text_file`: Path to a UTF-8 text file.
  - `--text_b64`: Base64-encoded UTF-8 string; recommended for long or multi-line content to avoid syntax errors when `script_args` are executed by the shell.

When the upstream (e.g. agent) passes long content already encoded, use `--text_b64`; the script decodes it and writes it to the PDF.

### Key Parameters at a Glance

- **Output**
  - `--output_pdf`: Output PDF path (required).
- **Text input (one of)**
  - `--text`: Text passed directly; suitable for short content.
  - `--text_file`: Path to a UTF-8 text file.
  - `--text_b64`: Base64-encoded UTF-8 text; suitable for long/multi-line content to avoid shell quoting and newline issues. Priority: `--text_b64` > `--text_file` > `--text`.

### Design Notes

- No fixed business structure (no “Task Name / Data Knowledge” sections); the script simply writes the given text into the PDF.
- The script is self-contained; all generation logic lives in `scripts/create_pdf_from_text.py`. This reference only describes how to pass parameters and recommended usage in agent/run_script scenarios.
