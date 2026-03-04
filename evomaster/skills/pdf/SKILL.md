---
name: pdf
description: Comprehensive PDF manipulation toolkit for extracting text and tables, creating new PDFs, merging/splitting documents, and handling forms. When Claude needs to fill in a PDF form or programmatically process, generate, or analyze PDF documents at scale.
license: Proprietary. LICENSE.txt has complete terms
---

# PDF Skill Overview

This skill provides building blocks for **programmatic PDF processing at scale**, including:

- Creating new PDFs from text (via `reportlab`);
- Filling existing PDF forms and annotations;
- Extracting text, tables, images, and metadata from PDFs;
- Merging/splitting/rotating PDFs using Python libraries and command-line tools.

> Detailed, script-level explanations are split into standalone documents under the `reference/` directory (and a couple of legacy root-level docs). This file focuses on overall structure and navigation.

## Directory Structure

```text
pdf/
├── SKILL.md                      # This file: overview and navigation
├── scripts/                      # Executable scripts (core logic)
│   ├── create_pdf_from_text.py   # Write text to a PDF file
│   ├── convert_pdf_to_images.py
│   ├── create_validation_image.py
│   ├── extract_form_field_info.py
│   ├── fill_fillable_fields.py
│   ├── fill_pdf_form_with_annotations.py
│   ├── check_fillable_fields.py
│   ├── check_bounding_boxes.py
│   └── check_bounding_boxes_test.py
├── reference/                    # Script-level docs for this skill
│   └── create_pdf_from_text.md   # Usage and examples for create_pdf_from_text.py
├── reference.md                  # Advanced PDF processing reference (legacy, kept for compatibility)
├── forms.md                      # PDF form-filling guide (legacy, kept for compatibility)
└── LICENSE.txt
```

## When to Use This Skill

Use the `pdf` skill when you need any of the following:

- **Generate PDF documents** from text (e.g., write a piece of text into a PDF file);
- **Fill PDF forms** (either via fillable fields or annotations);
- **Extract text, tables, or images** from existing PDFs with Python tools or command-line utilities;
- **Manipulate PDFs structurally** (merge, split, rotate, crop, encrypt/decrypt).

If your task only needs to read or write plain text/CSV without any PDF artifacts, you do not need this skill.

## Reference Documentation Navigation (`reference/`)

To keep `SKILL.md` concise, script-specific details live in separate reference documents. Load only what you need for the current task:

- `reference/create_pdf_from_text.md`  
  - **Scope**: Documentation for `scripts/create_pdf_from_text.py`.  
  - **Content**: Write text to a PDF; parameters `--output_pdf`, `--text` / `--text_file` / `--text_b64`. Use `--text_b64` for long content to avoid shell errors.

- `reference.md`  
  - **Scope**: Advanced PDF processing reference.  
  - **Content**: Deeper coverage of `pypdf`, `pdfplumber`, `pypdfium2`, JavaScript libraries (`pdf-lib`, `pdfjs-dist`), and advanced command-line tools (`pdftotext`, `qpdf`, `pdfimages`).

- `forms.md`  
  - **Scope**: PDF form-filling workflows.  
  - **Content**: How to inspect form fields, validate bounding boxes, and fill forms using the scripts under `scripts/` (e.g., `fill_pdf_form_with_annotations.py`, `fill_fillable_fields.py`).

When using this skill, **avoid pulling all of these documents at once**; select only the reference files that are directly relevant to the current operation (e.g. `reference/create_pdf_from_text.md` when you only need to write text to a PDF).

---

## Quick Start

```python
from pypdf import PdfReader, PdfWriter

# Read a PDF
reader = PdfReader("document.pdf")
print(f"Pages: {len(reader.pages)}")

# Extract text
text = ""
for page in reader.pages:
    text += page.extract_text()
```

---

## Quick Library Guide (Python)

The following sections provide a compact reminder of common PDF operations using Python libraries. For more advanced or specialized usage, see `reference.md`.

### pypdf - Basic Operations

#### Merge PDFs
```python
from pypdf import PdfWriter, PdfReader

writer = PdfWriter()
for pdf_file in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        writer.add_page(page)

with open("merged.pdf", "wb") as output:
    writer.write(output)
```

#### Split PDF
```python
reader = PdfReader("input.pdf")
for i, page in enumerate(reader.pages):
    writer = PdfWriter()
    writer.add_page(page)
    with open(f"page_{i+1}.pdf", "wb") as output:
        writer.write(output)
```

#### Extract Metadata
```python
reader = PdfReader("document.pdf")
meta = reader.metadata
print(f"Title: {meta.title}")
print(f"Author: {meta.author}")
print(f"Subject: {meta.subject}")
print(f"Creator: {meta.creator}")
```

#### Rotate Pages
```python
reader = PdfReader("input.pdf")
writer = PdfWriter()

page = reader.pages[0]
page.rotate(90)  # Rotate 90 degrees clockwise
writer.add_page(page)

with open("rotated.pdf", "wb") as output:
    writer.write(output)
```

### pdfplumber - Text and Table Extraction

#### Extract Text with Layout
```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        print(text)
```

#### Extract Tables
```python
with pdfplumber.open("document.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        for j, table in enumerate(tables):
            print(f"Table {j+1} on page {i+1}:")
            for row in table:
                print(row)
```

#### Advanced Table Extraction
```python
import pandas as pd

with pdfplumber.open("document.pdf") as pdf:
    all_tables = []
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            if table:  # Check if table is not empty
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables.append(df)

# Combine all tables
if all_tables:
    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_df.to_excel("extracted_tables.xlsx", index=False)
```

### reportlab - Create PDFs

#### Basic PDF Creation
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

c = canvas.Canvas("hello.pdf", pagesize=letter)
width, height = letter

# Add text
c.drawString(100, height - 100, "Hello World!")
c.drawString(100, height - 120, "This is a PDF created with reportlab")

# Add a line
c.line(100, height - 140, 400, height - 140)

# Save
c.save()
```

#### Create PDF with Multiple Pages
```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

# Add content
title = Paragraph("Report Title", styles['Title'])
story.append(title)
story.append(Spacer(1, 12))

body = Paragraph("This is the body of the report. " * 20, styles['Normal'])
story.append(body)
story.append(PageBreak())

# Page 2
story.append(Paragraph("Page 2", styles['Heading1']))
story.append(Paragraph("Content for page 2", styles['Normal']))

# Build PDF
doc.build(story)
```

## Command-Line Tools

### pdftotext (poppler-utils)
```bash
# Extract text
pdftotext input.pdf output.txt

# Extract text preserving layout
pdftotext -layout input.pdf output.txt

# Extract specific pages
pdftotext -f 1 -l 5 input.pdf output.txt  # Pages 1-5
```

### qpdf
```bash
# Merge PDFs
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf

# Split pages
qpdf input.pdf --pages . 1-5 -- pages1-5.pdf
qpdf input.pdf --pages . 6-10 -- pages6-10.pdf

# Rotate pages
qpdf input.pdf output.pdf --rotate=+90:1  # Rotate page 1 by 90 degrees

# Remove password
qpdf --password=mypassword --decrypt encrypted.pdf decrypted.pdf
```

### pdftk (if available)
```bash
# Merge
pdftk file1.pdf file2.pdf cat output merged.pdf

# Split
pdftk input.pdf burst

# Rotate
pdftk input.pdf rotate 1east output rotated.pdf
```

## Common Tasks

### Extract Text from Scanned PDFs
```python
# Requires: pip install pytesseract pdf2image
import pytesseract
from pdf2image import convert_from_path

# Convert PDF to images
images = convert_from_path('scanned.pdf')

# OCR each page
text = ""
for i, image in enumerate(images):
    text += f"Page {i+1}:\n"
    text += pytesseract.image_to_string(image)
    text += "\n\n"

print(text)
```

### Add Watermark
```python
from pypdf import PdfReader, PdfWriter

# Create watermark (or load existing)
watermark = PdfReader("watermark.pdf").pages[0]

# Apply to all pages
reader = PdfReader("document.pdf")
writer = PdfWriter()

for page in reader.pages:
    page.merge_page(watermark)
    writer.add_page(page)

with open("watermarked.pdf", "wb") as output:
    writer.write(output)
```

### Extract Images
```bash
# Using pdfimages (poppler-utils)
pdfimages -j input.pdf output_prefix

# This extracts all images as output_prefix-000.jpg, output_prefix-001.jpg, etc.
```

### Password Protection
```python
from pypdf import PdfReader, PdfWriter

reader = PdfReader("input.pdf")
writer = PdfWriter()

for page in reader.pages:
    writer.add_page(page)

# Add password
writer.encrypt("userpassword", "ownerpassword")

with open("encrypted.pdf", "wb") as output:
    writer.write(output)
```

## Quick Reference

| Task | Best Tool | Command/Code |
|------|-----------|--------------|
| Merge PDFs | pypdf | `writer.add_page(page)` |
| Split PDFs | pypdf | One page per file |
| Extract text | pdfplumber | `page.extract_text()` |
| Extract tables | pdfplumber | `page.extract_tables()` |
| Create PDFs | reportlab | Canvas or Platypus |
| Command line merge | qpdf | `qpdf --empty --pages ...` |
| OCR scanned PDFs | pytesseract | Convert to image first |
| Fill PDF forms | pdf-lib or pypdf (see forms.md) | See forms.md |

## Next Steps

- For advanced pypdfium2 usage, see reference.md
- For JavaScript libraries (pdf-lib), see reference.md
- If you need to fill out a PDF form, follow the instructions in forms.md
- For troubleshooting guides, see reference.md
