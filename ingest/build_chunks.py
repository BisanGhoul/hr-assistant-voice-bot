"""
Turn HR procedure PDFs into chunks for retrieval.

Reads every PDF under data/ (including subfolders), extracts the text,
cleans it, splits it into chunks, and writes them to data/chunks.jsonl.

Run from the server/ folder:
    uv run ingest/build_chunks.py
"""

import json
import re
from pathlib import Path

import pymupdf
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "chunks.jsonl"

SINGLE_CHUNK_LIMIT = 1500  # cleaned docs shorter than this stay as one chunk
CHUNK_SIZE = 1000          # characters, roughly 250 tokens
CHUNK_OVERLAP = 150

# Matches "**Label:** value" pairs in the header line, e.g. "**Department:** Administration"
HEADER_PAIR = re.compile(r"\*\*\s*([A-Za-z ]+?)\s*:\s*\*\*\s*([^*]+)")
# Bullet symbols from Symbol/Wingdings fonts come out as Unicode private-use characters
PRIVATE_USE_CHARS = re.compile(r"[\ue000-\uf8ff]")
LEADING_BULLETS = re.compile(r"^(?:-\s*)+")
# Checklist-style PDFs use a letter "o" as the bullet, sometimes alone on its line
LONE_O_BULLET = re.compile(r"(?m)^[ \t]*o[ \t]*\n\s*")
O_BULLET = re.compile(r"(?m)^[ \t]*[oO][ \t]+")
# Markdown table separator rows like |---|---|
TABLE_RULE = re.compile(r"(?m)^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def parse_header(md):
    """Find the 'Issued Date / Procedure Number / ...' line.
    Returns (fields dict, markdown with that line removed)."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if "Procedure Number" in line:
            fields = {
                label.strip().lower().replace(" ", "_"): value.strip()
                for label, value in HEADER_PAIR.findall(line)
            }
            return fields, "\n".join(lines[:i] + lines[i + 1:])
    return {}, md


def flatten_tables(md):
    """Turn Markdown tables into plain lines: one cell per line."""
    md = TABLE_RULE.sub("", md)
    return md.replace("<br>", "\n").replace("|", "\n")


def clean(md):
    """Strip formatting noise and fix broken bullets."""
    md = flatten_tables(md)
    md = O_BULLET.sub("- ", md)
    md = md.replace("<mark>", "").replace("</mark>", "").replace("**", "")
    md = PRIVATE_USE_CHARS.sub("\n- ", md)

    seen_titles = set()
    out = []
    for raw in md.splitlines():
        line = raw.strip()

        # '#' lines are the title repeated on every page: keep the first, drop repeats
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            line = title

        is_bullet = line.startswith("-")
        text = re.sub(r"\s+", " ", LEADING_BULLETS.sub("", line)).strip()
        if not text:
            continue  # empty bullet or blank line

        # A line starting lowercase is a sentence the PDF broke in two: join it back
        if out and text[0].islower():
            out[-1] += " " + text
        else:
            out.append(("- " if is_bullet else "") + text)

    return "\n".join(out)


def plain_text(pdf):
    """Plain text extraction. Cleaner than Markdown for table-style PDFs,
    where the Markdown version splits words across table cells."""
    with pymupdf.open(pdf) as doc:
        text = "\n".join(page.get_text() for page in doc)
    text = LONE_O_BULLET.sub("- ", text)
    return O_BULLET.sub("- ", text)


def build_chunks(pdf):
    # For image-only pages, to_markdown runs OCR automatically (needs rapidocr installed)
    md = pymupdf4llm.to_markdown(str(pdf))
    header, body = parse_header(md)
    raw = plain_text(pdf)
    used_ocr = not raw.strip()  # no selectable text at all = scanned PDF

    if header:
        text = clean(body)
    else:
        # No standard procedure header: a table/checklist doc, or a scanned one
        if used_ocr:
            print(f"NOTE: {pdf.name} has no selectable text, using OCR. Check its numbers against the PDF!")
            text = clean(md)
        else:
            print(f"NOTE: no standard header in {pdf.name}, using plain-text extraction")
            text = clean(raw)
        first_line = text.splitlines()[0] if text else pdf.stem
        header = {"procedure": first_line, "procedure_number": pdf.stem}

    if not text:
        return header, []

    title = header.get("procedure", pdf.stem)
    number = header.get("procedure_number", "")
    label = f"Procedure: {title}" + (f" ({number})" if number else "")
    pieces = [text] if len(text) <= SINGLE_CHUNK_LIMIT else splitter.split_text(text)
    source = pdf.relative_to(DATA_DIR).as_posix()

    chunks = [
        {
            "id": f"{pdf.stem}-{i}",
            "text": f"{label}\n{piece}",  # prefix gives every chunk its context
            # ocr=True marks text that may contain OCR mistakes
            "metadata": {"source": source, "chunk": i, "ocr": used_ocr, **header},
        }
        for i, piece in enumerate(pieces)
    ]
    return header, chunks


def main():
    pdfs = sorted(DATA_DIR.glob("**/*.pdf"))
    total = 0
    # utf-8 matters on Windows: the default encoding breaks on characters like ’ and Arabic
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            header, chunks = build_chunks(pdf)
            if not chunks:
                print(f"WARNING: no text in {pdf.name} (is rapidocr installed?)")
                continue
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"{pdf.name}: {len(chunks)} chunk(s)")

    print(f"\n{total} chunks from {len(pdfs)} PDFs -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()