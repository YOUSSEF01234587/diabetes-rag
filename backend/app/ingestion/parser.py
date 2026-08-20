"""PDF parser - extracts text, tables, and structure from PDFs."""
import re

import pymupdf
from pathlib import Path
from typing import Optional


def parse_pdf(pdf_path: Path) -> list[dict]:
    """Extract structured page data from a PDF."""
    doc = pymupdf.open(str(pdf_path))
    pages = []

    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text("text")

        blocks_dict = page.get_text("dict")["blocks"]
        structured_lines = []
        tables_detected = False

        for block in blocks_dict:
            if block["type"] == 0:
                for line in block.get("lines", []):
                    spans = []
                    for span in line.get("spans", []):
                        spans.append({
                            "text": span["text"],
                            "font_size": round(span["size"], 1),
                            "flags": span["flags"],
                            "is_bold": bool(span["flags"] & (1 << 4)),
                            "is_italic": bool(span["flags"] & (1 << 2)),
                        })
                    line_text = "".join(s["text"] for s in spans)
                    if not line_text.strip():
                        continue
                    max_font = max((s["font_size"] for s in spans), default=0)
                    any_bold = any(s["is_bold"] for s in spans)
                    structured_lines.append({
                        "text": line_text.strip(),
                        "font_size": max_font,
                        "is_bold": any_bold,
                    })

        drawings = page.get_drawings()
        if len(drawings) > 20:
            tables_detected = True

        images = page.get_images(full=True)

        pages.append({
            "page_pdf": i + 1,
            "text": text,
            "text_length": len(text.strip()),
            "structured_lines": structured_lines,
            "has_images": len(images) > 0,
            "num_images": len(images),
            "tables_detected": tables_detected,
            "num_drawings": len(drawings),
        })

    doc.close()
    return pages
