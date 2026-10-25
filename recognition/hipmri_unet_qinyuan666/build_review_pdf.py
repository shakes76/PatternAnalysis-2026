"""Render the checked-in feasibility Markdown as exactly one readable A4 page.

The Markdown remains the content source. This small renderer supports the
headings, lists, links and table used by that source; it is not a general parser.
It refuses a multipage result instead of silently shrinking the font.
"""

from __future__ import annotations

import argparse
from functools import partial
from html import escape
from pathlib import Path
import re

from matplotlib import font_manager
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  r'<link href="\2" color="#175d88">\1</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r'<font name="ReviewMono" size="8.5">\1</font>', text)


def render(source: Path, output: Path) -> None:
    for name, family, weight in (("Review", "DejaVu Sans", "normal"),
                                 ("ReviewBold", "DejaVu Sans", "bold"),
                                 ("ReviewMono", "DejaVu Sans Mono", "normal")):
        font = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight))
        pdfmetrics.registerFont(TTFont(name, font))
    pdfmetrics.registerFontFamily("Review", normal="Review", bold="ReviewBold")
    body = ParagraphStyle("Body", fontName="Review", fontSize=9.8, leading=12.3,
                          spaceAfter=4.5, textColor=colors.HexColor("#172535"))
    title = ParagraphStyle("Title", parent=body, fontName="ReviewBold", fontSize=14,
                           leading=17, spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=body, fontName="ReviewBold", fontSize=10,
                             leading=12.5, spaceBefore=7, spaceAfter=3, keepWithNext=True)
    item = ParagraphStyle("Item", parent=body, leftIndent=10, firstLineIndent=-10,
                          spaceAfter=2)
    cell = ParagraphStyle("Cell", parent=body, fontSize=9, leading=11.5, spaceAfter=0)
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=35,
                                  rightMargin=35, topMargin=29, bottomMargin=30,
                                  title="COMP3710 HipMRI midpoint feasibility review",
                                  author="s4906920", pageCompression=1)
    lines = source.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [part.strip() for part in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", part) for part in cells):
                    rows.append([Paragraph(inline(part), cell) for part in cells])
                index += 1
            table = Table(rows, colWidths=[245, 140, A4[0] - 70 - 385], hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f7")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9eb4c7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.extend([table, Spacer(1, 5)])
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), title))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), heading))
            index += 1
            continue
        is_item = bool(re.match(r"\d+\. ", line))
        paragraph = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if (not next_line or next_line.startswith(("#", "|"))
                    or re.match(r"\d+\. ", next_line)):
                break
            paragraph.append(next_line)
            index += 1
        story.append(Paragraph(inline(" ".join(paragraph)), item if is_item else body))
    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, canvasmaker=partial(Canvas, invariant=1))
    pages = len(PdfReader(output).pages)
    if pages != 1:
        raise RuntimeError(f"Review produced {pages} pages; shorten the source before submission.")
    print(f"Created {output.name}: one A4 page; body text 9.8 pt.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "feasibility_review.md")
    parser.add_argument("--output", type=Path, default=root / "feasibility_review.pdf")
    args = parser.parse_args()
    render(args.source, args.output)
