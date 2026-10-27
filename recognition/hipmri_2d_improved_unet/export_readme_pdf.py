"""Export this project's README to PDF without running any project code.

Supported input is deliberately limited: ATX headings, paragraphs, one-level or
indented list items, fenced code, horizontal rules, inline code/emphasis/links,
standalone local PNG/JPEG Markdown images, span/s/a/b/strong/i/em/code/br HTML,
and rectangular HTML tables (table/tr/th/td). Only CSS color is accepted.
The exact div class="feasibility-review" marks an isolated one-page section.
Unsupported block syntax, HTML tags/attributes, unsafe links, and oversized
one-page sections raise an error. Code fences are printed, never executed.

Only README.md, its local images, system font files and runtime dependencies
are read. There are no project imports, model operations or network requests.
Requires reportlab and Pillow; see requirements-docs.txt. PDF generation is a
separate authoring operation and must follow the applicable PDF review process.
"""

import argparse
from dataclasses import dataclass, field
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, HRFlowable, Image, KeepTogether,
    LongTable, PageBreak, PageTemplate, Paragraph, Spacer, TableStyle,
)


PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "README.md"
PURPLE = "#800080"
CJK_FONT = "ReadmeCJK"
BODY_SIZE = 9.0
BODY_LEADING = 13.0
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = MARGIN_RIGHT = 42.0
MARGIN_TOP = 44.0
MARGIN_BOTTOM = 40.0
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
CONTENT_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
INLINE_TAGS = {"span", "s", "a", "b", "strong", "i", "em", "code", "br"}
TABLE_TAGS = {"table", "tr", "th", "td"}
VOID_TAGS = {"br"}
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_ITEM = re.compile(r"^( *)([-+*]|\d+[.)])\s+(.+)$")
IMAGE_LINE = re.compile(r'^!\[([^\]]*)\]\(([^\s)]+)(?:\s+"[^\"]*")?\)\s*$')
HTML_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]*)(?:\s[^<>]*|/?)>")
KNOWN_HTML = re.compile(r"</?(?:span|s|a|b|strong|i|em|code|br|table|tr|th|td)\b[^<>]*>", re.I)
FEASIBILITY_OPEN = '<div class="feasibility-review">'


@dataclass
class Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_color(value):
    value = value.strip().lower()
    allowed = {"purple": PURPLE, "black": "#000000", "white": "#ffffff",
               "gray": "#808080", "grey": "#808080"}
    require(value in allowed or re.fullmatch(r"#[0-9a-f]{6}", value),
            f"Unsupported CSS color: {value!r}")
    return allowed.get(value, value)


def style_color(attrs):
    result = None
    for declaration in attrs.get("style", "").split(";"):
        if not declaration.strip():
            continue
        key, separator, value = declaration.partition(":")
        require(separator and key.strip().lower() == "color",
                f"Only CSS color is supported: {declaration!r}")
        result = checked_color(value)
    return result


def safe_link(target):
    target = html.unescape(target.strip())
    require(target and not any(ord(char) < 32 for char in target), "Empty or control-character link")
    parts = urlsplit(target)
    require(parts.scheme.lower() in {"", "http", "https", "mailto"},
            f"Unsupported link scheme: {target!r}")
    require(not (not parts.scheme and parts.netloc), "Protocol-relative links are unsupported")
    require(not target.startswith("#"), "In-document anchor links are unsupported")
    return target


class FragmentParser(HTMLParser):
    """Validate HTML and retain every text fragment in a small DOM."""

    def __init__(self, allow_tables=False):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]
        self.allowed = INLINE_TAGS | (TABLE_TAGS if allow_tables else set())

    def handle_starttag(self, tag, attrs):
        require(tag in self.allowed, f"Unsupported HTML tag: <{tag}>")
        require(len({key for key, _ in attrs}) == len(attrs), f"Duplicate attributes on <{tag}>")
        attributes = dict(attrs)
        allowed_attrs = {"style", "href"} if tag == "a" else {"style"}
        require(set(attributes) <= allowed_attrs,
                f"Unsupported attributes on <{tag}>: {set(attributes) - allowed_attrs}")
        require(all(value is not None for value in attributes.values()), "Boolean HTML attributes are unsupported")
        style_color(attributes)
        if tag == "a":
            require("href" in attributes, "Every HTML anchor needs href")
            safe_link(attributes["href"])
        node = Node(tag, attributes)
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        require(tag not in VOID_TAGS, f"Unexpected closing void tag: </{tag}>")
        require(len(self.stack) > 1 and self.stack[-1].tag == tag,
                f"Unbalanced or unsupported closing tag: </{tag}>")
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].children.append(data)

    def handle_comment(self, data):
        raise ValueError("HTML comments are unsupported; do not hide README content")

    def handle_decl(self, declaration):
        raise ValueError(f"HTML declarations are unsupported: {declaration}")

    def unknown_decl(self, declaration):
        raise ValueError(f"Unknown HTML declaration: {declaration}")

    def finished(self):
        self.close()
        require(len(self.stack) == 1, f"Unclosed HTML tag: <{self.stack[-1].tag}>")
        return self.root


def markdown_to_html(text, allow_tables=False):
    """Translate supported inline Markdown while escaping literal comparisons."""
    output = []
    index = 0
    while index < len(text):
        rest = text[index:]
        if rest.startswith("<!--"):
            raise ValueError("HTML comments are unsupported")
        entity = re.match(r"&(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);", rest)
        if entity:
            output.append(entity.group(0))
            index += entity.end()
            continue
        if rest.startswith("!["):
            raise ValueError("Images must be standalone Markdown blocks")
        if rest.startswith("`"):
            marker = re.match(r"`+", rest).group(0)
            end = text.find(marker, index + len(marker))
            require(end >= 0, "Unclosed inline-code delimiter")
            output.append("<code>" + html.escape(text[index + len(marker):end]) + "</code>")
            index = end + len(marker)
            continue
        link = re.match(r'\[([^\]]+)\]\(([^\s)]+)(?:\s+"[^\"]*")?\)', rest)
        if link:
            target = safe_link(link.group(2))
            output.append('<a href="' + html.escape(target, quote=True) + '">'
                          + markdown_to_html(link.group(1)) + "</a>")
            index += link.end()
            continue
        matched_emphasis = False
        for marker, tag in (("**", "b"), ("__", "b"), ("~~", "s"), ("*", "i")):
            if not rest.startswith(marker):
                continue
            end = text.find(marker, index + len(marker))
            if end > index + len(marker):
                output.append(f"<{tag}>" + markdown_to_html(text[index + len(marker):end]) + f"</{tag}>")
                index = end + len(marker)
                matched_emphasis = True
                break
        if matched_emphasis:
            continue
        if rest.startswith("<"):
            autolink = re.match(r"<(https?://[^<>\s]+|[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>", rest)
            if autolink:
                label = autolink.group(1)
                target = label if label.startswith(("http://", "https://")) else "mailto:" + label
                output.append('<a href="' + html.escape(safe_link(target), quote=True) + '">'
                              + html.escape(label) + "</a>")
                index += autolink.end()
                continue
            known = KNOWN_HTML.match(rest)
            if known:
                tag_name = re.match(r"</?([A-Za-z]+)", known.group(0)).group(1).lower()
                require(allow_tables or tag_name not in TABLE_TAGS, "Tables must be standalone HTML blocks")
                output.append(known.group(0))
                index += known.end()
                continue
            unknown = HTML_TAG.match(rest)
            require(not unknown, f"Unsupported HTML syntax: {unknown.group(0) if unknown else rest[:30]}")
            require(not rest.startswith(("<!", "<?")), "Unsupported HTML declaration")
        if rest.startswith("\\") and len(rest) > 1 and rest[1] in r"\`*_{}[]()#+-.!<>":
            output.append(html.escape(rest[1]))
            index += 2
            continue
        output.append(html.escape(text[index]))
        index += 1
    return "".join(output)


def parse_fragment(text, allow_tables=False):
    parser = FragmentParser(allow_tables)
    parser.feed(markdown_to_html(text, allow_tables))
    return parser.finished()


def plain_text(node):
    if isinstance(node, str):
        return node
    return "".join(plain_text(child) for child in node.children)


def text_markup(text, font=None):
    """Use an embedded Chinese font for non-ASCII characters."""
    output = []
    for group in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", text):
        escaped = html.escape(group)
        face = CJK_FONT if ord(group[0]) > 127 else font
        output.append(f'<font name="{face}">{escaped}</font>' if face else escaped)
    return "".join(output)


def inline_markup(node, code=False):
    if isinstance(node, str):
        return text_markup(node, "ReadmeMono" if code else None)
    content = "".join(inline_markup(child, code or node.tag == "code") for child in node.children)
    if node.tag == "br":
        return "<br/>"
    if node.tag in {"b", "strong"}:
        content = "<b>" + content + "</b>"
    elif node.tag in {"i", "em"}:
        content = "<i>" + content + "</i>"
    elif node.tag == "s":
        content = "<strike>" + content + "</strike>"
    elif node.tag == "a":
        content = '<link href="' + html.escape(safe_link(node.attrs["href"]), quote=True) + '">' + content + "</link>"
    elif node.tag == "code":
        content = '<font name="ReadmeMono" size="8.1">' + content + "</font>"
    color = style_color(node.attrs)
    return '<font color="' + color + '">' + content + "</font>" if color else content


def register_fonts():
    global CJK_FONT
    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    variants = {"ReadmeSans": "arial.ttf", "ReadmeSans-Bold": "arialbd.ttf",
                "ReadmeSans-Italic": "ariali.ttf", "ReadmeSans-BoldItalic": "arialbi.ttf",
                "ReadmeMono": "consola.ttf"}
    latin_embedded = all((font_dir / name).is_file() for name in variants.values())
    if latin_embedded:
        for name, filename in variants.items():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        pdfmetrics.registerFontFamily("ReadmeSans", normal="ReadmeSans", bold="ReadmeSans-Bold",
                                      italic="ReadmeSans-Italic", boldItalic="ReadmeSans-BoldItalic")
        pdfmetrics.registerFontFamily("ReadmeMono", normal="ReadmeMono", bold="ReadmeMono",
                                      italic="ReadmeMono", boldItalic="ReadmeMono")
    else:
        for name, base in (("ReadmeSans", "Helvetica"), ("ReadmeSans-Bold", "Helvetica-Bold"),
                           ("ReadmeSans-Italic", "Helvetica-Oblique"),
                           ("ReadmeSans-BoldItalic", "Helvetica-BoldOblique"), ("ReadmeMono", "Courier")):
            from reportlab.pdfbase.pdfmetrics import Font
            pdfmetrics.registerFont(Font(name, base, "WinAnsiEncoding"))
        pdfmetrics.registerFontFamily("ReadmeSans", normal="ReadmeSans", bold="ReadmeSans-Bold",
                                      italic="ReadmeSans-Italic", boldItalic="ReadmeSans-BoldItalic")
        pdfmetrics.registerFontFamily("ReadmeMono", normal="ReadmeMono", bold="ReadmeMono",
                                      italic="ReadmeMono", boldItalic="ReadmeMono")
    chinese = font_dir / "simsun.ttc"
    if chinese.is_file():
        pdfmetrics.registerFont(TTFont("ReadmeCJK", str(chinese), subfontIndex=0))
        CJK_FONT = "ReadmeCJK"
        chinese_description = "embedded SimSun (simsun.ttc subfont 0)"
    else:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        CJK_FONT = "STSong-Light"
        chinese_description = "STSong-Light CID fallback (viewer Chinese font support required)"
    pdfmetrics.registerFontFamily(CJK_FONT, normal=CJK_FONT, bold=CJK_FONT,
                                  italic=CJK_FONT, boldItalic=CJK_FONT)
    return {"latin": "embedded Arial and Consolas" if latin_embedded else "PDF standard Helvetica/Courier fallback",
            "chinese": chinese_description}


def make_styles(compact=False):
    size, leading = (8.5, 11.7) if compact else (BODY_SIZE, BODY_LEADING)
    body = ParagraphStyle("body", fontName="ReadmeSans", fontSize=size, leading=leading,
                          alignment=TA_LEFT, textColor=colors.HexColor("#222222"),
                          wordWrap="CJK", splitLongWords=True, spaceAfter=5 if compact else 6)
    styles = {"body": body,
              "cell": ParagraphStyle("cell", parent=body, fontSize=7.2 if compact else 7.4,
                                     leading=9.6 if compact else 10.1, spaceAfter=0),
              "caption": ParagraphStyle("caption", parent=body, fontSize=7.5, leading=10,
                                        textColor=colors.HexColor("#555555"), spaceAfter=8)}
    for level in range(1, 7):
        sizes = (17, 12.5, 10.5, 9.5, 9, 9)
        heading_size = sizes[level - 1] if not compact else min(sizes[level - 1], 13)
        styles[f"h{level}"] = ParagraphStyle(f"h{level}", parent=body, fontName="ReadmeSans-Bold",
                                             fontSize=heading_size, leading=heading_size * 1.35,
                                             spaceBefore=9 if level > 1 else 3, spaceAfter=5,
                                             keepWithNext=True)
    return styles


class CodeBlock(Flowable):
    """Wrap literal code by measured glyph widths, and split only between lines."""

    def __init__(self, lines, prepared=False):
        super().__init__()
        self.source_lines = list(lines) or [""]
        self.prepared = prepared
        self.lines = self.source_lines if prepared else []
        self.font_size = 7.2
        self.leading = 10.0
        self.padding = 6.0
        self.spaceBefore = 4
        self.spaceAfter = 7

    @staticmethod
    def font_for(char):
        return CJK_FONT if ord(char) > 127 else "ReadmeMono"

    def wrap(self, available_width, available_height):
        self.width = available_width
        if not self.prepared:
            self.lines = []
            limit = max(20, available_width - 2 * self.padding)
            for source in self.source_lines:
                current, width = "", 0.0
                for char in source.expandtabs(4):
                    advance = pdfmetrics.stringWidth(char, self.font_for(char), self.font_size)
                    if current and width + advance > limit:
                        self.lines.append(current)
                        current, width = "", 0.0
                    current += char
                    width += advance
                self.lines.append(current)
        self.height = len(self.lines) * self.leading + 2 * self.padding
        return self.width, self.height

    def split(self, available_width, available_height):
        self.wrap(available_width, available_height)
        count = int((available_height - 2 * self.padding) // self.leading)
        if count <= 0 or count >= len(self.lines):
            return []
        return [CodeBlock(self.lines[:count], prepared=True), CodeBlock(self.lines[count:], prepared=True)]

    def draw(self):
        self.canv.setFillColor(colors.HexColor("#f4f4f6"))
        self.canv.setStrokeColor(colors.HexColor("#dddddf"))
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=1)
        self.canv.setFillColor(colors.HexColor("#222222"))
        y = self.height - self.padding - self.font_size
        for line in self.lines:
            x = self.padding
            for group in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", line):
                face = self.font_for(group[0])
                self.canv.setFont(face, self.font_size)
                self.canv.drawString(x, y, group)
                x += pdfmetrics.stringWidth(group, face, self.font_size)
            y -= self.leading


class OnePageSection(Flowable):
    """Draw a measured section atomically; never silently spill to another page."""

    def __init__(self, items):
        super().__init__()
        self.items = items
        self.measurements = []

    def wrap(self, available_width, available_height):
        self.width = available_width
        y = 0.0
        self.measurements = []
        for item in self.items:
            require(not isinstance(item, PageBreak), "Nested page break in feasibility section")
            width, height = item.wrapOn(self.canv, available_width, CONTENT_HEIGHT)
            before, after = item.getSpaceBefore(), item.getSpaceAfter()
            self.measurements.append((item, height, before, after))
            y += before + height + after
        require(y <= CONTENT_HEIGHT - 0.1,
                f"feasibility-review needs {y:.1f} pt; one page allows {CONTENT_HEIGHT:.1f} pt. Shorten the source section.")
        self.height = y
        return self.width, self.height

    def split(self, available_width, available_height):
        return []

    def draw(self):
        y = self.height
        for item, height, before, after in self.measurements:
            y -= before + height
            item.drawOn(self.canv, 0, y)
            y -= after


class Renderer:
    def __init__(self, project):
        self.project = project.resolve()
        self.image_paths = []
        self.tables = 0
        self.feasibility_sections = 0

    def paragraph(self, text, style):
        return Paragraph(inline_markup(parse_fragment(text)), style)

    def table(self, text, styles):
        root = parse_fragment(text, allow_tables=True)
        actual = [node for node in root.children if not isinstance(node, str) or node.strip()]
        require(len(actual) == 1 and isinstance(actual[0], Node) and actual[0].tag == "table",
                "A table block must contain exactly one HTML table")
        table = actual[0]
        rows = []
        header = False
        for row in table.children:
            if isinstance(row, str):
                require(not row.strip(), "Text outside a table row is unsupported")
                continue
            require(row.tag == "tr", "HTML table requires direct tr children")
            cells = []
            for cell in row.children:
                if isinstance(cell, str):
                    require(not cell.strip(), "Text outside a table cell is unsupported")
                    continue
                require(cell.tag in {"td", "th"}, "Table rows require td/th cells")
                cells.append(cell)
            require(cells, "Empty HTML table row")
            if not rows:
                header = any(cell.tag == "th" for cell in cells)
            rows.append(cells)
        require(rows and len({len(row) for row in rows}) == 1, "HTML tables must be nonempty and rectangular")
        count = len(rows[0])
        weights = []
        for column in range(count):
            longest = max(sum(1 if ord(char) < 128 else 1.7 for char in plain_text(row[column])) for row in rows)
            weights.append(math.sqrt(max(7, min(70, longest))))
        widths = [CONTENT_WIDTH * weight / sum(weights) for weight in weights]
        color = style_color(table.attrs)
        matrix = []
        for row in rows:
            rendered = []
            for cell in row:
                markup = inline_markup(cell)
                if cell.tag == "th":
                    markup = "<b>" + markup + "</b>"
                if color:
                    markup = '<font color="' + color + '">' + markup + "</font>"
                rendered.append(Paragraph(markup, styles["cell"]))
            matrix.append(rendered)
        result = LongTable(matrix, colWidths=widths, repeatRows=1 if header else 0,
                           hAlign="LEFT", splitByRow=1, splitInRow=1)
        commands = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfcfd5")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0eaf4")))
        result.setStyle(TableStyle(commands))
        result.spaceBefore, result.spaceAfter = 4, 8
        self.tables += 1
        return result

    def image_items(self, match, styles, reserved_height=0):
        alt, location = match.groups()
        parts = urlsplit(location)
        require(not parts.scheme and not parts.netloc and not parts.query and not parts.fragment,
                "README images must be local files, with no URL/query/fragment")
        path = (self.project / unquote(location)).resolve()
        require(path.is_relative_to(self.project) and path.is_file(), f"Missing or escaping local image: {location}")
        require(path.suffix.lower() in {".png", ".jpg", ".jpeg"}, f"Unsupported image format: {location}")
        with PILImage.open(path) as source:
            source.verify()
        with PILImage.open(path) as source:
            width, height = source.size
        require(width > 0 and height > 0, "Empty image")
        caption = self.paragraph(alt, styles["caption"]) if alt else None
        caption_height = caption.wrap(CONTENT_WIDTH, CONTENT_HEIGHT)[1] + 12 if caption else 0
        image_height = CONTENT_HEIGHT - caption_height - reserved_height - 16
        require(image_height > 0, "Image heading/caption leaves no printable image area")
        scale = min(CONTENT_WIDTH / width, image_height / height, 72 / 105)
        figure = Image(str(path), width=width * scale, height=height * scale)
        figure.hAlign = "LEFT"
        self.image_paths.append(path.relative_to(self.project).as_posix())
        return [figure, Spacer(1, 4)] + ([caption] if caption else [])

    def image(self, match, styles):
        return KeepTogether(self.image_items(match, styles))

    @staticmethod
    def starts_block(line):
        text = line.strip()
        return bool(not text or HEADING.match(line) or LIST_ITEM.match(line)
                    or text.startswith(("```", "<table", "<div", "![", "|", ">", "~~~", "$$"))
                    or re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", text))

    def blocks(self, lines, compact=False):
        styles = make_styles(compact)
        story = []
        index = 0
        while index < len(lines):
            line, text = lines[index], lines[index].strip()
            if not text:
                index += 1
                continue
            if text == FEASIBILITY_OPEN:
                require(not compact, "Nested feasibility sections are unsupported")
                end = index + 1
                while end < len(lines) and lines[end].strip() != "</div>":
                    end += 1
                require(end < len(lines), "Unclosed feasibility-review div")
                inside = self.blocks(lines[index + 1:end], compact=True)
                require(inside, "Empty feasibility-review section")
                if story:
                    story.append(PageBreak())
                story.append(OnePageSection(inside))
                if any(item.strip() for item in lines[end + 1:]):
                    story.append(PageBreak())
                self.feasibility_sections += 1
                index = end + 1
                continue
            if text.startswith("<div") or text == "</div>":
                raise ValueError('Only a standalone <div class="feasibility-review"> is supported')
            if text.startswith("```"):
                opener = re.fullmatch(r"(`{3,})([A-Za-z0-9_+.-]*)\s*", text)
                require(opener is not None, "Unsupported code-fence opener")
                fence = opener.group(1)
                end = index + 1
                while end < len(lines) and lines[end].strip() != fence:
                    end += 1
                require(end < len(lines), "Unclosed fenced-code block")
                story.append(CodeBlock(lines[index + 1:end]))
                index = end + 1
                continue
            if text.startswith("<table"):
                end = index
                while end < len(lines) and "</table>" not in lines[end]:
                    end += 1
                require(end < len(lines), "Unclosed HTML table")
                story.append(self.table("\n".join(lines[index:end + 1]), styles))
                index = end + 1
                continue
            heading = HEADING.match(line)
            if heading:
                title = self.paragraph(heading.group(2), styles[f"h{len(heading.group(1))}"])
                following = index + 1
                while following < len(lines) and not lines[following].strip():
                    following += 1
                next_image = IMAGE_LINE.fullmatch(lines[following].strip()) if following < len(lines) else None
                if next_image:
                    require(not compact, "Images are unsupported inside the one-page feasibility section")
                    title_height = title.wrap(CONTENT_WIDTH, CONTENT_HEIGHT)[1]
                    reserve = title_height + title.getSpaceBefore() + title.getSpaceAfter() + 4
                    # A single flat group prevents a nested image KeepTogether
                    # from splitting away from a heading at a page boundary.
                    story.append(KeepTogether([title] + self.image_items(next_image, styles, reserve)))
                    index = following + 1
                else:
                    story.append(title)
                    index += 1
                continue
            image = IMAGE_LINE.fullmatch(text)
            if image:
                story.append(self.image(image, styles))
                index += 1
                continue
            item = LIST_ITEM.match(line)
            if item:
                indent, marker, content = item.groups()
                extra = min(len(indent) * 2.8, 60)
                style = ParagraphStyle("list-item", parent=styles["body"], leftIndent=12 + extra,
                                       firstLineIndent=0, bulletIndent=extra, bulletFontName="ReadmeSans",
                                       bulletFontSize=styles["body"].fontSize)
                bullet = marker if marker[0].isdigit() else "-"
                story.append(Paragraph(inline_markup(parse_fragment(content)), style, bulletText=bullet))
                index += 1
                continue
            if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", text):
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccccd2"),
                                        spaceBefore=5, spaceAfter=7))
                index += 1
                continue
            require(not text.startswith(("|", ">", "~~~", "$$", "![")),
                    f"Unsupported Markdown block at line {index + 1}: {text[:80]}")
            require(not re.match(r"^\[[^\]]+\]:", text), "Reference-style Markdown links are unsupported")
            require(not line.startswith(("    ", "\t")), "Indented code is unsupported; use fenced code")
            paragraph = [line]
            end = index + 1
            while end < len(lines) and not self.starts_block(lines[end]):
                require(not lines[end].startswith(("    ", "\t")), "Indented continuation is unsupported")
                paragraph.append(lines[end])
                end += 1
            combined = "".join(value.rstrip() + ("<br/>" if value.endswith("  ") else " ")
                               for value in paragraph).strip()
            story.append(self.paragraph(combined, styles["body"]))
            index = end
        return story


def page_furniture(canvas, document):
    canvas.saveState()
    canvas.setTitle("HipMRI 2D Segmentation - COMP3710 Report")
    canvas.setAuthor("Project README; PDF export by Codex automation")
    canvas.setFont("ReadmeSans", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(MARGIN_LEFT, PAGE_HEIGHT - 25, "HipMRI 2D Segmentation | COMP3710")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_RIGHT, 22, f"Page {document.page}")
    canvas.restoreState()


def export(output):
    output = Path(output).resolve()
    require(output.suffix.lower() == ".pdf", "Output must use the .pdf extension")
    require(not output.exists(), f"Refusing to overwrite an existing output: {output}")
    require(SOURCE.is_file(), "Project README.md is missing")
    text = SOURCE.read_text(encoding="utf-8-sig")
    require(text.strip(), "README.md is empty")
    require(not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text), "README contains unsupported control characters")
    fonts = register_fonts()
    renderer = Renderer(PROJECT)
    story = renderer.blocks(text.splitlines())
    require(story, "README produced no printable blocks")
    require(renderer.feasibility_sections == 1, "README must contain exactly one feasibility-review section")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    created_output = False
    try:
        with tempfile.NamedTemporaryFile(prefix=output.stem + ".", suffix=".partial.pdf",
                                         dir=output.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
        document = BaseDocTemplate(str(temp_path), pagesize=A4, leftMargin=MARGIN_LEFT,
                                   rightMargin=MARGIN_RIGHT, topMargin=MARGIN_TOP,
                                   bottomMargin=MARGIN_BOTTOM, allowSplitting=1)
        frame = Frame(MARGIN_LEFT, MARGIN_BOTTOM, CONTENT_WIDTH, CONTENT_HEIGHT,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="body")
        document.addPageTemplates(PageTemplate(id="readme", frames=[frame], onPage=page_furniture))
        document.build(story)
        # Exclusive destination creation preserves a file that appeared after preflight.
        with output.open("xb") as target, temp_path.open("rb") as source:
            created_output = True
            shutil.copyfileobj(source, target)
    except Exception:
        if created_output:
            output.unlink(missing_ok=True)
        raise
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return {"source": str(SOURCE), "output": str(output), "fonts": fonts,
            "tables": renderer.tables, "local_images": renderer.image_paths,
            "feasibility_sections": renderer.feasibility_sections,
            "layout_review_required": True, "project_modules_imported": False,
            "network_used": False, "code_fences_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "README.pdf",
                        help="New PDF destination; existing files are never overwritten.")
    args = parser.parse_args()
    print(json.dumps(export(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
