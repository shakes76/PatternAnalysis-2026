"""Export this project's Markdown report and local PNG figures to an A4 PDF.

Supports the report's headings, paragraphs/lists, pipe tables, code fences and
images. The historical feasibility section is rendered from its Markdown source
with build_review_pdf, then inserted as one selectable-text page. No web fetches.
"""
from __future__ import annotations

import argparse
from html import escape
from io import BytesIO
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

from matplotlib import font_manager
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_review_pdf import render as render_review


ROOT_URL = 'https://github.com/qinyuan666/PatternAnalysis-2026/blob/topic-recognition/recognition/hipmri_unet_qinyuan666/'


def inline(value):
    value = escape(value)
    def link(m):
        label, target = m.groups()
        if not target.startswith(('https://', 'http://')):
            target = ROOT_URL + quote(target, safe='/#')
        return f'<link href="{target}" color="#175d88">{label}</link>'
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'`([^`]+)`', r'<font name="ReportMono" size="8">\1</font>', value)
    return re.sub(r'\*([^*]+)\*', r'<i>\1</i>', value)


def render_part(markdown, root, output):
    body = ParagraphStyle('Body', fontName='Report', fontSize=9.5, leading=13,
                          spaceAfter=7, splitLongWords=True)
    title = ParagraphStyle('Title', parent=body, fontName='ReportBold', fontSize=17, leading=21, spaceAfter=13)
    heading = ParagraphStyle('Heading', parent=body, fontName='ReportBold', fontSize=12,
                             leading=16, spaceBefore=11, spaceAfter=6, keepWithNext=True)
    sub = ParagraphStyle('Subheading', parent=heading, fontSize=10.5, leading=14)
    cell = ParagraphStyle('Cell', parent=body, fontSize=8.1, leading=10.5, spaceAfter=0)
    code = ParagraphStyle('Code', parent=body, fontName='ReportMono', fontSize=7.4,
                          leading=10, backColor=colors.HexColor('#f3f5f7'), spaceAfter=2)
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=38,rightMargin=38,
                                 topMargin=35,bottomMargin=38, title='COMP3710 HipMRI final report',author='s4906920')
    width = A4[0]-76
    lines = markdown.splitlines()
    story=[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('<!--'):
            i+=1; continue
        if line.startswith('```'):
            i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'):
                story.append(Paragraph(escape(lines[i]) or ' ',code)); i+=1
            i+=1; story.append(Spacer(1,6)); continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in cells):
                    rows.append([Paragraph(inline(x),cell) for x in cells])
                i+=1
            count=len(rows[0])
            first=.42 if count==2 else .34 if count==3 else .32
            widths=[width*first]+[width*(1-first)/(count-1)]*(count-1)
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4ecf3')),
                ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#bdc8d2')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([KeepTogether([table]) if len(rows)<=9 else table,Spacer(1,8)]);continue
        picture=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if picture:
            image=Image(str(root/picture[2]))
            scale=min(width/image.imageWidth,370/image.imageHeight)
            image.drawWidth=image.imageWidth*scale;image.drawHeight=image.imageHeight*scale
            image.keepWithNext=True
            story.extend([image,Paragraph(inline(picture[1]),cell),Spacer(1,9)])
            i+=1;continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'))
            paragraph=Paragraph(inline(line[level:].strip()),title if level==1 else heading if level==2 else sub)
            paragraph.report_heading=True
            paragraph.failure_start=bool(re.match(r'### F[1-5]:',line))
            story.append(paragraph)
            i+=1;continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|!\[|<!--|\d+\. |[-*] )',lines[i].strip()):
            parts.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(parts)),body))
    grouped=[];j=0
    while j<len(story):
        if getattr(story[j],'failure_start',False):
            chunk=[story[j]];j+=1
            while j<len(story) and not getattr(story[j],'report_heading',False):
                chunk.append(story[j]);j+=1
            grouped.append(KeepTogether(chunk))
        else:
            grouped.append(story[j]);j+=1
    document.build(grouped)


def render(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    for name,family,weight,style in [('Report','DejaVu Sans','normal','normal'),
                                    ('ReportBold','DejaVu Sans','bold','normal'),
                                    ('ReportItalic','DejaVu Sans','normal','oblique'),
                                    ('ReportMono','DejaVu Sans Mono','normal','normal')]:
        font=font_manager.findfont(font_manager.FontProperties(family=family,weight=weight,style=style))
        pdfmetrics.registerFont(TTFont(name,font))
    pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBold')
    text=source.read_text(encoding='utf-8')
    before,rest=text.split('<!-- FEASIBILITY_START -->',1)
    feasibility,after=rest.split('<!-- FEASIBILITY_END -->',1)
    original=(source.parent/'feasibility_review.md').read_text(encoding='utf-8')
    # README supplies the surrounding heading; the standalone file has its own H1.
    original=re.sub(r'\A# [^\n]+\n', '', original)
    # The embedded historical section may use different heading levels only.
    normalize=lambda s: re.sub(r'(?m)^#+\s*','',s).strip()
    if normalize(feasibility)!=normalize(original):
        raise ValueError('Embedded feasibility content differs from its one-page source')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        render_part(before,source.parent,root/'before.pdf')
        render_review(source.parent/'feasibility_review.md',root/'review.pdf')
        render_part(after,source.parent,root/'after.pdf')
        merged=PdfWriter()
        for name in ('before.pdf','review.pdf','after.pdf'):
            merged.append(str(root/name))
        total=len(merged.pages)
        for index,page in enumerate(merged.pages,1):
            stream=BytesIO();c=canvas.Canvas(stream,pagesize=A4)
            c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#455868'))
            c.drawString(38,18,'COMP3710 | s4906920')
            c.drawRightString(A4[0]-38,18,f'{index} / {total}');c.save();stream.seek(0)
            page.merge_page(PdfReader(stream).pages[0])
        merged.add_metadata({'/Title':'HipMRI segmentation: U-Net versus Improved U-Net','/Author':'s4906920'})
        with output.open('wb') as stream:merged.write(stream)
    print(f'Exported {total} pages to {output.name}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).with_name('README.md'))
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('final_report.pdf'))
    render(**vars(p.parse_args()))
