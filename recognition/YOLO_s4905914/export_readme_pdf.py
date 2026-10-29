"""Export this report's Markdown subset to a plain PDF with no header/footer.

README.md is the sole prose source. Supports headings, paragraphs, lists,
tables, fenced code, local raster figures and explicit pagebreak comments.
Fails on unsupported nonempty HTML rather than silently omitting content.
"""
import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, Preformatted)


def inline(text):
    text=escape(text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',r'<link href="\2" color="#000000"><u>\1</u></link>',text)
    return text


def export(source, output):
    source=source.resolve();output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    styles={
        'body':ParagraphStyle('body',fontName='Helvetica',fontSize=9.5,leading=12.5,spaceAfter=7),
        'h1':ParagraphStyle('h1',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=12),
        'h2':ParagraphStyle('h2',fontName='Helvetica-Bold',fontSize=13,leading=16,spaceBefore=8,spaceAfter=8,keepWithNext=True),
        'h3':ParagraphStyle('h3',fontName='Helvetica-Bold',fontSize=10.5,leading=14,spaceBefore=6,spaceAfter=6,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.7,leading=11),
        'code':ParagraphStyle('code',fontName='Courier',fontSize=7.3,leading=9,spaceAfter=8),
    }
    width=A4[0]-84; story=[]; lines=source.read_text().splitlines();i=0;figures=[];paragraphs=[]
    def paragraph(t,style='body'):
        paragraphs.append(t);story.append(Paragraph(inline(t),styles[style]))
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- pagebreak -->':story.append(PageBreak());i+=1;continue
        if line.startswith('<'):raise ValueError('Unsupported HTML: '+line)
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'):
                code.extend(textwrap.wrap(lines[i],width=102,replace_whitespace=False,drop_whitespace=False) or ['']);i+=1
            if i==len(lines):raise ValueError('Unclosed code fence')
            story.append(Preformatted('\n'.join(code),styles['code']));i+=1;continue
        image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if image:
            path=source.parent/image.group(2)
            with PILImage.open(path) as im:w,h=im.size
            scale=min(width/w,260/h)
            story.extend([Image(str(path),width=w*scale,height=h*scale),Spacer(1,7)])
            figures.append({'path':image.group(2),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});i+=1;continue
        heading=re.match(r'^(#{1,3}) (.+)$',line)
        if heading:paragraph(heading.group(2),'h'+str(len(heading.group(1))));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            cols=len(rows[0]);assert all(len(r)==cols for r in rows)
            widths=([width*.56,width*.22,width*.22] if cols==3 else
                    [width*.40]+[width*.60/(cols-1)]*(cols-1) if cols>1 else [width])
            obj=Table([[Paragraph(inline(c),styles['cell']) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
            obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,-1),(-1,-1),.4,colors.black),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([obj,Spacer(1,9)]);continue
        if line.startswith('- ') or re.match(r'^\d+\. ',line):
            paragraph(line);i+=1;continue
        block=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','```','![','<!--','- ')) and not re.match(r'^\d+\. ',lines[i].strip()):
            block.append(lines[i].strip());i+=1
        paragraph(' '.join(block))
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=38,
                       title='COMP3710 Project 3: lesion detection and classification',author='s4905914').build(story)
    record={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'pdf':str(output),'pdf_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'paragraphs':len(paragraphs),'figures':figures,'running_headers_footers_page_numbers':False}
    output.with_suffix('.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path(__file__).with_name('README.md'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.source,a.output)
