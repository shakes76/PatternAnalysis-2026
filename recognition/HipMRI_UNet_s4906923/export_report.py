"""Create REPORT.pdf from the README (requires reportlab==5.0.1).

Render the report's headings, paragraphs, lists, code blocks, Markdown tables,
links and PNG figures. Text and figures come directly from the README;
there is no separate report text to keep in sync.
"""
import argparse,html,re,textwrap
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,Preformatted,KeepTogether,PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parent


def inline(text):
    text=html.escape(text)
    text=re.sub(r"`([^`]+)`",r'<font name="Courier">\1</font>',text)
    text=re.sub(r"\*\*([^*]+)\*\*",r'<b>\1</b>',text)
    text=re.sub(r"\*([^*]+)\*",r'<i>\1</i>',text)
    def link(match):
        label,target=match.groups()
        if target.startswith("#"):
            return f'<link href="{target}" color="#245b84"><u>{label}</u></link>'
        return label
    text=re.sub(r"\[([^]]+)\]\(([^)]+)\)",link,text)
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"REPORT.pdf")
    args=parser.parse_args()
    fontdir=Path("C:/Windows/Fonts")
    if (fontdir/"arial.ttf").exists():
        for name,file in [("Body","arial.ttf"),("BodyBold","arialbd.ttf"),("BodyItalic","ariali.ttf")]:
            pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
        pdfmetrics.registerFontFamily("Body",normal="Body",bold="BodyBold",italic="BodyItalic",boldItalic="BodyBold")
        font="Body"
    else:font="Helvetica"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Text",fontName=font,fontSize=9,leading=12,spaceAfter=6,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle(name="Cell",parent=styles["Text"],fontSize=7,leading=9,spaceAfter=0,splitLongWords=True))
    styles.add(ParagraphStyle(name="CodeSmall",fontName="Courier",fontSize=6.7,leading=9,spaceAfter=8))
    for name,size in [("Title",20),("Heading1",15),("Heading2",12),("Heading3",10)]:
        styles[name].fontName=font;styles[name].fontSize=size
        styles[name].leading=size*1.25;styles[name].spaceBefore=12;styles[name].spaceAfter=7;styles[name].keepWithNext=True
    width=A4[0]-84
    doc=SimpleDocTemplate(str(args.output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=40,
                          title="HipMRI Segmentation - COMP3710 Final Assessment",author="RuiJie Gong")
    lines=(ROOT/"README.md").read_text(encoding="utf-8").splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith("```"):
            i+=1;code=[]
            while i<len(lines) and not lines[i].strip().startswith("```"):
                code.extend(textwrap.wrap(lines[i],width=112,replace_whitespace=False,drop_whitespace=False) or [""])
                i+=1
            story.append(KeepTogether([Preformatted("\n".join(code),styles["CodeSmall"])]));i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                row=[cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?",cell.replace(" ","")) for cell in row):
                    rows.append([Paragraph(inline(cell),styles["Cell"]) for cell in row])
                i+=1
            n=len(rows[0])
            development_table = rows[0][0].text == "Run / parent"
            tab=Table(rows,colWidths=[width/n]*n,repeatRows=1,hAlign="LEFT",splitByRow=int(development_table),splitInRow=0)
            tab.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7edf3")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#c9d0d8")),
                ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            story.extend([tab,Spacer(1,8)]);continue
        image=re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)",line)
        if image:
            path=ROOT/image[2]
            from PIL import Image as PILImage
            with PILImage.open(path) as im:w,h=im.size
            scale=min(width/w,390/h)
            story.append(KeepTogether([Image(str(path),width=w*scale,height=h*scale),
                Paragraph(inline(image[1]),styles["Cell"]),Spacer(1,10)]))
            i+=1;continue
        heading=re.match(r"^(#{1,4}) (.*)",line)
        if heading:
            if heading[2].startswith(("2. Materials and methods", "Appendix C. Feasibility")): story.append(PageBreak())
            level=len(heading[1]);name="Title" if level==1 else f"Heading{level-1}"
            slug=re.sub(r"[^\w -]","",heading[2].lower()).replace(" ","-")
            item=Paragraph(f'<a name="{slug}"/>'+inline(heading[2]),styles[name])
            item._report_heading=heading[2]
            story.append(item);i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r"^(#|\||```|!\[|\d+\.\s)",lines[i].strip()):
            paragraph.append(lines[i].strip());i+=1
        content = Paragraph(inline(" ".join(paragraph)), styles["Text"])
        if paragraph[0].startswith(("Every model in the table", "These panels use real MRI", "These diagnostics were calculated", "The documentation replay on 22 October captured")):
            content.keepWithNext = True
        story.append(KeepTogether([content]) if paragraph[0].startswith("The current nine-test suite") else content)
    # Keep a code block with its heading or introductory paragraph.
    linked=[];index=0
    while index<len(story):
        if (isinstance(story[index],Paragraph) and index+1<len(story)
                and isinstance(story[index+1],KeepTogether)
                and isinstance(story[index+1]._content[0],Preformatted)):
            linked.append(KeepTogether([story[index],*story[index+1]._content]));index+=2
        else:
            linked.append(story[index]);index+=1
    story=linked
    # Keep each clinical heading, figure and autopsy together on a page.
    def keep_block(items):
        flat=[]
        for item in items:
            flat.extend(item._content if isinstance(item,KeepTogether) else [item])
        return KeepTogether(flat)
    grouped=[];block=[];in_case=False;in_references=False
    for flow in story:
        text=getattr(flow,"_report_heading",getattr(flow,"text",""))
        is_case=bool(re.match(r"3\.6\.[1-5] ",text))
        is_next=text.startswith("4. ")
        if is_case or (in_case and is_next):
            if block:grouped.append(keep_block(block));block=[]
            in_case=is_case
        if in_references and text.startswith("Appendix A."):
            grouped.append(keep_block(block));block=[];in_references=False
        if text == "References":
            in_references=True
        if in_case or in_references:block.append(flow)
        else:grouped.append(flow)
    if block:grouped.append(keep_block(block))
    story=grouped
    def footer(canvas,doc):
        canvas.setFont("Helvetica",7)
        canvas.setFillColor(colors.HexColor("#5a6470"))
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)


if __name__=="__main__":main()
