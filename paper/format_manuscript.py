"""Rebuild the manuscript Word file in the house layout of the brain-tumour paper.

Layout copied from BRAINTUMOR_Manuscript_FORMATTED_v3.pdf: Times New Roman, centred blue
upper-case title with a rule, boxed abstract, shaded section bars, single-column justified body,
navy table headers with captions above, blue figure captions below, reference list, red-boxed author action list.

Usage: python paper/format_manuscript.py <source.docx exported from the draft> <out.docx>
Needs: python-docx.
"""
import io, re, sys, copy
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.run import Run

NAVY, NAVY_RGB = "1F4E79", RGBColor(0x1F, 0x4E, 0x79)
BAR, BAND = "EAF1F8", "F3F7FB"
RED, RED_RGB, REDBG = "C00000", RGBColor(0xC0, 0, 0), "FDECEC"
FONT = "Times New Roman"

src = Document(sys.argv[1])
out = Document()

# ------------------------------------------------------------------ page and base style
sec = out.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.9)
sec.top_margin = sec.bottom_margin = Inches(0.9)
st = out.styles["Normal"]
st.font.name = FONT; st.font.size = Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st.paragraph_format.space_after = Pt(6); st.paragraph_format.space_before = Pt(0)
st.paragraph_format.line_spacing = 1.0
st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

def set_cols(section, n):
    sp = section._sectPr
    for c in sp.findall(qn("w:cols")): sp.remove(c)
    c = OxmlElement("w:cols"); c.set(qn("w:num"), str(n)); c.set(qn("w:space"), "432")
    sp.append(c)

_cur = {"cols": None}
def begin(cols):
    """Single-column layout: every request is mapped to one column."""
    cols = 1
    if _cur["cols"] == cols: return
    if _cur["cols"] is not None: out.add_section(WD_SECTION.CONTINUOUS)
    set_cols(out.sections[-1], cols); _cur["cols"] = cols

def shade(p, fill):
    pPr = p._p.get_or_add_pPr(); s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill); pPr.append(s)

def borders(p, sides, color, sz=6, space=4):
    pPr = p._p.get_or_add_pPr(); b = OxmlElement("w:pBdr")
    for side in sides:
        e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), str(space)); e.set(qn("w:color"), color); b.append(e)
    pPr.append(b)

def style_run(r, size=None, bold=None, italic=None, color=None):
    r.font.name = FONT; r._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size: r.font.size = Pt(size)
    if bold is not None: r.bold = bold
    if italic is not None: r.italic = italic
    if color is not None: r.font.color.rgb = color

def copy_runs(dst, sp, size=10, bold=None, color=None):
    for item in sp.iter_inner_content():
        text = item.text
        if not text: continue
        r = dst.add_run(text)
        isb = getattr(item, "bold", None) if isinstance(item, Run) else None
        isi = getattr(item, "italic", None) if isinstance(item, Run) else None
        style_run(r, size, bold if bold is not None else bool(isb), bool(isi), color)

def para(text="", size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False,
         color=None, before=0, after=6, keep=False):
    p = out.add_paragraph(); p.alignment = align
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
    p.paragraph_format.keep_with_next = keep
    if text: style_run(p.add_run(text), size, bold, italic, color)
    return p

def bar(text):
    p = para(text.upper(), 11, WD_ALIGN_PARAGRAPH.LEFT, True, color=NAVY_RGB, before=10, after=6, keep=True)
    shade(p, BAR); borders(p, ["left"], NAVY, sz=36, space=6)
    p.paragraph_format.left_indent = Inches(0.08)
    return p

def caption(label_text, kind_color=NAVY_RGB, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=8, keep=False):
    """Bold-blue 'Figure N.' plus first sentence, rest regular blue."""
    m = re.match(r"((?:Figure|Table) \d+\.)\s*(.*)", label_text, flags=re.S)
    lab, body = m.group(1), m.group(2)
    p = para("", 8.5, align, before=before, after=after, keep=keep)
    style_run(p.add_run(f"{lab} {body}"), 8.5, True, False, kind_color)
    return p

def src_table(tbl, cap):
    begin(1)
    caption(cap, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=3, keep=True)
    rows = tbl.rows; n = len(rows[0].cells)
    t = out.add_table(rows=len(rows), cols=n); t.style = "Table Grid"; t.autofit = False
    width = Inches(6.7) / n
    for i, row in enumerate(rows):
        for j, cell in enumerate(row.cells):
            c = t.cell(i, j); c.width = width
            c.paragraphs[0].text = ""
            p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
            r = p.add_run(cell.text.strip())
            if i == 0:
                style_run(r, 8.5, True, False, RGBColor(255, 255, 255))
                tcPr = c._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
                s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), NAVY); tcPr.append(s)
            else:
                style_run(r, 8.5, False, False, None)
                if i % 2 == 0:
                    tcPr = c._tc.get_or_add_tcPr(); s = OxmlElement("w:shd")
                    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), BAND); tcPr.append(s)
    para("", 4, after=4)

def picture(p_src, cap, narrow=False):
    rid = p_src._p.xpath(".//a:blip/@r:embed")[0]
    blob = src.part.related_parts[rid].blob
    begin(2 if narrow else 1)
    p = para("", 10, WD_ALIGN_PARAGRAPH.CENTER, before=4, after=2, keep=True)
    p.add_run().add_picture(io.BytesIO(blob), width=Inches(4.4 if narrow else 6.3))
    caption(cap)

# ------------------------------------------------------------------ walk the source
body = list(src.element.body.iterchildren())
from docx.table import Table
from docx.text.paragraph import Paragraph

items = []
for el in body:
    tag = el.tag.split("}")[1]
    if tag == "p": items.append(("p", Paragraph(el, src)))
    elif tag == "tbl": items.append(("t", Table(el, src)))

def text_of(p): return p.text.strip()
def is_img(p): return bool(p._p.xpath(".//a:blip"))
def style_of(p): return (p.style.name if p.style is not None else "")

NARROW_FIGS = {"Figure 2."}
i, mode = 0, "front"
ref_no = act_no = 0
while i < len(items):
    kind, o = items[i]
    if kind == "t":
        # caption above: previous paragraph if it starts with Table N, else the next one
        prev = items[i - 1][1] if items[i - 1][0] == "p" else None
        nxt = items[i + 1][1] if i + 1 < len(items) and items[i + 1][0] == "p" else None
        if nxt is not None and re.match(r"Table \d+\.", text_of(nxt)): cap = text_of(nxt); items[i + 1] = ("skip", nxt)
        else: cap = text_of(prev)
        src_table(o, cap); i += 1; continue
    if kind == "skip": i += 1; continue
    p, t, sty = o, text_of(o), style_of(o)
    if not t and not is_img(p): i += 1; continue
    # captions that a table consumed from above are already printed
    if re.match(r"Table \d+\.", t) and i + 1 < len(items) and items[i + 1][0] == "t": i += 1; continue
    if is_img(p):
        cap = text_of(items[i + 1][1]); picture(p, cap, cap[:9] in NARROW_FIGS); i += 2; continue
    if re.match(r"Figure \d+\.", t): i += 1; continue
    if sty == "Heading 1":
        begin(1)
        tp = para(t.upper(), 15, WD_ALIGN_PARAGRAPH.CENTER, True, color=NAVY_RGB, before=6, after=8)
        borders(tp, ["bottom"], NAVY, sz=8, space=8); i += 1; continue
    if mode == "front" and t.startswith("Authors:"):
        para(re.sub(r"^Authors:\s*", "", t), 9.5, WD_ALIGN_PARAGRAPH.CENTER, italic=True, before=2, after=10)
        i += 1; continue
    if mode == "front" and re.match(r"(Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep)\b", t): i += 1; continue
    if sty == "Heading 2":
        low = t.lower()
        if low.startswith("author action list"):
            mode = "actions"; begin(1)
            out.add_page_break()
            hp = para("AUTHOR ACTION LIST - NOT PART OF THE MANUSCRIPT; DELETE THIS PAGE BEFORE SUBMISSION", 11,
                      WD_ALIGN_PARAGRAPH.LEFT, True, color=RED_RGB, before=0, after=8)
            shade(hp, REDBG); borders(hp, ["top", "left", "bottom", "right"], RED, sz=8, space=6)
            i += 1; continue
        if low == "abstract": mode = "abstract"; begin(1)
        elif low == "references": mode = "refs"; begin(1)
        elif mode in ("front", "abstract"): mode = "body"
        if low == "references":
            bar(t); begin(2)
        elif low == "abstract":
            bar(t)
        else:
            begin(2); bar(t)
        i += 1; continue
    if sty == "Heading 3":
        begin(2); para(t, 10, WD_ALIGN_PARAGRAPH.LEFT, True, color=NAVY_RGB, before=6, after=3, keep=True)
        i += 1; continue
    # ---- ordinary paragraphs by mode
    if mode == "abstract":
        if t.startswith(("Categories:", "Keywords:")):
            q = para("", 9.5, before=4, after=4)
            lab, _, rest = t.partition(":")
            style_run(q.add_run(lab + ":"), 9.5, True, False, NAVY_RGB); style_run(q.add_run(rest), 9.5)
            if t.startswith("Keywords:"): mode = "after_abstract"
        else:
            q = para("", 9.5, before=0, after=0)
            copy_runs(q, p, 9.5); borders(q, ["top", "left", "bottom", "right"], NAVY, sz=6, space=6)
            q.paragraph_format.left_indent = q.paragraph_format.right_indent = Inches(0.1)
            q.paragraph_format.space_after = Pt(0)
    elif mode == "refs":
        ref_no += 1
        q = para("", 8.5, WD_ALIGN_PARAGRAPH.LEFT, before=0, after=3)
        style_run(q.add_run(f"{ref_no}. "), 8.5)
        copy_runs(q, p, 8.5, bold=False)
    elif mode == "actions":
        t2 = t
        if t2.startswith("Reference checks."): i += 1; continue
        act_no += 1
        q = para("", 9.5, WD_ALIGN_PARAGRAPH.LEFT, before=0, after=4)
        q.paragraph_format.left_indent = Inches(0.3); q.paragraph_format.first_line_indent = Inches(-0.3)
        style_run(q.add_run(f"{act_no}.  "), 9.5)
        for it in p.iter_inner_content():
            tx = it.text.replace("All 17 were", "All 22 were")
            r = q.add_run(tx); style_run(r, 9.5, bool(getattr(it, "bold", None)))
    elif sty == "List Paragraph" and mode == "body":
        begin(2)
        q = para("", 10, before=0, after=4)
        q.paragraph_format.left_indent = Inches(0.2); q.paragraph_format.first_line_indent = Inches(-0.2)
        style_run(q.add_run("•  "), 10)
        copy_runs(q, p, 10)
    else:
        begin(2); q = para("", 10); copy_runs(q, p, 10)
    i += 1

out.save(sys.argv[2])
print("saved", sys.argv[2], "refs", ref_no, "actions", act_no)
