#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор PDF-отчёта v5.0 «Мокрое чернило»: от Nemotron к LLaDA2.2-mini.

Обновляет отчёт v4.0: добавлен раздел 9 с результатами v5 — собственный
wet-ink декодер (фронтир 1 токен/шаг + скользящее окно W, Kaggle COMPLETE).
Маршрут: Report (ReportLab) + обложка (HTML/Playwright).
Язык: русский. Шрифты: FreeSerif (кириллица), DejaVuSansMono (код/формулы).
"""
import hashlib
import json
import math
import os
import subprocess
import sys

PDF_SKILL_DIR = "/home/z/my-project/skills/pdf"
sys.path.insert(0, os.path.join(PDF_SKILL_DIR, "scripts"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# ---------------------------------------------------------------- fonts
FONT_DIR = "/usr/share/fonts"
pdfmetrics.registerFont(TTFont("FreeSerif", f"{FONT_DIR}/truetype/freefont/FreeSerif.ttf"))
pdfmetrics.registerFont(TTFont("FreeSerif-Bold", f"{FONT_DIR}/truetype/freefont/FreeSerifBold.ttf"))
pdfmetrics.registerFont(TTFont("FreeSerif-Italic", f"{FONT_DIR}/truetype/freefont/FreeSerifItalic.ttf"))
pdfmetrics.registerFont(TTFont("FreeSerif-BoldItalic", f"{FONT_DIR}/truetype/freefont/FreeSerifBoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("NotoSerifSC", f"{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSerifSC-Bold", f"{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSans", f"{FONT_DIR}/truetype/dejavu/DejaVuSansMono.ttf"))
registerFontFamily("FreeSerif", normal="FreeSerif", bold="FreeSerif-Bold",
                   italic="FreeSerif-Italic", boldItalic="FreeSerif-BoldItalic")
registerFontFamily("NotoSerifSC", normal="NotoSerifSC", bold="NotoSerifSC-Bold")
registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans")

from pdf import install_font_fallback  # noqa: E402
install_font_fallback()

# ---------------------------------------------------------------- palette
PAGE_BG      = colors.HexColor("#f5f8fc")
SECTION_BG   = colors.HexColor("#edf2f9")
CARD_BG      = colors.HexColor("#e4ecf5")
TABLE_STRIPE = colors.HexColor("#eef3fa")
HEADER_FILL  = colors.HexColor("#1a4a7a")
BORDER       = colors.HexColor("#c0d0e2")
ACCENT       = colors.HexColor("#2d7ab3")
ACCENT_2     = colors.HexColor("#4da8da")
TEXT_PRIMARY = colors.HexColor("#142840")
TEXT_MUTED   = colors.HexColor("#5a7a96")
TABLE_HEADER_COLOR = HEADER_FILL
TABLE_ROW_EVEN = colors.white
TABLE_ROW_ODD = TABLE_STRIPE

PAGE_W, PAGE_H = A4
MARGIN = 0.9 * inch
AVAIL_W = PAGE_W - 2 * MARGIN
OUT_DIR = "/home/z/my-project/download"
WORK_DIR = "/home/z/my-project/scripts/report_assets"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- styles
S = {}
S["body"] = ParagraphStyle("body", fontName="FreeSerif", fontSize=10.5, leading=16,
                           alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY,
                           spaceBefore=0, spaceAfter=8)
S["body_c"] = ParagraphStyle("body_c", parent=S["body"], alignment=TA_CENTER)
S["lead"] = ParagraphStyle("lead", parent=S["body"], fontSize=11, leading=17.5,
                           textColor=TEXT_PRIMARY)
S["h1"] = ParagraphStyle("h1", fontName="FreeSerif-Bold", fontSize=19, leading=24,
                         textColor=HEADER_FILL, spaceBefore=16, spaceAfter=8,
                         keepWithNext=1)
S["h2"] = ParagraphStyle("h2", fontName="FreeSerif-Bold", fontSize=14, leading=18,
                         textColor=HEADER_FILL, spaceBefore=14, spaceAfter=6,
                         keepWithNext=1)
S["h3"] = ParagraphStyle("h3", fontName="FreeSerif-Bold", fontSize=11.5, leading=15,
                         textColor=TEXT_PRIMARY, spaceBefore=10, spaceAfter=5,
                         keepWithNext=1)
S["code"] = ParagraphStyle("code", fontName="DejaVuSans", fontSize=8.4, leading=11.6,
                           alignment=TA_LEFT, textColor=TEXT_PRIMARY,
                           backColor=SECTION_BG, borderPadding=(6, 6, 6, 6),
                           spaceBefore=6, spaceAfter=6)
S["caption"] = ParagraphStyle("caption", fontName="FreeSerif", fontSize=8.5, leading=11,
                              alignment=TA_CENTER, textColor=TEXT_MUTED,
                              spaceBefore=3, spaceAfter=6)
S["quote"] = ParagraphStyle("quote", fontName="FreeSerif-Italic", fontSize=10.5,
                            leading=16, leftIndent=24, textColor=TEXT_MUTED,
                            borderPadding=0, spaceBefore=6, spaceAfter=8)
S["th"] = ParagraphStyle("th", fontName="FreeSerif-Bold", fontSize=9, leading=11.5,
                         alignment=TA_CENTER, textColor=colors.white)
S["td"] = ParagraphStyle("td", fontName="FreeSerif", fontSize=9, leading=11.5,
                         alignment=TA_LEFT, textColor=TEXT_PRIMARY)
S["td_c"] = ParagraphStyle("td_c", parent=S["td"], alignment=TA_CENTER)
S["td_code"] = ParagraphStyle("td_code", fontName="DejaVuSans", fontSize=7.8, leading=10.2,
                              alignment=TA_LEFT, textColor=TEXT_PRIMARY)
S["stat"] = ParagraphStyle("stat", fontName="FreeSerif-Bold", fontSize=19, leading=23,
                           textColor=ACCENT, alignment=TA_CENTER)
S["stat_lbl"] = ParagraphStyle("stat_lbl", fontName="FreeSerif", fontSize=8.5, leading=11,
                               textColor=TEXT_MUTED, alignment=TA_CENTER)
S["toc0"] = ParagraphStyle("toc0", fontName="FreeSerif-Bold", fontSize=11, leading=17,
                           leftIndent=16, textColor=TEXT_PRIMARY)
S["toc1"] = ParagraphStyle("toc1", fontName="FreeSerif", fontSize=9.5, leading=14.5,
                           leftIndent=34, textColor=TEXT_MUTED)


# ---------------------------------------------------------------- helpers
def P(text, style="body"):
    text = text.replace(" — ", "\u00a0— ")
    return Paragraph(text, S[style])


def add_heading(text, level=0):
    style = S["h1"] if level == 0 else (S["h2"] if level == 1 else S["h3"])
    key = "h_" + hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph(f'<a name="{key}"/>{text}', style)
    p.bookmark_name = key
    p.bookmark_level = level
    p.bookmark_text = text.replace("<b>", "").replace("</b>", "")
    p.bookmark_key = key
    return p


def h1(story, text):
    story.append(CondPageBreak(PAGE_H * 0.18))
    story.append(add_heading(text, 0))
    hr = HRFlowable(width="100%", color=ACCENT, thickness=1.4,
                    spaceBefore=0, spaceAfter=10)
    hr.keepWithNext = 1
    story.append(hr)


def h2(story, text):
    story.append(CondPageBreak(PAGE_H * 0.12))
    story.append(add_heading(text, 1))


def h3(story, text):
    story.append(CondPageBreak(PAGE_H * 0.09))
    story.append(add_heading(text, 2))


def safe_keep(elements, max_frac=0.4):
    """Связать блок (таблица+подпись) и предыдущий заголовок в одну группу.

    Вложенный KeepTogether здесь нельзя: doc-template собирает цепочку
    keepWithNext в один KeepTogether, но _ktAllow запрещает вложенные
    контейнеры — группа «заголовок+таблица» распалась бы на границе.
    Вместо этого ставим keepWithNext на все элементы, кроме последнего:
    цепочка заголовок→HR→спейсер→таблица→подпись собирается в один
    KeepTogether, который при нехватке места уходит целиком на новую
    страницу (распаковывается посимвольно, так что TOC-закладки работают).
    """
    for el in elements[:-1]:
        try:
            el.keepWithNext = 1
        except Exception:
            pass
    return list(elements)


def _nb(text):
    return str(text).replace(" — ", "\u00a0— ")


def make_table(header, rows, ratios, style_extra=None, font="td", header_font="th",
               repeat=True, align_center_cols=None):
    assert abs(sum(ratios) - 1.0) < 1e-6, ratios
    col_w = [r * AVAIL_W for r in ratios]
    align_center_cols = align_center_cols or set()
    data = [[Paragraph(f"<b>{_nb(c)}</b>", S[header_font]) for c in header]]
    for row in rows:
        cells = []
        for j, c in enumerate(row):
            st = S["td_c"] if (j in align_center_cols or font == "td_c") else (
                S["td_code"] if font == "td_code" else S[font])
            cells.append(Paragraph(_nb(c), st))
        data.append(cells)
    t = Table(data, colWidths=col_w, hAlign="CENTER",
              repeatRows=1 if repeat else 0)
    base = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        base.append(("BACKGROUND", (0, i), (-1, i),
                     TABLE_ROW_EVEN if i % 2 == 1 else TABLE_ROW_ODD))
    if style_extra:
        base.extend(style_extra)
    t.setStyle(TableStyle(base))
    return t


def add_table(story, header, rows, ratios, caption, **kw):
    sp = Spacer(1, 10)
    sp.keepWithNext = 1
    story.append(sp)
    tbl = make_table(header, rows, ratios, **kw)
    cap = Paragraph(_nb(caption), S["caption"])
    if len(rows) <= 12:
        story.extend(safe_keep([tbl, cap]))
    else:
        tbl.keepWithNext = 1
        story.append(tbl)
        story.append(cap)
    story.append(Spacer(1, 10))


def callout(story, stat, label, width=150):
    st = S["stat"] if len(stat) <= 7 else ParagraphStyle(
        "stat_sm", parent=S["stat"], fontSize=14.5, leading=18)
    box = Table([[Paragraph(f"<b>{stat}</b>", st)],
                 [Paragraph(label, S["stat_lbl"])]], colWidths=[width])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return box


def callout_row(story, items):
    n = len(items)
    gap = 10
    w = (AVAIL_W - gap * (n - 1)) / n
    cells = [callout(story, s, l, width=w - 0) for s, l in items]
    row = Table([cells], colWidths=[w] * n, hAlign="CENTER")
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(Spacer(1, 8))
    story.append(row)
    story.append(Spacer(1, 10))


def code_block(story, text):
    story.append(Paragraph(text.replace("\n", "<br/>").replace(" ", "\u00a0"), S["code"]))


# ---------------------------------------------------------------- data
V3_PATH = "/home/z/my-project/wet_ink/results/results_v3.json"
with open(V3_PATH) as f:
    V3 = json.load(f)

V4_PATH = "/home/z/my-project/wet_ink/results/results_v4.json"
with open(V4_PATH) as f:
    V4 = json.load(f)

V5_PATH = "/home/z/my-project/wet_ink/results/results_v5.json"
with open(V5_PATH) as f:
    V5 = json.load(f)

TOK_CAP = 160  # gen_length адаптивного прогона v4


def _arm_stats(a):
    det = [d for d in a.get("details", []) if "error" not in d]
    n = len(det)
    k = sum(bool(d.get("ok", d.get("passed"))) for d in det)
    p = k / max(n, 1)
    z = 1.96
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    wall = a.get("wall_seconds",
                 a.get("gen_wall_seconds", 0) + a.get("exec_wall_seconds", 0))
    return dict(
        n=n, k=k, acc=100 * p, lo=100 * max(0.0, c - h), hi=100 * min(1.0, c + h),
        nfe=sum(d.get("nfe", 0) for d in det) / max(n, 1),
        churn=sum(d.get("churn", 0) for d in det) / max(n, 1),
        spt=wall / max(n, 1),
        trunc=sum(1 for d in det if d.get("n_tokens", 0) >= TOK_CAP),
        fin=[d for d in det if d.get("n_tokens", 0) < TOK_CAP],
    )


def _mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    kk = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(kk + 1)) / 2 ** n)


V4_NOEDIT = _arm_stats(V4["gsm8k"]["noedit_b32"])
V4_STOCK = _arm_stats(V4["gsm8k"]["stock_b32"])
_m0 = {d["i"]: d for d in V4["gsm8k"]["noedit_b32"]["details"] if "error" not in d}
_m1 = {d["i"]: d for d in V4["gsm8k"]["stock_b32"]["details"] if "error" not in d}
V4_COMMON = sorted(set(_m0) & set(_m1))
_v_b = sum(1 for i in V4_COMMON if not _m0[i]["ok"] and _m1[i]["ok"])
_v_c = sum(1 for i in V4_COMMON if _m0[i]["ok"] and not _m1[i]["ok"])
V4_PAIR = dict(
    n=len(V4_COMMON), fixed=_v_b, broken=_v_c, p=_mcnemar_p(_v_b, _v_c),
    nfe0=sum(_m0[i]["nfe"] for i in V4_COMMON) / len(V4_COMMON),
    nfe1=sum(_m1[i]["nfe"] for i in V4_COMMON) / len(V4_COMMON),
    ch0=sum(_m0[i].get("churn", 0) for i in V4_COMMON) / len(V4_COMMON),
    ch1=sum(_m1[i].get("churn", 0) for i in V4_COMMON) / len(V4_COMMON),
)
V4_PAIR["nfe_ratio"] = V4_PAIR["nfe1"] / V4_PAIR["nfe0"]
V4_PAIR["ch_ratio"] = V4_PAIR["ch1"] / V4_PAIR["ch0"]
V4_FIN_ALL = V4_NOEDIT["fin"] + V4_STOCK["fin"]
V4_FIN_OK = sum(bool(d["ok"]) for d in V4_FIN_ALL)


# ---------------- v5: вычисляемая статистика (ничего не хардкодим) ----------
def _v5_arm(name):
    a = V5["gsm8k"][name]
    det = [d for d in a.get("details", []) if "error" not in d]
    n = len(det)
    k = sum(bool(d.get("ok")) for d in det)
    p = k / max(n, 1)
    z = 1.96
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    wall = a.get("wall_seconds", 0)
    return dict(
        n=n, k=k, acc=100 * p, lo=100 * max(0.0, c - h), hi=100 * min(1.0, c + h),
        nfe=sum(d.get("nfe", 0) for d in det) / max(n, 1),
        churn=sum(d.get("churn", 0) for d in det) / max(n, 1),
        rewrites=sum(d.get("rewrites", 0) for d in det) / max(n, 1),
        opened=sum(d.get("opened", 0) for d in det) / max(n, 1),
        spt=wall / max(n, 1), wall=wall,
        trunc=sum(1 for d in det if d.get("n_tokens", 0) >= TOK_CAP),
        fin=[d for d in det if d.get("n_tokens", 0) < TOK_CAP],
        det=det, cfg=a,
    )


V5_W16 = _v5_arm("wetink_w16")
V5_NE = _v5_arm("wetink_noedit_w16")
V5_W8 = _v5_arm("wetink_w8")
V5_W16_FIN_OK = sum(bool(d["ok"]) for d in V5_W16["fin"])
V5_NE_FIN_OK = sum(bool(d["ok"]) for d in V5_NE["fin"])
V5_W8_FIN_OK = sum(bool(d["ok"]) for d in V5_W8["fin"])


def _v5_pair(base_arm, v5_stats):
    """Парная статистика v5-руки против v4b-базлайна по индексу задач."""
    m0 = {d["i"]: d for d in V4["gsm8k"][base_arm]["details"] if "error" not in d}
    m1 = {d["i"]: d for d in v5_stats["det"]}
    com = sorted(set(m0) & set(m1))
    b = sum(1 for i in com if not m0[i]["ok"] and m1[i]["ok"])
    c = sum(1 for i in com if m0[i]["ok"] and not m1[i]["ok"])
    k0 = sum(m0[i]["ok"] for i in com)
    k1 = sum(m1[i]["ok"] for i in com)
    nfe0 = sum(m0[i]["nfe"] for i in com) / len(com)
    nfe1 = sum(m1[i]["nfe"] for i in com) / len(com)
    return dict(n=len(com), b=b, c=c, k0=k0, k1=k1, p=_mcnemar_p(b, c),
                nfe0=nfe0, nfe1=nfe1, ratio=nfe1 / nfe0)


V5P_STOCK = _v5_pair("stock_b32", V5_W16)
V5P_NOED = _v5_pair("noedit_b32", V5_W16)
V5P_STOCK_NE = _v5_pair("stock_b32", V5_NE)
V5P_STOCK_W8 = _v5_pair("stock_b32", V5_W8)

# внутренний контроль v5: w16 vs noedit_w16 (оба из v5, общие задачи)
_mn = {d["i"]: d for d in V5_NE["det"]}
_mw = {d["i"]: d for d in V5_W16["det"]}
_com = sorted(set(_mn) & set(_mw))
V5P_CTRL = dict(
    n=len(_com),
    b=sum(1 for i in _com if not _mn[i]["ok"] and _mw[i]["ok"]),
    c=sum(1 for i in _com if _mn[i]["ok"] and not _mw[i]["ok"]),
    kn=sum(_mn[i]["ok"] for i in _com),
    kw=sum(_mw[i]["ok"] for i in _com),
)
V5P_CTRL["p"] = _mcnemar_p(V5P_CTRL["b"], V5P_CTRL["c"])

# сегменты задач: 0–21 (где остановился контроль) и 22–44 (хвост w16)
V5_SEG_A = (0, 21)
V5_SEG_B = (22, 44)


def _seg_ok(det, lo, hi):
    seg = [d for d in det if lo <= d["i"] <= hi]
    return 100 * sum(bool(d["ok"]) for d in seg) / max(len(seg), 1), len(seg)


V5_SEG_W16_A = _seg_ok(V5_W16["det"], *V5_SEG_A)
V5_SEG_W16_B = _seg_ok(V5_W16["det"], *V5_SEG_B)
V5_SEG_NE_A = _seg_ok(V5_NE["det"], *V5_SEG_A)
_STOCK_DET = [d for d in V4["gsm8k"]["stock_b32"]["details"] if "error" not in d]
V5_SEG_STOCK_A = _seg_ok(_STOCK_DET, *V5_SEG_A)
V5_SEG_STOCK_B = _seg_ok(_STOCK_DET, *V5_SEG_B)

# телеметрия sanity/equivalence из results_v5.json
V5_SAN = V5.get("sanity", {})
V5_EQ = V5.get("equivalence", {})

# v3 paired stats
V3_STATS = V3.get("stats", {})


def v3_acc(name):
    g = V3.get("gsm8k", {}).get(name) or V3.get("humaneval", {}).get(name)
    if not g:
        return "—"
    return g.get("accuracy", g.get("pass_at_1", "—"))


def v3_nfe(name):
    g = V3.get("gsm8k", {}).get(name) or V3.get("humaneval", {}).get(name)
    if not g:
        return "—"
    return g.get("avg_nfe", "—")


def v3_pair(bench, name):
    s = V3_STATS.get(bench, {}).get("vs_stock_same_block", {}).get(name)
    if not s:
        return dict(fixed="—", broken="—", p="—", nfe_ratio="—")
    return dict(
        fixed=s.get("fixed", "—"),
        broken=s.get("broken", "—"),
        p=f"{s['mcnemar_p']:.4f}" + (" *" if s.get("mcnemar_p", 1) < 0.05 else ""),
        nfe_ratio=f"{s.get('nfe_ratio', 0):.2f}×" if s.get("nfe_ratio") else "—",
    )


# ---------------------------------------------------------------- doc template
class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, "bookmark_name"):
            level = getattr(flowable, "bookmark_level", 0)
            text = getattr(flowable, "bookmark_text", "")
            key = getattr(flowable, "bookmark_key", "")
            self.notify("TOCEntry", (level, text, self.page, key))


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("FreeSerif", 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(MARGIN, PAGE_H - 0.55 * inch,
                      "«Мокрое чернило»: от Nemotron к LLaDA2.2-mini — технический отчёт v5.0")
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.2)
    canvas.line(MARGIN, PAGE_H - 0.62 * inch, PAGE_W - MARGIN, PAGE_H - 0.62 * inch)
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 0.62 * inch, PAGE_W - MARGIN, 0.62 * inch)
    canvas.setFont("FreeSerif", 7.5)
    canvas.drawString(MARGIN, 0.44 * inch, "Asilbekov · проект Wet-Ink · ноябрь 2026")
    canvas.drawRightString(PAGE_W - MARGIN, 0.44 * inch, f"стр. {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------- content
def build_story():
    story = []

    # ---------- TOC ----------
    toc = TableOfContents()
    toc.levelStyles = [S["toc0"], S["toc1"]]
    story.append(Paragraph("<b>Содержание</b>", S["h1"]))
    story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.4, spaceAfter=12))
    story.append(toc)
    story.append(PageBreak())

    # ================================================= 1. Резюме
    h1(story, "1. Резюме")
    story.append(P(
        "Отчёт завершает второй виток проекта «мокрое чернило» (wet-ink): перенос "
        "методов ревизии открытых токенов с базы <b>Nemotron-Labs-Diffusion-3B</b> "
        "(где ревизии нет в стоке) на новую базу <b>LLaDA2.2-mini</b> (16B MoE, "
        "1.4B активированных параметров, Apache 2.0, inclusionAI 2026), где ревизия "
        "уже обучена как часть самой модели через механизм Levenshtein Editing "
        "(L-EBPO). Постановка владельца идеи однозначна: если новая база уже "
        "содержит более эффективный механизм для той же задачи, то устаревший "
        "внешний цикл отбрасывается по правилу (г) — никаких самодельных формул "
        "поверх обученного механизма.", "lead"))
    story.append(P(
        "В версии v3 на Nemotron (полный прогон, 200 задач GSM8K × 9 конфигов + "
        "164 HumanEval × 5) консервативный CoDR остался нейтральным (−0,6…+1,5 п.п., "
        "McNemar p > 0,2), тогда как агрессивный RemeDi значимо вредил (−12…−26 п.п., "
        "p < 0,001): повторилось ровно то, что предсказывала калибровка NeurIPS 2026 "
        "на коротких блоках. В версии v4 мы берём LLaDA2.2-mini со всеми её "
        "обученными формулами как новую базу и настраиваем поверх неё только те "
        "параметры, которые сама модель публикует как часть generate(): "
        "<font name='DejaVuSans' size='8.4'>block_length</font> (ширина окна wet-ink), "
        "<font name='DejaVuSans' size='8.4'>editing_threshold</font> (гейт T2T-правки), "
        "<font name='DejaVuSans' size='8.4'>max_post_steps</font> (бюджет "
        "пост-маскового уточнения). RemeDi и CoDR отброшены как дубликаты.", "lead"))
    callout_row(story, [
        ("v3 → v4 → v5", "Nemotron → LLaDA2.2-mini → свой декодер"),
        ("2×T4 + NF4", "4-бит: целая модель на каждой T4"),
        ("COMPLETE", "v4b: 10 рук · v5: 3 руки + 2 скипа"),
        ("p = 1,0", "v5 парно не отличим от стока (n=45)"),
    ])
    story.append(P(
        "Ключевая находка дословного чтения кода LLaDA2.2-mini: T2T-правка "
        "<font name='DejaVuSans' size='8.4'>t2t_index = (p0 > editing_threshold) "
        "& (old != x0)</font> — это ровно confidence-гейт, который во v1 был "
        "training-free и порождал churn, а теперь <b>обучен через L-EBPO и ограничен "
        "max_post_steps + антилуп-ресемплером</b>. Это закрытая, завершимая система: "
        "окно правки = текущий блок (ширина = block_length), число шагов ограничено, "
        "SPLIT/DELETE подавляются на финальном раунде, остаточные mask/split/delete "
        "детектируются как warning. По правилу (г) мы используем её «как есть» и "
        "сканируем собственные параметры как абляции гейта."))
    story.append(P(
        "<b>Итог v4 (раздел 8).</b> Ядро v4b (4-бит NF4, по полной модели на каждой "
        "T4, два параллельных воркера) завершилось со статусом COMPLETE: суммарная "
        "работа двух воркеров — 13,45 ч в сессии ~7 ч из 7,5 ч бюджета; исполнено "
        "10 из 12 рук (8 GSM8K + 2 HumanEval, 403 генерации). Механика порта "
        "подтверждена: sanity без остаточных спец-токенов, жадное декодирование "
        "побитово детерминировано (ids_equal=True для b32 и b64). Главный результат "
        "P0-пары (n=80+80): встроенный T2T-гейт <b>не даёт сдвига качества</b> на "
        "коротком блоке 32 — 22,5% против 23,8% у контроля (−1,2 п.п., McNemar "
        "p = 1,0; 3 исправлено / 4 сломано); экономический сигнал устойчив — "
        "экономия 15% NFE (×0,85) и ~15% времени на задачу. Абляции гейта "
        "(edit0.5/edit0.7), блок 64, урезанный пост-бюджет mps4 — всё в пределах "
        "шума; порог M2T 0.9 удваивает NFE и режет точность. Абсолютные точности "
        "занижены усечением: gen_length=160, 82–86% генераций упёрлись в лимит до "
        "\\boxed{}; среди естественно завершённых (EOS) верны 25 из 26 (96%). "
        "Вывод: штатное блочное редактирование LLaDA2.2 — ещё не wet-ink в смысле "
        "§2.1 (окно = текущий блок, правки прекращаются с его закрытием). По "
        "решению владельца от 7 ноября v4b фиксируется как базлайн, а v5 "
        "(freedomform/wet-ink-v5, собственный декодер: фронтир по одному токену за "
        "шаг, скользящее окно W за фронтиром) реализует идею §2.1 в точности."))
    story.append(P(
        "<b>Итог v5 (раздел 9).</b> Ядро freedomform/wet-ink-v5 завершилось со "
        "статусом <b>COMPLETE</b> (рабочий цикл 6 ч 04 м из бюджета 7,5 ч): 77 "
        "генераций в трёх руках — wetink_w16 (гипотеза, n=45), "
        "wetink_noedit_w16 (контроль: фронтир без правки, n=22), wetink_w8 "
        "(узкое окно, n=10); руки w32 и HumanEval пропущены планировщиком честно "
        "(доступного бюджета хватало лишь на n=5 &lt; минимальных 8 задач — «skipped: "
        "time budget»). Главный результат: <b>собственный wet-ink декодер парно "
        "неотличим от штатного параллельного</b> — на 45 общих задачах GSM8K обе "
        "руки решили ровно по 11 (McNemar p = 1,0; 2 исправлено / 2 сломано) при "
        "честной цене ×4,88 NFE (42 → 207 на задачу). Второй результат — "
        "методологический: видимые +10,8 п.п. к внутреннему контролю целиком "
        "разложились на состав задач (контроль остановился на первых 22, более "
        "трудных: на них все руки дают 13,6%, на задачах 22–44 — 34,8%); на "
        "одних и тех же задачах правки в окне (6,9 реврайта на задачу, churn 5,6) "
        "<b>не перевернули ни одного исхода</b> — 0 расхождений из 22 пар. Механика "
        "подтверждена: детерминизм (ids_equal=True), ноль остаточных спец-токенов, "
        "телеметрия фронтира/реврайтов работает. Вердикт и следующий шаг — §9.5."))

    # ================================================= 2. Идея и правила
    h1(story, "2. Идея и правила проекта")
    h2(story, "2.1. Идея «мокрого чернила»")
    story.append(P(
        "Диффузионная модель генерирует последовательность слева направо, открывая "
        "по одному токену за шаг диффузии (фронтир); уже открытые токены позади "
        "фронтира остаются поправимыми, пока фронтир пишет следующий блок. По мере "
        "удаления от фронтира «чернила высыхают», и токены фиксируются. Схема "
        "образует непрерывный спектр между чистым AR-декодированием (окно нулевой "
        "ширины) и полнопоследовательностной диффузией (окно на всю длину). "
        "Практический смысл: токен, открытый при бедном контексте, может оказаться "
        "ошибочным после того, как правые соседи дописаны; без ревизии такая "
        "ошибка необратима."))
    h2(story, "2.2. Правила, заданные владельцем идеи")
    for t in [
        "(а) полностью изучить формулы базовой модели по первоисточникам;",
        "(б) проверить все ограничения, которые налагает запрет на переобучение;",
        "(в) выбрать оптимальный путь реализации и найти самые эффективные формулы, "
        "а не самые простые;",
        "(г) запрещено придумывать собственные формулы; <b>комбинировать "
        "опубликованные формулы и окружающий код из любых архитектур и моделей "
        "разрешено после проверки совместимости</b>; цель — максимально эффективная "
        "реализация идеи wet-ink; если новая база уже содержит обученный механизм "
        "для нашей задачи, устаревший внешний цикл отбрасывается.",
    ]:
        story.append(P(t, "body"))
    story.append(P(
        "Расширение правила (г), явно сформулированное владельцем идеи при смене "
        "базы: «если раньше у нас была формула маскировки неуверенных прошлых "
        "ответов и открытия на следующем шаге диффузии, а теперь в новой базе "
        "уже есть определённый механизм для этого и он лучше — мы берём новую "
        "базу со всеми её формулами, а старую отбрасываем как менее эффективную». "
        "Это определило дизайн v4. Редакция от 7 ноября 2026 года добавила два "
        "уточнения, определившие v5: во-первых, заимствование окружающего кода из "
        "любых архитектур разрешено на тех же условиях — после проверки "
        "совместимости; во-вторых, в самом декодере не должно быть самодельных "
        "формул — только формулы самой модели плюс наше расписание открытия "
        "токенов (фронтир и скользящее окно). v4 это правило выполняет тривиально "
        "(используется только штатный generate()); v5 — уже нетривиально: "
        "собственный цикл декодирования, но каждая формула внутри — из самой "
        "модели или из опубликованных источников."))

    # ================================================= 3. Прошлая база (Nemotron) — что было сделано в v3
    h1(story, "3. Прошлая база: Nemotron-Labs-Diffusion-3B")
    story.append(P(
        "В v1/v2/v3 базой служила <b>nvidia/Nemotron-Labs-Diffusion-3B</b>. "
        "Стоковый generate() замораживает открытые токены (confidence = −inf на "
        "открытых позициях), поэтому ревизию приходилось добавлять внешними "
        "циклами. v3 расширила протокол v2 до 200 задач GSM8K × {32, 64, 128} и "
        "164 задач HumanEval × {32, 64}, и проверила две опубликованные политики — "
        "RemeDi (ICLR 2026) и CoDR (EMNLP 2026) — против стока. Эквивалентность "
        "W=0 ≡ сток подтверждена на реальной модели для всех политик:"))
    code_block(story,
               "СТОК Nemotron:  confidence = where(masked, x0_p, -inf)\n"
               "                -> открытые токены НЕ перепредсказываются\n"
               "v3 RemeDi:      keep = topk(score, k=cumsum[i])\n"
               "                x = where(keep, x0, ())  — всё вне top-k в маску\n"
               "v3 CoDR:        после блока: probe -> drift = old_conf - new_conf > tau -> remask")
    story.append(P(
        "Прошлая база не имела встроенной ревизии — это и была «дыра», которую v1 "
        "закрыла training-free, а v2/v3 — двумя полными опубликованными циклами. "
        "Все формулы воспроизведены дословно из официальных репозиториев, провенанс "
        "документирован в коде и README."))

    # ================================================= 4. v3: результаты на Nemotron
    h1(story, "4. Результаты v3 на Nemotron (200+164 задач)")
    h2(story, "4.1. GSM8K: точность и NFE")
    rows = []
    for name in ["stock_b32", "codr_w1_b32", "remedi_w1_s32_b32",
                 "stock_b64", "codr_w1_b64", "remedi_w1_s64_b64",
                 "stock_b128", "codr_w1_b128", "remedi_w1_s128_b128",
                 "stock_b32_t0.8"]:
        g = V3.get("gsm8k", {}).get(name)
        if not g or "accuracy" not in g:
            continue
        bl = g.get("block_length", 32)
        rows.append([
            name,
            f"{g['accuracy']:.1f}",
            f"{g['avg_nfe']:.1f}",
            f"{g['avg_tokens']:.0f}",
            str(bl),
            str(g.get("attempted", 200)),
        ])
    add_table(story,
              ["Конфигурация", "Точн., %", "Ср. NFE", "Ср. ток.", "block", "n"],
              rows,
              [0.30, 0.13, 0.13, 0.13, 0.13, 0.18],
              "Таблица 1. GSM8K v3 (Nemotron), n=200, протокол: инструкция + chat-template "
              "+ \\boxed{} или последнее число. Порог стока 0.9; RemeDi-варианты с 32/64/128 "
              "шагами denoise. McNemar — в таблице 2.",
              font="td", align_center_cols={1, 2, 3, 4, 5})

    h2(story, "4.2. Парный McNemar против стока (тот же блок)")
    rows = []
    for name in ["codr_w1_b32", "remedi_w1_s32_b32",
                 "codr_w1_b64", "remedi_w1_s64_b64",
                 "codr_w1_b128", "remedi_w1_s128_b128"]:
        s = v3_pair("gsm8k", name)
        rows.append([name, s["fixed"], s["broken"], s["p"], s["nfe_ratio"]])
    add_table(story,
              ["Конфиг vs сток (блок)", "исправлено", "сломано", "McNemar p", "NFE ×сток"],
              rows, [0.36, 0.16, 0.16, 0.16, 0.16],
              "Таблица 2. Парный точный McNemar на одинаковых задачах (n=200); "
              "(*) p < 0,05. CoDR нейтрален; RemeDi значимо вредит.",
              font="td", align_center_cols={1, 2, 3, 4})

    story.append(P(
        "<b>Чтение.</b> CoDR на b64 дал +1,5 п.п. (7/4, p=0,55 — не значимо), на b32 и "
        "b128 — около нуля. RemeDi систематически и значимо вредил: −12,5 п.п. "
        "(b32, p=0,0002), −15,5 п.п. (b64, p<0,0001), −26,0 п.п. (b128, p<0,0001), "
        "при цене ×2,97–3,19 NFE. Этот паттерн точно соответствует калибровке "
        "NeurIPS 2026: training-free ревизия на коротких блоках при жадном "
        "декодировании даёт нулевой или отрицательный эффект, а агрессивное "
        "разжижение окна размазывает локальную связность текста. Сигнал ревизии "
        "при этом реален — фиксированные задачи есть во всех конфигах, но "
        "ломается больше."))
    story.append(Spacer(1, 6))
    img = Image(os.path.join(WORK_DIR, "chart_v3_gsm8k.png"),
                width=AVAIL_W, height=AVAIL_W * 5.2 / 7.6)
    img.keepWithNext = 1
    story.append(img)
    story.append(Paragraph(
        "Рис. 1. Точность (вверху) и NFE на задачу (внизу, лог-шкала) для всех "
        "GSM8K-конфигов v3. Сток (тёмно-синий) — стабильный базовик; CoDR "
        "(голубой) — нейтрален; RemeDi (серо-синий) — значимо вредит.", S["caption"]))
    story.append(Spacer(1, 8))

    h2(story, "4.3. HumanEval")
    rows = []
    for name in ["he_stock_b32", "he_codr_w1_b32", "he_remedi_w1_s32_b32",
                 "he_stock_b64", "he_codr_w1_b64"]:
        g = V3.get("humaneval", {}).get(name)
        if not g or "pass_at_1" not in g:
            continue
        rows.append([
            name,
            f"{g['pass_at_1']:.1f}",
            f"{g['avg_nfe']:.1f}",
            f"{g['avg_tokens']:.0f}",
            str(g.get("block_length", 32)),
            str(g.get("attempted", 164)),
        ])
    add_table(story,
              ["Конфигурация", "pass@1, %", "Ср. NFE", "Ср. ток.", "block", "n"],
              rows, [0.34, 0.13, 0.13, 0.13, 0.13, 0.14],
              "Таблица 3. HumanEval v3 (Nemotron), n=164. Протокол Saber (ACL 2026); "
              "выполнение — канонический OpenAI HumanEval (subprocess, 10 c, pass = exit 0).",
              font="td", align_center_cols={1, 2, 3, 4, 5})
    rows = []
    for name in ["he_codr_w1_b32", "he_remedi_w1_s32_b32", "he_codr_w1_b64"]:
        s = v3_pair("humaneval", name)
        rows.append([name, s["fixed"], s["broken"], s["p"], s["nfe_ratio"]])
    add_table(story,
              ["Конфиг vs сток (блок)", "исправлено", "сломано", "McNemar p", "NFE ×сток"],
              rows, [0.36, 0.16, 0.16, 0.16, 0.16],
              "Таблица 4. Парный McNemar на HumanEval. RemeDi значимо вредит и здесь "
              "(−19,5 п.п., p<0,0001); CoDR нейтрален.",
              font="td", align_center_cols={1, 2, 3, 4})
    story.append(P(
        "<b>Итог v3 по Nemotron.</b> Сдвига парадигмы не произошло: на коротких блоках "
        "training-free ревизия либо нейтральна (CoDR), либо значимо вредна (RemeDi). "
        "Контрольный сток b32 с t=0.8 (81% против 86,5% у t=0.9) подтвердил, что "
        "порог M2T 0.9 — разумный компромисс; понижение до 0.8 без лицензии на "
        "правку теряет качество. Это и стало мотивацией смены базы: нужен был "
        "механизм, в котором ревизия <b>обучена</b>, а не добавлена training-free "
        "поверх чужой модели. Следующий раздел описывает такую базу."))

    # ================================================= 5. Новая база: LLaDA2.2-mini — что уже есть
    h1(story, "5. Новая база: LLaDA2.2-mini")
    story.append(P(
        "Базой v4 служит <b>inclusionAI/LLaDA2.2-mini</b> (выпуск 2026, Apache 2.0): "
        "MoE-диффузионная LM, 16B всего / 1.4B активированных (256 экспертов × 8 "
        "активированы), 20 слоёв, RoPE, vocab 157K, контекст 128K. Ключевая "
        "новинка серии 2.2 — <b>Levenshtein Editing</b>: специальные токены "
        "<font name='DejaVuSans' size='8.4'>DELETE</font> и "
        "<font name='DejaVuSans' size='8.4'>SPLIT</font>, обученные через L-EBPO "
        "(Levenshtein Editing via Block-wise Policy Optimization). Это в точности "
        "механизм «мокрого чернила», обученный производителем как часть самой "
        "модели. Все факты этого раздела установлены дословным чтением "
        "<font name='DejaVuSans' size='8.4'>modeling_llada2_moe.py</font> (76 КБ, "
        "1 745 строк) и <font name='DejaVuSans' size='8.4'>config.json</font>."))
    h2(story, "5.1. Стоковый generate(): встроенный wet-ink")
    code_block(story,
               "block_length     = 32     # ширина окна wet-ink (по умолчанию)\n"
               "steps            = 32     # число денойз-шагов = block_length\n"
               "threshold        = 0.5    # M2T: p_s > threshold -> unmask\n"
               "editing_threshold = 0.0   # T2T: p0 > editing_threshold & old != x0 -> rewrite\n"
               "max_post_steps   = 16     # пост-масковый бюджет до завершения\n"
               "anti-loop resampler           # защита от зацикливаний (resample changed pos)\n"
               "SPLIT/DELETE suppression      # на финальном раунде -> гарантия терминции")
    story.append(P(
        "Семантика дословно из <font name='DejaVuSans' size='8.4'>_joint_decode_block</font>: "
        "на каждом шаге блок получает M2T (маска → токен с p_s > threshold) и "
        "<b>T2T</b> (токен → токен, если greedy p0 > editing_threshold и "
        "greedy-токен отличается). Это и есть confidence-гейт, который в v1 был "
        "training-free и порождал churn; здесь он обучен и ограничен. Окно правки "
        "по построению равно текущему блоку, что точно соответствует идее «мокрого "
        "чернила». Завершимость гарантирована: фиксированное расписание transfer "
        "schedule, max_steps_per_block, suppression SPLIT/DELETE на финальном "
        "раунде, детектор остаточных mask/split/delete как warning."))
    h2(story, "5.2. Правило (г) в применении к v4: что отбрасывается")
    add_table(story,
              ["Что было в v3", "Что есть в LLaDA2.2-mini", "Решение v4"],
              [
                  ["RemeDi (cumulative top-k re-masking всего окна)",
                   "T2T: greedy-перезапись по p0 > editing_threshold (обучено)",
                   "<b>ОТБРОСИТЬ</b> (дубликат, менее эффективен)"],
                  ["CoDR (post-block drift-detection + refill)",
                   "T2T: тот же механизм реагирует на drift (old != x0)",
                   "<b>ОТБРОСИТЬ</b> (дубликат, без обучения)"],
                  ["Идея wet-ink: окно = block_length",
                   "block_length — штатный параметр generate()",
                   "сохранить; сканировать {32, 64, 128}"],
                  ["McNemar + Wilson CI + NFE-учёт",
                   "(не относится к базе; методология)",
                   "сохранить из v3 без изменений"],
                  ["GSM8K n=200, HumanEval 164 (протоколы)",
                   "(не относится к базе)",
                   "сохранить из v3 для прямой сравнимости"],
              ],
              [0.32, 0.36, 0.32],
              "Таблица 5. Что переносим, что отбрасываем при смене базы. Никаких новых "
              "формул: только конфигурация штатных параметров LLaDA2.",
              font="td")
    story.append(P(
        "Иначе говоря: «наши методы поверх базы» в v4 — это не новые формулы, а "
        "<b>конфигурация собственных параметров LLaDA2.2-mini</b> (README «Best "
        "Practices» + generation_config defaults): block_length, editing_threshold, "
        "max_post_steps, threshold. Это полностью отвечает правилу (г) — мы "
        "используем формулы самой модели, не добавляем своих."))

    # ================================================= 6. Дизайн v4
    h1(story, "6. Дизайн v4")
    h2(story, "6.1. Сетка конфигураций")
    add_table(story,
              ["Приоритет", "Конфиг", "block_length", "editing_threshold", "max_post_steps", "threshold M2T"],
              [
                  ["P0", "noedit_b32", "32", "2.0 (OFF)", "16", "0.5"],
                  ["P0", "stock_b32", "32", "0.0 (default)", "16", "0.5"],
                  ["P0", "he_noedit_b32 / he_stock_b32", "32", "2.0 / 0.0", "16", "0.5"],
                  ["P1", "edit0.5_b32", "32", "0.5", "16", "0.5"],
                  ["P1", "edit0.7_b32", "32", "0.7", "16", "0.5"],
                  ["P2", "noedit_b64 / stock_b64 / edit0.7_b64", "64", "—", "16", "0.5"],
                  ["P2", "he_noedit_b64 / he_stock_b64", "64", "—", "16", "0.5"],
                  ["P2", "noedit_b128 / stock_b128", "128", "—", "16", "0.5"],
                  ["P3", "mps4_b32 (max_post_steps=4)", "32", "0.0", "4", "0.5"],
                  ["P4", "stock_b32_t0.9 (v3-control)", "32", "0.0", "16", "0.9"],
              ],
              [0.08, 0.30, 0.15, 0.20, 0.13, 0.14],
              "Таблица 6. Сетка v4. noedit = T2T выключен (editing_threshold=2.0, "
              "недостижимое значение p0 ∈ [0,1]) — контроль «база без правки». "
              "stock = editing_threshold=0.0 (default) — встроенный wet-ink. "
              "edit0.5/0.7 — абляции гейта.",
              font="td", align_center_cols={0, 2, 3, 4, 5})
    story.append(P(
        "Логика приоритетов: P0 — главное сравнение «с правкой vs без правки» "
        "на коротком блоке 32; P1 — где на шкале [0,1] сидит обученный гейт; "
        "P2 — длинные блоки 64/128, где, по калибровке NeurIPS 2026, ревизия "
        "должна начать выигрывать; P3 — speed-вариант с урезанным пост-бюджетом; "
        "P4 — порог M2T=0.9 (как в v3) для прямой сравнимости с прошлой базой. "
        "Фактический прогон (раздел 8.1) исполнял эту сетку через адаптивный "
        "планировщик plan_n(): приоритетный порядок с капами, остаток бюджета — "
        "более низким приоритетам."))
    h2(story, "6.2. Гарантия завершимости")
    story.append(P(
        "Все гарантии — штатные LLaDA2: (1) transfer_schedule — фиксированное "
        "расписание из _get_num_transfer_tokens (чётное разбиение, остаток "
        "вперёд); (2) max_steps_per_block = 1000 (модельный default); (3) на "
        "финальном раунде SPLIT/DELETE подавляются, а все оставшиеся маски "
        "разрешаются; (4) антилуп-ресемплер выбивает зацикливания до финального "
        "раунда; (5) residual-special-token детектор выводит warning, если "
        "что-то прорвалось. Никаких собственных лимитов, хвостов или хардкапов "
        "v4 не добавляет — только обёртка подсчёта NFE/churn поверх "
        "model.forward() (только наблюдение, не изменение математики)."))
    h2(story, "6.3. 2×Tesla T4: от int8-пайплайна к 4-бит NF4")
    story.append(P(
        "16B параметров в bf16 = 32 ГБ → на одну T4 (16 ГБ) не помещается. Первый "
        "подход (v4a, 6 ноября) — 8-битная квантизация с раскладкой слоёв по двум "
        "GPU (device_map=auto): сошлось, но медленно — ~3,8–4,2 с на forward, "
        "220 с/задача: конвейер через две GPU узким местом делает пересылки "
        "активаций. v4a успел выполнить только P0-пару (noedit 20,0% n=75; stock "
        "22,4% n=49) и сохранился как референс робастности к квантизации. "
        "Финальный прогон v4b пошёл другим путём:"))
    code_block(story,
               "BitsAndBytesConfig(load_in_4bit=True,\n"
               "    bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,\n"
               "    bnb_4bit_compute_dtype=torch.float16,\n"
               "    llm_int8_skip_modules=['lm_head'])   # 157K vocab -> fp16\n"
               "AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb,\n"
               "    device_map={'': gpu},                # ЦЕЛАЯ модель на ОДНОЙ T4\n"
               "    torch_dtype=torch.float16,            # у T4 нет bf16\n"
               "    attn_implementation='eager')          # SDPA отвергает bf16-маску\n"
               "# -> MoE 16.3B = ~9.4 GiB NF4, помещается в 15.8 GiB T4\n"
               "# -> два независимых воркера: по модели на каждой GPU, очередь задач")
    story.append(P(
        "Итог: forward подешевел до ~2,9 с (−25–30% против int8-конвейера), "
        "задача GSM8K — 111–143 с, и два воркера работают параллельно без "
        "пересылок: суммарно 13,45 ч полезной работы за сессию ~7 ч. Проверки "
        "совместимости квантизации с LLaDA2 (всё дословно из кода модели): "
        "(а) MoE-эксперты — обычные "
        "<font name='DejaVuSans' size='8.4'>nn.Linear</font>, bnb их квантует; "
        "(б) роутер — сырой "
        "<font name='DejaVuSans' size='8.4'>nn.Parameter + F.linear</font> в fp32, "
        "bnb его не трогает, skip-лист не нужен; (в) RMSNorm апкастит в fp32 и "
        "возвращает входной dtype (fp16); (г) eager attention мягко складывает "
        "fp16+bf16 с промоушеном в fp32 через softmax(..., dtype=torch.float32), "
        "поэтому bf16 edit-маска работает на T4 без шимов. Все проверки выполнены "
        "локально на tiny-модели той же архитектуры."))

    # ================================================= 7. Верификация
    h1(story, "7. Локальная верификация v4")
    add_table(story,
              ["Проверка", "Результат"],
              [
                  ["bf16-маска + fp16/fp32 logits: сложение промоутится в fp32", "OK (T4 path)"],
                  ["Квантизация int8 (v4a) и NF4 (v4b) + MoE-роутер: роутер в fp32 автоматически", "OK"],
                  ["eager attention принимает bf16-маску (vs SDPA: rejects)", "OK"],
                  ["Все конфиги arm-сетки (12 рук v4b) завершаются на tiny-модели", "OK (0 residual special)"],
                  ["Жадное декодирование детерминировано (двойной прогон ==)", "OK (ids_equal=True)"],
                  ["NFE/churn-обёртка: только наблюдение, не меняет математику", "OK"],
                  ["Prompt длины 37 и 64 (не кратные block_size): terminate", "OK"],
                  ["AST ядра v4 проходит, ключевые символы на месте", "OK"],
              ],
              [0.62, 0.38],
              "Таблица 7. Локальная проверка механики v4 на tiny-модели "
              "(та же архитектура LLaDA2MoeModelLM, 20,3M параметров, случайные веса). "
              "Тест только механический — о качестве говорят только Kaggle-результаты.",
              font="td", align_center_cols={1})

    # ================================================= 8. Результаты v4
    h1(story, "8. Результаты v4b (Kaggle: 2×T4, 4-бит NF4)")
    story.append(P(
        "Ядро <b>kaggle: freedomform/wet-ink-llada22-mini</b> (версия 5) завершено "
        "со статусом <b>COMPLETE</b>: старт 7 ноября ~03:17 UTC, финальный дамп "
        "10:14 UTC — сессия ~7 ч из 7,5 ч бюджета. Конфигурация: 4-бит NF4 "
        "(double-quant, fp16 compute), <b>по полной модели на каждой T4</b>, два "
        "независимых воркера с приоритетной очередью рук — суммарно 13,45 ч "
        "полезной работы. Исполнено <b>10 из 12 рук</b> (8 GSM8K + 2 HumanEval, "
        "403 генерации); блоки 128 отложены планировщиком по бюджету. Артефакт — "
        "<font name='DejaVuSans' size='8.4'>results_v4.json</font>: sanity, "
        "проверка детерминизма, детали каждой руки на задачу (NFE, churn, токены, "
        "префикс вывода), парная статистика и бюджетный учёт; инкрементальные "
        "дампы — каждые 10 задач."))

    h2(story, "8.1. Выполнение: приоритетная очередь + адаптивный планировщик")
    story.append(P(
        "Перед стартом рук — sanity и проверка эквивалентности. Sanity (задача "
        "про клипы): чистый markdown-вывод с LaTeX-блоками, nfe=27, churn=30, "
        "<b>0 остаточных спец-токенов</b> (mask/DELETE/SPLIT), 20 с. "
        "Эквивалентность: жадное декодирование побитово детерминировано — "
        "ids_equal=True для b32 (nfe 21=21, churn 29=29) и b64 (nfe 18=18, "
        "churn 27=27). Затем воркеры разбирали руки из приоритетной очереди; "
        "планировщик plan_n() пересчитывал допустимые n по остатку бюджета и "
        "фактической скорости каждой руки:"))
    add_table(story,
              ["Рука", "Приоритет", "n", "Часы", "с/задача", "Итог"],
              [
                  ["noedit_b32 (контроль)", "P0", "80", "3,19", "143",
                   "исполнено полностью"],
                  ["stock_b32 (T2T ON)", "P0", "80", "2,70", "121",
                   "исполнено полностью"],
                  ["he_noedit_b32 / he_stock_b32", "P0", "23+20", "0,37+0,27", "57/49",
                   "исполнено"],
                  ["edit0.7_b32", "P1", "60", "2,07", "124", "кап по бюджету"],
                  ["edit0.5_b32", "P1", "58", "1,91", "118", "кап по бюджету"],
                  ["noedit_b64 / stock_b64", "P2", "29+26", "1,07+0,80", "133/111",
                   "кап по бюджету"],
                  ["mps4_b32", "P3", "14", "0,48", "124", "остаток бюджета"],
                  ["stock_b32_t0.9", "P4", "13", "0,59", "165", "остаток бюджета"],
                  ["noedit_b128 / stock_b128", "P2", "0", "—", "—",
                   "SKIPPED: plan_n()=0 при остатке"],
              ],
              [0.26, 0.10, 0.09, 0.12, 0.11, 0.32],
              "Таблица 8. Исполнение рук v4b: приоритетная очередь, два воркера "
              "(по модели на каждой T4), суммарно 13,45 ч. Капы P0 гарантируют "
              "главную пару; инкрементальные дампы после каждых 10 задач.",
              font="td", align_center_cols={1, 2, 3, 4})
    story.append(P(
        "Против прошлого прогона (v4a, int8-конвейер) скорость выросла вдвое за "
        "счёт отказа от пересылок активаций между GPU: 2,86 с/NFE против ~4 с, "
        "121–143 с/задача против 186–220 с. Два независимых воркера удвоили "
        "пропускную способность: в тот же 7,5-часовой бюджет уместилось 403 "
        "генерации против 124 у v4a. Адаптивный планировщик отработал как "
        "задуман: защитил P0-пару гарантированными капами 80+80, дозировал "
        "P1/P2 по остатку и чисто пропустил блоки 128, сохранив все измерения."))

    h2(story, "8.2. GSM8K: все руки (блок 32 и 64)")
    _order = ["noedit_b32", "stock_b32", "edit0.5_b32", "edit0.7_b32",
              "noedit_b64", "stock_b64", "mps4_b32", "stock_b32_t0.9"]
    _stats = {k: _arm_stats(V4["gsm8k"][k]) for k in _order}
    rows = []
    for name in _order:
        a = _stats[name]
        arm = V4["gsm8k"][name]
        rows.append([
            name,
            f"{arm.get('editing_threshold', 0.0):.1f}" if name != "stock_b32_t0.9" else "0.0 (t=0.9)",
            str(arm.get("block_length", 32)),
            str(a["n"]), f"{a['acc']:.1f}",
            f"[{a['lo']:.1f}; {a['hi']:.1f}]",
            f"{a['nfe']:.1f}", f"{a['churn']:.2f}", f"{a['spt']:.0f}",
        ])
    add_table(story,
              ["Рука", "edit_thr", "block", "n", "Точн., %", "Wilson 95%",
               "Ср. NFE", "Ср. churn", "с/зад"],
              rows,
              [0.20, 0.10, 0.07, 0.06, 0.10, 0.17, 0.10, 0.10, 0.10],
              "Таблица 9. Все руки GSM8K v4b: greedy (temperature=0), "
              "gen_length=160, threshold M2T=0.5 (кроме t0.9), max_post_steps=16 "
              "(кроме mps4=4). Только штатные параметры generate() LLaDA2.2-mini.",
              font="td", align_center_cols={1, 2, 3, 4, 5, 6, 7, 8})

    def _pairstats(ctrl, arm):
        m0 = {d["i"]: d for d in V4["gsm8k"][ctrl]["details"] if "error" not in d}
        m1 = {d["i"]: d for d in V4["gsm8k"][arm]["details"] if "error" not in d}
        com = sorted(set(m0) & set(m1))
        b = sum(1 for i in com if not m0[i]["ok"] and m1[i]["ok"])
        c = sum(1 for i in com if m0[i]["ok"] and not m1[i]["ok"])
        n0 = sum(m0[i]["nfe"] for i in com) / len(com)
        n1 = sum(m1[i]["nfe"] for i in com) / len(com)
        c0 = sum(m0[i].get("churn", 0) for i in com) / len(com)
        c1 = sum(m1[i].get("churn", 0) for i in com) / len(com)
        return (len(com), b, c, _mcnemar_p(b, c), n0, n1, n0 and n1 / n0, c0, c1)

    rows = []
    for arm, ctrl, lab in [
            ("stock_b32", "noedit_b32", "T2T ON (default)"),
            ("edit0.5_b32", "noedit_b32", "T2T @ 0.5"),
            ("edit0.7_b32", "noedit_b32", "T2T @ 0.7"),
            ("mps4_b32", "noedit_b32", "T2T ON, пост-бюджет 4"),
            ("stock_b32_t0.9", "noedit_b32", "T2T ON, M2T @ 0.9"),
            ("stock_b64", "noedit_b64", "T2T ON, блок 64")]:
        n_, b_, c_, p_, n0, n1, r_, c0, c1 = _pairstats(ctrl, arm)
        rows.append([
            f"{arm} ({lab})", f"{ctrl}", str(n_), f"{b_}", f"{c_}",
            f"{p_:.4f}", f"×{r_:.2f} ({n0:.1f}→{n1:.1f})",
            f"×{c1 / c0:.1f} ({c0:.2f}→{c1:.2f})",
        ])
    add_table(story,
              ["Рука (vs контроль)", "контроль", "n пар", "испр.", "слом.",
               "McNemar p", "NFE", "churn"],
              rows,
              [0.21, 0.12, 0.07, 0.07, 0.07, 0.11, 0.19, 0.16],
              "Таблица 10. Парный точный McNemar против noedit того же блока; NFE "
              "и churn — средние по одним и тем же задачам. Ни одного значимого "
              "сдвига качества (все p ≥ 0,25).",
              font="td", align_center_cols={2, 3, 4, 5, 6, 7})
    story.append(P(
        "<b>Чтение таблицы 10.</b> Ни одна конфигурация правки не сдвинула "
        "качество: главный кандидат stock_b32 дал 3 исправлено / 4 сломано "
        "(−1,2 п.п., p = 1,0 на 80 парах), абляции гейта — 2/3 и 0/3. Единственная "
        "рука с видимым трендом — порог M2T 0.9: −8,4 п.п. (15,4%) при удвоении "
        "NFE (×1,93), но n=13 мало для значимости. Длинный блок 64 у стока "
        "выглядит лучше (26,9%, +4,4 п.п. к stock_b32), однако интервалы Wilson "
        "перекрываются и парных расхождений ноль (0/0)."))
    story.append(Spacer(1, 6))
    img = Image(os.path.join(WORK_DIR, "chart_v4_gsm8k.png"),
                width=AVAIL_W, height=AVAIL_W * 5.8 / 7.6)
    img.keepWithNext = 1
    story.append(img)
    story.append(Paragraph(
        "Рис. 2. Результаты v4b. Слева сверху — точность всех 8 рук GSM8K с "
        "Wilson 95% CI; справа сверху — парные NFE P0-пары (×0,85); слева снизу — "
        "парный churn (×10,5); справа снизу — эффект лимита gen_length=160: "
        "82–85% генераций усечены, но среди естественно завершённых (EOS) верны "
        "93–100%.", S["caption"]))
    story.append(Spacer(1, 8))

    h2(story, "8.3. Конфаунд усечения: gen_length=160")
    story.append(P(
        "Адаптивный планировщик сжал gen_length со штатных 256 до 160 токенов, "
        "чтобы уместить больше задач. На LLaDA2.2 это привело к систематическому "
        "усечению: модель пишет развёрнутые markdown-решения (заголовки и "
        "LaTeX-блоки), которые не помещаются в 160 токенов. Усечены "
        f"{V4_NOEDIT['trunc']} из {V4_NOEDIT['n']} генераций контроля (82%) и "
        f"{V4_STOCK['trunc']} из {V4_STOCK['n']} у стока (85%); по остальным рукам "
        "76–86%. Практически ни одна усечённая генерация не дошла до "
        "\\boxed{} (0–1 на руку), и экстрактор брал «последнее число» из середины "
        "решения. Следствие: абсолютные 20–27% — это мера усечения, а не "
        "способности модели, и напрямую несравнимы с v3 (86,5% — другая база, "
        "256 токенов на решение, другой стиль). Ключевой контроль: среди "
        f"{len(V4_FIN_ALL)} естественно завершённых (EOS) генераций P0-пары "
        f"верны {V4_FIN_OK} из {len(V4_FIN_ALL)} — 96% (13/14 у контроля, 12/12 "
        "у стока); по всем рукам доля верных среди EOS — 86–100%. Способность "
        "базы в порядке; выводы раздела 8.2 — о методе, а не о модели."))

    h2(story, "8.4. HumanEval (блок 32)")
    he_rows = []
    for name, lab in [("he_noedit_b32", "контроль (T2T OFF)"),
                      ("he_stock_b32", "сток (T2T ON)")]:
        a = V4["humaneval"][name]
        det = [d for d in a["details"] if "error" not in d]
        n = len(det)
        k = sum(bool(d.get("passed")) for d in det)
        _p = k / n
        _z = 1.96
        _dd = 1 + _z * _z / n
        _c = (_p + _z * _z / (2 * n)) / _dd
        _h = _z * math.sqrt(_p * (1 - _p) / n + _z * _z / (4 * n * n)) / _dd
        he_rows.append([
            f"{name} — {lab}", str(n), str(k),
            f"{100 * _p:.1f}", f"[{100 * max(0, _c - _h):.1f}; {100 * min(1, _c + _h):.1f}]",
            f"{sum(d.get('nfe', 0) for d in det) / n:.1f}",
            f"{sum(d.get('churn', 0) for d in det) / n:.2f}",
        ])
    add_table(story,
              ["Рука", "n", "прошло", "pass@1, %", "Wilson 95%", "Ср. NFE", "Ср. churn"],
              he_rows,
              [0.30, 0.07, 0.09, 0.11, 0.17, 0.13, 0.13],
              "Таблица 11. HumanEval v4b (протокол Saber ACL 2026, канонический "
              "runner OpenAI): блок 32, greedy, gen_length=160. Парное сравнение: "
              "n=20 общих задач, 1 исправлено / 0 сломано, McNemar p = 1,0; "
              "NFE парно 14,8 = 14,8 (×1,00), churn 0,00 → 2,00.",
              font="td", align_center_cols={1, 2, 3, 4, 5, 6})
    story.append(P(
        "На HumanEval разницы между рукой с правкой и без правки нет (5,0% "
        "против 0,0%, единственный прошедший тест — как раз у стока). "
        "Абсолютный уровень низкий: в режиме greedy с жёстким лимитом 160 "
        "токенов модель часто не укладывает полное решение, а проходных "
        "генераций на этих n единицы — CI шириной 14–23 п.п. это прямо "
        "показывают. Сравнение с v3 на Nemotron здесь некорректно по той же "
        "причине усечения (§8.3)."))

    h2(story, "8.5. Вторичные сигналы (устойчивые на 80 парах)")
    for t in [
        "<b>Эффективность.</b> На одних и тех же 80 задачах сток сходится за "
        "×0,85 NFE (50,1 → 42,5) и на 15% быстрее по часам (143 → 121 с/задача): "
        "обученная T2T-правка экономит проходы даже на коротком блоке — модель "
        "раньше достигает порога уверенности по блоку. Тот же коэффициент "
        "×0,86–0,88 у всех остальных рук с правкой.",
        "<b>Чернила мокрые.</b> Churn ×10,5 (1,14 → 11,97 перезаписей на задачу): "
        "гейт действительно переписывает открытые токены — телеметрия NFE/churn "
        "работает, эффект правки наблюдаем, просто не конвертируется в качество "
        "на коротком блоке.",
        "<b>Цена высокого порога M2T.</b> Порог 0.9 (как был оптимален у "
        "Nemotron в v3) на LLaDA2.2 удваивает NFE (×1,93: 50,5 → 97,3) и "
        "обваливает точность (15,4% на n=13): калибровка уверенности у новой "
        "базы другая, штатный порог 0.5 оправдан.",
        "<b>Блок 64.</b> У контроля эффекта нет (24,1% против 23,8%); у стока "
        "+4,4 п.п. (26,9%), но n=26 и 0 парных расхождений — в пределах шума.",
        "<b>Детерминизм.</b> ids_equal=True для b32 и b64 — жадный режим "
        "воспроизводим побитово; это условие осмысленности всех парных "
        "сравнений.",
        "<b>Завершимость.</b> 0 остаточных mask/DELETE/SPLIT в sanity-проверке; "
        "все 403 генерации завершились штатно — гарантии LLaDA2 работают на "
        "2×T4 + NF4 без шимов.",
    ]:
        story.append(P(t, "body"))

    h2(story, "8.6. Вердикт и следующий шаг (v5)")
    story.append(P(
        "Итог v4b по качеству: <b>обученный T2T-гейт на коротком блоке не даёт "
        "сдвига точности</b> (−1,2 п.п., p = 1,0 на 80 парах), абляции порога, "
        "пост-бюджета и длины блока — в пределах шума; единственная значимая "
        "находка — экономическая (×0,85 NFE, −15% времени). Но главный вывод "
        "v4b концептуальный: <b>штатный generate() LLaDA2.2 — это ещё не "
        "wet-ink из §2.1</b>. Встроенное блочное редактирование правит токены "
        "только внутри текущего блока и только до его закрытия: блок закрылся — "
        "чернила высохли навсегда, окно не скользит за фронтиром. Никакая "
        "конфигурация штатных параметров (весь v4 — 12 рук) не может этого "
        "изменить: расписание открытия токенов зашито в transfer schedule. "
        "Поэтому, как и сформулировал владелец идеи 7 ноября, v4b фиксируется "
        "как базлайн, а идея wet-ink реализуется собственным циклом "
        "декодирования — v5."))
    story.append(P(
        "<b>Дизайн v5</b> (ядро freedomform/wet-ink-v5, запуск 7 ноября 11:46:56 UTC; "
        "завершено COMPLETE — результаты в разделе 9). Фронтир открывает "
        "<b>ровно один токен за шаг</b> — M2T-гейтом самой модели на фронтирной "
        "позиции; позади фронтира живёт <b>скользящее окно W</b>, в котором "
        "обученный T2T-гейт модели (та же штатная формула "
        "<font name='DejaVuSans' size='8.4'>p0 &gt; editing_threshold &amp; "
        "old != x0</font>) продолжает править открытые токены; при продвижении "
        "фронтира окно скользит вместе с ним, и токены за пределами окна "
        "«высыхают» — жёсткая граница, без каких-либо новых формул. Руки: "
        "wetink_w16 (ядро гипотезы, W=16), wetink_noedit_w16 (контроль: "
        "одно-токенный фронтир без правки — изолирует вклад самого фронтира), "
        "wetink_w8 и wetink_w32 (абляции ширины; W=32 = «весь блок мокрый», "
        "мостик к стоку), he_wetink_w16 (перенос на HumanEval). Базлайны — "
        "прямо из v4b: та же модель, тот же 4-бит NF4, те же промпты и "
        "gen_length=160; парный McNemar считается офлайн на общих задачах. "
        "Честная цена метода заявлена заранее: NFE ≈ один forward на открытый "
        "токен ≈ 160–200 на задачу против ~42 у стока (×4–5) — она будет "
        "фиксироваться как NFE-ratio в каждом сравнении."))

    # ================================================= 9. Результаты v5
    h1(story, "9. Результаты v5: собственный wet-ink декодер (Kaggle: 2×T4, NF4)")
    story.append(P(
        "Ядро <b>kaggle: freedomform/wet-ink-v5</b> завершено со статусом "
        "<b>COMPLETE</b>: старт 7 ноября 11:46:56 UTC, рабочий цикл окончен на "
        "отметке 6 ч 04 м (21 865 с) при внутреннем бюджете 7,5 ч — итоговый дамп "
        "около 17:52 UTC. Конфигурация та же, что у базлайнов v4b: LLaDA2.2-mini "
        "в 4-бит NF4 (double-quant, fp16 compute), по полной модели на каждой "
        "T4, два независимых воркера, greedy, gen_length=160, те же промпты и "
        "порядок задач GSM8K — поэтому пары «v5-рука против v4b-базлайна» "
        "сопоставимы по индексу задачи. Исполнено <b>77 генераций в трёх руках</b> "
        "(wetink_w16 n=45, wetink_noedit_w16 n=22, wetink_w8 n=10); руки "
        "wetink_w32 и he_wetink_w16 пропущены планировщиком с честной отметкой "
        "<font name='DejaVuSans' size='8.4'>\"skipped\": \"time budget\"</font>. "
        "Артефакт — <font name='DejaVuSans' size='8.4'>results_v5.json</font>: "
        "sanity, эквивалентность, детали на задачу (NFE, churn, opened, "
        "rewrites, resid), агрегаты и бюджетный учёт."))
    callout_row(story, [
        ("COMPLETE", "v5: цикл 6 ч 04 м из бюджета 7,5 ч"),
        ("77", "генераций: 45 + 22 + 10 в трёх руках"),
        ("11/45 = 11/45", "парно со стоком, McNemar p = 1,0"),
        ("×4,88", "честная цена фронтира в NFE"),
    ])

    h2(story, "9.1. Исполнение: приоритетная очередь и адаптивный план")
    story.append(P(
        "Перед руками — sanity обеих механик и проверка эквивалентности "
        "(§9.4). Затем воркеры разбирали руки из приоритетной очереди; "
        "планировщик plan_n() пересчитывал допустимое n по остатку бюджета, "
        "фактической скорости и числу активных воркеров (приор 557 с/задача, "
        "откалиброванный по v4b, подтвердился: фактическая скорость 473–587 "
        "с/задача):"))
    add_table(story,
              ["Рука", "Приоритет", "n", "Часы", "с/задача", "Итог"],
              [
                  ["wetink_w16 (гипотеза, W=16)", "P0", "45", "5,91", "473",
                   "исполнена полностью (кап 45)"],
                  ["wetink_noedit_w16 (фронтир без правки)", "P0", "22", "3,59", "587",
                   "кап по бюджету при 2 воркерах"],
                  ["wetink_w8 (узкое окно)", "P1", "10", "1,63", "586",
                   "кап по бюджету"],
                  ["wetink_w32 (весь блок мокрый)", "P2", "0", "—", "—",
                   "SKIPPED: plan_n → 5 < минимальных 8"],
                  ["he_wetink_w16 (HumanEval)", "P2", "0", "—", "—",
                   "SKIPPED: plan_n → 5 < минимальных 8"],
              ],
              [0.34, 0.10, 0.07, 0.10, 0.11, 0.28],
              "Таблица 12. Исполнение рук v5: очередь P0→P2, два воркера, план "
              "plan_n() с резервом 900 с. Суммарная занятость воркеров "
              "40 041 с ≈ 11,1 ч GPU-времени в сессии 6,1 ч.",
              font="td", align_center_cols={1, 2, 3, 4})
    story.append(P(
        "Два пропуска — не сбой, а штатная защита статистики: на отметке 5 ч "
        "24 м свободного бюджета (7 546 с) с учётом двух активных воркеров "
        "хватило лишь на n_run=5 задач для каждой из оставшихся рук, а руки с "
        "n &lt; 8 не дают осмысленного McNemar — планировщик честно записал "
        "пропуск и отдал остаток времени доисполнению главной руки w16. "
        "Контроль noedit_w16 получил n=22, а не 45, потому что стартовал вторым "
        "при уже занятом воркере: (26 318 с − резерв)/(2 воркера × 557 с) ≈ 22. "
        "Это единственное место, где бюджет подрезал дизайн, — и оно учтено в "
        "анализе ниже (парные сравнения идут только по общим задачам)."))

    h2(story, "9.2. GSM8K: точность и парная статистика")
    _v5rows = []
    for lab, st in [("wetink_w16 (v5, гипотеза)", V5_W16),
                    ("wetink_noedit_w16 (v5, контроль)", V5_NE),
                    ("wetink_w8 (v5, абляция)", V5_W8),
                    ("stock_b32 (v4b, базлайн)", V4_STOCK),
                    ("noedit_b32 (v4b, базлайн)", V4_NOEDIT)]:
        _v5rows.append([
            lab, str(st["n"]), f"{st['acc']:.1f}",
            f"[{st['lo']:.1f}; {st['hi']:.1f}]",
            f"{st['nfe']:.1f}", f"{st['churn']:.2f}",
            f"{st.get('rewrites', 0):.1f}" if "rewrites" in st else "—",
        ])
    add_table(story,
              ["Рука", "n", "Точн., %", "Wilson 95%", "Ср. NFE",
               "Ср. churn", "Реврайты/зад"],
              _v5rows,
              [0.30, 0.07, 0.10, 0.15, 0.12, 0.13, 0.13],
              "Таблица 13. Все руки v5 и базлайны v4b на GSM8K: greedy, "
              "gen_length=160, блок 32, threshold M2T=0.5, max_post_steps=4. "
              "«Реврайты» — среднее число T2T-переписываний открытых токенов "
              "за задачу (телеметрия v5); у v4b эта метрика не собиралась.",
              font="td", align_center_cols={1, 2, 3, 4, 5, 6})

    add_table(story,
              ["Сравнение (v5 vs v4b)", "n пар", "испр.", "слом.", "McNemar p",
               "NFE (парно)"],
              [
                  ["wetink_w16 vs stock_b32",
                   str(V5P_STOCK["n"]), str(V5P_STOCK["b"]), str(V5P_STOCK["c"]),
                   f"{V5P_STOCK['p']:.4f}",
                   f"×{V5P_STOCK['ratio']:.2f} "
                   f"({V5P_STOCK['nfe0']:.0f}→{V5P_STOCK['nfe1']:.0f})"],
                  ["wetink_w16 vs noedit_b32",
                   str(V5P_NOED["n"]), str(V5P_NOED["b"]), str(V5P_NOED["c"]),
                   f"{V5P_NOED['p']:.4f}",
                   f"×{V5P_NOED['ratio']:.2f} "
                   f"({V5P_NOED['nfe0']:.0f}→{V5P_NOED['nfe1']:.0f})"],
                  ["wetink_noedit_w16 vs stock_b32",
                   str(V5P_STOCK_NE["n"]), str(V5P_STOCK_NE["b"]),
                   str(V5P_STOCK_NE["c"]), f"{V5P_STOCK_NE['p']:.4f}",
                   f"×{V5P_STOCK_NE['ratio']:.2f} "
                   f"({V5P_STOCK_NE['nfe0']:.0f}→{V5P_STOCK_NE['nfe1']:.0f})"],
                  ["wetink_w8 vs stock_b32",
                   str(V5P_STOCK_W8["n"]), str(V5P_STOCK_W8["b"]),
                   str(V5P_STOCK_W8["c"]), f"{V5P_STOCK_W8['p']:.4f}",
                   f"×{V5P_STOCK_W8['ratio']:.2f} "
                   f"({V5P_STOCK_W8['nfe0']:.0f}→{V5P_STOCK_W8['nfe1']:.0f})"],
              ],
              [0.28, 0.08, 0.08, 0.08, 0.13, 0.35],
              "Таблица 14. Парный точный McNemar рук v5 против базлайнов v4b "
              "(сопоставление по индексу задачи; модель, квантизация, промпты и "
              "gen_length идентичны). Все p = 1,0000 — значимых сдвигов качества "
              "нет.",
              font="td", align_center_cols={1, 2, 3, 4, 5})
    story.append(P(
        "<b>Чтение таблицы 14.</b> Главная пара — wetink_w16 против штатного "
        "параллельного stock_b32 на 45 общих задачах: обе руки решили "
        f"<b>ровно по {V5P_STOCK['k0']} из {V5P_STOCK['n']}</b> "
        f"({V5P_STOCK['b']} исправлено / {V5P_STOCK['c']} сломано, p = 1,0). "
        "Собственный wet-ink декодер с фронтиром «один токен за шаг», "
        "скользящим окном W=16 и обученным T2T-гейтом <b>сохраняет качество "
        "в точности</b> — ценой "
        f"×{V5P_STOCK['ratio']:.2f} NFE ({V5P_STOCK['nfe0']:.0f} → "
        f"{V5P_STOCK['nfe1']:.0f} forwards на задачу), ровно в заявленном "
        "диапазоне ×4–5 из §8.6. Контроль «фронтир без правки» ведёт себя так "
        "же (p = 1,0 к стоку): одно-токенный фронтир сам по себе ничего не "
        "ломает — и ничего не добавляет."))

    h2(story, "9.3. Ноль расхождений: что на самом деле делают правки в окне")
    story.append(P(
        "Самый тонкий результат прогона — внутренний контроль. Увидев 24,4% "
        "(w16, n=45) против 13,6% (контроль noedit_w16, n=22), легко прочесть "
        "это как +10,8 п.п. от T2T-правок в окне. Это чтение неверно, и "
        "парный анализ показывает это напрямую: на <b>22 общих задачах обе "
        f"руки решили по {V5P_CTRL['kn']}</b> — расхождения "
        f"<b>{V5P_CTRL['b']} исправлено / {V5P_CTRL['c']} сломано из "
        f"{V5P_CTRL['n']} пар</b>. Вся видимая разница — состав задач: "
        "контроль остановился на первых 22 (трудных для всех рук), а w16 "
        "успела пройти и лёгкий хвост 22–44:"))
    add_table(story,
              ["Сегмент задач", "wetink_w16", "noedit_w16", "stock_b32 (v4b)"],
              [
                  ["задачи 0–21 (трудные)",
                   f"{V5_SEG_W16_A[0]:.1f}% (n={V5_SEG_W16_A[1]})",
                   f"{V5_SEG_NE_A[0]:.1f}% (n={V5_SEG_NE_A[1]})",
                   f"{V5_SEG_STOCK_A[0]:.1f}% (n={V5_SEG_STOCK_A[1]})"],
                  ["задачи 22–44 (лёгкий хвост)",
                   f"{V5_SEG_W16_B[0]:.1f}% (n={V5_SEG_W16_B[1]})", "—",
                   f"{V5_SEG_STOCK_B[0]:.1f}% (n={V5_SEG_STOCK_B[1]})"],
              ],
              [0.30, 0.24, 0.22, 0.24],
              "Таблица 15. Точность по сегментам: на общих задачах 0–21 все "
              "три руки дают одинаковые 13,6%; разница общих accuracy — "
              "исключительно доля лёгкого хвоста в выборке руки.",
              font="td", align_center_cols={1, 2, 3})
    story.append(P(
        "Итог по механизму: T2T-правки внутри окна <b>реально происходят</b> — "
        f"{V5_W16['rewrites']:.1f} реврайта на задачу (в sanity-задаче — 79), "
        f"churn {V5_W16['churn']:.1f} против 0 у контроля — но на этой базе, "
        "бенче и n они <b>не перевернули ни одного исхода</b>: переписываются "
        "в основном формулировки и разметка (markdown-выделения, порядок слов "
        "в решении), а последний числовой ответ остаётся тем же. Это третий "
        "подряд виток одного и того же сигнала: v3 — training-free ревизия "
        "нейтральна/вредна; v4b — штатный блочный гейт нейтрален; v5 — "
        "скользящее окно с тем же обученным гейтом нейтрально. Ревизия "
        "открытых токенов на коротких генерациях GSM8K не является узким "
        "местом качества."))

    h2(story, "9.4. Механика, телеметрия и конфаунд усечения")
    for t in [
        f"<b>Sanity обеих механик.</b> Стоковая задача про клипы: nfe=27, "
        "churn=30, чистый markdown с LaTeX, 0 остаточных спец-токенов, 20 с. "
        "Wet-ink та же задача: 187 шагов, 167 открытых токенов, 79 реврайтов, "
        "0 остаточных mask/DELETE/SPLIT, корректный ответ, 137 с — фронтир, "
        "окно и высыхание работают как задумано.",
        "<b>Детерминизм.</b> Эквивалентность повторного запуска wet-ink: "
        "ids_equal=True (nfe 97 = 97) — жадный собственный декодер "
        "воспроизводим побитово; условие осмысленности парных сравнений "
        "выполнено.",
        f"<b>Телеметрия фронтира.</b> Открыто {V5_W16['opened']:.0f} токенов "
        f"на задачу при генерации ~157 токенов — каждый открытый токен "
        f"получает собственный forward; NFE {V5_W16['nfe']:.1f} ≈ открытые "
        "токены × пост-шаги. Реврайты {V5_W16['rewrites']:.1f}/задачу "
        f"(W=16) против {V5_W8['rewrites']:.1f} (W=8) — окно шире, правок "
        "пропорционально больше; resid=0 во всех 77 генерациях.",
        f"<b>Усечение.</b> Как и в v4b, лимит gen_length=160 режет "
        f"{V5_W16['trunc']} из {V5_W16['n']} генераций w16 (84%), "
        f"{V5_NE['trunc']} из {V5_NE['n']} у контроля; среди естественно "
        f"завершённых (EOS) верны {V5_W16_FIN_OK} из {len(V5_W16['fin'])} "
        f"у w16 и {V5_NE_FIN_OK} из {len(V5_NE['fin'])} у контроля — "
        "абсолютные точности занижены лимитом, выводы раздела 9.2–9.3 — о "
        "методе, а не о модели.",
        "<b>Бюджет.</b> Полезная занятость двух воркеров 40 041 с ≈ 11,1 ч "
        "GPU-времени; инкрементальные дампы после каждых 10 задач; резерв "
        "900 с на парную статистику и финальный дамп не понадобился полностью "
        "— очередь исчерпана на 6 ч 04 м.",
    ]:
        story.append(P(t, "body"))
    story.append(Spacer(1, 6))
    img = Image(os.path.join(WORK_DIR, "chart_v5_gsm8k.png"),
                width=AVAIL_W, height=AVAIL_W * 5.8 / 7.6)
    img.keepWithNext = 1
    story.append(img)
    story.append(Paragraph(
        "Рис. 3. Результаты v5. Слева сверху — точность рук v5 и базлайнов "
        "v4b с Wilson 95% CI; справа сверху — парный NFE главной пары "
        "(×4,88); слева снизу — парные расхождения McNemar: все сравнения "
        "p = 1,0, у внутреннего контроля 0/0; справа снизу — точность по "
        "сегментам задач: на общих задачах 0–21 все руки одинаковы, разница "
        "общих accuracy — доля лёгкого хвоста.",
        S["caption"]))
    story.append(Spacer(1, 8))

    h2(story, "9.5. Вердикт v5 и следующий шаг")
    for t in [
        "<b>(1) Идея §2.1 реализована в точности.</b> Собственный цикл "
        "декодирования: фронтир открывает ровно один токен за шаг (M2T-гейт "
        "модели), позади — скользящее окно W с обученным T2T-гейтом "
        "(editing_threshold=0.0), за окном чернила сухие навсегда. Никаких "
        "новых формул поверх опубликованных — только композиция штатных "
        "механизмов, как требует правило (г).",
        "<b>(2) Качество сохранено ровно.</b> 11/45 = 11/45 со стоком "
        "(p = 1,0), контроль фронтира тоже нейтрален. Одно-токенный фронтир и "
        "скользящее окно ничего не ломают — но и не добавляют.",
        "<b>(3) Правки не конвертируются в качество.</b> Третий виток "
        "подтверждает: на коротких задачах GSM8K у LLaDA2.2-mini ревизия "
        "открытых токенов (хоть training-free, хоть штатная блочная, хоть "
        "скользящим окном) не двигает точность. Утверждение идеи «далёкие "
        "токены меняются реже» реализовано и механически работает, но "
        "изменения в основном косметические.",
        "<b>(4) Цена честно измерена.</b> ×4,88 NFE к стоку (42 → 207) — "
        "структурная цена «forward на каждый токен». Это верхняя граница: "
        "никакого churn-взрыва, как в v1 (13,2×), — обученный гейт и жёсткое "
        "окно держат бюджет предсказуемым.",
        "<b>Следующий шаг (v6) — не пороги, а структура.</b> Правило (в) "
        "запрещает модифицировать обученный гейт, а v4b + v5 вместе показали, "
        "что настройка порогов (θ=0,0 / 0,5 / 0,7 / OFF) качество не двигает. "
        "Остаётся структурный рычаг: открывать за шаг k &gt; 1 токенов "
        "(блочный фронтир с сохранением скользящего окна) — это сокращает NFE "
        "в k раз, не требуя новых формул; патч телеметрии — гистограмма "
        "реврайтов по дистанции до фронтира, чтобы прямо измерить профиль "
        "высыхания. Для мощности по качеству нужен n ≥ 500 на руку (CI "
        "сузится с ±12 до ±4 п.п.).",
    ]:
        story.append(P(t, "body"))

    # ================================================= 10. Артефакты
    h1(story, "10. Артефакты")
    add_table(story,
              ["Артефакт", "Содержание"],
              [
                  ["github.com/Asilbekov/Explore", "код v1/v2/v3/v4/v5 (wet_ink.py, kaggle/, "
                   "kaggle_v4/, kaggle_v5/), тесты, results_v3/v4/v5.json, "
                   "анализ и генераторы отчётов, README с полным провенансом"],
                  ["wet_ink/kaggle/kernel.py", "ядро v3 (Nemotron-Labs-Diffusion-3B) — 1 235 строк"],
                  ["wet_ink/kaggle_v4/kernel.py", "ядро v4b (LLaDA2.2-mini, 2×T4, 4-бит NF4, "
                   "2 воркера, приоритетная очередь + plan_n) — 900 строк"],
                  ["wet_ink/kaggle_v5/kernel.py", "ядро v5: собственный декодер wet-ink "
                   "(фронтир 1 токен/шаг + скользящее окно W, гейты модели) — 1 153 строки"],
                  ["wet_ink/results/results_v3.json", "результаты v3: 15 конфигов × детали на задачу"],
                  ["wet_ink/results/results_v4.json", "результаты v4b: sanity, эквивалентность, "
                   "10 рук (8 GSM8K + 2 HumanEval, 403 генерации) с деталями на задачу, "
                   "парная статистика, бюджетный учёт"],
                  ["wet_ink/results/results_v5.json", "результаты v5: sanity, эквивалентность, "
                   "3 руки (45+22+10 генераций, телеметрия opened/rewrites/resid на каждую "
                   "задачу), агрегаты, бюджетный учёт; + вырезка лога ядра "
                   "v5_kernel_log_excerpt.txt"],
                  ["wet_ink/scripts/analyze_v5.py", "офлайн-анализ v5: парный McNemar "
                   "против базлайнов v4b, Wilson CI, NFE-ratio, доза-реакция окна W"],
                  ["research/llada22/", "дословный код модели LLaDA2.2-mini (ModelScope mirror)"],
                  ["kaggle: wet-ink-nemotron-3b", "v3 (Nemotron, 1×T4, fp16) — COMPLETE"],
                  ["kaggle: wet-ink-llada22-mini", "v4a int8 (частичный, референс) и v4b NF4 "
                   "(10 из 12 рук, 13,45 ч в бюджете 7,5 ч) — COMPLETE"],
                  ["kaggle: wet-ink-v5", "v5: декодер wet-ink — 3 руки исполнены "
                   "(45+22+10), w32 и HumanEval пропущены по бюджету (n_run=5 < 8); "
                   "цикл 6 ч 04 м — COMPLETE (7 ноября, старт 11:46:56 UTC)"],
              ],
              [0.34, 0.66],
              "Таблица 16. Артефакты v3, v4 и v5.",
              font="td")
    story.append(P(
        "Хронология проекта: (1) постановка идеи и правил; (2) выбор Nemotron как "
        "базы v1–v3; (3) извлечение дословных формул стока и доноров; (4) v1 — "
        "training-free ревизия, рост качества, обнаружен churn; (5) диагностика "
        "первопричины (потерянный инвариант прогресса); (6) пере-аудит 13 семейств; "
        "(7) v2/v3 — пересборка на RemeDi + CoDR с гарантиями завершимости; "
        "(8) v3 на полной модели — CoDR нейтрален, RemeDi значимо вредит; "
        "(9) смена базы по правилу (г): LLaDA2.2-mini как база с обученной правкой; "
        "(10) дословное чтение кода новой базы, локальная верификация механики; "
        "(11) v4a (int8) — пробный прогон, референс робастности квантизации; "
        "(12) v4b (NF4, 2 воркера) — полный прогон 10 рук: правка экономит 15% NFE, "
        "но качества не сдвигает, и главное — штатный generate() не реализует "
        "скользящее окно за фронтиром; (13) решение владельца (7 ноября): v4b = "
        "базлайны, v5 = собственный декодер wet-ink; (14) ядро v5 написано, "
        "проверено локально, запущено и завершено (COMPLETE, цикл 6 ч 04 м); "
        "(15) этот отчёт v5.0 фиксирует итог: декодер wet-ink реализует идею §2.1 "
        "в точности, парно сохраняет качество штатного декодера при ×4,88 NFE, "
        "правки в окне наблюдаемы, но исходов не переворачивают; следующий шаг — "
        "структурное удешевление фронтира (v6, §9.5)."))
    return story


def main():
    body_path = os.path.join(WORK_DIR, "body_v5.pdf")
    doc = TocDocTemplate(
        body_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title="«Мокрое чернило»: от Nemotron к LLaDA2.2-mini — технический отчёт v5.0",
        author="Asilbekov · проект Wet-Ink",
        subject="Wet-ink diffusion: Nemotron-Labs-Diffusion-3B to LLaDA2.2-mini, custom wet-ink decoder (v5)",
        creator="Z.ai",
    )
    story = build_story()
    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote body:", body_path)

    # обложка (HTML/Playwright, 1 стр.) + тело -> финальный PDF
    from pypdf import PdfReader, PdfWriter
    cover_path = os.path.join(WORK_DIR, "cover_v5.pdf")
    final_path = os.path.join(OUT_DIR, "Wet-Ink_Nemotron-to-LLaDA22_отчёт.pdf")
    writer = PdfWriter()
    for part in (cover_path, body_path):
        reader = PdfReader(part)
        for pg in reader.pages:
            writer.add_page(pg)
    with open(final_path, "wb") as f:
        writer.write(f)
    n = len(PdfReader(final_path).pages)
    print(f"wrote final: {final_path} ({1 + len(PdfReader(body_path).pages)} expected, {n} actual)")


if __name__ == "__main__":
    main()
