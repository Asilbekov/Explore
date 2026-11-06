#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор PDF-отчёта v3.0 «Мокрое чернило»: от Nemotron к LLaDA2.2-mini.

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
                         textColor=HEADER_FILL, spaceBefore=16, spaceAfter=8)
S["h2"] = ParagraphStyle("h2", fontName="FreeSerif-Bold", fontSize=14, leading=18,
                         textColor=HEADER_FILL, spaceBefore=14, spaceAfter=6)
S["h3"] = ParagraphStyle("h3", fontName="FreeSerif-Bold", fontSize=11.5, leading=15,
                         textColor=TEXT_PRIMARY, spaceBefore=10, spaceAfter=5)
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
    story.append(HRFlowable(width="100%", color=ACCENT, thickness=1.4,
                            spaceBefore=0, spaceAfter=10))


def h2(story, text):
    story.append(CondPageBreak(PAGE_H * 0.12))
    story.append(add_heading(text, 1))


def h3(story, text):
    story.append(CondPageBreak(PAGE_H * 0.09))
    story.append(add_heading(text, 2))


def safe_keep(elements, max_frac=0.4):
    total = 0
    for el in elements:
        try:
            _, h = el.wrap(AVAIL_W, PAGE_H)
        except Exception:
            h = 60
        total += h
    if total <= PAGE_H * max_frac:
        return [KeepTogether(elements)]
    if len(elements) >= 2:
        return [KeepTogether(elements[:2])] + list(elements[2:])
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
    story.append(Spacer(1, 10))
    tbl = make_table(header, rows, ratios, **kw)
    cap = Paragraph(_nb(caption), S["caption"])
    if len(rows) <= 12:
        story.extend(safe_keep([tbl, cap]))
    else:
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
                      "«Мокрое чернило»: от Nemotron к LLaDA2.2-mini — технический отчёт v3.0")
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
        ("v3 → v4", "смена базы: Nemotron → LLaDA2.2-mini"),
        ("2×T4 + int8", "максимальная скорость, 16B в 2×16 ГБ"),
        ("0×", "самодельных формул поверх новой базы"),
        ("COMPLETE", "ядро v4: 7,30 ч из 7,5 ч бюджета"),
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
        "<b>Итог v4 (раздел 8).</b> Ядро завершилось за 7,30 ч из 7,5 ч: механика "
        "порта подтверждена (sanity без остаточных спец-токенов; жадное декодирование "
        "побитово детерминировано на 2×T4-pipeline — ids_equal=True для b32 и b64). "
        "В рамках бюджета уместилась только P0-пара GSM8K (блок 32, контроль против "
        "wet-ink): +2,4 п.п. (22,4% против 20,0%), McNemar p = 1,0 на 49 парных "
        "задачах — мощности не хватило для вывода о качестве. Положительный "
        "побочный сигнал: обученная T2T-правка экономит 16% NFE (×0,84) и ~15% "
        "времени на задачу. Абсолютные точности занижены усечением: адаптивный "
        "планировщик сжал gen_length до 160 токенов, и 84% генераций упёрлись в "
        "лимит до \\boxed{}; среди 20 естественно завершённых (EOS) ответов верны "
        "все 20. Вывод о методе — за прогоном v4.1 с gen_length=256."))

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
        "(г) запрещено придумывать собственные формулы или изменять чужие; "
        "допускается комбинировать формулы из разных источников только после "
        "проверки совместимости; <b>если новая база уже содержит обученный "
        "механизм для нашей задачи, устаревший внешний цикл отбрасывается</b>.",
    ]:
        story.append(P(t, "body"))
    story.append(P(
        "Расширение правила (г), явно сформулированное владельцем идеи при смене "
        "базы: «если раньше у нас была формула маскировки неуверенных прошлых "
        "ответов и открытия на следующем шаге диффузии, а теперь в новой базе "
        "уже есть определённый механизм для этого и он лучше — мы берём новую "
        "базу со всеми её формулами, а старую отбрасываем как менее эффективную». "
        "Это и определило дизайн v4."))

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
    h2(story, "6.3. 2×Tesla T4 + 8-бит: скорость как первая цель")
    story.append(P(
        "16B параметров в bf16 = 32 ГБ → на одну T4 (16 ГБ) не помещается. "
        "Решение по требованию владельца идеи: 2×T4 параллельно + 8-битная "
        "квантизация. Конкретно:"))
    code_block(story,
               "BitsAndBytesConfig(load_in_8bit=True,\n"
               "                   llm_int8_skip_modules=['lm_head'])  # 157K vocab -> fp16\n"
               "AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb,\n"
               "    device_map='auto',          # accelerate сам раскидает по 2×T4\n"
               "    torch_dtype=torch.float16,   # у T4 нет bf16\n"
               "    attn_implementation='eager') # SDPA отвергает bf16-маску; eager промоутит в fp32")
    story.append(P(
        "Проверки совместимости int8 + LLaDA2 (всё дословно из кода модели): "
        "(а) MoE-эксперты — обычные <font name='DejaVuSans' size='8.4'>nn.Linear</font>, "
        "bnb их квантизует; (б) роутер — сырой "
        "<font name='DejaVuSans' size='8.4'>nn.Parameter + F.linear</font> в fp32, "
        "bnb его не трогает, skip-лист не нужен; (в) RMSNorm апкастит в fp32 и "
        "возвращает входной dtype (fp16); (г) eager attention мягко складывает "
        "fp16+bf16 с промоушеном в fp32 через softmax(..., dtype=torch.float32), "
        "поэтому bf16 edit-маска работает на T4 без шимов. Все проверки "
        "выполнены локально на tiny-модели той же архитектуры."))

    # ================================================= 7. Верификация
    h1(story, "7. Локальная верификация v4")
    add_table(story,
              ["Проверка", "Результат"],
              [
                  ["bf16-маска + fp16/fp32 logits: сложение промоутится в fp32", "OK (T4 path)"],
                  ["Квантизация int8 + MoE-роутер: роутер в fp32 автоматически", "OK"],
                  ["eager attention принимает bf16-маску (vs SDPA: rejects)", "OK"],
                  ["Все 9 конфигов arm-сетки завершаются на tiny-модели", "OK (0 residual special)"],
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
    h1(story, "8. Результаты v4 (Kaggle: 2×T4, int8)")
    story.append(P(
        "Ядро <b>kaggle: freedomform/wet-ink-llada22-mini</b> завершено со статусом "
        "<b>COMPLETE</b> за 7,30 ч из 7,5 ч внутреннего бюджета (старт 6 ноября "
        "14:25 UTC, финиш 21:53 UTC). Артефакт — "
        "<font name='DejaVuSans' size='8.4'>results_v4.json</font>: sanity, "
        "проверка детерминизма, детали P0-пары на задачу (n, NFE, churn, токены, "
        "префикс вывода) и парная статистика. Ниже — фактическое выполнение и "
        "все полученные числа."))

    h2(story, "8.1. Выполнение: адаптивный планировщик в действии")
    add_table(story,
              ["Фаза", "Время", "Результат"],
              [
                  ["Установка и загрузка", "0:00–0:04",
                   "pip (transformers 5.2.0, bitsandbytes 0.50.2), снапшот модели, "
                   "int8-квантование на 2×T4, device_map раскидал 20 слоёв по двум GPU"],
                  ["Sanity (задача про клипы)", "82 с",
                   "чистый markdown-вывод с LaTeX-формулами, nfe=21, churn=16, "
                   "0 остаточных спец-токенов (mask/DELETE/SPLIT)"],
                  ["Эквивалентность", "0:05–0:09",
                   "жадный режим побитово детерминирован: ids_equal=True для b32 "
                   "(nfe 16=16) и b64 (nfe 15=15) на 2-GPU pipeline"],
                  ["noedit_b32 (P0, контроль)", "0:09–4:44",
                   "n=75 задач, 220 с/задача — калибровка предполагала 140 с/задачу"],
                  ["stock_b32 (P0, wet-ink)", "4:44–7:16",
                   "план 64 → ранний стоп на 49/64 по бюджету; 186 с/задача"],
                  ["Остальные 12 конфигов", "7:16–7:30",
                   "plan_n() вернул n_run=0 при остатке 807 с; парная статистика, "
                   "summary, инкрементальный dump"],
              ],
              [0.24, 0.12, 0.64],
              "Таблица 8. Хронология ядра v4 по логу и results_v4.json. "
              "Сессия 7,30 ч из 7,5 ч бюджета.",
              font="td", align_center_cols={1})
    story.append(P(
        "Измеренная скорость — 3,8–4,2 с на model.forward() (eager attention + "
        "int8 + pipeline через два GPU) — оказалась в ~1,6 раза хуже до-пусковой "
        "калибровки, поэтому полный прогон сетки (14 конфигов × 200/164 задач) "
        "потребовал бы порядка десяти таких бюджетов. Адаптивный планировщик "
        "отработал как задуман: пересчитал лимиты по фактической скорости, "
        "защитил P0-пару (запущена первой с гарантированными капами 75+75) и "
        "чисто пропустил остальные 12 конфигов, сохранив всё измеренное "
        "инкрементально после каждой конфигурации."))

    h2(story, "8.2. P0-пара: встроенный T2T-гейт против контроля (GSM8K, блок 32)")
    rows = []
    for name, a, lab in [("noedit_b32", V4_NOEDIT, "контроль (T2T OFF)"),
                         ("stock_b32", V4_STOCK, "wet-ink (T2T ON, default)")]:
        rows.append([
            f"{name} — {lab}", "2.0" if name == "noedit_b32" else "0.0",
            str(a["n"]), f"{a['acc']:.1f}",
            f"[{a['lo']:.1f}; {a['hi']:.1f}]",
            f"{a['nfe']:.1f}", f"{a['churn']:.2f}", f"{a['spt']:.0f}",
        ])
    add_table(story,
              ["Конфигурация", "edit_thr", "n", "Точн., %", "Wilson 95%",
               "Ср. NFE", "Ср. churn", "с/задача"],
              rows,
              [0.28, 0.08, 0.06, 0.10, 0.16, 0.10, 0.11, 0.11],
              "Таблица 9. P0-пара v4: GSM8K, block_length=32, threshold=0.5, "
              "max_post_steps=16, greedy (temperature=0), gen_length=160. Обе "
              "конфигурации используют только штатные параметры generate() "
              "LLaDA2.2-mini.",
              font="td", align_center_cols={1, 2, 3, 4, 5, 6, 7})
    add_table(story,
              ["Сравнение (парное)", "исправлено", "сломано", "McNemar p (точн.)",
               "NFE ×контроль", "churn ×контроль"],
              [[f"stock_b32 vs noedit_b32, n={V4_PAIR['n']} общих задач",
                f"{V4_PAIR['fixed']}", f"{V4_PAIR['broken']}",
                f"{V4_PAIR['p']:.4f}",
                f"{V4_PAIR['nfe_ratio']:.2f} ({V4_PAIR['nfe0']:.1f}→{V4_PAIR['nfe1']:.1f})",
                f"{V4_PAIR['ch_ratio']:.1f} ({V4_PAIR['ch0']:.2f}→{V4_PAIR['ch1']:.2f})"]],
              [0.30, 0.11, 0.10, 0.15, 0.17, 0.17],
              "Таблица 10. Парный точный McNemar на общих задачах двух конфигураций; "
              "NFE и churn — средние по одним и тем же задачам.",
              font="td", align_center_cols={1, 2, 3, 4, 5})
    story.append(Spacer(1, 6))
    img = Image(os.path.join(WORK_DIR, "chart_v4_gsm8k.png"),
                width=AVAIL_W, height=AVAIL_W * 5.8 / 7.6)
    story.append(img)
    story.append(Paragraph(
        "Рис. 2. Результаты v4, P0-пара. Слева сверху — точность с Wilson 95% CI; "
        "справа — парные NFE (×0,84) и churn (×5,5); снизу — эффект лимита "
        "gen_length=160: 84% генераций усечены, но все естественно завершённые "
        "(EOS) ответы верны (100%).", S["caption"]))
    story.append(Spacer(1, 8))

    h2(story, "8.3. Конфаунд усечения: gen_length=160")
    story.append(P(
        f"Адаптивный планировщик сжал gen_length со штатных 256 до 160 токенов, "
        f"чтобы уместить больше задач. На LLaDA2.2 это привело к систематическому "
        f"усечению: модель пишет развёрнутые markdown-решения (заголовки и "
        f"LaTeX-блоки), которые не помещаются в 160 токенов. Усечены "
        f"{V4_NOEDIT['trunc']} из {V4_NOEDIT['n']} генераций контроля (84%) и "
        f"{V4_STOCK['trunc']} из {V4_STOCK['n']} у wet-ink (84%); ни одна из "
        f"усечённых не дошла до \\boxed{{}}, и экстрактор падал на «последнее "
        f"число» из середины решения. Следствие: абсолютные 20–22% — это мера "
        f"усечения, а не способности модели, и напрямую несравнимы с v3 "
        f"(86,5% — другая база, 256 токенов на решение, другой стиль). "
        f"Ключевой контроль: среди "
        f"{len(V4_FIN_ALL)} естественно завершённых (EOS-ранний стоп) генераций "
        f"верны {V4_FIN_OK} из {len(V4_FIN_ALL)} — 100% (12/12 у контроля, 8/8 у "
        f"wet-ink). Способность базы в порядке; эксперимент же по методу "
        f"недостаточно мощный: одно расхождение на 49 пар, p = 1,0."))

    h2(story, "8.4. Вторичные сигналы (значимые уже на этой паре)")
    for t in [
        "<b>Эффективность.</b> На одних и тех же задачах wet-ink сходится за "
        "×0,84 NFE (46,8 → 39,4) и ~15% меньше времени на задачу (220 → 186 с): "
        "обученная T2T-правка экономит проходы даже на коротком блоке — модель "
        "раньше достигает порога уверенности по блоку.",
        "<b>Чернила мокрые.</b> Churn ×5,5 (1,4 → 7,6 перезаписей на задачу): "
        "гейт реально переписывает открытые токены, телеметрия NFE/churn работает.",
        "<b>Детерминизм на гетерогенном pipeline.</b> ids_equal=True для b32 и "
        "b64 при device_map, размазывающем слои по двум GPU, — жадный режим "
        "воспроизводим побитово; это условие осмысленности парных сравнений.",
        "<b>Завершимость.</b> 0 остаточных mask/DELETE/SPLIT во всех 124 "
        "генерациях — штатные гарантии LLaDA2 работают на 2×T4 + int8 без шимов.",
    ]:
        story.append(P(t, "body"))

    h2(story, "8.5. Вердикт и следующий шаг (v4.1)")
    story.append(P(
        "Главная гипотеза v4 — обученный T2T-гейт даёт положительный сдвиг "
        "качества на коротких блоках — <b>не подтверждена и не опровергнута</b>: "
        "одно расхождение на 49 парах не даёт ни значимости, ни уверенности в "
        "нулевом эффекте (в v3 для значимости требовались десятки расхождений). "
        "Экономический сигнал при этом устойчиво положительный (×0,84 NFE в "
        "паре). Шаг v4.1 не требует изменений в коде харнесса — только "
        "конфигурация запуска: (а) gen_length=256 (протокол v3, без усечения); "
        "(б) полный 12-часовой бюджет Kaggle; (в) калибровка plan_n() от "
        "измеренных 186–220 с/задачу. Арифметика: P0-пара на n=200 — это "
        "400 задач × ~200 с ≈ 22 ч на 2×T4, то есть либо A100-ресурс (в 4–6 раз "
        "быстрее, без int8), либо две сессии с возобновлением от "
        "промежуточного JSON. Никаких новых формул: всё те же штатные параметры "
        "generate() LLaDA2.2-mini."))

    # ================================================= 9. Артефакты
    h1(story, "9. Артефакты")
    add_table(story,
              ["Артефакт", "Содержание"],
              [
                  ["github.com/Asilbekov/Explore", "код v1/v2/v3/v4 (wet_ink.py, kaggle/, kaggle_v4/), "
                   "тесты, results_v3.json, results_v4.json, README с полным провенансом"],
                  ["wet_ink/kaggle/kernel.py", "ядро v3 (Nemotron-Labs-Diffusion-3B) — 1 235 строк"],
                  ["wet_ink/kaggle_v4/kernel.py", "ядро v4 (LLaDA2.2-mini, 2×T4, int8, адаптивный "
                   "планировщик plan_n) — 844 строки"],
                  ["wet_ink/results/results_v3.json", "результаты v3: 15 конфигов × детали на задачу"],
                  ["wet_ink/results/results_v4.json", "результаты v4: sanity, эквивалентность, "
                   "P0-пара GSM8K (n=75+49) с деталями на задачу, парная статистика"],
                  ["research/llada22/", "дословный код модели LLaDA2.2-mini (ModelScope mirror)"],
                  ["kaggle: wet-ink-nemotron-3b", "v3 (Nemotron, 1×T4, fp16) — COMPLETE"],
                  ["kaggle: wet-ink-llada22-mini", "v4 (LLaDA2.2-mini, 2×T4, int8) — COMPLETE "
                   "(7,30 ч из 7,5 ч; P0-пара, 2 из 14 конфигов, остальные — по решению "
                   "адаптивного планировщика)"],
              ],
              [0.34, 0.66],
              "Таблица 11. Артефакты v3 и v4.",
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
        "(11) v4 — запуск на Kaggle 2×T4 + int8; (12) v4 завершён за 7,30 ч: "
        "механика и детерминизм подтверждены, P0-пара не набрала мощности "
        "(p = 1,0), сигнал эффективности ×0,84 NFE; финальный отчёт обновлён; "
        "следующий шаг — v4.1 с gen_length=256 и полным бюджетом."))
    return story


def main():
    body_path = os.path.join(WORK_DIR, "body_v4.pdf")
    doc = TocDocTemplate(
        body_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title="«Мокрое чернило»: от Nemotron к LLaDA2.2-mini — технический отчёт v3.0",
        author="Asilbekov · проект Wet-Ink",
        subject="Wet-ink diffusion: migration from Nemotron-Labs-Diffusion-3B to LLaDA2.2-mini",
        creator="Z.ai",
    )
    story = build_story()
    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote body:", body_path)

    # обложка (HTML/Playwright, 1 стр.) + тело -> финальный PDF
    from pypdf import PdfReader, PdfWriter
    cover_path = os.path.join(WORK_DIR, "cover_v4.pdf")
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
