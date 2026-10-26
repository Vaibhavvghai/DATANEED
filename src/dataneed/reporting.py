from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BRAND = colors.HexColor("#FF4B4B")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5F6B7A")
PALE = colors.HexColor("#F3F6FA")


def _money(value: float) -> str:
    return f"INR {float(value):,.0f}"


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E0E8"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9 * mm, "DataNeed AI - Decision-support estimate, not a performance guarantee")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf_report(
    *,
    input_mode: str,
    dataset_name: str,
    target_name: str,
    dataset_summary: dict[str, str],
    winner: dict,
    comparison: pd.DataFrame,
    health_checks: Iterable[dict[str, str]],
    explanation: str,
) -> bytes:
    """Build a compact, polished DataNeed AI result report."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=19 * mm,
        title="DataNeed AI Analysis Report",
        author="DataNeed AI",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=24, leading=28, textColor=INK, alignment=TA_CENTER, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], fontSize=10, leading=14,
        textColor=MUTED, alignment=TA_CENTER, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=17, textColor=INK, spaceBefore=10, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=13,
        textColor=INK,
    ))

    story = [
        Paragraph("DataNeed AI", styles["ReportTitle"]),
        Paragraph("Training-data requirement and algorithm comparison report", styles["Subtitle"]),
    ]

    summary_data = [["Input mode", input_mode], ["Dataset", dataset_name], ["Target", target_name]]
    summary_data.extend([[key, value] for key, value in dataset_summary.items()])
    summary_table = Table(summary_data, colWidths=[48 * mm, 122 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), PALE),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E0E8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([Paragraph("Input summary", styles["Section"]), summary_table])

    cards = [
        ["Recommended algorithm", "Conservative total", "Additional samples"],
        [str(winner["Algorithm"]), f"{int(winner['Conservative total']):,}", f"{int(winner['Additional samples']):,}"],
        ["Target F1", "Estimated accuracy", "Estimated labelling cost"],
        [f"{float(winner['Target F1']):.1%}", f"{float(winner['Estimated accuracy']):.1%}", _money(winner["Estimated cost (INR)"])],
    ]
    card_table = Table(cards, colWidths=[56.5 * mm] * 3)
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND),
        ("BACKGROUND", (0, 2), (-1, 2), BRAND),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 2), (-1, 2), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTNAME", (0, 3), (-1, 3), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E0E8")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([Paragraph("Recommendation", styles["Section"]), card_table])
    story.extend([
        Paragraph("Why this algorithm?", styles["Section"]),
        Paragraph(explanation, styles["BodySmall"]),
    ])

    table_rows = [["Algorithm", "Pilot F1", "Conservative", "Additional", "Accuracy", "Cost"]]
    for _, row in comparison.iterrows():
        table_rows.append([
            str(row["Algorithm"]), f"{float(row['Pilot F1']):.1%}",
            f"{int(row['Conservative total']):,}", f"{int(row['Additional samples']):,}",
            f"{float(row['Estimated accuracy']):.1%}", _money(row["Estimated cost (INR)"]),
        ])
    comparison_table = Table(table_rows, repeatRows=1, colWidths=[39 * mm, 23 * mm, 29 * mm, 27 * mm, 25 * mm, 27 * mm])
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E0E8")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([Paragraph("Three-algorithm comparison", styles["Section"]), comparison_table])

    health_rows = [["Status", "Check", "Finding"]]
    for item in health_checks:
        health_rows.append([item["status"], item["check"], Paragraph(item["detail"], styles["BodySmall"])])
    health_table = Table(health_rows, repeatRows=1, colWidths=[22 * mm, 40 * mm, 108 * mm])
    health_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E0E8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([PageBreak(), Paragraph("Dataset health checks", styles["Section"]), health_table])

    method_text = (
        "DataNeed AI extracts dataset meta-features and early learning-curve signals, then applies a "
        "random-forest meta-model trained on benchmark datasets. The point estimate is corrected using "
        "the 90th percentile of leave-dataset-out log residuals. When pilot F1 is below the target, a "
        "learning-curve extrapolation guard prevents a false claim that the current data are sufficient. "
        "Accuracy is separately estimated and has a benchmark mean absolute error of about 6.52 percentage points."
    )
    limitation_text = (
        "Use this report for planning, not as a guarantee. Validate the selected model on an untouched test "
        "set. Results currently support tabular classification with 2-20 classes; they do not cover regression, "
        "time series, images, or transformer-based text models."
    )
    story.append(KeepTogether([
        Paragraph("Methodology", styles["Section"]),
        Paragraph(method_text, styles["BodySmall"]),
        Spacer(1, 4),
        Paragraph("Limitations", styles["Section"]),
        Paragraph(limitation_text, styles["BodySmall"]),
    ]))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()

