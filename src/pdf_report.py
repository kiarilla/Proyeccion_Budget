"""
pdf_report.py -- Generador de informe PDF corporativo para el Forecast 5+7.

Construye un documento ejecutivo y detallado (portada, resumen, metodologia,
detalle mensual, analisis por dimension, top desviaciones y conclusiones)
usando exclusivamente ReportLab. No requiere dependencias graficas externas.
"""

from datetime import datetime
from io import BytesIO

import pandas as pd

try:
    import zoneinfo
except ImportError:  # pragma: no cover
    zoneinfo = None

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Paleta corporativa (coherente con el resto de la plataforma)
# ---------------------------------------------------------------------------
AZUL_CORP = colors.HexColor("#1d3557")
AZUL_MEDIO = colors.HexColor("#457b9d")
ROJO_CORP = colors.HexColor("#e63946")
VERDE_CORP = colors.HexColor("#2a9d8f")
GRIS_SUAVE = colors.HexColor("#f1faee")
GRIS_BORDE = colors.HexColor("#dee2e6")
TEXTO = colors.HexColor("#2b2d42")


# ---------------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------------

def format_currency(value, decimals: int = 0) -> str:
    """Formatea un monto con sufijo de magnitud (coherente con viz.format_currency)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "0"
    if abs(value) >= 1e9:
        return f"{value / 1e9:,.{decimals}f} MM"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:,.{decimals}f} M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:,.{decimals}f} K"
    return f"{value:,.{decimals}f}"


def _money_full(value) -> str:
    try:
        return f"$ {float(value):,.0f}"
    except (TypeError, ValueError):
        return "$ 0"


def _pct(value) -> str:
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _safe(value, default="-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


# ---------------------------------------------------------------------------
# Encabezado / pie de pagina con numeracion
# ---------------------------------------------------------------------------

def _on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(AZUL_CORP)
    canvas.setLineWidth(0.8)
    canvas.line(40, height - 32, width - 40, height - 32)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(AZUL_MEDIO)
    canvas.drawString(40, height - 28, "Forecast 5+7 - Proyeccion de Gastos Operacionales (OPEX)")

    canvas.setStrokeColor(GRIS_BORDE)
    canvas.setLineWidth(0.5)
    canvas.line(40, 38, width - 40, 38)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(40, 28, "Documento Confidencial - Control de Gestion")
    canvas.drawRightString(width - 40, 28, f"Pagina {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Construccion de estilos y tablas reutilizables
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle(
            "FcTitulo", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=23, leading=27, textColor=AZUL_CORP, alignment=1, spaceAfter=14,
        ),
        "sub": ParagraphStyle(
            "FcSub", parent=base["Normal"], fontName="Helvetica",
            fontSize=13, leading=17, textColor=AZUL_MEDIO, alignment=1, spaceAfter=28,
        ),
        "h1": ParagraphStyle(
            "FcH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, textColor=AZUL_CORP, spaceBefore=14, spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "FcBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=TEXTO, spaceAfter=8,
        ),
        "bold": ParagraphStyle(
            "FcBold", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9.5, leading=13.5, textColor=TEXTO,
        ),
        "analisis": ParagraphStyle(
            "FcAnalisis", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, leading=13, textColor=AZUL_CORP, spaceBefore=4, spaceAfter=10,
        ),
        "nota": ParagraphStyle(
            "FcNota", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8, leading=11, textColor=colors.grey, spaceBefore=2, spaceAfter=8,
        ),
    }


def _data_table(rows, col_widths, header_color=AZUL_CORP, align_right_from_col=1):
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXTO),
        ("GRID", (0, 0), (-1, -1), 0.4, GRIS_BORDE),
        ("ALIGN", (align_right_from_col, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS_SUAVE]),
    ]
    table.setStyle(TableStyle(style))
    return table


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def build_forecast_pdf(
    kpis: dict,
    metodo_ganador: str,
    backtesting_df: pd.DataFrame,
    agg_classif: pd.DataFrame,
    agg_vp: pd.DataFrame,
    agg_gerencia: pd.DataFrame,
    deviation_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    filtros: dict,
) -> BytesIO:
    """
    Genera el informe PDF detallado del Forecast 5+7.

    Parameters
    ----------
    kpis : dict
        KPIs de compute_kpis() (Budget_FY_Total, Forecast_5plus7_Total, etc.).
    metodo_ganador : str
        Metodo de proyeccion seleccionado.
    backtesting_df : pd.DataFrame
        Resultados de run_backtesting().
    agg_classif, agg_vp, agg_gerencia : pd.DataFrame
        Agregados por dimension (aggregate_forecast()).
    deviation_df : pd.DataFrame
        Desviaciones por linea (compute_deviations()).
    monthly_df : pd.DataFrame
        Detalle mensual (Mes, Real / Proyeccion, Budget Mensual, Forecast Oficial, Var vs Budget).
    filtros : dict
        Filtros activos {"VP": ..., "Classif": ..., "CLASS": ...}.

    Returns
    -------
    BytesIO
        Buffer con el PDF listo para descargar.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=40, leftMargin=40, topMargin=52, bottomMargin=48,
        title="Informe Forecast 5+7",
    )
    s = _styles()
    story = []

    # Fecha viva (zona horaria Chile)
    ahora = None
    if zoneinfo is not None:
        try:
            ahora = datetime.now(zoneinfo.ZoneInfo("America/Santiago"))
        except Exception:
            ahora = None
    if ahora is None:
        ahora = datetime.now()
    fecha = ahora.strftime("%d/%m/%Y %H:%M")

    vp_txt = _safe(filtros.get("VP"), "Todas")
    classif_txt = _safe(filtros.get("Classif"), "Todas")
    class_txt = _safe(filtros.get("CLASS"), "Todas")

    # ---------------------------------------------------------------- Portada
    story.append(Spacer(1, 46))
    story.append(Paragraph("INFORME DETALLADO DE FORECAST 5+7", s["titulo"]))
    story.append(Paragraph(
        "Proyeccion No Lineal de Gastos Operacionales (OPEX)<br/>"
        "5 meses reales (Ene-May) + 7 meses proyectados (Jun-Dic)",
        s["sub"],
    ))
    story.append(Spacer(1, 24))

    portada_rows = [
        [Paragraph("<b>Preparado para:</b>", s["body"]),
         Paragraph("Comite de Finanzas y Control de Gestion", s["body"])],
        [Paragraph("<b>Fecha de emision (Chile):</b>", s["body"]),
         Paragraph(fecha, s["body"])],
        [Paragraph("<b>Metodo de proyeccion:</b>", s["body"]),
         Paragraph(f"<b>{_safe(metodo_ganador)}</b> (ganador por RMSE en backtesting)", s["body"])],
        [Paragraph("<b>Filtro Vicepresidencia (VP):</b>", s["body"]),
         Paragraph(vp_txt, s["body"])],
        [Paragraph("<b>Filtro Clasificacion:</b>", s["body"]),
         Paragraph(classif_txt, s["body"])],
        [Paragraph("<b>Filtro CLASS (Grupos):</b>", s["body"]),
         Paragraph(class_txt, s["body"])],
    ]
    t_portada = Table(portada_rows, colWidths=[150, 360])
    t_portada.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("GRID", (0, 0), (-1, -1), 0.5, GRIS_BORDE),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_portada)
    story.append(PageBreak())

    # ----------------------------------------------- 1. Resumen Ejecutivo
    budget_fy = kpis.get("Budget_FY_Total", 0) or 0
    forecast_total = kpis.get("Forecast_5plus7_Total", 0) or 0
    var_abs = kpis.get("Var_vs_Budget_Abs", 0) or 0
    var_pct = kpis.get("Var_vs_Budget_Pct", 0) or 0
    real_ytd = kpis.get("Real_YTD_Total", 0) or 0
    avance = kpis.get("Pct_Avance_Real", 0) or 0
    oficial = kpis.get("Forecast_Oficial_Total", 0) or 0
    var_of_pct = kpis.get("Var_vs_Oficial_Pct", 0) or 0

    story.append(Paragraph("1. Resumen Ejecutivo", s["h1"]))
    story.append(Paragraph(
        "Este informe documenta la proyeccion de cierre de ano (Forecast 5+7) de los gastos "
        "operacionales, combinando la ejecucion real de los primeros cinco meses con una "
        f"proyeccion no lineal de los siete meses restantes mediante el metodo <b>{_safe(metodo_ganador)}</b>. "
        "Los montos reflejan el alcance definido por los filtros indicados en la portada.",
        s["body"],
    ))

    kpi_rows = [
        ["Metrica", "Valor"],
        ["Budget FY (presupuesto)", format_currency(budget_fy)],
        ["Forecast 5+7 (proyeccion)", format_currency(forecast_total)],
        ["Variacion vs Budget (abs.)", format_currency(var_abs)],
        ["Variacion vs Budget (%)", _pct(var_pct)],
        ["Real YTD (Ene-May)", format_currency(real_ytd)],
        ["% Avance real sobre Budget", f"{avance:.1f}%"],
        ["Forecast Oficial", format_currency(oficial)],
        ["Variacion 5+7 vs Oficial (%)", _pct(var_of_pct)],
    ]
    story.append(_data_table(kpi_rows, [300, 210]))

    if var_pct < -0.5:
        analisis = (
            f"<b>Analisis:</b> el Forecast 5+7 proyecta un cierre <b>{abs(var_pct):.1f}% por debajo</b> "
            f"del Budget FY ({format_currency(var_abs)}), consistente con un patron de sub-ejecucion "
            "presupuestaria. Se recomienda revisar partidas con ratios de ejecucion atipicos antes de "
            "liberar el presupuesto comprometido."
        )
    elif var_pct > 0.5:
        analisis = (
            f"<b>Analisis:</b> el Forecast 5+7 proyecta un cierre <b>{var_pct:.1f}% por encima</b> "
            f"del Budget FY ({format_currency(var_abs)}), lo que sugiere presion al alza en los OPEX. "
            "Se recomienda activar controles de gasto en las partidas con mayor desviacion positiva."
        )
    else:
        analisis = (
            "<b>Analisis:</b> el Forecast 5+7 se mantiene practicamente alineado con el Budget FY, "
            "mostrando una ejecucion controlada dentro de las bandas presupuestarias esperadas."
        )
    story.append(Paragraph(analisis, s["analisis"]))

    # ----------------------------------------------- 2. Metodologia
    story.append(Paragraph("2. Metodologia y Validacion (Backtesting)", s["h1"]))
    story.append(Paragraph(
        f"El metodo <b>{_safe(metodo_ganador)}</b> fue seleccionado por obtener el menor RMSE en la "
        "validacion walk-forward (entrenar con meses 1-3, predecir meses 4-5). Preserva la "
        "estacionalidad del presupuesto y amortigua los ratios de ejecucion extremos, lo que lo hace "
        "robusto frente a la limitada cantidad de datos reales disponibles.",
        s["body"],
    ))

    if backtesting_df is not None and not backtesting_df.empty:
        bt_rows = [["Metodo", "MAPE", "RMSE", "MAE", "Lineas"]]
        bt = backtesting_df.copy()
        if "rmse_mean" in bt.columns:
            bt = bt.sort_values("rmse_mean")
        for _, r in bt.iterrows():
            bt_rows.append([
                _safe(r.get("method")),
                f"{r.get('mape_mean', 0):,.1f}",
                f"{r.get('rmse_mean', 0):,.0f}",
                f"{r.get('mae_mean', 0):,.0f}",
                f"{int(r.get('n_lines', 0)):,}",
            ])
        story.append(_data_table(bt_rows, [150, 90, 100, 100, 70], header_color=AZUL_MEDIO))

    # ----------------------------------------------- 3. Detalle Mensual
    story.append(Paragraph("3. Detalle Mensual (Real + Proyeccion)", s["h1"]))
    if monthly_df is not None and not monthly_df.empty:
        m_rows = [["Mes", "Real / Proy.", "Budget", "Oficial", "Var vs Budget"]]
        for _, r in monthly_df.iterrows():
            m_rows.append([
                _safe(r.get("Mes")),
                _money_full(r.get("Real / Proyeccion", r.get("Real / Proyección", 0))),
                _money_full(r.get("Budget Mensual", 0)),
                _money_full(r.get("Forecast Oficial", 0)),
                _money_full(r.get("Var vs Budget", 0)),
            ])
        story.append(_data_table(m_rows, [70, 115, 110, 110, 110], header_color=AZUL_MEDIO))
    else:
        story.append(Paragraph("No hay datos mensuales para el alcance seleccionado.", s["body"]))

    story.append(PageBreak())

    # ----------------------------------------------- 4. Analisis por Dimension
    story.append(Paragraph("4. Analisis por Dimension", s["h1"]))

    def _dim_block(title, df, dim_col, top_n):
        story.append(Paragraph(title, s["bold"]))
        if df is None or df.empty or dim_col not in df.columns:
            story.append(Paragraph("Sin datos para el alcance seleccionado.", s["body"]))
            story.append(Spacer(1, 6))
            return
        d = df.sort_values("Forecast_5+7", ascending=False)
        truncated = len(d) > top_n
        d = d.head(top_n)
        rows = [[dim_col, "Forecast 5+7", "Budget FY", "Var Abs", "Var %"]]
        for _, r in d.iterrows():
            rows.append([
                _safe(r.get(dim_col)),
                _money_full(r.get("Forecast_5+7", 0)),
                _money_full(r.get("Budget_FY", 0)),
                _money_full(r.get("Var_Abs", 0)),
                _pct(r.get("Var_Pct", 0)),
            ])
        story.append(_data_table(rows, [150, 100, 100, 95, 70]))
        if truncated:
            story.append(Paragraph(f"Se muestran las {top_n} mayores; tabla resumida.", s["nota"]))
        story.append(Spacer(1, 6))

    _dim_block("4.1 Por Clasificacion (Classif)", agg_classif, "Classif", 15)
    _dim_block("4.2 Por Vicepresidencia (VP)", agg_vp, "VP", 12)
    _dim_block("4.3 Por Gerencia (Top 12)", agg_gerencia, "Gerencia", 12)

    # Conclusion dinamica de dimensiones
    if agg_classif is not None and not agg_classif.empty and "Var_Pct" in agg_classif.columns:
        valid = agg_classif[agg_classif["Budget_FY"].abs() > 0]
        if not valid.empty:
            over = valid.loc[valid["Var_Pct"].idxmax()]
            under = valid.loc[valid["Var_Pct"].idxmin()]
            story.append(Paragraph(
                f"<b>Lectura por clasificacion:</b> <b>{_safe(over['Classif'])}</b> es la partida con "
                f"mayor ejecucion relativa ({_pct(over['Var_Pct'])} vs Budget), mientras que "
                f"<b>{_safe(under['Classif'])}</b> presenta la mayor sub-ejecucion "
                f"({_pct(under['Var_Pct'])}). Estas dos partidas concentran el mayor riesgo de desviacion "
                "y deben priorizarse en la revision de gestion.",
                s["analisis"],
            ))

    # ----------------------------------------------- 5. Top Desviaciones
    story.append(Paragraph("5. Top Desviaciones por Item (vs Budget)", s["h1"]))
    if (deviation_df is not None and not deviation_df.empty
            and "Var_vs_Budget_Abs" in deviation_df.columns):
        dv = deviation_df.reindex(
            deviation_df["Var_vs_Budget_Abs"].abs().sort_values(ascending=False).index
        ).head(15)
        dv_rows = [["Item", "Var Abs", "Var %"]]
        for _, r in dv.iterrows():
            item = _safe(r.get("Desc Item"))
            if len(item) > 48:
                item = item[:45] + "..."
            dv_rows.append([
                item,
                _money_full(r.get("Var_vs_Budget_Abs", 0)),
                _pct(r.get("Var_vs_Budget_Pct", 0)),
            ])
        story.append(_data_table(dv_rows, [290, 120, 100]))
    else:
        story.append(Paragraph("No hay desviaciones para el alcance seleccionado.", s["body"]))

    # ----------------------------------------------- 6. Hallazgos y Conclusion
    story.append(Paragraph("6. Hallazgos y Conclusion", s["h1"]))
    sentido = "por debajo del" if var_abs < 0 else ("por encima del" if var_abs > 0 else "alineado con el")
    story.append(Paragraph(
        f"A nivel agregado, el Forecast 5+7 proyecta un cierre de {format_currency(forecast_total)}, "
        f"{_pct(var_pct)} {sentido} Budget FY de {format_currency(budget_fy)}. La ejecucion real "
        f"acumulada (Ene-May) alcanza {format_currency(real_ytd)}, equivalente a un {avance:.1f}% del "
        "presupuesto anual. Se recomienda actualizar la proyeccion mensualmente incorporando el nuevo "
        "dato real y validar con los gerentes de area las partidas con desviaciones extremas antes de "
        "tomar decisiones presupuestarias.",
        s["body"],
    ))
    story.append(Paragraph(
        "Informe generado automaticamente por la Plataforma de Control de Gestion. "
        "Los resultados se basan exclusivamente en el analisis cuantitativo de los datos disponibles.",
        s["nota"],
    ))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    buffer.seek(0)
    return buffer
