"""
data_loader.py -- Carga, limpieza y cruce de datos del workbook de gastos mineros.

Hojas relevantes del archivo Excel:
    - Budget (3046 filas): Presupuesto mensualizado, columnas mensuales Jan-25...Dec-25
      mas columnas anuales FY25...FY29 y BYTD.
    - Gastos (2575 filas): Forecast 5+7 existente. Columnas mensuales Jan-25...Dec-25
      (Ene--May = reales, Jun--Dic = proyeccion), mas YTD, Forecast FY, Budget FY,
      Var, BYTD y Forecast Actual.
    - GRUPOS (99 filas): Mapeo RESPONSABILIDAD -> CLASS (RH, OP, OM, SG, SO, AS, PR)
      y GRUPOS (rangos tipo "1 - 6", "5 - 10", etc.).
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = "02+Gastos+Proy+Mejor+01-2025.xlsx"

MONTH_COLS = [
    "Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25",
    "Jun-25", "Jul-25", "Aug-25", "Sep-25", "Oct-25", "Nov-25", "Dec-25",
]
REAL_MONTHS = MONTH_COLS[:5]   # Jan-25 ... May-25
PROJ_MONTHS = MONTH_COLS[5:]   # Jun-25 ... Dec-25
DIM_COLS = ["Resp", "Desc Resp", "VP", "Gerencia", "Proc", "Desc Proc",
            "Item", "Desc Item", "Classif", "CC"]


def _resolve_path(filename: str | None = None) -> Path:
    """Resuelve la ruta al archivo de datos, con fallback para despliegues."""
    fname = filename or DATA_FILE
    full = DATA_DIR / fname

    if not full.exists():
        # Fallback: probar version con underscores (comun en depliegues Linux)
        alt_name = fname.replace("+", "_")
        alt_full = DATA_DIR / alt_name
        if alt_full.exists():
            return alt_full
        # Fallback 2: buscar cualquier .xlsx en data/
        xlsx_files = list(DATA_DIR.glob("*.xlsx"))
        if xlsx_files:
            return xlsx_files[0]
        raise FileNotFoundError(
            f"No se encuentra el archivo de datos: {full}. "
            f"Coloque '{fname}' dentro de la carpeta 'data/'."
        )
    return full


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina espacios sobrantes (strip) en columnas tipo string."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convierte columnas a numerico, reemplazando errores con NaN."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_total_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas que sean totales o subtotales (Resp nulo o 'Total')."""
    if "Resp" in df.columns:
        df = df.dropna(subset=["Resp"])
        df = df[~df["Resp"].astype(str).str.upper().str.contains("TOTAL")]
    return df


# ---------------------------------------------------------------------------
# Carga individual de hojas
# ---------------------------------------------------------------------------

def load_budget_detail(filename: str | None = None) -> pd.DataFrame:
    """
    Carga la hoja 'Budget' con el presupuesto mensualizado multi-anio.

    Returns
    -------
    pd.DataFrame
        Columnas: Resp, Desc Resp, VP, Gerencia, Proc, Desc Proc, Item,
        Desc Item, Classif, CC, Jan-25...Dec-25, FY25...FY29, BYTD.
        Incluye columna derivada 'BUDGET_REAL_MONTHS' = suma Jan--May (presupuesto).
    """
    path = _resolve_path(filename)
    df = pd.read_excel(path, sheet_name="Budget")
    df = _clean_strings(df)
    df = _drop_total_rows(df)

    monthly_budget = MONTH_COLS.copy()
    annual_cols = ["FY25", "FY26", "FY27", "FY28", "FY29", "BYTD"]
    numeric_cols = monthly_budget + annual_cols
    df = _coerce_numeric(df, numeric_cols)

    df["BUDGET_REAL_MONTHS"] = df[REAL_MONTHS].sum(axis=1)
    df["BUDGET_PROJ_MONTHS"] = df[PROJ_MONTHS].sum(axis=1)
    return df


def load_forecast_detail(filename: str | None = None) -> pd.DataFrame:
    """
    Carga la hoja 'Gastos' con el detalle de forecast existente.

    Las columnas mensuales Jan--May contienen valores REALES (actuals).
    Las columnas Jun--Dic contienen proyecciones del forecast oficial.

    Returns
    -------
    pd.DataFrame
        Columnas dims + Jan-25...Dec-25, YTD, Forecast FY, Budget FY,
        Var, BYTD, Forecast Actual.
        Incluye columnas derivadas:
            - REAL_MONTHS_SUM = sum Jan--May (actuals)
            - PROJ_MONTHS_SUM = sum Jun--Dic (proyeccion oficial)
    """
    path = _resolve_path(filename)
    df = pd.read_excel(path, sheet_name="Gastos")
    df = _clean_strings(df)
    df = _drop_total_rows(df)

    monthly = MONTH_COLS.copy()
    extra_cols = ["YTD", "Forecast FY", "Budget FY", "Var", "BYTD", "Forecast Actual"]
    numeric_cols = monthly + extra_cols
    df = _coerce_numeric(df, numeric_cols)

    df["REAL_MONTHS_SUM"] = df[REAL_MONTHS].sum(axis=1)
    df["PROJ_MONTHS_SUM"] = df[PROJ_MONTHS].sum(axis=1)
    return df


def load_grupos_mapping(filename: str | None = None) -> pd.DataFrame:
    """
    Carga la hoja 'GRUPOS' que mapea cada RESPONSABILIDAD a CLASS y GRUPOS.

    Returns
    -------
    pd.DataFrame
        Columnas: RESPONSABILIDAD, CLASS, GRUPOS.
        CLASS: RH, OP, OM, SG, SO, AS, PR.
        GRUPOS: rangos de cantidad de personas (ej. "1 - 6", "5 - 10", "13").
    """
    path = _resolve_path(filename)
    df = pd.read_excel(path, sheet_name="GRUPOS")
    df = _clean_strings(df)
    return df


# ---------------------------------------------------------------------------
# Union y data final
# ---------------------------------------------------------------------------

def get_merged_data(filename: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cruza la data de forecast con la de budget y el mapeo GRUPOS.

    El cruce se hace por 'Desc Resp' con 'RESPONSABILIDAD' del mapeo.
    Forecast y Budget se cruzan por las columnas dimensionales comunes.

    Returns
    -------
    forecast : pd.DataFrame
        Forecast detail enriquecido con columnas CLASS y GRUPOS.
    budget : pd.DataFrame
        Budget detail enriquecido con columnas CLASS y GRUPOS.
    """
    forecast = load_forecast_detail(filename)
    budget = load_budget_detail(filename)
    grupos = load_grupos_mapping(filename)

    # Cruzar con GRUPOS usando Desc Resp <-> RESPONSABILIDAD
    forecast = forecast.merge(
        grupos, left_on="Desc Resp", right_on="RESPONSABILIDAD", how="left",
    )
    budget = budget.merge(
        grupos, left_on="Desc Resp", right_on="RESPONSABILIDAD", how="left",
    )

    return forecast, budget


def load_pivot_summary(filename: str | None = None) -> pd.DataFrame:
    """
    Carga la hoja 'Pivote (2)' con el resumen por Classif.

    Returns
    -------
    pd.DataFrame con columnas: Etiquetas de fila, Suma de YTD,
    Suma de Forecast FY, Suma de Budget FY, Suma de BYTD,
    Suma de Forecast Actual.
    """
    path = _resolve_path(filename)
    df = pd.read_excel(path, sheet_name="Pivote (2)", header=4)
    df = _clean_strings(df)
    metric_cols = [
        "Suma de YTD", "Suma de Forecast FY", "Suma de Budget FY",
        "Suma de BYTD", "Suma de Forecast Actual",
    ]
    df = _coerce_numeric(df, metric_cols)
    return df


def load_tabla_control(filename: str | None = None) -> pd.DataFrame:
    """
    Carga la hoja 'Tabla de Control' con el resumen por Naturaleza de Gasto.
    """
    path = _resolve_path(filename)
    df = pd.read_excel(path, sheet_name="Tabla de Control", header=None)
    return df
