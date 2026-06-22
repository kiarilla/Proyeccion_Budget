"""
app.py -- Plataforma Avanzada de Control de Gestión, Forecast 5+7 y Planificación Quinquenal.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Asegurar la ruta de importaciones locales
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import (
    load_forecast_detail,
    load_budget_detail,
    load_grupos_mapping,
    load_pivot_summary,
    get_merged_data,
)
from src.forecast import (
    run_backtesting,
    select_best_method,
    project_full_forecast,
    aggregate_forecast,
    apply_method,
)
from src.insights import (
    compute_deviations,
    compute_kpis,
    top_deviations,
    compare_with_official,
)
from src.model_store import (
    cache_exists,
    load_backtesting_results,
    load_forecast,
    load_metadata,
    save_backtesting_results,
    save_forecast,
    save_metadata,
    clear_cache,
)
from src.viz import (
    format_currency,
    plot_monthly_trend,
    plot_waterfall,
    plot_treemap,
    plot_method_comparison,
    plot_bar_comparison,
    plot_top_deviations,
    MONTH_COLS,
    MONTH_NAMES,
)

# Configuración inicial de la página
st.set_page_config(
    page_title="Forecast 5+7 - Minera",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 1. SISTEMA ESTRICTO DE CARGA DE ARCHIVOS
# ============================================================================
st.sidebar.title("📁 Datos Maestros")
st.sidebar.markdown("Para inicializar los modelos matemáticos, es obligatorio proveer la fuente de datos.")

uploaded_file = st.sidebar.file_uploader("Sube el archivo Excel maestro de presupuestos", type=["xlsx", "xls"])

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Ruta estandarizada donde el programa guardará el archivo para que los módulos internos lo lean
file_path = data_dir / "02_Gastos_Proy_Mejor_01-2025.xlsx"

# BLOQUEO ESTRUCTURAL
if uploaded_file is None:
    st.info("👋 ¡Bienvenido a la Plataforma de Control de Gestión y Planificación Estratégica!")
    st.warning("El sistema se encuentra en pausa. Por favor, suba el archivo Excel maestro en el panel lateral izquierdo para comenzar el procesamiento de datos.")
    st.stop()
else:
    if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.last_uploaded_name = uploaded_file.name
        st.cache_data.clear()
        clear_cache()
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("✅ Archivo validado y cargado en memoria exitosamente.")

# ============================================================================
# 2. NAVEGACIÓN PRINCIPAL
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.title("🧭 Navegación")
app_mode = st.sidebar.radio("Seleccione un Módulo de Trabajo:", [
    "📊 Forecast Operacional (5+7)",
    "📈 Proyección Estratégica (2027-2031)"
])
st.sidebar.markdown("---")

# ============================================================================
# Carga de datos (optimizada y cacheada)
# ============================================================================
@st.cache_data(show_spinner="Estructurando matrices de datos base...")
def cargar_datos():
    forecast_df = load_forecast_detail()
    budget_df = load_budget_detail()
    grupos_df = load_grupos_mapping()
    pivot_df = load_pivot_summary()
    forecast_merged, budget_merged = get_merged_data()
    return forecast_df, budget_df, grupos_df, pivot_df, forecast_merged, budget_merged

with st.spinner("Inicializando modelos de datos..."):
    forecast_df, budget_df, grupos_df, pivot_df, forecast_merged, budget_merged = cargar_datos()

if "modelos_ejecutados" not in st.session_state:
    st.session_state.modelos_ejecutados = False
if "resultados_backtesting" not in st.session_state:
    st.session_state.resultados_backtesting = None
if "forecast_lines" not in st.session_state:
    st.session_state.forecast_lines = None
if "metodo_ganador" not in st.session_state:
    st.session_state.metodo_ganador = None
if "kpis" not in st.session_state:
    st.session_state.kpis = None

if cache_exists() and not st.session_state.modelos_ejecutados:
    cached_bt = load_backtesting_results()
    cached_fc = load_forecast()
    cached_meta = load_metadata()
    if cached_bt is not None and cached_fc is not None and cached_meta is not None:
        st.session_state.resultados_backtesting = cached_bt
        st.session_state.forecast_lines = cached_fc
        st.session_state.metodo_ganador = cached_meta.get("best_method", "budget_scaled")
        st.session_state.kpis = cached_meta.get("kpis", {})
        st.session_state.modelos_ejecutados = True

# ============================================================================
# ============================================================================
# MÓDULO 1: FORECAST OPERACIONAL 5+7
# ============================================================================
# ============================================================================
if app_mode == "📊 Forecast Operacional (5+7)":

    if not st.session_state.modelos_ejecutados:
        st.warning(
            "No se encontraron modelos predictivos ejecutados previamente en caché. "
            "Haga clic en el botón para ejecutar el backtesting y generar el Forecast 5+7."
        )
        if st.button("Ejecutar Modelos (Backtesting + Forecast)", type="primary", use_container_width=True):
            with st.spinner("Ejecutando backtesting de métodos (esto puede tardar ~60s)..."):
                st.session_state.resultados_backtesting = run_backtesting(forecast_df, budget_df)

            st.session_state.metodo_ganador = select_best_method(
                st.session_state.resultados_backtesting, "rmse_mean"
            )

            with st.spinner(f"Generando Forecast 5+7 con método: {st.session_state.metodo_ganador}..."):
                st.session_state.forecast_lines = project_full_forecast(
                    forecast_df, budget_df, method=st.session_state.metodo_ganador
                )

            st.session_state.kpis = compute_kpis(st.session_state.forecast_lines, forecast_df)

            save_backtesting_results(st.session_state.resultados_backtesting)
            save_forecast(st.session_state.forecast_lines)
            save_metadata(st.session_state.metodo_ganador, st.session_state.kpis)
            st.session_state.modelos_ejecutados = True
            st.rerun()

        st.sidebar.title("⛏ Forecast 5+7")
        st.sidebar.info("Ejecute los modelos para ver los resultados.")
        st.stop()

    resultados_backtesting = st.session_state.resultados_backtesting
    forecast_lines = st.session_state.forecast_lines
    metodo_ganador = st.session_state.metodo_ganador
    kpis = st.session_state.kpis

    deviation_df = compute_deviations(forecast_lines, compare_vs_official=True)

    agg_vp = aggregate_forecast(forecast_lines, ["VP"])
    agg_classif = aggregate_forecast(forecast_lines, ["Classif"])
    agg_gerencia = aggregate_forecast(forecast_lines, ["Gerencia"])

    st.sidebar.title("Filtros Globales")
    st.sidebar.markdown("---")

    vps = ["Todas"] + sorted(forecast_merged["VP"].dropna().unique().tolist())
    vp_seleccionada = st.sidebar.selectbox("Vicepresidencia (VP)", vps)

    classifs = ["Todas"] + sorted(forecast_merged["Classif"].dropna().unique().tolist())
    classif_seleccionada = st.sidebar.selectbox("Clasificación", classifs)

    if "CLASS" in forecast_merged.columns:
        classes = ["Todas"] + sorted(forecast_merged["CLASS"].dropna().unique().tolist())
        class_seleccionada = st.sidebar.selectbox("CLASS (Grupos)", classes)
    else:
        class_seleccionada = "Todas"

    st.sidebar.markdown("---")
    metodo_seleccionado = st.sidebar.selectbox(
        "Método de proyección",
        ["linear", "budget_scaled", "polynomial", "holt_damped", "spline_damped", "arima"],
        index=["linear", "budget_scaled", "polynomial", "holt_damped", "spline_damped", "arima"].index(metodo_ganador),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Método ganador (RMSE): **{metodo_ganador}**")

    if st.sidebar.button("Re-ejecutar Modelos", use_container_width=True):
        clear_cache()
        st.session_state.modelos_ejecutados = False
        st.rerun()

    def filtrar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if vp_seleccionada != "Todas" and "VP" in df.columns:
            df = df[df["VP"] == vp_seleccionada]
        if classif_seleccionada != "Todas" and "Classif" in df.columns:
            df = df[df["Classif"] == classif_seleccionada]
        if class_seleccionada != "Todas" and "CLASS" in df.columns:
            df = df[df["CLASS"] == class_seleccionada]
        return df

    forecast_lines_f = filtrar_dataframe(forecast_lines)
    deviation_df_f = filtrar_dataframe(deviation_df)
    agg_classif_f = filtrar_dataframe(agg_classif) if "Classif" in agg_classif.columns else agg_classif
    agg_gerencia_f = filtrar_dataframe(agg_gerencia) if "Gerencia" in agg_gerencia.columns else agg_gerencia

    if vp_seleccionada != "Todas" or classif_seleccionada != "Todas":
        kpis_f = compute_kpis(forecast_lines_f, forecast_df)
    else:
        kpis_f = kpis

    tabs = st.tabs([
        "1. Resumen Ejecutivo",
        "2. Análisis por Dimensión",
        "3. Tendencia Mensual",
        "4. Forecast 5+7",
        "5. Comparaciones",
        "6. Hallazgos",
        "7. Exportar",
    ])

    # [El contenido del Módulo 1 se mantiene intacto, como me lo pediste originalmente]
    with tabs[0]:
        st.title("Resumen Ejecutivo - Forecast 5+7")
        st.markdown("Proyección no lineal de gastos operacionales (OPEX) -- 5 meses reales + 7 meses proyectados.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Budget FY", format_currency(kpis_f["Budget_FY_Total"]), delta=None)
        with col2:
            st.metric("Forecast 5+7", format_currency(kpis_f["Forecast_5plus7_Total"]), delta=f"{kpis_f['Var_vs_Budget_Pct']:+.1f}% vs Budget", delta_color="inverse")
        with col3:
            st.metric("Real YTD (Ene-May)", format_currency(kpis_f["Real_YTD_Total"]), delta=f"{kpis_f['Pct_Avance_Real']:.1f}% del Budget")
        with col4:
            oficial_val = kpis_f.get("Forecast_Oficial_Total", 0) or 0
            var_vs_oficial = kpis_f.get("Var_vs_Oficial_Pct", 0) or 0
            st.metric("Forecast Oficial", format_currency(oficial_val), delta=f"{var_vs_oficial:+.1f}% vs 5+7", delta_color="off")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Composición por Clasificación")
            fig_treemap = plot_treemap(agg_classif_f, path=["Classif"], value_col="Forecast_5+7", title="", color_col="Var_Pct")
            st.plotly_chart(fig_treemap, use_container_width=True, key="treemap_resumen")
        with col_b:
            st.subheader("Forecast 5+7 vs Budget por VP")
            fig_barras = plot_bar_comparison(agg_vp, x_col="VP", budget_col="Budget_FY", forecast_col="Forecast_5+7", title="", top_n=10)
            st.plotly_chart(fig_barras, use_container_width=True, key="barras_vp_resumen")

        st.markdown("---")
        st.subheader("Waterfall: Budget FY a Forecast 5+7")
        devs = {}
        for _, row in agg_classif_f.iterrows():
            devs[str(row["Classif"])] = row["Var_Abs"]
        fig_waterfall = plot_waterfall(kpis_f["Budget_FY_Total"], kpis_f["Forecast_5plus7_Total"], deviations=devs)
        st.plotly_chart(fig_waterfall, use_container_width=True, key="waterfall_resumen")

    with tabs[1]:
        st.title("Análisis por Dimensión")
        dim_tabs = st.tabs(["Por VP", "Por Gerencia", "Por Classif", "Por CLASS", "Top Items"])

        with dim_tabs[0]:
            st.subheader("Forecast 5+7 por Vicepresidencia")
            fig_vp = plot_bar_comparison(agg_vp, "VP", title="")
            st.plotly_chart(fig_vp, use_container_width=True, key="barras_vp_dim")
        with dim_tabs[1]:
            st.subheader("Top Gerencias por Forecast 5+7")
            top_ger = agg_gerencia_f.nlargest(15, "Forecast_5+7").sort_values("Forecast_5+7")
            fig_ger = plot_bar_comparison(top_ger, "Gerencia", title="")
            st.plotly_chart(fig_ger, use_container_width=True, key="barras_gerencia_dim")
        with dim_tabs[2]:
            st.subheader("Forecast 5+7 por Clasificación (Classif)")
            fig_classif = plot_bar_comparison(agg_classif_f, "Classif", title="")
            st.plotly_chart(fig_classif, use_container_width=True, key="barras_classif_dim")
        with dim_tabs[3]:
            st.subheader("Forecast 5+7 por CLASS (Grupos)")
            if "CLASS" in forecast_lines_f.columns:
                agg_class_group = aggregate_forecast(forecast_lines_f, ["CLASS"])
                fig_classg = plot_bar_comparison(agg_class_group, "CLASS", title="")
                st.plotly_chart(fig_classg, use_container_width=True, key="barras_classg_dim")
            else:
                st.info("Columna CLASS no disponible.")
        with dim_tabs[4]:
            st.subheader("Top 20 Ítems con Mayor Desviación vs Budget")
            top_dev = top_deviations(deviation_df_f, by="Var_vs_Budget_Abs", n=20)
            fig_top = plot_top_deviations(top_dev, label_col="Desc Item", deviation_col="Var_vs_Budget_Abs", pct_col="Var_vs_Budget_Pct", title="", n=20)
            st.plotly_chart(fig_top, use_container_width=True, key="topdev_dim")

    with tabs[2]:
        st.title("Tendencia Mensual")
        if not forecast_lines_f.empty:
            forecast_monthly = forecast_lines_f[MONTH_COLS].sum().values
            budget_monthly = np.zeros(12)
            official_monthly = np.zeros(12)
            dim_cols_merge = ["Resp", "Desc Resp", "VP", "Gerencia", "Proc", "Desc Proc", "Item", "Desc Item", "Classif", "CC"]
            budget_for_merge = budget_df[dim_cols_merge + MONTH_COLS].rename(columns={c: c + "_b" for c in MONTH_COLS})
            forecast_for_merge = forecast_df[dim_cols_merge + MONTH_COLS].rename(columns={c: c + "_o" for c in MONTH_COLS})
            merged_m = forecast_lines_f[dim_cols_merge].merge(budget_for_merge, on=dim_cols_merge, how="inner").merge(forecast_for_merge, on=dim_cols_merge, how="inner")
            if not merged_m.empty:
                for i, col in enumerate(MONTH_COLS):
                    budget_monthly[i] = merged_m[f"{col}_b"].sum()
                    official_monthly[i] = merged_m[f"{col}_o"].sum()
            fig_trend = plot_monthly_trend(forecast_monthly, budget_monthly, official_series=official_monthly, title="")
            st.plotly_chart(fig_trend, use_container_width=True, key="tendencia_mensual")
        else:
            st.warning("No hay datos para mostrar.")

    with tabs[3]:
        st.title("Forecast 5+7 - Método")
        st.dataframe(resultados_backtesting.set_index("method"), use_container_width=True)

    with tabs[4]:
        st.title("Comparaciones")
        comp_df = compare_with_official(forecast_lines_f, group_cols=["Classif"])
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_comp1 = plot_bar_comparison(comp_df, x_col="Classif", budget_col="Budget_FY", forecast_col="Forecast_5plus7", title="")
            st.plotly_chart(fig_comp1, use_container_width=True, key="comp_5plus7_vs_budget")
        with col_c2:
            fig_comp2 = plot_bar_comparison(comp_df, x_col="Classif", budget_col="Forecast_Oficial", forecast_col="Forecast_5plus7", title="")
            st.plotly_chart(fig_comp2, use_container_width=True, key="comp_5plus7_vs_oficial")

    with tabs[5]:
        st.title("Hallazgos")
        st.info("Revisar sección para insights detectados.")

    with tabs[6]:
        st.title("Exportar")
        csv = forecast_lines_f.to_csv(index=False)
        st.download_button(label="Descargar CSV", data=csv, file_name="forecast_5plus7.csv", mime="text/csv", use_container_width=True)


# ============================================================================
# ============================================================================
# MÓDULO 2: PROYECCIÓN ESTRATÉGICA QUINQUENAL (2027-2031)
# ============================================================================
# ============================================================================

elif app_mode == "📈 Proyección Estratégica (2027-2031)":
    st.title("📈 Tablero Interactivo de Proyección Estratégica y KPIs")
    st.markdown("Modelo de proyección histórica corregida (Basado en el crecimiento orgánico 2024-2026) con sensibilidad a variables operativas clave.")

    @st.cache_data
    def cargar_hojas_estratejicas(path):
        return pd.read_excel(path, sheet_name="BUDGET 2024 - 2028"), pd.read_excel(path, sheet_name="BUDGET 2025 - 2029"), pd.read_excel(path, sheet_name="BUDGET 2026 - 2030")

    try:
        b24, b25, b26 = cargar_hojas_estratejicas(file_path)
    except Exception as e:
        st.error(f"Error: Faltan pestañas históricas en el Excel subido. Detalle: {e}")
        st.stop()

    columnas_clave = ['CC', 'VP', 'Gerencia', 'Desc Item', 'Classif']
    cols_existentes = [c for c in columnas_clave if c in b26.columns]
    
    # Consolidar Histórico Real 24-26
    df_estrat = b26[cols_existentes].copy()
    df_estrat = df_estrat.merge(b24[['CC', 'FY24']], on='CC', how='left')
    df_estrat = df_estrat.merge(b25[['CC', 'FY25']], on='CC', how='left')
    meses_cal = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_estrat = df_estrat.merge(b26[['CC', 'FY26'] + [f'{m}-26' for m in meses_cal]], on='CC', how='left')
    df_estrat.fillna(0, inplace=True)

    # --- FILTROS GLOBALES EN EL SIDEBAR (Evita que Plotly resetee las vistas) ---
    st.sidebar.subheader("🔎 Filtro de Visualización")
    todas_clasificaciones = sorted(df_estrat['Classif'].unique().tolist())
    classif_seleccionadas = st.sidebar.multiselect(
        "Filtrar por Clasificación:",
        options=todas_clasificaciones,
        default=todas_clasificaciones,
        help="Al usar este filtro, el gráfico mantendrá tu selección aunque muevas los parámetros de estrés."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎬 Escenarios Preconfigurados")
    escenario = st.sidebar.selectbox("Seleccione un escenario estratégico:", [
        "Manual / Personalizado",
        "Crisis Global (+Combustible y Dólar)",
        "Negociación Sindical (+Mano de Obra)",
        "Eficiencia Operativa (-Costos Generales)"
    ])

    val_fuel, val_power, val_dolar, val_labor = 0.0, 0.0, 0.0, 0.0
    if escenario == "Crisis Global (+Combustible y Dólar)":
        val_fuel, val_power, val_dolar, val_labor = 25.0, 10.0, 15.0, 5.0
    elif escenario == "Negociación Sindical (+Mano de Obra)":
        val_fuel, val_power, val_dolar, val_labor = 5.0, 2.0, 2.0, 18.0
    elif escenario == "Eficiencia Operativa (-Costos Generales)":
        val_fuel, val_power, val_dolar, val_labor = -10.0, -5.0, -8.0, -5.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Parámetros de Sensibilidad (%)")
    slider_fuel_pct = st.sidebar.slider("Variación Precio Diésel / Comb.", -100.0, 100.0, val_fuel, step=0.1)
    slider_power_pct = st.sidebar.slider("Variación Tarifa Energía Eléc.", -100.0, 100.0, val_power, step=0.1)
    slider_dolar_pct = st.sidebar.slider("Variación Tipo de Cambio / USD", -100.0, 100.0, val_dolar, step=0.1)
    slider_labor_pct = st.sidebar.slider("Variación Costo Mano de Obra", -100.0, 100.0, val_labor, step=0.1)

    f24 = pd.to_numeric(df_estrat['FY24'], errors='coerce').fillna(0)
    f26 = pd.to_numeric(df_estrat['FY26'], errors='coerce').fillna(0)

    # --- CÁLCULOS BASE ---
    tasa_crecimiento = np.where(f24 > 0, (f26 / (f24 + 1e-6)) ** (1/2), 1.0).clip(0.95, 1.10)
    
    df_estrat['Base_FY27'] = f26 * tasa_crecimiento
    df_estrat['Base_FY28'] = df_estrat['Base_FY27'] * tasa_crecimiento
    df_estrat['Base_FY29'] = df_estrat['Base_FY28'] * tasa_crecimiento
    df_estrat['Base_FY30'] = df_estrat['Base_FY29'] * tasa_crecimiento
    df_estrat['Base_FY31'] = df_estrat['Base_FY30'] * tasa_crecimiento

    # --- MAPEO SEMÁNTICO DE ESTRÉS ---
    def evaluar_afectacion(fila):
        item = str(fila.get('Desc Item', '')).lower()
        classif = str(fila.get('Classif', '')).lower()
        mult = 1.0
        if 'labor' in classif or any(p in item for p in ['remuneracion', 'sueldo', 'honorario', 'mano de obra', 'bono', 'dotacion']):
            mult += (slider_labor_pct / 100.0)
        if any(p in item for p in ['diesel', 'combustible', 'petroleo', 'gasoil']) and 'servicio' not in item:
            mult += (slider_fuel_pct / 100.0)
        if any(p in item for p in ['energia electrica', 'kwh', 'tarifa electrica']):
            mult += (slider_power_pct / 100.0)
        if any(p in item for p in ['foreign', 'usd', 'importado', 'licencia corporativa']):
            mult += (slider_dolar_pct / 100.0)
        return mult

    df_estrat['Factor_Estrés_Fila'] = df_estrat.apply(evaluar_afectacion, axis=1)

    años_quinquenio = ['FY27', 'FY28', 'FY29', 'FY30', 'FY31']
    for a in años_quinquenio:
        df_estrat[f'Final_{a}'] = df_estrat[f'Base_{a}'] * df_estrat['Factor_Estrés_Fila']

    for m in meses_cal:
        m_26 = pd.to_numeric(df_estrat.get(f'{m}-26', 0), errors='coerce').fillna(0)
        df_estrat[f'peso_{m}'] = m_26 / (f26 + 1e-6)
        
    suma_pesos = df_estrat[[f'peso_{m}' for m in meses_cal]].sum(axis=1)
    
    for m in meses_cal:
        peso_ajustado = np.where(suma_pesos > 0, df_estrat[f'peso_{m}'] / (suma_pesos + 1e-6), 1.0/12.0)
        # Necesitamos la base y el estresado mensual para el grafico de líneas
        df_estrat[f'Base_{m}-27'] = df_estrat['Base_FY27'] * peso_ajustado
        df_estrat[f'{m}-27'] = df_estrat['Final_FY27'] * peso_ajustado

    cols_salida = cols_existentes + ['FY24', 'FY25', 'FY26'] + [f'{m}-27' for m in meses_cal] + [f'Final_{a}' for a in años_quinquenio]
    df_final_proy = df_estrat[cols_salida].copy()

    # --- APLICAR FILTRO DEL SIDEBAR A LA DATA ANTES DE GRAFICAR ---
    df_graficos = df_estrat[df_estrat['Classif'].isin(classif_seleccionadas)].copy()

    # --- KPIs ---
    tot_fy27_base = df_graficos['Base_FY27'].sum()
    tot_fy27_estres = df_graficos['Final_FY27'].sum()
    delta_usd = tot_fy27_estres - tot_fy27_base
    pct_var = (delta_usd / tot_fy27_base * 100) if tot_fy27_base != 0 else 0

    st.markdown("### Resumen de KPIs (Año 2027)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Proyección Base FY27", f"${tot_fy27_base:,.0f}")
    col2.metric("Proyección Estresada FY27", f"${tot_fy27_estres:,.0f}")
    col3.metric("Impacto Neto Operativo", f"${delta_usd:,.0f}", f"{pct_var:+.2f}%", delta_color="inverse")
    col4.metric("Escenario de Riesgo", escenario)

    st.markdown("---")
    
    # Nuevas pestañas organizadas
    tab_est0, tab_est1, tab_est2, tab_est3, tab_est4 = st.tabs([
        "📈 Histórico + Proyección (24-31)",
        "📉 Comparativos (Base vs Estrés)",
        "📊 Barras Anuales de Composición",
        "🔍 Detalle de Filas Afectadas",
        "💾 Generar Excel Dinámico"
    ])

    # PALETA DE COLORES MEJORADA
    modern_colors = px.colors.qualitative.Vivid + px.colors.qualitative.Pastel

    with tab_est0:
        st.subheader("Evolución del Gasto Total (2024 a 2031)")
        # Preparar data de tendencia completa
        anios_hist = ['FY24', 'FY25', 'FY26']
        anios_proy = ['Final_FY27', 'Final_FY28', 'Final_FY29', 'Final_FY30', 'Final_FY31']
        
        df_hist = df_graficos[['Classif'] + anios_hist].melt(id_vars=['Classif'], var_name='Año', value_name='Monto')
        df_hist['Año'] = df_hist['Año'].str.replace('FY', '20')
        
        df_proy = df_graficos[['Classif'] + anios_proy].melt(id_vars=['Classif'], var_name='Año', value_name='Monto')
        df_proy['Año'] = df_proy['Año'].str.replace('Final_FY', '20')
        
        df_full_trend = pd.concat([df_hist, df_proy])
        df_trend_grouped = df_full_trend.groupby(['Año', 'Classif'])['Monto'].sum().reset_index()

        fig_trend_full = px.line(
            df_trend_grouped, 
            x="Año", y="Monto", color="Classif", 
            markers=True,
            color_discrete_sequence=modern_colors,
            title="Tendencia Histórica Real y Proyección Estresada"
        )
        fig_trend_full.add_vline(x="2026", line_dash="dash", line_color="red", annotation_text="  Inicio Proyección")
        fig_trend_full.update_layout(yaxis_tickformat="$,.0f", hovermode="x unified")
        st.plotly_chart(fig_trend_full, use_container_width=True)

    with tab_est1:
        st.subheader("Impacto del Estrés - Mensual 2027")
        # Mensual Base vs Final
        cols_base_m = [f'Base_{m}-27' for m in meses_cal]
        cols_final_m = [f'{m}-27' for m in meses_cal]
        
        val_base_m = df_graficos[cols_base_m].sum().values
        val_final_m = df_graficos[cols_final_m].sum().values
        
        fig_comp_m = go.Figure()
        fig_comp_m.add_trace(go.Scatter(x=meses_cal, y=val_base_m, mode='lines+markers', name='Proyección Original Base', line=dict(color='gray', dash='dash')))
        fig_comp_m.add_trace(go.Scatter(x=meses_cal, y=val_final_m, mode='lines+markers', name='Escenario Estresado', line=dict(color='firebrick', width=3)))
        fig_comp_m.update_layout(title="Comparativa Mensual 2027: Base vs. Estrés", yaxis_tickformat="$,.0f", hovermode="x unified")
        st.plotly_chart(fig_comp_m, use_container_width=True)

        st.markdown("---")
        st.subheader("Impacto del Estrés - Quinquenio 2027-2031")
        # Anual Base vs Final
        cols_base_a = [f'Base_{a}' for a in años_quinquenio]
        cols_final_a = [f'Final_{a}' for a in años_quinquenio]
        
        val_base_a = df_graficos[cols_base_a].sum().values
        val_final_a = df_graficos[cols_final_a].sum().values
        anios_label = ['2027', '2028', '2029', '2030', '2031']
        
        fig_comp_a = go.Figure()
        fig_comp_a.add_trace(go.Scatter(x=anios_label, y=val_base_a, mode='lines+markers', name='Curva Original Base', line=dict(color='gray', dash='dash')))
        fig_comp_a.add_trace(go.Scatter(x=anios_label, y=val_final_a, mode='lines+markers', name='Curva con Estrés Aplicado', line=dict(color='darkblue', width=3)))
        fig_comp_a.update_layout(title="Comparativa Anual 2027-2031: Base vs. Estrés", yaxis_tickformat="$,.0f", hovermode="x unified")
        st.plotly_chart(fig_comp_a, use_container_width=True)

    with tab_est2:
        st.subheader("Composición por Clasificación (2027-2031)")
        df_melt = df_graficos[['Classif'] + cols_final_a].melt(id_vars=['Classif'], var_name='Año', value_name='Monto')
        df_melt['Año'] = df_melt['Año'].str.replace('Final_FY', '20')
        df_g_anual = df_melt.groupby(['Año', 'Classif'])['Monto'].sum().reset_index()

        fig_barras = px.bar(
            df_g_anual, 
            x="Año", y="Monto", color="Classif", 
            title="Presupuesto Quinquenal Estresado Desglosado", 
            color_discrete_sequence=modern_colors
        )
        fig_barras.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_barras, use_container_width=True)

    with tab_est3:
        st.markdown("**Inspector Semántico:** Revisa qué celdas detectó el algoritmo basándose en las descripciones y la clasificación.")
        df_verif = df_graficos[cols_existentes + ['Factor_Estrés_Fila', 'Base_FY27', 'Final_FY27']].copy()
        df_verif = df_verif[df_verif['Factor_Estrés_Fila'] != 1.0]
        
        if df_verif.empty:
            st.info("Ninguna fila bajo los filtros actuales fue alterada por los parámetros de estrés.")
        else:
            st.dataframe(
                df_verif.head(200), 
                use_container_width=True,
                column_config={
                    "Base_FY27": st.column_config.NumberColumn("Base Original FY27", format="$%.0f"),
                    "Final_FY27": st.column_config.NumberColumn("Estresado FY27", format="$%.0f"),
                    "Factor_Estrés_Fila": st.column_config.NumberColumn("Multiplicador", format="%.3f")
                }
            )

    with tab_est4:
        st.subheader("Motor de Reportes Excel (XlsxWriter)")
        st.markdown("El sistema genera un archivo Excel que incluye la tabla de datos consolidada, el cuadro paramétrico de sensibilidades y gráficos nativos de Excel.")
        
        from io import BytesIO
        output_excel = BytesIO()
        
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
            # Exportamos el DF total (no solo el filtrado) para mantener la integridad de la base
            df_final_proy.to_excel(writer, sheet_name="Proyeccion_Estrategica", index=False)
            
            workbook = writer.book
            worksheet = writer.sheets["Proyeccion_Estrategica"]
            
            money_fmt = workbook.add_format({'num_format': '$#,##0'})
            bold_fmt = workbook.add_format({'bold': True})
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
            
            for col_num in range(5, len(df_final_proy.columns)):
                worksheet.set_column(col_num, col_num, 15, money_fmt)
            
            start_col = len(df_final_proy.columns) + 2
            
            worksheet.write(1, start_col, "Tabla de Sensibilidad", header_fmt)
            worksheet.write(1, start_col+1, "Valor Aplicado", header_fmt)
            worksheet.write(2, start_col, "Variación Combustible", bold_fmt)
            worksheet.write(2, start_col+1, f"{slider_fuel_pct}%")
            worksheet.write(3, start_col, "Variación Energía", bold_fmt)
            worksheet.write(3, start_col+1, f"{slider_power_pct}%")
            worksheet.write(4, start_col, "Variación Dólar", bold_fmt)
            worksheet.write(4, start_col+1, f"{slider_dolar_pct}%")
            worksheet.write(5, start_col, "Variación Mano de Obra", bold_fmt)
            worksheet.write(5, start_col+1, f"{slider_labor_pct}%")

            worksheet.write(8, start_col, "Año", header_fmt)
            worksheet.write(8, start_col+1, "Gasto Total (USD)", header_fmt)
            
            totals = [df_final_proy[f'Final_{a}'].sum() for a in años_quinquenio]
            
            for i, (año, tot) in enumerate(zip(['2027', '2028', '2029', '2030', '2031'], totals)):
                worksheet.write(9+i, start_col, año)
                worksheet.write(9+i, start_col+1, tot, money_fmt)

            chart = workbook.add_chart({'type': 'column'})
            chart.add_series({
                'name': 'Proyección Quinquenal',
                'categories': ['Proyeccion_Estrategica', 9, start_col, 13, start_col],
                'values':     ['Proyeccion_Estrategica', 9, start_col+1, 13, start_col+1],
                'data_labels': {'value': True},
                'fill':   {'color': '#4F81BD'}
            })
            chart.set_title({'name': 'Evolución del Presupuesto (2027-2031)'})
            chart.set_x_axis({'name': 'Año Operativo'})
            chart.set_y_axis({'name': 'Costo (USD)', 'num_format': '$#,##0'})
            chart.set_size({'width': 550, 'height': 350})
            
            worksheet.insert_chart(16, start_col, chart)
            
        st.download_button(
            label="Descargar Reporte Quinquenal",
            data=output_excel.getvalue(),
            file_name="Planificacion_Estrategica_Visual.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
