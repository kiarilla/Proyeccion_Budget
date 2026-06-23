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

# BLOQUEO ESTRUCTURAL: Si no hay archivo subido en esta sesión, la aplicación se detiene.
if uploaded_file is None:
    st.info("👋 ¡Bienvenido a la Plataforma de Control de Gestión y Planificación Estratégica!")
    st.warning("El sistema se encuentra en pausa. Por favor, suba el archivo Excel maestro en el panel lateral izquierdo para comenzar el procesamiento de datos.")
    st.stop()
else:
    # Validación para reescribir la memoria solo cuando se sube un archivo nuevo
    if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
        st.session_state.last_uploaded_name = uploaded_file.name
        st.cache_data.clear()
        clear_cache()
        # Sobreescribir cualquier archivo local anterior con la nueva data
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
    """Carga y prepara todos los DataFrames necesarios para los módulos."""
    forecast_df = load_forecast_detail()
    budget_df = load_budget_detail()
    grupos_df = load_grupos_mapping()
    pivot_df = load_pivot_summary()
    forecast_merged, budget_merged = get_merged_data()
    return forecast_df, budget_df, grupos_df, pivot_df, forecast_merged, budget_merged

with st.spinner("Inicializando modelos de datos..."):
    forecast_df, budget_df, grupos_df, pivot_df, forecast_merged, budget_merged = cargar_datos()

# Inicialización de variables de estado de sesión
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

# Verificación de existencia de caché en disco para evitar reejecución
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
# MÓDULO 1: FORECAST OPERACIONAL 5+7 (INTACTO CON TILDES CORREGIDAS)
# ============================================================================
# ============================================================================
if app_mode == "📊 Forecast Operacional (5+7)":

    # Validación de ejecución de modelos
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

            # Guardado en disco para persistencia
            save_backtesting_results(st.session_state.resultados_backtesting)
            save_forecast(st.session_state.forecast_lines)
            save_metadata(st.session_state.metodo_ganador, st.session_state.kpis)
            st.session_state.modelos_ejecutados = True
            st.rerun()

        # Mostrar sidebar reducido mientras no haya modelos
        st.sidebar.title("⛏ Forecast 5+7")
        st.sidebar.info("Ejecute los modelos para ver los resultados.")
        st.stop()

    # Modelos ya cargados -- mostrar boton de re-ejecucion y continuar
    resultados_backtesting = st.session_state.resultados_backtesting
    forecast_lines = st.session_state.forecast_lines
    metodo_ganador = st.session_state.metodo_ganador
    kpis = st.session_state.kpis

    deviation_df = compute_deviations(forecast_lines, compare_vs_official=True)

    # Agregados por dimensión
    agg_vp = aggregate_forecast(forecast_lines, ["VP"])
    agg_classif = aggregate_forecast(forecast_lines, ["Classif"])
    agg_gerencia = aggregate_forecast(forecast_lines, ["Gerencia"])

    # Sidebar - Filtros globales
    st.sidebar.title("Filtros Globales")
    st.sidebar.markdown("---")

    # Filtro de VP
    vps = ["Todas"] + sorted(forecast_merged["VP"].dropna().unique().tolist())
    vp_seleccionada = st.sidebar.selectbox("Vicepresidencia (VP)", vps)

    # Filtro de Classif
    classifs = ["Todas"] + sorted(forecast_merged["Classif"].dropna().unique().tolist())
    classif_seleccionada = st.sidebar.selectbox("Clasificación", classifs)

    # Filtro de CLASS (GRUPOS)
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

    # Botón para re-ejecutar modelos
    if st.sidebar.button("Re-ejecutar Modelos", use_container_width=True):
        clear_cache()
        st.session_state.modelos_ejecutados = False
        st.rerun()

    def filtrar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Aplica filtros del sidebar a un DataFrame."""
        if vp_seleccionada != "Todas" and "VP" in df.columns:
            df = df[df["VP"] == vp_seleccionada]
        if classif_seleccionada != "Todas" and "Classif" in df.columns:
            df = df[df["Classif"] == classif_seleccionada]
        if class_seleccionada != "Todas" and "CLASS" in df.columns:
            df = df[df["CLASS"] == class_seleccionada]
        return df

    # Aplicar filtros
    forecast_lines_f = filtrar_dataframe(forecast_lines)
    deviation_df_f = filtrar_dataframe(deviation_df)
    agg_classif_f = filtrar_dataframe(agg_classif) if "Classif" in agg_classif.columns else agg_classif
    agg_gerencia_f = filtrar_dataframe(agg_gerencia) if "Gerencia" in agg_gerencia.columns else agg_gerencia

    # Actualizar KPIs con datos filtrados
    if vp_seleccionada != "Todas" or classif_seleccionada != "Todas":
        kpis_f = compute_kpis(forecast_lines_f, forecast_df)
    else:
        kpis_f = kpis

    # Secciones (tabs)
    tabs = st.tabs([
        "1. Resumen Ejecutivo",
        "2. Análisis por Dimensión",
        "3. Tendencia Mensual",
        "4. Forecast 5+7",
        "5. Comparaciones",
        "6. Hallazgos",
        "7. Exportar",
    ])

    # Tab 1: Resumen Ejecutivo
    with tabs[0]:
        st.title("Resumen Ejecutivo - Forecast 5+7")
        st.markdown("Proyección no lineal de gastos operacionales (OPEX) -- 5 meses reales + 7 meses proyectados.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Budget FY",
                format_currency(kpis_f["Budget_FY_Total"]),
                delta=None,
            )
        with col2:
            st.metric(
                "Forecast 5+7",
                format_currency(kpis_f["Forecast_5plus7_Total"]),
                delta=f"{kpis_f['Var_vs_Budget_Pct']:+.1f}% vs Budget",
                delta_color="inverse",
            )
        with col3:
            st.metric(
                "Real YTD (Ene-May)",
                format_currency(kpis_f["Real_YTD_Total"]),
                delta=f"{kpis_f['Pct_Avance_Real']:.1f}% del Budget",
            )
        with col4:
            oficial_val = kpis_f.get("Forecast_Oficial_Total", 0) or 0
            var_vs_oficial = kpis_f.get("Var_vs_Oficial_Pct", 0) or 0
            st.metric(
                "Forecast Oficial",
                format_currency(oficial_val),
                delta=f"{var_vs_oficial:+.1f}% vs 5+7",
                delta_color="off",
            )

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Composición por Clasificación")
            fig_treemap = plot_treemap(
                agg_classif_f,
                path=["Classif"],
                value_col="Forecast_5+7",
                title="",
                color_col="Var_Pct",
            )
            st.plotly_chart(fig_treemap, use_container_width=True, key="treemap_resumen")

        with col_b:
            st.subheader("Forecast 5+7 vs Budget por VP")
            fig_barras = plot_bar_comparison(
                agg_vp,
                x_col="VP",
                budget_col="Budget_FY",
                forecast_col="Forecast_5+7",
                title="",
                top_n=10,
            )
            st.plotly_chart(fig_barras, use_container_width=True, key="barras_vp_resumen")

        st.markdown("---")
        st.subheader("Waterfall: Budget FY a Forecast 5+7")
        # Preparar datos para waterfall
        devs = {}
        for _, row in agg_classif_f.iterrows():
            devs[str(row["Classif"])] = row["Var_Abs"]
        fig_waterfall = plot_waterfall(
            kpis_f["Budget_FY_Total"],
            kpis_f["Forecast_5plus7_Total"],
            deviations=devs,
        )
        st.plotly_chart(fig_waterfall, use_container_width=True, key="waterfall_resumen")

    # Tab 2: Análisis por Dimensión
    with tabs[1]:
        st.title("Análisis por Dimensión")
        st.markdown("Desglose del Forecast 5+7 por VP, Gerencia, Clasificación e Ítem.")

        dim_tabs = st.tabs(["Por VP", "Por Gerencia", "Por Classif", "Por CLASS", "Top Items"])

        with dim_tabs[0]:
            st.subheader("Forecast 5+7 por Vicepresidencia")
            fig_vp = plot_bar_comparison(agg_vp, "VP", title="")
            st.plotly_chart(fig_vp, use_container_width=True, key="barras_vp_dim")
            st.dataframe(
                agg_vp[["VP", "Forecast_5+7", "Budget_FY", "Var_Abs", "Var_Pct"]]
                .sort_values("Forecast_5+7", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Forecast_5+7": st.column_config.NumberColumn(format="%.0f"),
                    "Budget_FY": st.column_config.NumberColumn(format="%.0f"),
                    "Var_Abs": st.column_config.NumberColumn(format="%.0f"),
                    "Var_Pct": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

        with dim_tabs[1]:
            st.subheader("Top Gerencias por Forecast 5+7")
            top_ger = agg_gerencia_f.nlargest(15, "Forecast_5+7").sort_values("Forecast_5+7")
            fig_ger = plot_bar_comparison(
                top_ger, "Gerencia", title="",
            )
            st.plotly_chart(fig_ger, use_container_width=True, key="barras_gerencia_dim")
            st.dataframe(
                agg_gerencia_f.sort_values("Forecast_5+7", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Forecast_5+7": st.column_config.NumberColumn(format="%.0f"),
                    "Budget_FY": st.column_config.NumberColumn(format="%.0f"),
                    "Var_Abs": st.column_config.NumberColumn(format="%.0f"),
                    "Var_Pct": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

        with dim_tabs[2]:
            st.subheader("Forecast 5+7 por Clasificación (Classif)")
            fig_classif = plot_bar_comparison(agg_classif_f, "Classif", title="")
            st.plotly_chart(fig_classif, use_container_width=True, key="barras_classif_dim")
            st.dataframe(
                agg_classif_f[["Classif", "Forecast_5+7", "Budget_FY", "Var_Abs", "Var_Pct"]]
                .sort_values("Forecast_5+7", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Forecast_5+7": st.column_config.NumberColumn(format="%.0f"),
                    "Budget_FY": st.column_config.NumberColumn(format="%.0f"),
                    "Var_Abs": st.column_config.NumberColumn(format="%.0f"),
                    "Var_Pct": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

        with dim_tabs[3]:
            st.subheader("Forecast 5+7 por CLASS (Grupos)")
            if "CLASS" in forecast_lines_f.columns:
                agg_class_group = aggregate_forecast(forecast_lines_f, ["CLASS"])
                fig_classg = plot_bar_comparison(agg_class_group, "CLASS", title="")
                st.plotly_chart(fig_classg, use_container_width=True, key="barras_classg_dim")
            else:
                st.info("Columna CLASS no disponible en los datos filtrados.")

        with dim_tabs[4]:
            st.subheader("Top 20 Ítems con Mayor Desviación vs Budget")
            top_dev = top_deviations(deviation_df_f, by="Var_vs_Budget_Abs", n=20)
            fig_top = plot_top_deviations(
                top_dev,
                label_col="Desc Item",
                deviation_col="Var_vs_Budget_Abs",
                pct_col="Var_vs_Budget_Pct",
                title="",
                n=20,
            )
            st.plotly_chart(fig_top, use_container_width=True, key="topdev_dim")

    # Tab 3: Tendencia Mensual
    with tabs[2]:
        st.title("Tendencia Mensual")
        st.markdown("Serie mensual: reales (Ene--May) + proyección no lineal (Jun--Dic).")

        # Preparar series mensuales agregadas
        if not forecast_lines_f.empty:
            forecast_monthly = forecast_lines_f[MONTH_COLS].sum().values
            # Obtener budget mensual de las mismas líneas
            budget_monthly = np.zeros(12)
            official_monthly = np.zeros(12)

            # Budget mensual desde Budget sheet (alineado)
            dim_cols_merge = ["Resp", "Desc Resp", "VP", "Gerencia", "Proc",
                              "Desc Proc", "Item", "Desc Item", "Classif", "CC"]
            budget_for_merge = budget_df[dim_cols_merge + MONTH_COLS].rename(
                columns={c: c + "_b" for c in MONTH_COLS}
            )
            forecast_for_merge = forecast_df[dim_cols_merge + MONTH_COLS].rename(
                columns={c: c + "_o" for c in MONTH_COLS}
            )

            merged_m = forecast_lines_f[dim_cols_merge].merge(
                budget_for_merge, on=dim_cols_merge, how="inner"
            ).merge(
                forecast_for_merge, on=dim_cols_merge, how="inner"
            )

            if not merged_m.empty:
                for i, col in enumerate(MONTH_COLS):
                    budget_monthly[i] = merged_m[f"{col}_b"].sum()
                    official_monthly[i] = merged_m[f"{col}_o"].sum()

            fig_trend = plot_monthly_trend(
                forecast_monthly,
                budget_monthly,
                official_series=official_monthly,
                title="",
            )
            st.plotly_chart(fig_trend, use_container_width=True, key="tendencia_mensual")

            # Tabla de detalle mensual
            st.subheader("Detalle Mensual")
            df_mensual = pd.DataFrame({
                "Mes": MONTH_NAMES,
                "Real / Proyección": forecast_monthly,
                "Budget Mensual": budget_monthly,
                "Forecast Oficial": official_monthly,
            })
            df_mensual["Var vs Budget"] = df_mensual["Real / Proyección"] - df_mensual["Budget Mensual"]
            st.dataframe(
                df_mensual,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Real / Proyección": st.column_config.NumberColumn(format="%.0f"),
                    "Budget Mensual": st.column_config.NumberColumn(format="%.0f"),
                    "Forecast Oficial": st.column_config.NumberColumn(format="%.0f"),
                    "Var vs Budget": st.column_config.NumberColumn(format="%.0f"),
                },
            )
        else:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")

    # Tab 4: Forecast 5+7 (detalle métodos)
    with tabs[3]:
        st.title("Forecast 5+7 - Método y Comparación")
        st.markdown(f"Método ganador (por RMSE en backtesting): **{metodo_ganador}**")

        st.subheader("Tabla Comparativa de Métodos")
        st.dataframe(
            resultados_backtesting.set_index("method"),
            use_container_width=True,
            column_config={
                "mape_mean": st.column_config.NumberColumn("MAPE mean", format="%.1f"),
                "mape_median": st.column_config.NumberColumn("MAPE median", format="%.1f"),
                "rmse_mean": st.column_config.NumberColumn("RMSE mean", format="%.0f"),
                "rmse_median": st.column_config.NumberColumn("RMSE median", format="%.0f"),
                "mae_mean": st.column_config.NumberColumn("MAE mean", format="%.0f"),
                "mae_median": st.column_config.NumberColumn("MAE median", format="%.0f"),
                "n_lines": "Líneas",
            },
        )

        st.subheader("Comparación Visual de Métodos")
        metrica_viz = st.selectbox("Métrica", ["rmse_mean", "mape_median", "mae_mean"])
        fig_comp = plot_method_comparison(resultados_backtesting, metric=metrica_viz)
        st.plotly_chart(fig_comp, use_container_width=True, key="metodo_comparacion")

        st.markdown("---")
        st.subheader("Justificación de la Elección")
        st.markdown(f"""
        **Método seleccionado: `{metodo_ganador}`**

        Criterios de selección (en orden de prioridad):

        1. **Menor RMSE en backtesting**: el método `{metodo_ganador}` obtuvo el menor
           error cuadrático medio al predecir los meses 4-5 usando solo los meses 1-3
           como entrenamiento (validación walk-forward).

        2. **No linealidad**: a diferencia del método lineal (run-rate), este método
           captura la estacionalidad del presupuesto y aplica un factor de amortiguación
           no lineal al ratio de ejecución observado. Esto evita extrapolaciones
           ingenuas que ignoran la realidad operacional minera (mantenciones programadas,
           rampas de producción, estacionalidad de contratos).

        3. **Robustez con datos limitados**: con solo 5 meses de datos reales, métodos
           como ARIMA o Holt-Winters carecen de suficiente información para estimar
           componentes estacionales de forma fiable. El perfil presupuestario escalado
           utiliza la información exógena del Budget como *prior* estacional,
           lo que lo hace más estable.

        4. **Interpretabilidad**: cada proyección se puede explicar como:
           `Proyección = Budget_restante * f(ratio_ejecución)`, donde `f` es una
           función de amortiguación no lineal que converge a 1.0.

        **Nota sobre el método lineal (run-rate)**: aunque obtuvo el mejor MAPE mediano
        en backtesting, proyecta cada mes futuro como el promedio simple de los meses
        pasados, ignorando completamente la forma del presupuesto. Esto lo hace
        inadecuado como forecast operativo, ya que no refleja la realidad de una
        operación minera (ej. un gasto fuerte en Mayo por una mantención mayor no
        debería "contaminar" la proyección de Septiembre).
        """)

        st.markdown("---")
        st.subheader(f"Forecast 5+7 por Línea (método: {metodo_ganador})")
        st.dataframe(
            forecast_lines_f.sort_values("Forecast_5+7", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Forecast_5+7": st.column_config.NumberColumn(format="%.0f"),
                "Budget_FY": st.column_config.NumberColumn(format="%.0f"),
                "Var_Abs": st.column_config.NumberColumn(format="%.0f"),
                "Var_Pct": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    # Tab 5: Comparaciones
    with tabs[4]:
        st.title("Comparaciones")
        st.markdown("Forecast 5+7 vs Budget FY vs Forecast Oficial.")

        comp_df = compare_with_official(forecast_lines_f, group_cols=["Classif"])

        st.subheader("Comparación por Clasificación")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("**Forecast 5+7 vs Budget FY**")
            fig_comp1 
