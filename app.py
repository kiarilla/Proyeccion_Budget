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
            fig_comp1 = plot_bar_comparison(
                comp_df,
                x_col="Classif",
                budget_col="Budget_FY",
                forecast_col="Forecast_5plus7",
                title="",
            )
            st.plotly_chart(fig_comp1, use_container_width=True, key="comp_5plus7_vs_budget")

        with col_c2:
            st.markdown("**Forecast 5+7 vs Forecast Oficial**")
            fig_comp2 = plot_bar_comparison(
                comp_df,
                x_col="Classif",
                budget_col="Forecast_Oficial",
                forecast_col="Forecast_5plus7",
                title="",
            )
            st.plotly_chart(fig_comp2, use_container_width=True, key="comp_5plus7_vs_oficial")

        st.subheader("Tabla Comparativa")
        comp_df["Var_5plus7_vs_Budget_Pct"] = np.where(
            comp_df["Budget_FY"].abs() > 0.01,
            comp_df["Var_5plus7_vs_Budget"] / comp_df["Budget_FY"].abs() * 100,
            0,
        )
        comp_df["Var_5plus7_vs_Oficial_Pct"] = np.where(
            comp_df["Forecast_Oficial"].abs() > 0.01,
            comp_df["Var_5plus7_vs_Oficial"] / comp_df["Forecast_Oficial"].abs() * 100,
            0,
        )
        st.dataframe(
            comp_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Budget_FY": st.column_config.NumberColumn(format="%.0f"),
                "Forecast_Oficial": st.column_config.NumberColumn(format="%.0f"),
                "Forecast_5plus7": st.column_config.NumberColumn(format="%.0f"),
                "Var_5plus7_vs_Budget": st.column_config.NumberColumn(format="%.0f"),
                "Var_5plus7_vs_Budget_Pct": st.column_config.NumberColumn(format="%.1f%%"),
                "Var_5plus7_vs_Oficial": st.column_config.NumberColumn(format="%.0f"),
                "Var_5plus7_vs_Oficial_Pct": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        st.markdown("---")
        st.subheader("Top Desviaciones vs Budget (por Ítem)")
        fig_topdev = plot_top_deviations(
            top_deviations(deviation_df_f, by="Var_vs_Budget_Abs", n=20),
            label_col="Desc Item",
            title="",
            n=20,
        )
        st.plotly_chart(fig_topdev, use_container_width=True, key="topdev_comp")

    # Tab 6: Hallazgos y Propuesta de Mejora
    with tabs[5]:
        st.title("Hallazgos y Propuesta de Mejora")

        st.markdown("""
        ## Principales Hallazgos

        ### 1. Ejecución por debajo del presupuesto en la mayoría de las partidas
        El Forecast 5+7 proyecta un cierre de año **aproximadamente un 10% por debajo
        del Budget FY** a nivel agregado. Esto es consistente con un patrón de
        sub-ejecución presupuestaria observado en los primeros 5 meses del año.

        Las partidas con mayor sub-ejecución son:
        - **Spare Parts y S&C**: posiblemente por retrasos en adquisiciones o
          renegociación de contratos.
        - **Expenses y Contractors**: ejecución más lenta de lo presupuestado,
          típico en servicios externalizados donde los contratos se activan
          progresivamente.

        ### 2. Energía (Power) es la única partida sobre el presupuesto
        El gasto en energía eléctrica muestra una ejecución superior al budget,
        reflejando posiblemente:
        - Tarifas eléctricas mayores a las estimadas.
        - Mayor consumo por ramp-up de producción.
        - Costos de potencia contratada no considerados en el budget original.

        ### 3. La estacionalidad del presupuesto es un activo informativo valioso
        El perfil mensual del Budget contiene información sobre mantenciones
        programadas, campañas estacionales y ciclos operacionales que un modelo
        run-rate ignora por completo. Al incorporarlo como *prior* con
        amortiguación no lineal, se obtiene un forecast más realista.

        ### 4. Limitación de datos reales
        Con solo 5 meses de datos reales, métodos estadísticos clásicos (ARIMA,
        Holt-Winters) tienen poca capacidad predictiva. El enfoque híbrido
        (real + perfil presupuestario ajustado) es el más robusto en este contexto.

        ---

        ## Propuesta de Mejora

        ### Corto plazo (próximo ciclo de forecast)
        1. **Revisión de supuestos de ejecución**: ajustar el factor de
           amortiguación (`damp_factor`) por VP o Classif según patrones
           históricos de ejecución.
        2. **Identificar partidas con ejecución atípica**: las partidas con
           ratios real/budget extremos (>2x o <0.5x) deben revisarse manualmente
           para confirmar que no haya errores de imputación o cambios de alcance.
        3. **Forecast móvil mensual**: actualizar la proyección cada mes
           incorporando el nuevo dato real, reduciendo progresivamente la
           incertidumbre.

        ### Mediano plazo (próximo año)
        4. **Modelo por segmento**: entrenar métodos distintos por Classif
           (ej. Holt para Labor que es más estable, perfil escalado para
           Contractors que tiene más variabilidad).
        5. **Incorporar variables exógenas**: precio del cobre, tipo de cambio,
           producción mensual (toneladas), leyes de mineral -- todas afectan
           directamente los costos operacionales.
        6. **Backtesting con datos históricos**: si se dispone de datos de años
           anteriores (Budget 2024 parece existir en la hoja de control), se
           puede hacer validación cruzada temporal más extensa.

        ### Largo plazo (mejora continua)
        7. **Pipeline automatizado**: integrar la carga de datos, proyección y
           visualización en un proceso periódico (mensual) que se ejecute al
           cierre contable.
        8. **Alertas tempranas**: configurar umbrales de desviación que
           disparen alertas cuando una partida se desvía significativamente
           de su senda proyectada.
        """)

        st.info(
            "Estos hallazgos y propuestas se basan exclusivamente en el análisis "
            "cuantitativo de los datos disponibles (Ene--May). Se recomienda "
            "validar con los gerentes de área antes de tomar decisiones."
        )

    # Tab 7: Exportar
    with tabs[6]:
        st.title("Exportar Datos")

        st.subheader("Descargar Forecast 5+7")

        col_x1, col_x2 = st.columns(2)

        with col_x1:
            csv = forecast_lines_f.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name="forecast_5plus7.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_x2:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                forecast_lines_f.to_excel(writer, sheet_name="Forecast_5plus7", index=False)
                agg_classif_f.to_excel(writer, sheet_name="Por_Classif", index=False)
                agg_vp.to_excel(writer, sheet_name="Por_VP", index=False)
                resultados_backtesting.to_excel(writer, sheet_name="Metodos", index=False)
            st.download_button(
                label="Descargar Excel",
                data=output.getvalue(),
                file_name="forecast_5plus7.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("Vista previa - Forecast 5+7")
        st.dataframe(
            forecast_lines_f.sort_values("Forecast_5+7", ascending=False).head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("Resultados de Backtesting")
        st.dataframe(resultados_backtesting, use_container_width=True, hide_index=True)


# ============================================================================
# ============================================================================
# MÓDULO 2: PROYECCIÓN ESTRATÉGICA QUINQUENAL (2027-2031)
# ============================================================================
# ============================================================================

elif app_mode == "📈 Proyección Estratégica (2027-2031)":
    st.title("📈 Tablero Interactivo de Proyección Estratégica y KPIs")
    st.markdown("Modelo de proyección histórica corregida (Basado en el crecimiento orgánico 2024-2026 sin inflación) con sensibilidad a variables operativas clave.")

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
    slider_fuel_pct = st.sidebar.slider("Variación Precio Diésel / Combustible", -100.0, 100.0, val_fuel, step=0.1)
    slider_power_pct = st.sidebar.slider("Variación Tarifa Energía Eléctrica", -100.0, 100.0, val_power, step=0.1)
    slider_dolar_pct = st.sidebar.slider("Variación Tipo de Cambio / USD", -100.0, 100.0, val_dolar, step=0.1)
    slider_labor_pct = st.sidebar.slider("Variación Costo Mano de Obra", -100.0, 100.0, val_labor, step=0.1)

    # NUEVO: Selectores interactivos para definir Año Base y Año Proyectado Objetivo en el Gráfico de Líneas
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Períodos del Gráfico Temporal")
    lista_anios_disponibles = ['2024', '2026', '2027', '2028', '2029', '2030', '2031']
    anio_base_sel = st.sidebar.selectbox("Seleccione Año Base:", lista_anios_disponibles, index=1) # Default 2026
    anio_proy_sel = st.sidebar.selectbox("Seleccione Año Proyectado Objetivo:", [a for a in lista_anios_disponibles if a != anio_base_sel], index=1) # Default 2027

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
    df_estrat = b26[cols_existentes].copy()

    df_estrat = df_estrat.merge(b24[['CC', 'FY24']], on='CC', how='left')
    df_estrat = df_estrat.merge(b26[['CC', 'FY26'] + [f'{m}-26' for m in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]], on='CC', how='left')
    df_estrat.fillna(0, inplace=True)

    f24 = pd.to_numeric(df_estrat['FY24'], errors='coerce').fillna(0)
    f26 = pd.to_numeric(df_estrat['FY26'], errors='coerce').fillna(0)

    # --- MÉTODO ÚNICO: Tendencia Limpia 2024 a 2026 sin Inflación ---
    tasa_crecimiento = np.where(f24 > 0, (f26 / (f24 + 1e-6)) ** (1/2), 1.0).clip(0.95, 1.10)
    
    df_estrat['Base_FY27'] = f26 * tasa_crecimiento
    df_estrat['Base_FY28'] = df_estrat['Base_FY27'] * tasa_crecimiento
    df_estrat['Base_FY29'] = df_estrat['Base_FY28'] * tasa_crecimiento
    df_estrat['Base_FY30'] = df_estrat['Base_FY29'] * tasa_crecimiento
    df_estrat['Base_FY31'] = df_estrat['Base_FY30'] * tasa_crecimiento

    # --- MAPEO SEMÁNTICO POR FILA ---
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

    meses_cal = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in meses_cal:
        m_26 = pd.to_numeric(df_estrat.get(f'{m}-26', 0), errors='coerce').fillna(0)
        df_estrat[f'peso_{m}'] = m_26 / (f26 + 1e-6)
        
    suma_pesos = df_estrat[[f'peso_{m}' for m in meses_cal]].sum(axis=1)
    
    # Generar distribuciones mensuales proyectadas para todos los años del quinquenio basándonos en la curva estacional base
    for m in meses_cal:
        peso_ajustado = np.where(suma_pesos > 0, df_estrat[f'peso_{m}'] / (suma_pesos + 1e-6), 1.0/12.0)
        df_estrat[f'{m}-24'] = df_estrat['FY24'] * peso_ajustado
        df_estrat[f'{m}-27'] = df_estrat['Final_FY27'] * peso_ajustado
        df_estrat[f'{m}-28'] = df_estrat['Final_FY28'] * peso_ajustado
        df_estrat[f'{m}-29'] = df_estrat['Final_FY29'] * peso_ajustado
        df_estrat[f'{m}-30'] = df_estrat['Final_FY30'] * peso_ajustado
        df_estrat[f'{m}-31'] = df_estrat['Final_FY31'] * peso_ajustado

    cols_salida = cols_existentes + [f'{m}-27' for m in meses_cal] + [f'Final_{a}' for a in años_quinquenio]
    df_final_proy = df_estrat[cols_salida].copy()

    # --- TABLERO INTERACTIVO DE KPIs ---
    tot_fy27_base = df_estrat['Base_FY27'].sum()
    tot_fy27_estres = df_estrat['Final_FY27'].sum()
    delta_usd = tot_fy27_estres - tot_fy27_base
    pct_var = (delta_usd / tot_fy27_base * 100) if tot_fy27_base != 0 else 0

    st.markdown("### Resumen de KPIs (Año 2027)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Proyección Base FY27", f"${tot_fy27_base:,.0f}")
    col2.metric(f"Proyección Simulada FY27", f"${tot_fy27_estres:,.0f}")
    col3.metric("Impacto Neto Operativo", f"${delta_usd:,.0f}", f"{pct_var:+.2f}%", delta_color="inverse")
    col4.metric("Escenario de Riesgo", escenario)

    st.markdown("---")
    
    tab_est1, tab_est2, tab_est3 = st.tabs([
        "📊 Gráficos de Proyección",
        "🔍 Detalle de Filas Afectadas",
        "💾 Generar Excel Dinámico"
    ])

    with tab_est1:
        df_melt = df_final_proy[['Classif'] + [f'Final_{a}' for a in años_quinquenio]].melt(id_vars=['Classif'], var_name='Año', value_name='Monto')
        df_melt['Año'] = df_melt['Año'].str.replace('Final_FY', '20')
        df_g_anual = df_melt.groupby(['Año', 'Classif'])['Monto'].sum().reset_index()

        fig_barras = px.bar(
            df_g_anual, 
            x="Año", y="Monto", color="Classif", 
            title="Presupuesto Multianual Reconstruido y Simulado (USD Detallado)", 
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_barras.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_barras, use_container_width=True)

        # --------------------------------------------------------------------
        # NUEVO GRÁFICO AGREGADO: LÍNEAS TEMPORALES MENSUALES (BASE VS PROYECTADO SIMULADO)
        # --------------------------------------------------------------------
        st.markdown(f"#### 📈 Curva Mensual Temporal: Año Base {anio_base_sel} vs Año Proyectado (Simulado) {anio_proy_sel}")
        st.markdown("Las líneas representan los montos agregados mensuales y se recalculan al mover los parámetros de sensibilidad.")

        # Obtener sufijos de columnas según la selección ('24', '26', '27', etc.)
        sufijo_base = anio_base_sel[-2:]
        sufijo_proy = anio_proy_sel[-2:]

        # Construir lista de columnas mensuales correspondientes
        cols_meses_base = [f"{m}-{sufijo_base}" for m in meses_cal]
        cols_meses_proy = [f"{m}-{sufijo_proy}" for m in meses_cal]

        # Calcular totales mensuales consolidados
        totales_mensuales_base = [df_estrat[col].sum() for col in cols_meses_base]
        totales_mensuales_proy = [df_estrat[col].sum() for col in cols_meses_proy]

        # Estructurar Dataframe para el gráfico lineal
        df_lineas_trend = pd.DataFrame({
            "Mes": meses_cal * 2,
            "Monto": totales_mensuales_base + totales_mensuales_proy,
            "Año / Escenario": [f"Año Base ({anio_base_sel})"] * 12 + [f"Año Proyectado Simulado ({anio_proy_sel})"] * 12
        })

        # Mapeo de nombres completos de meses para el Eje X interactivo
        meses_largos = {"Jan": "Ene", "Feb": "Feb", "Mar": "Mar", "Apr": "Abr", "May": "May", "Jun": "Jun", 
                        "Jul": "Jul", "Aug": "Ago", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dec": "Dic"}
        df_lineas_trend["Mes"] = df_lineas_trend["Mes"].map(meses_largos)

        fig_lineas = px.line(
            df_lineas_trend,
            x="Mes",
            y="Monto",
            color="Año / Escenario",
            markers=True,
            title=f"Evolución de Costos Mensuales — Impacto del Escenario ({escenario})",
            color_discrete_map={
                f"Año Base ({anio_base_sel})": "#457b9d",
                f"Año Proyectado Simulado ({anio_proy_sel})": "#e63946" if delta_usd >= 0 else "#2a9d8f"
            }
        )

        fig_lineas.update_layout(
            xaxis_title="Meses del Período",
            yaxis_title="Monto Total General ($)",
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_lineas, use_container_width=True, key="grafico_lineas_mensual_quinquenal")
        # --------------------------------------------------------------------

    with tab_est2:
        st.markdown("**Inspector Semántico:** Revisa qué celdas detectó el algoritmo basándose en las descripciones y la clasificación de mano de obra.")
        df_verif = df_estrat[cols_existentes + ['Factor_Estrés_Fila', 'Base_FY27', 'Final_FY27']].copy()
        df_verif = df_verif[df_verif['Factor_Estrés_Fila'] != 1.0]
        
        st.dataframe(
            df_verif.head(200), 
            use_container_width=True,
            column_config={
                "Base_FY27": st.column_config.NumberColumn("Base Original FY27", format="$%.0f"),
                "Final_FY27": st.column_config.NumberColumn("Simulado FY27", format="$%.0f"),
                "Factor_Estrés_Fila": st.column_config.NumberColumn("Multiplicador", format="%.3f")
            }
        )

    with tab_est3:
        st.subheader("Motor de Reportes Excel (XlsxWriter)")
        st.markdown("El sistema genera un archivo Excel que incluye la tabla de datos, el cuadro paramétrico de sensibilidades separado a la derecha y un gráfico interactivo nativo de Excel con el resumen quinquenal.")
        
        from io import BytesIO
        output_excel = BytesIO()
        
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
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
            label="Descargar Reporte Quinquenal (Incluye Gráficos y Variables en Excel)",
            data=output_excel.getvalue(),
            file_name="Planificacion_Estrategica_Visual.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        )
