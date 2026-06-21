"""
app.py -- Aplicacion Streamlit para visualizacion y gestion del Forecast 5+7.

Entrypoint principal. La logica de negocio reside en src/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Asegurar que src este en el path
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

st.set_page_config(
    page_title="Forecast 5+7 - Minera",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Carga de datos (cacheada)
# ============================================================================

@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    """Carga y prepara todos los datos necesarios."""
    forecast_df = load_forecast_detail()
    budget_df = load_budget_detail()
    grupos_df = load_grupos_mapping()
    pivot_df = load_pivot_summary()
    forecast_merged, budget_merged = get_merged_data()
    return forecast_df, budget_df, grupos_df, pivot_df, forecast_merged, budget_merged


# ============================================================================
# Carga inicial con soporte de cache en disco
# ============================================================================

with st.spinner("Cargando datos del workbook..."):
    forecast_df, budget_df, grupos_df, pivot_df, forecast_merged, budget_merged = cargar_datos()

# Inicializar estado de sesion para modelos
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

# Intentar cargar desde cache en disco
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

# Si aun no hay modelos, mostrar boton para ejecutar
if not st.session_state.modelos_ejecutados:
    st.warning(
        "No se encontraron modelos previamente ejecutados. "
        "Haga clic en el boton para ejecutar el backtesting y generar el Forecast 5+7."
    )
    if st.button("Ejecutar Modelos (Backtesting + Forecast)", type="primary", use_container_width=True):
        with st.spinner("Ejecutando backtesting de metodos (esto puede tardar ~60s)..."):
            st.session_state.resultados_backtesting = run_backtesting(forecast_df, budget_df)

        st.session_state.metodo_ganador = select_best_method(
            st.session_state.resultados_backtesting, "rmse_mean"
        )

        with st.spinner(f"Generando Forecast 5+7 con metodo: {st.session_state.metodo_ganador}..."):
            st.session_state.forecast_lines = project_full_forecast(
                forecast_df, budget_df, method=st.session_state.metodo_ganador
            )

        st.session_state.kpis = compute_kpis(st.session_state.forecast_lines, forecast_df)

        # Guardar a disco
        save_backtesting_results(st.session_state.resultados_backtesting)
        save_forecast(st.session_state.forecast_lines)
        save_metadata(st.session_state.metodo_ganador, st.session_state.kpis)
        st.session_state.modelos_ejecutados = True
        st.rerun()

    # Mostrar sidebar reducido mientras no haya modelos
    st.sidebar.title("⛏ Forecast 5+7")
    st.sidebar.info("Ejecute los modelos para ver los resultados.")
    st.stop()


# ============================================================================
# Modelos ya cargados -- mostrar boton de re-ejecucion y continuar
# ============================================================================

resultados_backtesting = st.session_state.resultados_backtesting
forecast_lines = st.session_state.forecast_lines
metodo_ganador = st.session_state.metodo_ganador
kpis = st.session_state.kpis

deviation_df = compute_deviations(forecast_lines, compare_vs_official=True)

# Agregados por dimension
agg_vp = aggregate_forecast(forecast_lines, ["VP"])
agg_classif = aggregate_forecast(forecast_lines, ["Classif"])
agg_gerencia = aggregate_forecast(forecast_lines, ["Gerencia"])

# ============================================================================
# Sidebar - Filtros globales
# ============================================================================

st.sidebar.title("⛏ Forecast 5+7")
st.sidebar.markdown("---")

# Filtro de VP
vps = ["Todas"] + sorted(forecast_merged["VP"].dropna().unique().tolist())
vp_seleccionada = st.sidebar.selectbox("Vicepresidencia (VP)", vps)

# Filtro de Classif
classifs = ["Todas"] + sorted(forecast_merged["Classif"].dropna().unique().tolist())
classif_seleccionada = st.sidebar.selectbox("Clasificacion", classifs)

# Filtro de CLASS (GRUPOS)
if "CLASS" in forecast_merged.columns:
    classes = ["Todas"] + sorted(forecast_merged["CLASS"].dropna().unique().tolist())
    class_seleccionada = st.sidebar.selectbox("CLASS (Grupos)", classes)
else:
    class_seleccionada = "Todas"

st.sidebar.markdown("---")
metodo_seleccionado = st.sidebar.selectbox(
    "Metodo de proyeccion",
    ["linear", "budget_scaled", "polynomial", "holt_damped", "spline_damped", "arima"],
    index=["linear", "budget_scaled", "polynomial", "holt_damped", "spline_damped", "arima"].index(metodo_ganador),
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Metodo ganador (RMSE): **{metodo_ganador}**")

# Boton para re-ejecutar modelos
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

# ============================================================================
# Secciones (tabs)
# ============================================================================

tabs = st.tabs([
    "1. Resumen Ejecutivo",
    "2. Analisis por Dimension",
    "3. Tendencia Mensual",
    "4. Forecast 5+7",
    "5. Comparaciones",
    "6. Hallazgos",
    "7. Exportar",
    "8. Budget 2027-2031",
])

# --------------------------------------------------------------------------
# Tab 1: Resumen Ejecutivo
# --------------------------------------------------------------------------

with tabs[0]:
    st.title("Resumen Ejecutivo - Forecast 5+7")
    st.markdown("Proyeccion no lineal de gastos operacionales (OPEX) -- 5 meses reales + 7 meses proyectados.")

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
        st.subheader("Composicion por Clasificacion")
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

# --------------------------------------------------------------------------
# Tab 2: Analisis por Dimension
# --------------------------------------------------------------------------

with tabs[1]:
    st.title("Analisis por Dimension")
    st.markdown("Desglose del Forecast 5+7 por VP, Gerencia, Clasificacion e Item.")

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
        st.subheader("Forecast 5+7 por Clasificacion (Classif)")
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
        st.subheader("Top 20 Items con Mayor Desviacion vs Budget")
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

# --------------------------------------------------------------------------
# Tab 3: Tendencia Mensual
# --------------------------------------------------------------------------

with tabs[2]:
    st.title("Tendencia Mensual")
    st.markdown("Serie mensual: reales (Ene--May) + proyeccion no lineal (Jun--Dic).")

    # Preparar series mensuales agregadas
    if not forecast_lines_f.empty:
        forecast_monthly = forecast_lines_f[MONTH_COLS].sum().values
        # Obtener budget mensual de las mismas lineas
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
            "Real / Proyeccion": forecast_monthly,
            "Budget Mensual": budget_monthly,
            "Forecast Oficial": official_monthly,
        })
        df_mensual["Var vs Budget"] = df_mensual["Real / Proyeccion"] - df_mensual["Budget Mensual"]
        st.dataframe(
            df_mensual,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Real / Proyeccion": st.column_config.NumberColumn(format="%.0f"),
                "Budget Mensual": st.column_config.NumberColumn(format="%.0f"),
                "Forecast Oficial": st.column_config.NumberColumn(format="%.0f"),
                "Var vs Budget": st.column_config.NumberColumn(format="%.0f"),
            },
        )
    else:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")

# --------------------------------------------------------------------------
# Tab 4: Forecast 5+7 (detalle metodos)
# --------------------------------------------------------------------------

with tabs[3]:
    st.title("Forecast 5+7 - Metodo y Comparacion")
    st.markdown(f"Metodo ganador (por RMSE en backtesting): **{metodo_ganador}**")

    st.subheader("Tabla Comparativa de Metodos")
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
            "n_lines": "Lineas",
        },
    )

    st.subheader("Comparacion Visual de Metodos")
    metrica_viz = st.selectbox("Metrica", ["rmse_mean", "mape_median", "mae_mean"])
    fig_comp = plot_method_comparison(resultados_backtesting, metric=metrica_viz)
    st.plotly_chart(fig_comp, use_container_width=True, key="metodo_comparacion")

    st.markdown("---")
    st.subheader("Justificacion de la Eleccion")
    st.markdown(f"""
    **Metodo seleccionado: `{metodo_ganador}`**

    Criterios de seleccion (en orden de prioridad):

    1. **Menor RMSE en backtesting**: el metodo `{metodo_ganador}` obtuvo el menor
       error cuadratico medio al predecir los meses 4-5 usando solo los meses 1-3
       como entrenamiento (validacion walk-forward).

    2. **No linealidad**: a diferencia del metodo lineal (run-rate), este metodo
       captura la estacionalidad del presupuesto y aplica un factor de amortiguacion
       no lineal al ratio de ejecucion observado. Esto evita extrapolaciones
       ingenuas que ignoran la realidad operacional minera (mantenciones programadas,
       rampas de produccion, estacionalidad de contratos).

    3. **Robustez con datos limitados**: con solo 5 meses de datos reales, metodos
       como ARIMA o Holt-Winters carecen de suficiente informacion para estimar
       componentes estacionales de forma fiable. El perfil presupuestario escalado
       utiliza la informacion exogena del Budget como *prior* estacional,
       lo que lo hace mas estable.

    4. **Interpretabilidad**: cada proyeccion se puede explicar como:
       `Proyeccion = Budget_restante * f(ratio_ejecucion)`, donde `f` es una
       funcion de amortiguacion no lineal que converge a 1.0.

    **Nota sobre el metodo lineal (run-rate)**: aunque obtuvo el mejor MAPE mediano
    en backtesting, proyecta cada mes futuro como el promedio simple de los meses
    pasados, ignorando completamente la forma del presupuesto. Esto lo hace
    inadecuado como forecast operativo, ya que no refleja la realidad de una
    operacion minera (ej. un gasto fuerte en Mayo por una mantencion mayor no
    deberia "contaminar" la proyeccion de Septiembre).
    """)

    st.markdown("---")
    st.subheader(f"Forecast 5+7 por Linea (metodo: {metodo_ganador})")
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

# --------------------------------------------------------------------------
# Tab 5: Comparaciones
# --------------------------------------------------------------------------

with tabs[4]:
    st.title("Comparaciones")
    st.markdown("Forecast 5+7 vs Budget FY vs Forecast Oficial.")

    comp_df = compare_with_official(forecast_lines_f, group_cols=["Classif"])

    st.subheader("Comparacion por Clasificacion")
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
    st.subheader("Top Desviaciones vs Budget (por Item)")
    fig_topdev = plot_top_deviations(
        top_deviations(deviation_df_f, by="Var_vs_Budget_Abs", n=20),
        label_col="Desc Item",
        title="",
        n=20,
    )
    st.plotly_chart(fig_topdev, use_container_width=True, key="topdev_comp")

# --------------------------------------------------------------------------
# Tab 6: Hallazgos y Propuesta de Mejora
# --------------------------------------------------------------------------

with tabs[5]:
    st.title("Hallazgos y Propuesta de Mejora")

    st.markdown("""
    ## Principales Hallazgos

    ### 1. Ejecucion por debajo del presupuesto en la mayoria de las partidas
    El Forecast 5+7 proyecta un cierre de ano **aproximadamente un 10% por debajo
    del Budget FY** a nivel agregado. Esto es consistente con un patron de
    sub-ejecucion presupuestaria observado en los primeros 5 meses del ano.

    Las partidas con mayor sub-ejecucion son:
    - **Spare Parts y S&C**: posiblemente por retrasos en adquisiciones o
      renegociacion de contratos.
    - **Expenses y Contractors**: ejecucion mas lenta de lo presupuestado,
      tipico en servicios externalizados donde los contratos se activan
      progresivamente.

    ### 2. Energia (Power) es la unica partida sobre el presupuesto
    El gasto en energia electrica muestra una ejecucion superior al budget,
    reflejando posiblemente:
    - Tarifas electricas mayores a las estimadas.
    - Mayor consumo por ramp-up de produccion.
    - Costos de potencia contratada no considerados en el budget original.

    ### 3. La estacionalidad del presupuesto es un activo informativo valioso
    El perfil mensual del Budget contiene informacion sobre mantenciones
    programadas, campañas estacionales y ciclos operacionales que un modelo
    run-rate ignora por completo. Al incorporarlo como *prior* con
    amortiguacion no lineal, se obtiene un forecast mas realista.

    ### 4. Limitacion de datos reales
    Con solo 5 meses de datos reales, metodos estadisticos clasicos (ARIMA,
    Holt-Winters) tienen poca capacidad predictiva. El enfoque hibrido
    (real + perfil presupuestario ajustado) es el mas robusto en este contexto.

    ---

    ## Propuesta de Mejora

    ### Corto plazo (proximo ciclo de forecast)
    1. **Revision de supuestos de ejecucion**: ajustar el factor de
       amortiguacion (`damp_factor`) por VP o Classif segun patrones
       historicos de ejecucion.
    2. **Identificar partidas con ejecucion atipica**: las partidas con
       ratios real/budget extremos (>2x o <0.5x) deben revisarse manualmente
       para confirmar que no haya errores de imputacion o cambios de alcance.
    3. **Forecast movil mensual**: actualizar la proyeccion cada mes
       incorporando el nuevo dato real, reduciendo progresivamente la
       incertidumbre.

    ### Mediano plazo (proximo ano)
    4. **Modelo por segmento**: entrenar metodos distintos por Classif
       (ej. Holt para Labor que es mas estable, perfil escalado para
       Contractors que tiene mas variabilidad).
    5. **Incorporar variables exogenas**: precio del cobre, tipo de cambio,
       produccion mensual (toneladas), leyes de mineral -- todas afectan
       directamente los costos operacionales.
    6. **Backtesting con datos historicos**: si se dispone de datos de años
       anteriores (Budget 2024 parece existir en la hoja de control), se
       puede hacer validacion cruzada temporal mas extensa.

    ### Largo plazo (mejora continua)
    7. **Pipeline automatizado**: integrar la carga de datos, proyeccion y
       visualizacion en un proceso periodico (mensual) que se ejecute al
       cierre contable.
    8. **Alertas tempranas**: configurar umbrales de desviacion que
       disparen alertas cuando una partida se desvia significativamente
       de su senda proyectada.
    """)

    st.info(
        "Estos hallazgos y propuestas se basan exclusivamente en el analisis "
        "cuantitativo de los datos disponibles (Ene--May 2025). Se recomienda "
        "validar con los gerentes de area antes de tomar decisiones."
    )

# --------------------------------------------------------------------------
# Tab 7: Exportar
# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# Tab 8: Budget 2027-2031
# --------------------------------------------------------------------------
with tabs[7]:
    st.title("Budget 2027-2031")

    st.markdown(
        "Proyección presupuestaria basada en los valores históricos del Budget."
    )

    tasa_crecimiento = st.slider(
        "Tasa de crecimiento anual (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    tasa = tasa_crecimiento / 100

    if "Budget_FY" in forecast_lines_f.columns:

        budget_base = forecast_lines_f["Budget_FY"].sum()

        budget_2027 = budget_base * (1 + tasa)
        budget_2028 = budget_2027 * (1 + tasa)
        budget_2029 = budget_2028 * (1 + tasa)
        budget_2030 = budget_2029 * (1 + tasa)
        budget_2031 = budget_2030 * (1 + tasa)

        budget_df = pd.DataFrame({
            "Año": [2027, 2028, 2029, 2030, 2031],
            "Budget": [
                budget_2027,
                budget_2028,
                budget_2029,
                budget_2030,
                budget_2031,
            ]
        })

        st.subheader("Proyección Budget")

        st.dataframe(
            budget_df,
            use_container_width=True,
            hide_index=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Budget FY27",
                format_currency(budget_2027)
            )

        with col2:
            st.metric(
                "Budget FY29",
                format_currency(budget_2029)
            )

        with col3:
            st.metric(
                "Budget FY31",
                format_currency(budget_2031)
            )

        import plotly.express as px

        fig = px.line(
            budget_df,
            x="Año",
            y="Budget",
            markers=True,
            title="Evolución Budget 2027-2031"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning(
            "No se encontró la columna Budget_FY para generar el Budget."
        )

from pdf_generator import generar_pdf
import streamlit as st

# df ya cargado en tu app

if st.button("📄 Generar PDF Budget 2027-2031"):

    pdf = generar_pdf(df)

    st.download_button(
        label="⬇️ Descargar PDF",
        data=pdf,
        file_name="Budget_2027_2031.pdf",
        mime="application/pdf"
    )
