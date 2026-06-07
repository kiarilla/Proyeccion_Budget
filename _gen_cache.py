"""Generate full cache for deployment."""
from src.data_loader import load_forecast_detail, load_budget_detail
from src.forecast import run_backtesting, select_best_method, project_full_forecast
from src.insights import compute_kpis
from src.model_store import (
    save_backtesting_results, save_forecast, save_metadata, clear_cache,
)

print("Cargando datos...")
f = load_forecast_detail()
b = load_budget_detail()

print("Ejecutando backtesting con 6 metodos (esto tarda ~2-3 min)...")
r = run_backtesting(f, b)  # all 6 methods
best = select_best_method(r, "rmse_mean")
print(f"Ganador: {best}")
print(r.to_string())

print("Generando forecast...")
fl = project_full_forecast(f, b, method=best)
k = compute_kpis(fl, f)
print(f"Budget FY: {k['Budget_FY_Total']:,.0f}")
print(f"Forecast 5+7: {k['Forecast_5plus7_Total']:,.0f}")

print("Guardando cache...")
clear_cache()
save_backtesting_results(r)
save_forecast(fl)
save_metadata(best, k)
print("Cache guardado. Listo para commit y deploy.")
