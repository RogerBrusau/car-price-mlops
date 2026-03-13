# predict.py
# Uso: integrar en FastAPI para predecir p10, p50, p90 con el modelo guardado en joblib.

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Importa la misma limpieza y feature engineering que usaste al entrenar
# Asegúrate de que baseline_train.py esté en el mismo directorio o en el PYTHONPATH
from baseline_train import clean_and_engineer


CAT_COLS = [
    "make_clean", "model_clean", "trim_clean",
    "fuel_clean", "transmission_clean", "body_type_clean",
    "province_clean", "seller_type_clean", "emissions_clean",
]

NUM_COLS = [
    "mileage_km_clean", "power_kw_clean", "engine_cc_clean",
    "vehicle_age_years", "listing_age_days",
    "owners_clean", "accidents_clean", "damages_clean",
    "equip_count", "has_navegador", "has_carplay", "has_android_auto", "has_cruise", "has_camera",
]

FEATURE_COLS = CAT_COLS + NUM_COLS

_DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/baseline_quantiles.joblib")

_MODELS_CACHE: Optional[dict] = None


def load_models(model_path: str = _DEFAULT_MODEL_PATH) -> dict:
    """
    Carga el diccionario de modelos cuantílicos: {'p10': pipe, 'p50': pipe, 'p90': pipe}.
    Usa caché en memoria para que FastAPI no recargue el archivo en cada request.
    """
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        _MODELS_CACHE = joblib.load(model_path)
        if not isinstance(_MODELS_CACHE, dict) or not {"p10", "p50", "p90"}.issubset(_MODELS_CACHE.keys()):
            raise ValueError("El joblib no contiene el dict esperado con claves: p10, p50, p90.")
    return _MODELS_CACHE


def predict_price(car_dict: Dict[str, Any], model_path: str = _DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Recibe un JSON como dict con campos tipo scraping y devuelve rango p10, p50, p90.

    car_dict puede contener:
    make / make_raw, model / model_raw, trim_raw
    fuel / fuel_raw, transmission / transmission_raw, body_type / body_type_raw
    province / province_raw, seller_type / seller_type_raw, emissions_label / emissions_label_raw
    mileage_km o mileage_raw, power_kw o power_raw, engine_size_cc o engine_raw
    first_registration_date o first_registration_raw
    listing_date o listing_date_raw
    scrape_datetime (recomendado, si no lo das se quedará NaT y algunas features serán NaN)
    equipment_list (separado por ;) o equipment_raw
    owners_count / owners_raw, accidents_count / accidents_raw, damages_flag / damages_flag_raw
    price_eur o price_raw (no hace falta para predecir)

    Devuelve:
    {'p10_eur': ..., 'p50_eur': ..., 'p90_eur': ..., 'model_version': 'baseline_quantiles'}
    """
    models = load_models(model_path=model_path)

    if not isinstance(car_dict, dict):
        raise TypeError("car_dict debe ser un dict (JSON ya parseado).")

    df = pd.DataFrame([car_dict])
    dfc = clean_and_engineer(df)

    # Asegura que están todas las columnas esperadas para el modelo
    for c in FEATURE_COLS:
        if c not in dfc.columns:
            dfc[c] = pd.NA

    X = dfc[FEATURE_COLS]

    p10 = float(models["p10"].predict(X)[0])
    p50 = float(models["p50"].predict(X)[0])
    p90 = float(models["p90"].predict(X)[0])

    lo, hi = (p10, p90) if p10 <= p90 else (p90, p10)

    return {
        "p10_eur": lo,
        "p50_eur": p50,
        "p90_eur": hi,
        "model_version": "baseline_quantiles",
    }


if __name__ == "__main__":
    # Ejemplo rápido por CLI
    example = {
        "scrape_datetime": "2026-01-10T10:12:00",
        "first_registration_date": "2018-05-01",
        "make": "BMW",
        "model": "Serie 3",
        "trim_raw": "Sport",
        "fuel": "diésel",
        "transmission": "automático",
        "body_type": "berlina",
        "province": "Barcelona",
        "seller_type": "professional",
        "emissions_label": "C",
        "mileage_raw": "145.000 km",
        "power_raw": "190 CV",
        "engine_raw": "1995 cc",
        "equipment_list": "navegador; apple carplay; android auto; control crucero; cámara trasera",
    }
    print(predict_price(example))
