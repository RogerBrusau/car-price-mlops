# app.py (Streamlit)
# Ejecutar:
#   streamlit run app.py
#
# Requisitos:
#   pip install streamlit fastapi uvicorn  (fastapi/uvicorn no hacen falta aquí)
#   pip install streamlit
#
# Estructura esperada:
#   baseline_train.py
#   predict.py
#   app.py
#   artifacts/baseline_quantiles.joblib

import json
import streamlit as st

from predict import predict_price, load_models

st.set_page_config(page_title="Tasador coche segunda mano", layout="centered")

@st.cache_resource
def _load():
    # carga una vez y queda cacheado
    return load_models()

_load()

st.title("Tasador de coche (baseline)")

st.write("Pega un JSON con los datos del coche y obtén p10, p50 y p90 en euros.")

default_json = {
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
    "equipment_list": "navegador; apple carplay; android auto; control crucero; cámara trasera"
}

txt = st.text_area(
    "JSON del coche",
    value=json.dumps(default_json, ensure_ascii=False, indent=2),
    height=320
)

col1, col2 = st.columns(2)

with col1:
    run = st.button("Calcular precio", type="primary")

with col2:
    st.caption("El modelo usa artifacts/baseline_quantiles.joblib (o MODEL_PATH).")

if run:
    try:
        car = json.loads(txt)
        if not isinstance(car, dict):
            st.error("El JSON debe ser un objeto (dict), no una lista.")
        else:
            res = predict_price(car)
            st.subheader("Resultado")
            st.metric("p50 (precio central)", f"{res['p50_eur']:,.0f} €".replace(",", "."))
            st.write(
                f"Rango (p10–p90): **{res['p10_eur']:,.0f} € – {res['p90_eur']:,.0f} €**".replace(",", ".")
            )
            st.json(res)
    except json.JSONDecodeError as e:
        st.error(f"JSON inválido: {e}")
    except Exception as e:
        st.error(f"Error al predecir: {e}")
