# baseline_train.py
# Requisitos: pandas, numpy, scikit-learn, joblib

from __future__ import annotations

import re
import json
import inspect
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

import joblib


MISSING_TOKENS = {
    "", "n/d", "nd", "no disponible", "sin especificar", "—", "-", "null",
    "none", "nan", "a convenir", "consultar", "precio a consultar"
}

def _is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in MISSING_TOKENS:
        return True
    return False

def _norm_text(x: object) -> str:
    if _is_missing(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_price_eur(price_raw: object) -> Optional[float]:
    if _is_missing(price_raw):
        return None
    s = str(price_raw).strip().lower()
    if "consult" in s or "a convenir" in s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*k\s*€", s)
    if m:
        return float(m.group(1).replace(",", ".")) * 1000.0
    s2 = s.replace("€", "").replace("eur", "").replace("euros", "").replace(" ", "")
    nums = re.findall(r"\d+", s2.replace(".", ""))
    if not nums:
        return None
    v = float(int("".join(nums)))
    return None if v <= 0 else v

def parse_mileage_km(mileage_raw: object) -> Optional[float]:
    if _is_missing(mileage_raw):
        return None
    s = str(mileage_raw).strip().lower().replace("kms", "").replace("km", "").replace(" ", "")
    nums = re.findall(r"\d+", s.replace(".", ""))
    if not nums:
        return None
    v = float(int("".join(nums)))
    return None if v <= 0 else v

def parse_power_kw(power_raw: object) -> Optional[float]:
    if _is_missing(power_raw):
        return None
    s = str(power_raw).strip().lower().replace(" ", "")
    nums = re.findall(r"\d+(?:[.,]\d+)?", s)
    if not nums:
        return None
    val = float(nums[0].replace(",", "."))
    if "kw" in s:
        return val
    if "cv" in s or "hp" in s:
        return val * 0.73549875
    if 40 <= val <= 450:
        return val * 0.73549875
    return None

def parse_engine_cc(engine_raw: object) -> Optional[float]:
    if _is_missing(engine_raw):
        return None
    s = str(engine_raw).strip().lower().replace(" ", "")
    m = re.search(r"(\d+(?:[.,]\d+)?)l", s)
    if m:
        cc = float(m.group(1).replace(",", ".")) * 1000.0
        return cc if cc >= 500 else None
    nums = re.findall(r"\d+", s.replace(".", ""))
    if not nums:
        return None
    v = float(int("".join(nums)))
    return None if v < 500 else v

def parse_bool_spanish(x: object) -> Optional[int]:
    if _is_missing(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    s = str(x).strip().lower()
    if s in {"sí", "si", "s", "true", "1", "t"}:
        return 1
    if s in {"no", "n", "false", "0", "f"}:
        return 0
    return None

def parse_int_from_text(x: object) -> Optional[int]:
    if _is_missing(x):
        return None
    nums = re.findall(r"\d+", str(x).strip().lower())
    return None if not nums else int(nums[0])

def parse_date_any(x: object) -> pd.Timestamp:
    """
    Parse flexible sin warnings:
    si empieza por año (YYYY-.. o YYYY/..), dayfirst=False; si no, dayfirst=True.
    Devuelve pd.NaT si no se puede parsear (nunca None).
    """
    if _is_missing(x):
        return pd.NaT
    s = re.sub(r"(?i)matric\.?\s*", "", str(x)).strip()
    if not s:
        return pd.NaT

    if re.match(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$", s):
        return pd.to_datetime(s, errors="coerce", dayfirst=False)
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected = [
        "listing_id","scrape_datetime",
        "listing_date","listing_date_raw",
        "first_registration_date","first_registration_raw",
        "make","make_raw","model","model_raw","trim_raw",
        "fuel","fuel_raw","transmission","transmission_raw",
        "body_type","body_type_raw",
        "province","province_raw",
        "seller_type","seller_type_raw",
        "emissions_label","emissions_label_raw",
        "price_eur","price_raw",
        "mileage_km","mileage_raw",
        "power_kw","power_raw",
        "engine_size_cc","engine_raw",
        "owners_count","owners_raw",
        "accidents_count","accidents_raw",
        "damages_flag","damages_flag_raw",
        "equipment_list"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # --- CORRECCIÓN FECHAS Y ZONAS HORARIAS ---
    df["scrape_dt"] = pd.to_datetime(df["scrape_datetime"], errors="coerce", utc=True).dt.tz_localize(None)

    # Procesar listing_dt de forma segura
    df["listing_dt"] = pd.to_datetime(df["listing_date"], errors="coerce")
    m = df["listing_dt"].isna()
    df.loc[m, "listing_dt"] = pd.to_datetime(df.loc[m, "listing_date_raw"].map(parse_date_any), errors="coerce")
    df["listing_dt"] = pd.to_datetime(df["listing_dt"], errors="coerce").dt.tz_localize(None)

    # Procesar first_reg_dt de forma segura
    df["first_reg_dt"] = pd.to_datetime(df["first_registration_date"], errors="coerce")
    m = df["first_reg_dt"].isna()
    df.loc[m, "first_reg_dt"] = pd.to_datetime(df.loc[m, "first_registration_raw"].map(parse_date_any), errors="coerce")
    df["first_reg_dt"] = pd.to_datetime(df["first_reg_dt"], errors="coerce").dt.tz_localize(None)

    # Si aún no hay first_reg_dt, usamos la columna 'year' si existe
    if 'year' in df.columns:
        m2 = df["first_reg_dt"].isna() & df["year"].notna()
        df.loc[m2, "first_reg_dt"] = pd.to_datetime(df.loc[m2, "year"].astype(int).astype(str) + "-07-01", format="%Y-%m-%d", errors="coerce")
    
    # --- CORRECCIÓN TIPOS NUMÉRICOS ---
    df["price_eur_clean"] = pd.to_numeric(df["price_eur"], errors="coerce")
    m = df["price_eur_clean"].isna()
    df.loc[m, "price_eur_clean"] = pd.to_numeric(df.loc[m, "price_raw"].map(parse_price_eur), errors="coerce")

    df["mileage_km_clean"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    m = df["mileage_km_clean"].isna()
    df.loc[m, "mileage_km_clean"] = pd.to_numeric(df.loc[m, "mileage_raw"].map(parse_mileage_km), errors="coerce")

    df["power_kw_clean"] = pd.to_numeric(df["power_kw"], errors="coerce")
    m = df["power_kw_clean"].isna()
    df.loc[m, "power_kw_clean"] = pd.to_numeric(df.loc[m, "power_raw"].map(parse_power_kw), errors="coerce")
    
    # Si power_kw_clean sigue vacío pero existe power_cv, lo convertimos
    if 'power_cv' in df.columns:
        m_cv = df["power_kw_clean"].isna() & pd.to_numeric(df["power_cv"], errors="coerce").notna()
        df.loc[m_cv, "power_kw_clean"] = pd.to_numeric(df.loc[m_cv, "power_cv"], errors="coerce") * 0.73549875

    df["engine_cc_clean"] = pd.to_numeric(df["engine_size_cc"], errors="coerce")
    m = df["engine_cc_clean"].isna()
    df.loc[m, "engine_cc_clean"] = pd.to_numeric(df.loc[m, "engine_raw"].map(parse_engine_cc), errors="coerce")

    df["owners_clean"] = pd.to_numeric(df["owners_count"], errors="coerce")
    m = df["owners_clean"].isna()
    df.loc[m, "owners_clean"] = pd.to_numeric(df.loc[m, "owners_raw"].map(parse_int_from_text), errors="coerce")

    df["accidents_clean"] = pd.to_numeric(df["accidents_count"], errors="coerce")
    m = df["accidents_clean"].isna()
    df.loc[m, "accidents_clean"] = pd.to_numeric(df.loc[m, "accidents_raw"].map(parse_int_from_text), errors="coerce")

    df["damages_clean"] = pd.to_numeric(df["damages_flag"], errors="coerce")
    m = df["damages_clean"].isna()
    df.loc[m, "damages_clean"] = pd.to_numeric(df.loc[m, "damages_flag_raw"].map(parse_bool_spanish), errors="coerce")

    # --- FECHAS Y ANTIGÜEDAD ---
    df["vehicle_age_years"] = (df["scrape_dt"] - df["first_reg_dt"]).dt.days / 365.25
    df.loc[df["vehicle_age_years"] < 0, "vehicle_age_years"] = np.nan

    df["listing_age_days"] = (df["scrape_dt"] - df["listing_dt"]).dt.days
    df.loc[df["listing_age_days"] < 0, "listing_age_days"] = np.nan

    def equip_count(x):
        if _is_missing(x):
            return 0
        parts = [p.strip() for p in str(x).split(";") if p.strip()]
        return len(parts)

    def has_token(x, token):
        if _is_missing(x):
            return 0
        return int(token in str(x).lower())

    df["equip_count"] = df["equipment_list"].map(equip_count)
    df["has_navegador"] = df["equipment_list"].map(lambda x: has_token(x, "navegador"))
    df["has_carplay"] = df["equipment_list"].map(lambda x: has_token(x, "carplay"))
    df["has_android_auto"] = df["equipment_list"].map(lambda x: has_token(x, "android auto"))
    df["has_cruise"] = df["equipment_list"].map(lambda x: has_token(x, "crucero"))
    df["has_camera"] = df["equipment_list"].map(lambda x: has_token(x, "cámara"))

    def pick_norm(norm_col, raw_col):
        a = df[norm_col].map(_norm_text)
        b = df[raw_col].map(_norm_text)
        out = np.where(a != "", a, b)
        out = pd.Series(out, index=df.index).str.strip().str.lower()
        out = out.str.replace(r"\s+", " ", regex=True)
        return out.replace({"": "missing"})

    df["make_clean"] = pick_norm("make", "make_raw")
    df["model_clean"] = pick_norm("model", "model_raw")
    df["trim_clean"] = df["trim_raw"].map(_norm_text).str.strip().str.lower().replace({"": "missing"})
    df["fuel_clean"] = pick_norm("fuel", "fuel_raw")
    df["transmission_clean"] = pick_norm("transmission", "transmission_raw")
    df["body_type_clean"] = pick_norm("body_type", "body_type_raw")
    df["province_clean"] = pick_norm("province", "province_raw")
    df["seller_type_clean"] = pick_norm("seller_type", "seller_type_raw")
    df["emissions_clean"] = pick_norm("emissions_label", "emissions_label_raw")

    if "listing_id" in df.columns:
        df["_scrape_sort"] = df["scrape_dt"].fillna(pd.Timestamp("1900-01-01"))
        ok_id = df["listing_id"].notna() & (df["listing_id"].astype(str).str.len() > 0)
        if ok_id.any():
            df = df.sort_values("_scrape_sort").drop_duplicates(subset=["listing_id"], keep="last")

    # outliers duros a NaN
    df.loc[(df["price_eur_clean"] < 200) | (df["price_eur_clean"] > 250000), "price_eur_clean"] = np.nan
    df.loc[(df["mileage_km_clean"] < 1) | (df["mileage_km_clean"] > 700000), "mileage_km_clean"] = np.nan
    df.loc[(df["power_kw_clean"] < 25) | (df["power_kw_clean"] > 350), "power_kw_clean"] = np.nan
    df.loc[(df["engine_cc_clean"] < 600) | (df["engine_cc_clean"] > 6000), "engine_cc_clean"] = np.nan

    return df

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe()),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def make_quantile_regressor(q: float):
    """
    Intenta HistGradientBoostingRegressor con quantile.
    Si no está disponible en tu sklearn, cae a GradientBoostingRegressor.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        sig = inspect.signature(HistGradientBoostingRegressor).parameters
        if "quantile" in sig:
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                learning_rate=0.06,
                max_depth=8,
                max_leaf_nodes=63,
                min_samples_leaf=40,
                random_state=7,
            )
    except Exception:
        pass

    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=q,
        learning_rate=0.06,
        max_depth=5,
        n_estimators=400,
        random_state=7,
    )

def temporal_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split temporal por scrape_dt.
    Train: scrape_dt < cutoff
    Test: scrape_dt >= cutoff
    cutoff se elige para que el test sea aprox test_size del total (por cuantiles).
    """
    d = df.copy()
    d["scrape_dt"] = pd.to_datetime(d["scrape_dt"], errors="coerce")
    d = d.sort_values("scrape_dt")
    dt_non_null = d["scrape_dt"].dropna()
    if len(dt_non_null) == 0:
        # sin fechas, cae a split aleatorio (pero no rompe)
        idx = np.arange(len(d))
        np.random.seed(7)
        np.random.shuffle(idx)
        cut = int((1 - test_size) * len(d))
        train = d.iloc[idx[:cut]]
        test = d.iloc[idx[cut:]]
        return train, test, pd.Timestamp("1900-01-01")

    cutoff = dt_non_null.quantile(1 - test_size)
    train = d[d["scrape_dt"] < cutoff]
    test = d[d["scrape_dt"] >= cutoff]

    # si por empates queda test vacío, fuerza último tramo
    if len(test) == 0:
        cutoff = dt_non_null.iloc[int(0.8 * len(dt_non_null))]
        train = d[d["scrape_dt"] < cutoff]
        test = d[d["scrape_dt"] >= cutoff]

    return train, test, cutoff

def main(input_data: str, out_dir: str = "artifacts", test_size: float = 0.2):
    # Detecta si es csv o parquet
    if input_data.endswith('.parquet'):
        df = pd.read_parquet(input_data)
    else:
        df = pd.read_csv(input_data)
    dfc = clean_and_engineer(df)

    # target definido solo donde hay precio
    dfc = dfc.dropna(subset=["price_eur_clean"]).copy()

    cat_cols = [
        "make_clean","model_clean","trim_clean",
        "fuel_clean","transmission_clean","body_type_clean",
        "province_clean","seller_type_clean","emissions_clean"
    ]
    num_cols = [
        "mileage_km_clean","power_kw_clean","engine_cc_clean",
        "vehicle_age_years","listing_age_days",
        "owners_clean","accidents_clean","damages_clean",
        "equip_count","has_navegador","has_carplay","has_android_auto","has_cruise","has_camera"
    ]

    feat_cols = cat_cols + num_cols

    train_df, test_df, cutoff = temporal_split(dfc, test_size=test_size)

    X_train = train_df[feat_cols]
    y_train = train_df["price_eur_clean"].astype(float)

    X_test = test_df[feat_cols]
    y_test = test_df["price_eur_clean"].astype(float)

    pre = build_preprocessor(cat_cols, num_cols)

    models: Dict[str, Pipeline] = {}
    preds: Dict[str, np.ndarray] = {}

    for name, q in [("p10", 0.10), ("p50", 0.50), ("p90", 0.90)]:
        reg = make_quantile_regressor(q)
        pipe = Pipeline([("pre", pre), ("model", reg)])
        pipe.fit(X_train, y_train)
        yhat = pipe.predict(X_test)
        models[name] = pipe
        preds[name] = yhat

    mae_p50 = mean_absolute_error(y_test, preds["p50"])

    # cobertura del intervalo [p10, p90]
    lower = np.minimum(preds["p10"], preds["p90"])
    upper = np.maximum(preds["p10"], preds["p90"])
    coverage_10_90 = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
    mean_interval_width = float(np.mean(upper - lower))

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, outp / "baseline_quantiles.joblib")

    summary = {
        "rows_raw": int(df.shape[0]),
        "rows_used_after_clean": int(dfc.shape[0]),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "cutoff_scrape_datetime": None if pd.isna(cutoff) else str(pd.Timestamp(cutoff)),
        "mae_p50_eur": float(mae_p50),
        "coverage_p10_p90": coverage_10_90,
        "mean_interval_width_eur": mean_interval_width,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    (outp / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(f"Cutoff temporal: {summary['cutoff_scrape_datetime']}")
    print(f"Train: {summary['rows_train']}  Test: {summary['rows_test']}")
    print(f"MAE p50 (EUR): {mae_p50:,.0f}".replace(",", "."))
    print(f"Cobertura [p10,p90]: {coverage_10_90:.3f}")
    print(f"Anchura media intervalo (EUR): {mean_interval_width:,.0f}".replace(",", "."))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Archivo de datos (csv o parquet)")
    ap.add_argument("--out", default="artifacts", help="Carpeta de salida")
    ap.add_argument("--test_size", type=float, default=0.2, help="Proporción aproximada para test temporal")
    args = ap.parse_args()
    main(args.input, args.out, args.test_size)
