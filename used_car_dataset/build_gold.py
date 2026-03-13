import pandas as pd
import glob
import os
import warnings

def build_gold_dataset():
    # Ignorar warnings de pandas al concatenar archivos
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    print("🔍 Buscando historial de lecturas en Silver...")
    files = glob.glob('data/silver/listings/**/*.parquet', recursive=True)
    if not files:
        print("❌ No se encontraron datos en Silver.")
        return
    
    # 1. Cargar todo el historial (las 19.000+ filas)
    df_full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"📊 Total de lecturas en bruto: {len(df_full)}")
    
    # 2. Filtrar solo los válidos (quitar vendidos y errores)
    df_clean = df_full[df_full['parse_ok'] == True].copy()
    
    # 3. DESDUPLICAR: Nos quedamos con la última actualización de cada coche
    df_clean['scrape_datetime'] = pd.to_datetime(df_clean['scrape_datetime'])
    df_clean = df_clean.sort_values('scrape_datetime', ascending=False)
    df_clean = df_clean.drop_duplicates(subset=['url'], keep='first')
    
    print(f"🏎️ Total de coches ÚNICOS y VÁLIDOS para ML: {len(df_clean)}")

    # 4. ADAPTACIÓN PARA EL MODELO (baseline_train.py)
    print("⚙️ Adaptando columnas para el entrenamiento de Machine Learning...")
    
    # El modelo espera 'first_registration_date' (YYYY-MM-DD), nosotros tenemos 'year' (YYYY)
    df_clean['year'] = df_clean['year'].fillna(0).astype(int)
    df_clean['first_registration_date'] = df_clean['year'].apply(lambda x: f"{x}-01-01" if x > 0 else None)
    
    # El modelo espera 'power_raw' (ej: "150 CV"), nosotros tenemos 'power_cv' (ej: 150)
    df_clean['power_cv'] = df_clean['power_cv'].fillna(0).astype(int)
    df_clean['power_raw'] = df_clean['power_cv'].apply(lambda x: f"{x} CV" if x > 0 else None)
    
    # Rellenar columnas vacías que el modelo de ML necesita para no fallar
    cols_to_add = ['listing_date', 'body_type', 'seller_type', 'emissions_label', 
                   'engine_size_cc', 'owners_count', 'accidents_count', 
                   'damages_flag', 'equipment_list']
    for c in cols_to_add:
        if c not in df_clean.columns:
            df_clean[c] = None

    # 5. Guardar en la capa Gold como PARQUET
    os.makedirs('data/gold', exist_ok=True)
    gold_path = 'data/gold/train_ocasionplus.parquet'
    df_clean.to_parquet(gold_path, index=False)
    
    print(f"🎉 ¡Dataset Gold creado con éxito en: {gold_path}!")

if __name__ == "__main__":
    build_gold_dataset()