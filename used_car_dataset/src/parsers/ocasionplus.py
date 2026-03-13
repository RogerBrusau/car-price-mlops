import json
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# --- CONFIGURACIÓN DE MARCAS PARA FALLBACK URL ---
KNOWN_MAKES = [
    "alfa-romeo", "aston-martin", "land-rover", "mercedes-benz", "rolls-royce",
    "dr-automobiles", "lynk-co", "cupra", "dacia", "tesla", "smart", "mini",
    "abarth", "audi", "bmw", "byd", "citroen", "ds", "fiat", "ford", 
    "honda", "hyundai", "infiniti", "isuzu", "jaguar", "jeep", "kia", "lancia", "lexus", 
    "mazda", "mg", "mitsubishi", "nissan", "opel", "peugeot", "porsche", "renault", 
    "seat", "skoda", "ssangyong", "subaru", "suzuki", "toyota", "volkswagen", "volvo", 
    "polestar", "kgm", "swm", "giottiline", "iveco", "man", "mercedes",
    "niesmann-and-bischoff", "benimar"
]

MAKE_NORMALIZATION = {
    "mercedes": "Mercedes-Benz", "dr-automobiles": "DR Automobiles",
    "lynk-co": "Lynk & Co", "kgm": "KGM", "swm": "SWM", "bmw": "BMW",
    "mg": "MG", "ds": "DS", "honda": "Honda", "seat": "SEAT",
    "fiat": "Fiat", "kia": "Kia", "niesmann-and-bischoff": "Niesmann+Bischoff"
}

def _clean_number(text: Optional[str]) -> Optional[int]:
    """Limpia textos como '43.514 Km' o '17.990 €' a enteros."""
    if not text: return None
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None

def _extract_from_url(url: str) -> Dict:
    """Fallback: Extrae datos de la URL si el JSON falla."""
    data = {}
    try:
        path = urlparse(url).path
        slug = path.strip("/").split("/")[-1]
    except: return data

    # Año y KM de la URL
    year_match = re.search(r'-(\d{4})-[a-z0-9]+$', slug)
    if year_match:
        data["year"] = int(year_match.group(1))
        slug = slug[:year_match.start()] # Cortar para limpiar
        
    km_match = re.search(r'-con-(\d+)km', slug)
    if km_match:
        data["mileage_km"] = int(km_match.group(1))
        slug = slug[:km_match.start()]

    # Marca y Modelo
    for make_slug in KNOWN_MAKES:
        if slug.lower().startswith(make_slug + "-") or slug.lower() == make_slug:
            raw_make = make_slug.replace("-", " ").title()
            data["make"] = MAKE_NORMALIZATION.get(make_slug, raw_make)
            
            rest = slug[len(make_slug):].strip("-")
            if rest:
                parts = rest.split("-")
                if make_slug == "mercedes" and parts[0] == "clase" and len(parts) > 1:
                    data["model"] = f"Clase {parts[1].upper()}"
                    data["trim_raw"] = " ".join(parts[2:]) if len(parts) > 2 else None
                else:
                    data["model"] = parts[0].capitalize()
                    data["trim_raw"] = " ".join(parts[1:]) if len(parts) > 1 else None
            break
    return data

def _extract_nextjs_json(html: str) -> Optional[Dict]:
    """Intenta extraer el JSON de Next.js (método frágil pero rico en datos)."""
    candidates = re.findall(r'self\.__next_f\.push\(\[[0-9]+,"(.*?)"\]\)', html, re.DOTALL)
    for candidate in candidates:
        unescaped = candidate.replace('\\"', '"').replace('\\\\', '\\')
        if '"vehicle":{' in unescaped:
            start_marker = '"vehicle":{'
            start_idx = unescaped.find(start_marker)
            if start_idx == -1: continue
            obj_start = start_idx + len('"vehicle":')
            balance = 0
            for i in range(obj_start, len(unescaped)):
                if unescaped[i] == '{': balance += 1
                elif unescaped[i] == '}':
                    balance -= 1
                    if balance == 0:
                        try:
                            return json.loads(unescaped[obj_start : i+1])
                        except: pass
                        break
    return None

def _extract_price_from_html(html: str) -> Optional[int]:
    """Busca el precio visual en el HTML (método robusto)."""
    # Patrón típico: 17.900 € o 17.900€
    # Buscamos precios grandes (>1000) para evitar cuotas mensuales
    prices = re.findall(r'(\d{1,3}(?:\.\d{3})*)\s*€', html)
    for p in prices:
        val = _clean_number(p)
        if val and val > 1000 and val < 200000: # Rango razonable
            return val
    return None

def parse(html_bytes: bytes, url: str = None, config: dict = None) -> Dict[str, Any]:
    if not html_bytes: return {}
    html = html_bytes.decode("utf-8", errors="ignore")
    
    # 0. Descartar basura
    if "coche-vendido" in (url or "") or "<title>Vehículo vendido" in html:
        return {"error": "sold_vehicle"}

    data = {}

    # ---------------------------------------------------------
    # ESTRATEGIA 1: JSON (Datos Puros)
    # ---------------------------------------------------------
    vehicle_json = _extract_nextjs_json(html)
    if vehicle_json:
        data["make"] = vehicle_json.get("brand", {}).get("name")
        data["model"] = vehicle_json.get("model", {}).get("name")
        data["trim_raw"] = vehicle_json.get("version")
        price_data = vehicle_json.get("price", {})
        data["price_eur"] = price_data.get("cash") or price_data.get("withoutFinancing")
        data["source_id"] = vehicle_json.get("id")
        if vehicle_json.get("images"):
            data["image_url"] = vehicle_json.get("images")[0].get("medium")

    # ---------------------------------------------------------
    # ESTRATEGIA 2: HTML Tags (Datos Técnicos muy fiables)
    # ---------------------------------------------------------
    if not data.get("year"):
        m = re.search(r'data-test="span-registration-date"[^>]*>(\d+)<', html)
        if m: data["year"] = int(m.group(1))

    if not data.get("mileage_km"):
        m = re.search(r'data-test="span-km"[^>]*>([^<]+)<', html)
        if m: data["mileage_km"] = _clean_number(m.group(1))

    # Combustible y Cambio
    m_fuel = re.search(r'data-test="span-fuel-type"[^>]*>([^<]+)<', html)
    if m_fuel: data["fuel"] = m_fuel.group(1).strip()
    
    m_trans = re.search(r'data-test="span-engine-transmission"[^>]*>([^<]+)<', html)
    if m_trans: data["transmission"] = m_trans.group(1).strip()

    # Provincia
    if not data.get("province"):
        m_prov = re.search(r' en ([^<|\n]+)(?:<|\|)', html) # Busca "Seat Ibiza... en Madrid"
        if m_prov: data["province"] = m_prov.group(1).strip()

    # ---------------------------------------------------------
    # ESTRATEGIA 3: FALLBACKS (Si todo lo anterior falló)
    # ---------------------------------------------------------
    
    # Si falta el Precio (Vital)
    if not data.get("price_eur"):
        data["price_eur"] = _extract_price_from_html(html)

    # Si falta Marca/Modelo (Vital) -> Usar URL
    if url and (not data.get("make") or not data.get("model")):
        url_data = _extract_from_url(url)
        # Solo rellenamos lo que falte
        for k, v in url_data.items():
            if not data.get(k): data[k] = v

    return data