
import joblib, json, math, os
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY", "")

BASE_DIR = Path(__file__).parent
model    = joblib.load(BASE_DIR / "crop_model.joblib")
eco      = pd.read_csv(BASE_DIR / "economics.csv").set_index("crop")
with open(BASE_DIR / "model_meta.json") as f:
    meta = json.load(f)
CLASSES  = meta["classes"]
ACCURACY = meta["accuracy"]

FALLBACK_PRICES = {
    "rice":         2183,
    "wheat":        2275,
    "maize":        1962,
    "chickpea":     5440,
    "kidneybeans":  6000,
    "pigeonpeas":   7000,
    "mothbeans":    8558,
    "mungbean":     8558,
    "blackgram":    7400,
    "lentil":       6000,
    "pomegranate":  4500,   # fixed — was 7000
    "banana":       1200,   # fixed — was 2000
    "mango":        2000,   # fixed — was 3000
    "grapes":       3500,   # fixed — was 8000
    "watermelon":    600,   # fixed — was 1200
    "muskmelon":     500,   # fixed — was 1000
    "apple":        5500,   # fixed — was 12000
    "orange":       2500,   # fixed — was 4000
    "papaya":        600,   # fixed — was 1500
    "coconut":      2400,   # fixed — was 3200
    "cotton":       6620,
    "jute":         5050,
    "coffee":      23000,
}

CITY_SOIL_DB = {
    # West Bengal
    "kolkata":            (22.57, 88.36, 82, 36, 214, 6.5),
    "murshidabad":        (24.18, 88.27, 78, 32, 198, 6.3),
    "siliguri":           (26.72, 88.43, 90, 42, 220, 5.8),
    "howrah":             (22.59, 88.31, 80, 34, 210, 6.4),
    "durgapur":           (23.48, 87.32, 76, 31, 192, 6.6),

    # Maharashtra
    "mumbai":             (19.08, 72.88, 68, 28, 156, 7.2),
    "pune":               (18.52, 73.86, 72, 30, 175, 7.5),
    "nagpur":             (21.15, 79.09, 65, 26, 145, 7.8),
    "nashik":             (19.99, 73.79, 70, 29, 168, 7.3),
    "aurangabad":         (19.88, 75.34, 68, 27, 160, 7.6),

    # Tamil Nadu
    "chennai":            (13.08, 80.27, 55, 22, 132, 7.8),
    "coimbatore":         (11.02, 76.97, 62, 25, 148, 6.9),
    "madurai":            (9.93,  78.12, 58, 23, 138, 7.5),
    "trichy":             (10.79, 78.70, 60, 24, 142, 7.2),
    "salem":              (11.65, 78.16, 58, 22, 135, 7.4),

    # Punjab / Haryana
    "amritsar":           (31.63, 74.87, 95, 48, 185, 7.8),
    "ludhiana":           (30.90, 75.85, 92, 46, 180, 7.9),
    "chandigarh":         (30.73, 76.78, 88, 44, 175, 7.7),
    "jalandhar":          (31.33, 75.58, 90, 45, 178, 7.8),

    # UP / Bihar
    "lucknow":            (26.85, 80.95, 85, 38, 190, 7.2),
    "patna":              (25.60, 85.14, 80, 35, 195, 6.8),
    "varanasi":           (25.32, 83.01, 83, 37, 188, 7.0),
    "agra":               (27.18, 78.01, 82, 36, 185, 7.5),
    "kanpur":             (26.46, 80.35, 84, 38, 188, 7.3),
    "allahabad":          (25.44, 81.84, 82, 36, 190, 7.1),

    # Andhra / Telangana
    "hyderabad":          (17.38, 78.49, 62, 24, 142, 7.6),
    "vijayawada":         (16.51, 80.64, 68, 26, 158, 7.2),
    "visakhapatnam":      (17.69, 83.22, 65, 25, 152, 7.0),
    "warangal":           (18.00, 79.59, 64, 25, 148, 7.4),

    # Karnataka
    "bangalore":          (12.97, 77.59, 58, 20, 125, 6.2),
    "mysore":             (12.30, 76.65, 62, 22, 135, 6.5),
    "hubli":              (15.36, 75.12, 65, 24, 142, 7.2),
    "mangalore":          (12.87, 74.84, 70, 28, 162, 5.9),

    # Gujarat
    "ahmedabad":          (23.02, 72.57, 48, 18, 118, 8.1),
    "surat":              (21.17, 72.83, 52, 20, 125, 7.8),
    "vadodara":           (22.31, 73.18, 50, 19, 120, 8.0),
    "rajkot":             (22.30, 70.80, 45, 17, 112, 8.2),

    # Rajasthan
    "jaipur":             (26.91, 75.79, 38, 15,  98, 7.9),
    "jodhpur":            (26.29, 73.02, 32, 12,  88, 8.2),
    "udaipur":            (24.58, 73.71, 40, 16, 102, 7.8),
    "kota":               (25.18, 75.83, 42, 17, 105, 7.7),

    # Himachal / Uttarakhand
    "shimla":             (31.10, 77.17, 75, 32, 165, 5.8),
    "dehradun":           (30.32, 78.03, 78, 34, 172, 6.2),
    "haridwar":           (29.95, 78.16, 80, 35, 175, 7.0),
    "nainital":           (29.38, 79.46, 72, 30, 158, 5.5),

    # Assam / NE India
    "guwahati":           (26.14, 91.74, 92, 44, 225, 5.5),
    "shillong":           (25.57, 91.88, 88, 42, 215, 5.2),
    "dibrugarh":          (27.48, 95.00, 90, 43, 220, 5.4),

    # Kerala
    "kochi":              (9.93,  76.27, 75, 35, 185, 5.8),
    "thiruvananthapuram": (8.52,  76.94, 72, 33, 178, 5.9),
    "kozhikode":          (11.25, 75.78, 78, 36, 190, 5.7),
    "thrissur":           (10.52, 76.21, 76, 35, 186, 5.8),

    # Madhya Pradesh
    "bhopal":             (23.26, 77.40, 65, 25, 148, 7.5),
    "indore":             (22.72, 75.86, 68, 27, 155, 7.3),
    "jabalpur":           (23.18, 79.94, 66, 26, 150, 7.2),
    "gwalior":            (26.22, 78.18, 70, 28, 160, 7.6),

    # Odisha
    "bhubaneswar":        (20.30, 85.82, 72, 30, 175, 6.5),
    "cuttack":            (20.46, 85.88, 75, 32, 180, 6.3),
    "rourkela":           (22.26, 84.86, 70, 29, 170, 6.4),

    # Jharkhand
    "ranchi":             (23.34, 85.31, 68, 27, 165, 5.8),
    "jamshedpur":         (22.80, 86.18, 65, 26, 158, 6.0),

    # Chhattisgarh
    "raipur":             (21.25, 81.63, 70, 28, 168, 6.5),
    "bilaspur":           (22.09, 82.15, 68, 27, 162, 6.4),

    # Delhi / NCR
    "delhi":              (28.67, 77.21, 75, 32, 175, 7.8),
    "noida":              (28.54, 77.39, 76, 33, 178, 7.7),
    "gurugram":           (28.46, 77.03, 74, 31, 172, 7.9),
}

def get_nearest_city_soil(lat: float, lon: float):
    """
    Find nearest city in CITY_SOIL_DB using Euclidean distance.
    Returns (N, P, K, ph, city_name, distance_km)
    """
    best_city = None
    best_dist = float("inf")

    for city, data in CITY_SOIL_DB.items():
        city_lat, city_lon = data[0], data[1]
        dist = math.sqrt(
            ((lat - city_lat) * 111) ** 2 +
            ((lon - city_lon) * 91)  ** 2
        )
        if dist < best_dist:
            best_dist = dist
            best_city = city

    d = CITY_SOIL_DB[best_city]
    return d[2], d[3], d[4], d[5], best_city, round(best_dist, 1)


def is_cold_climate(lat, lon):
    return lat > 30.0 or (lat > 27.0 and lon < 80.0)

def filter_impossible_crops(top_raw, lat, lon, temperature):
    filtered = []
    for crop, prob in top_raw:
        remove = False
        if crop in ["apple", "grapes"]:
            if temperature > 22 or not is_cold_climate(lat, lon):
                remove = True
        if crop == "coffee" and (temperature > 30):
            remove = True
        if not remove:
            filtered.append((crop, prob))
    # Always return at least 5
    if len(filtered) < 5:
        for crop, prob in top_raw:
            if (crop, prob) not in filtered:
                filtered.append((crop, prob))
            if len(filtered) >= 5:
                break
    return filtered[:5]


DISEASE_RULES = {
    # ── Cereals ───────────────────────────────────────────────────────────────
    "rice":        {"humidity_gt":80, "temp_gt":25,
                    "disease":"Brown Leaf Spot / Blast",
                    "action":"Apply Tricyclazole fungicide. Monitor weekly."},
    "wheat":       {"humidity_gt":75, "temp_lt":15,
                    "disease":"Yellow Rust",
                    "action":"Apply Propiconazole at first sign."},
    "maize":       {"humidity_gt":70, "rainfall_gt":100,
                    "disease":"Maydis Leaf Blight",
                    "action":"Ensure drainage. Reduce plant density."},
    # ── Pulses ────────────────────────────────────────────────────────────────
    "mungbean":    {"humidity_gt":80,
                    "disease":"Powdery Mildew",
                    "action":"Spray Wettable Sulphur 0.3%."},
    "blackgram":   {"humidity_gt":80,
                    "disease":"Yellow Mosaic Virus",
                    "action":"Control whitefly with Imidacloprid."},
    "pigeonpeas":  {"humidity_gt":75, "rainfall_gt":80,
                    "disease":"Fusarium Wilt",
                    "action":"Use resistant varieties. Apply Carbendazim soil drench."},
    "chickpea":    {"humidity_gt":70, "temp_lt":18,
                    "disease":"Ascochyta Blight",
                    "action":"Spray Mancozeb 0.25% at first sign of lesions."},
    "lentil":      {"humidity_gt":75, "temp_lt":15,
                    "disease":"Rust / Stemphylium Blight",
                    "action":"Apply Propiconazole. Avoid overhead irrigation."},
    # ── Fruits ────────────────────────────────────────────────────────────────
    "banana":      {"humidity_gt":85, "temp_gt":26,
                    "disease":"Panama Wilt / Sigatoka",
                    "action":"Remove infected leaves. Apply Bordeaux mixture."},
    "papaya":      {"humidity_gt":85, "temp_gt":28,
                    "disease":"Papaya Ring Spot Virus / Anthracnose",
                    "action":"Control aphid vectors with Imidacloprid. Remove infected plants."},
    "mango":       {"humidity_gt":80, "temp_gt":28,
                    "disease":"Anthracnose / Powdery Mildew",
                    "action":"Apply Carbendazim or Wettable Sulphur before flowering."},
    "grapes":      {"humidity_gt":80,
                    "disease":"Downy Mildew / Powdery Mildew",
                    "action":"Apply Metalaxyl + Mancozeb. Ensure good air circulation."},
    "watermelon":  {"humidity_gt":80, "rainfall_gt":80,
                    "disease":"Gummy Stem Blight / Downy Mildew",
                    "action":"Apply Chlorothalonil. Avoid waterlogging."},
    "muskmelon":   {"humidity_gt":80, "rainfall_gt":80,
                    "disease":"Powdery Mildew / Downy Mildew",
                    "action":"Spray Wettable Sulphur 0.3%. Improve drainage."},
    "pomegranate": {"humidity_gt":80, "rainfall_gt":100,
                    "disease":"Bacterial Blight / Fruit Rot",
                    "action":"Apply Copper Oxychloride spray. Remove infected fruits."},
    "orange":      {"humidity_gt":80, "temp_gt":28,
                    "disease":"Citrus Canker / Greening",
                    "action":"Apply Copper-based fungicide. Remove affected branches."},
    "apple":       {"humidity_gt":75, "temp_lt":18,
                    "disease":"Apple Scab / Fire Blight",
                    "action":"Apply Captan or Mancozeb at bud-break stage."},
    "coconut":     {"humidity_gt":85,
                    "disease":"Bud Rot / Root Wilt",
                    "action":"Apply Bordeaux mixture to crown. Remove infected palms."},
    "papaya":      {"humidity_gt":85, "temp_gt":28,
                    "disease":"Papaya Ring Spot Virus / Anthracnose",
                    "action":"Control aphid vectors with Imidacloprid. Remove infected plants."},
    # ── Cash Crops ────────────────────────────────────────────────────────────
    "cotton":      {"humidity_gt":75, "temp_gt":30,
                    "disease":"Bollworm / Leaf Curl Virus",
                    "action":"Apply Bt spray. Monitor pheromone traps. Use Bt cotton varieties."},
    "jute":        {"humidity_gt":85, "rainfall_gt":150,
                    "disease":"Stem Rot / Soft Rot",
                    "action":"Improve drainage. Apply Carbendazim 0.1% spray."},
    "coffee":      {"humidity_gt":85, "temp_lt":18,
                    "disease":"Coffee Leaf Rust",
                    "action":"Apply Copper Oxychloride spray every 3 weeks."},
}


app = FastAPI(
    title="Smart Crop Advisor API",
    description="""
Give your **latitude, longitude, and land area** — get back everything:
- City-specific soil data (N, P, K, pH from ICAR soil survey database)
- Live weather (temperature, humidity, rainfall from Open-Meteo)
- Top 5 crop recommendations (ML model, 99.32% accuracy)
- Yield, revenue, cost, profit, ROI for each crop
- Disease risk alerts based on current weather
    """,
    version="2.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class LocationInput(BaseModel):
    latitude:          float         = Field(..., ge=-90,  le=90,    example=22.5726)
    longitude:         float         = Field(..., ge=-180, le=180,   example=88.3639)
    land_area_acres:   float         = Field(..., gt=0,    le=10000, example=5.0)
    agmarknet_api_key: Optional[str] = Field(None)
    soil_N: Optional[float]          = Field(None, ge=0, le=140,  description="Override N from soil test")
    soil_P: Optional[float]          = Field(None, ge=0, le=200,  description="Override P from soil test")
    soil_K: Optional[float]          = Field(None, ge=0, le=250,  description="Override K from soil test")
    soil_ph: Optional[float]         = Field(None, ge=0, le=14,   description="Override pH from soil test")

class WeatherData(BaseModel):
    temperature_c: float
    humidity_pct:  float
    rainfall_mm:   float
    source:        str

class SoilData(BaseModel):
    N:             float
    P:             float
    K:             float
    ph:            float
    source:        str
    nearest_city:  str
    distance_km:   float

class CropEconomics(BaseModel):
    yield_kg:           int
    yield_low_kg:       int
    yield_high_kg:      int
    price_per_kg_inr:   float
    price_source:       str
    total_cost_inr:     int
    revenue_inr:        int
    profit_inr:         int
    roi_pct:            float
    profit_margin_pct:  float
    breakeven_yield_kg: int
    profitable:         bool

class DiseaseAlert(BaseModel):
    disease: str
    action:  str
    risk:    str

class CropResult(BaseModel):
    rank:           int
    crop:           str
    confidence_pct: float
    season:         str
    category:       str
    duration_days:  int
    water_req_mm:   int
    economics:      CropEconomics
    disease_alert:  Optional[DiseaseAlert]

class FullResponse(BaseModel):
    location:         dict
    land_area_acres:  float
    land_area_ha:     float
    weather:          WeatherData
    soil:             SoilData
    top_5_crops:      List[CropResult]
    best_crop:        str
    best_crop_profit: int
    summary:          str


def fetch_weather(lat, lon):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": ["temperature_2m_max","temperature_2m_min",
                          "precipitation_sum","relative_humidity_2m_max"],
                "timezone": "auto", "past_days": 30, "forecast_days": 1
            }, timeout=30
        )
        if r.status_code == 200:
            d     = r.json()["daily"]
            tmax  = [t for t in d["temperature_2m_max"]        if t is not None]
            tmin  = [t for t in d["temperature_2m_min"]        if t is not None]
            rain  = [p for p in d["precipitation_sum"]         if p is not None]
            humid = [h for h in d["relative_humidity_2m_max"]  if h is not None]
            return (
                round((sum(tmax)+sum(tmin))/(len(tmax)+len(tmin)), 2),
                round(sum(rain), 2),
                round(sum(humid)/len(humid), 2),
                "Open-Meteo live (last 30 days)"
            )
    except Exception as e:
        print(f"[WEATHER] Error: {e}")
    return 27.0, 100.0, 75.0, "India average (Open-Meteo unavailable)"


def fetch_price(crop, api_key):
    if api_key:
        try:
            r = requests.get(
                "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070",
                params={"api-key": api_key, "format": "json", "limit": 10,
                        "filters[Commodity.keyword]": crop.capitalize()},
                timeout=20
            )
            if r.status_code == 200:
                records = r.json().get("records", [])
                prices  = [float(rec["Modal_Price"]) for rec in records if rec.get("Modal_Price")]
                if prices:
                    return round(sum(prices)/len(prices)), "Agmarknet live mandi price"
        except Exception:
            pass
    return FALLBACK_PRICES.get(crop, 2000), "2024 static price (MSP/Agmarknet)"


def get_disease_alert(crop, temp, humidity, rainfall):
    rule = DISEASE_RULES.get(crop)
    if not rule:
        return None
    triggered = True
    if "humidity_gt"  in rule and humidity  <= rule["humidity_gt"]:  triggered = False
    if "temp_gt"      in rule and temp      <= rule["temp_gt"]:       triggered = False
    if "temp_lt"      in rule and temp      >= rule["temp_lt"]:       triggered = False
    if "rainfall_gt"  in rule and rainfall  <= rule["rainfall_gt"]:  triggered = False
    if not triggered:
        return None
    return DiseaseAlert(disease=rule["disease"], action=rule["action"], risk="HIGH")


def calc_economics(crop, land_ha, price_per_quintal, price_source):
    row        = eco.loc[crop]
    price_kg   = round(price_per_quintal / 100, 2)
    yield_kg   = int(row["avg_yield_kg_per_ha"]         * land_ha)
    yield_low  = int(row["yield_low_kg_per_ha"]         * land_ha)
    yield_high = int(row["yield_high_kg_per_ha"]        * land_ha)
    total_cost = int(row["cultivation_cost_inr_per_ha"] * land_ha)
    revenue    = int(yield_kg * price_kg)
    profit     = revenue - total_cost
    roi        = round(profit / total_cost * 100, 1) if total_cost > 0 else 0.0
    margin     = round(profit / revenue   * 100, 1) if revenue    > 0 else 0.0
    breakeven  = int(total_cost / price_kg)          if price_kg  > 0 else 0
    return CropEconomics(
        yield_kg=yield_kg, yield_low_kg=yield_low, yield_high_kg=yield_high,
        price_per_kg_inr=price_kg, price_source=price_source,
        total_cost_inr=total_cost, revenue_inr=revenue,
        profit_inr=profit, roi_pct=roi, profit_margin_pct=margin,
        breakeven_yield_kg=breakeven, profitable=profit>0
    )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {"status":"running","version":"2.2.0",
            "model":"RandomForest","accuracy":ACCURACY,
            "endpoint":"POST /predict/bylocation","docs":"/docs"}


@app.post("/predict/bylocation", response_model=FullResponse, tags=["Prediction"])
def predict_by_location(data: LocationInput):
    land_ha = round(data.land_area_acres * 0.4047, 4)

    # 1. Soil — city-specific DB lookup
    db_N, db_P, db_K, db_ph, nearest_city, dist_km = get_nearest_city_soil(
        data.latitude, data.longitude
    )
    # Allow user overrides for any individual value
    soil_N  = data.soil_N  if data.soil_N  is not None else db_N
    soil_P  = data.soil_P  if data.soil_P  is not None else db_P
    soil_K  = data.soil_K  if data.soil_K  is not None else db_K
    soil_ph = data.soil_ph if data.soil_ph is not None else db_ph

    override_fields = []
    if data.soil_N:  override_fields.append("N")
    if data.soil_P:  override_fields.append("P")
    if data.soil_K:  override_fields.append("K")
    if data.soil_ph: override_fields.append("pH")

    soil_source = f"ICAR soil survey — nearest city: {nearest_city} ({dist_km} km away)"
    if override_fields:
        soil_source += f" | User override: {', '.join(override_fields)}"

    # 2. Weather
    temperature, rainfall, humidity, weather_source = fetch_weather(
        data.latitude, data.longitude
    )

    # 3. ML prediction
    X      = np.array([[soil_N, soil_P, soil_K, temperature, humidity, soil_ph, rainfall]])
    probas = model.predict_proba(X)[0]
    top_raw = sorted(zip(CLASSES, probas.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # 4. Filter impossible crops
    top5 = filter_impossible_crops(top_raw, data.latitude, data.longitude, temperature)

    # 5. Build results
    used_key     = data.agmarknet_api_key or AGMARKNET_API_KEY
    crop_results = []
    for rank, (crop, prob) in enumerate(top5, 1):
        if crop not in eco.index:
            continue
        price_q, price_src = fetch_price(crop, used_key)
        econ               = calc_economics(crop, land_ha, price_q, price_src)
        alert              = get_disease_alert(crop, temperature, humidity, rainfall)
        row                = eco.loc[crop]
        crop_results.append(CropResult(
            rank=rank, crop=crop,
            confidence_pct=round(prob*100, 2),
            season=row["season"], category=row["category"],
            duration_days=int(row["duration_days"]),
            water_req_mm=int(row["water_requirement_mm"]),
            economics=econ, disease_alert=alert,
        ))

    if not crop_results:
        raise HTTPException(status_code=500, detail="No crop results generated")

    best        = crop_results[0]
    best_profit = best.economics.profit_inr
    status_str  = "profit" if best_profit > 0 else "loss"
    summary = (
        f"For {data.land_area_acres} acres at ({data.latitude}N, {data.longitude}E) "
        f"near {nearest_city.title()}: grow {best.crop.upper()} this {best.season} season. "
        f"Expected {status_str} of Rs {best_profit:+,} "
        f"(ROI: {best.economics.roi_pct:+.1f}%). Harvest in {best.duration_days} days."
    )

    return FullResponse(
        location={"latitude":data.latitude,"longitude":data.longitude},
        land_area_acres=data.land_area_acres, land_area_ha=land_ha,
        weather=WeatherData(temperature_c=temperature, humidity_pct=humidity,
                            rainfall_mm=rainfall, source=weather_source),
        soil=SoilData(N=soil_N, P=soil_P, K=soil_K, ph=soil_ph,
                      source=soil_source, nearest_city=nearest_city,
                      distance_km=dist_km),
        top_5_crops=crop_results, best_crop=best.crop,
        best_crop_profit=best_profit, summary=summary,
    )


@app.get("/crops", tags=["Info"])
def list_crops():
    return {"crops": CLASSES, "total": len(CLASSES)}


@app.get("/soil/cities", tags=["Info"])
def list_soil_cities():
    """Returns all cities in the soil database"""
    return {
        "total": len(CITY_SOIL_DB),
        "cities": [
            {"city": k, "lat": v[0], "lon": v[1],
             "N": v[2], "P": v[3], "K": v[4], "ph": v[5]}
            for k, v in CITY_SOIL_DB.items()
        ]
    }


@app.get("/features", tags=["Info"])
def features_info():
    return {
        "auto_fetched": [
            {"name": "N,P,K,ph", "source": "ICAR city soil survey DB (60+ Indian cities)"},
            {"name": "temperature,humidity,rainfall", "source": "Open-Meteo live weather"},
        ],
        "optional_overrides": [
            "soil_N", "soil_P", "soil_K", "soil_ph", "agmarknet_api_key"
        ]
    }