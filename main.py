# -*- coding: utf-8 -*-
"""
Smart Crop Advisor API v2.1
===========================
Input  : latitude, longitude, land area (acres)
Output : top 5 crop predictions + weather + soil + yield + profit/loss + disease alerts

Run locally:
    uvicorn main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""

import joblib
import json
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY", "")

# ── Load model & data ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
model    = joblib.load(BASE_DIR / "crop_model.joblib")
eco      = pd.read_csv(BASE_DIR / "crop_economics.csv").set_index("crop")

with open(BASE_DIR / "model_meta.json") as f:
    meta = json.load(f)

CLASSES  = meta["classes"]
ACCURACY = meta["accuracy"]

# ── Fallback prices ───────────────────────────────────────────────────────────
FALLBACK_PRICES = {
    "rice":2183,"wheat":2275,"maize":1962,"chickpea":5440,
    "kidneybeans":6000,"pigeonpeas":7000,"mothbeans":8558,
    "mungbean":8558,"blackgram":7400,"lentil":6000,
    "pomegranate":7000,"banana":2000,"mango":3000,"grapes":8000,
    "watermelon":1200,"muskmelon":1000,"apple":12000,"orange":4000,
    "papaya":1500,"coconut":3200,"cotton":6620,"jute":5050,"coffee":23000
}

# ── FIX 1: Region-based crop filter ──────────────────────────────────────────
# Crops that cannot grow in certain climate zones
# Key = (min_lat, max_lat, min_lon, max_lon) roughly
COLD_CLIMATE_ONLY = ["apple", "grapes"]      # need cold winters, high altitude
ARID_ONLY         = ["mothbeans", "muskmelon", "watermelon"]  # need dry climate
TROPICAL_ONLY     = ["coconut", "banana", "papaya", "mango",
                      "coffee", "jute"]       # need high rainfall + heat

def is_cold_climate(lat, lon):
    """Himachal Pradesh, Kashmir, Uttarakhand, Sikkim, hilly NE India"""
    return lat > 30.0 or (lat > 27.0 and lon < 80.0)

def is_arid(lat, lon):
    """Rajasthan, Gujarat dry belt"""
    return lat > 23.0 and lon < 74.0

def filter_impossible_crops(top5_raw, lat, lon, temperature, humidity, rainfall):
    """
    Remove crops that are geographically/climatically impossible
    for this location based on weather and coordinates.
    """
    filtered = []
    for crop, prob in top5_raw:
        remove = False

        # Apple & Grapes need cold climate (winter chill hours)
        if crop in COLD_CLIMATE_ONLY:
            if temperature > 22 or not is_cold_climate(lat, lon):
                remove = True

        # Coconut needs very high rainfall + coastal/tropical climate
        if crop == "coconut" and rainfall < 50 and humidity < 70:
            remove = True

        # Coffee needs specific altitude + humidity
        if crop == "coffee" and (humidity < 70 or temperature > 30):
            remove = True

        if not remove:
            filtered.append((crop, prob))

    # Always return at least 5 — fill from original if filtered too many
    if len(filtered) < 5:
        for crop, prob in top5_raw:
            if (crop, prob) not in filtered:
                filtered.append((crop, prob))
            if len(filtered) >= 5:
                break

    return filtered[:5]

# ── Disease rules ─────────────────────────────────────────────────────────────
DISEASE_RULES = {
    "rice":      {"humidity_gt":80,"temp_gt":25,"disease":"Brown Leaf Spot / Blast","action":"Apply Tricyclazole fungicide. Monitor weekly."},
    "wheat":     {"humidity_gt":75,"temp_lt":15,"disease":"Yellow Rust","action":"Apply Propiconazole at first sign."},
    "maize":     {"humidity_gt":70,"rainfall_gt":100,"disease":"Maydis Leaf Blight","action":"Ensure drainage. Reduce plant density."},
    "mungbean":  {"humidity_gt":80,"disease":"Powdery Mildew","action":"Spray Wettable Sulphur 0.3%."},
    "blackgram": {"humidity_gt":80,"disease":"Yellow Mosaic Virus","action":"Control whitefly with Imidacloprid."},
    "banana":    {"humidity_gt":85,"temp_gt":26,"disease":"Panama Wilt / Sigatoka","action":"Remove infected leaves. Apply Bordeaux mixture."},
    "coconut":   {"humidity_gt":85,"disease":"Bud Rot","action":"Apply Bordeaux mixture to crown."},
    "cotton":    {"humidity_gt":75,"temp_gt":30,"disease":"Bollworm","action":"Apply Bt spray. Monitor bollworm traps."},
    "coffee":    {"humidity_gt":85,"temp_lt":18,"disease":"Coffee Leaf Rust","action":"Apply Copper Oxychloride spray."},
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Crop Advisor API",
    description="""
Give your **latitude, longitude, and land area** — get back everything:
- Live soil data (N, pH from SoilGrids satellite)
- Live weather (temperature, humidity, rainfall from Open-Meteo)
- Top 5 crop recommendations (ML model, 99.32% accuracy)
- Yield, revenue, cost, profit, ROI for each crop
- Disease risk alerts based on current weather

**No manual soil input needed. Just your location.**
    """,
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class LocationInput(BaseModel):
    latitude:          float = Field(..., ge=-90,   le=90,    description="Latitude",           example=22.5726)
    longitude:         float = Field(..., ge=-180,  le=180,   description="Longitude",          example=88.3639)
    land_area_acres:   float = Field(..., gt=0,     le=10000, description="Land area in acres", example=5.0)
    agmarknet_api_key: Optional[str]   = Field(None, description="Optional: data.gov.in API key for live mandi prices")
    soil_P:            Optional[float] = Field(None, ge=0, le=200, description="Optional: Phosphorus from soil test (kg/ha)", example=42.0)
    soil_K:            Optional[float] = Field(None, ge=0, le=250, description="Optional: Potassium from soil test (kg/ha)",  example=43.0)

class WeatherData(BaseModel):
    temperature_c: float
    humidity_pct:  float
    rainfall_mm:   float
    source:        str

class SoilData(BaseModel):
    N:        float
    P:        float
    K:        float
    ph:       float
    N_source: str
    P_source: str
    K_source: str
    ph_source: str

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

# ── Helper: estimate P and K ──────────────────────────────────────────────────
def estimate_P_from_soc(soc):
    return round(10.0 + (soc * 1.8), 1)

def estimate_K_from_cec(cec):
    return round(80.0 + (cec * 3.2), 1)

# ── FIX 2: Corrected N conversion from SoilGrids ─────────────────────────────
def convert_N_for_model(raw_N_after_divisor):
    """
    SoilGrids nitrogen after dividing by d_factor gives values in cg/kg (centigrams/kg).
    The crop recommendation model was trained on data where N is in range 0-140.
    Raw SoilGrids values after division are typically 0.5-5.0 for Indian soils.
    We multiply by 10 to bring into the correct 5-50 range.

    Example: raw=1.5 cg/kg -> 1.5*10 = 15 (realistic Indian soil N)
    """
    if raw_N_after_divisor is None:
        return 40.0
    corrected = round(raw_N_after_divisor * 10, 2)
    # Clamp to valid model range
    return max(0.0, min(140.0, corrected))

# ── Helper: fetch soil ────────────────────────────────────────────────────────
def fetch_soil(lat, lon, user_P, user_K):
    N, ph    = 40.0, 6.5
    soc, cec = None, None
    N_src    = "India regional average (SoilGrids unavailable)"
    ph_src   = "India regional average (SoilGrids unavailable)"

    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lon": lon, "lat": lat,
            "property": ["nitrogen", "phh2o", "soc", "cec"],
            "depth": ["0-5cm"],
            "value": ["mean"]
        }
        r = requests.get(url, params=params, timeout=25)
        if r.status_code == 200:
            layers = r.json()["properties"]["layers"]
            soil   = {}
            for layer in layers:
                val     = layer["depths"][0]["values"]["mean"]
                divisor = layer["unit_measure"]["d_factor"]
                soil[layer["name"]] = round(val / divisor, 2) if val is not None else None

            # FIX 2 applied here — correct N conversion
            if soil.get("nitrogen") is not None:
                N     = convert_N_for_model(soil["nitrogen"])
                N_src = f"SoilGrids satellite API (raw={soil['nitrogen']} cg/kg → model input={N})"

            if soil.get("phh2o") is not None:
                ph     = soil["phh2o"]
                ph_src = "SoilGrids satellite API"

            soc = soil.get("soc")
            cec = soil.get("cec")

    except Exception:
        pass

    # P estimation
    if user_P is not None:
        P     = round(user_P, 1)
        P_src = "User soil test (override)"
    elif soc is not None:
        P     = estimate_P_from_soc(soc)
        P_src = f"Estimated from SoilGrids SOC ({soc} g/kg)"
    else:
        P     = 42.0
        P_src = "India regional average"

    # K estimation
    if user_K is not None:
        K     = round(user_K, 1)
        K_src = "User soil test (override)"
    elif cec is not None:
        K     = estimate_K_from_cec(cec)
        K_src = f"Estimated from SoilGrids CEC ({cec} cmol/kg)"
    else:
        K     = 120.0
        K_src = "India regional average"

    return N, P, K, ph, N_src, P_src, K_src, ph_src

# ── Helper: fetch weather ─────────────────────────────────────────────────────
def fetch_weather(lat, lon):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ["temperature_2m_max","temperature_2m_min",
                      "precipitation_sum","relative_humidity_2m_max"],
            "timezone":      "auto",
            "past_days":     30,
            "forecast_days": 1
        }
        r = requests.get(url, params=params, timeout=25)
        if r.status_code == 200:
            daily    = r.json()["daily"]
            tmax     = [t for t in daily["temperature_2m_max"]        if t is not None]
            tmin     = [t for t in daily["temperature_2m_min"]        if t is not None]
            rain     = [p for p in daily["precipitation_sum"]         if p is not None]
            humid    = [h for h in daily["relative_humidity_2m_max"]  if h is not None]
            temp     = round((sum(tmax)+sum(tmin))/(len(tmax)+len(tmin)), 2)
            rainfall = round(sum(rain), 2)
            humidity = round(sum(humid)/len(humid), 2)
            return temp, rainfall, humidity, "Open-Meteo live (last 30 days)"
    except Exception:
        pass
    return 27.0, 100.0, 75.0, "India average (Open-Meteo unavailable)"

# ── Helper: fetch mandi price ─────────────────────────────────────────────────
def fetch_price(crop, api_key):
    if api_key:
        try:
            url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            params = {
                "api-key": api_key,
                "format":  "json",
                "limit":   10,
                "filters[Commodity.keyword]": crop.capitalize(),
            }
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                records = r.json().get("records", [])
                prices  = [float(rec["Modal_Price"]) for rec in records if rec.get("Modal_Price")]
                if prices:
                    return round(sum(prices)/len(prices)), "Agmarknet live mandi price"
        except Exception:
            pass
    return FALLBACK_PRICES.get(crop, 2000), "2024 static price (MSP/Agmarknet)"

# ── Helper: disease alert ─────────────────────────────────────────────────────
def get_disease_alert(crop, temp, humidity, rainfall):
    rule = DISEASE_RULES.get(crop)
    if not rule:
        return None
    triggered = True
    if "humidity_gt"  in rule and humidity  <= rule["humidity_gt"]:  triggered = False
    if "humidity_lt"  in rule and humidity  >= rule["humidity_lt"]:  triggered = False
    if "temp_gt"      in rule and temp      <= rule["temp_gt"]:       triggered = False
    if "temp_lt"      in rule and temp      >= rule["temp_lt"]:       triggered = False
    if "rainfall_gt"  in rule and rainfall  <= rule["rainfall_gt"]:  triggered = False
    if not triggered:
        return None
    return DiseaseAlert(disease=rule["disease"], action=rule["action"], risk="HIGH")

# ── Helper: economics ─────────────────────────────────────────────────────────
def calc_economics(crop, land_ha, price_per_quintal):
    row        = eco.loc[crop]
    price_kg   = round(price_per_quintal / 100, 2)
    yield_kg   = int(row["avg_yield_kg_per_ha"]        * land_ha)
    yield_low  = int(row["yield_low_kg_per_ha"]        * land_ha)
    yield_high = int(row["yield_high_kg_per_ha"]       * land_ha)
    total_cost = int(row["cultivation_cost_inr_per_ha"]* land_ha)
    revenue    = int(yield_kg * price_kg)
    profit     = revenue - total_cost
    roi        = round(profit / total_cost * 100, 1)
    margin     = round(profit / revenue   * 100, 1) if revenue > 0 else 0.0
    breakeven  = int(total_cost / price_kg)
    return CropEconomics(
        yield_kg           = yield_kg,
        yield_low_kg       = yield_low,
        yield_high_kg      = yield_high,
        price_per_kg_inr   = price_kg,
        price_source       = "",
        total_cost_inr     = total_cost,
        revenue_inr        = revenue,
        profit_inr         = profit,
        roi_pct            = roi,
        profit_margin_pct  = margin,
        breakeven_yield_kg = breakeven,
        profitable         = profit > 0
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health():
    return {
        "status":   "running",
        "version":  "2.1.0",
        "model":    "RandomForest (300 trees)",
        "accuracy": ACCURACY,
        "endpoint": "POST /predict/bylocation",
        "docs":     "/docs"
    }


@app.post("/predict/bylocation", response_model=FullResponse, tags=["Prediction"])
def predict_by_location(data: LocationInput):
    """
    Main endpoint.
    Input : latitude, longitude, land_area_acres
    Output: weather, soil, top 5 crops, yield, revenue, cost, profit, ROI, disease alerts
    """
    land_ha = round(data.land_area_acres * 0.4047, 4)

    # 1. Soil
    soil_N, soil_P, soil_K, soil_ph, N_src, P_src, K_src, ph_src = fetch_soil(
        data.latitude, data.longitude, data.soil_P, data.soil_K
    )

    # 2. Weather
    temperature, rainfall, humidity, weather_source = fetch_weather(
        data.latitude, data.longitude
    )

    # 3. ML prediction — raw top N
    X       = np.array([[soil_N, soil_P, soil_K, temperature, humidity, soil_ph, rainfall]])
    probas  = model.predict_proba(X)[0]
    top_raw = sorted(zip(CLASSES, probas.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # FIX 1 applied — filter geographically impossible crops
    top5 = filter_impossible_crops(
        top_raw, data.latitude, data.longitude, temperature, humidity, rainfall
    )

    # 4. Build results
    crop_results = []
    used_key = AGMARKNET_API_KEY if not data.agmarknet_api_key else data.agmarknet_api_key

    for rank, (crop, prob) in enumerate(top5, 1):
        if crop not in eco.index:
            continue
        price_q, price_src = fetch_price(crop, used_key)
        econ               = calc_economics(crop, land_ha, price_q)
        econ.price_source  = price_src
        alert              = get_disease_alert(crop, temperature, humidity, rainfall)
        row                = eco.loc[crop]

        crop_results.append(CropResult(
            rank           = rank,
            crop           = crop,
            confidence_pct = round(prob * 100, 2),
            season         = row["season"],
            category       = row["category"],
            duration_days  = int(row["duration_days"]),
            water_req_mm   = int(row["water_requirement_mm"]),
            economics      = econ,
            disease_alert  = alert,
        ))

    if not crop_results:
        raise HTTPException(status_code=500, detail="No crop results generated")

    best        = crop_results[0]
    best_profit = best.economics.profit_inr
    status_str  = "profit" if best_profit > 0 else "loss"

    summary = (
        f"For {data.land_area_acres} acres at ({data.latitude}N, {data.longitude}E): "
        f"grow {best.crop.upper()} this {best.season} season. "
        f"Expected {status_str} of Rs {best_profit:+,} "
        f"(ROI: {best.economics.roi_pct:+.1f}%). "
        f"Harvest in {best.duration_days} days."
    )

    return FullResponse(
        location        = {"latitude": data.latitude, "longitude": data.longitude},
        land_area_acres = data.land_area_acres,
        land_area_ha    = land_ha,
        weather         = WeatherData(
            temperature_c = temperature,
            humidity_pct  = humidity,
            rainfall_mm   = rainfall,
            source        = weather_source,
        ),
        soil            = SoilData(
            N         = soil_N,
            P         = soil_P,
            K         = soil_K,
            ph        = soil_ph,
            N_source  = N_src,
            P_source  = P_src,
            K_source  = K_src,
            ph_source = ph_src,
        ),
        top_5_crops     = crop_results,
        best_crop       = best.crop,
        best_crop_profit= best_profit,
        summary         = summary,
    )


@app.get("/crops", tags=["Info"])
def list_crops():
    return {"crops": CLASSES, "total": len(CLASSES)}


@app.get("/features", tags=["Info"])
def features_info():
    return {
        "description": "This API auto-fetches all features. You only need latitude, longitude, and land area.",
        "auto_fetched": [
            {"name": "N",           "source": "SoilGrids satellite API (corrected scaling)"},
            {"name": "ph",          "source": "SoilGrids satellite API"},
            {"name": "P",           "source": "Estimated from SOC (SoilGrids) or regional average"},
            {"name": "K",           "source": "Estimated from CEC (SoilGrids) or regional average"},
            {"name": "temperature", "source": "Open-Meteo live weather"},
            {"name": "humidity",    "source": "Open-Meteo live weather"},
            {"name": "rainfall",    "source": "Open-Meteo live weather"},
        ],
        "optional_inputs": [
            {"name": "soil_P",            "description": "Override P with your soil test value"},
            {"name": "soil_K",            "description": "Override K with your soil test value"},
            {"name": "agmarknet_api_key", "description": "For live mandi prices"},
        ]
    }
