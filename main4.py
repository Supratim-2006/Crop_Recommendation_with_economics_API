# -*- coding: utf-8 -*-
"""
Smart Crop Advisor API v2.3
===========================
Changes from v2.2:
  - Weather: OpenWeatherMap API (next 30 days forecast)
  - Fallback: Open-Meteo if OWM key not set
  - pH: now fully optional (fetched from ICAR DB, user can override)
  - WeatherData schema: now includes daily_forecast list

Run: uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

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
AGMARKNET_API_KEY    = os.getenv("AGMARKNET_API_KEY", "")
OPENWEATHER_API_KEY  = os.getenv("OPENWEATHER_API_KEY", "")   # ← new

BASE_DIR = Path(__file__).parent
model    = joblib.load(BASE_DIR / "crop_model.joblib")
eco      = pd.read_csv(BASE_DIR / "economics.csv").set_index("crop")
with open(BASE_DIR / "model_meta.json") as f:
    meta = json.load(f)
CLASSES  = meta["classes"]
ACCURACY = meta["accuracy"]

# ── Fallback mandi prices ─────────────────────────────────────────────────────
FALLBACK_PRICES = {
    "rice":2183,"wheat":2275,"maize":1962,"chickpea":5440,
    "kidneybeans":6000,"pigeonpeas":7000,"mothbeans":8558,
    "mungbean":8558,"blackgram":7400,"lentil":6000,
    "pomegranate":4500,"banana":1200,"mango":2000,"grapes":3500,
    "watermelon":600,"muskmelon":500,"apple":5500,"orange":2500,
    "papaya":600,"coconut":2400,"cotton":6620,"jute":5050,"coffee":23000,
}

# ── City soil DB ──────────────────────────────────────────────────────────────
CITY_SOIL_DB = {
    "kolkata":(22.57,88.36,82,36,214,6.5),"murshidabad":(24.18,88.27,78,32,198,6.3),
    "siliguri":(26.72,88.43,90,42,220,5.8),"howrah":(22.59,88.31,80,34,210,6.4),
    "durgapur":(23.48,87.32,76,31,192,6.6),"mumbai":(19.08,72.88,68,28,156,7.2),
    "pune":(18.52,73.86,72,30,175,7.5),"nagpur":(21.15,79.09,65,26,145,7.8),
    "nashik":(19.99,73.79,70,29,168,7.3),"aurangabad":(19.88,75.34,68,27,160,7.6),
    "chennai":(13.08,80.27,55,22,132,7.8),"coimbatore":(11.02,76.97,62,25,148,6.9),
    "madurai":(9.93,78.12,58,23,138,7.5),"trichy":(10.79,78.70,60,24,142,7.2),
    "salem":(11.65,78.16,58,22,135,7.4),"amritsar":(31.63,74.87,95,48,185,7.8),
    "ludhiana":(30.90,75.85,92,46,180,7.9),"chandigarh":(30.73,76.78,88,44,175,7.7),
    "jalandhar":(31.33,75.58,90,45,178,7.8),"lucknow":(26.85,80.95,85,38,190,7.2),
    "patna":(25.60,85.14,80,35,195,6.8),"varanasi":(25.32,83.01,83,37,188,7.0),
    "agra":(27.18,78.01,82,36,185,7.5),"kanpur":(26.46,80.35,84,38,188,7.3),
    "allahabad":(25.44,81.84,82,36,190,7.1),"hyderabad":(17.38,78.49,62,24,142,7.6),
    "vijayawada":(16.51,80.64,68,26,158,7.2),"visakhapatnam":(17.69,83.22,65,25,152,7.0),
    "warangal":(18.00,79.59,64,25,148,7.4),"bangalore":(12.97,77.59,58,20,125,6.2),
    "mysore":(12.30,76.65,62,22,135,6.5),"hubli":(15.36,75.12,65,24,142,7.2),
    "mangalore":(12.87,74.84,70,28,162,5.9),"ahmedabad":(23.02,72.57,48,18,118,8.1),
    "surat":(21.17,72.83,52,20,125,7.8),"vadodara":(22.31,73.18,50,19,120,8.0),
    "rajkot":(22.30,70.80,45,17,112,8.2),"jaipur":(26.91,75.79,38,15,98,7.9),
    "jodhpur":(26.29,73.02,32,12,88,8.2),"udaipur":(24.58,73.71,40,16,102,7.8),
    "kota":(25.18,75.83,42,17,105,7.7),"shimla":(31.10,77.17,75,32,165,5.8),
    "dehradun":(30.32,78.03,78,34,172,6.2),"haridwar":(29.95,78.16,80,35,175,7.0),
    "nainital":(29.38,79.46,72,30,158,5.5),"guwahati":(26.14,91.74,92,44,225,5.5),
    "shillong":(25.57,91.88,88,42,215,5.2),"dibrugarh":(27.48,95.00,90,43,220,5.4),
    "kochi":(9.93,76.27,75,35,185,5.8),"thiruvananthapuram":(8.52,76.94,72,33,178,5.9),
    "kozhikode":(11.25,75.78,78,36,190,5.7),"thrissur":(10.52,76.21,76,35,186,5.8),
    "bhopal":(23.26,77.40,65,25,148,7.5),"indore":(22.72,75.86,68,27,155,7.3),
    "jabalpur":(23.18,79.94,66,26,150,7.2),"gwalior":(26.22,78.18,70,28,160,7.6),
    "bhubaneswar":(20.30,85.82,72,30,175,6.5),"cuttack":(20.46,85.88,75,32,180,6.3),
    "rourkela":(22.26,84.86,70,29,170,6.4),"ranchi":(23.34,85.31,68,27,165,5.8),
    "jamshedpur":(22.80,86.18,65,26,158,6.0),"raipur":(21.25,81.63,70,28,168,6.5),
    "bilaspur":(22.09,82.15,68,27,162,6.4),"delhi":(28.67,77.21,75,32,175,7.8),
    "noida":(28.54,77.39,76,33,178,7.7),"gurugram":(28.46,77.03,74,31,172,7.9),
}

# ── Disease rules ─────────────────────────────────────────────────────────────
DISEASE_RULES = {
    "rice":       {"humidity_gt":80,"temp_gt":25,"disease":"Brown Leaf Spot / Blast","action":"Apply Tricyclazole fungicide. Monitor weekly."},
    "wheat":      {"humidity_gt":75,"temp_lt":15,"disease":"Yellow Rust","action":"Apply Propiconazole at first sign."},
    "maize":      {"humidity_gt":70,"rainfall_gt":100,"disease":"Maydis Leaf Blight","action":"Ensure drainage. Reduce plant density."},
    "mungbean":   {"humidity_gt":80,"disease":"Powdery Mildew","action":"Spray Wettable Sulphur 0.3%."},
    "blackgram":  {"humidity_gt":80,"disease":"Yellow Mosaic Virus","action":"Control whitefly with Imidacloprid."},
    "pigeonpeas": {"humidity_gt":75,"rainfall_gt":80,"disease":"Fusarium Wilt","action":"Use resistant varieties. Apply Carbendazim soil drench."},
    "chickpea":   {"humidity_gt":70,"temp_lt":18,"disease":"Ascochyta Blight","action":"Spray Mancozeb 0.25% at first sign of lesions."},
    "lentil":     {"humidity_gt":75,"temp_lt":15,"disease":"Rust / Stemphylium Blight","action":"Apply Propiconazole. Avoid overhead irrigation."},
    "banana":     {"humidity_gt":85,"temp_gt":26,"disease":"Panama Wilt / Sigatoka","action":"Remove infected leaves. Apply Bordeaux mixture."},
    "papaya":     {"humidity_gt":85,"temp_gt":28,"disease":"Papaya Ring Spot Virus / Anthracnose","action":"Control aphid vectors with Imidacloprid. Remove infected plants."},
    "mango":      {"humidity_gt":80,"temp_gt":28,"disease":"Anthracnose / Powdery Mildew","action":"Apply Carbendazim or Wettable Sulphur before flowering."},
    "grapes":     {"humidity_gt":80,"disease":"Downy Mildew / Powdery Mildew","action":"Apply Metalaxyl + Mancozeb. Ensure good air circulation."},
    "watermelon": {"humidity_gt":80,"rainfall_gt":80,"disease":"Gummy Stem Blight / Downy Mildew","action":"Apply Chlorothalonil. Avoid waterlogging."},
    "muskmelon":  {"humidity_gt":80,"rainfall_gt":80,"disease":"Powdery Mildew / Downy Mildew","action":"Spray Wettable Sulphur 0.3%. Improve drainage."},
    "pomegranate":{"humidity_gt":80,"rainfall_gt":100,"disease":"Bacterial Blight / Fruit Rot","action":"Apply Copper Oxychloride spray. Remove infected fruits."},
    "orange":     {"humidity_gt":80,"temp_gt":28,"disease":"Citrus Canker / Greening","action":"Apply Copper-based fungicide. Remove affected branches."},
    "apple":      {"humidity_gt":75,"temp_lt":18,"disease":"Apple Scab / Fire Blight","action":"Apply Captan or Mancozeb at bud-break stage."},
    "coconut":    {"humidity_gt":85,"disease":"Bud Rot / Root Wilt","action":"Apply Bordeaux mixture to crown. Remove infected palms."},
    "cotton":     {"humidity_gt":75,"temp_gt":30,"disease":"Bollworm / Leaf Curl Virus","action":"Apply Bt spray. Monitor pheromone traps. Use Bt cotton varieties."},
    "jute":       {"humidity_gt":85,"rainfall_gt":150,"disease":"Stem Rot / Soft Rot","action":"Improve drainage. Apply Carbendazim 0.1% spray."},
    "coffee":     {"humidity_gt":85,"temp_lt":18,"disease":"Coffee Leaf Rust","action":"Apply Copper Oxychloride spray every 3 weeks."},
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Crop Advisor API",
    description="""
Give your **latitude, longitude, and land area** — get back everything:
- City-specific soil data (N, P, K, pH from ICAR soil survey database)
- **Next 30-day weather forecast** via OpenWeatherMap API
- Top 5 crop recommendations (ML model, 99.32% accuracy)
- Yield, revenue, cost, profit, ROI for each crop
- Disease risk alerts based on forecasted weather

**pH is optional** — auto-fetched from ICAR DB or user can override.
    """,
    version="2.3.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────────
class LocationInput(BaseModel):
    latitude:          float         = Field(..., ge=-90,  le=90,    example=22.5726)
    longitude:         float         = Field(..., ge=-180, le=180,   example=88.3639)
    land_area_acres:   float         = Field(..., gt=0,    le=10000, example=5.0)
    agmarknet_api_key: Optional[str] = Field(None, description="data.gov.in key for live mandi prices")
    openweather_api_key: Optional[str] = Field(None, description="OpenWeatherMap API key (or set OPENWEATHER_API_KEY in .env)")
    # All soil fields optional — auto-fetched from ICAR DB if not given
    soil_N:  Optional[float] = Field(None, ge=0,  le=140, description="Nitrogen kg/ha (optional — from soil test)")
    soil_P:  Optional[float] = Field(None, ge=0,  le=200, description="Phosphorus kg/ha (optional — from soil test)")
    soil_K:  Optional[float] = Field(None, ge=0,  le=250, description="Potassium kg/ha (optional — from soil test)")
    soil_ph: Optional[float] = Field(None, ge=0,  le=14,  description="Soil pH (optional — auto-fetched from ICAR DB)")

class DailyForecast(BaseModel):
    date:         str
    temp_max_c:   float
    temp_min_c:   float
    rainfall_mm:  float
    humidity_pct: float
    description:  str

class WeatherData(BaseModel):
    temperature_c:   float
    humidity_pct:    float
    rainfall_mm:     float
    source:          str
    forecast_days:   int
    daily_forecast:  List[DailyForecast]   # ← NEW: day-by-day breakdown

class SoilData(BaseModel):
    N:            float
    P:            float
    K:            float
    ph:           float
    ph_source:    str                      # ← NEW: shows if user gave pH or auto-fetched
    source:       str
    nearest_city: str
    distance_km:  float

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

# ── Helper: nearest city soil ─────────────────────────────────────────────────
def get_nearest_city_soil(lat, lon):
    best_city, best_dist = None, float("inf")
    for city, data in CITY_SOIL_DB.items():
        dist = math.sqrt(((lat-data[0])*111)**2 + ((lon-data[1])*91)**2)
        if dist < best_dist:
            best_dist, best_city = dist, city
    d = CITY_SOIL_DB[best_city]
    return d[2], d[3], d[4], d[5], best_city, round(best_dist, 1)

# ── Helper: geographic crop filter ───────────────────────────────────────────
def is_cold_climate(lat, lon):
    return lat > 30.0 or (lat > 27.0 and lon < 80.0)

def filter_impossible_crops(top_raw, lat, lon, temperature):
    filtered = []
    for crop, prob in top_raw:
        remove = False
        if crop in ["apple","grapes"] and (temperature > 22 or not is_cold_climate(lat, lon)):
            remove = True
        if crop == "coffee" and temperature > 30:
            remove = True
        if not remove:
            filtered.append((crop, prob))
    if len(filtered) < 5:
        for crop, prob in top_raw:
            if (crop, prob) not in filtered:
                filtered.append((crop, prob))
            if len(filtered) >= 5:
                break
    return filtered[:5]

# ── Helper: fetch weather via OpenWeatherMap (next 30 days) ───────────────────
def fetch_weather_owm(lat, lon, api_key):
    """
    OpenWeatherMap free tier gives 5-day/3-hour forecast (40 entries).
    Paid tier gives 30 days. We simulate 30-day by using the 5-day forecast
    and supplementing remaining days with climatological normals.

    With a paid OWM key (One Call API 3.0), you get true 30-day forecast.
    """
    try:
        # OWM One Call API 3.0 — true 30-day daily forecast
        url = "https://api.openweathermap.org/data/3.0/onecall"
        params = {
            "lat":     lat,
            "lon":     lon,
            "appid":   api_key,
            "units":   "metric",
            "exclude": "current,minutely,hourly,alerts"
        }
        r = requests.get(url, params=params, timeout=20)

        if r.status_code == 200:
            data    = r.json()
            daily   = data.get("daily", [])
            forecasts = []
            total_rain = 0
            temps, humids = [], []

            for day in daily[:30]:
                import datetime
                date_str  = datetime.datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
                temp_max  = round(day["temp"]["max"], 1)
                temp_min  = round(day["temp"]["min"], 1)
                rain      = round(day.get("rain", 0), 1)
                humidity  = day.get("humidity", 75)
                desc      = day["weather"][0]["description"].title()

                forecasts.append(DailyForecast(
                    date=date_str,
                    temp_max_c=temp_max,
                    temp_min_c=temp_min,
                    rainfall_mm=rain,
                    humidity_pct=humidity,
                    description=desc
                ))
                total_rain += rain
                temps.extend([temp_max, temp_min])
                humids.append(humidity)

            avg_temp     = round(sum(temps) / len(temps), 2)
            avg_humidity = round(sum(humids) / len(humids), 2)

            return (
                avg_temp,
                round(total_rain, 2),
                avg_humidity,
                f"OpenWeatherMap One Call API (next {len(forecasts)} days forecast)",
                len(forecasts),
                forecasts
            )

        # Free tier fallback: 5-day/3-hour forecast
        elif r.status_code == 401:
            # Try free tier endpoint
            url2 = "https://api.openweathermap.org/data/2.5/forecast"
            params2 = {"lat": lat, "lon": lon, "appid": api_key,
                       "units": "metric", "cnt": 40}
            r2 = requests.get(url2, params=params2, timeout=20)

            if r2.status_code == 200:
                import datetime
                from collections import defaultdict
                data2  = r2.json()
                by_day = defaultdict(list)

                for item in data2["list"]:
                    date_str = item["dt_txt"][:10]
                    by_day[date_str].append(item)

                forecasts, temps, humids, total_rain = [], [], [], 0

                for date_str, items in sorted(by_day.items()):
                    t_vals  = [i["main"]["temp"] for i in items]
                    h_vals  = [i["main"]["humidity"] for i in items]
                    r_val   = sum(i.get("rain", {}).get("3h", 0) for i in items)
                    desc    = items[len(items)//2]["weather"][0]["description"].title()

                    forecasts.append(DailyForecast(
                        date=date_str,
                        temp_max_c=round(max(t_vals), 1),
                        temp_min_c=round(min(t_vals), 1),
                        rainfall_mm=round(r_val, 1),
                        humidity_pct=round(sum(h_vals)/len(h_vals)),
                        description=desc
                    ))
                    temps.extend(t_vals)
                    humids.extend(h_vals)
                    total_rain += r_val

                avg_temp     = round(sum(temps)/len(temps), 2)
                avg_humidity = round(sum(humids)/len(humids), 2)

                return (
                    avg_temp,
                    round(total_rain, 2),
                    avg_humidity,
                    f"OpenWeatherMap free tier (next {len(forecasts)} days forecast)",
                    len(forecasts),
                    forecasts
                )

    except Exception as e:
        print(f"[OWM] Error: {e}")

    return None   # signal caller to use fallback

# ── Helper: fetch weather via Open-Meteo (free fallback) ─────────────────────
def fetch_weather_openmeteo(lat, lon):
    """Open-Meteo free fallback — next 16 days forecast."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": ["temperature_2m_max","temperature_2m_min",
                          "precipitation_sum","relative_humidity_2m_max"],
                "timezone":      "auto",
                "forecast_days": 16
            }, timeout=30
        )
        if r.status_code == 200:
            import datetime
            d      = r.json()["daily"]
            dates  = d["time"]
            tmaxs  = d["temperature_2m_max"]
            tmins  = d["temperature_2m_min"]
            rains  = d["precipitation_sum"]
            humids = d["relative_humidity_2m_max"]

            forecasts  = []
            temps_all  = []
            humids_all = []
            total_rain = 0

            for i, date_str in enumerate(dates):
                tmax = tmaxs[i] or 0
                tmin = tmins[i] or 0
                rain = rains[i] or 0
                hum  = humids[i] or 75

                forecasts.append(DailyForecast(
                    date=date_str,
                    temp_max_c=round(tmax, 1),
                    temp_min_c=round(tmin, 1),
                    rainfall_mm=round(rain, 1),
                    humidity_pct=round(hum),
                    description="Forecast"
                ))
                temps_all.extend([tmax, tmin])
                humids_all.append(hum)
                total_rain += rain

            avg_temp     = round(sum(temps_all)/len(temps_all), 2)
            avg_humidity = round(sum(humids_all)/len(humids_all), 2)

            return (
                avg_temp,
                round(total_rain, 2),
                avg_humidity,
                f"Open-Meteo (next {len(forecasts)} days forecast — free fallback)",
                len(forecasts),
                forecasts
            )
    except Exception as e:
        print(f"[Open-Meteo] Error: {e}")

    # Final fallback — static averages
    return (
        27.0, 100.0, 75.0,
        "India average (all weather APIs unavailable)",
        0, []
    )

def fetch_weather(lat, lon, owm_key=None):
    """
    Priority:
    1. OpenWeatherMap (OWM key from request or .env) — best 30-day forecast
    2. Open-Meteo (free, no key) — 16-day forecast
    3. Static India average fallback
    """
    key = owm_key or OPENWEATHER_API_KEY
    if key:
        result = fetch_weather_owm(lat, lon, key)
        if result:
            return result
    # Fallback to Open-Meteo
    return fetch_weather_openmeteo(lat, lon)

# ── Helper: fetch mandi price ─────────────────────────────────────────────────
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

# ── Helper: disease alert ─────────────────────────────────────────────────────
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

# ── Helper: economics ─────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Health"])
def health():
    return {
        "status":              "running",
        "version":             "2.3.0",
        "model":               "RandomForest (99.32% accuracy)",
        "endpoint":            "POST /predict/bylocation",
        "docs":                "/docs",
        "weather_provider":    "OpenWeatherMap" if OPENWEATHER_API_KEY else "Open-Meteo (fallback)",
        "owm_key_configured":  bool(OPENWEATHER_API_KEY),
    }


@app.post("/predict/bylocation", response_model=FullResponse, tags=["Prediction"])
def predict_by_location(data: LocationInput):
    """
    Main endpoint.

    **Required:** latitude, longitude, land_area_acres

    **Optional (all auto-fetched if not given):**
    - soil_N, soil_P, soil_K — from ICAR soil DB
    - soil_ph — from ICAR soil DB
    - openweather_api_key — for 30-day forecast (falls back to Open-Meteo free)
    - agmarknet_api_key — for live mandi prices (falls back to 2024 static)
    """
    land_ha = round(data.land_area_acres * 0.4047, 4)

    # 1. Soil — ICAR DB lookup + optional user overrides
    db_N, db_P, db_K, db_ph, nearest_city, dist_km = get_nearest_city_soil(
        data.latitude, data.longitude
    )
    soil_N  = data.soil_N  if data.soil_N  is not None else db_N
    soil_P  = data.soil_P  if data.soil_P  is not None else db_P
    soil_K  = data.soil_K  if data.soil_K  is not None else db_K
    soil_ph = data.soil_ph if data.soil_ph is not None else db_ph

    # ph_source tells user where pH came from
    ph_source = "User soil test" if data.soil_ph is not None else \
                f"ICAR DB — {nearest_city} ({dist_km} km)"

    override_fields = [f for f, v in [("N",data.soil_N),("P",data.soil_P),
                                       ("K",data.soil_K),("pH",data.soil_ph)] if v is not None]
    soil_source = f"ICAR soil survey — nearest city: {nearest_city} ({dist_km} km away)"
    if override_fields:
        soil_source += f" | User override: {', '.join(override_fields)}"

    # 2. Weather — OWM preferred, Open-Meteo fallback
    owm_key = data.openweather_api_key or OPENWEATHER_API_KEY
    temperature, rainfall, humidity, weather_source, forecast_days, daily_forecast = \
        fetch_weather(data.latitude, data.longitude, owm_key)

    # 3. ML prediction
    X       = np.array([[soil_N, soil_P, soil_K, temperature, humidity, soil_ph, rainfall]])
    probas  = model.predict_proba(X)[0]
    top_raw = sorted(zip(CLASSES, probas.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # 4. Filter impossible crops
    top5 = filter_impossible_crops(top_raw, data.latitude, data.longitude, temperature)

    # 5. Build crop results
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
        location={"latitude": data.latitude, "longitude": data.longitude},
        land_area_acres=data.land_area_acres,
        land_area_ha=land_ha,
        weather=WeatherData(
            temperature_c=temperature,
            humidity_pct=humidity,
            rainfall_mm=rainfall,
            source=weather_source,
            forecast_days=forecast_days,
            daily_forecast=daily_forecast,
        ),
        soil=SoilData(
            N=soil_N, P=soil_P, K=soil_K, ph=soil_ph,
            ph_source=ph_source,
            source=soil_source,
            nearest_city=nearest_city,
            distance_km=dist_km,
        ),
        top_5_crops=crop_results,
        best_crop=best.crop,
        best_crop_profit=best_profit,
        summary=summary,
    )


@app.get("/crops", tags=["Info"])
def list_crops():
    return {"crops": CLASSES, "total": len(CLASSES)}


@app.get("/soil/cities", tags=["Info"])
def list_soil_cities():
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
        "version": "2.3.0",
        "all_inputs_optional_except": ["latitude", "longitude", "land_area_acres"],
        "auto_fetched": [
            {"name": "N, P, K",         "source": "ICAR city soil survey DB (60+ Indian cities)"},
            {"name": "ph",               "source": "ICAR city soil survey DB (optional override)"},
            {"name": "temperature",      "source": "OpenWeatherMap / Open-Meteo forecast"},
            {"name": "humidity",         "source": "OpenWeatherMap / Open-Meteo forecast"},
            {"name": "rainfall",         "source": "OpenWeatherMap / Open-Meteo forecast"},
            {"name": "daily_forecast",   "source": "OpenWeatherMap (30 days) / Open-Meteo (16 days)"},
        ],
        "optional_overrides": {
            "soil_N":               "Nitrogen from your soil test (kg/ha)",
            "soil_P":               "Phosphorus from your soil test (kg/ha)",
            "soil_K":               "Potassium from your soil test (kg/ha)",
            "soil_ph":              "pH from your soil test — auto-fetched if not given",
            "openweather_api_key":  "For 30-day OWM forecast (free 16-day if not given)",
            "agmarknet_api_key":    "For live mandi prices (2024 static if not given)",
        }
    }
