# 🌾 Smart Crop Advisor API

> Give your location. Get the best crop to grow, expected profit, live weather, soil data, and disease alerts — instantly.

---

## 📌 What is this?

**Smart Crop Advisor** is an end-to-end ML-powered REST API that helps farmers decide **what to grow, when to grow it, and how much profit to expect** — using just their GPS coordinates and land area.

No soil testing kit required. No manual weather lookup. Just latitude, longitude, and acres.

---

## 🚀 Live Demo

```
POST https://your-app-name.onrender.com/predict/bylocation
```

Interactive docs (Swagger UI):
```
https://your-app-name.onrender.com/docs
```

---

## 🧠 How it works

```
You provide       Auto-fetched              ML Model          Output
─────────────     ──────────────────        ──────────        ──────────────────────
latitude    ──►   SoilGrids API  → N, pH                     Top 5 crop predictions
longitude   ──►   Open-Meteo    → temp,   ──► Random    ──►  Yield & profit per crop
land acres  ──►                   humidity,   Forest         Disease risk alerts
                                  rainfall    (99.32%)        Live mandi prices
                  Agmarknet API → market                      Full financial summary
                                  prices
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (300 trees) |
| Accuracy | **99.32%** on test set |
| Classes | 22 crops |
| Training data | Kaggle Crop Recommendation Dataset |
| Cross-validation | 99.1% ± 0.3% (5-fold) |

### Supported crops
`rice` `wheat` `maize` `chickpea` `kidneybeans` `pigeonpeas` `mothbeans` `mungbean`
`blackgram` `lentil` `pomegranate` `banana` `mango` `grapes` `watermelon` `muskmelon`
`apple` `orange` `papaya` `coconut` `cotton` `jute` `coffee`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — model info and status |
| `GET` | `/docs` | Interactive Swagger UI |
| `POST` | `/predict/bylocation` | **Main endpoint** — full crop analysis |
| `GET` | `/crops` | List all 22 supported crops |
| `GET` | `/features` | What's auto-fetched vs overrideable |

---

## 📥 Request

### Minimum (just 3 fields)

```json
{
  "latitude": 22.5726,
  "longitude": 88.3639,
  "land_area_acres": 5.0
}
```

### Full (with optional overrides)

```json
{
  "latitude": 22.5726,
  "longitude": 88.3639,
  "land_area_acres": 5.0,
  "soil_P": 38.0,
  "soil_K": 45.0,
  "agmarknet_api_key": "your_data_gov_in_key"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `latitude` | ✅ | Latitude of the farm |
| `longitude` | ✅ | Longitude of the farm |
| `land_area_acres` | ✅ | Farm size in acres |
| `soil_P` | ⬜ Optional | Phosphorus from soil test (kg/ha) — uses regional average if not given |
| `soil_K` | ⬜ Optional | Potassium from soil test (kg/ha) — uses regional average if not given |
| `agmarknet_api_key` | ⬜ Optional | data.gov.in key for live mandi prices — uses 2024 static prices if not given |

---

## 📤 Response

```json
{
  "location": { "latitude": 22.5726, "longitude": 88.3639 },
  "land_area_acres": 5.0,
  "land_area_ha": 2.0235,

  "weather": {
    "temperature_c": 27.87,
    "humidity_pct": 79.2,
    "rainfall_mm": 142.6,
    "source": "Open-Meteo live (last 30 days)"
  },

  "soil": {
    "N": 15.2,
    "P": 44.9,
    "K": 138.9,
    "ph": 6.7,
    "N_source": "SoilGrids satellite API",
    "P_source": "Estimated from SoilGrids SOC",
    "K_source": "Estimated from SoilGrids CEC",
    "ph_source": "SoilGrids satellite API"
  },

  "top_5_crops": [
    {
      "rank": 1,
      "crop": "rice",
      "confidence_pct": 87.33,
      "season": "Kharif",
      "category": "Cereal",
      "duration_days": 120,
      "water_req_mm": 1200,
      "economics": {
        "yield_kg": 5261,
        "yield_low_kg": 3642,
        "yield_high_kg": 9106,
        "price_per_kg_inr": 21.83,
        "price_source": "2024 static price (MSP/Agmarknet)",
        "total_cost_inr": 76893,
        "revenue_inr": 114844,
        "profit_inr": 37951,
        "roi_pct": 49.4,
        "profit_margin_pct": 33.0,
        "breakeven_yield_kg": 3522,
        "profitable": true
      },
      "disease_alert": null
    }
  ],

  "best_crop": "rice",
  "best_crop_profit": 37951,
  "summary": "For 5.0 acres at (22.57N, 88.36E): grow RICE this Kharif season. Expected profit of Rs +37,951 (ROI: +49.4%). Harvest in 120 days."
}
```

---

## ⚠️ Disease Alerts

The API automatically checks current weather against known disease risk thresholds:

| Crop | Disease | Trigger condition |
|------|---------|-------------------|
| Rice | Brown Leaf Spot / Blast | Humidity > 80% and Temp > 25°C |
| Wheat | Yellow Rust | Humidity > 75% and Temp < 15°C |
| Maize | Maydis Leaf Blight | Humidity > 70% and Rainfall > 100mm |
| Banana | Panama Wilt / Sigatoka | Humidity > 85% and Temp > 26°C |
| Coconut | Bud Rot | Humidity > 85% |
| Cotton | Bollworm | Humidity > 75% and Temp > 30°C |
| Coffee | Leaf Rust | Humidity > 85% and Temp < 18°C |

When triggered:
```json
"disease_alert": {
  "disease": "Brown Leaf Spot / Blast",
  "action": "Apply Tricyclazole fungicide. Monitor weekly.",
  "risk": "HIGH"
}
```

---


## 🛠️ Data Sources

| Data | Source | Cost |
|------|--------|------|
| Soil N, pH | [SoilGrids (ISRIC)](https://soilgrids.org) | Free, no key |
| Soil P, K | Estimated from SOC/CEC via SoilGrids | Free, no key |
| Temperature, Humidity, Rainfall | [Open-Meteo](https://open-meteo.com) | Free, no key |
| Live mandi prices | [Agmarknet / data.gov.in](https://data.gov.in) | Free, key required |
| Static fallback prices | MSP 2024 + Agmarknet averages | Built-in |
| Yield & cost data | CACP Reports 2023-24 + FAOSTAT | Built-in |

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Supratim-2006/Crop_Recommendation_API.git
cd Crop_Recommendation_API
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env` file

```bash
# Create .env in the project folder
AGMARKNET_API_KEY=your_key_here
```

Get a free key from [data.gov.in](https://data.gov.in) → Register → API Keys.
The API works without a key — uses 2024 static prices as fallback.

### 5. Start the server

```bash
uvicorn main:app --reload --port 8000
```

### 6. Test it

Open `http://localhost:8000/docs` in your browser for the interactive Swagger UI.

Or use Python:

```python
import requests

response = requests.post("http://localhost:8000/predict/bylocation", json={
    "latitude": 22.5726,
    "longitude": 88.3639,
    "land_area_acres": 5.0
})
print(response.json()["summary"])
```

---

## ☁️ Deploy to Render

### 1. Push to GitHub

```bash
git add .
git commit -m "initial deployment"
git push
```

### 2. Create Web Service on Render

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repository
3. Configure:

| Setting | Value |
|---------|-------|
| Environment | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

### 3. Add environment variable

Render dashboard → your service → **Environment** tab:

```
Key   : AGMARKNET_API_KEY
Value : your_actual_key
```

### 4. Done

Your API is live at `https://your-app-name.onrender.com`

---

## 🧪 Example Usage

### Python

```python
import requests

# Basic prediction
r = requests.post("https://your-app.onrender.com/predict/bylocation", json={
    "latitude": 22.5726,
    "longitude": 88.3639,
    "land_area_acres": 5.0
})

data = r.json()
print(data["summary"])
print("Best crop:", data["best_crop"])
print("Expected profit: Rs", data["best_crop_profit"])

# With soil test override
r = requests.post("https://your-app.onrender.com/predict/bylocation", json={
    "latitude": 22.5726,
    "longitude": 88.3639,
    "land_area_acres": 5.0,
    "soil_P": 38.0,
    "soil_K": 45.0
})
```

### curl

```bash
curl -X POST https://your-app.onrender.com/predict/bylocation \
  -H "Content-Type: application/json" \
  -d '{"latitude": 22.5726, "longitude": 88.3639, "land_area_acres": 5.0}'
```

### JavaScript

```javascript
const response = await fetch("https://your-app.onrender.com/predict/bylocation", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    latitude: 22.5726,
    longitude: 88.3639,
    land_area_acres: 5.0
  })
});
const data = await response.json();
console.log(data.summary);
```

---

## 📦 Requirements

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
joblib>=1.3.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.0.0
pydantic>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## 🚧 Known Limitations

| Limitation | Details |
|------------|---------|
| P and K accuracy | Phosphorus and Potassium are estimated from soil proxies — provide your own soil test values for higher accuracy |
| Soil data coverage | SoilGrids has limited resolution in remote areas — falls back to India regional average |
| Market prices | Live prices require Agmarknet API key — falls back to 2024 static prices without one |
| 22 crops only | Model supports 22 crops from the training dataset |
| Geographic filter | Apple/grapes filtered out for low-altitude hot regions automatically |

---

## 🔮 Future Improvements

- [ ] Crop yield prediction model (input: crop + location → expected yield)
- [ ] SMS output via Twilio for farmers without smartphones
- [ ] Hindi and Bengali language support
- [ ] More crops (vegetables, spices)
- [ ] Historical price trend charts
- [ ] Crop rotation recommendations
- [ ] Integration with satellite NDVI for field health monitoring

---

## 👨‍💻 Author

**Supratim** — built for Hackathon

---

## 📄 License

MIT License — free to use, modify, and distribute.
