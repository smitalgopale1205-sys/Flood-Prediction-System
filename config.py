"""
config.py — Single source of truth for the Flood Prediction System.
Both train_model.py and app.py import from here to stay in sync.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH  = "flood_risk_FINAL_DATSET.xlsx"
MODELS_DIR    = "models"
OUTPUTS_DIR   = "outputs"

MODEL_PATH    = f"{MODELS_DIR}/best_model.pkl"
SCALER_PATH   = f"{MODELS_DIR}/scaler.pkl"
LE_LC_PATH    = f"{MODELS_DIR}/le_lc.pkl"
LE_ST_PATH    = f"{MODELS_DIR}/le_st.pkl"
META_PATH     = f"{MODELS_DIR}/meta.json"

# ── Features ──────────────────────────────────────────────────────────────────
# 9 features selected from the dataset (Lat/Lon dropped as non-causal)
FEATURES = [
    "Rainfall",           # mm  – primary flood driver
    "Humidity",           # %   – reduces soil absorption
    "Water_Level",        # m   – direct river gauge
    "River_Discharge",    # m³/s – overflow volume
    "Elevation",          # m   – low = flood-prone (inverted in score)
    "Population_Density", # p/km² – impervious surface proxy
    "Historical_Floods",  # 0/1 – recurrence indicator
    "Land_Cover_enc",     # encoded categorical
    "Soil_Type_enc",      # encoded categorical
]

TARGET = "Flood_Risk_Binary"

# ── Domain-Knowledge Flood Risk Score Weights ─────────────────────────────────
# Based on hydrology literature; weights sum to 1.0
SCORE_WEIGHTS = {
    "Rainfall":          0.25,
    "Water_Level":       0.20,
    "River_Discharge":   0.18,
    "Humidity":          0.12,
    "Elevation":         0.12,   # inverted: 1 - norm
    "Historical_Floods": 0.07,
    "Population_Density":0.04,
    "Infrastructure":    0.02,   # inverted: 1 - value
}

# Normalization ranges (min, max) for each feature
NORM_RANGES = {
    "Rainfall":          (0,   300),
    "Water_Level":       (0,    10),
    "River_Discharge":   (0,  5000),
    "Humidity":          (0,   100),
    "Elevation":         (0, 10000),
    "Population_Density":(0, 10000),
}

RISK_THRESHOLD = 50   # score > 50 → High Risk (1)

# ── Categorical Risk Multipliers ──────────────────────────────────────────────
# Land cover: higher = more runoff, less absorption
LAND_COVER_RISK = {
    "Water Body":   1.10,
    "Urban":        1.05,
    "Agricultural": 1.02,
    "Forest":       0.95,
    "Desert":       0.90,
}

# Soil type: higher = lower permeability = more waterlogging
SOIL_TYPE_RISK = {
    "Clay":  1.08,
    "Peat":  1.05,
    "Silt":  1.02,
    "Loam":  0.98,
    "Sandy": 0.90,
}

# ── Best Hyperparameters (from GridSearchCV) ──────────────────────────────────
BEST_PARAMS = {
    "n_estimators":  300,
    "learning_rate": 0.15,
    "max_depth":     7,
    "subsample":     0.8,
    "random_state":  42,
}

# ── Feature display labels (for plots and UI) ─────────────────────────────────
FEATURE_LABELS = {
    "Rainfall":           "Rainfall (mm)",
    "Humidity":           "Humidity (%)",
    "Water_Level":        "Water Level (m)",
    "River_Discharge":    "River Discharge (m³/s)",
    "Elevation":          "Elevation (m)",
    "Population_Density": "Population Density (p/km²)",
    "Historical_Floods":  "Historical Floods",
    "Land_Cover_enc":     "Land Cover",
    "Soil_Type_enc":      "Soil Type",
}

FEATURE_EXPLANATIONS = {
    "Rainfall":           "Higher rainfall → ground saturation → surface runoff → flooding",
    "Humidity":           "High humidity reduces soil absorption capacity",
    "Water_Level":        "Direct river gauge; high levels precede bank overflow",
    "River_Discharge":    "Large water volume = imminent overflow risk",
    "Elevation":          "Low terrain = natural flood accumulation zones",
    "Population_Density": "Dense areas have more impervious surfaces → less infiltration",
    "Historical_Floods":  "Past flood events strongly predict future susceptibility",
    "Land_Cover_enc":     "Water Body & Urban cover increase runoff (lower infiltration)",
    "Soil_Type_enc":      "Clay & Peat have low permeability → waterlogging risk",
}

# ── UI Slider ranges for app.py ───────────────────────────────────────────────
SLIDER_CONFIG = {
    "Rainfall":          {"min": 0.0,    "max": 300.0,   "default": 150.0, "step": 1.0,  "unit": "mm"},
    "Humidity":          {"min": 0.0,    "max": 100.0,   "default": 60.0,  "step": 0.5,  "unit": "%"},
    "Water_Level":       {"min": 0.0,    "max": 10.0,    "default": 5.0,   "step": 0.1,  "unit": "m"},
    "River_Discharge":   {"min": 0.0,    "max": 5000.0,  "default": 2500.0,"step": 10.0, "unit": "m³/s"},
    "Elevation":         {"min": 0.0,    "max": 10000.0, "default": 500.0, "step": 10.0, "unit": "m"},
    "Population_Density":{"min": 0,      "max": 10000,   "default": 5000,  "step": 50,   "unit": "p/km²"},
}
