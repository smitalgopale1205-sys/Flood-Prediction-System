
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from streamlit.components.v1 import html

# ── Shared config & utilities ─────────────────────────────────────────────────
from config import (
    MODEL_PATH, SCALER_PATH, LE_LC_PATH, LE_ST_PATH, META_PATH,
    FEATURES, FEATURE_LABELS, FEATURE_EXPLANATIONS, SLIDER_CONFIG,
)
from utils import compute_flood_risk_score, risk_level, prepare_input_row

# ── Page setup with custom theme ──────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Prediction System | Advanced Flood Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── INJECT ADVANCED CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Orbitron:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none;}
    
    /* Vibrant gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0f2a 0%, #1a1a3e 25%, #0d2b3e 50%, #1a2a4f 75%, #0f1535 100%);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0f2a;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #FF6B6B;
    }
    
    /* Typography with vibrant accents */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', monospace !important;
        font-weight: 600;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    p, span, div, label, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: #E0E5F0 !important;
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 42, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(78, 205, 196, 0.3) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #E0E5F0 !important;
    }
    
    /* Glassmorphism cards with vibrant borders */
    .custom-card {
        background: rgba(10, 15, 42, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(78, 205, 196, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .custom-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(78, 205, 196, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .custom-card:hover::before {
        left: 100%;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        border-color: #FF6B6B;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
    }
    
    /* Metric cards with vibrant colors */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15), rgba(78, 205, 196, 0.1));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(69, 183, 209, 0.3);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3);
        border-color: #FF6B6B;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Custom buttons with vibrant gradient */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%) !important;
        color: white !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 0 !important;
        height: 0 !important;
        border-radius: 50% !important;
        background: rgba(255, 255, 255, 0.3) !important;
        transform: translate(-50%, -50%) !important;
        transition: width 0.6s, height 0.6s !important;
    }
    
    .stButton > button:hover::before {
        width: 300px !important;
        height: 300px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.4);
    }
    
    /* Custom sliders with vibrant colors */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%) !important;
    }
    
    .stSlider > div > div > div > div {
        background: #FFD93D !important;
        box-shadow: 0 0 10px #FFD93D !important;
    }
    
    /* Custom select boxes */
    .stSelectbox > div > div {
        background: rgba(10, 15, 42, 0.8) !important;
        border: 1px solid rgba(78, 205, 196, 0.4) !important;
        border-radius: 10px !important;
        color: #E0E5F0 !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #FF6B6B !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #45B7D1 0%, #4ECDC4 50%, #FF6B6B 100%) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(10, 15, 42, 0.6) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(78, 205, 196, 0.2) !important;
    }
    
    /* Info/Warning/Error boxes with vibrant colors */
    .stAlert {
        background: rgba(10, 15, 42, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(78, 205, 196, 0.3) !important;
        border-radius: 15px !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent 0%, #FF6B6B 25%, #4ECDC4 50%, #45B7D1 75%, transparent 100%) !important;
        margin: 2rem 0 !important;
    }
    
    /* Custom section divider */
    .section-divider {
        background: linear-gradient(90deg, transparent 0%, #FF6B6B 25%, #4ECDC4 50%, #45B7D1 75%, transparent 100%);
        height: 2px;
        margin: 2rem 0;
    }
    
    /* Floating animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Pulse animation */
    @keyframes vibrantPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.02); }
    }
    
    .vibrant-pulse {
        animation: vibrantPulse 2s ease-in-out infinite;
    }
    
    /* Glowing text */
    .glow-text {
        text-shadow: 0 0 10px #FF6B6B, 0 0 20px #4ECDC4;
    }
    
    /* Radar scan effect */
    @keyframes radar {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .radar-loading::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 100px;
        height: 100px;
        border: 2px solid rgba(78, 205, 196, 0.3);
        border-top: 2px solid #FF6B6B;
        border-radius: 50%;
        animation: radar 1s linear infinite;
    }
    
    /* Colorful badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-blue {
        background: rgba(69, 183, 209, 0.2);
        border: 1px solid #45B7D1;
        color: #45B7D1;
    }
    
    .badge-teal {
        background: rgba(78, 205, 196, 0.2);
        border: 1px solid #4ECDC4;
        color: #4ECDC4;
    }
    
    .badge-coral {
        background: rgba(255, 107, 107, 0.2);
        border: 1px solid #FF6B6B;
        color: #FF6B6B;
    }
    
    .badge-gold {
        background: rgba(255, 217, 61, 0.2);
        border: 1px solid #FFD93D;
        color: #FFD93D;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with st.spinner("🌊 Loading Flood Prediction System..."):
        time.sleep(0.5)
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le_lc = joblib.load(LE_LC_PATH)
        le_st = joblib.load(LE_ST_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        return model, scaler, le_lc, le_st, meta

try:
    model, scaler, le_lc, le_st, meta = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - REDESIGNED
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 3rem; animation: float 3s ease-in-out infinite;'>🌊</div>
        <h2 style='font-size: 1.5rem; margin: 0;'>FLOOD PREDICTION</h2>
        <p style='font-size: 0.8rem; opacity: 0.7;'>Advanced Monitoring System</p>
        <div style='margin-top: 0.5rem;'>
            <span class='badge badge-blue'>AI-Powered</span>
            <span class='badge badge-teal'>Real-time</span>
            <span class='badge badge-coral'>95% Accuracy</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if model_loaded:
        st.markdown("""
        <div class='metric-card'>
            <p style='margin: 0; font-size: 0.8rem; opacity: 0.7;'>System Status</p>
            <p style='margin: 0.5rem 0; color: #4ECDC4; font-weight: 600;'>● OPERATIONAL</p>
            <span class='badge badge-gold'>Live</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='metric-card' style='margin-top: 1rem;'>
            <p style='margin: 0; font-size: 0.8rem; opacity: 0.7;'>Model Performance</p>
            <p style='margin: 0.5rem 0;'><span class='metric-value'>{:.0f}%</span> Accuracy</p>
            <p style='margin: 0.5rem 0;'><span class='metric-value'>{:.3f}</span> F1 Score</p>
            <div style='margin-top: 0.5rem;'>
                <span class='badge badge-blue'>Gradient Boosting</span>
                <span class='badge badge-teal'>15k Samples</span>
            </div>
        </div>
        """.format(meta['final_accuracy']*100, meta['final_f1']), unsafe_allow_html=True)
    else:
        st.error("⚠️ Model not found. Run `python train_model.py` first.")
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style='font-size: 1rem; margin-bottom: 1rem;'>📊 RISK WEIGHTS</h3>
    """, unsafe_allow_html=True)
    
    weights = {
        "Rainfall": "25%",
        "Water Level": "20%",
        "River Discharge": "18%",
        "Humidity": "12%",
        "Low Elevation": "12%",
        "Historical Floods": "7%",
        "Population Density": "4%",
        "Infrastructure": "2%",
    }
    
    for feat, w in weights.items():
        color = "#FF6B6B" if int(w[:-1]) > 15 else "#4ECDC4" if int(w[:-1]) > 10 else "#45B7D1"
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; margin: 0.5rem 0; font-size: 0.85rem;'>
            <span>{feat}</span>
            <span style='color: {color}; font-weight: 600;'>{w}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style='font-size: 1rem; margin-bottom: 1rem;'>ℹ️ FEATURE INFO</h3>
    """, unsafe_allow_html=True)
    
    for feat, exp in FEATURE_EXPLANATIONS.items():
        label = FEATURE_LABELS.get(feat, feat)
        with st.expander(label):
            st.caption(exp)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE - REDESIGNED
# ─────────────────────────────────────────────────────────────────────────────
# Hero section
st.markdown("""
<div style='text-align: center; padding: 1rem 0 2rem 0;'>
    <div style='font-size: 4rem; animation: float 3s ease-in-out infinite;'>🌊</div>
    <h1 style='font-size: 3rem; margin: 0;'>FLOOD PREDICTION SYSTEM</h1>
    <p style='font-size: 1.2rem; opacity: 0.8; margin-top: 0.5rem;'>Advanced Environmental Monitoring & Prediction System</p>
    <div style='margin: 1rem 0;'>
        <span class='badge badge-blue'>🚀 Real-time Analysis</span>
        <span class='badge badge-teal'>🎯 94.9% Accuracy</span>
        <span class='badge badge-coral'>⚡ Instant Predictions</span>
        <span class='badge badge-gold'>🌍 9 Features Analyzed</span>
    </div>
    <div class='section-divider'></div>
</div>
""", unsafe_allow_html=True)

if model_loaded:
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-family: "JetBrains Mono", monospace; font-size: 0.8rem; opacity: 0.6;'>
            <span style='color: #FF6B6B;'>●</span> ACTIVE MONITORING 
            | Tuned Gradient Boosting v2.1 
            | <span style='color: #4ECDC4;'>Real-time Prediction Engine</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if not model_loaded:
    st.error("🚨 Model artifacts not found in `models/` folder.")
    st.info("Please run `python train_model.py` first to train and save the model.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM - REDESIGNED WITH GLASS CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h2 style='text-align: center; margin: 2rem 0;'>📡 ENVIRONMENTAL PARAMETERS</h2>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class='custom-card'>
        <h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>🌧 HYDROLOGICAL DATA</h3>
        <div style='margin-bottom: 1rem;'>
            <span class='badge badge-blue'>Rainfall</span>
            <span class='badge badge-teal'>Water Levels</span>
            <span class='badge badge-coral'>Flow Rate</span>
        </div>
    """, unsafe_allow_html=True)
    
    cfg = SLIDER_CONFIG
    
    rainfall = st.slider(
        "RAINFALL INTENSITY",
        cfg["Rainfall"]["min"], cfg["Rainfall"]["max"],
        cfg["Rainfall"]["default"], cfg["Rainfall"]["step"],
        help=FEATURE_EXPLANATIONS["Rainfall"],
        format="%.0f mm"
    )
    
    water_level = st.slider(
        "WATER LEVEL",
        cfg["Water_Level"]["min"], cfg["Water_Level"]["max"],
        cfg["Water_Level"]["default"], cfg["Water_Level"]["step"],
        help=FEATURE_EXPLANATIONS["Water_Level"],
        format="%.1f m"
    )
    
    river_discharge = st.slider(
        "RIVER DISCHARGE RATE",
        cfg["River_Discharge"]["min"], cfg["River_Discharge"]["max"],
        cfg["River_Discharge"]["default"], cfg["River_Discharge"]["step"],
        help=FEATURE_EXPLANATIONS["River_Discharge"],
        format="%.0f m³/s"
    )
    
    humidity = st.slider(
        "ATMOSPHERIC HUMIDITY",
        cfg["Humidity"]["min"], cfg["Humidity"]["max"],
        cfg["Humidity"]["default"], cfg["Humidity"]["step"],
        help=FEATURE_EXPLANATIONS["Humidity"],
        format="%.0f%%"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class='custom-card'>
        <h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>🏔 GEOGRAPHICAL DATA</h3>
        <div style='margin-bottom: 1rem;'>
            <span class='badge badge-coral'>Elevation</span>
            <span class='badge badge-gold'>Population</span>
            <span class='badge badge-blue'>Land Cover</span>
        </div>
    """, unsafe_allow_html=True)
    
    elevation = st.slider(
        "TERRAIN ELEVATION",
        cfg["Elevation"]["min"], cfg["Elevation"]["max"],
        cfg["Elevation"]["default"], cfg["Elevation"]["step"],
        help=FEATURE_EXPLANATIONS["Elevation"],
        format="%.0f m"
    )
    
    pop_density = st.slider(
        "POPULATION DENSITY",
        cfg["Population_Density"]["min"], cfg["Population_Density"]["max"],
        cfg["Population_Density"]["default"], cfg["Population_Density"]["step"],
        help=FEATURE_EXPLANATIONS["Population_Density"],
        format="%.0f people/km²"
    )
    
    hist_floods = st.selectbox(
        "HISTORICAL FLOOD EVENTS",
        options=[0, 1],
        format_func=lambda x: "✅ NO PRIOR FLOODS" if x == 0 else "⚠️ PRIOR FLOODS RECORDED",
        help=FEATURE_EXPLANATIONS["Historical_Floods"]
    )
    
    land_cover = st.selectbox(
        "LAND COVER CLASSIFICATION",
        options=meta["land_covers"],
        help=FEATURE_EXPLANATIONS["Land_Cover_enc"]
    )
    
    soil_type = st.selectbox(
        "SOIL COMPOSITION TYPE",
        options=meta["soil_types"],
        help=FEATURE_EXPLANATIONS["Soil_Type_enc"]
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION - REDESIGNED WITH ANIMATIONS
# ─────────────────────────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔍 ANALYZE FLOOD RISK", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("🌊 Analyzing environmental patterns..."):
        time.sleep(0.8)
        
        lc_enc = int(le_lc.transform([land_cover])[0])
        st_enc = int(le_st.transform([soil_type])[0])
        
        X_input = prepare_input_row(
            rainfall, humidity, water_level, river_discharge,
            elevation, pop_density, hist_floods, lc_enc, st_enc
        )
        
        prob = model.predict_proba(X_input)[0][1]
        prediction = int(model.predict(X_input)[0])
        
        risk_score = compute_flood_risk_score({
            "Rainfall": rainfall, "Humidity": humidity,
            "Water_Level": water_level, "River_Discharge": river_discharge,
            "Elevation": elevation, "Population_Density": pop_density,
            "Historical_Floods": hist_floods, "Infrastructure": 0,
            "Land_Cover": land_cover, "Soil_Type": soil_type,
        })
        
        level, icon = risk_level(prob)
    
    st.markdown("<h2 style='text-align: center; margin: 2rem 0 1rem 0;'>📊 PREDICTION ANALYSIS</h2>", unsafe_allow_html=True)
    
    # Animated result alert with vibrant colors
    if prob < 0.35:
        st.markdown("""
        <div class='custom-card' style='border-color: #4ECDC4; text-align: center; animation: vibrantPulse 2s ease-in-out infinite;'>
            <div style='font-size: 3rem;'>✅</div>
            <h3 style='color: #4ECDC4;'>LOW RISK DETECTED</h3>
            <p>Environmental conditions are stable. Monitoring continues at standard intervals.</p>
            <span class='badge badge-teal'>Safe Zone</span>
        </div>
        """, unsafe_allow_html=True)
    elif prob < 0.65:
        st.markdown("""
        <div class='custom-card' style='border-color: #FFD93D; text-align: center;'>
            <div style='font-size: 3rem;'>⚠️</div>
            <h3 style='color: #FFD93D;'>MODERATE RISK ALERT</h3>
            <p>Monitor hydrological conditions closely. Increased surveillance recommended.</p>
            <span class='badge badge-gold'>Caution Zone</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='custom-card' style='border-color: #FF6B6B; text-align: center; animation: vibrantPulse 1.5s ease-in-out infinite;'>
            <div style='font-size: 3rem;'>🚨</div>
            <h3 style='color: #FF6B6B;'>HIGH RISK WARNING</h3>
            <p>Immediate precautions and alerts are advised. Monitor situation closely.</p>
            <span class='badge badge-coral'>Danger Zone</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Metric row with vibrant cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.8rem; opacity: 0.7; margin: 0;'>PREDICTION</p>
            <div class='metric-value' style='font-size: 1.5rem;'>{'⚠️ FLOOD' if prediction == 1 else '✅ SAFE'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.8rem; opacity: 0.7; margin: 0;'>RISK LEVEL</p>
            <div class='metric-value' style='font-size: 1.5rem;'>{icon} {level}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.8rem; opacity: 0.7; margin: 0;'>CONFIDENCE</p>
            <div class='metric-value' style='font-size: 1.5rem;'>{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='font-size: 0.8rem; opacity: 0.7; margin: 0;'>RISK SCORE</p>
            <div class='metric-value' style='font-size: 1.5rem;'>{risk_score:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability bar
    st.markdown("#### FLOOD PROBABILITY METER")
    st.progress(float(prob))
    
    # Two-column detail view
    detail_left, detail_right = st.columns(2)
    
    with detail_left:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='font-size: 1.1rem; margin-bottom: 1rem;'>📋 INPUT PARAMETERS</h3>
        """, unsafe_allow_html=True)
        
        summary_df = pd.DataFrame({
            "Feature": [
                "Rainfall", "Humidity", "Water Level", "River Discharge",
                "Elevation", "Population Density", "Historical Floods",
                "Land Cover", "Soil Type"
            ],
            "Value": [
                f"{rainfall} mm", f"{humidity}%", f"{water_level} m",
                f"{river_discharge} m³/s", f"{elevation} m",
                f"{pop_density} p/km²",
                "YES ⚠️" if hist_floods else "NO ✅",
                land_cover, soil_type
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with detail_right:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='font-size: 1.1rem; margin-bottom: 1rem;'>📊 RISK CONTRIBUTION ANALYSIS</h3>
        """, unsafe_allow_html=True)
        
        components = {
            "Rainfall": min(1, rainfall / 300) * 0.25,
            "Water Level": min(1, water_level / 10) * 0.20,
            "River Discharge": min(1, river_discharge / 5000) * 0.18,
            "Humidity": min(1, humidity / 100) * 0.12,
            "Low Elevation": (1 - min(1, elevation / 10000)) * 0.12,
            "Hist. Floods": hist_floods * 0.07,
            "Pop. Density": min(1, pop_density / 10000) * 0.04,
        }
        
        # Custom vibrant theme for matplotlib
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(5, 3.5))
        labels = list(components.keys())
        values = list(components.values())
        colors = ["#FF6B6B" if v > 0.06 else "#4ECDC4" if v > 0.03 else "#45B7D1" for v in values]
        bars = ax.barh(labels, values, color=colors, edgecolor="#1a1a3e", linewidth=2)
        ax.set_xlabel("Weighted Contribution", color="#E0E5F0", fontsize=9)
        ax.set_title("Feature Impact Analysis", color="#FF6B6B", fontsize=10, fontweight='bold')
        ax.axvline(sum(values) / len(values), color="#FFD93D", linestyle="--", linewidth=1.5, label="Threshold")
        ax.legend(fontsize=8, facecolor='#0a0f2a', edgecolor='#4ECDC4')
        ax.tick_params(colors='#E0E5F0')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)
