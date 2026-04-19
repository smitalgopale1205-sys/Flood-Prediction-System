# 🌊 Flood Prediction System

A machine learning system that predicts flood risk from environmental and hydrological
features, with a Streamlit web app for real-time inference. Achieves **94.9% accuracy**.

---

## 📁 Folder Structure

```
flood_project/
│
├── flood_risk_FINAL_DATSET.xlsx   ← Your dataset (place here before training)
│
├── config.py                      ← ⭐ Single source of truth: all constants,
│                                      feature lists, weights, file paths
├── utils.py                       ← ⭐ Shared functions used by both train & app
│                                      (flood risk scoring, input prep, risk labels)
├── train_model.py                 ← Full ML pipeline (imports config + utils)
├── app.py                         ← Streamlit web app  (imports config + utils)
├── requirements.txt
└── README.md
│
├── models/                        ← Auto-created by train_model.py
│   ├── best_model.pkl             ← Tuned Gradient Boosting
│   ├── scaler.pkl                 ← StandardScaler
│   ├── le_lc.pkl                  ← LabelEncoder (Land Cover)
│   ├── le_st.pkl                  ← LabelEncoder (Soil Type)
│   └── meta.json                  ← Metrics, params, feature lists
│
└── outputs/                       ← Auto-created: all EDA & evaluation plots
    ├── 01_correlation_heatmap.png
    ├── 02_feature_distributions.png
    ├── 03_risk_score_distribution.png
    ├── 04_categorical_flood_rates.png
    ├── 05_model_comparison.png
    ├── 06_feature_importance.png
    ├── cm_logistic_regression.png
    ├── cm_decision_tree.png
    ├── cm_random_forest.png
    └── cm_gradient_boosting.png
```

---

## 🔗 How Files Are Integrated

```
config.py  ──────────────────────────────────────────────
    │  (constants: FEATURES, WEIGHTS, PATHS, SLIDERS...)  │
    ▼                                                      ▼
train_model.py                                        app.py
    │  imports config + utils                              │  imports config + utils
    │  builds target via utils.build_target()              │  loads saved model
    │  trains & evaluates 4 models                         │  uses utils.compute_flood_risk_score()
    │  saves artifacts to models/                          │  uses utils.prepare_input_row()
    ▼                                                      │  uses utils.risk_level()
models/ ────────────────────────────────────────────────── ▲
    (best_model.pkl, scaler.pkl, le_lc.pkl, le_st.pkl, meta.json)
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
```
flood_project/flood_risk_FINAL_DATSET.xlsx
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Launch the web app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 🎯 Selected Features (9)

| Feature            | Weight | Why Selected |
|--------------------|--------|-------------|
| Rainfall           | 25%    | Primary flood driver |
| Water_Level        | 20%    | Direct river gauge |
| River_Discharge    | 18%    | Overflow volume |
| Humidity           | 12%    | Reduces soil absorption |
| Elevation          | 12%    | Low terrain = flood-prone (inverted) |
| Historical_Floods  | 7%     | Recurrence predictor |
| Population_Density | 4%     | Impervious surface proxy |
| Land_Cover         | mult.  | Urban/Water Body amplify risk |
| Soil_Type          | mult.  | Clay/Peat reduce infiltration |

---

## 📊 Model Results

| Model               | Accuracy | F1    |
|---------------------|----------|-------|
| Logistic Regression | 89.7%    | 0.908 |
| Decision Tree       | 83.2%    | 0.848 |
| Random Forest       | 89.9%    | 0.910 |
| **Gradient Boosting** | **94.9%** | **0.955** |
