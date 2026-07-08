"""
================================================================================
 SMART RESTAURANT AI - FOOD DEMAND FORECASTING SYSTEM
 Business Intelligence Dashboard (Streamlit)
================================================================================
This app loads a REAL trained XGBoost model and rebuilds the exact feature
pipeline used during training (see feature_columns.pkl) so predictions are
consistent with the model's training distribution.

UPGRADE NOTES (this version):
- Executive KPI dashboard with 6 headline business metrics
- Weekly / approximate-monthly demand trend toggle
- Cuisine, category, center and promotion performance visuals
- A dedicated "AI Business Insights" page that programmatically mines
  train.csv for rising/falling demand combinations, low performers,
  the top center, and promotion lift — every number below is computed
  live from the uploaded CSVs, nothing is hard-coded or fabricated.
- Demand Prediction page logic is UNCHANGED (byte-for-byte identical to the
  original) to preserve compatibility with your trained models.

Folder structure expected:
    Smart Restaurant AI/
    ├── app.py
    ├── models/
    │   ├── xgboost_model.pkl
    │   ├── feature_columns.pkl
    │   └── scaler.pkl
    ├── train.csv
    ├── meal_info.csv
    └── fulfilment_center_info.csv
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ANN support is optional - only used if models/ann_model.h5 is present later
try:
    from tensorflow.keras.models import load_model as load_keras_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Smart Restaurant AI | Demand Forecasting",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# FILE PATHS
# =============================================================================
MODEL_DIR = "models"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_model.h5")

TRAIN_CSV_PATH = "train.csv"
MEAL_INFO_PATH = "meal_info.csv"
CENTER_INFO_PATH = "fulfilment_center_info.csv"

# =============================================================================
# CUSTOM CSS - PREMIUM BUSINESS DASHBOARD THEME
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a1a40, #3a1c71);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1c3c 0%, #2c1a4d 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #f1f1f1 !important; }

.main-header {
    padding: 28px 32px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.02));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.main-header h1 {
    font-weight: 800; font-size: 2.3rem; margin-bottom: 4px;
    background: linear-gradient(90deg, #ff9966, #ff5e62, #ffd452);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header p { color: #d8d8f0; font-size: 1.02rem; }

.kpi-card {
    padding: 20px; border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.03));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.18);
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    transition: transform 0.25s ease;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-5px); }
.kpi-value { font-size: 1.7rem; font-weight: 800; color: #ffd452; }
.kpi-label { font-size: 0.86rem; color: #e0e0f5; font-weight: 500; margin-top: 4px; }
.kpi-icon { font-size: 1.7rem; margin-bottom: 4px; }
.kpi-sub { font-size: 0.74rem; color: #b9b9e0; margin-top: 2px; }

.section-card {
    padding: 22px; border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.01));
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.section-title {
    color: #ffd452; font-weight: 700; font-size: 1.25rem; margin-bottom: 14px;
    border-left: 4px solid #ff5e62; padding-left: 10px;
}

.badge-low    { background: linear-gradient(135deg,#2ecc71,#27ae60); padding:10px 20px; border-radius:30px; font-weight:700; color:white; display:inline-block; }
.badge-medium { background: linear-gradient(135deg,#f1c40f,#f39c12); padding:10px 20px; border-radius:30px; font-weight:700; color:#1c1c1c; display:inline-block; }
.badge-high   { background: linear-gradient(135deg,#e74c3c,#c0392b); padding:10px 20px; border-radius:30px; font-weight:700; color:white; display:inline-block; }

.stButton>button {
    background: linear-gradient(135deg, #ff5e62, #ff9966);
    color: white; font-weight: 700; border: none; border-radius: 12px;
    padding: 12px 26px; font-size: 1.02rem;
    box-shadow: 0 4px 15px rgba(255,94,98,0.4);
    transition: all 0.25s ease;
}
.stButton>button:hover { transform: scale(1.02); box-shadow: 0 6px 22px rgba(255,94,98,0.6); }

.rec-box {
    padding: 15px 18px; border-radius: 14px;
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #ffd452; margin-bottom: 12px; color: #f1f1f1;
}
.info-box {
    padding: 12px 16px; border-radius: 12px;
    background: rgba(255,255,255,0.06);
    border-left: 3px solid #6dd5fa; margin-bottom: 10px; color: #e8e8f5; font-size: 0.92rem;
}

.insight-card {
    padding: 16px 20px; border-radius: 14px;
    background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02));
    border-left: 4px solid #ff9966; margin-bottom: 14px; color: #f1f1f1;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.insight-title { font-weight: 700; color: #ffd452; margin-bottom: 4px; font-size: 1.0rem; }
.insight-text { color: #e6e6f5; font-size: 0.92rem; line-height: 1.45; }

hr { border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 16px 0; }
p, span, label, div { color: #f1f1f1; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA / MODEL LOADING (cached)
# =============================================================================
@st.cache_resource
def load_xgb_model():
    if os.path.exists(XGB_MODEL_PATH):
        try:
            return joblib.load(XGB_MODEL_PATH), None
        except Exception as e:
            return None, str(e)
    return None, "File not found"

@st.cache_resource
def load_feature_columns():
    if os.path.exists(FEATURE_COLUMNS_PATH):
        try:
            return joblib.load(FEATURE_COLUMNS_PATH), None
        except Exception as e:
            return None, str(e)
    return None, "File not found"

@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        try:
            return joblib.load(SCALER_PATH), None
        except Exception as e:
            return None, str(e)
    return None, "File not found"

@st.cache_resource
def load_ann_model():
    if TF_AVAILABLE and os.path.exists(ANN_MODEL_PATH):
        try:
            return load_keras_model(ANN_MODEL_PATH), None
        except Exception as e:
            return None, str(e)
    return None, "Not available (optional)"

@st.cache_data
def load_meal_info():
    if os.path.exists(MEAL_INFO_PATH):
        return pd.read_csv(MEAL_INFO_PATH)
    return None

@st.cache_data
def load_center_info():
    if os.path.exists(CENTER_INFO_PATH):
        return pd.read_csv(CENTER_INFO_PATH)
    return None

@st.cache_data
def load_train_data():
    if os.path.exists(TRAIN_CSV_PATH):
        return pd.read_csv(TRAIN_CSV_PATH)
    return None

xgb_model, xgb_err = load_xgb_model()
feature_columns, fc_err = load_feature_columns()
scaler, scaler_err = load_scaler()
ann_model, ann_err = load_ann_model()

meal_df = load_meal_info()
center_df = load_center_info()
train_df = load_train_data()

MODEL_READY = (xgb_model is not None) and (feature_columns is not None)
DATA_READY = (meal_df is not None) and (center_df is not None)
FULL_DATA_READY = DATA_READY and (train_df is not None)

# =============================================================================
# LABEL ENCODING MAPS
# Reconstructed exactly as sklearn's LabelEncoder does: sorted-alphabetical
# unique values -> integer codes. This matches the encoding used when the
# model was trained (see notebook Cell 46: LabelEncoder on category/cuisine/
# center_type).
# =============================================================================
def build_label_map(series):
    classes = sorted(series.dropna().unique().tolist())
    return {cls: idx for idx, cls in enumerate(classes)}

if DATA_READY:
    CATEGORY_MAP = build_label_map(meal_df["category"])
    CUISINE_MAP = build_label_map(meal_df["cuisine"])
    CENTER_TYPE_MAP = build_label_map(center_df["center_type"])
else:
    CATEGORY_MAP, CUISINE_MAP, CENTER_TYPE_MAP = {}, {}, {}

# =============================================================================
# DEMAND LEVEL THRESHOLDS (data-driven, from actual historical distribution)
# =============================================================================
if train_df is not None:
    LOW_THRESHOLD = float(train_df["num_orders"].quantile(0.33))
    HIGH_THRESHOLD = float(train_df["num_orders"].quantile(0.66))
else:
    LOW_THRESHOLD, HIGH_THRESHOLD = 70.0, 240.0

def classify_demand(value):
    if value < LOW_THRESHOLD:
        return "Low Demand", "badge-low", "🟢"
    elif value < HIGH_THRESHOLD:
        return "Medium Demand", "badge-medium", "🟡"
    else:
        return "High Demand", "badge-high", "🔴"

def generate_recommendations(level, predicted_value):
    recs = {}
    if level == "Low Demand":
        recs["prep"] = f"Prepare around {int(round(predicted_value * 1.05))} units — minimal safety buffer needed."
        recs["staff"] = "Schedule a lean shift (2-3 kitchen staff, 2 servers) to control labor cost."
        recs["inventory"] = "Order conservative stock; avoid over-purchasing perishable ingredients."
        recs["waste"] = "Portion sizes can stay standard; focus on using existing stock before reordering."
    elif level == "Medium Demand":
        recs["prep"] = f"Prepare around {int(round(predicted_value * 1.10))} units to absorb normal fluctuation."
        recs["staff"] = "Schedule a standard shift (4-5 kitchen staff, 3-4 servers)."
        recs["inventory"] = "Maintain standard inventory levels with a ~10% safety buffer."
        recs["waste"] = "Track sell-through hourly; adjust next-day prep based on actual uptake."
    else:
        recs["prep"] = f"Prepare around {int(round(predicted_value * 1.20))} units to avoid stockouts during peak demand."
        recs["staff"] = "Schedule full strength (6+ kitchen staff, 5+ servers) plus on-call backup."
        recs["inventory"] = "Increase orders by 20-25%, prioritizing the highest-demand ingredients."
        recs["waste"] = "Stagger prep in batches through the shift rather than one large batch to reduce spoilage risk."
    return recs

# =============================================================================
# FEATURE ENGINEERING HELPERS (mirrors the training notebook exactly)
# =============================================================================
def get_center_details(center_id):
    row = center_df[center_df["center_id"] == center_id].iloc[0]
    return {
        "city_code": int(row["city_code"]),
        "region_code": int(row["region_code"]),
        "center_type": row["center_type"],
        "op_area": float(row["op_area"]),
    }

def get_meal_details(meal_id):
    row = meal_df[meal_df["meal_id"] == meal_id].iloc[0]
    return {
        "category": row["category"],
        "cuisine": row["cuisine"],
    }

def get_recent_lags(center_id, meal_id):
    """Fetch the most recent 4 historical num_orders values for this
    center+meal combination from train.csv, matching the lag_1..lag_4
    engineering used during training. Falls back to zeros if no history
    exists (new combination), consistent with the notebook's fillna(0)."""
    if train_df is None:
        return [0, 0, 0, 0], False

    subset = train_df[
        (train_df["center_id"] == center_id) & (train_df["meal_id"] == meal_id)
    ].sort_values("week", ascending=False)

    if subset.empty:
        return [0, 0, 0, 0], False

    recent = subset["num_orders"].head(4).tolist()
    while len(recent) < 4:
        recent.append(0)
    return recent, True

def build_feature_row(week, center_id, meal_id, checkout_price, base_price,
                       emailer_for_promotion, homepage_featured,
                       lag_1, lag_2, lag_3, lag_4):
    center_details = get_center_details(center_id)
    meal_details = get_meal_details(meal_id)

    price_difference = base_price - checkout_price
    discount_percentage = ((base_price - checkout_price) / base_price * 100) if base_price != 0 else 0.0
    promotion = 1 if (emailer_for_promotion == 1 or homepage_featured == 1) else 0
    week_of_month = ((week - 1) % 4) + 1

    lags = [lag_1, lag_2, lag_3, lag_4]
    rolling_mean_4 = float(np.mean(lags))
    rolling_std_4 = float(np.std(lags, ddof=1)) if len(set(lags)) > 1 else 0.0

    row = {
        "week": week,
        "center_id": center_id,
        "meal_id": meal_id,
        "checkout_price": checkout_price,
        "emailer_for_promotion": emailer_for_promotion,
        "homepage_featured": homepage_featured,
        "category": CATEGORY_MAP.get(meal_details["category"], 0),
        "cuisine": CUISINE_MAP.get(meal_details["cuisine"], 0),
        "city_code": center_details["city_code"],
        "region_code": center_details["region_code"],
        "center_type": CENTER_TYPE_MAP.get(center_details["center_type"], 0),
        "op_area": center_details["op_area"],
        "price_difference": price_difference,
        "discount_percentage": discount_percentage,
        "promotion": promotion,
        "week_of_month": week_of_month,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_4": lag_4,
        "rolling_mean_4": rolling_mean_4,
        "rolling_std_4": rolling_std_4,
    }

    df_row = pd.DataFrame([row])
    df_row = df_row[feature_columns]  # enforce exact training column order
    return df_row, meal_details, center_details

def predict_demand(feature_df):
    """XGBoost was trained on RAW (unscaled) features - the scaler.pkl in
    this project was fit only for the separate ANN model. Applying it here
    would corrupt the XGBoost prediction, so we intentionally skip it for
    the XGBoost branch."""
    xgb_pred = None
    ann_pred = None

    if xgb_model is not None:
        try:
            xgb_pred = float(xgb_model.predict(feature_df)[0])
        except Exception as e:
            st.error(f"XGBoost prediction failed: {e}")

    if ann_model is not None and scaler is not None:
        try:
            scaled = scaler.transform(feature_df)
            ann_pred = float(ann_model.predict(scaled).flatten()[0])
        except Exception as e:
            st.warning(f"ANN prediction skipped: {e}")

    return xgb_pred, ann_pred

# =============================================================================
# EXECUTIVE KPI & GROWTH METRICS
# All figures below are computed directly from train.csv / meal_info.csv /
# fulfilment_center_info.csv — nothing is hard-coded or simulated.
# =============================================================================
@st.cache_data
def compute_executive_kpis(train_df, meal_df, center_df):
    """Builds the headline KPI numbers shown on the Dashboard page."""
    total_orders = int(train_df["num_orders"].sum())
    n_weeks = int(train_df["week"].nunique())

    # Weekly demand is the dataset's native granularity. We additionally
    # derive an estimated daily figure (weekly total / 7) purely as a
    # convenience metric - clearly labeled as an estimate in the UI.
    weekly_totals = train_df.groupby("week")["num_orders"].sum()
    avg_weekly_demand = float(weekly_totals.mean())
    avg_daily_demand_est = avg_weekly_demand / 7.0

    # Growth %: compare the average weekly total orders in the most recent
    # quarter of weeks against the earliest quarter of weeks in the dataset.
    weeks_sorted = sorted(train_df["week"].unique())
    q = max(1, len(weeks_sorted) // 4)
    early_weeks = weeks_sorted[:q]
    recent_weeks = weeks_sorted[-q:]
    early_avg = train_df[train_df["week"].isin(early_weeks)]["num_orders"].sum() / q
    recent_avg = train_df[train_df["week"].isin(recent_weeks)]["num_orders"].sum() / q
    growth_pct = ((recent_avg - early_avg) / early_avg * 100.0) if early_avg > 0 else 0.0

    # Highest performing meal (by total historical orders)
    meal_totals = train_df.groupby("meal_id")["num_orders"].sum().sort_values(ascending=False)
    top_meal_id = int(meal_totals.index[0])
    top_meal_orders = float(meal_totals.iloc[0])
    meal_row = meal_df[meal_df["meal_id"] == top_meal_id]
    top_meal_category = meal_row["category"].values[0] if not meal_row.empty else "Unknown"
    top_meal_cuisine = meal_row["cuisine"].values[0] if not meal_row.empty else "Unknown"

    # Best performing fulfilment center (by total historical orders)
    center_totals = train_df.groupby("center_id")["num_orders"].sum().sort_values(ascending=False)
    top_center_id = int(center_totals.index[0])
    top_center_orders = float(center_totals.iloc[0])
    center_row = center_df[center_df["center_id"] == top_center_id]
    top_center_type = center_row["center_type"].values[0] if not center_row.empty else "Unknown"

    return {
        "total_orders": total_orders,
        "n_weeks": n_weeks,
        "avg_weekly_demand": avg_weekly_demand,
        "avg_daily_demand_est": avg_daily_demand_est,
        "growth_pct": growth_pct,
        "top_meal_id": top_meal_id,
        "top_meal_orders": top_meal_orders,
        "top_meal_category": top_meal_category,
        "top_meal_cuisine": top_meal_cuisine,
        "top_center_id": top_center_id,
        "top_center_orders": top_center_orders,
        "top_center_type": top_center_type,
        "n_centers": int(center_df["center_id"].nunique()),
        "n_meals": int(meal_df["meal_id"].nunique()),
    }

# =============================================================================
# AI BUSINESS INSIGHTS ENGINE
# Mines train.csv for rising/falling demand pairs, weak performers, the top
# center, and promotion lift. Every insight is generated from real rows in
# the dataset — no synthetic examples are inserted.
# =============================================================================
@st.cache_data
def generate_ai_insights(train_df, meal_df, center_df):
    insights = []
    merged = train_df.merge(meal_df, on="meal_id", how="left").merge(center_df, on="center_id", how="left")

    # --- 1. Rising / falling demand center+meal combinations -----------------
    # Compare the average of the last 4 weeks of history against the prior
    # 4 weeks, per center+meal pair (mirrors the lag_1..lag_4 features used
    # by the model itself).
    growth_rows = []
    for (c, m), g in train_df.groupby(["center_id", "meal_id"]):
        if len(g) < 8:
            continue
        g = g.sort_values("week")
        prev4 = g["num_orders"].iloc[-8:-4].mean()
        last4 = g["num_orders"].iloc[-4:].mean()
        if prev4 < 10:  # ignore near-zero baselines to avoid noisy % swings
            continue
        growth_rows.append({
            "center_id": c, "meal_id": m,
            "prev4_avg": prev4, "last4_avg": last4,
            "growth_pct": (last4 - prev4) / prev4 * 100.0,
        })
    growth_df = pd.DataFrame(growth_rows)

    if not growth_df.empty:
        rising = growth_df.sort_values("growth_pct", ascending=False).head(3)
        for _, row in rising.iterrows():
            if row["growth_pct"] > 5:
                insights.append({
                    "icon": "📈",
                    "title": f"Center {int(row['center_id'])} + Meal {int(row['meal_id'])} — Rising Demand",
                    "text": (
                        f"Average weekly orders climbed from {row['prev4_avg']:.0f} to "
                        f"{row['last4_avg']:.0f} ({row['growth_pct']:+.1f}%) over the most recent 4 weeks. "
                        f"Recommendation: increase preparation quantity and safety stock for this "
                        f"center-meal combination during upcoming peak periods."
                    ),
                })

        falling = growth_df.sort_values("growth_pct", ascending=True).head(2)
        for _, row in falling.iterrows():
            if row["growth_pct"] < -10:
                insights.append({
                    "icon": "📉",
                    "title": f"Center {int(row['center_id'])} + Meal {int(row['meal_id'])} — Falling Demand",
                    "text": (
                        f"Average weekly orders dropped from {row['prev4_avg']:.0f} to "
                        f"{row['last4_avg']:.0f} ({row['growth_pct']:+.1f}%). "
                        f"Recommendation: scale back prep quantity and consider a targeted "
                        f"promotion to revive demand before over-ordering inventory."
                    ),
                })

    # --- 2. Lowest performing meals overall -----------------------------------
    meal_avg = (
        merged.groupby(["meal_id", "category", "cuisine"])["num_orders"]
        .mean().reset_index().sort_values("num_orders")
    )
    for _, row in meal_avg.head(2).iterrows():
        insights.append({
            "icon": "🥡",
            "title": f"Meal {int(row['meal_id'])} ({row['category']} · {row['cuisine']}) — Low Demand Item",
            "text": (
                f"Averages only {row['num_orders']:.0f} orders per week across all centers, "
                f"the lowest in the menu. Recommendation: bundle with a promotion, revisit pricing, "
                f"or reduce standing prep quantity to cut waste."
            ),
        })

    # --- 3. Best performing fulfilment center ---------------------------------
    center_perf = (
        merged.groupby(["center_id", "center_type"])["num_orders"]
        .sum().reset_index().sort_values("num_orders", ascending=False)
    )
    if not center_perf.empty:
        best = center_perf.iloc[0]
        insights.append({
            "icon": "🏆",
            "title": f"Center {int(best['center_id'])} ({best['center_type']}) — Top Performing Center",
            "text": (
                f"Generated the highest cumulative demand of {best['num_orders']:,.0f} orders "
                f"across the full history. Recommendation: use its staffing and inventory "
                f"cadence as the benchmark playbook for similar center types."
            ),
        })

    # --- 4. Promotion effectiveness -------------------------------------------
    merged_promo = merged.assign(
        promo_status=np.where(
            (merged["emailer_for_promotion"] == 1) | (merged["homepage_featured"] == 1),
            "Promoted", "Not Promoted",
        )
    )
    promo_avg = merged_promo.groupby("promo_status")["num_orders"].mean()
    if "Promoted" in promo_avg.index and "Not Promoted" in promo_avg.index and promo_avg["Not Promoted"] > 0:
        lift = (promo_avg["Promoted"] - promo_avg["Not Promoted"]) / promo_avg["Not Promoted"] * 100.0
        insights.append({
            "icon": "📢",
            "title": "Promotion Effectiveness",
            "text": (
                f"Promoted meals (emailer or homepage feature) average {promo_avg['Promoted']:.0f} "
                f"orders/week vs {promo_avg['Not Promoted']:.0f} for non-promoted meals — a "
                f"{lift:+.1f}% lift. Recommendation: prioritize promotions for medium/low "
                f"performing meals to close the gap."
            ),
        })

    return insights

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🍽️ Smart Restaurant AI")
    st.markdown("### Demand Forecasting System")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 Demand Prediction", "📈 Analytics & Insights",
         "🧠 AI Business Insights", "ℹ️ About Project"],
    )

    st.markdown("---")
    st.markdown("#### ⚙️ System Status")
    st.write(f"XGBoost Model: {'✅ Loaded' if xgb_model is not None else '⚠️ ' + str(xgb_err)}")
    st.write(f"Feature Columns: {'✅ Loaded' if feature_columns is not None else '⚠️ ' + str(fc_err)}")
    st.write(f"Scaler (ANN): {'✅ Loaded' if scaler is not None else '⚠️ ' + str(scaler_err)}")
    st.write(f"ANN Model: {'✅ Loaded' if ann_model is not None else 'ℹ️ Optional - not added yet'}")
    st.write(f"train.csv: {'✅ Loaded' if train_df is not None else '⚠️ Missing'}")
    st.write(f"meal_info.csv: {'✅ Loaded' if meal_df is not None else '⚠️ Missing'}")
    st.write(f"fulfilment_center_info.csv: {'✅ Loaded' if center_df is not None else '⚠️ Missing'}")

    st.markdown("---")
    st.caption("Built with Streamlit • XGBoost • Plotly")

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🍽 Smart Restaurant AI</h1>
    <p>Food Demand Forecasting & Business Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_READY:
    st.error(
        "⚠️ Model files not found. Place `xgboost_model.pkl` and `feature_columns.pkl` "
        "inside the `models/` folder to enable predictions."
    )
if not DATA_READY:
    st.error(
        "⚠️ Reference datasets not found. Place `meal_info.csv` and `fulfilment_center_info.csv` "
        "in the project root — these are required to build accurate feature inputs."
    )

# =============================================================================
# PAGE: DASHBOARD (Executive KPI Dashboard)
# =============================================================================
if page == "🏠 Dashboard":

    if not FULL_DATA_READY:
        st.info("Upload `train.csv`, `meal_info.csv` and `fulfilment_center_info.csv` to the project root to see the executive dashboard here.")
    else:
        kpis = compute_executive_kpis(train_df, meal_df, center_df)

        # ---- KPI Row 1 ----
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📦</div>
                <div class="kpi-value">{kpis['total_orders']:,}</div>
                <div class="kpi-label">Total Orders (all-time)</div>
            </div>""", unsafe_allow_html=True)
        with r1c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">📅</div>
                <div class="kpi-value">{kpis['avg_daily_demand_est']:,.0f}</div>
                <div class="kpi-label">Avg Daily Demand (est.)</div>
                <div class="kpi-sub">Weekly avg {kpis['avg_weekly_demand']:,.0f} ÷ 7 days</div>
            </div>""", unsafe_allow_html=True)
        with r1c3:
            arrow = "▲" if kpis['growth_pct'] >= 0 else "▼"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{arrow}</div>
                <div class="kpi-value">{kpis['growth_pct']:+.1f}%</div>
                <div class="kpi-label">Demand Growth</div>
                <div class="kpi-sub">Latest quarter vs earliest quarter of weeks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- KPI Row 2 ----
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🍜</div>
                <div class="kpi-value">Meal {kpis['top_meal_id']}</div>
                <div class="kpi-label">Highest Performing Meal</div>
                <div class="kpi-sub">{kpis['top_meal_category']} · {kpis['top_meal_cuisine']} — {kpis['top_meal_orders']:,.0f} orders</div>
            </div>""", unsafe_allow_html=True)
        with r2c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🏢</div>
                <div class="kpi-value">Center {kpis['top_center_id']}</div>
                <div class="kpi-label">Best Performing Center</div>
                <div class="kpi-sub">{kpis['top_center_type']} — {kpis['top_center_orders']:,.0f} orders</div>
            </div>""", unsafe_allow_html=True)
        with r2c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">🏬</div>
                <div class="kpi-value">{kpis['n_centers']}</div>
                <div class="kpi-label">Total Fulfilment Centers</div>
                <div class="kpi-sub">{kpis['n_meals']} menu items tracked</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Demand Trend: Weekly / Approx-Monthly toggle ----
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Demand Trend Over Time</div>', unsafe_allow_html=True)
        trend_tab_week, trend_tab_month = st.tabs(["📆 Weekly", "🗓️ Approx. Monthly"])

        with trend_tab_week:
            weekly = train_df.groupby("week")["num_orders"].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly["week"], y=weekly["num_orders"], mode="lines",
                line=dict(color="#ffd452", width=2.5),
                fill="tozeroy", fillcolor="rgba(255,94,98,0.15)",
                name="Total Weekly Orders"
            ))
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Week", yaxis_title="Total Orders"
            )
            st.plotly_chart(fig, use_container_width=True)

        with trend_tab_month:
            # The source data has no calendar date column — only a week number.
            # We approximate each 4-week block as one "month" purely for a
            # higher-level view; this is a real aggregation of real rows,
            # not fabricated data.
            monthly_df = train_df.copy()
            monthly_df["month_block"] = ((monthly_df["week"] - 1) // 4) + 1
            monthly = monthly_df.groupby("month_block")["num_orders"].sum().reset_index()
            fig_m = px.bar(monthly, x="month_block", y="num_orders",
                            color="num_orders", color_continuous_scale="sunset", template="plotly_dark")
            fig_m.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                 height=380, margin=dict(l=10, r=10, t=30, b=10),
                                 xaxis_title="Month (approx., 4-week blocks)", yaxis_title="Total Orders",
                                 showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True)
            st.caption("Each 'month' groups 4 consecutive weeks since the dataset has no calendar date field.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Category performance & Top centers ----
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🍲 Avg Demand by Meal Category</div>', unsafe_allow_html=True)
            merged = train_df.merge(meal_df, on="meal_id", how="left")
            cat_avg = merged.groupby("category")["num_orders"].mean().sort_values(ascending=False)
            fig2 = px.bar(x=cat_avg.index, y=cat_avg.values, color=cat_avg.values,
                          color_continuous_scale="sunset", template="plotly_dark")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                height=340, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="Category", yaxis_title="Avg Orders", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🏢 Top 10 Centers by Total Demand</div>', unsafe_allow_html=True)
            top_centers = train_df.groupby("center_id")["num_orders"].sum().sort_values(ascending=False).head(10)
            fig3 = px.bar(x=top_centers.index.astype(str), y=top_centers.values,
                          color=top_centers.values, color_continuous_scale="agsunset", template="plotly_dark")
            fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                height=340, margin=dict(l=10, r=10, t=20, b=10),
                                xaxis_title="Center ID", yaxis_title="Total Orders", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: DEMAND PREDICTION  (UNCHANGED - preserved exactly for compatibility)
# =============================================================================
elif page == "📊 Demand Prediction":

    if not (MODEL_READY and DATA_READY):
        st.warning("Prediction form requires the model files and reference CSVs described above.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏢 Restaurant & Order Details</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            center_ids = sorted(center_df["center_id"].unique().tolist())
            center_id = st.selectbox("Fulfilment Center ID", center_ids)
            center_details = get_center_details(center_id)
            st.markdown(f"""
            <div class="info-box">
            📍 City Code: <b>{center_details['city_code']}</b> &nbsp;|&nbsp;
            Region Code: <b>{center_details['region_code']}</b><br>
            🏬 Center Type: <b>{center_details['center_type']}</b> &nbsp;|&nbsp;
            Operational Area: <b>{center_details['op_area']}</b>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            meal_ids = sorted(meal_df["meal_id"].unique().tolist())
            meal_id = st.selectbox("Meal ID", meal_ids)
            meal_details = get_meal_details(meal_id)
            st.markdown(f"""
            <div class="info-box">
            🍽️ Category: <b>{meal_details['category']}</b><br>
            🌍 Cuisine: <b>{meal_details['cuisine']}</b>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            max_week = int(train_df["week"].max()) if train_df is not None else 145
            week = st.number_input("Forecast Week Number", min_value=1, value=max_week + 1, step=1)
            week_of_month = ((week - 1) % 4) + 1
            st.markdown(f"""
            <div class="info-box">
            🗓️ Week of Month (derived): <b>{week_of_month}</b>
            </div>
            """, unsafe_allow_html=True)
            weather_condition = st.selectbox(
                "Weather Condition (business context only)",
                ["Sunny", "Rainy", "Cloudy", "Stormy", "Windy"],
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            default_checkout = float(train_df[
                (train_df["center_id"] == center_id) & (train_df["meal_id"] == meal_id)
            ]["checkout_price"].mean()) if train_df is not None else 200.0
            if np.isnan(default_checkout):
                default_checkout = float(train_df["checkout_price"].mean())
            checkout_price = st.number_input("Checkout Price (₹)", min_value=0.0, value=round(default_checkout, 2))

        with c5:
            default_base = float(train_df[
                (train_df["center_id"] == center_id) & (train_df["meal_id"] == meal_id)
            ]["base_price"].mean()) if train_df is not None else 220.0
            if np.isnan(default_base):
                default_base = float(train_df["base_price"].mean())
            base_price = st.number_input("Base Price (₹)", min_value=0.0, value=round(default_base, 2))

        with c6:
            emailer_for_promotion = st.selectbox("Emailer Promotion Sent", ["No", "Yes"])
            homepage_featured = st.selectbox("Featured on Homepage", ["No", "Yes"])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**📊 Recent Order History (auto-filled from `train.csv`, editable)**")

        recent_lags, history_found = get_recent_lags(center_id, meal_id)
        if not history_found:
            st.info("No historical orders found for this exact Center + Meal combination — defaulting to 0. Adjust manually if you have estimates.")

        l1, l2, l3, l4 = st.columns(4)
        with l1:
            lag_1 = st.number_input("Last Week's Orders (lag_1)", min_value=0, value=int(recent_lags[0]))
        with l2:
            lag_2 = st.number_input("2 Weeks Ago (lag_2)", min_value=0, value=int(recent_lags[1]))
        with l3:
            lag_3 = st.number_input("3 Weeks Ago (lag_3)", min_value=0, value=int(recent_lags[2]))
        with l4:
            lag_4 = st.number_input("4 Weeks Ago (lag_4)", min_value=0, value=int(recent_lags[3]))

        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔮 Predict Demand", use_container_width=True)

        if predict_btn:
            feature_df, m_details, c_details = build_feature_row(
                week=int(week),
                center_id=int(center_id),
                meal_id=int(meal_id),
                checkout_price=float(checkout_price),
                base_price=float(base_price),
                emailer_for_promotion=1 if emailer_for_promotion == "Yes" else 0,
                homepage_featured=1 if homepage_featured == "Yes" else 0,
                lag_1=float(lag_1), lag_2=float(lag_2), lag_3=float(lag_3), lag_4=float(lag_4),
            )

            xgb_pred, ann_pred = predict_demand(feature_df)

            if xgb_pred is None:
                st.error("Prediction could not be generated. Please check the model files.")
            else:
                final_prediction = xgb_pred if ann_pred is None else (xgb_pred + ann_pred) / 2
                level, badge_class, emoji = classify_demand(final_prediction)

                st.markdown("<br>", unsafe_allow_html=True)
                kpi_cols = st.columns(3 if ann_pred is None else 4)

                with kpi_cols[0]:
                    st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🎯</div>
                        <div class="kpi-value">{final_prediction:.0f}</div>
                        <div class="kpi-label">Predicted Food Demand (Orders)</div></div>""", unsafe_allow_html=True)
                with kpi_cols[1]:
                    st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🌲</div>
                        <div class="kpi-value">{xgb_pred:.0f}</div>
                        <div class="kpi-label">XGBoost Prediction</div></div>""", unsafe_allow_html=True)
                if ann_pred is not None:
                    with kpi_cols[2]:
                        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">🧠</div>
                            <div class="kpi-value">{ann_pred:.0f}</div>
                            <div class="kpi-label">ANN Prediction</div></div>""", unsafe_allow_html=True)

                # Demand level card (own row for clarity)

                st.markdown(f"""<div class="kpi-card" style="margin-top:14px;">
                    <div class="kpi-icon">{emoji}</div>
                    <div class="kpi-value" style="font-size:1.3rem;">{level}</div>
                    <div class="kpi-label">Overall Demand Level</div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<span class="{badge_class}">{emoji} {level} — Recommended action plan below</span>', unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">🔍 Model Prediction Comparison</div>', unsafe_allow_html=True)
                    bars_x = ["XGBoost"]
                    bars_y = [xgb_pred]
                    bars_c = ["#ff5e62"]
                    if ann_pred is not None:
                        bars_x.append("ANN"); bars_y.append(ann_pred); bars_c.append("#ffd452")
                    bars_x.append("Final"); bars_y.append(final_prediction); bars_c.append("#2ecc71")
                    comp_fig = go.Figure(data=[go.Bar(x=bars_x, y=bars_y, marker_color=bars_c)])
                    comp_fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                            paper_bgcolor="rgba(0,0,0,0)", height=320,
                                            margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Predicted Orders")
                    st.plotly_chart(comp_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_chart2:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📉 Recent Order History Used</div>', unsafe_allow_html=True)
                    hist_fig = go.Figure()
                    hist_fig.add_trace(go.Scatter(
                        x=["Week -4", "Week -3", "Week -2", "Week -1", "Forecast"],
                        y=[lag_4, lag_3, lag_2, lag_1, final_prediction],
                        mode="lines+markers",
                        line=dict(color="#ff9966", width=3),
                        marker=dict(size=9, color=["#ffd452"] * 4 + ["#2ecc71"]),
                    ))
                    hist_fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                            paper_bgcolor="rgba(0,0,0,0)", height=320,
                                            margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Orders")
                    st.plotly_chart(hist_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🧑‍🍳 Business Recommendations</div>', unsafe_allow_html=True)
                recs = generate_recommendations(level, final_prediction)
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown(f'<div class="rec-box">🍲 <b>Food Prep</b><br>{recs["prep"]}</div>', unsafe_allow_html=True)
                with r2:
                    st.markdown(f'<div class="rec-box">👥 <b>Staffing</b><br>{recs["staff"]}</div>', unsafe_allow_html=True)
                with r3:
                    st.markdown(f'<div class="rec-box">📦 <b>Inventory</b><br>{recs["inventory"]}</div>', unsafe_allow_html=True)
                with r4:
                    st.markdown(f'<div class="rec-box">♻️ <b>Waste Reduction</b><br>{recs["waste"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.caption(f"Weather noted for context: {weather_condition} (not part of the trained model's feature set — the source dataset has no weather column).")

# =============================================================================
# PAGE: ANALYTICS & INSIGHTS
# =============================================================================
elif page == "📈 Analytics & Insights":

    if train_df is None or meal_df is None or center_df is None:
        st.warning("Analytics require train.csv, meal_info.csv, and fulfilment_center_info.csv in the project root.")
    else:
        merged = train_df.merge(meal_df, on="meal_id", how="left").merge(center_df, on="center_id", how="left")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🍜 Cuisine-wise Average Demand</div>', unsafe_allow_html=True)
        cuisine_avg = merged.groupby("cuisine")["num_orders"].mean().sort_values(ascending=False)
        fig_c = px.bar(x=cuisine_avg.index, y=cuisine_avg.values, color=cuisine_avg.values,
                       color_continuous_scale="thermal", template="plotly_dark")
        fig_c.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             height=350, margin=dict(l=10, r=10, t=20, b=10),
                             xaxis_title="Cuisine", yaxis_title="Avg Orders", showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🏢 Demand Share by Center Type</div>', unsafe_allow_html=True)
            type_share = merged.groupby("center_type")["num_orders"].sum()
            fig_pie = px.pie(names=type_share.index, values=type_share.values, hole=0.45,
                              color_discrete_sequence=px.colors.sequential.Sunset)
            fig_pie.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                   paper_bgcolor="rgba(0,0,0,0)", height=340,
                                   margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📢 Promotion Impact on Orders</div>', unsafe_allow_html=True)
            promo_impact = merged.assign(
                promo_status=np.where(
                    (merged["emailer_for_promotion"] == 1) | (merged["homepage_featured"] == 1),
                    "Promoted", "Not Promoted"
                )
            ).groupby("promo_status")["num_orders"].mean()
            fig_promo = px.bar(x=promo_impact.index, y=promo_impact.values, color=promo_impact.index,
                                template="plotly_dark",
                                color_discrete_sequence=["#ff5e62", "#2ecc71"])
            fig_promo.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                     height=340, margin=dict(l=10, r=10, t=20, b=10),
                                     yaxis_title="Avg Orders", showlegend=False)
            st.plotly_chart(fig_promo, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 Checkout Price vs Demand</div>', unsafe_allow_html=True)
        sample = merged.sample(min(8000, len(merged)), random_state=42)
        fig_scatter = px.scatter(sample, x="checkout_price", y="num_orders", color="category",
                                  opacity=0.5, template="plotly_dark")
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   height=400, margin=dict(l=10, r=10, t=20, b=10),
                                   xaxis_title="Checkout Price (₹)", yaxis_title="Number of Orders")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if xgb_model is not None and feature_columns is not None:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🧠 XGBoost Feature Importance</div>', unsafe_allow_html=True)
            try:
                importances = xgb_model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": importances
                }).sort_values("Importance", ascending=True)
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="agsunset", template="plotly_dark")
                fig_fi.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                      height=550, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.info(f"Feature importance not available: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: AI BUSINESS INSIGHTS
# =============================================================================
elif page == "🧠 AI Business Insights":

    if not FULL_DATA_READY:
        st.warning("AI insights require train.csv, meal_info.csv, and fulfilment_center_info.csv in the project root.")
    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Automated Business Insights</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#d8d8f0;'>Every insight below is mined directly from "
            "<code>train.csv</code> at run time — rising and falling demand pairs are found by "
            "comparing each center+meal combination's last 4 weeks against the prior 4 weeks, "
            "exactly like the model's own <code>lag</code> features.</p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Mining historical patterns..."):
            insights = generate_ai_insights(train_df, meal_df, center_df)

        if not insights:
            st.info("Not enough historical depth per center+meal combination to generate trend-based insights yet.")
        else:
            for ins in insights:
                st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-title">{ins['icon']} {ins['title']}</div>
                    <div class="insight-text">{ins['text']}</div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# PAGE: ABOUT PROJECT
# =============================================================================
elif page == "ℹ️ About Project":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    **Smart Restaurant AI – Food Demand Forecasting System** predicts weekly food demand
    for a fulfilment-center + meal combination using a trained **XGBoost Regressor**,
    based on the Food Demand Forecasting dataset (train.csv, meal_info.csv,
    fulfilment_center_info.csv).

    **Modeling pipeline:**
    - Merged transactional data with meal and fulfilment-center reference tables
    - Engineered `price_difference`, `discount_percentage`, `promotion`, `week_of_month`
    - Engineered `lag_1`–`lag_4` and `rolling_mean_4` / `rolling_std_4` per center+meal history
    - Label-encoded `category`, `cuisine`, `center_type`
    - Removed `base_price` due to high collinearity with `checkout_price`
    - Trained and compared Linear Regression, Decision Tree, Random Forest,
      Gradient Boosting, Extra Trees, XGBoost, and an ANN
    - **XGBoost was the top performer:** R² = 0.8217, MAE = 71.35, RMSE = 155.01

    **Tech stack:** Python, Streamlit, XGBoost, Scikit-learn, Plotly, Pandas, NumPy

    **Business value:** helps fulfilment centers plan food preparation quantity,
    staff scheduling, and inventory ordering ahead of time, reducing both
    stockouts during high-demand weeks and food waste during low-demand weeks.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#c9c9e8;'>Smart Restaurant AI © 2026 | Powered by XGBoost</p>",
    unsafe_allow_html=True
)
