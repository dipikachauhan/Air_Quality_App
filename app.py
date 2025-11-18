import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import pydeck as pdk
from datetime import datetime, timedelta
import datetime as dt

def load_model_safe(path):
    try:
        from tensorflow.keras.models import load_model
        return load_model(path, compile=False)
    except:
        import keras
        return keras.models.load_model(path, compile=False)

TIME_STEPS = 72
SITE_COORDS = {
    "site_1": {"name": "Site 1", "lat": 28.69536, "lon": 77.18168},
    "site_2": {"name": "Site 2", "lat": 28.57180, "lon": 77.07125},
    "site_3": {"name": "Site 3", "lat": 28.58278, "lon": 77.23441},
}
DELHI_POLYGON = [
    (28.83, 76.95),
    (28.60, 76.96),
    (28.45, 77.05),
    (28.42, 77.22),
    (28.47, 77.37),
    (28.62, 77.45),
    (28.78, 77.37),
    (28.86, 77.22),
    (28.83, 76.95),
]
HEATMAP_POINTS = 1800
FORECAST_HORIZONS = [24, 48, 72]
DEFAULT_CONTEXT = {
    "year": 2025,
    "month": 3,
    "day": 1,
    "hour": 9,
    "T_forecast": 26.0,
    "q_forecast": 11.5,
    "u_forecast": -1.0,
    "v_forecast": 0.5,
    "w_forecast": 0.0,
    "O3_forecast": 65.0,
    "NO2_forecast": 45.0,
}

O3_AQI = [(0, 50, 'Good'), (51, 100, 'Moderate'), (101, 168, 'Poor'), (169, 240, 'Severe')]
NO2_AQI = [(0, 40, 'Good'), (41, 80, 'Moderate'), (81, 180, 'Poor'), (181, 280, 'Severe')]

AQI_COLORS = {
    "Good": "#4CAF50",
    "Moderate": "#FFC107",
    "Poor": "#FF9800",
    "Severe": "#F44336"
}

AQI_RECOMMENDATIONS = {
    'Good': "Air quality is satisfactory. No health impacts expected.",
    'Moderate': "Air quality acceptable. Sensitive individuals may experience minor effects.",
    'Poor': "Air quality is unhealthy for sensitive groups. Limit outdoor activities.",
    'Severe': "Air quality is very unhealthy. Avoid outdoor exposure."
}

def load_models_and_scalers(site_name, pollutant):
    X_scaler = joblib.load(f"saved_scalers/{site_name}_X_scaler.pkl")
    y_scaler = joblib.load(f"saved_scalers/{site_name}_{pollutant}_Y_scaler.pkl")
    lstm = load_model_safe(f"saved_models/{site_name}/{site_name}_{pollutant}_LSTM.keras")
    gru  = load_model_safe(f"saved_models/{site_name}/{site_name}_{pollutant}_GRU.keras")
    trans= load_model_safe(f"saved_models/{site_name}/{site_name}_{pollutant}_TRANS.keras")
    return X_scaler, y_scaler, (lstm, gru, trans)

def get_model_bundle(site_name, pollutant):
    cache = st.session_state.setdefault("model_cache", {})
    key = f"{site_name}_{pollutant}"
    if key not in cache:
        cache[key] = load_models_and_scalers(site_name, pollutant)
    return cache[key]

def classify_aqi(value, pollutant):
    ranges = O3_AQI if pollutant == 'O3' else NO2_AQI
    for low, high, cat in ranges:
        if low <= value <= high:
            return cat
    return "Severe"
def prepare_input(data, X_scaler, time_steps=72):
    O3_curr = float(data.get("O3_forecast", 0))
    NO2_curr = float(data.get("NO2_forecast", 0))
    T_val = data["T_forecast"]
    q_val = data["q_forecast"]
    u_val = data["u_forecast"]
    v_val = data["v_forecast"]
    w_val = data["w_forecast"]
    base_dt = dt.datetime(int(data["year"]), int(data["month"]), int(data["day"]), int(data["hour"]))
    O3_lags = [O3_curr] * time_steps
    NO2_lags = [NO2_curr] * time_steps
    rows = []
    for t in range(time_steps):
        step_dt = base_dt - dt.timedelta(hours=(time_steps - 1 - t))
        hour = step_dt.hour
        month = step_dt.month
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        if t == 0:
            O3_diff = 0
            NO2_diff = 0
        else:
            O3_diff = O3_lags[t] - O3_lags[t-1]
            NO2_diff = NO2_lags[t] - NO2_lags[t-1]
        row = {
            'O3_forecast': O3_curr,
            'NO2_forecast': NO2_curr,
            'T_forecast': T_val,
            'q_forecast': q_val,
            'u_forecast': u_val,
            'v_forecast': v_val,
            'w_forecast': w_val,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'O3_diff': O3_diff,
            'NO2_diff': NO2_diff
        }
        for lag in range(1, time_steps + 1):
            row[f"O3_target_lag_{lag}"] = O3_lags[lag-1]
            row[f"NO2_target_lag_{lag}"] = NO2_lags[lag-1]
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[X_scaler.feature_names_in_]
    X_scaled = X_scaler.transform(df)
    X_seq = np.expand_dims(X_scaled, axis=0)
    return X_seq

def predict_ensemble(X_seq, models, y_scaler):
    lstm, gru, trans = models
    y_scaled = (0.4 * gru.predict(X_seq) +
                0.4 * lstm.predict(X_seq) +
                0.2 * trans.predict(X_seq))
    y = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    return y[0]

def _get_site_values(pollutant):
    preds = st.session_state.get("latest_preds", {})
    defaults = {"O3": 60, "NO2": 45}
    rows = []
    for site, info in SITE_COORDS.items():
        value = preds.get(site, {}).get(pollutant)
        if value is None or np.isnan(value):
            value = defaults[pollutant]
        rows.append(
            {
                "site_id": site,
                "site": info["name"],
                "lat": info["lat"],
                "lon": info["lon"],
                "value": float(value),
            }
        )
    return rows

def _point_in_polygon(lat, lon, polygon):
    inside = False
    x = lon
    y = lat
    n = len(polygon)
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[(i + 1) % n]
        intersects = ((xi > x) != (xj > x)) and (
            y < (yj - yi) * (x - xi) / (xj - xi + 1e-12) + yi
        )
        if intersects:
            inside = not inside
    return inside

def _sample_delhi_points(num_points):
    latitudes = [p[0] for p in DELHI_POLYGON]
    longitudes = [p[1] for p in DELHI_POLYGON]
    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(longitudes), max(longitudes)
    samples = []
    attempts = 0
    max_attempts = num_points * 12
    rng = np.random.default_rng()
    while len(samples) < num_points and attempts < max_attempts:
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        if _point_in_polygon(lat, lon, DELHI_POLYGON):
            samples.append((lat, lon))
        attempts += 1
    return samples or [(info["lat"], info["lon"]) for info in SITE_COORDS.values()]
def get_heatmap_dataframe(pollutant):
    sites = _get_site_values(pollutant)
    sampled_points = _sample_delhi_points(HEATMAP_POINTS)
    cloud = []
    for lat, lon in sampled_points:
        weights = []
        values = []
        for site in sites:
            dist = np.sqrt((lat - site["lat"]) ** 2 + (lon - site["lon"]) ** 2) + 1e-6
            weights.append(1 / dist)
            values.append(site["value"])
        grid_value = float(np.average(values, weights=weights))
        cloud.append({"lat": lat, "lon": lon, "value": grid_value})
    return pd.DataFrame(cloud)

def generate_forecast_series(base_input, site, o3_bundle=None, no2_bundle=None, horizon=24):
    base_dt = datetime(
        int(base_input["year"]),
        int(base_input["month"]),
        int(base_input["day"]),
        int(base_input["hour"]),
    )
    if o3_bundle is None:
        o3_bundle = load_models_and_scalers(site, "O3")
    if no2_bundle is None:
        no2_bundle = load_models_and_scalers(site, "NO2")
    X_scaler_o3, y_scaler_o3, models_o3 = o3_bundle
    X_scaler_no2, y_scaler_no2, models_no2 = no2_bundle
    series = []
    for step in range(horizon):
        current_dt = base_dt + timedelta(hours=step)
        step_input = base_input.copy()
        step_input.update(
            {
                "year": current_dt.year,
                "month": current_dt.month,
                "day": current_dt.day,
                "hour": current_dt.hour,
            }
        )
        X_seq_o3 = prepare_input(step_input, X_scaler_o3)
        value_o3 = predict_ensemble(X_seq_o3, models_o3, y_scaler_o3)
        X_seq_no2 = prepare_input(step_input, X_scaler_no2)
        value_no2 = predict_ensemble(X_seq_no2, models_no2, y_scaler_no2)
        series.append(
            {
                "timestamp": current_dt,
                "O₃ (µg/m³)": value_o3,
                "NO₂ (µg/m³)": value_no2,
            }
        )
    return pd.DataFrame(series)

def ensure_forecast_series(site, horizon, base_input=None):
    cache = st.session_state.setdefault("forecast_cache", {})
    key = f"{site}_{horizon}"
    if key in cache and not cache[key].empty:
        return cache[key]
    base = DEFAULT_CONTEXT.copy()
    if base_input:
        base.update(base_input)
    o3_bundle = get_model_bundle(site, "O3")
    no2_bundle = get_model_bundle(site, "NO2")
    df = generate_forecast_series(base, site, o3_bundle=o3_bundle, no2_bundle=no2_bundle, horizon=horizon)
    cache[key] = df
    return df

st.set_page_config(
    page_title="Delhi Air Quality Prediction",
    page_icon="🌫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        body {
            background: radial-gradient(circle at top, #101726 0%, #0d1117 35%, #06080d 100%);
            color: #f5f7fa;
        }
        section[data-testid="stSidebar"] {
            background: #0f1521;
        }
        .hero-card {
            background: linear-gradient(135deg, #1f2e5a, #131b2f);
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 30px 80px rgba(4,10,45,0.45);
            color: #f5f7fa;
        }
        .data-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 18px;
            min-height: 160px;
        }
        .aq-card {
            padding: 20px;
            border-radius: 16px;
            margin-top: 15px;
            color: white;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        .rec-card {
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 12px;
            background: rgba(255,255,255,0.03);
            border-left: 4px solid;
        }
        .stButton>button {
            background: linear-gradient(135deg, #22d3ee, #2563eb);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            font-size: 1rem;
            border-radius: 12px;
            font-weight: 600;
            transition: transform 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px) scale(1.01);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:center; margin-top:20px; margin-bottom:10px;'>
        <h1 style='font-size:2.5rem; font-weight:800;'>
            🌬 Air Predict — Delhi Air Quality Forecasting System
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["About", "Prediction", "Visualization"])

with tabs[0]:
    st.markdown(
        """
        <div class="hero-card">
            <h1>🌆 Delhi Air Quality Intelligence</h1>
            <p style="font-size:1.1rem; opacity:0.9;">
                Real-time inference powered by an ensemble of LSTM, GRU and Transformer models.
                Built for CPCB specifications with meteorology-aware forecasting.
            </p>
            <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:20px;">
                <div style="flex:1; min-width:220px;">
                    <h3>🎯 Ensemble Models</h3>
                    <p>Blends temporal strengths from three deep architectures for robust predictions.</p>
                </div>
                <div style="flex:1; min-width:220px;">
                    <h3>🪁 Weather Context</h3>
                    <p>Ingests synoptic-scale temperature, humidity and wind vectors.</p>
                </div>
                <div style="flex:1; min-width:220px;">
                    <h3>🩺 Health Guidance</h3>
                    <p>Maps concentrations to CPCB AQI bands with instant recommendations.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔍 Why This Matters")
    cols = st.columns(3)
    highlights = [
        ("⏱ 72-step temporal context", "Captures three-day lag structure for stable inference."),
        ("🧠 Transfer-ready models", "Swap saved Keras weights per monitoring site in seconds."),
        ("📡 Ready for live feeds", "Plug in forecast APIs or observational feeds without retraining.")
    ]
    for col, (title, desc) in zip(cols, highlights):
        with col:
            st.markdown(
                f"""
                <div class="data-card">
                    <h4>{title}</h4>
                    <p style="opacity:0.8;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
with tabs[1]:
    st.header("🧪 Predict Air Quality")
    st.caption("Configure meteorological context and optionally inject pollutant forecasts.")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗓 Temporal Context")
            year  = st.number_input("Year", 2000, 2030, 2025)
            month = st.slider("Month", 1, 12, 3)
            day   = st.slider("Day", 1, 31, 1)
            hour  = st.slider("Hour (24h)", 0, 23, 16)
            site  = st.selectbox("Monitoring Site", ["site_1", "site_2", "site_3"])

        with col2:
            st.subheader("🌡 Atmospheric Inputs")
            Temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
            Humidity = st.number_input("Specific Humidity (g/kg)", 0.0, 30.0, 10.0)
            Wind_U = st.number_input("U-component (m/s)", -15.0, 15.0, 0.0)
            Wind_V = st.number_input("V-component (m/s)", -15.0, 15.0, 0.0)
            Vertical_Wind = st.number_input("Vertical wind (m/s)", -5.0, 5.0, 0.0)
            O3_forecast  = st.number_input("Weather Model O₃ Forecast (µg/m³)", 0.0, 500.0, 65.0)
            NO2_forecast = st.number_input("Weather Model NO₂ Forecast (µg/m³)", 0.0, 500.0, 45.0)


    if st.button("🚀 Run Ensemble Inference"):

        input_data = {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "T_forecast": Temperature,
            "q_forecast": Humidity,
            "u_forecast": Wind_U,
            "v_forecast": Wind_V,
            "w_forecast": Vertical_Wind
        }

        if O3_forecast:
            input_data["O3_forecast"] = float(O3_forecast)
        if NO2_forecast:
            input_data["NO2_forecast"] = float(NO2_forecast)

        with st.spinner("Calibrating scalers and running ensemble..."):
            o3_bundle = get_model_bundle(site, "O3")
            X_scaler_o3, y_scaler_o3, models_o3 = o3_bundle
            X_seq_o3 = prepare_input(input_data, X_scaler_o3)
            o3_pred = predict_ensemble(X_seq_o3, models_o3, y_scaler_o3)
            o3_class = classify_aqi(o3_pred, "O3")

            no2_bundle = get_model_bundle(site, "NO2")
            X_scaler_no2, y_scaler_no2, models_no2 = no2_bundle
            X_seq_no2 = prepare_input(input_data, X_scaler_no2)
            no2_pred = predict_ensemble(X_seq_no2, models_no2, y_scaler_no2)
            no2_class = classify_aqi(no2_pred, "NO2")

            forecast_df = generate_forecast_series(
                input_data, site, o3_bundle=o3_bundle, no2_bundle=no2_bundle, horizon=24
            )

        latest_preds = st.session_state.setdefault("latest_preds", {})
        latest_preds[site] = {"O3": o3_pred, "NO2": no2_pred}
        st.session_state["latest_series"] = forecast_df
        st.session_state["last_site"] = site

        st.subheader("✨ Predicted Concentrations")
        col_a, col_b = st.columns(2)
        for col, label, value, cat in [
            (col_a, "O₃ Concentration", o3_pred, o3_class),
            (col_b, "NO₂ Concentration", no2_pred, no2_class)
        ]:
            with col:
                color = AQI_COLORS[cat]
                st.markdown(
                    f"""
                    <div class="aq-card" style="background:{color};">
                        <p style="margin:0; opacity:0.8;">{label}</p>
                        <h2 style="margin:5px 0 10px;">{value:.2f} µg/m³</h2>
                        <p style="margin:0; font-weight:600;">Status · {cat}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.subheader("🩺 Health Recommendations")
        rec_cols = st.columns(2)
        for col, pollutant, klass in [
            (rec_cols[0], "O₃", o3_class),
            (rec_cols[1], "NO₂", no2_class)
        ]:
            color = AQI_COLORS[klass]
            with col:
                st.markdown(
                    f"""
                    <div class="rec-card" style="border-color:{color};">
                        <strong>{pollutant} · {klass}</strong>
                        <p style="margin-top:6px;">{AQI_RECOMMENDATIONS[klass]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with tabs[2]:
    st.header("📈 Operational Insights")
    st.caption("Bring your own feeds later – currently seeded with placeholders.")

    default_site = st.session_state.get("last_site", "site_1")
    site_names = list(SITE_COORDS.keys())
    site_index = site_names.index(default_site) if default_site in site_names else 0

    control_col1, control_col2 = st.columns([2, 1])
    with control_col1:
        insight_site = st.selectbox(
            "Monitoring Site",
            site_names,
            index=site_index,
            format_func=lambda s: SITE_COORDS[s]["name"],
        )
    with control_col2:
        horizon = st.radio(
            "Forecast Horizon",
            options=FORECAST_HORIZONS,
            index=0,
            format_func=lambda h: f"{h} hours",
        )

    forecast_df = ensure_forecast_series(insight_site, horizon)

    latest_preds = st.session_state.get("latest_preds", {})
    latest_site_vals = latest_preds.get(insight_site, {})
    if not latest_site_vals:
        snapshot = forecast_df.iloc[0]
        latest_site_vals = {
            "O3": snapshot["O₃ (µg/m³)"],
            "NO2": snapshot["NO₂ (µg/m³)"],
        }

    def fmt_value(val):
        return f"{val:.1f} µg/m³" if val is not None else "–"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"O₃ Forecast · {SITE_COORDS[insight_site]['name']}",
            fmt_value(latest_site_vals.get("O3")),
        )
    with col2:
        st.metric(
            f"NO₂ Forecast · {SITE_COORDS[insight_site]['name']}",
            fmt_value(latest_site_vals.get("NO2")),
        )
    with col3:
        st.metric("Sites Online", f"{len(SITE_COORDS)} / {len(SITE_COORDS)}", "stable")

    st.markdown(f"#### {horizon}h Ensemble Forecast")
    if not forecast_df.empty:
        chart_df = forecast_df.set_index("timestamp")
        st.line_chart(chart_df)
        st.caption(
            f"Auto-generated from stored ensembles · {SITE_COORDS[insight_site]['name']} ({horizon}-hour lookahead)."
        )
    else:
        st.info("Forecast series unavailable. Please check saved models and scalers.")

    st.markdown("#### Spatial Scatter Map")

pollutant_choice = st.radio(
    "Select pollutant for display",
    options=["O3", "NO2"],
    format_func=lambda x: "O₃" if x == "O3" else "NO₂",
    horizontal=True,
)

site_data = []
for site_id, info in SITE_COORDS.items():
    pred_val = latest_preds.get(site_id, {}).get(pollutant_choice)
    if pred_val is None:
        pred_val = 0.0
    site_data.append(
        {
            "site": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "value": float(pred_val),
        }
    )

scatter_df = pd.DataFrame(site_data)

def value_to_color(value):
    if value < 40:
        return [76, 175, 80, 180]
    elif value < 80:
        return [255, 193, 7, 180]
    elif value < 150:
        return [255, 152, 0, 180]
    else:
        return [244, 67, 54, 180]

scatter_df["color"] = scatter_df["value"].apply(value_to_color)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=scatter_df,
    get_position="[lon, lat]",
    get_radius=350,
    get_fill_color="color",
    pickable=True,
    opacity=0.8,
)

view_state = pdk.ViewState(
    latitude=28.62,
    longitude=77.20,
    zoom=11,
    pitch=45,
)

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v11",
    initial_view_state=view_state,
    layers=[scatter_layer],
    tooltip={"text": "Site: {site}\\nValue: {value} µg/m³"},
)

st.pydeck_chart(deck, use_container_width=True)
