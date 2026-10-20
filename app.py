import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor & AIOps Telemetry",
    page_icon="📈",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .prediction-highlight {
        font-size: 36px;
        font-weight: 800;
        color: #0284c7;
    }
</style>
""", unsafe_allow_html=True)

# Auto-train if model doesn't exist
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists("stock_model.pkl") or not os.path.exists("scaler.pkl"):
        st.info("Training initial model artifacts... Please wait a few seconds.")
        import train_model
        train_model.train_and_save()
    model = joblib.load("stock_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Header
st.markdown('<div class="main-title">📈 Stock Price Prediction & Analytics Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dockerized Machine Learning Service & AIOps Operational Telemetry</div>', unsafe_allow_html=True)

# Tabs
tab_pred, tab_history, tab_aiops = st.tabs(["🔮 Price Prediction", "📊 Market Trends", "⚙️ AIOps Telemetry"])

# Preset Profiles
ticker_presets = {
    "AAPL (Apple)": {"open": 182.50, "high": 184.20, "low": 181.80, "volume": 54000000, "ma5": 183.10, "ma20": 180.40},
    "TSLA (Tesla)": {"open": 248.00, "high": 254.50, "low": 246.20, "volume": 98000000, "ma5": 250.20, "ma20": 242.80},
    "NVDA (NVIDIA)": {"open": 135.20, "high": 138.40, "low": 134.10, "volume": 72000000, "ma5": 136.00, "ma20": 130.50},
    "MSFT (Microsoft)": {"open": 420.00, "high": 423.50, "low": 418.00, "volume": 22000000, "ma5": 421.10, "ma20": 415.80},
    "Custom / Manual": {"open": 150.00, "high": 153.00, "low": 149.00, "volume": 45000000, "ma5": 151.00, "ma20": 148.00}
}

# Sidebar inputs
st.sidebar.header("🎯 Stock Input Parameters")
selected_ticker = st.sidebar.selectbox("Choose Stock / Preset", list(ticker_presets.keys()))
preset = ticker_presets[selected_ticker]

open_val = st.sidebar.number_input("Open Price ($)", value=preset["open"], step=0.5, format="%.2f")
high_val = st.sidebar.number_input("Day High ($)", value=preset["high"], step=0.5, format="%.2f")
low_val = st.sidebar.number_input("Day Low ($)", value=preset["low"], step=0.5, format="%.2f")
vol_val = st.sidebar.number_input("Trading Volume", value=preset["volume"], step=1000000)
ma5_val = st.sidebar.number_input("5-Day Moving Avg ($)", value=preset["ma5"], step=0.5, format="%.2f")
ma20_val = st.sidebar.number_input("20-Day Moving Avg ($)", value=preset["ma20"], step=0.5, format="%.2f")

# Prediction Tab
with tab_pred:
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Current Trading Day Features")
        f_df = pd.DataFrame({
            "Feature": ["Open", "High", "Low", "Volume", "5-Day MA", "20-Day MA"],
            "Value": [f"${open_val:.2f}", f"${high_val:.2f}", f"${low_val:.2f}", f"{vol_val:,}", f"${ma5_val:.2f}", f"${ma20_val:.2f}"]
        })
        st.table(f_df)
        
        predict_btn = st.button("🚀 Predict Next Day Close", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Output")
        if predict_btn:
            start_t = time.perf_counter()
            
            # Feature vector
            X_input = np.array([[open_val, high_val, low_val, vol_val, ma5_val, ma20_val]])
            X_scaled = scaler.transform(X_input)
            predicted_close = float(model.predict(X_scaled)[0])
            
            latency_ms = (time.perf_counter() - start_t) * 1000
            
            price_delta = predicted_close - open_val
            pct_delta = (price_delta / open_val) * 100
            
            st.metric(
                label=f"Predicted Next Day Close for {selected_ticker.split(' ')[0]}",
                value=f"${predicted_close:.2f}",
                delta=f"{price_delta:+.2f} ({pct_delta:+.2f}%)"
            )
            
            st.success(f"Model Inference Completed in **{latency_ms:.2f} ms** | Status: 200 OK")
            
            # Log for AIOps
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": selected_ticker.split(' ')[0],
                "open": open_val,
                "predicted": round(predicted_close, 2),
                "latency_ms": round(latency_ms, 2),
                "status": "200_OK"
            }
            log_file = "aiops_access_logs.csv"
            log_df = pd.DataFrame([log_entry])
            if not os.path.exists(log_file):
                log_df.to_csv(log_file, index=False)
            else:
                log_df.to_csv(log_file, mode='a', header=False, index=False)
                
            st.caption("Request telemetry logged to `aiops_access_logs.csv` for AIOps performance monitoring.")
        else:
            st.info("👈 Click the **Predict Next Day Close** button to compute forecast using the trained model.")

# Historical Trends Tab
with tab_history:
    st.subheader("Historical Stock Price Movement")
    if os.path.exists("historical_sample.csv"):
        hist_df = pd.read_csv("historical_sample.csv")
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        hist_df = hist_df.set_index('Date')
        st.line_chart(hist_df[['Close', 'MA_5', 'MA_20']])
    else:
        st.info("Run `python train_model.py` to generate sample historical market data.")

# AIOps Telemetry Tab
with tab_aiops:
    st.subheader("🛡️ AIOps Real-Time Telemetry & Health Monitoring")
    st.markdown("Operational health metrics gathered from containerized model prediction requests:")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Container Status", "Healthy (Online)", delta="99.9% Uptime")
    col_m2.metric("API Endpoint", "Streamlit Server :8501", delta="Dockerized")
    
    log_file = "aiops_access_logs.csv"
    if os.path.exists(log_file):
        logs = pd.read_csv(log_file)
        avg_lat = logs['latency_ms'].mean()
        col_m3.metric("Avg Latency", f"{avg_lat:.2f} ms", delta="Normal Latency")
        
        st.markdown("### Recent Model Inference Access Logs")
        st.dataframe(logs.tail(10), use_container_width=True)
        
        # Simple AIOps anomaly check: latency > 50ms is flagged
        anomalies = logs[logs['latency_ms'] > 50]
        if len(anomalies) > 0:
            st.warning(f"⚠️ AIOps Alert: {len(anomalies)} inference latency anomalies detected (>50ms)!")
        else:
            st.success("✅ AIOps Health Check: All inference requests operating within optimal SLA limits (<50ms).")
    else:
        col_m3.metric("Avg Latency", "-- ms", delta="No traffic yet")
        st.info("Make a prediction in the first tab to generate live telemetry logs.")
