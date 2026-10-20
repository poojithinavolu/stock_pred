# 📈 Stock Price Predictor & AIOps Telemetry Service

A complete machine learning web application built with **Streamlit**, containerized with **Docker**, version-controlled with **Git**, and instrumented with **AIOps Telemetry** for real-time inference latency and performance anomaly tracking.

---

## 🚀 Features

- **Interactive Streamlit Web Interface:** Live stock selection (`AAPL`, `TSLA`, `NVDA`, `MSFT`) or manual parameter sliders.
- **Machine Learning Forecast:** Random Forest Regressor predicting next-day close price based on technical features (`Open`, `High`, `Low`, `Volume`, `5-day MA`, `20-day MA`).
- **Interactive Historical Trend Charts:** Dynamic market data visualization.
- **AIOps Telemetry & Health Monitoring:** Real-time logging of prediction request latency and automated anomaly detection alerts.
- **Production-Ready Dockerization:** Multi-stage lightweight Python container with built-in healthchecks.

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit
- **ML Framework:** scikit-learn, NumPy, Pandas, joblib
- **Containerization:** Docker (Linux Alpine/Debian slim base)
- **Version Control:** Git & GitHub

---

## 💻 1. Local Setup & Execution

### Prerequisites
- Python 3.10+ installed

### Steps:
```bash
# 1. Clone repository
git clone https://github.com/poojithinavolu/stock_pred.git
cd stock_pred

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model artifacts
python train_model.py

# 5. Launch Streamlit App
streamlit run app.py
```
App will open at: `http://localhost:8501`

---

## 🐳 2. Docker Build & Deployment

### Build the Docker Image:
```bash
docker build -t stock-predictor:latest .
```

### Run the Container:
```bash
docker run -d -p 8501:8501 --name stock-app stock-predictor:latest
```

### Access App:
Open your browser and navigate to:
```text
http://localhost:8501
```

### Stop the Container:
```bash
docker stop stock-app
docker rm stock-app
```

---

## 🛡️ AIOps Operations

Every prediction automatically writes request telemetry to `aiops_access_logs.csv` including:
- Request Timestamp
- Stock Ticker
- Prediction Value
- Execution Latency ($ms$)
- Response Status (200 OK)

The built-in **AIOps Telemetry Tab** continuously evaluates latency thresholds and raises anomaly warnings when latency degrades beyond standard operational SLAs (>50ms).
