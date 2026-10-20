import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def generate_synthetic_stock_data(n_days=500):
    np.random.seed(42)
    # Generate realistic random walk for stock prices
    base_price = 150.0
    returns = np.random.normal(0.0008, 0.02, n_days)
    price_series = base_price * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
    
    data = []
    for i in range(20, n_days):
        close_today = price_series[i]
        high_today = close_today * (1 + abs(np.random.normal(0, 0.01)))
        low_today = close_today * (1 - abs(np.random.normal(0, 0.01)))
        open_today = low_today + np.random.uniform(0, 1) * (high_today - low_today)
        volume = int(np.random.normal(50000000, 12000000))
        ma_5 = np.mean(price_series[i-5:i])
        ma_20 = np.mean(price_series[i-20:i])
        
        # Target: next day close price
        if i + 1 < n_days:
            target_next_close = price_series[i+1]
        else:
            target_next_close = close_today * (1 + np.random.normal(0.0005, 0.015))
            
        data.append({
            'Date': dates[i].strftime('%Y-%m-%d'),
            'Open': round(float(open_today), 2),
            'High': round(float(high_today), 2),
            'Low': round(float(low_today), 2),
            'Close': round(float(close_today), 2),
            'Volume': max(1000000, volume),
            'MA_5': round(float(ma_5), 2),
            'MA_20': round(float(ma_20), 2),
            'Target_Next_Close': round(float(target_next_close), 2)
        })
        
    return pd.DataFrame(data)

def train_and_save():
    print("Generating training dataset...")
    df = generate_synthetic_stock_data(n_days=600)
    
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'MA_20']
    X = df[feature_cols]
    y = df['Target_Next_Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    
    # Save artifacts
    joblib.dump(model, 'stock_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save sample recent history for charting in Streamlit
    df.tail(60).to_csv('historical_sample.csv', index=False)
    
    metrics = {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4),
        'feature_names': feature_cols
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("Artifacts saved: stock_model.pkl, scaler.pkl, historical_sample.csv, metrics.json")

if __name__ == '__main__':
    train_and_save()
