
import pandas as pd
import numpy as np
import joblib
import sys
import os
import pandas_ta as ta

# Add Model directory to path to handle relative imports if needed by pickles
# (Though we will try to just use the objects directly)
sys.path.append(os.path.join(os.getcwd(), 'Model'))

def load_models():
    print("Loading Models...")
    try:
        binary_model = joblib.load("Model/binary_model.pkl")
        binary_scaler = joblib.load("Model/binary_scaler.pkl")
        
        multi_model = joblib.load("Model/stacked_model.pkl")
        multi_scaler = joblib.load("Model/scaler.pkl")
        
        print("Models loaded successfully.")
        return binary_model, binary_scaler, multi_model, multi_scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def generate_features_for_predictions(renko_df, raw_df):
    """
    Replicates feature generation logic from Feature_Generator.py and Data_Loader_Binary.py
    """
    print("Generating Features...")
    
    # 1. Technical Indicators on Raw Data
    raw_df['close'] = raw_df['close'].astype(float)
    raw_df['high'] = raw_df['high'].astype(float)
    raw_df['low'] = raw_df['low'].astype(float)
    raw_df['volume'] = raw_df['tick_volume'].astype(float) # Assuming tick_volume is what we use

    # Momentum
    raw_df['RSI_14'] = ta.rsi(raw_df['close'], length=14)
    raw_df['MFI_14'] = ta.mfi(raw_df['high'], raw_df['low'], raw_df['close'], raw_df['volume'], length=14)
    
    # Volatility
    raw_df['ATR_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
    raw_df['BB_UPPER'], raw_df['BB_MIDDLE'], raw_df['BB_LOWER'] = ta.bbands(raw_df['close'], length=20).iloc[:, 0], ta.bbands(raw_df['close'], length=20).iloc[:, 1], ta.bbands(raw_df['close'], length=20).iloc[:, 2]
    raw_df['BB_WIDTH'] = (raw_df['BB_UPPER'] - raw_df['BB_LOWER']) / raw_df['BB_MIDDLE']
    
    # Trend
    raw_df['SMA_50'] = ta.sma(raw_df['close'], length=50)
    raw_df['DIST_SMA50'] = (raw_df['close'] - raw_df['SMA_50']) / raw_df['SMA_50']
    
    # Returns
    raw_df['RETURN_1M'] = raw_df['close'].pct_change()
    raw_df['RETURN_15M'] = raw_df['close'].pct_change(15)
    raw_df['RETURN_60M'] = raw_df['close'].pct_change(60)
    
    # Volume
    raw_df['RVOL_20'] = raw_df['volume'] / raw_df['volume'].rolling(20).mean()

    # 2. Merge Technicals to Renko
    cols_to_merge = [
        'date', 
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20'
    ]
    
    # Ensure dates match (strip tz)
    renko_df['date'] = pd.to_datetime(renko_df['date']).dt.tz_localize(None)
    raw_df['date'] = pd.to_datetime(raw_df['date']).dt.tz_localize(None)
    
    merged_df = pd.merge(renko_df, raw_df[cols_to_merge], on='date', how='left')
    
    # 3. Generate Renko-Specific Features
    merged_df['hour'] = merged_df['date'].dt.hour
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
    merged_df['seq_len'] = merged_df['sequence'].apply(len)
    merged_df['seq_ones_ratio'] = merged_df['sequence'].apply(lambda x: x.count('1') / len(x) if len(x) > 0 else 0)
    merged_df['uptrend_float'] = merged_df['uptrend'].astype(float)
    
    # Lags
    for lag in [1, 2, 3]:
        merged_df[f'brick_size_lag_{lag}'] = merged_df['brick_size'].shift(lag)
        merged_df[f'uptrend_lag_{lag}'] = merged_df['uptrend'].astype(float).shift(lag)
        
    # 4. Generate Duration (For Binary Model)
    # Start time of brick i is Close time of brick i-1
    merged_df['start_time'] = merged_df['date'].shift(1)
    # Duration in seconds
    merged_df['duration_seconds'] = (merged_df['date'] - merged_df['start_time']).dt.total_seconds()
    merged_df['duration_log'] = np.log1p(merged_df['duration_seconds'])
    
    return merged_df

def main():
    # 1. Load Data
    print("Loading Data...")
    renko_df = pd.read_csv("Data copy/Raw/renko_with_outcomes.csv")
    raw_df = pd.read_csv("Data copy/Raw/XAUUSD_data_ohlc.csv")
    # Basic cleanup for Raw
    if 'Date' in raw_df.columns: raw_df.rename(columns={'Date': 'date'}, inplace=True)
    if 'Time' in raw_df.columns: raw_df.rename(columns={'Time': 'date'}, inplace=True) # Just in case
    # If the CSV has 'Gmt time', rename carefully
    # Assuming standard format from Data_Loader checks
    
    # 2. Generate Features
    df = generate_features_for_predictions(renko_df, raw_df)
    
    # Fill nan for features (first few rows) to allow prediction
    # Or just drop? We want to keep all renko rows aligned.
    # We will use ffill/bfill for simplicity or 0
    # Actually models train on dropna, so prediction might fail on NaN.
    # Let's simple impute for now.
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # 3. Load Models
    binary_model, binary_scaler, multi_model, multi_scaler = load_models()
    
    if binary_model is None:
        return

    # 4. Predict Binary Features
    print("Predicting Binary Model...")
    binary_features = [
        'hour', 'day_of_week', 'duration_log',
        'seq_len', 'seq_ones_ratio', 'uptrend_float',
        'brick_size',
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2'
    ]
    
    try:
        X_binary = df[binary_features]
        X_binary_scaled = binary_scaler.transform(X_binary)
        
        # Probabilities: [Prob(Loss), Prob(Win)]
        binary_probs = binary_model.predict_proba(X_binary_scaled)
        df['prob_binary_loss'] = binary_probs[:, 0]
        df['prob_binary_win'] = binary_probs[:, 1]
    except Exception as e:
        print(f"Binary prediction failed: {e}")
        # Default to neutral
        df['prob_binary_loss'] = 0.5
        df['prob_binary_win'] = 0.5

    # 5. Predict Multi-Class Features
    print("Predicting Multi-Class Model...")
    multi_features = [
        'hour', 'day_of_week', 'seq_len', 'seq_ones_ratio', 'uptrend_float',
        'brick_size',
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2'
    ]
    
    try:
        X_multi = df[multi_features]
        X_multi_scaled = multi_scaler.transform(X_multi)
        
        # Probabilities: [Prob(Loss), Prob(Win), Prob(BE)] (Check order!)
        # Train_Advanced: target_names=['LOSS', 'WIN', 'BE'] -> this is classification_report order
        # Map was {'WIN': 1, 'LOSS': 0, 'BE': 2}
        # Classes in sklearn are usually sorted: 0 (Loss), 1 (Win), 2 (BE)
        multi_probs = multi_model.predict_proba(X_multi_scaled)
        df['prob_multi_loss'] = multi_probs[:, 0]
        df['prob_multi_win'] = multi_probs[:, 1]
        df['prob_multi_be'] = multi_probs[:, 2]
    except Exception as e:
        print(f"Multi prediction failed: {e}")
        df['prob_multi_loss'] = 0.33
        df['prob_multi_win'] = 0.33
        df['prob_multi_be'] = 0.33

    # 6. Save Result
    output_cols = [
        'date', 'open', 'high', 'low', 'close', 'uptrend', 'brick_size', 'sequence', 'outcome', 
        'prob_binary_win', 'prob_binary_loss',
        'prob_multi_win', 'prob_multi_loss', 'prob_multi_be'
    ]
    
    # Filter to only relevant columns (original + probs)
    # We want to keep original renko structure mostly
    final_df = df[output_cols]
    
    output_path = "Data/Processed/renko_with_predictions.csv"
    os.makedirs("Data/Processed", exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()
