
import sys
import os
import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta
import xgboost
from tqdm import tqdm

# Ensure we can import from RL_Agent_RewardOpt
sys.path.append(os.getcwd())

try:
    from RL_Agent_RewardOpt.features.regime import RegimeIdentifier
    from RL_Agent_RewardOpt.features.bilstm import BiLSTMModel
    from RL_Agent_RewardOpt.features.indicators import TechnicalIndicators
    from RL_Agent_RewardOpt.features.structure import StructureFeatures
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Paths
RENKO_RAW_PATH = "Raw/renko_with_outcomes_USDMXN.csv"
OHLC_RAW_PATH = "Raw/USD_MXN_data_bid.csv"
RENKO_PRED_OUT = "Data/Processed/renko_with_predictions_USDMXN.csv"
STATES_OUT = "Data/Processed/renko_states_USDMXN.npy"

# --- PHASE 1: PREDICTIONS ---

def load_prediction_models():
    print("Loading Prediction Models (XAUUSD Trained)...")
    try:
        binary_model = joblib.load("Model/binary_model.pkl")
        binary_scaler = joblib.load("Model/binary_scaler.pkl")
        multi_model = joblib.load("Model/stacked_model.pkl")
        multi_scaler = joblib.load("Model/scaler.pkl")
        return binary_model, binary_scaler, multi_model, multi_scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def generate_features(renko_df, raw_df):
    print("Generating Features for Prediction...")
    
    # 1. Tech Indicators
    raw_df['close'] = raw_df['close'].astype(float)
    raw_df['high'] = raw_df['high'].astype(float)
    raw_df['low'] = raw_df['low'].astype(float)
    
    # Tick Volume normalization
    if 'tick_volume' in raw_df.columns:
        raw_df['volume'] = raw_df['tick_volume'].astype(float)
    else:
        raw_df['volume'] = 1.0 # Fallback
    
    # Indicators
    raw_df['RSI_14'] = ta.rsi(raw_df['close'], length=14)
    raw_df['MFI_14'] = ta.mfi(raw_df['high'], raw_df['low'], raw_df['close'], raw_df['volume'], length=14)
    raw_df['ATR_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
    
    bb = ta.bbands(raw_df['close'], length=20)
    if bb is not None:
        raw_df['BB_WIDTH'] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
    else:
        raw_df['BB_WIDTH'] = 0
        
    sma = ta.sma(raw_df['close'], length=50)
    raw_df['DIST_SMA50'] = (raw_df['close'] - sma) / sma
    
    raw_df['RETURN_1M'] = raw_df['close'].pct_change()
    raw_df['RETURN_15M'] = raw_df['close'].pct_change(15)
    raw_df['RETURN_60M'] = raw_df['close'].pct_change(60)
    
    vol_roll = raw_df['volume'].rolling(20).mean()
    raw_df['RVOL_20'] = raw_df['volume'] / vol_roll
    
    # 2. Merge
    cols_to_merge = [
        'date', 'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20'
    ]
    
    # Clean Dates
    renko_df['date'] = pd.to_datetime(renko_df['date'])
    if renko_df['date'].dt.tz is not None:
        renko_df['date'] = renko_df['date'].dt.tz_localize(None)
        
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    # Handle USDMXN raw format '2017-01-02 01:27:00+00:00'
    if raw_df['date'].dt.tz is not None:
        raw_df['date'] = raw_df['date'].dt.tz_localize(None)
        
    merged = pd.merge(renko_df, raw_df[cols_to_merge], on='date', how='left')
    
    # 3. Renko Features
    merged['hour'] = merged['date'].dt.hour
    merged['day_of_week'] = merged['date'].dt.dayofweek
    merged['seq_len'] = merged['sequence'].apply(lambda x: len(str(x)))
    merged['seq_ones_ratio'] = merged['sequence'].apply(lambda x: str(x).count('1') / len(str(x)) if len(str(x)) > 0 else 0)
    merged['uptrend_float'] = merged['uptrend'].astype(float)
    
    # Lags
    for lag in [1, 2, 3]:
        merged[f'brick_size_lag_{lag}'] = merged['brick_size'].shift(lag)
        merged[f'uptrend_lag_{lag}'] = merged['uptrend'].astype(float).shift(lag)
        
    # Duration
    merged['start_time'] = merged['date'].shift(1)
    merged['duration_seconds'] = (merged['date'] - merged['start_time']).dt.total_seconds()
    merged['duration_log'] = np.log1p(merged['duration_seconds'])
    
    # Fill Nans
    merged = merged.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return merged

def run_predictions():
    print(f"--- Processing Predictions for {RENKO_RAW_PATH} ---")
    if not os.path.exists(RENKO_RAW_PATH):
        print("Raw Renko File not found.")
        return False
        
    renko_df = pd.read_csv(RENKO_RAW_PATH)
    raw_df = pd.read_csv(OHLC_RAW_PATH)
    
    # Generate
    df = generate_features(renko_df, raw_df)
    
    # Load Models
    b_model, b_scaler, m_model, m_scaler = load_prediction_models()
    if b_model is None: return False
    
    # Predict Binary
    b_feats = [
        'hour', 'day_of_week', 'duration_log',
        'seq_len', 'seq_ones_ratio', 'uptrend_float',
        'brick_size',
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2'
    ]
    
    try:
        X_b = b_scaler.transform(df[b_feats])
        probs_b = b_model.predict_proba(X_b)
        df['prob_binary_loss'] = probs_b[:, 0]
        df['prob_binary_win'] = probs_b[:, 1]
    except Exception as e:
        print(f"Binary Prediction Error: {e}")
        df['prob_binary_win'] = 0.5
        
    # Predict Multi
    m_feats = [
        'hour', 'day_of_week', 'seq_len', 'seq_ones_ratio', 'uptrend_float',
        'brick_size',
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2'
    ]
    
    try:
        X_m = m_scaler.transform(df[m_feats])
        probs_m = m_model.predict_proba(X_m)
        df['prob_multi_loss'] = probs_m[:, 0]
        df['prob_multi_win'] = probs_m[:, 1]
        df['prob_multi_be'] = probs_m[:, 2]
    except Exception as e:
        print(f"Multi Prediction Error: {e}")
        df['prob_multi_win'] = 0.33
        
    # Save
    out_cols = [
        'date', 'open', 'high', 'low', 'close', 'uptrend', 'brick_size', 'sequence', 'outcome', 
        'prob_binary_win', 'prob_binary_loss',
        'prob_multi_win', 'prob_multi_loss', 'prob_multi_be'
    ]
    
    # We keep open/high/low from renko if they exist, or merged
    # Renko CSv has open, close, high, low
    final_df = df[out_cols]
    final_df.to_csv(RENKO_PRED_OUT, index=False)
    print(f"Saved Predictions to {RENKO_PRED_OUT}")
    return True

# --- PHASE 2: STATES ---

def run_states():
    print(f"--- Computing RL States for {RENKO_PRED_OUT} ---")
    
    if not os.path.exists(RENKO_PRED_OUT):
        print("Prediction file missing.")
        return
        
    renko_df = pd.read_csv(RENKO_PRED_OUT)
    renko_df['date'] = pd.to_datetime(renko_df['date'])
    
    # Engines
    regime_engine = RegimeIdentifier()
    regime_engine.load_models() # Loads XAUUSD regime models
    
    bilstm_engine = BiLSTMModel()
    bilstm_engine.load()
    
    # Important: Indicators Engine needs OHLC
    print(f"Loading Technical Indicators from {OHLC_RAW_PATH}...")
    indicators_engine = TechnicalIndicators(OHLC_RAW_PATH)
    indicators_engine.load_data()
    
    structure_engine = StructureFeatures()
    
    num_samples = len(renko_df)
    obs_dim = 21
    states = np.zeros((num_samples, obs_dim), dtype=np.float32)
    
    # Pre-calc Time Left helpers
    renko_df['date_only'] = renko_df['date'].dt.date
    day_counts = renko_df.groupby('date_only').size()
    renko_df['day_idx'] = renko_df.groupby('date_only').cumcount()
    
    print(f"Computing {num_samples} states...")
    
    for i in tqdm(range(num_samples)):
        brick = renko_df.iloc[i]
        current_time = brick['date']
        
        # 1. Regime
        try:
            latest_1m = indicators_engine.get_latest_regime_data(current_time)
            regime_1m = regime_engine.predict_1m(latest_1m)
        except: regime_1m = 0.0
        
        if i > 0: prev = renko_df.iloc[i-1]
        else: prev = brick
        
        try:
            regime_renko = regime_engine.predict_renko(brick, prev)
        except: regime_renko = 0.0
        
        # 2. LSTM
        seq = str(brick.get('sequence', ''))
        val = brick['uptrend']
        trend_val = 1 if (val == True or str(val).lower()=='true') else -1
        lstm_conf = bilstm_engine.predict(seq, trend_val)
        
        # 3. Structure
        struct = structure_engine.get_features(brick, prev)
        
        # 4. Indicators
        if i > 0: start = renko_df.iloc[i-1]['date']
        else: start = current_time - pd.Timedelta(minutes=60)
        ind_feats = indicators_engine.get_aggregated_features(start, current_time)
        
        # 5. PnL (Placeholder)
        pnl = 0.0
        
        # 6. Time Left
        d = brick['date_only']
        total = day_counts.get(d, 1)
        curr_idx = brick['day_idx']
        time_left = (total - curr_idx) / total if total > 0 else 0
        
        # 7. Predictions
        p_vec = [
            brick.get('prob_binary_win', 0.5),
            brick.get('prob_multi_win', 0.33),
            brick.get('prob_binary_loss', 0.5),
            brick.get('prob_multi_loss', 0.33)
        ]
        
        vector = np.concatenate([
            [regime_1m, regime_renko, lstm_conf],
            struct,
            ind_feats,
            [pnl, time_left],
            p_vec
        ])
        states[i] = vector
        
    np.save(STATES_OUT, states)
    print(f"States saved to {STATES_OUT}")

if __name__ == "__main__":
    if run_predictions():
        run_states()
