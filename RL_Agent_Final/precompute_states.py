
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

from .features.regime import RegimeIdentifier
from .features.bilstm import BiLSTMModel
from .features.indicators import TechnicalIndicators
from .features.structure import StructureFeatures

def precompute():
    print("Starting State Pre-computation...")
    
    # 1. Load Data
    renko_path = 'Data/Processed/renko_with_predictions.csv'
    ohlc_path = 'Data copy/Raw/XAUUSD_data_ohlc.csv' # Using the copy as discovered before
    
    if not os.path.exists(renko_path):
        print(f"Error: {renko_path} not found.")
        return
        
    print("Loading datasets...")
    renko_df = pd.read_csv(renko_path)
    # Restore datetime objects
    renko_df['date'] = pd.to_datetime(renko_df['date'])
    # Strip TZ if present
    if renko_df['date'].dt.tz is not None:
        renko_df['date'] = renko_df['date'].dt.tz_localize(None)
    
    # Sort
    renko_df = renko_df.sort_values('date').reset_index(drop=True)
    
    # Filter by days (train/test split happens in Env, but we precompute all)
    unique_days = renko_df['date'].dt.date.unique()
    print(f"Total Days: {len(unique_days)}")
    print(f"Total Bricks: {len(renko_df)}")
    
    # 2. Initialize Engines
    print("Initializing Feature Engines...")
    regime_engine = RegimeIdentifier()
    regime_engine.load_models()
    
    bilstm_engine = BiLSTMModel()
    bilstm_engine.load()
    
    indicators_engine = TechnicalIndicators(ohlc_path)
    indicators_engine.load_data()
    
    structure_engine = StructureFeatures()
    
    # 3. Precompute Loop
    # Obs Dim = 21
    # [Regime(2), LSTM(1), Struct(4), Ind(8), PnL(1), Time(1), Preds(4)]
    # PnL is dynamic (index 15), we will leave it as 0.0
    
    num_samples = len(renko_df)
    obs_dim = 21
    states = np.zeros((num_samples, obs_dim), dtype=np.float32)
    
    print("Computing states...")
    
    # We need to act brick by brick to maintain history context where needed
    # (e.g. prev_brick for structural flip, or regime transitions)
    
    for i in tqdm(range(num_samples)):
        brick = renko_df.iloc[i]
        current_time = brick['date']
        
        # --- 1. Regime (2) ---
        # 1m Regime
        latest_1m = indicators_engine.get_latest_regime_data(current_time)
        try:
             regime_1m = regime_engine.predict_1m(latest_1m)
        except:
             regime_1m = 0.0
             
        # Renko Regime
        if i > 0:
            prev_brick = renko_df.iloc[i - 1]
        else:
            prev_brick = brick # Fallback
            
        try:
            regime_renko = regime_engine.predict_renko(brick, prev_brick)
        except:
            regime_renko = 0.0
            
        # --- 2. BiLSTM (1) ---
        seq = brick.get('sequence', '')
        # Handle bool/str/int trend
        val = brick['uptrend']
        if isinstance(val, str):
            trend_val = 1 if val.lower() == 'true' else -1
        else:
            trend_val = 1 if val else -1
            
        lstm_conf = bilstm_engine.predict(seq, trend_val)
        
        # --- 3. Structure (4) ---
        struct_feats = structure_engine.get_features(brick, prev_brick if i > 0 else None)
        
        # --- 4. Indicators (8) ---
        if i > 0:
            start_time = renko_df.iloc[i - 1]['date']
        else:
            start_time = current_time - pd.Timedelta(minutes=60)
            
        ind_feats = indicators_engine.get_aggregated_features(start_time, current_time)
        
        # --- 5. PnL (1) ---
        # Dynamic, set to 0.0
        pnl_feat = 0.0
        
        # --- 6. Time Left (1) ---
        day_date = current_time.date()
        # To calculate time left accurately, we need total bricks for *this specific day*.
        # But that requires knowing the future count.
        # renko_env.py does: `total_bricks = len(self.day_data)`
        # `time_left = (total_bricks - self.current_step_idx) / total_bricks`
        # We can precompute this by grouping.
        # Doing it inside the loop is slow O(N^2) if we filter every time.
        # Let's pre-calculate day counts.
        
        # (Handling below outside loop for optimization, but for now placeholder)
        time_left = 0.0 # Will fill later
        
        # --- 7. Predictions (4) ---
        # Binary Win, Multi Win, Binary Loss, Multi Loss
        prob_bin_win = brick.get('prob_binary_win', 0.5)
        prob_multi_win = brick.get('prob_multi_win', 0.33)
        prob_bin_loss = brick.get('prob_binary_loss', 0.5)
        prob_multi_loss = brick.get('prob_multi_loss', 0.33)
        
        preds_feat = [prob_bin_win, prob_multi_win, prob_bin_loss, prob_multi_loss]
        
        # Concatenate
        # Order: [Regime(2), LSTM(1), Struct(4), Ind(8), PnL(1), Time(1), Preds(4)]
        vector = np.concatenate([
            [regime_1m, regime_renko, lstm_conf],
            struct_feats,
            ind_feats,
            [pnl_feat, time_left],
            preds_feat
        ])
        
        states[i] = vector

    # --- Fix Time Left ---
    print("Computing Time Left...")
    # Group by date to get daily counts
    renko_df['date_only'] = renko_df['date'].dt.date
    day_counts = renko_df.groupby('date_only').size()
    
    # We need to map each row to its index within the day and the total for that day
    # Cumulative count per day
    renko_df['day_idx'] = renko_df.groupby('date_only').cumcount()
    
    for i in range(num_samples):
        d = renko_df.iloc[i]['date_only']
        total = day_counts[d]
        current = renko_df.iloc[i]['day_idx']
        
        # Env logic: time_left = (total_bricks - current_step_idx) / total_bricks
        # In env, current_step_idx starts at 0.
        # So at start: (Total - 0) / Total = 1.0. 
        # At end (last step): (Total - (Total-1)) / Total = 1/Total.
        
        t_left = (total - current) / total if total > 0 else 0
        
        # Index of Time Left is: 2+1+4+8+1 = 16
        states[i, 16] = t_left

    # 4. Save
    out_path = 'Data/Processed/renko_states.npy'
    np.save(out_path, states)
    print(f"States saved to {out_path}. Shape: {states.shape}")

if __name__ == "__main__":
    precompute()
