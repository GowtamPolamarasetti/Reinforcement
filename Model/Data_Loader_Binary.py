import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Reuse the robust load function but we will process it further
from Data_Loader import load_and_merge_data

def load_binary_data():
    print("Loading and preparing Binary Data (Win/Loss)...")
    
    # 1. Load base data (Renko + Clean Raw, Merged is not fully needed yet if we want to calc start time on Renko first)
    # Actually load_and_merge_data returns (merged, raw).
    # We can use the 'renko_with_outcomes.csv' directly first to handle the chronological start time correctly, 
    # then merge.
    
    renko_path = "Data/Raw/renko_with_outcomes.csv"
    renko_df = pd.read_csv(renko_path)
    renko_df['date'] = pd.to_datetime(renko_df['date'])
    # Normalize TZ
    if renko_df['date'].dt.tz is not None:
        renko_df['date'] = renko_df['date'].dt.tz_localize(None)
        
    renko_df = renko_df.sort_values('date')
    
    # 2. Calculate Start Time
    # Start time of brick i is Close time of brick i-1
    renko_df['start_time'] = renko_df['date'].shift(1)
    
    # Drop first row as it has no start time
    renko_df = renko_df.dropna(subset=['start_time'])
    
    # Calculate Duration
    renko_df['duration'] = renko_df['date'] - renko_df['start_time']
    renko_df['duration_seconds'] = renko_df['duration'].dt.total_seconds()
    
    # 3. Filter Class
    # Drop 'BE'
    print(f"Records before filtering: {len(renko_df)}")
    renko_df = renko_df[renko_df['outcome'] != 'BE']
    print(f"Records after dropping BE: {len(renko_df)}")
    
    # 4. Merge with Raw Data Features
    # We need to re-run feature generation or import it.
    # To keep it simple, we'll re-load raw and generate.
    
    raw_path = "Data/Raw/XAUUSD_data_ohlc.csv"
    # Read raw for context
    # Use the robust reading logic from Data_Loader or just import Data_Loader's raw_df
    _, raw_df = load_and_merge_data() # We only need the raw_df part
    
    # Generate Features on raw_df
    from Feature_Generator import generate_features
    raw_w_features = generate_features(raw_df)
    
    # Select columns to merge
    cols_to_merge = [
        'date', 
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20'
    ]
    
    print("Merging features to binary dataset...")
    # Merge on 'date' (Close time of brick)
    binary_df = pd.merge(renko_df, raw_w_features[cols_to_merge], on='date', how='left')
    
    # Drop NaNs (from feature lags or merge missingness)
    binary_df = binary_df.dropna()
    
    print(f"Final Binary Dataset: {len(binary_df)} records")
    return binary_df

if __name__ == "__main__":
    df = load_binary_data()
    print(df.head())
    print(df['outcome'].value_counts())
