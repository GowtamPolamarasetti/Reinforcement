import pandas as pd
import numpy as np
import os

# Paths
RENKO_PATH = "Data/Raw/renko_with_outcomes.csv"
RAW_PATH = "Data/Raw/XAUUSD_data_ohlc.csv"

def load_and_merge_data():
    print("Loading Renko Data...")
    renko_df = pd.read_csv(RENKO_PATH)
    renko_df['date'] = pd.to_datetime(renko_df['date'])
    
    print("Loading Raw 1-Min Data (this might take a moment)...")
    # Assuming Raw data has columns: date, open, high, low, close, volume (or similar)
    # We'll read a subset if it's too huge, but let's try reading all first.
    # Read header first to find date column
    header = pd.read_csv(RAW_PATH, nrows=0)
    possible_date_cols = [c for c in header.columns if 'date' in c.lower() or 'time' in c.lower()]
    
    if not possible_date_cols:
        raise ValueError("Could not find date/time column in raw data")
        
    date_col = possible_date_cols[0]
    print(f"Detected raw date column: {date_col}")
    
    raw_df = pd.read_csv(RAW_PATH, parse_dates=[date_col])
    
    # Standardize to 'date'
    raw_df.rename(columns={date_col: 'date'}, inplace=True)
    
    # Normalize Timezones (Remove TZ info to avoid merge errors)
    if renko_df['date'].dt.tz is not None:
        renko_df['date'] = renko_df['date'].dt.tz_localize(None)
    if raw_df['date'].dt.tz is not None:
        raw_df['date'] = raw_df['date'].dt.tz_localize(None)
        
    # Sort both by date
    renko_df = renko_df.sort_values('date')
    raw_df = raw_df.sort_values('date')
    
    print(f"Renko records: {len(renko_df)}")
    print(f"Raw records: {len(raw_df)}")
    
    # We want to attach raw data context to each renko brick
    # Since Renko close time MUST exist in raw data (as it's derived from it),
    # we can try an inner join or asof join.
    # Inner join on 'date' is safest for exact matches.
    
    # Rename raw columns to have 'raw_' prefix to separate them for merging
    raw_df_merge = raw_df.add_prefix('raw_')
    raw_df_merge = raw_df_merge.rename(columns={'raw_date': 'date'})
    
    print("Merging datasets...")
    merged_df = pd.merge(renko_df, raw_df_merge, on='date', how='inner')
    
    print(f"Merged records: {len(merged_df)}")
    
    if len(merged_df) < len(renko_df) * 0.9:
        print("WARNING: Significant data loss during merge. Check timestamp alignment.")
        
    return merged_df, raw_df # Return clean raw_df for features

if __name__ == "__main__":
    merged, raw = load_and_merge_data()
    print(merged.head())
    print(merged.columns)
