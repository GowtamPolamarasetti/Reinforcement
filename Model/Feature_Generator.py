import pandas as pd
import pandas_ta as ta
import numpy as np

def generate_features(raw_df):
    print("Generating technical indicators on raw data...")
    # Ensure correct types
    raw_df['close'] = raw_df['close'].astype(float)
    raw_df['high'] = raw_df['high'].astype(float)
    raw_df['low'] = raw_df['low'].astype(float)
    raw_df['open'] = raw_df['open'].astype(float)
    raw_df['tick_volume'] = raw_df['tick_volume'].astype(float)
    
    # Check if 'volume' exists, if not use 'tick_volume'
    if 'volume' not in raw_df.columns and 'tick_volume' in raw_df.columns:
        raw_df['volume'] = raw_df['tick_volume']

    # Momentum
    raw_df['RSI_14'] = ta.rsi(raw_df['close'], length=14)
    raw_df['MFI_14'] = ta.mfi(raw_df['high'], raw_df['low'], raw_df['close'], raw_df['volume'], length=14)
    
    # Volatility
    raw_df['ATR_14'] = ta.atr(raw_df['high'], raw_df['low'], raw_df['close'], length=14)
    raw_df['BB_UPPER'], raw_df['BB_MIDDLE'], raw_df['BB_LOWER'] = ta.bbands(raw_df['close'], length=20).iloc[:, 0], ta.bbands(raw_df['close'], length=20).iloc[:, 1], ta.bbands(raw_df['close'], length=20).iloc[:, 2]
    raw_df['BB_WIDTH'] = (raw_df['BB_UPPER'] - raw_df['BB_LOWER']) / raw_df['BB_MIDDLE']
    
    # Trend
    raw_df['SMA_50'] = ta.sma(raw_df['close'], length=50)
    raw_df['EMA_200'] = ta.ema(raw_df['close'], length=200)
    raw_df['DIST_SMA50'] = (raw_df['close'] - raw_df['SMA_50']) / raw_df['SMA_50']
    
    # Returns
    raw_df['RETURN_1M'] = raw_df['close'].pct_change()
    raw_df['RETURN_15M'] = raw_df['close'].pct_change(15)
    raw_df['RETURN_60M'] = raw_df['close'].pct_change(60)
    
    # Volume
    raw_df['RVOL_20'] = raw_df['volume'] / raw_df['volume'].rolling(20).mean()
    
    print("Features generated.")
    return raw_df

if __name__ == "__main__":
    # Test
    from Data_Loader import load_and_merge_data
    merged, raw = load_and_merge_data()
    raw_with_features = generate_features(raw)
    
    # Merge features back to merged_df (which is just 'raw' features attached to renko events)
    # We need to act on the full 'raw' dataframe first to get correct indicator values, 
    # then merge those specific indicator columns to the renko dataframe.
    
    indicator_cols = ['date', 'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 'RETURN_15M', 'RETURN_60M', 'RVOL_20']
    
    final_df = pd.merge(merged, raw_with_features[indicator_cols], on='date', how='left')
    print(final_df.columns)
    print(final_df.head())
