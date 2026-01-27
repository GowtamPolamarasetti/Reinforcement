import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, data_path='Raw/XAUUSD_data_ohlc.csv'):
        self.data_path = data_path
        self.df = None
        
    def update_live_data(self, window_df):
        """
        Updates the internal dataframe with the live window and recalculates indicators.
        """
        if window_df is None or window_df.empty:
            return
            
        # Ensure proper format
        df = window_df.copy()
        
        # FIX: If 'date' is not in columns, assume index is date and reset.
        if 'date' not in df.columns:
            df = df.reset_index()
            # If index was unnamed, it becomes 'index'. Rename it.
            if 'date' not in df.columns:
                # First column is likely the old index
                df.rename(columns={df.columns[0]: 'date'}, inplace=True)
             
        # Ensure datetime
        if 'date' in df.columns:
             # If already datetime, good. If int/float, convert?
             # Assuming caller passes clean DF compatible with load_data equivalent logic
             if not pd.api.types.is_datetime64_any_dtype(df['date']):
                 try:
                     df['date'] = pd.to_datetime(df['date'], unit='s' if df['date'].iloc[0] < 1e11 else 'ms', utc=True).dt.tz_convert(None)
                 except:
                     pass
        
        self.df = df.sort_values('date').reset_index(drop=True)
        self._calculate_indicators()

    def load_data(self):
        # print(f"Loading 1m data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path)
            # Ensure proper datetime
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'], utc=True).dt.tz_convert(None)
            elif 'time' in self.df.columns:
                 self.df['date'] = pd.to_datetime(self.df['time'], utc=True).dt.tz_convert(None)
            else:
                 # Fallback for standard ohlc
                 self.df['date'] = pd.to_datetime(self.df.iloc[:, 0], utc=True).dt.tz_convert(None)
                 
            self.df = self.df.sort_values('date').reset_index(drop=True)
            self._calculate_indicators()
            # print("1m Indicators calculated.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _calculate_indicators(self):
        # Calculate Base Indicators
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # 1. RSI(14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Safe division
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi'] = self.df['rsi'].fillna(50)
        
        # 2. ATR(14)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=14).mean().fillna(0)
        
        # 3. SMA(20) & Slope
        self.df['sma20'] = close.rolling(window=20).mean()
        self.df['sma20_slope'] = self.df['sma20'].diff()
        
        # 4. SMA(50) & Slope
        self.df['sma50'] = close.rolling(window=50).mean()
        self.df['sma50_slope'] = self.df['sma50'].diff()
        
        # 5. Price-MA Distance %
        self.df['dist_ma20_pct'] = (close - self.df['sma20']) / self.df['sma20']
        self.df['dist_ma50_pct'] = (close - self.df['sma50']) / self.df['sma50']
        
        # 6. MA Spread %
        self.df['ma_spread_pct'] = (self.df['sma20'] - self.df['sma50']) / self.df['sma50']
        
        # 7. Volatility (Std Dev of returns)
        self.df['returns'] = close.pct_change()
        self.df['volatility'] = self.df['returns'].rolling(20).std()
        
        # 8. Momentum Acceleration (Change in slope)
        self.df['mom_acc'] = self.df['sma20_slope'].diff()

        # Calculate % Changes for Aggregation
        # User defined: "% change per candle"
        # For unbounded values like slope, simple diff might be better, but user asked for % change.
        # For bounded (0-100) like RSI, diff is better.
        # I will compute simple diffs for stationary things, pct_change for prices.
        
        self.df['rsi_chg'] = self.df['rsi'].diff()
        self.df['atr_pct_chg'] = self.df['atr'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        self.df['sma20_slope_chg'] = self.df['sma20_slope'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        
        # Fill NaNs
        self.df = self.df.fillna(0)

    def get_aggregated_features(self, start_time, end_time):
        """
        Aggregate indicators between start_time (exclusive) and end_time (inclusive).
        Returns a dictionary or series of features.
        """
        # Slice data
        mask = (self.df['date'] > start_time) & (self.df['date'] <= end_time)
        chunk = self.df.loc[mask]
        
        if chunk.empty:
            # Fallback: take the last known values before end_time
            # or just zeros
            return np.zeros(8) # Assuming 8 features
            
        # Aggregation: Mean of changes/values
        # User: "Aggregate (mean or sum)"
        
        feats = [
            chunk['rsi_chg'].mean(),
            chunk['atr_pct_chg'].mean(),
            chunk['sma20_slope'].mean(), # Mean slope during brick
            chunk['sma50_slope'].mean(),
            chunk['dist_ma20_pct'].mean(),
            chunk['dist_ma50_pct'].mean(),
            chunk['ma_spread_pct'].mean(),
            chunk['volatility'].mean()
        ]
        
        return np.array(feats)
        
    def get_latest_regime_data(self, time):
        # Get 1m data for regime prediction at time t
        # Need window of 20 for volatility/returns
        # return dataframe slice
        idx = self.df['date'].searchsorted(time)
        if idx < 20:
             return self.df.iloc[0:idx+1]
        return self.df.iloc[idx-20:idx+1]
