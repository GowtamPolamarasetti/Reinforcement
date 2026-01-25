import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import os

class RegimeIdentifier:
    def __init__(self, models_dir='RL_Agent_Filter/models'):
        self.models_dir = models_dir
        self.model_1m = None
        self.model_renko = None
        self.scaler_1m = None
        self.scaler_renko = None
        
        os.makedirs(models_dir, exist_ok=True)

    def _calculate_features(self, df, source='1m'):
        """
        Calculate features for regime detection.
        For 1m: Log Returns, Rolling Volatility
        For Renko: Brick Returns (Close-Open), Duration (Time diff)
        """
        df = df.copy()
        
        if source == '1m':
            # returns = df['close'].pct_change()
            # Handle potential zeros if needed, but close usually > 0
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_ret'].rolling(window=20).std()
            df = df.dropna()
            return df[['log_ret', 'volatility']]
            
        elif source == 'renko':
            # Renko features: 
            # 1. Brick Return: (Close - Prev_Close) / Prev_Close
            # 2. Duration: Time since prev brick
            
            # Ensure date is datetime
            if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.datetime64):
                 df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
            
            df['prev_close'] = df['close'].shift(1)
            df['log_ret'] = np.log(df['close'] / df['prev_close'])
            
            # Duration in seconds
            df['duration'] = (df['date'] - df['date'].shift(1)).dt.total_seconds()
            
            # Handle first row NaNs
            df = df.dropna()
            
            # Normalize duration by log just in case of huge outliers
            df['log_duration'] = np.log1p(df['duration'])
            
            return df[['log_ret', 'log_duration']]

    def fit_1m_model(self, csv_path, n_components=3):
        print("Loading 1m data for Regime Training...")
        df = pd.read_csv(csv_path)
        features = self._calculate_features(df, source='1m')
        
        print("Fitting GMM for 1m data...")
        self.model_1m = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.model_1m.fit(features)
        
        joblib.dump(self.model_1m, os.path.join(self.models_dir, 'gmm_1m.joblib'))
        print("1m Regime Model Saved.")
        
    def fit_renko_model(self, renko_df, n_components=3):
        print("Calculating features for Renko Regime Training...")
        features = self._calculate_features(renko_df, source='renko')
        
        print("Fitting GMM for Renko data...")
        self.model_renko = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.model_renko.fit(features)
        
        joblib.dump(self.model_renko, os.path.join(self.models_dir, 'gmm_renko.joblib'))
        print("Renko Regime Model Saved.")

    def load_models(self):
        try:
            self.model_1m = joblib.load(os.path.join(self.models_dir, 'gmm_1m.joblib'))
            self.model_renko = joblib.load(os.path.join(self.models_dir, 'gmm_renko.joblib'))
            print("Regime models loaded successfully.")
            return True
        except FileNotFoundError:
            print("Models not found. Please train first.")
            return False

    def predict_1m(self, df_window):
        """Predict regime for a window of 1m data (returns last point's regime)"""
        if self.model_1m is None:
            raise ValueError("1m Model not loaded")
        
        feats = self._calculate_features(df_window, source='1m')
        if feats.empty:
            return 0 # Default
        
        # Predict for all, return last
        regimes = self.model_1m.predict(feats)
        return regimes[-1]

    def predict_renko(self, renko_row, prev_renko_row):
        """
        Predict regime for a single Renko transition (requires prev row for diffs)
        This is a bit tricky for 'latest' if we only have one row.
        Ideally we pass a small history dataframe.
        """
        if self.model_renko is None:
             raise ValueError("Renko Model not loaded")
             
        # Reconstruct mini df
        mini_df = pd.DataFrame([prev_renko_row, renko_row])
        feats = self._calculate_features(mini_df, source='renko')
        
        if feats.empty:
            return 0
            
        return self.model_renko.predict(feats)[-1]
        
    def predict_renko_batch(self, renko_df):
        """Predict regimes for entire dataframe"""
        if self.model_renko is None:
            raise ValueError("Renko Model not loaded")
        
        feats = self._calculate_features(renko_df, source='renko')
        # Re-align with original DF? The features will be shorter by 1
        # We want to return an array matching the FEATURES length
        regimes = self.model_renko.predict(feats)
        return regimes, feats.index

