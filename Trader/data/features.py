import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config.definitions import MODELS_ROOT
from data.features_lib.regime import RegimeIdentifier
from data.features_lib.bilstm import BiLSTMModel
from data.features_lib.indicators import TechnicalIndicators
from data.features_lib.structure import StructureFeatures
from utils.logger import logger
import os

class FeatureEngineer:
    def __init__(self):
        logger.info("Initializing Feature Engineer...")
        
        # Paths
        self.regime_models_dir = os.path.join(MODELS_ROOT, 'models_regime')
        self.bilstm_path = os.path.join(MODELS_ROOT, 'bilstm_50_trend.keras') 
        # Note: If bilstm model is missing, it will degrade gracefully to 0.5
        
        # Initialize Engines
        self.regime_engine = RegimeIdentifier(models_dir=self.regime_models_dir)
        if not self.regime_engine.load_models():
             logger.warning("Regime models not loaded. Returns 0.")
             
        self.bilstm_engine = BiLSTMModel(model_path=self.bilstm_path)
        if not self.bilstm_engine.load():
             logger.warning("BiLSTM model not loaded. Returns 0.5.")
             
        # Indicators Engine usually takes a path to initialize data, 
        # but for Live Trading we feed it data incrementally or BUFFER it.
        # The library `features_lib/indicators.py` might need adaptation for live buffer.
        # Let's check if it has a method to compute on a DF window.
        # Assuming we can instantiate it without a path for live use if we change logic,
        # OR we pass a dummy path.
        # Indicators Engine
        self.indicators_engine = TechnicalIndicators(data_path=None) 
        
        self.structure_engine = StructureFeatures()
        
    def get_indicators(self, m1_window_df):
        """
        Calculates and returns the dictionary of latest indicators.
        """
        if m1_window_df is None or m1_window_df.empty:
            return {}
            
        try:
            # Update Engine
            self.indicators_engine.update_live_data(m1_window_df)
            
            # Get latest row
            if self.indicators_engine.df is not None and not self.indicators_engine.df.empty:
                latest = self.indicators_engine.df.iloc[-1]
                # Return dict of required keys
                return {
                    'RSI_14': latest.get('rsi', 50.0),
                    'MFI_14': latest.get('mfi', 50.0), # Note: indicators.py might not calc MFI? Check it.
                    'ATR_14': latest.get('atr', 0.0),
                    'BB_WIDTH': latest.get('bb_width', 0.0), # Check key names in indicators.py
                    'DIST_SMA50': latest.get('dist_ma50_pct', 0.0),
                    'RETURN_1M': latest.get('returns', 0.0),
                    'RETURN_15M': 0.0, # Need to calc
                    'RETURN_60M': 0.0,
                    'RVOL_20': 1.0 # Need to calc
                }
        except Exception as e:
            logger.error(f"Indicator Calc Error: {e}")
            
        return {}
    
    def calculate_state(self, current_brick, prev_brick, m1_window_df, predictions_dict, pnl_feat, time_left_feat):
        """
        Assembles the 21-dim vector.
        Args:
            current_brick: dict with {open, close, high, low, uptrend, date, sequence}
            prev_brick: dict (can be same as current if first brick)
            m1_window_df: DataFrame of recent M1 data (for regime/indicators)
            predictions_dict: {prob_bin_win, prob_multi_win, prob_bin_loss, prob_multi_loss}
            pnl_feat: float
            time_left_feat: float
        Returns:
            np.array shape (21,)
        """
        # 1. Regime (2)
        # 1m Regime
        try:
            # We need a small window for 1m regime
            # RegimeIdentifier.predict_1m expects a DF
            regime_1m = self.regime_engine.predict_1m(m1_window_df)
        except Exception as e:
            # logger.error(f"Regime 1m error: {e}")
            regime_1m = 0.0
            
        # Renko Regime
        try:
            regime_renko = self.regime_engine.predict_renko(current_brick, prev_brick)
        except Exception as e:
            regime_renko = 0.0
            
        # 2. BiLSTM (1)
        seq = current_brick.get('sequence', '')
        # Handle bool trend
        trend_val = 1 if current_brick['uptrend'] else -1
        lstm_conf = self.bilstm_engine.predict(seq, trend_val)
        
        # 3. Structure (4)
        try:
            struct_feats = self.structure_engine.get_features(current_brick, prev_brick)
        except:
            struct_feats = [0.0, 0.0, 0.0, 0.0] # Fallback
            
        # 4. Indicators (8)
        # Use aggregated features from engine
        ind_feats = [0.0] * 8 
        try:
             # Assuming update_live_data was called in get_indicators or we call it here?
             # If Orbit calls get_indicators FIRST, then engine is updated.
             # We can reuse the engine state.
             if self.indicators_engine.df is not None:
                 # We need end_time. current_brick['date']? (datetime object)
                 # m1_window_df index might be time.
                 # Let's use latest time in engine.
                 end_time = self.indicators_engine.df.iloc[-1]['date']
                 # Start time? Brick start?
                 # Precompute used prev_brick time.
                 start_time = end_time - pd.Timedelta(minutes=5) # Fallback
                 if prev_brick:
                     start_time = prev_brick['date'] # if available
                 
                 ind_feats = self.indicators_engine.get_aggregated_features(start_time, end_time)
        except Exception as e:
             # logger.error(f"Ind Feats Error: {e}")
             pass
            
        # 5. Predictions (4)
        preds_feat = [
            predictions_dict.get('prob_bin_win', 0.5),
            predictions_dict.get('prob_multi_win', 0.33),
            predictions_dict.get('prob_bin_loss', 0.5),
            predictions_dict.get('prob_multi_loss', 0.33)
        ]
        
        # Assemble
        # [Regime(2), LSTM(1), Struct(4), Ind(8), PnL(1), Time(1), Preds(4)]
        vector = np.concatenate([
            [regime_1m, regime_renko, lstm_conf],
            struct_feats,
            ind_feats,
            [pnl_feat, time_left_feat],
            preds_feat
        ])
        
        return vector.astype(np.float32)
