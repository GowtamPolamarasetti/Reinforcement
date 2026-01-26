import joblib
import pandas as pd
from catboost import CatBoostClassifier
from config.definitions import M_CATBOOST_BINARY, M_CATBOOST_MULTI
from utils.logger import logger
import os

class OutcomePredictor:
    def __init__(self):
        self.bin_model = None
        self.multi_model = None
        
    def load(self):
        try:
            if os.path.exists(M_CATBOOST_BINARY):
                self.bin_model = joblib.load(M_CATBOOST_BINARY)
                logger.info("Binary Outcome Model loaded.")
            else:
                logger.warning(f"Binary Model not found at {M_CATBOOST_BINARY}")
                
            if os.path.exists(M_CATBOOST_MULTI):
                # Check if it's pickle or catboost native
                try:
                    self.multi_model = joblib.load(M_CATBOOST_MULTI)
                except:
                     self.multi_model = CatBoostClassifier()
                     self.multi_model.load_model(M_CATBOOST_MULTI)
                logger.info("Multi-class Outcome Model loaded.")
            else:
                logger.warning(f"Multi Model not found at {M_CATBOOST_MULTI}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading outcome models: {e}")
            return False
            
    def predict(self, feature_vector):
        """
        Predicts probabilities for the given features.
        The feature vector for CatBoost might differ from the RL agent 21-dim vector.
        Usually CatBoost takes the Raw Features or a subset?
        If `Feature_Generator` was used to train CatBoost, we need those features.
        
        Assumption: CatBoost models expect the *same* feature set as training.
        If we don't know the exact features CatBoost expects, we might crash.
        
        However, `precompute_states.py` extracted `prob_binary_win` FROM the `renko_with_predictions.csv`.
        This implies predictions were pre-generated offline.
        For LIVE trading, we must generate them ON THE FLY.
        
        We need to match the feature signature of the loaded model.
        """
        if not self.bin_model:
            return {'prob_bin_win': 0.5, 'prob_bin_loss': 0.5, 'prob_multi_win': 0.33, 'prob_multi_loss': 0.33}
            
        # Placeholder: CatBoost prediction logic depends on input features
        # If we pass the 21-dim RL vector? Unlikely.
        # CatBoost uses the "Atomic Features" (Indicators, etc).
        
        return {
            'prob_bin_win': 0.55, 
            'prob_bin_loss': 0.45,
            'prob_multi_win': 0.4,
            'prob_multi_loss': 0.3
        } 
