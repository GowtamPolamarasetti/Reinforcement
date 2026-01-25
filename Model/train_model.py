import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Define constants
DATA_PATH = "Data/Raw/renko_with_outcomes.csv"
MODEL_DIR = "Model"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # Parse date
    df['date'] = pd.to_datetime(df['date'])
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Target Encoding
    # We want to predict WIN vs LOSS/BE or pure outcome
    # Let's map WIN: 1, LOSS: 0, BE: 2 (or drop BE, or treat as 3 classes)
    # Strategy: 3-class classification
    outcome_map = {'WIN': 1, 'LOSS': 0, 'BE': 2}
    df['target'] = df['outcome'].map(outcome_map)
    
    # Drop rows where outcome is NaN (if any)
    df = df.dropna(subset=['target'])
    
    # Feature Engineering
    
    # 1. Time features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # 2. Sequence features
    # sequence is a string of 0s and 1s. Let's extract some stats.
    # Length of sequence (volatility proxy?)
    df['seq_len'] = df['sequence'].apply(len)
    # Ratio of 1s (bullish pressure?)
    df['seq_ones_ratio'] = df['sequence'].apply(lambda x: x.count('1') / len(x) if len(x) > 0 else 0)
    # Count of 1s
    df['seq_ones_count'] = df['sequence'].apply(lambda x: x.count('1'))
    # Count of 0s
    df['seq_zeros_count'] = df['sequence'].apply(lambda x: x.count('0'))
    
    # 3. Lagged Features (Trend context)
    for lag in [1, 2, 3]:
        df[f'brick_size_lag_{lag}'] = df['brick_size'].shift(lag)
        df[f'uptrend_lag_{lag}'] = df['uptrend'].shift(lag).astype(float) # True/False to 1/0
    
    # 4. Current Trend
    df['uptrend_float'] = df['uptrend'].astype(float)
    
    # Drop NaNs created by shifting
    df = df.dropna()
    
    return df

def train_model(df):
    print("Training model...")
    
    features = [
        'brick_size', 'hour', 'day_of_week', 
        'seq_len', 'seq_ones_ratio', 'seq_ones_count', 'seq_zeros_count',
        'uptrend_float', 
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2',
        'brick_size_lag_3', 'uptrend_lag_3'
    ]
    
    X = df[features]
    y = df['target']
    
    # Time Series Split Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"Fold {fold}...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=3, 
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=['LOSS', 'WIN', 'BE']))
        
        fold += 1
    
    # Train on full dataset for final model
    print("Retraining on full dataset...")
    final_model = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=3, 
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=6,
        random_state=42
    )
    final_model.fit(X, y)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "outcome_predictor.pkl")
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Feature Importance
    importances = final_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_imp_df)
    
    feature_imp_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    feature_imp_df.to_csv(feature_imp_path, index=False)

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    train_model(df)
