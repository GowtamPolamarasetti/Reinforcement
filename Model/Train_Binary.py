import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
import sys

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Data_Loader_Binary import load_binary_data

MODEL_DIR = "Model"

def train_binary_model():
    print("--- Starting Binary (Win/Loss) Model Training ---")
    
    # 1. Load Data
    df = load_binary_data()
    
    # Target Map (Binary)
    outcome_map = {'WIN': 1, 'LOSS': 0}
    df['target'] = df['outcome'].map(outcome_map)
    df = df.dropna(subset=['target'])
    
    # Additional Binary Features
    # Duration was calculated in loader
    df['duration_log'] = np.log1p(df['duration_seconds'])
    
    # Renko Features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['seq_len'] = df['sequence'].apply(len)
    df['seq_ones_ratio'] = df['sequence'].apply(lambda x: x.count('1') / len(x) if len(x) > 0 else 0)
    df['uptrend_float'] = df['uptrend'].astype(float)
    
    # Lagged Renko features
    for lag in [1, 2, 3]:
        df[f'brick_size_lag_{lag}'] = df['brick_size'].shift(lag)
        df[f'uptrend_lag_{lag}'] = df['uptrend'].astype(float).shift(lag)

    # Drop NaNs
    df = df.dropna()
    
    # Feature Selection
    features = [
        'hour', 'day_of_week', 'duration_log',
        'seq_len', 'seq_ones_ratio', 'uptrend_float',
        'brick_size',
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2'
    ]
    
    X = df[features]
    y = df['target']
    
    print(f"Training Data Shape: {X.shape}")
    print(f"Class Balance:\n{y.value_counts(normalize=True)}")
    
    # Define Base Estimators
    estimators = [
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic', n_estimators=150, 
            learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss'
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31, 
            random_state=42, n_jobs=-1, verbose=-1
        )),
        ('cat', CatBoostClassifier(
            iterations=150, learning_rate=0.05, depth=6, 
            loss_function='Logloss', verbose=False, random_seed=42
        ))
    ]
    
    # Stacking Classifier
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracy_scores = []
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])
        
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        print(classification_report(y_test, preds, target_names=['LOSS', 'WIN']))
        
        accuracy_scores.append(acc)
        fold += 1
        
    avg_acc = np.mean(accuracy_scores)
    print(f"\nAverage Cross-Validation Accuracy: {avg_acc:.4f}")

    # Train Final Model
    print("Training Final Binary Model on Full Data...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    clf.fit(X_scaled, y)
    
    # Save artifacts
    joblib.dump(clf, os.path.join(MODEL_DIR, "binary_model.pkl"))
    joblib.dump(final_scaler, os.path.join(MODEL_DIR, "binary_scaler.pkl"))
    print("Binary Model saved.")

if __name__ == "__main__":
    train_binary_model()
