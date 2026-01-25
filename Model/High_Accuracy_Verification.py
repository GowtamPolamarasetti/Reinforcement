import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import accuracy_score, classification_report

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Data_Loader import load_and_merge_data
from Feature_Generator import generate_features

MODEL_DIR = "Model"

def verify_high_accuracy():
    print("--- High Accuracy Verification ---")
    
    # Load Models
    clf = joblib.load(os.path.join(MODEL_DIR, "stacked_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # Load Data (Process again to be safe)
    merged_with_raw, raw_df = load_and_merge_data()
    raw_w_features = generate_features(raw_df)
    
    cols_to_merge = [
        'date', 
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20'
    ]
    df = pd.merge(merged_with_raw, raw_w_features[cols_to_merge], on='date', how='left')
    
    outcome_map = {'WIN': 1, 'LOSS': 0, 'BE': 2}
    df['target'] = df['outcome'].map(outcome_map)
    df = df.dropna(subset=['target'])
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['seq_len'] = df['sequence'].apply(len)
    df['seq_ones_ratio'] = df['sequence'].apply(lambda x: x.count('1') / len(x) if len(x) > 0 else 0)
    df['uptrend_float'] = df['uptrend'].astype(float)
    
    for lag in [1, 2, 3]:
        df[f'brick_size_lag_{lag}'] = df['brick_size'].shift(lag)
        df[f'uptrend_lag_{lag}'] = df['uptrend'].astype(float).shift(lag)

    df = df.dropna()
    
    features = [
        'hour', 'day_of_week', 'seq_len', 'seq_ones_ratio', 'uptrend_float',
        'brick_size',
        'RSI_14', 'MFI_14', 'ATR_14', 'BB_WIDTH', 'DIST_SMA50', 
        'RETURN_1M', 'RETURN_15M', 'RETURN_60M', 'RVOL_20',
        'brick_size_lag_1', 'uptrend_lag_1',
        'brick_size_lag_2', 'uptrend_lag_2'
    ]
    
    X = df[features]
    y = df['target']
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict Probabilities
    probs = clf.predict_proba(X_scaled)
    # Probs is (n_samples, 3) for [0: LOSS, 1: WIN, 2: BE]
    
    # Get max prob and predicted class
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    results = pd.DataFrame({'Target': y, 'Pred': preds, 'Confidence': max_probs})
    
    print(f"\nGlobal Accuracy: {accuracy_score(y, preds):.4f}")
    
    # Threshold Analysis
    print("\n--- Confidence Threshold Analysis ---")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    for t in thresholds:
        high_conf = results[results['Confidence'] >= t]
        if len(high_conf) == 0:
            print(f"Threshold >= {t}: No samples")
            continue
            
        acc = accuracy_score(high_conf['Target'], high_conf['Pred'])
        print(f"Threshold >= {t}: Accuracy = {acc:.4f} (Samples: {len(high_conf)} / {len(results)}) - Coverage: {len(high_conf)/len(results)*100:.1f}%")
        
        if acc > 0.70:
            print(f"*** FOUND > 70% ACCURACY at threshold {t} ***")
            print("Class Balance in High Conf:")
            print(high_conf['Target'].value_counts(normalize=True))

if __name__ == "__main__":
    verify_high_accuracy()
