
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Constants
XAUUSD_PATH = 'Data/Processed/renko_with_predictions.csv'
USDMXN_PATH = 'Data/Processed/renko_with_predictions_USDMXN.csv'
DAILY_DD_LIMIT = -3.0 # -3% Stopout

def evaluate_baseline(name, csv_path):
    print(f"Evaluating Baseline for {name}...")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Ensure correct date format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Handle timezones if present, normalize to None
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        df['date_only'] = df['date'].dt.date
    else:
        # Fallback if no date column, assume sequential
        df['date_only'] = 0
        
    days = df['date_only'].unique()
    print(f"Total Days: {len(days)}")
    
    initial_cap = 100000.0
    cap = initial_cap
    curve = [initial_cap]
    
    trades_pnl = []
    daily_returns = []
    
    outcomes = df['outcome'].values
    
    # We iterate day by day to enforce daily stopout
    grouped = df.groupby('date_only', sort=False)
    
    for day, group in grouped:
        day_pnl_pct = 0.0
        day_start_cap = cap
        
        # Iterate trades in the day
        day_outcomes = group['outcome'].values
        
        for outcome in day_outcomes:
            # Check Daily Limit FIRST
            if day_pnl_pct <= DAILY_DD_LIMIT:
                break # Stop trading for the day
            
            # Take Trade (Action = 1)
            trade_res = 0.0
            if outcome == 'WIN':
                trade_res = 0.5
            elif outcome == 'LOSS':
                trade_res = -0.5
            elif outcome == 'BE' or outcome == 'BREAKEVEN':
                trade_res = 0.0
            
            # Update
            day_pnl_pct += trade_res
            cap += cap * (trade_res / 100.0)
            trades_pnl.append(trade_res)
            
        # End of Day
        curve.append(cap)
        dr = (cap - day_start_cap) / day_start_cap if day_start_cap > 0 else 0.0
        daily_returns.append(dr)
        
    # Metrics
    total_ret = ((cap - initial_cap) / initial_cap) * 100
    
    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    
    pf = sum(wins)/abs(sum(losses)) if losses else 0
    wr = len(wins)/len(trades_pnl)*100 if trades_pnl else 0
    
    eq_np = np.array(curve)
    peaks = np.maximum.accumulate(eq_np)
    dds = (eq_np - peaks) / peaks
    max_dd = np.min(dds) * 100
    
    rets_np = np.array(daily_returns)
    sharpe = (np.mean(rets_np) / np.std(rets_np)) * np.sqrt(252) if np.std(rets_np) > 0 else 0
    
    print("\n" + "="*40)
    print(f"       BASELINE RESULTS: {name}       ")
    print("="*40)
    print(f"Strategy: Take All Trades (No AI Filter)")
    print(f"Stopout : {DAILY_DD_LIMIT}% Daily")
    print("-" * 40)
    print(f"Final Capital     : ${cap:,.2f}")
    print(f"Total Return (%)  : {total_ret:,.2f}%")
    print("-" * 40)
    print(f"Total Trades      : {len(trades_pnl)}")
    print(f"Win Rate (%)      : {wr:.2f}%")
    print(f"Profit Factor     : {pf:.2f}")
    print(f"Max Drawdown (%)  : {max_dd:.2f}%")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print("="*40)
    
    return curve

def main():
    curve_xau = evaluate_baseline("XAUUSD", XAUUSD_PATH)
    curve_mxn = evaluate_baseline("USDMXN", USDMXN_PATH)
    
    plt.figure(figsize=(12, 6))
    if curve_xau: plt.plot(curve_xau, label='Baseline XAUUSD', color='gray')
    if curve_mxn: plt.plot(curve_mxn, label='Baseline USDMXN', color='orange')
    plt.title('Baseline Performance (No AI Filtering)')
    plt.yscale('log')
    plt.legend()
    plt.savefig('RL_Agent_Final/baseline_comparison.png')
    print("Saved plot to RL_Agent_Final/baseline_comparison.png")

if __name__ == "__main__":
    main()
