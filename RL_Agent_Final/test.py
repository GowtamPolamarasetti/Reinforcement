
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import seaborn as sns

try:
    from renko_env_fast import RenkoFilterFastEnv
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv

def evaluate():
    model_path = "RL_Agent_Final/models/final_model"
    if not os.path.exists(model_path + ".zip"):
         print(f"Model not found at {model_path}")
         return
         
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Init Env in Test Mode
    # IMPORTANT: Must use same feature mask as training
    env = RenkoFilterFastEnv(
        renko_path='Data/Processed/renko_with_predictions.csv',
        states_path='Data/Processed/renko_states.npy',
        reward_config=None, # Reward config irrelevant for inference/testing
        mode='test', # Uses last 20% of data
        split_ratio=0.8,
        mask_indices=[3, 4, 5, 6] 
    )
    
    num_days = len(env.days)
    print(f"Evaluating on {num_days} Test Days...")
    
    # Tracking
    initial_capital = 100000.0
    current_capital = initial_capital
    equity_curve = [initial_capital]
    
    # Trade Lists
    all_trades_pnl_pct = [] 
    
    win_count = 0
    loss_count = 0
    be_count = 0
    
    daily_returns = []
    dates = []
    
    for i in range(num_days):
        obs, info = env.reset()
        day_date = env.current_day_val
        dates.append(day_date)
        
        terminated = False
        truncated = False
        
        day_start_equity = current_capital
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            
            # Peek outcome
            brick = env.renko_df.iloc[env.current_step_idx]
            original_outcome = brick.get('outcome', None)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if action == 1: # TAKE
                pnl_pct = 0.0
                if original_outcome == 'WIN':
                    pnl_pct = 0.5
                    win_count += 1
                elif original_outcome == 'LOSS':
                    pnl_pct = -0.5
                    loss_count += 1
                elif original_outcome in ['BE', 'BREAKEVEN']:
                    pnl_pct = 0.0
                    be_count += 1
                
                # Update Capital
                pnl_amount = current_capital * (pnl_pct / 100.0)
                current_capital += pnl_amount
                
                all_trades_pnl_pct.append(pnl_pct)
        
        # End of Day
        equity_curve.append(current_capital)
        day_return = (current_capital - day_start_equity) / day_start_equity
        daily_returns.append(day_return)
        
        # Retrieve Agent PnL (Sum of %)
        pnl = env.daily_pnl
        
        print(f"Day {i+1}/{num_days}: {day_date} | Agent PnL: {pnl:.1f}%")

    # --- Metrics Calculation ---
    total_trades = win_count + loss_count + be_count
    total_return_pct = ((current_capital - initial_capital) / initial_capital) * 100
    
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (loss_count / total_trades * 100) if total_trades > 0 else 0
    be_rate = (be_count / total_trades * 100) if total_trades > 0 else 0
    
    wins = [p for p in all_trades_pnl_pct if p > 0]
    losses = [p for p in all_trades_pnl_pct if p < 0]
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Risk Metrics
    daily_returns_np = np.array(daily_returns)
    mean_daily_ret = np.mean(daily_returns_np)
    std_daily_ret = np.std(daily_returns_np)
    
    # Annualized Sharpe (assuming 252 trading days)
    if std_daily_ret > 1e-9:
        sharpe_ratio = (mean_daily_ret / std_daily_ret) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        
    # Sortino
    downside_returns = daily_returns_np[daily_returns_np < 0]
    std_downside = np.std(downside_returns) if len(downside_returns) > 0 else 0
    if std_downside > 1e-9:
        sortino_ratio = (mean_daily_ret / std_downside) * np.sqrt(252)
    else:
        sortino_ratio = 0.0 if len(downside_returns) > 0 else float('inf') 
        
    # Max Drawdown
    equity_np = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_np)
    drawdowns = (equity_np - peaks) / peaks
    max_drawdown_pct = np.min(drawdowns) * 100
    
    print("\n" + "="*40)
    print("       FAST AGENT PERFORMANCE       ")
    print("="*40)
    print(f"Initial Capital     : ${initial_capital:,.2f}")
    print(f"Final Capital       : ${current_capital:,.2f}")
    print(f"Total Return (%)    : {total_return_pct:.2f}%")
    print("-" * 40)
    print(f"Total Trades        : {total_trades}")
    print(f"Win Rate (%)        : {win_rate:.2f}%")
    print(f"Loss Rate (%)       : {loss_rate:.2f}%")
    print(f"BE Rate (%)         : {be_rate:.2f}%")
    print("-" * 40)
    print(f"Avg Win (%)         : {avg_win:.2f}%")
    print(f"Avg Loss (%)        : {avg_loss:.2f}%")
    print(f"Profit Factor       : {profit_factor:.2f}")
    print("-" * 40)
    print(f"Sharpe Ratio        : {sharpe_ratio:.2f}")
    print(f"Sortino Ratio       : {sortino_ratio:.2f}")
    print(f"Max Drawdown (%)    : {max_drawdown_pct:.2f}%")
    print("="*40)
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label='Fast Agent Equity', color='green')
    plt.title('Equity Curve - Optimized Agent')
    plt.xlabel('Trading Days')
    plt.ylabel('Capital ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('RL_Agent_Final/equity_curve.png')
    print("Equity curve saved to 'RL_Agent_Final/equity_curve.png'")
    
    # Save Heatmap data
    df_res = pd.DataFrame({
        'Date': dates,
        'Daily_Return': daily_returns
    })
    df_res['Date'] = pd.to_datetime(df_res['Date'])
    df_res['Year'] = df_res['Date'].dt.year
    df_res['Month'] = df_res['Date'].dt.month
    
    monthly_ret = df_res.groupby(['Year', 'Month'])['Daily_Return'].apply(lambda x: (np.prod(1 + x) - 1) * 100).reset_index()
    heatmap_data = monthly_ret.pivot(index='Year', columns='Month', values='Daily_Return')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Return (%)'})
    plt.title('Monthly Returns Heatmap (%)')
    plt.savefig('RL_Agent_Final/monthly_heatmap.png')
    print("Monthly heatmap saved to 'RL_Agent_Final/monthly_heatmap.png'")

if __name__ == "__main__":
    evaluate()
