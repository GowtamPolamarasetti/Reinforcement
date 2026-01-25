
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import os

try:
    from renko_env_fast import RenkoFilterFastEnv
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv

def evaluate_recurrent():
    # Paths
    model_path = "RL_Agent_Final/models_recurrent/final_model_recurrent"
    
    if not os.path.exists(model_path + ".zip"):
         print(f"Recurrent Model not found at {model_path}")
         return
         
    print(f"Loading Recurrent PPO model from {model_path}...")
    model = RecurrentPPO.load(model_path)
    
    # Env (XAUUSD Test Set)
    env = RenkoFilterFastEnv(
        renko_path='Data/Processed/renko_with_predictions.csv',
        states_path='Data/Processed/renko_states.npy',
        reward_config=None,
        mode='test',
        split_ratio=0.8,
        mask_indices=[3, 4, 5, 6] 
    )
    
    num_days = len(env.days)
    print(f"Evaluating Recurrent PPO on {num_days} Test Days...")
    
    initial_cap = 100000.0
    cap = initial_cap
    curve = [initial_cap]
    daily_returns = []
    trades_pnl = []
    dates = []
    
    for i in range(num_days):
        obs, info = env.reset()
        dates.append(env.current_day_val)
        
        terminated = False
        truncated = False
        day_start_cap = cap
        
        # Reset LSTM states for new episode
        lstm_states = None
        # We need to handle episode starts for LSTM masking
        episode_starts = np.ones((1,), dtype=bool)
        
        while not (terminated or truncated):
            # Recurrent Prediction
            # Predict returns: action, lstm_states
            # We must feed lstm_states back
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            
            # Reset episode start flag after first step
            episode_starts = np.zeros((1,), dtype=bool)
            
            # Peek Outcome
            brick = env.renko_df.iloc[env.current_step_idx]
            original_outcome = brick.get('outcome', None)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if action == 1:
                unit_pnl_pct = 0.0
                if original_outcome == 'WIN':
                    unit_pnl_pct = 0.5
                elif original_outcome == 'LOSS':
                    unit_pnl_pct = -0.5
                
                # Execute
                cap += cap * (unit_pnl_pct / 100.0)
                trades_pnl.append(unit_pnl_pct)
        
        # End of Day
        curve.append(cap)
        dr = (cap - day_start_cap) / day_start_cap if day_start_cap > 0 else 0.0
        daily_returns.append(dr)
        
        if (i+1) % 50 == 0:
            print(f"Day {i+1}: Cap=${cap:,.0f}")

    # Metrics
    print("\n" + "="*40)
    print(f"       RECURRENT PPO RESULTS       ")
    print("="*40)
    
    total_ret = ((cap - initial_cap) / initial_cap) * 100
    
    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    
    total_trades = len(trades_pnl)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (len(losses) / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    prof_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Risk
    rets_np = np.array(daily_returns)
    mean_ret = np.mean(rets_np)
    std_ret = np.std(rets_np)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0.0
    
    downside = rets_np[rets_np < 0]
    std_down = np.std(downside) if len(downside) > 0 else 0
    sortino = (mean_ret / std_down) * np.sqrt(252) if std_down > 1e-9 else 0.0
    
    eq_np = np.array(curve)
    peaks = np.maximum.accumulate(eq_np)
    dds = (eq_np - peaks) / peaks
    max_dd = np.min(dds) * 100
    
    print(f"Final Capital     : ${cap:,.2f}")
    print(f"Total Return (%)  : {total_ret:.2f}%")
    print("-" * 40)
    print(f"Total Trades      : {total_trades}")
    print(f"Win Rate (%)      : {win_rate:.2f}%")
    print("-" * 40)
    print(f"Profit Factor     : {prof_factor:.2f}")
    print("-" * 40)
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Sortino Ratio     : {sortino:.2f}")
    print(f"Max Drawdown (%)  : {max_dd:.2f}%")
    print("="*40)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(curve, label='Recurrent PPO', color='magenta')
    plt.title('Recurrent PPO Performance (XAUUSD)')
    plt.xlabel('Days')
    plt.ylabel('Capital')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('RL_Agent_Final/equity_curve_recurrent.png')
    print("Saved plot to RL_Agent_Final/equity_curve_recurrent.png")

if __name__ == "__main__":
    evaluate_recurrent()
