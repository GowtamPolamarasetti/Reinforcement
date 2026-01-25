
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt
import os
import seaborn as sns

try:
    from renko_env_fast import RenkoFilterFastEnv
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv

def evaluate_ensemble():
    # Paths
    ppo_path = "RL_Agent_Final/models/final_model"
    dqn_path = "RL_Agent_Final/models_dqn/final_model_dqn"
    
    # Check models
    if not os.path.exists(ppo_path + ".zip"):
         print(f"PPO Model not found at {ppo_path}")
         return
    if not os.path.exists(dqn_path + ".zip"):
         print(f"DQN Model not found at {dqn_path}")
         return
         
    print(f"Loading PPO model from {ppo_path}...")
    ppo_model = PPO.load(ppo_path)
    
    print(f"Loading DQN model from {dqn_path}...")
    dqn_model = DQN.load(dqn_path)
    
    # Init Env
    env = RenkoFilterFastEnv(
        renko_path='Data/Processed/renko_with_predictions.csv',
        states_path='Data/Processed/renko_states.npy',
        reward_config=None,
        mode='test',
        split_ratio=0.8,
        mask_indices=[3, 4, 5, 6] 
    )
    
    num_days = len(env.days)
    print(f"Evaluating Ensemble on {num_days} Test Days...")
    
    # Setup for Two Cases
    
    initial_cap = 100000.0
    
    cap_c1 = initial_cap
    cap_c2 = initial_cap
    
    curve_c1 = [initial_cap]
    curve_c2 = [initial_cap]
    
    # Detailed Tracking
    # Daily Returns
    daily_returns_c1 = []
    daily_returns_c2 = []
    
    # Trade Results (PnL %) for Advanced Metrics
    trades_pnl_c1 = []
    trades_pnl_c2 = []
    
    dates = []
    
    for i in range(num_days):
        obs, info = env.reset()
        day_date = env.current_day_val
        dates.append(day_date)
        
        terminated = False
        truncated = False
        
        day_start_c1 = cap_c1
        day_start_c2 = cap_c2
        
        while not (terminated or truncated):
            # Get Predictions
            action_ppo, _ = ppo_model.predict(obs, deterministic=True)
            action_dqn, _ = dqn_model.predict(obs, deterministic=True)
            
            # Action values are 1 (Take) or 0 (Skip)
            
            # Peek Outcome
            brick = env.renko_df.iloc[env.current_step_idx]
            original_outcome = brick.get('outcome', None)
            
            # Step Env
            obs, reward, terminated, truncated, info = env.step(action_ppo)
            
            # --- Logic ---
            # Base PnL Per Unit Risk (0.5% per trade)
            unit_pnl_pct = 0.0
            
            if original_outcome == 'WIN':
                unit_pnl_pct = 0.5
            elif original_outcome == 'LOSS':
                unit_pnl_pct = -0.5
            
            # Case 1: Additive (Combined)
            # If PPO=1, DQN=1 -> Action=2 -> 2 * unit_pnl
            # If PPO=1, DQN=0 -> Action=1 -> 1 * unit_pnl
            total_action = int(action_ppo + action_dqn)
            
            if total_action > 0:
                pnl_pct_c1_total = total_action * unit_pnl_pct
                # Update Capital
                cap_c1 += cap_c1 * (pnl_pct_c1_total / 100.0)
                # Track Trade (Count Contracts individually as requested)
                # This treats a 2x order as 2 separate trades of 0.5% each
                for _ in range(total_action):
                    trades_pnl_c1.append(unit_pnl_pct)
            
            # Case 2: Agreement (Ensemble)
            # Only if PPO=1 AND DQN=1 -> Action=1
            if action_ppo == 1 and action_dqn == 1:
                pnl_pct_c2 = 1.0 * unit_pnl_pct # Single Order
                cap_c2 += cap_c2 * (pnl_pct_c2 / 100.0)
                # Track Trade
                trades_pnl_c2.append(pnl_pct_c2)
                
        # End of Day
        curve_c1.append(cap_c1)
        curve_c2.append(cap_c2)
        
        # Calculate Daily Return
        dr_c1 = (cap_c1 - day_start_c1) / day_start_c1 if day_start_c1 > 0 else 0.0
        dr_c2 = (cap_c2 - day_start_c2) / day_start_c2 if day_start_c2 > 0 else 0.0
        
        daily_returns_c1.append(dr_c1)
        daily_returns_c2.append(dr_c2)
        
        if (i+1) % 50 == 0:
            print(f"Day {i+1}: C1=${cap_c1:,.0f} | C2=${cap_c2:,.0f}")

    # --- Helper Reporting Function ---
    def print_metrics(name, final_cap, trades_pnl, daily_rets, equity_curve):
        print("\n" + "="*40)
        print(f"       {name} RESULTS       ")
        print("="*40)
        
        total_ret = ((final_cap - initial_cap) / initial_cap) * 100
        
        # Trade Stats
        wins = [p for p in trades_pnl if p > 0]
        losses = [p for p in trades_pnl if p < 0]
        be = [p for p in trades_pnl if p == 0]
        
        total_trades = len(trades_pnl)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        loss_rate = (len(losses) / total_trades * 100) if total_trades > 0 else 0
        be_rate = (len(be) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        prof_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Risk Metrics
        rets_np = np.array(daily_rets)
        mean_ret = np.mean(rets_np)
        std_ret = np.std(rets_np)
        
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0.0
        
        downside = rets_np[rets_np < 0]
        std_down = np.std(downside) if len(downside) > 0 else 0
        sortino = (mean_ret / std_down) * np.sqrt(252) if std_down > 1e-9 else 0.0
        if len(downside) == 0 and mean_ret > 0: sortino = float('inf')
        
        # Drawdown
        eq_np = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq_np)
        dds = (eq_np - peaks) / peaks
        max_dd = np.min(dds) * 100
        
        print(f"Final Capital     : ${final_cap:,.2f}")
        print(f"Total Return (%)  : {total_ret:.2f}%")
        print("-" * 40)
        print(f"Total Trades      : {total_trades}")
        print(f"Win Rate (%)      : {win_rate:.2f}%")
        print(f"Loss Rate (%)     : {loss_rate:.2f}%")
        print(f"BE Rate (%)       : {be_rate:.2f}%")
        print("-" * 40)
        print(f"Avg Win (%)       : {avg_win:.2f}%")
        print(f"Avg Loss (%)      : {avg_loss:.2f}%")
        print(f"Profit Factor     : {prof_factor:.2f}")
        print("-" * 40)
        print(f"Sharpe Ratio      : {sharpe:.2f}")
        print(f"Sortino Ratio     : {sortino:.2f}")
        print(f"Max Drawdown (%)  : {max_dd:.2f}%")
        print("="*40)
        
    # --- Execute Reporting ---
    print_metrics("CASE 1 (Combined/Additive)", cap_c1, trades_pnl_c1, daily_returns_c1, curve_c1)
    print_metrics("CASE 2 (Agreement/Conservative)", cap_c2, trades_pnl_c2, daily_returns_c2, curve_c2)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(curve_c1, label='Case 1: Combined (Px2)', color='purple')
    plt.plot(curve_c2, label='Case 2: Agreement Only', color='orange')
    plt.title('Ensemble Strategies: PPO + DQN')
    plt.xlabel('Trading Days')
    plt.ylabel('Capital ($)')
    plt.yscale('log') # Log scale might be needed for these huge numbers
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('RL_Agent_Final/equity_curve_ensemble.png')
    print("Saved plot to RL_Agent_Final/equity_curve_ensemble.png")

if __name__ == "__main__":
    evaluate_ensemble()
