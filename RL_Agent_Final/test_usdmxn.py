
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

def evaluate_usdmxn():
    # Paths (XAUUSD Trained Models)
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
    
    # Data Paths for USDMXN
    renko_path = 'Data/Processed/renko_with_predictions_USDMXN.csv'
    states_path = 'Data/Processed/renko_states_USDMXN.npy'
    
    if not os.path.exists(renko_path) or not os.path.exists(states_path):
        print("Processed USDMXN data not found. Please run process_usdmxn.py first.")
        return

    # Init Env with USDMXN Data
    # Note: We use all data for test (no split needed really, or just set split 0 if env supports)
    # FastEnv 'test' mode usually takes the last split.
    # To test on FULL Dataset, we can trick it by setting split_ratio very low or modifying env.
    # Actually, let's just use mode='test' with split_ratio=0.0 (if checks allow) 
    # or better, just use the standard init and accept it tests on the last 20%.
    # Wait, the user likely wants to test on the *whole* new file.
    # The FastEnv splits based on `split_ratio` inside `_load_data`.
    # To use ALL data: set split_ratio=0.0 and mode='test' -> No, mode='test' takes `[train_size:]`. 
    # If split=0, train_size=0, test=all.
    
    env = RenkoFilterFastEnv(
        renko_path=renko_path,
        states_path=states_path,
        reward_config=None,
        mode='test',
        split_ratio=0.0, # This forces the 'test' set to be the entire dataset
        mask_indices=[3, 4, 5, 6] 
    )
    
    num_days = len(env.days)
    print(f"Evaluating USD/MXN on {num_days} Days (Full Dataset)...")
    
    initial_cap = 100000.0
    
    cap_c1 = initial_cap
    cap_c2 = initial_cap
    
    curve_c1 = [initial_cap]
    curve_c2 = [initial_cap]
    
    daily_returns_c1 = []
    daily_returns_c2 = []
    
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
            
            # Peek Outcome
            brick = env.renko_df.iloc[env.current_step_idx]
            original_outcome = brick.get('outcome', None)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action_ppo)
            
            # PnL Logic
            unit_pnl_pct = 0.0
            if original_outcome == 'WIN':
                unit_pnl_pct = 0.5
            elif original_outcome == 'LOSS':
                unit_pnl_pct = -0.5
            
            # Case 1: Additive (Treating as multiple contracts)
            total_action = int(action_ppo + action_dqn)
            
            if total_action > 0:
                pnl_pct_c1_total = total_action * unit_pnl_pct
                cap_c1 += cap_c1 * (pnl_pct_c1_total / 100.0)
                # Track INDIVIDUAL contracts
                for _ in range(total_action):
                    trades_pnl_c1.append(unit_pnl_pct)
            
            # Case 2: Agreement (Single Trade if Consensus)
            if action_ppo == 1 and action_dqn == 1:
                pnl_pct_c2 = 1.0 * unit_pnl_pct
                cap_c2 += cap_c2 * (pnl_pct_c2 / 100.0)
                trades_pnl_c2.append(pnl_pct_c2)
                
        # End of Day
        curve_c1.append(cap_c1)
        curve_c2.append(cap_c2)
        
        dr_c1 = (cap_c1 - day_start_c1) / day_start_c1 if day_start_c1 > 0 else 0.0
        dr_c2 = (cap_c2 - day_start_c2) / day_start_c2 if day_start_c2 > 0 else 0.0
        
        daily_returns_c1.append(dr_c1)
        daily_returns_c2.append(dr_c2)
        
        if (i+1) % 50 == 0:
            print(f"Day {i+1}: C1=${cap_c1:,.0f} | C2=${cap_c2:,.0f}")

    # Metrics Helper
    def print_metrics(name, final_cap, trades_pnl, daily_rets, equity_curve):
        print("\n" + "="*40)
        print(f"       {name} RESULTS       ")
        print("="*40)
        
        total_ret = ((final_cap - initial_cap) / initial_cap) * 100
        
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
        
        # Risk
        rets_np = np.array(daily_rets)
        mean_ret = np.mean(rets_np)
        std_ret = np.std(rets_np)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0.0
        
        downside = rets_np[rets_np < 0]
        std_down = np.std(downside) if len(downside) > 0 else 0
        sortino = (mean_ret / std_down) * np.sqrt(252) if std_down > 1e-9 else 0.0
        if len(downside) == 0 and mean_ret > 0: sortino = float('inf')
        
        # DD
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
        print("-" * 40)
        print(f"Avg Win (%)       : {avg_win:.2f}%")
        print(f"Avg Loss (%)      : {avg_loss:.2f}%")
        print(f"Profit Factor     : {prof_factor:.2f}")
        print("-" * 40)
        print(f"Sharpe Ratio      : {sharpe:.2f}")
        print(f"Sortino Ratio     : {sortino:.2f}")
        print(f"Max Drawdown (%)  : {max_dd:.2f}%")
        print("="*40)
        
    print_metrics("USDMXN Case 1 (Combined)", cap_c1, trades_pnl_c1, daily_returns_c1, curve_c1)
    print_metrics("USDMXN Case 2 (Agreement)", cap_c2, trades_pnl_c2, daily_returns_c2, curve_c2)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(curve_c1, label='Case 1: Combined', color='purple')
    plt.plot(curve_c2, label='Case 2: Agreement', color='orange')
    plt.title('USD/MXN Performance (Transfer Learning)')
    plt.xlabel('Days')
    plt.ylabel('Capital')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('RL_Agent_Final/equity_curve_usdmxn.png')
    print("Saved plot to RL_Agent_Final/equity_curve_usdmxn.png")

if __name__ == "__main__":
    evaluate_usdmxn()
