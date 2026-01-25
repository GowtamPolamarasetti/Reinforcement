
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt
import os

try:
    from renko_env_fast import RenkoFilterFastEnv
    from transformer_policy import TransformerExtractor
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv
    from .transformer_policy import TransformerExtractor

def evaluate_transformer():
    # Paths
    model_path = "RL_Agent_Final/models_transformer/final_model_transformer"
    
    if not os.path.exists(model_path + ".zip"):
         print(f"Model not found at {model_path}")
         return
         
    # Env (XAUUSD Test Set)
    # We must wrap in DummyVecEnv then VecFrameStack
    def make_test_env():
        return RenkoFilterFastEnv(
            renko_path='Data/Processed/renko_with_predictions.csv',
            states_path='Data/Processed/renko_states.npy',
            reward_config=None,
            mode='test',
            split_ratio=0.8,
            mask_indices=[3, 4, 5, 6] 
        )
    
    # 1. Create Vec Env
    env = DummyVecEnv([make_test_env])
    
    # 2. Stack Frames (Must match training!)
    N_STACK = 10
    env = VecFrameStack(env, n_stack=N_STACK)
    
    print(f"Loading Transformer PPO model from {model_path}...")
    # We must pass custom objects if the class isn't in scope (it is here, but good practice)
    model = PPO.load(model_path, custom_objects={'features_extractor_class': TransformerExtractor})
    
    # Access internal env to get metrics
    base_env = env.envs[0]
    num_days = len(base_env.days)
    print(f"Evaluating Transformer PPO on {num_days} Test Days...")
    
    initial_cap = 100000.0
    cap = initial_cap
    curve = [initial_cap]
    daily_returns = []
    trades_pnl = []
    
    # Reset
    obs = env.reset() # (1, N_Stack*F)
    
    for i in range(num_days):
        day_start_cap = cap
        terminated = False
        truncated = False
        
        # We need to track when the day ends.
        # VecEnv step() automatically resets if done=True.
        # So we check 'dones' array.
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            
            # Peek Outcome from base env
            brick = base_env.renko_df.iloc[base_env.current_step_idx]
            original_outcome = brick.get('outcome', None)
            
            # Step
            obs, rewards, dones, infos = env.step(action)
            
            if action[0] == 1:
                unit_pnl = 0.0
                if original_outcome == 'WIN': unit_pnl = 0.5
                elif original_outcome == 'LOSS': unit_pnl = -0.5
                
                cap += cap * (unit_pnl / 100.0)
                trades_pnl.append(unit_pnl)
            
            # Check done
            if dones[0]:
                break
                
        # End of Day
        curve.append(cap)
        dr = (cap - day_start_cap) / day_start_cap if day_start_cap > 0 else 0.0
        daily_returns.append(dr)
        
        if (i+1) % 50 == 0:
            print(f"Day {i+1}: Cap=${cap:,.0f}")

    # Metrics
    print("\n" + "="*40)
    print(f"       TRANSFORMER PPO RESULTS       ")
    print("="*40)
    
    total_ret = ((cap - initial_cap) / initial_cap) * 100
    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    
    total_trades = len(trades_pnl)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    prof_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
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
    print(f"Total Trades      : {total_trades}")
    print(f"Win Rate (%)      : {win_rate:.2f}%")
    print(f"Profit Factor     : {prof_factor:.2f}")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Sortino Ratio     : {sortino:.2f}")
    print(f"Max Drawdown (%)  : {max_dd:.2f}%")
    print("="*40)
    
    plt.figure(figsize=(12, 6))
    plt.plot(curve, label='Transformer PPO', color='brown')
    plt.title('Transformer PPO Performance (XAUUSD)')
    plt.xlabel('Days')
    plt.ylabel('Capital')
    plt.yscale('log')
    plt.legend()
    plt.savefig('RL_Agent_Final/equity_curve_transformer.png')
    print("Saved plot to RL_Agent_Final/equity_curve_transformer.png")

if __name__ == "__main__":
    evaluate_transformer()
