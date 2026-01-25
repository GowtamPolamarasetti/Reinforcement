
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

try:
    from renko_env_fast import RenkoFilterFastEnv
    from transformer_policy import TransformerExtractor
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv
    from .transformer_policy import TransformerExtractor

# Constants
RENKO_PATH = 'Data/Processed/renko_with_predictions_USDMXN.csv'
STATES_PATH = 'Data/Processed/renko_states_USDMXN.npy'

# Optimized Weights
WEIGHTS = {
    'ppo': 0.3735,
    'dqn': 1.3229,
    'qrdqn': 1.0548,
    'recurrent': 0.6007,
    'transformer': 1.2186
}
THRESHOLD = 4.2094

# --- REALISTIC SETTINGS ---
INITIAL_CAPITAL = 100000.0
RISK_PER_TRADE_PCT = 1.0 # Risk 1% of account per trade
STOP_LOSS_PCT = 0.5      # Renko brick size is roughly 0.5% move? Need to verify.
                         # If Win is 0.5% and Loss is -0.5%, then 1 Brick = 0.5% price move.
                         # Position Size = (Risk / Stop) * Cap = (1% / 0.5%) = 2x Leverage.
                         # This is standard conservative FX sizing.

def make_env(stacked=False):
    def _init():
        return RenkoFilterFastEnv(
            renko_path=RENKO_PATH,
            states_path=STATES_PATH,
            reward_config=None,
            mode='test',
            split_ratio=0.0,
            mask_indices=[3, 4, 5, 6] 
        )
    if stacked:
        env = DummyVecEnv([_init])
        env = VecFrameStack(env, n_stack=10)
        return env
    else:
        env = DummyVecEnv([_init])
        return env

def evaluate_realistic():
    print("Loading Models for Realistic Validation...")
    
    models = {}
    try: models['ppo'] = PPO.load("RL_Agent_Final/models/final_model")
    except: pass
    try: models['dqn'] = DQN.load("RL_Agent_Final/models_dqn/final_model_dqn")
    except: pass
    try: models['qrdqn'] = QRDQN.load("RL_Agent_Final/models_qrdqn/final_model_qrdqn")
    except: pass
    try: models['recurrent'] = RecurrentPPO.load("RL_Agent_Final/models_recurrent/final_model_recurrent")
    except: pass
    try: models['transformer'] = PPO.load("RL_Agent_Final/models_transformer/final_model_transformer", custom_objects={'features_extractor_class': TransformerExtractor})
    except: pass
    
    env_std = make_env(stacked=False)
    env_stack = make_env(stacked=True)
    
    base_env = env_std.envs[0]
    num_days = len(base_env.days)
    
    print(f"Running Realistic Backtest on {num_days} Days...")
    print(f"Risk Per Trade: {RISK_PER_TRADE_PCT}% | Stop Loss Estimate: {STOP_LOSS_PCT}%")
    print(f"Implied Leverage: {RISK_PER_TRADE_PCT/STOP_LOSS_PCT:.1f}x")
    
    cap = INITIAL_CAPITAL
    curve = [cap]
    daily_returns = []
    trades_pnl_dollars = []
    
    obs_std = env_std.reset()
    obs_stack = env_stack.reset()
    
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    # Reset loop indices
    base_env.current_day_idx = 0
    obs_std = env_std.reset()
    base_env_stack = env_stack.envs[0]
    base_env_stack.current_day_idx = 0
    obs_stack = env_stack.reset()

    for i in range(num_days):
        day_start_cap = cap
        
        while True:
            # 1. Gather Votes
            score = 0.0
            if 'ppo' in models:
                act, _ = models['ppo'].predict(obs_std, deterministic=True)
                score += (1 if act==1 else -1) * WEIGHTS['ppo']
            if 'dqn' in models:
                act, _ = models['dqn'].predict(obs_std, deterministic=True)
                score += (1 if act==1 else -1) * WEIGHTS['dqn']
            if 'qrdqn' in models:
                act, _ = models['qrdqn'].predict(obs_std, deterministic=True)
                score += (1 if act==1 else -1) * WEIGHTS['qrdqn']
            if 'recurrent' in models:
                act, lstm_states = models['recurrent'].predict(obs_std, state=lstm_states, episode_start=episode_starts, deterministic=True)
                score += (1 if act==1 else -1) * WEIGHTS['recurrent']
            if 'transformer' in models:
                act, _ = models['transformer'].predict(obs_stack, deterministic=True)
                score += (1 if act==1 else -1) * WEIGHTS['transformer']
                
            # 2. Decision
            final_action = 1 if score > THRESHOLD else 0
            
            # 3. Step
            brick = base_env.renko_df.iloc[base_env.current_step_idx]
            original_outcome = brick.get('outcome', 'UNKNOWN')
            
            obs_std, _, dones_std, _ = env_std.step(np.array([final_action]))
            obs_stack, _, dones_stack, _ = env_stack.step(np.array([final_action]))
            episode_starts = dones_std
            
            # 4. Realistic PnL Logic (SIMPLE INTEREST / FLAT STAKE)
            if final_action == 1:
                # Flat Stake: Always risk based on INITIAL capital, never compound
                risk_amount = INITIAL_CAPITAL * (RISK_PER_TRADE_PCT / 100.0)
                
                trade_pnl_dollars = 0.0
                if original_outcome == 'WIN':
                    trade_pnl_dollars = risk_amount
                elif original_outcome == 'LOSS':
                    trade_pnl_dollars = -risk_amount
                elif original_outcome == 'BE' or original_outcome == 'BREAKEVEN':
                    trade_pnl_dollars = 0.0
                
                # Update Capital
                cap += trade_pnl_dollars
                trades_pnl_dollars.append(trade_pnl_dollars)
                
            if dones_std[0]:
                break
        
        curve.append(cap)
        dr = (cap - day_start_cap) / day_start_cap if day_start_cap > 0 else 0.0
        daily_returns.append(dr)
    
    # --- REPORT GENERATION ---
    total_ret_pct = ((cap - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    years = num_days / 365.0
    cagr = ((cap / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    
    # DD
    eq_np = np.array(curve)
    peaks = np.maximum.accumulate(eq_np)
    dds = (eq_np - peaks) / peaks
    max_dd_pct = np.min(dds) * 100
    
    # Sharpe
    rets_np = np.array(daily_returns)
    sharpe = (np.mean(rets_np) / np.std(rets_np)) * np.sqrt(252) if np.std(rets_np) > 0 else 0
    
    # Trade Stats
    wins = [p for p in trades_pnl_dollars if p > 0]
    losses = [p for p in trades_pnl_dollars if p < 0]
    pf = sum(wins)/abs(sum(losses)) if losses else 0
    
    print("\n========================================")
    print("      REALISTIC BACKTEST REPORT         ")
    print("========================================")
    print(f"Risk Model       : Fixed {RISK_PER_TRADE_PCT}% Risk per Trade")
    print(f"Implied Leverage : ~{RISK_PER_TRADE_PCT/STOP_LOSS_PCT:.1f}x")
    print("-" * 40)
    print(f"Initial Capital  : ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital    : ${cap:,.2f}")
    print(f"Total Return     : {total_ret_pct:,.2f}%")
    print(f"CAGR             : {cagr:.2f}%")
    print("-" * 40)
    print(f"Max Drawdown     : {max_dd_pct:.2f}%")
    print(f"Sharpe Ratio     : {sharpe:.2f}")
    print(f"Profit Factor    : {pf:.2f}")
    print("========================================")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(curve, label=f'Realistic Equity (2% Risk)', color='green')
    plt.title('Realistic Strategy Performance (No Infinite Leverage)')
    plt.ylabel('Capital')
    plt.xlabel('Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('RL_Agent_Final/equity_curve_realistic.png')
    print("Saved plot to RL_Agent_Final/equity_curve_realistic.png")

if __name__ == "__main__":
    evaluate_realistic()
