
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN, RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Imports
try:
    from renko_env_fast import RenkoFilterFastEnv
    from transformer_policy import TransformerExtractor
except ImportError:
    from .renko_env_fast import RenkoFilterFastEnv
    from .transformer_policy import TransformerExtractor

# Constants
RENKO_PATH = 'Data/Processed/renko_with_predictions_USDMXN.csv'
STATES_PATH = 'Data/Processed/renko_states_USDMXN.npy'
OUTPUT_DIR = 'RL_Agent_Final/Benchmark_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_cagr(start_val, end_val, days):
    if days <= 0: return 0.0
    # Annualized based on 365 calendar days assumption for Crypto/Forex continuous markets
    # Or 252 for Stock. Renko bars are continuous. Let's use 365 to be conservative on "Years".
    years = days / 252.0 
    if years < 0.1: return 0.0 # Too short
    try:
        cagr = (end_val / start_val) ** (1 / years) - 1
        return cagr * 100
    except:
        return 0.0

def calculate_drawdown_series(curve):
    eq_np = np.array(curve)
    peaks = np.maximum.accumulate(eq_np)
    dds = (eq_np - peaks) / peaks
    return dds * 100

def plot_combined_curves(results):
    plt.figure(figsize=(14, 10))
    
    # 1. Equity Curves
    plt.subplot(2, 1, 1)
    for name, data in results.items():
        plt.plot(data['curve'], label=f"{name} (Ret: {data['total_ret']:.0f}%)")
    
    plt.title('Combined Equity Curves (USD/MXN)')
    plt.ylabel('Capital (Log Scale)')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 2. Drawdowns
    plt.subplot(2, 1, 2)
    for name, data in results.items():
        plt.plot(data['drawdowns'], label=f"{name} (MaxDD: {data['max_dd']:.2f}%)")
        
    plt.title('Drawdown Profiles')
    plt.ylabel('Drawdown %')
    plt.xlabel('Days')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'combined_equity_and_drawdowns.png')
    plt.savefig(path)
    print(f"Saved Combined Plot to {path}")
    plt.close()

def plot_heatmap(name, dates, daily_rets):
    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'return': daily_rets})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Pivot
    pivot = df.pivot_table(index='year', columns='month', values='return', aggfunc='sum')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0)
    plt.title(f'Monthly Returns Heatmap - {name}')
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, f'heatmap_{name.replace(" ", "_").lower()}.png')
    plt.savefig(path)
    plt.close()

def print_metrics(name, initial_cap, final_cap, trades_pnl, daily_rets, curve, days_count):
    total_ret = ((final_cap - initial_cap) / initial_cap) * 100
    cagr = calculate_cagr(initial_cap, final_cap, days_count)
    
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
    
    rets_np = np.array(daily_rets)
    mean_ret = np.mean(rets_np)
    std_ret = np.std(rets_np)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0.0
    
    downside = rets_np[rets_np < 0]
    std_down = np.std(downside) if len(downside) > 0 else 0
    sortino = (mean_ret / std_down) * np.sqrt(252) if std_down > 1e-9 else 0.0
    
    dds = calculate_drawdown_series(curve)
    max_dd = np.min(dds)
    
    print("\n" + "="*40)
    print(f"       {name} RESULTS       ")
    print("="*40)
    print(f"Initial Capital     : ${initial_cap:,.2f}")
    print(f"Final Capital       : ${final_cap:,.2f}")
    print(f"Total Return (%)    : {total_ret:.2f}%")
    print(f"CAGR (%)            : {cagr:.2f}%")
    print("-" * 40)
    print(f"Total Trades        : {total_trades}")
    print(f"Win Rate (%)        : {win_rate:.2f}%")
    print(f"Loss Rate (%)       : {loss_rate:.2f}%")
    print(f"BE Rate (%)         : {be_rate:.2f}%")
    print("-" * 40)
    print(f"Avg Win (%)         : {avg_win:.2f}%")
    print(f"Avg Loss (%)        : {avg_loss:.2f}%")
    print(f"Profit Factor       : {prof_factor:.2f}")
    print("-" * 40)
    print(f"Sharpe Ratio        : {sharpe:.2f}")
    print(f"Sortino Ratio       : {sortino:.2f}")
    print(f"Max Drawdown (%)    : {max_dd:.2f}%")
    print("="*40)
    
    return {
        'total_ret': total_ret,
        'max_dd': max_dd,
        'curve': curve,
        'drawdowns': dds,
        'days': days_count
    }

def run_test(agent_name, model, env_type='standard'):
    if model is None:
        print(f"Skipping {agent_name} (Model not found)")
        return None

    # Create Env
    def make_env():
        return RenkoFilterFastEnv(
            renko_path=RENKO_PATH,
            states_path=STATES_PATH,
            reward_config=None,
            mode='test',
            split_ratio=0.0,
            mask_indices=[3, 4, 5, 6] 
        )
        
    if env_type == 'stacked':
        env = DummyVecEnv([make_env])
        env = VecFrameStack(env, n_stack=10)
        base_env = env.envs[0]
    else:
        env = DummyVecEnv([make_env])
        base_env = env.envs[0]
        
    # Capture Dates
    dates_list = base_env.days # This is a numpy array of dates
    num_days = len(dates_list)
    print(f"Testing {agent_name} on {num_days} Days...")
    
    initial_cap = 100000.0
    cap = initial_cap
    curve = [initial_cap]
    daily_returns = []
    trades_pnl = []
    
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    # Need to align curve with dates for heatmap
    # Curve len = num_days + 1 (initial). Returns len = num_days.
    
    for i in range(num_days):
        day_start_cap = cap
        
        while True:
            if agent_name == 'Recurrent PPO':
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            brick = base_env.renko_df.iloc[base_env.current_step_idx]
            original_outcome = brick.get('outcome', None)
            
            obs, rewards, dones, infos = env.step(action)
            episode_starts = dones
            
            if action[0] == 1:
                unit_pnl = 0.0
                if original_outcome == 'WIN': unit_pnl = 0.5
                elif original_outcome == 'LOSS': unit_pnl = -0.5
                cap += cap * (unit_pnl / 100.0)
                trades_pnl.append(unit_pnl)
                
            if dones[0]:
                break
        
        curve.append(cap)
        dr = (cap - day_start_cap) / day_start_cap if day_start_cap > 0 else 0.0
        daily_returns.append(dr)
        
    metrics = print_metrics(agent_name.upper(), initial_cap, cap, trades_pnl, daily_returns, curve, num_days)
    
    # Generate Heatmap immediately
    plot_heatmap(agent_name, dates_list, daily_returns)
    
    return metrics

def main():
    if not os.path.exists(RENKO_PATH):
        print("USDMXN Data missing. Run process_usdmxn.py first.")
        return

    results = {}

    # 1. PPO
    try:
        path = "RL_Agent_Final/models/final_model"
        if os.path.exists(path + ".zip"):
            model = PPO.load(path)
            res = run_test("Standard PPO", model, 'standard')
            if res: results['Standard PPO'] = res
    except Exception as e: print(f"PPO Failed: {e}")

    # 2. DQN
    try:
        path = "RL_Agent_Final/models_dqn/final_model_dqn"
        if os.path.exists(path + ".zip"):
            model = DQN.load(path)
            res = run_test("Standard DQN", model, 'standard')
            if res: results['Standard DQN'] = res
    except Exception as e: print(f"DQN Failed: {e}")
            
    # 3. QR-DQN
    try:
        path = "RL_Agent_Final/models_qrdqn/final_model_qrdqn"
        if os.path.exists(path + ".zip"):
            model = QRDQN.load(path)
            res = run_test("QR-DQN", model, 'standard')
            if res: results['QR-DQN'] = res
    except Exception as e: print(f"QR-DQN Failed: {e}")
            
    # 4. Recurrent PPO
    try:
        path = "RL_Agent_Final/models_recurrent/final_model_recurrent"
        if os.path.exists(path + ".zip"):
            model = RecurrentPPO.load(path)
            res = run_test("Recurrent PPO", model, 'standard')
            if res: results['Recurrent PPO'] = res
    except Exception as e: print(f"Recurrent PPO Failed: {e}")
            
    # 5. Transformer PPO
    try:
        path = "RL_Agent_Final/models_transformer/final_model_transformer"
        if os.path.exists(path + ".zip"):
            model = PPO.load(path, custom_objects={'features_extractor_class': TransformerExtractor})
            res = run_test("Transformer PPO", model, 'stacked')
            if res: results['Transformer PPO'] = res
    except Exception as e: print(f"Transformer PPO Failed: {e}")
    
    # Final Plots
    if len(results) > 0:
        plot_combined_curves(results)

if __name__ == "__main__":
    main()
