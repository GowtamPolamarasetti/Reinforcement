
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

def calculate_advanced_metrics(initial_cap, final_cap, trades_pnl, daily_returns, curve, days_count):
    # 1. Basic
    total_ret = ((final_cap - initial_cap) / initial_cap) * 100
    years = days_count / 365.0
    cagr = ((final_cap / initial_cap) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # 2. Risk
    rets_np = np.array(daily_returns)
    ann_vol = np.std(rets_np) * np.sqrt(252) * 100
    mean_ret = np.mean(rets_np)
    
    sharpe = (mean_ret / np.std(rets_np)) * np.sqrt(252) if np.std(rets_np) > 0 else 0
    
    downside = rets_np[rets_np < 0]
    std_down = np.std(downside) if len(downside) > 0 else 0
    sortino = (mean_ret / std_down) * np.sqrt(252) if std_down > 0 else 0
    
    # DD
    eq_np = np.array(curve)
    peaks = np.maximum.accumulate(eq_np)
    dds = (eq_np - peaks) / peaks
    max_dd = np.min(dds) * 100
    
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    
    # 3. Trade Analysis
    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p < 0]
    
    total_trades = len(trades_pnl)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (len(losses) / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Payoff Ratio
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Profit Factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # Expectancy (Average R per trade)
    # Win% * AvgWin% - Loss% * AvgLoss%
    expectancy_pct = (len(wins)/total_trades * avg_win) + (len(losses)/total_trades * avg_loss)
    
    # SQN: Square Root of N * (Expectancy / StdDev of R-multiples)
    # R-multiples: Trade / Risk. Assuming Risk is constant unit (0.5% in logic)
    # So PnL% can be treated directly for SQN relative calc
    trades_np = np.array(trades_pnl)
    sqn = np.sqrt(total_trades) * (np.mean(trades_np) / np.std(trades_np)) if len(trades_np) > 0 else 0
    
    # Kelly Criterion
    # K = W - (1-W)/R  where W=WinProb, R=Payoff
    W = win_rate / 100.0
    R = payoff
    kelly = W - (1-W)/R if R > 0 else 0
    kelly_pct = kelly * 100
    
    # Formatted Report
    report = []
    report.append("==================================================")
    report.append("           QUANTITATIVE ANALYSIS REPORT           ")
    report.append("             WEIGHTED ENSEMBLE MODEL              ")
    report.append("==================================================")
    report.append(f"Metric Used: USD/MXN Transfer Learning (2015-2024)")
    report.append(f"No. Days: {days_count} | Years: {years:.2f}")
    report.append("-" * 50)
    report.append("RETURN METRICS")
    report.append(f"  Final Capital    : ${final_cap:,.2f}")
    report.append(f"  Total Return     : {total_ret:,.2f}%")
    report.append(f"  CAGR             : {cagr:.2f}%")
    report.append(f"  Ann. Volatility  : {ann_vol:.2f}%")
    report.append("-" * 50)
    report.append("RISK METRICS")
    report.append(f"  Max Drawdown     : {max_dd:.2f}%")
    report.append(f"  Sharpe Ratio     : {sharpe:.2f}")
    report.append(f"  Sortino Ratio    : {sortino:.2f}")
    report.append(f"  Calmar Ratio     : {calmar:.2f}")
    report.append("-" * 50)
    report.append("TRADE ANALYSIS")
    report.append(f"  Total Trades     : {total_trades}")
    report.append(f"  Win Rate         : {win_rate:.2f}%")
    report.append(f"  Loss Rate        : {loss_rate:.2f}%")
    report.append(f"  Profit Factor    : {pf:.2f}")
    report.append(f"  Payoff Ratio     : {payoff:.2f}")
    report.append(f"  Expectancy       : {expectancy_pct:.2f}% per trade")
    report.append(f"  SQN Score        : {sqn:.2f}")
    report.append(f"  Kelly Criterion  : {kelly_pct:.2f}%")
    report.append("==================================================")
    
    return "\n".join(report)

def plot_heatmap(dates, daily_rets):
    df = pd.DataFrame({'date': dates, 'return': daily_rets})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    pivot = df.pivot_table(index='year', columns='month', values='return', aggfunc='sum')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0)
    plt.title('Monthly Returns Heatmap - Weighted Ensemble')
    plt.tight_layout()
    plt.savefig('RL_Agent_Final/heatmap_weighted.png')
    print("Saved heatmap to RL_Agent_Final/heatmap_weighted.png")
    plt.close()

def evaluate_weighted_ensemble():
    print("Loading Models for Weighted Ensemble...")
    
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
    dates = base_env.days
    
    print(f"Evaluating Weighted Ensemble on {num_days} Days...")
    
    initial_cap = 100000.0
    cap = initial_cap
    curve = [initial_cap]
    daily_returns = [] # Store daily raw returns
    trades_pnl = []
    
    obs_std = env_std.reset()
    obs_stack = env_stack.reset()
    
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
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
            
            # 4. PnL Logic
            if final_action == 1:
                unit_pnl = 0.0
                if original_outcome == 'WIN': unit_pnl = 0.5
                elif original_outcome == 'LOSS': unit_pnl = -0.5
                cap += cap * (unit_pnl / 100.0)
                trades_pnl.append(unit_pnl)
                
            if dones_std[0]:
                break
                
        # Day End
        curve.append(cap)
        dr = (cap - day_start_cap) / day_start_cap if day_start_cap > 0 else 0.0
        daily_returns.append(dr)
        
    # Generate Report
    report = calculate_advanced_metrics(initial_cap, cap, trades_pnl, daily_returns, curve, num_days)
    print(report)
    
    with open('RL_Agent_Final/Weighted_Ensemble_Report.txt', 'w') as f:
        f.write(report)
    print("\nSaved detailed report to RL_Agent_Final/Weighted_Ensemble_Report.txt")
    
    # Heatmap
    plot_heatmap(dates, daily_returns)
    
    # Equity Plot (Recalculate to be sure)
    plt.figure(figsize=(10, 6))
    plt.plot(curve, label='Weighted Ensemble', color='gold')
    plt.title(f'Weighted Ensemble (CAGR: {((cap/initial_cap)**(365/num_days)-1)*100:.0f}%)')
    plt.yscale('log')
    plt.legend()
    plt.savefig('RL_Agent_Final/equity_curve_weighted.png')
    print("Saved plot to RL_Agent_Final/equity_curve_weighted.png")

if __name__ == "__main__":
    evaluate_weighted_ensemble()
