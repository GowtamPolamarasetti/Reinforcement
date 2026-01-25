
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

# Load
CSV_PATH = 'Data/Processed/agent_actions_usdmxn.csv'
df = pd.read_csv(CSV_PATH)

# Agents: ppo, dqn, qrdqn, recurrent, transformer
AGENTS = ['ppo', 'dqn', 'qrdqn', 'recurrent', 'transformer']
outcomes = df['outcome'].values

# Map Actions to Signs: 0 -> -1, 1 -> +1
action_matrix = df[AGENTS].values
signed_matrix = np.where(action_matrix == 1, 1, -1)

# Pre-compute PnL for Win/Loss
# If decision is 1: Win=+0.5, Loss=-0.5, BE=0
# If decision is 0: 0
pnl_vector = np.zeros(len(df))
pnl_vector[np.isin(outcomes, ['WIN'])] = 0.5
pnl_vector[np.isin(outcomes, ['LOSS'])] = -0.5
pnl_vector[np.isin(outcomes, ['BE', 'BREAKEVEN'])] = 0.0

def evaluate_ensemble(params):
    # Params: [w1, w2, w3, w4, w5, threshold]
    weights = np.array(params[:5])
    threshold = params[5]
    
    # Weighted Sum
    # (N_Samples, 5) dot (5,) -> (N_Samples,)
    scores = np.dot(signed_matrix, weights)
    
    # Decision
    decisions = (scores > threshold).astype(int)
    
    # Calculate PnL
    # decisions is 0 or 1. pnl_vector is +0.5/-0.5/0.
    # realized_pnl = decisions * pnl_vector
    realized_pnl = decisions * pnl_vector
    
    # Metrics
    total_trades = np.sum(decisions)
    if total_trades < 100: return 9999 # Penalty for inactivity (Minimize means return large value)
    
    wins = realized_pnl[realized_pnl > 0]
    losses = realized_pnl[realized_pnl < 0]
    
    gross_profit = np.sum(wins)
    gross_loss = np.abs(np.sum(losses))
    
    if gross_loss == 0: pf = 100
    else: pf = gross_profit / gross_loss
    
    # Sharpe (approx using per-trade Returns, assuming constant time/risk which is false but okay for optimization proxy)
    # Better: Cumulative Curve Sharpe? No, too slow.
    # Just usage mean/std of PnL stream (including zeros)
    # Actually, we usually calculate Sharpe on Daily Returns.
    # Here we have trade-by-trade or step-by-step.
    # Valid approximation: Mean/Std of realized_pnl vector (including zeros) * sqrt(TradesPerYear)
    # Let's stick to Profit Factor * log(Trades) to balance Quality vs Quantity
    
    # Objective: Maximize Profit Factor, but ensure decent volume.
    score = pf * np.log10(total_trades)
    
    return -score # Minize negative score

def optimize():
    print("Optimizing Ensemble Weights...")
    
    # Bounds: Weights [0, 10], Threshold [-10, 10]
    bounds = [(0, 5)] * 5 + [(-5, 5)]
    
    result = differential_evolution(
        evaluate_ensemble,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )
    
    best_params = result.x
    print("\noptimization Complete!")
    print("-" * 30)
    print("Best Weights:")
    for agent, w in zip(AGENTS, best_params[:5]):
        print(f"  {agent.upper()}: {w:.4f}")
    print(f"  THRESHOLD: {best_params[5]:.4f}")
    print("-" * 30)
    
    # Run Final Validation
    evaluate_final(best_params)

def evaluate_final(params):
    weights = np.array(params[:5])
    threshold = params[5]
    scores = np.dot(signed_matrix, weights)
    decisions = (scores > threshold).astype(int)
    realized_pnl = decisions * pnl_vector
    
    wins = realized_pnl[realized_pnl > 0]
    losses = realized_pnl[realized_pnl < 0]
    total_trades = np.sum(decisions)
    
    pf = np.sum(wins) / np.abs(np.sum(losses))
    win_rate = len(wins) / total_trades * 100
    
    print(f"Projected Performance:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")

if __name__ == "__main__":
    optimize()
