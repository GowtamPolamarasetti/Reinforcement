
# 🏗️ RL_Agent_Final: The Comprehensive Technical Repository

This directory represents the "Production Cluster" of the Renko Trading Reinforcement Learning project. It contains the most optimized agents, the final benchmarking suites, and the **Weighted Ensemble** solution that consistently outperforms all single-agent benchmarks.

---

## 📖 Table of Contents
1.  [Philosophy & Strategy](#philosophy--strategy)
2.  [Atomic Feature Engineering](#atomic-feature-engineering)
3.  [The Reward Logic (Risk Management)](#the-reward-logic)
4.  [Architecture Deep-Dive](#architecture-deep-dive)
5.  [The Weighted Ensemble Pipeline](#the-weighted-ensemble-pipeline)
6.  [Performance Discovery & Benchmarks](#performance-discovery--benchmarks)
7.  [File-by-File Technical Guide](#file-by-file-technical-guide)

---

## 🎯 Philosophy & Strategy

The core strategy focuses on its role as a **Signal Filter**. Rather than predicting price directly, the RL agents look at a candidate trade (a Renko brick formation) and decide: **"Is the current market environment conducive to this trade's success?"**

We evolved from basic PPO agents to a diversified committee of 5 advanced neural networks (PPO, DQN, QR-DQN, LSTM, and Transformer) to ensure robustness across different assets and regimes.

---

## 🧬 Atomic Feature Engineering

The observation space is a **21-dimensional vector**. While the underlying code supports all 21, our final training runs utilized a **Specific Feature Mask** to prioritize high-alpha signals over "Structural Noise."

### **The Full 21-Dim Vector**
| Index | Name | Class | Description |
| :--- | :--- | :--- | :--- |
| **0** | `Regime_1m` | Regime | HMM-based state of the 1-minute OHLC market (Trend vs Chop). |
| **1** | `Regime_Renko` | Regime | Relative trend of the last 10 Renko bricks. |
| **2** | `BiLSTM_Conf` | Regime | Probability of trend continuation from a separate BiLSTM model. |
| **[3-6]** | **Structural** | **MASKED** | **Masked in Final Model**: (Brick Size, Trend Duration, Reversal Status). This was found to be redundant and noisy during our ablation study. |
| **7** | `SMA_Cross` | Indicator | Distance between 20-period and 50-period SMA. |
| **8** | `RSI` | Indicator | Relative Strength Index (Momentum). |
| **9** | `ATR_Ratio` | Indicator | Current Volatility relative to 24h average. |
| **10** | `VWAP_Dist` | Indicator | Price distance from the Volume Weighted Average Price. |
| **11** | `Bollinger_W` | Indicator | Bollinger Band Width (Volatilty Squeeze). |
| **12-14** | `Oscillators` | Indicator | Stochastics and MACD histogram values. |
| **15** | `Daily_PnL` | Dynamic | **Critical**: Current realized profit for the day in percentage points. |
| **16** | `Time_Left` | Dynamic | % of the trading day remaining (0.0 to 1.0). |
| **17** | `Prob_Bin_Win` | Outcome | Probability of WIN (from CatBoost Binary Model). |
| **18** | `Prob_Multi_Win`| Outcome | Probability of WIN (from CatBoost Multi-class Model). |
| **19** | `Prob_Bin_Loss`| Outcome | Probability of LOSS (from CatBoost Binary Model). |
| **20** | `Prob_Multi_Loss`| Outcome | Probability of LOSS (from CatBoost Multi-class Model). |

---

## 💰 The Reward Logic (Risk Management)

The agents are not just reward-seekers; they are **loss-avoiders**. The reward function is heavily weighted toward preserving capital.

### **The Atomic Reward Formula**
*   **Trade Win**: `+8.0` (Standard profit capture).
*   **Trade Loss**: `-7.2` (Asymmetric penalty. Slightly lower than Win to keep the agent active, but the high precision of the agent makes this extremely profitable).
*   **Daily Positive Bonus**: `+67.0` (Massive reward for ending the day in profit). This encourages the agent to "defend its gains" once it is up for the day.
*   **Daily Negative Penalty**: `-27.0` (Penalty for ending the day red).
*   **Daily Drawdown Penalty**: `-21.0` (**The Kill Switch**). If the agent hits the -3.0% daily stop-limit, it is severely penalized. This teaches the agent to stop trading *before* hitting the limit.

---

## 🏗️ Architecture Deep-Dive

We implemented five distinct learning strategies to ensure the ensemble isn't just "averaging the same mistake."

1.  **PPO (Proximal Policy Optimization)**: Our baseline. Good for general trend following.
2.  **DQN (Deep Q-Network)**: Better at identifying discrete "Value" areas. It acts as the anchor for the ensemble.
3.  **QR-DQN (Quantile Regression DQN)**: Instead of predicting a single expected reward, it models the **entire distribution of possible outcomes**. This makes it much more resilient during high-volatility events.
4.  **Recurrent PPO (LSTM)**: Incorporates a Long Short-Term Memory cell. It "remembers" the previous 64 time-steps without needing an explicit window, allowing it to detect complex time-series patterns.
5.  **Transformer PPO**: Uses Multi-Head Self-Attention to look across a fixed context window (10 bricks). It is the most modern architecture, excellent at filtering "noise" by attending only to the most relevant previous bricks.

---

## 🚢 The Weighted Ensemble Pipeline

The final solution is a **committee of specialists**. 

### **The Voting Mechanism**
Every model outputs a "Signed Vote":
*   `Action 1 (TRADE)` -> `+1`
*   `Action 0 (SKIP)` -> `-1`

The Final Decision is calculated as:
$$\text{Score} = \sum (\text{Vote}_i \times \text{Weight}_i)$$

If **Score > 4.21**, the trade is executed.

### **Optimal Weights (Result of Differential Evolution Optimization)**
*   **Standard DQN (1.32)**: The primary decider.
*   **Transformer (1.22)**: The contextual validator.
*   **QR-DQN (1.05)**: The risk manager.
*   **Recurrent (0.60)**: The sequence checker.
*   **PPO (0.37)**: The trend enhancer.

---

## 📈 Performance Discovery & Benchmarks

### **The "Holy Grail" Discovery**
In our final tests on **USD/MXN** (a completely different asset from what the agent was trained on), the **Weighted Ensemble** achieved:
*   **Profit Factor**: **3.56**
*   **Win Rate**: **45.8%**
*   **Max Drawdown**: **-2.97%** (compared to -19% for the baseline strategy).

### **Verification Table (Flat Staking / Simple Interest)**
*Removing the "Infinite Compounding" distortion to prove real alpha.*

| Setup | Trades | Win Rate | Profit Factor | Return (6.5y) |
| :--- | :--- | :--- | :--- | :--- |
| **No AI (Baseline)**| 49,423 | 35.2% | 1.64 | +3,800% |
| **Weighted Ensemble AI**| **23,847** | **45.8%** | **3.56** | **+7,850%** |

**Discovery**: The AI effectively filtered out **25,576 low-probability trades**, doubling the Profit Factor and protecting the account from the deep drawdowns of the raw strategy.

---

## 📂 File-by-File Technical Guide

### **Execution & Benchmarking**
*   `test_weighted_ensemble.py`: The production execution script. Implements the 5-agent weighted voting logic.
*   `test_realistic.py`: The validation script using **Simple Interest** (Fixed $1k bets) to prove the strategy's validity to professional quants.
*   `baseline_test.py`: Calculates the performance of the underlying strategy without any AI filtering.
*   `benchmark_all_usdmxn.py`: Individual performance comparison of all agents on the transfer asset.

### **Optimization & Tools**
*   `optimize_weights.py`: Uses a genetic algorithm (Differential Evolution) to find the optimal weights for the ensemble.
*   `collect_agent_actions.py`: A utility to run all agents and save their actions to a single CSV for rapid offline optimization.

### **Training Scripts**
*   `train_fast.py`: Modern PPO training using vectorized environments.
*   `train_dqn.py`: DQN training.
*   `train_qrdqn.py`: Quantile Regression DQN training.
*   `train_recurrent.py`: LSTM (Recurrent) PPO training.
*   **`train_transformer.py`**: Transformer-based training with custom `TransformerExtractor`.

###  **Support Files**
*   `renko_env_fast.py`: The ultra-optimized environment that loads pre-computed `numpy` states for 100x faster training.
*   `transformer_policy.py`: The PyTorch implementation of the Transformer Feature Extractor.
