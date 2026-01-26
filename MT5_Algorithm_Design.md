# 🏛️ MT5-Based Weighted Ensemble Trading Algorithm: Technical Design

## 1. System Architecture Overview

The system operates as a **Python-based Execution Engine** connected to the **MetaTrader 5 (MT5)** terminal. It uses a **Gap-Less Tick Batching** approach for live execution to ensure no price movement is missed, guaranteeing precise Renko construction and risk management.

### High-Level Components
1.  **Daily Optimizer (Calibration Engine)**: Runs once per session to calculate the optimal **Brick Size** and **Grid Offset** using the last 5 days of M1 data.
2.  **Tick Batcher (Gap-Less Bridge)**: Polls `mt5.copy_ticks_from` to retrieve *all* ticks that occurred since the last cycle, ensuring zero data loss even if the loop lags.
3.  **Renko Builder (Tick-Perfect)**: A high-precision engine that consumes the tick buffer sequentially to reconstruct the exact path of price.
4.  **Feature Engineer**: Calculates 21 atomic features on the fly whenever a *committed* Renko brick is formed.
5.  **Ensemble Core**: Hosts the 5 RL Agents + CatBoost Models for inference.
6.  **Orbit Execution (Micro-Manager)**: Manages trade lifecycles, processing price updates tick-by-tick for **Intra-Brick Break-Even** triggers.

---

## 2. Session Initialization & Calibration (Daily)

Before the trading loop begins (e.g., at 00:00 server time), the system performs a rigid calibration routine.

### Step 2.1: Data Warmup & History Fetch
*   **Fetch M1 History**: Request last 5-6 days of M1 OHLC data (`mt5.copy_rates_from_pos`).
*   **Purpose**: Used for the Daily Optimization routine and to seed indicators.

### Step 2.2: The Optimization Routine (Brute-Force Grid Search)
*   **Objective**: Find the `Grid Offset` (price anchor) that maximized trend efficiency over the last 5 days.
*   **Inputs**: T-5 Days M1 Data.
*   **Brick Size Calculation**: `(Daily Open * 0.00236) / 2`.
*   **Search Space**: Scan price offsets between `[Min_Price, Max_Price]` of the anchor window.
*   **Scoring**: Reconstruct Renko history for each offset candidate.
    *   Score = Sum(PnL of simulated standard trades).
*   **Output**: `Optimal_Offset` and `Brick_Size` for the current trading session.

### Step 2.3: Model Loading
*   Load RL Agents and CatBoost/HMM models.
*   **Sync State**: Initialize `last_processed_time_ms` to `mt5.symbol_info_tick(SYMBOL).time_msc`.

---

## 3. The Live Trading Loop (Batch-Pulse)

The system uses a **Batch Polling** pattern. This is superior to simple tick polling because it survives network/CPU latency spikes.

### Step 3.1: Gap-Less Data Ingestion
*   **State**: Maintain `last_processed_time_ms` (in milliseconds).
*   **Fetch**: `new_ticks = mt5.copy_ticks_from(SYMBOL, last_processed_time_ms, count=1000, flags=mt5.COPY_TICKS_ALL)`
    *   *Why `copy_ticks_from`?* It provides the sequence of *all* ticks since timestamp `T`.
    *   *Why not `symbol_info_tick`?* That only gives the *current* state. If price jumps from 100 -> 102 -> 101 in 100ms, and our loop takes 200ms, we would miss the 102 spike which could have triggered a Stop Loss or Renko Brick. Batching captures the 102.
*   **Filter**: Remove ticks with `time_msc <= last_processed_time_ms` (overlap protection).
*   **Update**: Set `last_processed_time_ms = new_ticks[-1].time_msc`.

### Step 3.2: Sequential Tick Replay
*   **Loop**: Iterate through `new_ticks` array one by one.
*   **Pass**: Send each tick `(Bid, Ask)` to the **Renko Builder**.
    *   This effectively "replays" the market action at high speed inside the engine.

### Step 3.3: Renko Construction Logic
*   **Virtual Grid**: Offset + N * Size.
*   **Traversal Check**: 
    *   Does this individual tick breach `Next_Brick_Top` or `Next_Brick_Bottom`?
    *   **Yes**: Form Brick -> Trigger Feature Engineer -> Trigger Agent Inference.
*   **Break-Even Monitor**:
    *   Inside the phantom brick, check: `Tick_Price >= Entry + (Brick_Size * 0.3125)`.
    *   **Action**: If true, add "Modify SL to BE" to the *Order_Queue* immediately.

### Step 3.4: Signal Generation & Execution
*   If a brick is completed during the replay:
    1.  Update `Renko_History`.
    2.  Run HMM/Indicators/CatBoost.
    3.  Run RL Ensemble (Weighted Vote).
    4.  If `Score > 4.21` -> **Output Signal**.
    5.  **Execute Immediately**: Do not wait for the batch to finish. If the 5th tick in a 20-tick batch triggers a trade, execute it before processing the 6th tick.

---

## 4. Real-Time Feature Engineering Pipeline

Triggered only on **New Brick Formation**.

### Step 4.1: Market Regime (HMM)
*   Update return series. Predict Hidden State (0=Chop, 1=Trend, 2=Vol).

### Step 4.2: Technical Indicators
*   Update RSI, ATR, MACD using the committed Renko History.

### Step 4.3: ML Outcomes (CatBoost)
*   Predict `Prob_Win`, `Prob_Loss`.

### Step 4.4: Context
*   Update `Daily_PnL` using `mt5.account_info().equity`.

---

## 5. The Ensemble Decision Engine

*   **Inputs**: 21-dim vector.
*   **Process**: Query PPO, DQN, QR-DQN, LSTM, Transformer.
*   **Output**: Weighted Vote Score.

---

## 6. Orbit Execution System (High-Fidelity)

### Step 6.1: Signal Execution
*   **Direction**: Based on Renko Trend & Agent Vote.
*   **Execution**: Market Order via `mt5.order_send`.

### Step 6.2: Tick-Level Risk Management
*   **Hard Stop (SL)**: 2 Bricks away.
*   **Trailing Stop**: 1 Brick trailing.
*   **Intra-Brick Break-Even**:
    *   This is the primary reason for the **Batch Tick** architecture.
    *   Even if the market spikes and reverts within 500ms, the *Tick Batcher* will see the spike in the history array, trigger the virtual condition, and update the SL logic (or recognize that the SL *would have* moved).
    *   *Reality Check*: If the spike was too fast for execution latency, we accept slippage, but the Logic State remains correct.

### Step 6.3: Portfolio Protection
*   **Kill Switch**: -3% Daily Drawdown (Equity-based).

---

## 7. Safety & Recovery

*   **State Persistence**: Save `last_processed_time_ms`, `Grid_Offset`, `Renko_History` to `daily_state.json`.
*   **Crash Recovery**:
    *   Load state.
    *   Call `mt5.copy_ticks_from(SYMBOL, loaded_last_time_ms, ...)` to replay everything missed during downtime.
    *   The system naturally catches up to the present millisecond.

## 8. Summary of Data Flow

```mermaid
graph TD
    A[MT5 History (M1)] -->|00:00 Daily| B(Daily Optimizer)
    B -->|Offset & Size| C[Renko Builder State]
    D[MT5 Live Ticks] -->|Batch Request| E(Tick Batcher)
    E -->|Sequential Replay| C
    C -->|Tick Check| F{Intra-Brick BE?}
    F -->|Yes| G[Modify SL -> BE]
    C -->|Brick Completed?| H[Feature Engineer]
    H --> I[Ensemble Inference]
    I -->|Score > 4.21| J[Orbit Execution]
    J -->|Send Order| K[MT5 Terminal]
```
