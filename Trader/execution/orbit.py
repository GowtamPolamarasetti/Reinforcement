import time
import numpy as np
from collections import deque
from datetime import datetime
import pandas as pd
import os
from config.settings import DEFAULT_BRICK_SIZE, DEFAULT_OFFSET, SYMBOL
from data.connector import MT5Connector
from data.tick_buffer import TickStream
from data.renko import RenkoBuilder
from data.features import FeatureEngineer
from models.ensemble import EnsembleAgent
from models.predictors import OutcomePredictor
from execution.orders import OrderExecutor
from execution.risk import RiskManager
from utils.logger import logger
from utils.state import StateManager

class OrbitEngine:
    def __init__(self):
        # Components
        self.connector = MT5Connector()
        self.state = StateManager()
        self.clock = None # Initialized in start() after connection
        
        # Load Optimization params from state or default
        saved_brick = self.state.get("optimization", {}).get("brick_size", 0.0)
        self.brick_size = saved_brick 
        self.offset = self.state.get("optimization", {}).get("grid_offset", DEFAULT_OFFSET)
        
        self.renko = None 
        
        self.features = FeatureEngineer()
        
        self.ensemble = EnsembleAgent()
        self.ensemble.load_all()
        
        self.predictors = OutcomePredictor()
        self.predictors.load()
        
        self.orders = OrderExecutor()
        self.risk = RiskManager(self.state)
        
        # Runtime State
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        
        # Transformer Buffer (Stack of 10 Observations)
        self.obs_stack = deque(maxlen=10)
        
        # M1 Data Buffer (for Indicators)
        # We accumulate ticks and resample to M1
        self.tick_accumulator = []
        self.m1_buffer = pd.DataFrame() # Holds recent M1 bars
        
    def start(self):
        if not self.connector.connect():
            return
            
        # Initialize TickStream NOW, when MT5 is connected
        self.clock = TickStream()
            
        # Optimization & Warmup Logic
        import MetaTrader5 as mt5
        from data.renko_optimizer import RenkoOptimizer
        
        logger.info("Fetching M1 History for Optimization (7 Days)...")
        days_back = 7
        
        ticks = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1440 * days_back)
        
        if ticks is None or len(ticks) == 0:
            logger.warning("History Fetch Failed. Using default Fallback.")
            start_price = mt5.symbol_info_tick(SYMBOL).ask
            self.brick_size = start_price * 0.00118
            self.renko = RenkoBuilder(self.brick_size, start_price, 0.0)
        else:
            # Convert to DF
            history_df = pd.DataFrame(ticks)
            history_df['date'] = pd.to_datetime(history_df['time'], unit='s')
            # Ensure proper columns
            history_df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            optimizer = RenkoOptimizer()
            best_bs, best_offset, _ = optimizer.optimize(history_df)
            
            self.brick_size = best_bs
            self.offset = best_offset
            
            # Initialize M1 Buffer with recent history (last 500 bars for indicators)
            self.m1_buffer = history_df.tail(1000).copy()
            self.m1_buffer.set_index('date', inplace=True)
            
            # Initialize Renko State
            # We must replay history to align the Renko Sequence and Trend
            logger.info(f"Replaying history to sync Renko state (Size: {best_bs:.5f})...")
            
            # Start from the anchor (best_offset usually aligns with a min/max)
            # Simplification: Start Renko at first close of replay window
            start_row = self.m1_buffer.iloc[0]
            self.renko = RenkoBuilder(self.brick_size, start_row['close'], 0.0)
            
            # Replay M1 bars as virtual ticks
            for idx, row in self.m1_buffer.iterrows():
                ts_ms = int(row.name.timestamp() * 1000)
                # Feed Open, High, Low, Close traversal to capture intra-bar bricks?
                # RenkoBuilder.update_tick handles single price.
                # Approx: Open -> High -> Low -> Close (if Bullish bar) could be better
                # But simple Close update is standard for fast warmup
                self.renko.update_tick(row['close'], ts_ms)
                
            # Pre-fill Transformer Stack (Warmup with dummy or real obs)
            # ideally we run process_signal logic during replay but without executing order.
            # For simplicity: Pad with zeros first
            dummy_obs = np.zeros(21, dtype=np.float32)
            for _ in range(10):
                self.obs_stack.append(dummy_obs)
                
            # Save Warmup Renko Snapshot
            self._save_renko_snapshot()
            
        logger.info(f"Orbit Started. Brick: {self.brick_size:.4f}")
        
    def _save_renko_snapshot(self):
        """
        Saves the current Renko history (after warmup) to a CSV file.
        """
        try:
            if not self.renko or not self.renko.history:
                logger.warning("No Renko history to save.")
                return

            # Directory
            save_dir = "renkos"
            os.makedirs(save_dir, exist_ok=True)
            
            # Filename based on current time
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"renko_{timestamp_str}.csv"
            filepath = os.path.join(save_dir, filename)
            
            # Convert to DataFrame
            # Renko history is list of NewBrickEvent namedtuples
            df = pd.DataFrame(self.renko.history)
            
            # Add readable date
            if 'timestamp' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save
            df.to_csv(filepath, index=False)
            logger.info(f"Saved Warmup Renko Snapshot to: {filepath} ({len(df)} bricks)")
            
        except Exception as e:
            logger.error(f"Failed to save Renko snapshot: {e}")
        
    def pulse(self):
        """
        Single heartbeat of the loop.
        """
        # 1. Risk Check
        if not self.risk.check_daily_limit():
            return False
            
        # 2. Fetch Ticks (Gap-less)
        new_ticks = self.clock.fetch()
        if not new_ticks:
            time.sleep(0.001) 
            return True
            
        # 3. Process Ticks
        for t in new_ticks:
            price = t['bid'] 
            ts = t['time_msc']
            
            # Update M1 Accumulator
            # Real-time M1 bar construction
            self.update_m1_buffer(t)
            
            # A. Update Renko
            new_bricks = self.renko.update_tick(price, ts)
            
            # B. Intra-Brick Logic (BE Check)
            # Only if we have an active position managed by us
            # We assume One Active Trade per Symbol
            be_price = self.renko.get_be_price()
            if be_price:
                 # Check condition
                 # If Long (Uptrend=1): Trigger if Price >= BE_Price
                 # If Short (Uptrend=-1): Trigger if Price <= BE_Price
                 
                 # Optimization: Only check if BE not already moved?
                 # OrderExecutor should handle idempotency
                 should_trigger = False
                 if self.renko.uptrend == 1 and price >= be_price:
                     should_trigger = True
                 elif self.renko.uptrend == -1 and price <= be_price:
                     should_trigger = True
                     
                 if should_trigger:
                     # Move SL to Entry
                     # We need the current active ticket. Assuming OrderExecutor knows it 
                     # or we find it via positions.
                     # We'll call a generic method
                     self.orders.move_sl_to_entry(SYMBOL)
            
            # C. New Brick Handling
            for brick in new_bricks:
                logger.info(f"New Brick: {brick}")
                self.process_signal(brick)
                
        return True
        
    def update_m1_buffer(self, tick):
        """
        Maintains the self.m1_buffer DataFrame in real-time.
        """
        ts_sec = tick['time'] # Epoch seconds
        dt = pd.to_datetime(ts_sec, unit='s')
        # Floor to minute
        dt_floored = dt.floor('min')
        
        # Check if we have a bar for this minute
        if dt_floored not in self.m1_buffer.index:
            # Create new bar
            new_row = pd.DataFrame([{
                'open': tick['bid'],
                'high': tick['bid'],
                'low': tick['bid'],
                'close': tick['bid'],
                'volume': tick['volume'] 
            }], index=[dt_floored])
            self.m1_buffer = pd.concat([self.m1_buffer, new_row])
            
            # Keep buffer size manageable (e.g. 1000 bars)
            if len(self.m1_buffer) > 2000:
                self.m1_buffer = self.m1_buffer.iloc[-1000:]
        else:
            # Update current bar
            idx = dt_floored
            self.m1_buffer.at[idx, 'high'] = max(self.m1_buffer.at[idx, 'high'], tick['bid'])
            self.m1_buffer.at[idx, 'low'] = min(self.m1_buffer.at[idx, 'low'], tick['bid'])
            self.m1_buffer.at[idx, 'close'] = tick['bid']
            self.m1_buffer.at[idx, 'volume'] += tick['volume']
            
    def process_signal(self, brick):
        # 1. Feature Engineering
        prev = self.renko.history[-2] if len(self.renko.history) > 1 else brick
        b_dict = brick._asdict() 
        p_dict = prev._asdict() # Contains 'sequence' now
        
        # Pass the M1 Buffer!
        ind_dict = self.features.get_indicators(self.m1_buffer)
        
        preds = self.predictors.predict(brick, self.renko.history[:-1], ind_dict) 
        
        obs = self.features.calculate_state(
            b_dict, p_dict, self.m1_buffer,
            preds, 
            self.state.get("daily_pnl", 0.0),
            0.5 
        )
        
        # Update Stack
        self.obs_stack.append(obs)
        # Prepare Stack for Transformer (1, 10, 21)
        # Pad if not full (though we warmed up)
        stack_arr = np.array(self.obs_stack)
        # Ensure shape (10, 21)
        if len(stack_arr) < 10:
             # Pad
             padding = np.zeros((10 - len(stack_arr), 21))
             stack_arr = np.vstack([padding, stack_arr])
        
        # 2. Inference
        action, self.lstm_states, score = self.ensemble.predict(
            obs, 
            lstm_states=self.lstm_states, 
            episode_starts=self.episode_starts,
            obs_stack=stack_arr # Pass stack to ensemble!
        )
        self.episode_starts = np.array([False])
        
        logger.info(f"Ensemble Vote: {score:.4f} -> Action: {action}")
        
        # 3. Execution (1:1 Ratio)
        if action == 1:
            # Entry at Brick Close (Current Price)
            entry = brick.close
            
            # R/R = 1:1
            # SL Distance = 1 Brick
            # TP Distance = 1 Brick
            # BE Logic is handled in pulse()
            
            dist = self.brick_size
            
            if brick.uptrend:
                # Buy
                sl = entry - dist
                tp = entry + dist
                direction = 1
            else:
                # Sell
                sl = entry + dist
                tp = entry - dist
                direction = -1
                
            self.orders.send_market_order(direction, sl, tp)

    def run(self):
        self.start()
        try:
            while True:
                if not self.pulse():
                    break
        except KeyboardInterrupt:
            logger.info("Orbit Stopped by User")
            self.connector.shutdown()

