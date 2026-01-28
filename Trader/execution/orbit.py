import time
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import pandas as pd
import os
from config.settings import DEFAULT_BRICK_SIZE, DEFAULT_OFFSET, SYMBOL, TIMEZONE_OFFSET
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
import MetaTrader5 as mt5

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
            # Apply Timezone Offset to History (Shift UTC -> Broker Time)
            # history_df['time'] is epoch seconds (UTC).
            # We add offset seconds.
            history_df['time'] = history_df['time'] + (TIMEZONE_OFFSET * 3600)
            
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
                # Row name is index (shifted date). timestamp() returns float seconds.
                # We need ms.
                ts_ms = int(row.name.timestamp() * 1000)
                # Feed Open, High, Low, Close traversal to capture intra-bar bricks?
                # OLD: self.renko.update_tick(row['close'], ts_ms)
                
                # NEW: Intra-candle traversal (High/Low) to capture wicks
                # Logic matches create_renko_optimized_t6.py
                p_open = row['open']
                p_close = row['close']
                p_high = row['high']
                p_low = row['low']
                
                if p_close >= p_open:
                    # Bullish Candle: Low -> High -> Close
                    prices = [p_low, p_high, p_close]
                else:
                    # Bearish Candle: High -> Low -> Close
                    prices = [p_high, p_low, p_close]
                    
                for p in prices:
                    self.renko.update_tick(p, ts_ms)
                
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
            
            # Add readable date (from SHIFTED timestamps)
            if 'timestamp' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save
            df.to_csv(filepath, index=False)
            logger.info(f"Saved Warmup Renko Snapshot to: {filepath} ({len(df)} bricks)")
            
        except Exception as e:
            logger.error(f"Failed to save Renko snapshot: {e}")
        
    def get_normalized_time_left(self):
        """
        Calculates normalized time left in the trading day [0, 1].
        """
        # UTC+Offset (Broker Time)
        now = datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET)
        
        # Session End: 23:59:00
        session_end = now.replace(hour=23, minute=59, second=0, microsecond=0)
        
        # If we are past 23:59, maybe it's next day? Or wait for reset?
        # Assuming we run 24h, session resets at 00:00.
        # Simple distance: Seconds until 23:59 today.
        
        time_left_sec = (session_end - now).total_seconds()
        
        # Total seconds in day (86400) or trading session?
        # Predictor trained on full day decay usually.
        # Normalize by 24h
        norm = time_left_sec / 86400.0
        
        # Clip
        return max(0.0, min(1.0, norm))

    def pulse(self):
        """
        Single heartbeat of the loop.
        """
        # 1. Risk Check
        if not self.risk.check_daily_limit():
            return False
            
        # 2. Check Active Trade Outcome
        # We check deals to see if our active trade closed
        # We rely on OrderExecutor to track 'active_ticket' or we check positions
        # Simplified: Check active positions. If our stored ticket handles missing position, it closed.
        
        active_ticket = self.state.get("active_ticket")
        if active_ticket:
            # Check if still open
            import MetaTrader5 as mt5
            positions = mt5.positions_get(ticket=active_ticket)
            
            if not positions:
                # IT CLOSED! Find out why.
                # history_deals_get (from=start of today, to=now)
                # Optimization: just check last few deals
                from_time = datetime.now() - timedelta(hours=24) # Safe window
                deals = mt5.history_deals_get(date_from=from_time, date_to=datetime.now() + timedelta(minutes=1))
                
                my_deal = None
                if deals:
                    # Find our ticket
                    for d in reversed(deals): # check latest first
                         if d.position_id == active_ticket and d.entry == mt5.DEAL_ENTRY_OUT:
                             my_deal = d
                             break
                             
                if my_deal:
                    # Determine Outcome
                    # 1. BE Check (Approximation)
                    # Deviation from expected PnL?
                    # Or Deviation from Entry Price
                    # Close Price vs Market Price at Entry (hard to get exact)
                    # Use Profit.
                    # BUT user said: BE can have +ve or -ve.
                    # Strict check: Distance from Entry.
                    
                    entry_price = self.state.get("active_entry_price", 0.0)
                    close_price = my_deal.price
                    direction = self.state.get("active_direction", 0) # 1 or -1
                    
                    # Thresholds
                    be_threshold = self.brick_size * 0.1 # 10% of brick
                    win_threshold = self.brick_size * 0.8 # 80% of brick
                    
                    price_diff = (close_price - entry_price) * direction
                    # Warning: 'price_diff' is Points * Dir.
                    
                    unit_pnl = 0.0
                    outcome_str = "BE"
                    
                    if abs(price_diff) < be_threshold or my_deal.reason == mt5.DEAL_REASON_SL:
                        # Re-check SL reason - could be Real SL or BE SL.
                        # If price is near entry, it's BE.
                        if abs(close_price - entry_price) < be_threshold:
                            unit_pnl = 0.0 # BE
                            outcome_str = "BE"
                        elif price_diff <= -win_threshold:
                            unit_pnl = -0.5 # LOSS
                            outcome_str = "LOSS"
                        else:
                            # Grey area? Treat as BE/Scratch or Loss?
                            # Default to Loss if negative
                            if price_diff < 0: 
                                unit_pnl = -0.5
                                outcome_str = "LOSS"
                            else:
                                unit_pnl = 0.5
                                outcome_str = "WIN"
                    else:
                        # Standard checking
                        if price_diff >= win_threshold:
                            unit_pnl = 0.5
                            outcome_str = "WIN"
                        elif price_diff <= -win_threshold:
                            unit_pnl = -0.5
                            outcome_str = "LOSS"
                        else:
                            unit_pnl = 0.0
                            outcome_str = "BE"
                            
                    logger.info(f"Trade Closed call. Outcome: {outcome_str} (PnL: {unit_pnl})")
                    
                    # UPDATE STATE
                    current_daily = self.state.get("daily_pnl", 0.0)
                    self.state.update("daily_pnl", current_daily + unit_pnl)
                else:
                    logger.warning(f"Trade {active_ticket} closed but Deal not found based on history lookup.")
                
                # Clear Active (ALWAYS, if position is gone)
                self.state.update("active_ticket", 0)
                self.state.update("active_entry_price", 0.0)

        # 3. Check Pending Orders (Limit Fallback Monitoring)
        pending_ticket = self.state.get("pending_ticket")
        if pending_ticket:
            import MetaTrader5 as mt5
            # Is it still pending?
            orders = mt5.orders_get(ticket=pending_ticket)
            
            if not orders:
                # It's gone from Pending. Did it fill?
                # Check Positions
                positions = mt5.positions_get(ticket=pending_ticket)
                if positions:
                    logger.info(f"Limit Order {pending_ticket} FILLED!")
                    # Promote to Active
                    pos = positions[0]
                    self.state.update("active_ticket", pending_ticket)
                    self.state.update("active_entry_price", pos.price_open)
                    dir = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                    self.state.update("active_direction", dir)
                    
                    # Clear Pending State
                    self.state.update("pending_ticket", 0)
                    self.state.update("pending_be_level", 0.0)
                else:
                    # Cancelled or Expired?
                    logger.info(f"Limit Order {pending_ticket} expired or cancelled externally.")
                    self.state.update("pending_ticket", 0)
            else:
                # Still Pending. Check Price for Invalidation (Runs Away).
                be_level = self.state.get("pending_be_level", 0.0)
                direction = self.state.get("pending_direction", 0)
                
                cancel_needed = False
                if direction == 1: # Buy Limit
                    # If Bid rises ABOVE BE Level (Run away profit)
                    curr_bid = mt5.symbol_info_tick(SYMBOL).bid
                    if curr_bid > be_level:
                        logger.warning(f"Pending Buy Missed Move ({curr_bid} > {be_level}). Cancelling.")
                        cancel_needed = True
                elif direction == -1: # Sell Limit
                    # If Ask drops BELOW BE Level (Run away profit)
                    curr_ask = mt5.symbol_info_tick(SYMBOL).ask
                    if curr_ask < be_level:
                        logger.warning(f"Pending Sell Missed Move ({curr_ask} < {be_level}). Cancelling.")
                        cancel_needed = True
                        
                if cancel_needed:
                    if self.orders.cancel_order(pending_ticket):
                        self.state.update("pending_ticket", 0)
            
        # 4. Fetch Ticks (Gap-less)
        new_ticks = self.clock.fetch()
        if not new_ticks:
            time.sleep(0.001) 
            return True
            
        # 4. Process Ticks
        for t in new_ticks:
            price = t['bid'] 
            # Apply Timezone Offset to Live Ticks
            ts = t['time_msc'] + (TIMEZONE_OFFSET * 3600 * 1000)
            
            # Apply Offset to tick Dict for M1 Buffer
            t_shifted = t.copy()
            t_shifted['time'] = t['time'] + (TIMEZONE_OFFSET * 3600)
            
            # Update M1 Accumulator
            self.update_m1_buffer(t_shifted)
            
            # A. Update Renko
            new_bricks = self.renko.update_tick(price, ts)
            
            # B. Intra-Brick Logic (BE Check)
            be_price = self.renko.get_be_price()
            if be_price:
                 should_trigger = False
                 if self.renko.uptrend == 1 and price >= be_price:
                     should_trigger = True
                 elif self.renko.uptrend == -1 and price <= be_price:
                     should_trigger = True
                     
                 if should_trigger:
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
        
        # DYNAMIC TIME LEFT and PNL from State
        time_left = self.get_normalized_time_left()
        pnl = self.state.get("daily_pnl", 0.0)
        # Fix: Clip PnL to match Training Limits [-5, 5]
        pnl = max(-5.0, min(5.0, pnl))
        
        obs = self.features.calculate_state(
            b_dict, p_dict, self.m1_buffer,
            preds, 
            pnl,
            time_left 
        )
        
        # --- FEATURE MASKING FIX ---
        # Training Environment masks Structure Features [3, 4, 5, 6]
        # Indices: 
        # 0,1: Regime
        # 2: BiLSTM
        # 3,4,5,6: Structure (Uptrend, BrickSize, Duration, Flip)
        obs[3:7] = 0.0
        # ---------------------------
        
        # Update Stack
        self.obs_stack.append(obs)
        # Prepare Stack for Transformer (1, 10, 21)
        stack_arr = np.array(self.obs_stack)
        if len(stack_arr) < 10:
             padding = np.zeros((10 - len(stack_arr), 21))
             stack_arr = np.vstack([padding, stack_arr])
        
        # 2. Latency Check (Catch-Up Mode)
        # Check if brick is "Live" or "History"
        # We compare brick timestamp (adjusted) to System Time (adjusted same way if needed)
        # Simplest: Compare to current UTC timestamp + Offset
        
        system_time_ms = time.time() * 1000
        # If brick has offset applied, we apply same to system? 
        # In pulse: ts = t['time_msc'] + (TIMEZONE_OFFSET * 3600 * 1000)
        # brick.timestamp from pulse ts.
        # So we shift system time too.
        current_adjusted_ms = system_time_ms + (TIMEZONE_OFFSET * 3600 * 1000)
        
        latency_ms = current_adjusted_ms - brick.timestamp
        is_catchup = latency_ms > 60000 # 60 Seconds lag
        
        if is_catchup:
             if len(self.renko.history) % 10 == 0:
                 logger.info(f"Catching up... Brick {brick.date} (Latency: {latency_ms/1000:.1f}s)")
             
             # Skip Inference and Execution
             # But we MUST update the stack (done above)
             return
             
        # 3. Inference
        action, self.lstm_states, score = self.ensemble.predict(
            obs, 
            lstm_states=self.lstm_states, 
            episode_starts=self.episode_starts,
            obs_stack=stack_arr 
        )
        self.episode_starts = np.array([False])
        
        logger.info(f"Ensemble Vote: {score:.4f} -> Action: {action}")
        
        # 3. Execution (1:1 Ratio)
        if action == 1:
            # Check if we already have an active trade?
            # Ideally the RL agent decides WHEN to enter.
            # If we limit to 1 trade at a time, we skip.
            if self.state.get("active_ticket", 0) != 0:
                logger.info("Signal Skipped: Trade already active.")
                return
            
            # Check if we have a pending Limit Order?
            if self.state.get("pending_ticket", 0) != 0:
                logger.info("Signal Skipped: Pending Limit Order active.")
                return

            # Entry at Brick Close (Current Price)
            entry = brick.close
            
            # R/R = 1:1
            dist = self.brick_size
            
            if brick.uptrend:
                # Buy
                sl = entry - dist
                tp = entry + dist
                direction = 1
                
                # SLIPPAGE CHECK
                # Get current Ask
                current_price = mt5.symbol_info_tick(SYMBOL).ask
                slippage = current_price - entry
                
                # Max Slippage: 8% of Brick
                max_slip = self.brick_size * 0.08
                
                # BE Level (Run-Away Invalidation): 0.3125 * Brick *IN PROFIT*
                # If Price > Entry + 0.3125*Brick, we missed the move.
                be_dist = self.brick_size * 0.3125
                be_level = entry + be_dist
                
                if slippage > max_slip:
                    logger.warning(f"High Slippage ({slippage:.5f} > {max_slip:.5f}). switch to Limit.")
                    
                    # Pre-Check: Has price already run away?
                    current_bid = mt5.symbol_info_tick(SYMBOL).bid
                    if current_bid > be_level:
                         logger.warning(f"Limit Order Skipped: Price {current_bid} ran away past {be_level}")
                         return
                         
                    # Place Limit @ Entry
                    ticket = self.orders.send_limit_order(direction, entry, sl, tp)
                    if ticket:
                        self.state.update("pending_ticket", ticket)
                        self.state.update("pending_be_level", be_level)
                        self.state.update("pending_direction", direction)
                    return
            else:
                # Sell
                sl = entry + dist
                tp = entry - dist
                direction = -1
                
                # SLIPPAGE CHECK
                # Get current Bid
                current_price = mt5.symbol_info_tick(SYMBOL).bid
                slippage = entry - current_price 
                
                max_slip = self.brick_size * 0.08
                
                # BE Level (Run-Away Invalidation): 0.3125 * Brick *IN PROFIT* (Below Entry)
                be_dist = self.brick_size * 0.3125
                be_level = entry - be_dist
                
                if slippage > max_slip:
                     logger.warning(f"High Slippage ({slippage:.5f} > {max_slip:.5f}). switch to Limit.")
                     
                     current_ask = mt5.symbol_info_tick(SYMBOL).ask
                     if current_ask < be_level:
                          logger.warning(f"Limit Order Skipped: Price {current_ask} ran away past {be_level}")
                          return
                          
                     ticket = self.orders.send_limit_order(direction, entry, sl, tp)
                     if ticket:
                        self.state.update("pending_ticket", ticket)
                        self.state.update("pending_be_level", be_level)
                        self.state.update("pending_direction", direction)
                     return
                
            ticket = self.orders.send_market_order(direction, sl, tp)
            
            if ticket:
                self.state.update("active_ticket", ticket)
                self.state.update("active_entry_price", entry)
                self.state.update("active_direction", direction)

    def run(self):
        self.start()
        try:
            while True:
                if not self.pulse():
                    break
        except KeyboardInterrupt:
            logger.info("Orbit Stopped by User")
            self.connector.shutdown()

