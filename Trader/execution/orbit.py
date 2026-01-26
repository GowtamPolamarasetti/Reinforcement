import time
import numpy as np
from datetime import datetime
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
        self.clock = TickStream() # State loaded inside or fresh?
        
        # Load Optimization params from state or default
        saved_brick = self.state.get("optimization", {}).get("brick_size", 0.0)
        self.brick_size = saved_brick # Will be finalized in start()
        self.offset = self.state.get("optimization", {}).get("grid_offset", DEFAULT_OFFSET)
        
        # Core Logic
        # We need a start price for Renko. 
        # Ideally we fetch history to find where we are.
        # For now, start from current price.
        # TODO: Implement "Warmup" to replay history and align Renko.
        
        self.renko = None # Initialized in start()
        
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
        
        # Buffer for Transformer (Stack of 10)
        # We need to store previous OBSERVATIONS.
        self.obs_buffer = [] 
        
    def start(self):
        if not self.connector.connect():
            return
            
        # T6 Optimization Logic
        import MetaTrader5 as mt5
        from data.renko_optimizer import RenkoOptimizer
        import pandas as pd
        
        logger.info("Fetching M1 History for Optimization (7 Days)...")
        # Fetch 7 days
        days_back = 7
        date_from = datetime.now() - pd.Timedelta(days=days_back)
        
        ticks = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_M1, datetime.now(), 1440 * days_back)
        
        if ticks is None or len(ticks) == 0:
            logger.warning("History Fetch Failed. Using default Fallback.")
            start_price = mt5.symbol_info_tick(SYMBOL).ask
            self.brick_size = start_price * 0.00118
            self.renko = RenkoBuilder(self.brick_size, start_price, 0.0)
        else:
            # Convert to DF
            history_df = pd.DataFrame(ticks)
            history_df['date'] = pd.to_datetime(history_df['time'], unit='s')
            history_df.sort_values('time', inplace=True) # Ensure chronological order before replay
            
            optimizer = RenkoOptimizer()
            best_bs, best_offset, _ = optimizer.optimize(history_df)
            
            self.brick_size = best_bs
            # Initialize Renko from the OPTIMIZED START PRICE (best_offset)
            # The RenkoBuilder needs proper history initialization.
            # Best Offset is effectively the Anchor Price.
            # We initialize the builder starting from that price.
            # Ideally we perform "catch up" processing of the history to reach current state.
            
            # For simplicity & correctness:
            # 1. Init Renko with best_offset as start_price.
            # 2. Feed it the history_df ticks/M1 to build up to "Now".
            
            self.renko = RenkoBuilder(self.brick_size, best_offset, 0.0)
            
            # Catch up from Anchor (Optimization Start) to Now
            # We use the M1 closes for speed
            logger.info("Replaying history to sync Renko state...")
            for row in history_df.itertuples():
                 ts_ms = int(row.time * 1000)
                 self.renko.update_tick(row.close, ts_ms)
            
        logger.info(f"Orbit Started. Brick: {self.brick_size:.4f}")
        
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
            time.sleep(0.001) # Tiny sleep to prevent CPU burn
            return True
            
        # 3. Process Ticks
        for t in new_ticks:
            price = t['bid'] # Use Bid for chart construction usually
            ts = t['time_msc']
            
            # A. Update Renko
            new_bricks = self.renko.update_tick(price, ts)
            
            # B. Intra-Brick Logic (BE Check)
            # Check active orders logic here...
            # if self.orders.check_be_trigger(price, self.renko.uptrend): ...
            # TODO: Link active ticket to Renko state
            
            # C. New Brick Handling
            for brick in new_bricks:
                logger.info(f"New Brick: {brick}")
                self.process_signal(brick)
                
        return True
        
    def process_signal(self, brick):
        # 1. Feature Engineering
        # We need prev_brick, m1_window, preds, etc.
        # Placeholder M1 window
        prev = self.renko.history[-2] if len(self.renko.history) > 1 else brick
        # Convert namedtuple to dict if needed by features
        b_dict = brick._asdict() # namedtuple method
        p_dict = prev._asdict()
        
        # Predictions
        # Predictions
        # 1. Get Indicators
        # TODO: We need M1 Window DF. 
        # Hack: Pass None for now if we don't have buffering of M1 data yet.
        # OrbitEngine needs to maintain M1 Buffer from TickStream?
        # TickStream gives ticks. We need M1 candles.
        # For now, we will pass None, and FeatureEngineer returns default dict.
        # Real implementation: Accumulate ticks into M1 bars in Data Layer.
        
        ind_dict = self.features.get_indicators(None) 
        
        preds = self.predictors.predict(brick, self.renko.history[:-1], ind_dict) 
        
        # Assemble Vector
        obs = self.features.calculate_state(
            b_dict, p_dict, None, # No M1 DF yet
            preds, 
            self.state.get("daily_pnl", 0.0),
            0.5 # Dummy time left
        )
        
        # 2. Inference
        action, self.lstm_states, score = self.ensemble.predict(
            obs, self.lstm_states, self.episode_starts
        )
        self.episode_starts = np.array([False])
        
        logger.info(f"Ensemble Vote: {score:.4f} -> Action: {action}")
        
        # 3. Execution
        if action == 1:
            # Calculate SL/TP
            # SL = 2 Bricks, TP = 1 Brick? Or Agent Exits?
            # Design: "Hard Stop 2 Bricks". "Trade Win +8.0" implies logic.
            # Strategy usually takes Fixed TP or trailing?
            # Design says "Trade Win" is +8.
            # Implementation: Send Market Order with SL.
            
            sl_dist = 2 * self.brick_size
            tp_dist = 1 * self.brick_size # Conservative?
            
            entry = brick.close
            if brick.uptrend:
                sl = entry - sl_dist
                tp = entry + tp_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
                
            self.orders.send_market_order(1 if brick.uptrend else -1, sl, tp)

    def run(self):
        self.start()
        try:
            while True:
                if not self.pulse():
                    break
        except KeyboardInterrupt:
            logger.info("Orbit Stopped by User")
            self.connector.shutdown()
