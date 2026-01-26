import time
import numpy as np
from datetime import datetime
from config.settings import DEFAULT_BRICK_SIZE, DEFAULT_OFFSET
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
        self.brick_size = saved_brick if saved_brick > 0 else DEFAULT_BRICK_SIZE
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
            
        # Initialize Renko with current price (or history replay)
        current_tick = self.connector.get_info() # or symbol_info_tick
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick("XAUUSD") # Use config symbol
        start_price = tick.ask # Approximate
        
        self.renko = RenkoBuilder(self.brick_size, start_price, self.offset)
        logger.info(f"Orbit Started. Brick: {self.brick_size}, Price: {start_price}")
        
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
        # TODO: Get proper features for predictor
        preds = self.predictors.predict(None) 
        
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
