import numpy as np
from collections import namedtuple

# Lightweight Event
# Lightweight Event
NewBrickEvent = namedtuple('NewBrickEvent', ['open', 'close', 'high', 'low', 'uptrend', 'timestamp', 'sequence'])

class RenkoBuilder:
    def __init__(self, brick_size, start_price, grid_offset=0.0):
        self.brick_size = brick_size
        self.grid_offset = grid_offset
        self.current_price = start_price
        
        self.history = [] # List of NewBrickEvent
        self.uptrend = 0 # 1 or -1
        
        # State for "Phantom" (Intra-brick)
        self.virtual_high = start_price
        self.virtual_low = start_price
        
        # Sequence Tracking (binary string "1011...")
        self.sequence = ""
        
    def update_tick(self, price, timestamp):
        """
        Process a single price update.
        Returns: list of NewBrickEvent (could be multiple if gap fill)
        """
        new_bricks = []
        
        # Check Uptrend Formation
        if self.uptrend in [0, 1]:
            # Potential UP brick: Close = Current + BrickSize
            thresh_up = self.current_price + self.brick_size
            if price >= thresh_up:
                while price >= self.current_price + self.brick_size:
                    self.current_price += self.brick_size
                    self.uptrend = 1
                    
                    # Update Sequence
                    self.sequence += "1"
                    if len(self.sequence) > 100: self.sequence = self.sequence[-100:]
                    
                    brick = NewBrickEvent(
                        open=self.current_price - self.brick_size,
                        close=self.current_price,
                        high=self.current_price,
                        low=self.current_price - self.brick_size,
                        uptrend=True,
                        timestamp=timestamp,
                        sequence=self.sequence
                    )
                    self.history.append(brick)
                    new_bricks.append(brick)
                    # Reset phantom
                    self.virtual_high = self.current_price
                    self.virtual_low = self.current_price
                    
        # Check Downtrend Formation
        if self.uptrend in [0, -1]:
            thresh_down = self.current_price - self.brick_size
            
            # Reversal logic check (Standard Renko vs user simplified?)
            # User said "simple 1:1 system".
            # Standard Renko requires 2 bricks to reverse direction visually.
            # But the logic we optimized in simulate_profit_jit was:
            # "price <= current_brick_price - 2 * brick_size" for reversal.
            # Let's stick to that robust logic if uptrend==1.
            
            threshold = thresh_down
            if self.uptrend == 1:
                threshold = self.current_price - (2 * self.brick_size)
                
            if price <= threshold:
                 # If previous was UP, and we cross threshold, we form a DOWN brick.
                 # The "skipped" brick in the gap is part of the reversal.
                 
                 while price <= (self.current_price - self.brick_size if self.uptrend!=1 else self.current_price - 2*self.brick_size):
                    if self.uptrend == 1:
                         # Reversal Jump
                         self.current_price -= self.brick_size 
                         
                    self.current_price -= self.brick_size
                    self.uptrend = -1
                    
                    # Update Sequence
                    self.sequence += "0"
                    if len(self.sequence) > 100: self.sequence = self.sequence[-100:]

                    brick = NewBrickEvent(
                        open=self.current_price + self.brick_size,
                        close=self.current_price,
                        high=self.current_price + self.brick_size,
                        low=self.current_price,
                        uptrend=False,
                        timestamp=timestamp,
                        sequence=self.sequence
                    )
                    self.history.append(brick)
                    new_bricks.append(brick)
                    self.virtual_high = self.current_price
                    self.virtual_low = self.current_price

        # Update Intra-brick High/Low
        self.virtual_high = max(self.virtual_high, price)
        self.virtual_low = min(self.virtual_low, price)
        
        return new_bricks

    def get_be_price(self):
        """
        Calculates the Break-Even trigger price for the current forming brick.
        """
        if self.uptrend == 1:
            # We are looking for next UP brick.
            # Entry would be at Current + Size? No, Entry was previous.
            # We are monitoring the *active* trade.
            # If we bought at 110 (Current).
            # We want BE trigger at 110 + 0.3125*Size.
            return self.current_price + (self.brick_size * 0.3125)
        elif self.uptrend == -1:
            # Sold at 90.
            # BE Trigger at 90 - 0.3125*Size.
            return self.current_price - (self.brick_size * 0.3125)
        return None
