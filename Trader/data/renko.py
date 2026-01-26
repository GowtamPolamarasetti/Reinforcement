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
        
        # 1. Check Continuation / Formation UP
        # Trigger: 
        #   If Neutral: Price >= Current + Size
        #   If UP: Price >= Current + Size
        #   If DOWN: Price >= Current + 2*Size (Reversal)
        
        up_threshold = self.current_price + self.brick_size
        if self.uptrend == -1:
            up_threshold = self.current_price + (2 * self.brick_size)
            
        if price >= up_threshold:
            # We have an UP move (Continuation or Reversal)
            while price >= (self.current_price + self.brick_size if self.uptrend != -1 else self.current_price + 2*self.brick_size):
                
                if self.uptrend == -1:
                    # Reversal Jump (Ghost Brick)
                    # Move Current Price UP by 1 brick to the "pivot"
                    self.current_price += self.brick_size
                    # Now we are "at" the pivot, next brick is the real new UP brick
                    
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

        # 2. Check Continuation / Formation DOWN
        # Trigger:
        #   If Neutral: Price <= Current - Size
        #   If DOWN: Price <= Current - Size
        #   If UP: Price <= Current - 2*Size (Reversal)
        
        down_threshold = self.current_price - self.brick_size
        if self.uptrend == 1:
            down_threshold = self.current_price - (2 * self.brick_size)
            
        if price <= down_threshold:
            # We have a DOWN move
            while price <= (self.current_price - self.brick_size if self.uptrend != 1 else self.current_price - 2*self.brick_size):
                
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
