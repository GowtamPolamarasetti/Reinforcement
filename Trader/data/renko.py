import numpy as np
from collections import namedtuple

# Lightweight Event
NewBrickEvent = namedtuple('NewBrickEvent', ['open', 'close', 'high', 'low', 'uptrend', 'timestamp'])

class RenkoBuilder:
    def __init__(self, brick_size, start_price, grid_offset=0.0):
        self.brick_size = brick_size
        self.grid_offset = grid_offset
        self.current_price = start_price
        
        # Align start price to grid? 
        # Logic: If we start at X, do we snap to grid?
        # The design says "Virtual Grid: Offset + N * Size".
        # We'll trust the start_price is chemically the last brick close (anchor).
        
        self.history = [] # List of NewBrickEvent
        self.uptrend = 0 # 1 or -1
        
        # State for "Phantom" (Intra-brick)
        self.virtual_high = start_price
        self.virtual_low = start_price
        
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
                # How many bricks? (Gap Handling)
                while price >= self.current_price + self.brick_size:
                    self.current_price += self.brick_size
                    self.uptrend = 1
                    brick = NewBrickEvent(
                        open=self.current_price - self.brick_size,
                        close=self.current_price,
                        high=self.current_price,
                        low=self.current_price - self.brick_size,
                        uptrend=True,
                        timestamp=timestamp
                    )
                    self.history.append(brick)
                    new_bricks.append(brick)
                    # Reset phantom
                    self.virtual_high = self.current_price
                    self.virtual_low = self.current_price
                    
        # Check Downtrend Formation
        if self.uptrend in [0, -1]:
            # Potential DOWN brick: Close = Current - BrickSize
            # Note: If we just reversed from UP, we need 2 bricks?
            # Standard Renko reverses if price moves 2 bricks from High/Low?
            # OR simple "Next Brick" logic?
            # The Design says: "Does this individual tick breach Next_Brick_Top or Next_Brick_Bottom?"
            # Implies symmetric Grid.
            
            # If we are strictly on a grid, reversing requires crossing the previous line?
            # Let's assume Standard Renko: 
            # If UP, we need to drop by 2 bricks to reverse? Or 1 brick below open?
            # "Current Brick" is [Close-Size, Close]. 
            # To Reverse Down, we need to go to Close-Size-Size = Open-Size.
            
            # Simplified Grid Logic (as per design hint): 
            # Current Price is the "Locked" level.
            # Next Up = Current + Size.
            # Next Down = Current - Size.
            # If we are UpTrend, Current is the Top of the brick.
            # If we are DownTrend, Current is the Bottom of the brick.
            
            # Wait, Standard Renko (wicks excluded):
            # Up Brick: Open=100, Close=110. Current=110.
            # Next Up: 120. (Diff +10)
            # Next Down: 90. (Diff -20) -> Requires 2x size to reverse.
            
            # Let's implement this Standard 2x Reverse Logic.
            
            thresh_down = self.current_price - self.brick_size
            
            # Special Case: Reversal requires 2 bricks if we are in trend?
            # Actually, standard Renko usually says "Price must close below previous brick open".
            # Which is Current - 2*Size.
            
            threshold = thresh_down
            if self.uptrend == 1:
                threshold = self.current_price - (2 * self.brick_size)
                
            if price <= threshold:
                 while price <= (self.current_price - self.brick_size if self.uptrend!=1 else self.current_price - 2*self.brick_size):
                    # If reversing
                    if self.uptrend == 1:
                        # First brick of reversal is "double" distance visually or just jumps?
                        # Usually we form 1 brick at [100, 90].
                        # Jump down 2 slots.
                        self.current_price -= (2 * self.brick_size)
                        # Actually Renko prints distinct bricks.
                        # It prints a brick from 100->90.
                        # The brick before was 100->110.
                        # So we form a DOWN brick.
                        # Wait, logic: Current=110.
                        # We cross 90.
                        # We form brick 100->90.
                        # So current becomes 90.
                        pass
                        # To simplify loop, handling 2x jump is tricky in while.
                        # Let's do step by step.
                        
                    # Simple Step logic:
                    # If we are UP (110), and price hits 90.
                    # We form brick 100->90. current becomes 90. uptrend becomes -1.
                    if self.uptrend == 1:
                        # Fake intermediate update to align to grid
                         self.current_price -= self.brick_size # to 100 (Open of prev)
                         # Now we are at 100, neutral state? 
                         # Proceed to form the down brick
                         
                    self.current_price -= self.brick_size
                    self.uptrend = -1
                    brick = NewBrickEvent(
                        open=self.current_price + self.brick_size,
                        close=self.current_price,
                        high=self.current_price + self.brick_size,
                        low=self.current_price,
                        uptrend=False,
                        timestamp=timestamp
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
