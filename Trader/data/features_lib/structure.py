import pandas as pd
import numpy as np

class StructureFeatures:
    def __init__(self):
        pass
        
    def get_features(self, renko_row, prev_renko_row=None):
        """
        Extract Renko-specific features.
        Input: Series (row of renko dataframe) 'brick_size', 'uptrend', etc.
        """
        uptrend = 1 if renko_row['uptrend'] else -1
        brick_size = renko_row['brick_size']
        
        # Time delta
        if prev_renko_row is not None:
             dt = (renko_row['date'] - prev_renko_row['date']).total_seconds()
             # Trend flip?
             prev_uptrend = 1 if prev_renko_row['uptrend'] else -1
             flip = 1 if uptrend != prev_uptrend else 0
        else:
             dt = 0
             flip = 0
             
        # Streak (consecutive same direction)
        # Assuming 'sequence' column might have this info, but user said 'digit concatenated'.
        # We can implement a simple counter if we iterate sequentially in the Env.
        # But for stateless usage, we might depend on existing columns or pass state.
        # I'll rely on the Env valid logic for streak.
        
        # For now, return [direction, brick_size, duration, flip]
        return np.array([uptrend, brick_size, dt, flip])
