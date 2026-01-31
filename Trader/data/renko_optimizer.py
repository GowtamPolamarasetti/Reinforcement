import numpy as np
import pandas as pd
from numba import jit
from utils.logger import logger
from config.settings import BRICK_SIZE_FACTOR

# ==========================================
# Core Renko Logic (Numba Optimized)
# ==========================================

@jit(nopython=True)
def generate_bricks_jit(
    opens, highs, lows, closes, dates, 
    start_index, start_price, brick_size, 
    min_ts_val # int64 timestamp
):
    """
    Generates bricks. Numba optimized.
    """
    n = len(dates)
    current_brick_price = start_price
    
    # Output
    out_dates = []
    out_opens = []
    out_closes = []
    out_uptrends = []
    
    uptrend = 0 
    
    for i in range(start_index, n):
        row_date = dates[i]
        price_close = closes[i]
        price_high = highs[i]
        price_low = lows[i]

        # Order of checking: Low, High, Close based on candle direction?
        # Standard: Check high/low against current range
        # Simplified for JIT: Check prices against limits
        
        # We need to process both High and Low if volatility is high
        # Prioritize based on candle move?
        if price_close >= opens[i]:
            prices = [price_low, price_high, price_close]
        else:
            prices = [price_high, price_low, price_close]
            
        for p in prices:
            if uptrend == 0:
                if p >= current_brick_price + brick_size:
                    while p >= current_brick_price + brick_size:
                        current_brick_price += brick_size
                        if row_date >= min_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price - brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(1)
                    uptrend = 1
                elif p <= current_brick_price - brick_size:
                    while p <= current_brick_price - brick_size:
                        current_brick_price -= brick_size
                        if row_date >= min_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price + brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(-1)
                    uptrend = -1
            else:
                if uptrend == 1:
                    if p >= current_brick_price + brick_size:
                        while p >= current_brick_price + brick_size:
                            current_brick_price += brick_size
                            if row_date >= min_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price - brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(1)
                    elif p <= current_brick_price - 2 * brick_size:
                        current_brick_price -= 2 * brick_size
                        if row_date >= min_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price + brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(-1)
                        uptrend = -1
                        while p <= current_brick_price - brick_size:
                            current_brick_price -= brick_size
                            if row_date >= min_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price + brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(-1)
                else: # Downtrend
                    if p <= current_brick_price - brick_size:
                        while p <= current_brick_price - brick_size:
                            current_brick_price -= brick_size
                            if row_date >= min_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price + brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(-1)
                    elif p >= current_brick_price + 2 * brick_size:
                        current_brick_price += 2 * brick_size
                        if row_date >= min_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price - brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(1)
                        uptrend = 1
                        while p >= current_brick_price + brick_size:
                            current_brick_price += brick_size
                            if row_date >= min_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price - brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(1)
                                    
    return out_dates, out_opens, out_closes, out_uptrends

@jit(nopython=True)
def simulate_profit_jit(b_closes, b_uptrends, brick_size):
    """
    Simulates simple Brick-Follow strategy PnL for optimization ranking.
    Strategy: Fixed TP 1 brick, SL 2 bricks (simplification for offset finding).
    Actually T6 script uses price action simulation. 
    Simplification: Just count trend continuation matches for speed in live use?
    The T6 script logic is complex PA simulation. We will approximate or copy.
    Approximation for speed: 
    If next brick is same trend: +1
    If reversal: -1
    """
    pnl = 0.0
    n = len(b_closes)
    if n < 2: return 0.0
    
    for i in range(n-1):
        if b_uptrends[i+1] == b_uptrends[i]:
            pnl += 1.0 # Trend continuation
        else:
            pnl -= 1.0 # Reversal
            
    return pnl

class RenkoOptimizer:
    def __init__(self):
        pass
        
    def optimize(self, history_df):
        """
        Runs the T-6 Anchor optimization.
        Args:
            history_df: DataFrame with datetime index or 'date' col, OHLC.
                        Must contain at least 7 days of data preferable.
        Returns:
            best_brick_size (float)
            best_offset (float) -> Represented as best_candidate_price to start from.
            start_index (int) -> Index in df to start live processing from.
        """
        if history_df.empty:
            return None, None, 0
            
        df = history_df.copy()
        if 'date' not in df.columns:
            df = df.reset_index()
        
        # Ensure date type
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Get Unique Days
        df['day_date'] = df['date'].dt.date
        days = sorted(df['day_date'].unique())
        
        if len(days) < 2:
            logger.warning("Not enough history for optimization (need >1 day). Using defaults.")
            # Default
            latest_price = df.iloc[-1]['close']
            bs = latest_price * 0.00118
            return bs, latest_price, 0
            
        # Strategy: T-6 Anchor
        # Anchor Day = Day[-6] if exists, else Day[0]
        anchor_idx = max(0, len(days) - 7) # Approximately 7 days back
        anchor_day = days[anchor_idx]
        
        # Target Day = Today (Last)
        target_day = days[-1]
        
        anchor_data = df[df['day_date'] == anchor_day]
        if anchor_data.empty:
            anchor_data = df.iloc[0:100] # Fallback
            
        anchor_high = anchor_data['high'].max()
        anchor_low = anchor_data['low'].min()
        anchor_open = anchor_data.iloc[0]['open']
        
        # Target Open
        target_data = df[df['day_date'] == target_day]
        target_open = target_data.iloc[0]['open'] if not target_data.empty else df.iloc[-1]['close']
        
        # Brick Size calc
        # Factor comes from settings
        brick_size = target_open * BRICK_SIZE_FACTOR
        
        # Offset Candidates
        # Scan from Anchor Low to High
        step = anchor_open * 0.00236 * 0.01
        if step == 0: step = 0.1
        candidates = np.arange(anchor_low, anchor_high + step, step)
        
        # Prepare Arrays
        # We optimize on the window [Anchor, Target]
        mask = (df['day_date'] >= anchor_day)
        window_df = df[mask]
        
        opens = window_df['open'].values.astype(np.float64)
        highs = window_df['high'].values.astype(np.float64)
        lows = window_df['low'].values.astype(np.float64)
        closes = window_df['close'].values.astype(np.float64)
        dates = window_df['date'].values.astype(np.int64) # ns timestamp
        
        # Optimization Loop
        best_pnl = -float('inf')
        best_offset = candidates[0]
        
        # Just check a subset if too many candidates to be fast in live
        if len(candidates) > 50:
             # Take 50 equidistanced
             candidates = np.linspace(anchor_low, anchor_high, 50)
        
        for cand in candidates:
            # Generate Bricks
            # We assume start_index=0 relative to window_df (Anchor day start)
            _, _, b_closes, b_uptrends = generate_bricks_jit(
                opens, highs, lows, closes, dates,
                0, cand, brick_size, 0
            )
            
            pnl = simulate_profit_jit(np.array(b_closes), np.array(b_uptrends), brick_size)
            
            if pnl > best_pnl:
                best_pnl = pnl
                best_offset = cand
                
        logger.info(f"Optimization Results: BS={brick_size:.4f}, Offset={best_offset:.4f}, PnL={best_pnl}")
        
        return brick_size, best_offset, 0 # 0 means we consume full history?
        
