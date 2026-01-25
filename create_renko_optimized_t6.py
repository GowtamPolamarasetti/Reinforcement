import pandas as pd
import numpy as np
import os
import sys
import multiprocessing
from glob import glob
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Global variable for worker processes
df_global = None

def _init_worker(data):
    """Initializer to share dataframe across processes."""
    global df_global
    df_global = data

# ==========================================
# Core Renko Logic (Optimized for Speed)
# ==========================================

import pandas as pd
import numpy as np
import os
import sys
import multiprocessing
from glob import glob
from datetime import datetime, timedelta
import warnings
from numba import jit, int64, float64, boolean, types 

warnings.filterwarnings('ignore')

# Global variable for worker processes
# Global variable for worker processes
df_global = None
days_global = None

def _init_worker(data, days):
    """Initializer to share dataframe and day list across processes."""
    global df_global, days_global
    df_global = data
    days_global = np.array(days) # Cache as numpy array

# ==========================================
# Core Renko Logic (Numba Optimized)
# ==========================================

# We use int64 for timestamps (nanoseconds) because Numba handles numpy datetime64 as int64 seamlessly usually,
# or we cast.

@jit(nopython=True)
def generate_bricks_with_window_jit(
    opens, highs, lows, closes, dates, 
    start_index, start_price, brick_size, 
    t_minus_5_ts_val, # int64 timestamp
):
    """
    Generates bricks from start_index forward. 
    Returns arrays of brick properties for bricks that fall into our relevant window.
    
    Returns: (dates_out, opens_out, closes_out, uptrends_out)
    """
    n = len(dates)
    current_brick_price = start_price
    
    # Pre-allocate estimation (safe upper bound? 5 days of 1-min data -> max 10k bricks usually)
    # We use dynamic list and convert or just fixed array if possible.
    # Numba Lists are experimental but work for append.
    
    # Output storage
    out_dates = []
    out_opens = []
    out_closes = []
    out_uptrends = []
    out_highs = [] # Renko High (formed)
    out_lows = []
    
    uptrend = 0 # 0: None, 1: True, -1: False
    
    for i in range(start_index, n):
        row_date = dates[i] # int64 usually
        
        price_open = opens[i]
        price_close = closes[i]
        price_high = highs[i]
        price_low = lows[i]

        if price_close >= price_open:
            if uptrend == 0: # Init logic inside loop to avoid code dup?
               # Check Low, High, Close
               prices = [price_low, price_high, price_close]
            else:
               prices = [price_low, price_high, price_close]
        else:
            prices = [price_high, price_low, price_close]
            
        # Unroll small loop manually or iterate
        for p_idx in range(3): 
            price = prices[p_idx]
            
            if uptrend == 0:
                if price >= current_brick_price + brick_size:
                    while price >= current_brick_price + brick_size:
                        current_brick_price += brick_size
                        if row_date >= t_minus_5_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price - brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(1)
                            out_highs.append(current_brick_price)
                            out_lows.append(current_brick_price - brick_size)
                    uptrend = 1
                elif price <= current_brick_price - brick_size:
                    while price <= current_brick_price - brick_size:
                        current_brick_price -= brick_size
                        if row_date >= t_minus_5_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price + brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(-1)
                            out_highs.append(current_brick_price + brick_size)
                            out_lows.append(current_brick_price)
                    uptrend = -1
            else:
                if uptrend == 1:
                    if price >= current_brick_price + brick_size:
                        while price >= current_brick_price + brick_size:
                            current_brick_price += brick_size
                            if row_date >= t_minus_5_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price - brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(1)
                                out_highs.append(current_brick_price)
                                out_lows.append(current_brick_price - brick_size)
                                
                    elif price <= current_brick_price - 2 * brick_size:
                        current_brick_price -= 2 * brick_size
                        if row_date >= t_minus_5_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price + brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(-1)
                            out_highs.append(current_brick_price + brick_size)
                            out_lows.append(current_brick_price)
                        uptrend = -1
                        while price <= current_brick_price - brick_size:
                            current_brick_price -= brick_size
                            if row_date >= t_minus_5_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price + brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(-1)
                                out_highs.append(current_brick_price + brick_size)
                                out_lows.append(current_brick_price)
                
                else: # Downtrend (-1)
                    if price <= current_brick_price - brick_size:
                        while price <= current_brick_price - brick_size:
                            current_brick_price -= brick_size
                            if row_date >= t_minus_5_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price + brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(-1)
                                out_highs.append(current_brick_price + brick_size)
                                out_lows.append(current_brick_price)
                                
                    elif price >= current_brick_price + 2 * brick_size:
                        current_brick_price += 2 * brick_size
                        if row_date >= t_minus_5_ts_val:
                            out_dates.append(row_date)
                            out_opens.append(current_brick_price - brick_size)
                            out_closes.append(current_brick_price)
                            out_uptrends.append(1)
                            out_highs.append(current_brick_price)
                            out_lows.append(current_brick_price - brick_size)
                        uptrend = 1
                        while price >= current_brick_price + brick_size:
                            current_brick_price += brick_size
                            if row_date >= t_minus_5_ts_val:
                                out_dates.append(row_date)
                                out_opens.append(current_brick_price - brick_size)
                                out_closes.append(current_brick_price)
                                out_uptrends.append(1)
                                out_highs.append(current_brick_price)
                                out_lows.append(current_brick_price - brick_size)
                                    
    return out_dates, out_opens, out_closes, out_uptrends, out_highs, out_lows

@jit(nopython=True)
def simulate_profit_jit(
    b_dates, b_closes, b_uptrends,
    df_dates, df_highs, df_lows, 
    brick_size
):
    """
    Calculates profit.
    b_* are arrays of brick data (filtered for simulation window).
    df_* are arrays of 1-min data (filtered for simulation window).
    """
    n_bricks = len(b_dates)
    if n_bricks == 0:
        return 0.0
    
    daily_pnl = 0.0
    last_idx = 0
    max_idx = len(df_dates)
    
    for i in range(n_bricks):
        entry_price = b_closes[i]
        uptrend = b_uptrends[i] # 1 or -1
        start_time = b_dates[i]
        
        # Determine Trade params
        if uptrend == 1:
            tp_price = entry_price + brick_size
            sl_price = entry_price - brick_size
            be_trigger = entry_price + (0.3125 * brick_size)
            trade_type = 1 # BUY
        else:
            tp_price = entry_price - brick_size
            sl_price = entry_price + brick_size
            be_trigger = entry_price - (0.3125 * brick_size)
            trade_type = -1 # SELL
            
        # Outcome Deterministic Check via Next Brick
        if i < n_bricks - 1:
            next_start = b_dates[i+1]
            if start_time == next_start:
                # Same time brick logic
                next_trend = b_uptrends[i+1]
                if next_trend == uptrend:
                    daily_pnl += 1.0
                else:
                    daily_pnl -= 1.0
                continue
            end_time = next_start # Actually next brick formation time is limit
        else:
            end_time = df_dates[-1] 
            
        # Price Action Check
        # Fast forward
        curr_idx = last_idx
        while curr_idx < max_idx and df_dates[curr_idx] <= start_time:
            curr_idx += 1
            
        outcome = 0 # 0: Open/Unknown, 1: Win, -1: Loss
        sl_moved_to_be = False
        
        scan_idx = curr_idx
        while scan_idx < max_idx:
            # Check time bound
            # In simulation, we typically check until next brick formation.
            # If next brick forms at T2, price action T1->T2 applies.
            if df_dates[scan_idx] >= end_time: # Inclusive or exclusive?
                # Usually exclusive of the exact tick that formed next brick? 
                # Or inclusive? Assuming exclusive to match "between bricks".
                # If we hit end_time, the NEXT brick logic takes over?
                # Actually original logic: "date <= end_time".
                # Numba efficient check: if date > end_time break
                if df_dates[scan_idx] > end_time:
                    break
                # But wait, next brick *start_time* is the time it formed.
                # So PA up to that time is valid.
            
            h = df_highs[scan_idx]
            l = df_lows[scan_idx]
            
            if trade_type == 1: # BUY
                current_sl = entry_price if sl_moved_to_be else sl_price
                if l <= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0 # Loss or BE (0 profit)
                    break
                if h >= tp_price:
                    outcome = 1
                    break
                if h >= be_trigger:
                    sl_moved_to_be = True
            else: # SELL
                current_sl = entry_price if sl_moved_to_be else sl_price
                if h >= current_sl:
                    outcome = -1 if not sl_moved_to_be else 0
                    break
                if l <= tp_price:
                    outcome = 1
                    break
                if l <= be_trigger:
                    sl_moved_to_be = True
            
            scan_idx += 1
        
        last_idx = curr_idx 
        
        if outcome == 1:
            daily_pnl += 0.5
        elif outcome == -1:
            daily_pnl -= 0.5
            
    return daily_pnl


def process_day_task(target_dir_path):
    global df_global, days_global # Use shared memory DF
    
    try:
        # Extract Day
        day_str = os.path.basename(target_dir_path)
        target_day = pd.to_datetime(day_str).date()
        target_day_ts = pd.Timestamp(target_day)
        
        # Fast Lookup in pre-calculated days
        # days_global is numpy array of dates
        day_locs = np.where(days_global == target_day)[0]
        if len(day_locs) == 0: 
            return # Target day not in data?
            
        day_idx = day_locs[0]
            
        if day_idx < 7:
            return 
            
        t_minus_5 = days_global[day_idx-5]
        t_minus_5_ts = pd.Timestamp(t_minus_5)
        
        # Data Slice: Anchor Start -> Target End
        # df_global is indexed by 'day' (sorted)
        # We can slice directly
        
        # ANCHOR CHANGE: Rolling Window T-6
        anchor_day = days_global[day_idx - 6] 
        
        # Anchor Data (Instant lookup)
        # Handle case where .loc returns Series (unlikely for intraday)
        try:
            anchor_rows = df_global.loc[anchor_day]
            if isinstance(anchor_rows, pd.Series):
                 anchor_rows = anchor_rows.to_frame().T
        except KeyError:
            return
            
        anchor_high = anchor_rows['high'].max()
        anchor_low = anchor_rows['low'].min()
        anchor_open = anchor_rows.iloc[0]['open']
        
        # Target Data
        try:
            target_rows = df_global.loc[target_day]
            if isinstance(target_rows, pd.Series):
                target_rows = target_rows.to_frame().T
        except KeyError:
             return
             
        # Use first open of target day
        target_open = target_rows.iloc[0]['open']
        
        brick_size = (target_open * 0.00236)/2
        step_size = anchor_open * 0.00236 * 0.01
        
        # Candidates (High to Low)
        candidates = np.arange(anchor_low, anchor_high + step_size/1000, step_size)[::-1]
        
        # Prepare Data Arrays for Window
        # Slice including Anchor Day through Target Day
        # Since index is sorted, this is fast and simple
        slice_df = df_global.loc[anchor_day : target_day]
        
        # Ensure Types for Numba
        opens = slice_df['open'].values.astype(np.float64)
        highs = slice_df['high'].values.astype(np.float64)
        lows = slice_df['low'].values.astype(np.float64)
        closes = slice_df['close'].values.astype(np.float64)
        
        # Dates as int64 (nanoseconds)
        dates = slice_df['date'].values.astype(np.int64)
        
        # Anchor Hit Check Arrays
        n_anchor = len(anchor_rows)
        a_highs = highs[:n_anchor] # relative to slice
        a_lows = lows[:n_anchor]
        
        # Timestamps as int64
        t_minus_5_int = t_minus_5_ts.value
        target_day_int = target_day_ts.value # 00:00:00 of Target Day
        
        # Optimization
        best_profit = -float('inf')
        best_cand = candidates[0]
        
        # Slice for Simulation Profit (Pre-filter rows)
        # We need rows where date >= T-5 and date < Target
        # Use numpy mask
        sim_dates = dates
        
        # Limits
        # t_minus_5_ts is Timestamp
        t5_limit = t_minus_5_ts.value
        tgt_limit = target_day_ts.value
        
        sim_mask = (sim_dates >= t5_limit) & (sim_dates < tgt_limit)
        
        sim_dates_filtered = sim_dates[sim_mask]
        sim_highs_filtered = highs[sim_mask]
        sim_lows_filtered = lows[sim_mask]
        
        # --- Optimization Loop ---
        for cand in candidates:
            # Check Hit
            hits = (a_highs >= cand) & (a_lows <= cand)
            if not hits.any():
                continue
            hit_idx = np.argmax(hits)
            
            # Build JIT
            b_dates, b_opens, b_closes, b_uptrends, _, _ = generate_bricks_with_window_jit(
                opens, highs, lows, closes, dates,
                hit_idx, cand, brick_size,
                t_minus_5_int
            )
            
            # Convert outputs (Numba typed lists) to simple mechanism for simulation?
            # simulate_profit_jit expects arrays. 
            # numba list -> np array is implicit or explicit needed?
            # We can iterate numba list in jit function? Yes.
            # But simulate function signature: b_dates, b_closes, etc.
            
            # If nothing returned
            if not b_dates:
               continue
               
            # Numba lists to arrays (if needed, or just slice in python)
            # We need history bricks (date < target day)
            
            # Filter manually or pass to JIT with threshold?
            # Let's filter in python - list comprehension on Numba list works but slow?
            # Better: Pass full lists + target threshold to `simulate_profit_jit`?
            # Actually our `simulate_profit_jit` simulates PnL based on matches.
            # It expects brick arrays.
            
            # Helper to filter:
            # Iterating numba list in python is bad.
            # We assume small number of bricks (5 days ~ 500-2000 bricks).
            # Convert to numpy?
            
            bd = np.array(b_dates)
            bc = np.array(b_closes)
            bu = np.array(b_uptrends)
            
            # Mask for history
            hist_mask = (bd < tgt_limit)
            
            hist_dates = bd[hist_mask]
            hist_closes = bc[hist_mask]
            hist_uptrends = bu[hist_mask]
            
            pnl = simulate_profit_jit(
                hist_dates, hist_closes, hist_uptrends,
                sim_dates_filtered, sim_highs_filtered, sim_lows_filtered,
                brick_size
            )
            
            if pnl > best_profit:
                best_profit = pnl
                best_cand = cand
                
        # --- Generation ---
        # Re-build best
        hits = (a_highs >= best_cand) & (a_lows <= best_cand)
        hit_idx = np.argmax(hits)
        
        b_dates, b_opens, b_closes, b_uptrends, b_highs, b_lows = generate_bricks_with_window_jit(
            opens, highs, lows, closes, dates,
            hit_idx, best_cand, brick_size,
            t_minus_5_int
        )
        
        if not b_dates:
            return 
            
        # Reconstruct Objects for Saving
        final_bricks = []
        for i in range(len(b_dates)):
            final_bricks.append({
                'date': pd.Timestamp(b_dates[i]), # Convert back
                'open': b_opens[i],
                'close': b_closes[i],
                'uptrend': (b_uptrends[i] == 1),
                'high': b_highs[i],
                'low': b_lows[i],
                'brick_size': brick_size
            })
        
        # Target only
        target_only = [b for b in final_bricks if b['date'].date() == target_day]
        
        # Sequence
        for i, b in enumerate(final_bricks):
            if b['date'].date() == target_day:
                # Calc sequence
                start = max(0, i-50)
                seq_bricks = final_bricks[start:i]
                seq_str = "".join(['1' if x['uptrend'] else '0' for x in seq_bricks])
                b['sequence'] = seq_str
                
        # Filter again enriched bricks
        target_only = [b for b in final_bricks if b['date'].date() == target_day and 'sequence' in b]
        
        if not target_only:
            return 
            
        res_df = pd.DataFrame(target_only)
        # Drop internal fields if needed or keep
        out_path = os.path.join(target_dir_path, 'renko_optimized2_t6.csv')
        res_df.to_csv(out_path, index=False)
        
        return f"Done {day_str} | Best: {best_cand:.2f} | PnL: {best_profit}"
        
    except Exception as e:
        return f"Error {day_str}: {e}"


def main():
    print("Starting Renko Optimization (Rolling T-6 Anchor)...")
    
    # Load Data
    raw_path = 'Data/Raw/XAUUSD_data_ohlc.csv'
    if not os.path.exists(raw_path):
        print("Raw data not found.")
        return
        
    print("Reading CSV...")
    df = pd.read_csv(raw_path)
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.date
    
    # Drop 1st day strictly
    days = sorted(df['day'].unique())
    if len(days) > 1:
        # Filter rows
        df = df[df['day'] != days[0]].reset_index(drop=True)
        # Update days list
        days = sorted(df['day'].unique())
        
    print(f"Data Loaded: {len(df)} rows. Anchoring on {days[0]} (Index 0)")

    # OPTIMIZATION: Set Index for Fast Slicing
    df = df.set_index('day').sort_index()
    
    # SHARE MEMORY VIA GLOBALS (FOR FORK CONTEXT)
    # This avoids pickling 300MB+ for every worker.
    global df_global, days_global
    df_global = df
    days_global = np.array(days)

    # Get Targets
    processed_dir = 'Data/Processed/XAUUSD_dukas'
    dirs = sorted(glob(os.path.join(processed_dir, "*")))
    targets = [d for d in dirs if os.path.isdir(d) and len(os.path.basename(d)) == 10]
    
    print(f"Targets: {len(targets)}")
    
    # Process
    num_cores = max(1, multiprocessing.cpu_count() - 2) # Leave 2 cores free
    print(f"Pool: {num_cores} workers (Using FORK context for memory efficiency).")
    
    # Dry Run Check
    if "--dry-run" in sys.argv:
        print("DRY RUN: Processing last 3 targets only.")
        targets = targets[-3:]
    
    # Use 'fork' context
    # This copies the current process memory (with loaded df_global) to workers COW.
    try:
        ctx = multiprocessing.get_context('fork')
    except ValueError:
        # Fallback for Windows or systems without fork (unlikely here as user is on Mac)
        ctx = multiprocessing.get_context('spawn')
        print("Warning: Forced to use 'spawn'. Memory usage may be high.")
        # If spawn, we NEED initializer
        # But we refactored main... let's assume fork works on Mac/Linux.
        
    if ctx.get_start_method() == 'spawn':
         # Fallback logic if fork fails
         with ctx.Pool(num_cores, initializer=_init_worker, initargs=(df, days)) as pool:
            for result in pool.imap_unordered(process_day_task, targets):
                if result: print(result, flush=True)
    else:
        # Fork logic (Faster, Shared Memory)
        with ctx.Pool(num_cores) as pool:
            for result in pool.imap_unordered(process_day_task, targets):
                if result: print(result, flush=True)

    print("Complete.")

if __name__ == "__main__":
    main()
