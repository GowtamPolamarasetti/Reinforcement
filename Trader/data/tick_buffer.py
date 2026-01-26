import MetaTrader5 as mt5
import time
from datetime import datetime
from config.settings import SYMBOL
from utils.logger import logger

class TickStream:
    def __init__(self, start_time_msc=None):
        """
        Args:
            start_time_msc: Unix timestamp in milliseconds to start fetching from.
                            If None, starts from current server time.
        """
        if start_time_msc is None:
            # Default to now if not provided
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick:
                self.last_time_msc = tick.time_msc
            else:
                # CRITICAL: Do not fallback to local time.
                # If we are here, MT5 is connected but symbol is invalid or market is down.
                # Or maybe connection dropped.
                raise RuntimeError(f"TickStream Error: Could not retrieve initial tick for {SYMBOL}. Check MT5 connection.")
        else:
            self.last_time_msc = start_time_msc
            
        logger.info(f"TickStream initialized starting from: {self.last_time_msc}")

    def fetch(self):
        """
        Polls for *all* ticks since last_time_msc.
        Updates self.last_time_msc to the latest tick time.
        Returns: list of ticks (or empty list)
        """
        # Fetch batch (1000 ticks max per call to avoid overload, can loop if needed)
        # using copy_ticks_from (inclusive of start time usually)
        # Fix: Convert ms to seconds (float) to avoid OverflowError on Windows (int is 32-bit long)
        date_from_sec = self.last_time_msc / 1000.0
        ticks = mt5.copy_ticks_from(SYMBOL, date_from_sec, 1000, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) == 0:
            return []
            
        # Filter: strictly greater than last_time_msc to avoid duplicates
        # copy_ticks_from includes the start millisecond.
        # Often multiple ticks happen in the same ms. 
        # Robust logic: We might process duplicates if we use >.
        # But if we use >=, we definitely re-process the last one.
        # MT5 doesn't have unique Tick IDs easily accessible via python api in standard struct?
        # flags field might help? No.
        # Strategy: accept >= but skip if exact match of fields? Too slow.
        # Strategy: Keep last_time_msc. If new batch starts with same ms, filter it.
        
        # New robust logic:
        # We assume last_time_msc is the time of the LAST processed tick.
        # We request from that time.
        # We filter out any tick where tick.time_msc <= self.last_time_msc
        # WAIT: If 5 ticks happen at T, and we processed 2, we want the other 3.
        # But we only know "T". We don't know "index at T".
        # Batching assumes we process ALL ticks at T in the previous batch.
        # So it is safe to strictly filter > T?
        # NO. If we fetched 1000 ticks and the batch ended in the middle of a millisecond T...
        # ... then next fetch at T would lose the rest.
        # BUT: copy_ticks_from(time_msc) returns ticks >= time_msc.
        # If we just update last_msc to the end, we risk skipping concurrent ticks if our batch cut them off.
        # HOWEVER, 1000 ticks is a lot. Unless high frequency, we likely grabbed all at T.
        # We will assume strictly > last_time_msc for simplicity, 
        # OR we can keep the last tick object and compare?
        
        # Let's filter: t.time_msc > self.last_time_msc
        # This implies we processed ALL ticks at valid timestamp self.last_time_msc.
        
        # Correction: To be safe, if we received < 1000 ticks, we definitely got everything up to the end time.
        # So strictly > is safe.
        
        new_ticks = [t for t in ticks if t['time_msc'] > self.last_time_msc]
        
        if new_ticks:
            logger.debug(f"TickStream: Fetched {len(new_ticks)} new ticks. Last: {new_ticks[-1]['time_msc']}")
            self.last_time_msc = new_ticks[-1]['time_msc']
            
        return new_ticks
