import MetaTrader5 as mt5
from config.settings import SYMBOL
from utils.logger import logger
import time

class MT5Connector:
    def __init__(self):
        self.connected = False
        
    def connect(self):
        if not mt5.initialize():
            logger.error("MT5 Initialize failed")
            return False
            
        # Ensure Symbol is valid
        if not mt5.symbol_select(SYMBOL, True):
            logger.error(f"Failed to select symbol {SYMBOL}")
            return False
            
        logger.info(f"MT5 Connected. Code: {mt5.last_error()}")
        self.connected = True
        return True
        
    def shutdown(self):
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 Shutdown")
        
    def get_info(self):
        return mt5.terminal_info()

    def check_connection(self):
        # Re-connect if needed
        term_info = mt5.terminal_info()
        if term_info is None:
            logger.warning("Connection lost. Reconnecting...")
            return self.connect()
        return True
