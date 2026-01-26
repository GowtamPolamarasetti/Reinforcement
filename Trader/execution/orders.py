import MetaTrader5 as mt5
import time
from config.settings import SYMBOL, LOT_SIZE, DEVIATION, MAGIC_NUMBER
from utils.logger import logger

class OrderExecutor:
    def __init__(self):
        pass
        
    def send_market_order(self, action, sl_price=None, tp_price=None):
        """
        Action: 1 (BUY), -1 (SELL)
        """
        # Determine Type
        order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(SYMBOL).ask if action == 1 else mt5.symbol_info_tick(SYMBOL).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": float(LOT_SIZE),
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": "RL_Ensemble",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl_price:
            request["sl"] = float(sl_price)
        if tp_price:
            request["tp"] = float(tp_price)
            
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Send Failed: {result.retcode} - {result.comment}")
            return None
            
        logger.info(f"Order Executed: {result.order} | Price: {result.price} | SL: {sl_price}")
        return result.order

    def modify_sl(self, ticket, new_sl):
        # Get existing position to keep TP
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Ticket {ticket} not found for modification.")
            return False
            
        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": SYMBOL,
            "position": ticket, # Position ID
            "sl": float(new_sl),
            "tp": pos.tp,
            "magic": MAGIC_NUMBER,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"SL Modify Failed: {result.retcode}")
            return False
            
        logger.info(f"Modified SL for {ticket} to {new_sl}")
        return True
        
    def close_all(self):
        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        if positions:
            for pos in positions:
                type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(SYMBOL).bid if type_close == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "position": pos.ticket,
                    "volume": pos.volume,
                    "type": type_close,
                    "price": price,
                    "deviation": DEVIATION,
                    "magic": MAGIC_NUMBER,
                }
                mt5.order_send(request)
                logger.info(f"Closed position {pos.ticket}")
