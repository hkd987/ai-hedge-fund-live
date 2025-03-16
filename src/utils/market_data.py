"""
Market Data Utilities
--------------------
Functions for retrieving market data useful for risk assessment and trading decisions.
Uses Alpaca's Market Data API to ensure consistency with our trading platform.
"""

import logging
import os
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('market_data')

# Import Alpaca market data client
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("Alpaca SDK not installed. Market data functions will not work properly.")
    ALPACA_AVAILABLE = False

def get_alpaca_data_client():
    """Initialize and return Alpaca data client if credentials are available"""
    if not ALPACA_AVAILABLE:
        logger.warning("Alpaca SDK not installed. Run 'pip install alpaca-py' to enable market data retrieval.")
        return None

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found in environment variables.")
        return None
    
    try:
        return StockHistoricalDataClient(api_key, api_secret)
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca data client: {e}")
        return None

def get_alpaca_trading_client():
    """Initialize and return Alpaca trading client if credentials are available"""
    if not ALPACA_AVAILABLE:
        logger.warning("Alpaca SDK not installed. Run 'pip install alpaca-py' to enable market data retrieval.")
        return None

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found in environment variables.")
        return None
    
    try:
        # Check if live trading is enabled
        live_trading = os.getenv("LIVE_TRADING", "false").lower() == "true"
        
        # Use paper trading unless LIVE_TRADING is explicitly set to true
        is_paper = not live_trading
        client = TradingClient(api_key, api_secret, paper=is_paper)
        
        # Log which environment we're using
        env_type = "paper trading" if is_paper else "live trading"
        logger.info(f"Initialized Alpaca trading client for {env_type}")
        
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca trading client: {e}")
        return None

def get_current_vix():
    """
    Fetches the current VIX index value as a volatility indicator using Alpaca.
    The VIX is a real-time volatility index that represents market expectations 
    of 30-day forward-looking volatility.
    
    Returns:
        float: The current VIX value, or None if data can't be retrieved
    """
    if not ALPACA_AVAILABLE:
        logger.error("Alpaca SDK not available. Unable to fetch VIX data.")
        return None
        
    try:
        # Get Alpaca client
        client = get_alpaca_data_client()
        if not client:
            return None
            
        # VIX ticker in Alpaca is "VIX"
        request_params = StockBarsRequest(
            symbol_or_symbols=["VIX"],
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=5),  # Get data for the last 5 days to ensure we have recent data
            end=datetime.now(),
            adjustment='all'
        )
        
        # Get the bars data
        bars = client.get_stock_bars(request_params)
        
        # Convert to dataframe and get the latest close
        if bars and "VIX" in bars:
            df = bars["VIX"].df
            if not df.empty:
                # Get the most recent close price
                vix_value = df['close'].iloc[-1]
                logger.info(f"Current VIX value from Alpaca: {vix_value}")
                return vix_value
        
        logger.warning("No VIX data found in Alpaca response")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching VIX data from Alpaca: {e}")
        return None

def get_market_status():
    """
    Checks if the US market is currently open using Alpaca's API.
    
    Returns:
        dict: Dictionary with market status information
    """
    if not ALPACA_AVAILABLE:
        logger.error("Alpaca SDK not available. Unable to check market status.")
        return {"is_open": None, "next_open": None, "next_close": None}
        
    try:
        # Get Alpaca client
        client = get_alpaca_trading_client()
        if not client:
            return {"is_open": None, "next_open": None, "next_close": None}
            
        # Get clock information
        clock = client.get_clock()
        
        # Format the response
        market_status = {
            "is_open": clock.is_open,
            "next_open": clock.next_open.isoformat() if clock.next_open else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None
        }
        
        return market_status
        
    except Exception as e:
        logger.error(f"Error checking market status with Alpaca: {e}")
        return {"is_open": None, "next_open": None, "next_close": None}

def get_market_breadth():
    """
    Gets market breadth indicators (advancers vs decliners) to gauge market health using Alpaca data.
    
    This is an approximation using a sample of major stocks from S&P 500 to determine
    the general market breadth.
    
    Returns:
        dict: Dictionary with advance-decline indicators
    """
    if not ALPACA_AVAILABLE:
        logger.error("Alpaca SDK not available. Unable to calculate market breadth.")
        return None
        
    try:
        # Get Alpaca client
        client = get_alpaca_data_client()
        if not client:
            return None
            
        # Sample of major S&P 500 stocks to measure market breadth
        # This is a simplification since we don't have direct access to all NYSE/NASDAQ stocks
        sample_tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "V", "JPM", "JNJ",
            "UNH", "PG", "HD", "BAC", "MA", "XOM", "DIS", "CSCO", "VZ", "INTC",
            "NFLX", "ADBE", "CRM", "PYPL", "CMCSA", "PEP", "ABT", "TMO", "COST", "AVGO"
        ]
        
        # Get yesterday's date for comparison
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # Request for today's data
        request_params = StockBarsRequest(
            symbol_or_symbols=sample_tickers,
            timeframe=TimeFrame.Day,
            start=datetime.combine(yesterday, datetime.min.time()),
            end=datetime.now(),
            adjustment='all'
        )
        
        # Get the bars data
        bars = client.get_stock_bars(request_params)
        
        # Process the data to determine advancers and decliners
        advancers = 0
        decliners = 0
        unchanged = 0
        adv_volume = 0
        dec_volume = 0
        
        # Convert to DataFrame for each symbol and compare today to yesterday
        if bars:
            for symbol in sample_tickers:
                if symbol in bars:
                    symbol_bars = bars[symbol].df
                    if len(symbol_bars) >= 2:  # Need at least 2 days to compare
                        # Get the last two days
                        yesterday_close = symbol_bars['close'].iloc[-2]
                        today_close = symbol_bars['close'].iloc[-1]
                        today_volume = symbol_bars['volume'].iloc[-1]
                        
                        # Compare prices
                        if today_close > yesterday_close:
                            advancers += 1
                            adv_volume += today_volume
                        elif today_close < yesterday_close:
                            decliners += 1
                            dec_volume += today_volume
                        else:
                            unchanged += 1
        
        # Calculate ratios and metrics
        total_stocks = advancers + decliners + unchanged
        adv_dec_ratio = advancers / decliners if decliners > 0 else float('inf')
        
        # Simple advance-decline line (can be improved with historical tracking)
        adv_dec_line = advancers - decliners
        
        market_breadth = {
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "adv_vol": int(adv_volume),
            "dec_vol": int(dec_volume),
            "adv_dec_ratio": float(adv_dec_ratio),
            "adv_dec_line": adv_dec_line,
            "sample_size": total_stocks
        }
        
        logger.info(f"Market breadth from Alpaca: {advancers} advancers, {decliners} decliners")
        return market_breadth
        
    except Exception as e:
        logger.error(f"Error calculating market breadth with Alpaca: {e}")
        return None 