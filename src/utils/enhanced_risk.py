"""
Enhanced Risk Management Module
------------------------------
Additional risk management features for the AI Hedge Fund.
This module extends the basic risk management functionality with more sophisticated
risk assessment and management techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
import json
from datetime import datetime, date, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enhanced_risk')

# Try to import yfinance for sector mapping
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Some sector mapping features won't be available.")

# Directory for storing risk data
RISK_DATA_DIR = 'data/risk'
os.makedirs(RISK_DATA_DIR, exist_ok=True)

# ------------------- SECTOR EXPOSURE TRACKING -------------------

# Simplified sector mapping for common stocks
SECTOR_MAPPINGS = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology', 'INTC': 'Technology',
    'AMD': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology', 'CSCO': 'Technology',
    'AVGO': 'Technology', 'ORCL': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
    'IBM': 'Technology', 'SHOP': 'Technology', 'PYPL': 'Technology', 'SQ': 'Technology',
    'TSM': 'Technology', 'ACN': 'Technology', 'NOW': 'Technology', 'AMAT': 'Technology',
    'MU': 'Technology', 'LRCX': 'Technology', 'ABNB': 'Technology', 'UBER': 'Technology',
    'DASH': 'Technology', 'LYFT': 'Technology',
    
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare', 
    'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'MDT': 'Healthcare',
    'ISRG': 'Healthcare', 'GILD': 'Healthcare', 'CVS': 'Healthcare', 'MRNA': 'Healthcare',
    'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'BSX': 'Healthcare', 'BDX': 'Healthcare',
    'BIIB': 'Healthcare', 'ZTS': 'Healthcare', 'HCA': 'Healthcare', 'IDXX': 'Healthcare',
    
    # Financial
    'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial', 
    'MS': 'Financial', 'C': 'Financial', 'BLK': 'Financial', 'AXP': 'Financial',
    'SPGI': 'Financial', 'SCHW': 'Financial', 'CME': 'Financial', 'CB': 'Financial',
    'PNC': 'Financial', 'TFC': 'Financial', 'USB': 'Financial', 'ICE': 'Financial',
    'MCO': 'Financial', 'AIG': 'Financial', 'MET': 'Financial', 'COF': 'Financial',
    'PRU': 'Financial', 'TROW': 'Financial', 'NTRS': 'Financial', 'ALL': 'Financial',
    'V': 'Financial', 'MA': 'Financial',
    
    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
    'SBUX': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
    'TJX': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary', 'ROST': 'Consumer Discretionary',
    'YUM': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'GM': 'Consumer Discretionary',
    'DPZ': 'Consumer Discretionary', 'BBY': 'Consumer Discretionary', 'EBAY': 'Consumer Discretionary',
    
    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'CL': 'Consumer Staples',
    'EL': 'Consumer Staples', 'GIS': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'KHC': 'Consumer Staples', 'SYY': 'Consumer Staples', 'STZ': 'Consumer Staples',
    'MO': 'Consumer Staples', 'PM': 'Consumer Staples', 'KR': 'Consumer Staples',
    'HSY': 'Consumer Staples', 'K': 'Consumer Staples', 'CAG': 'Consumer Staples',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 
    'EOG': 'Energy', 'OXY': 'Energy', 'PXD': 'Energy', 'KMI': 'Energy',
    'VLO': 'Energy', 'WMB': 'Energy', 'BP': 'Energy', 'SHEL': 'Energy',
    'PSX': 'Energy', 'MPC': 'Energy', 'DVN': 'Energy', 'HAL': 'Energy',
    
    # Industrial
    'CAT': 'Industrial', 'DE': 'Industrial', 'BA': 'Industrial', 'HON': 'Industrial', 
    'MMM': 'Industrial', 'UPS': 'Industrial', 'FDX': 'Industrial', 'RTX': 'Industrial',
    'UNP': 'Industrial', 'GE': 'Industrial', 'LMT': 'Industrial', 'ADP': 'Industrial',
    'ETN': 'Industrial', 'EMR': 'Industrial', 'CSX': 'Industrial', 'NSC': 'Industrial',
    'WM': 'Industrial', 'JCI': 'Industrial', 'CARR': 'Industrial', 'OTIS': 'Industrial',
    'PCAR': 'Industrial', 'CTAS': 'Industrial', 'ROK': 'Industrial', 'CMI': 'Industrial',
    
    # Communication Services
    'NFLX': 'Communication', 'DIS': 'Communication', 'CMCSA': 'Communication',
    'VZ': 'Communication', 'T': 'Communication', 'TMUS': 'Communication',
    'CHTR': 'Communication', 'ATVI': 'Communication', 'EA': 'Communication',
    'TTWO': 'Communication', 'MTCH': 'Communication', 'OMC': 'Communication',
    'GOOG': 'Communication', 'GOOGL': 'Communication', 'META': 'Communication',
    'TWTR': 'Communication', 'SNAP': 'Communication', 'PINS': 'Communication',
    
    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities', 
    'AEP': 'Utilities', 'SRE': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities',
    'PCG': 'Utilities', 'ED': 'Utilities', 'AEE': 'Utilities', 'DTE': 'Utilities',
    'ES': 'Utilities', 'PEG': 'Utilities', 'WEC': 'Utilities', 'ETR': 'Utilities',
    
    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 
    'EQIX': 'Real Estate', 'PSA': 'Real Estate', 'O': 'Real Estate',
    'SPG': 'Real Estate', 'WELL': 'Real Estate', 'ARE': 'Real Estate',
    'DLR': 'Real Estate', 'VICI': 'Real Estate', 'AVB': 'Real Estate',
    
    # Materials
    'LIN': 'Materials', 'FCX': 'Materials', 'APD': 'Materials', 
    'SHW': 'Materials', 'NEM': 'Materials', 'ECL': 'Materials',
    'NUE': 'Materials', 'DOW': 'Materials', 'DD': 'Materials',
    'PPG': 'Materials', 'IP': 'Materials', 'VMC': 'Materials',
}

# In-memory cache for sector data to avoid repeated API calls
SECTOR_CACHE = {}
SECTOR_CACHE_FILE = os.path.join(RISK_DATA_DIR, 'sector_cache.json')

# Try to import Alpaca client from market_data module
try:
    from utils.market_data import get_alpaca_trading_client, ALPACA_AVAILABLE
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca market data module not available. Using fallback for sector data.")

def save_sector_cache():
    """Save the sector cache to a file for persistence between runs"""
    try:
        with open(SECTOR_CACHE_FILE, 'w') as f:
            json.dump(SECTOR_CACHE, f, indent=2)
        logger.info(f"Saved sector cache to {SECTOR_CACHE_FILE} with {len(SECTOR_CACHE)} entries")
    except Exception as e:
        logger.error(f"Error saving sector cache: {e}")

def load_sector_cache():
    """Load the sector cache from file if it exists"""
    global SECTOR_CACHE
    try:
        if os.path.exists(SECTOR_CACHE_FILE):
            with open(SECTOR_CACHE_FILE, 'r') as f:
                loaded_cache = json.load(f)
                SECTOR_CACHE.update(loaded_cache)
            logger.info(f"Loaded sector cache from {SECTOR_CACHE_FILE} with {len(SECTOR_CACHE)} entries")
    except Exception as e:
        logger.error(f"Error loading sector cache: {e}")

# Load the cache at module import time
load_sector_cache()

def get_sector_for_ticker(ticker: str) -> str:
    """
    Get the sector for a given ticker, using multiple sources in order of preference:
    1. In-memory cache (for speed)
    2. yfinance (for comprehensive sector information)
    3. Predefined mappings (as fallback)
    
    Args:
        ticker: The stock ticker symbol
        
    Returns:
        sector: The sector name or "Unknown" if not found
    """
    # Check the cache first for performance
    if ticker in SECTOR_CACHE:
        return SECTOR_CACHE[ticker]
    
    # Try using yfinance if available (most reliable source of sector data)
    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'sector' in info and info['sector']:
                # Cache the result
                sector = info['sector']
                SECTOR_CACHE[ticker] = sector
                logger.info(f"Retrieved sector for {ticker} from yfinance: {sector}")
                # Save the updated cache
                save_sector_cache()
                return sector
            
            # Also try the industry field if sector is not available
            if 'industry' in info and info['industry']:
                # For now, just log the industry
                industry = info['industry']
                logger.info(f"Retrieved industry for {ticker} from yfinance: {industry}")
                # Not using industry for sector categorization yet
        except Exception as e:
            logger.warning(f"Error getting sector from yfinance for {ticker}: {e}")
    
    # Try to get basic information from Alpaca
    # Note: Alpaca's standard API doesn't provide sector data directly
    if ALPACA_AVAILABLE:
        try:
            client = get_alpaca_trading_client()
            if client:
                try:
                    asset = client.get_asset(ticker)
                    if asset:
                        # Log asset information for debugging
                        logger.info(f"Asset info for {ticker} from Alpaca: {asset}")
                        # Alpaca doesn't provide sector data, so we don't return anything here
                except Exception as inner_e:
                    logger.debug(f"Could not get asset details from Alpaca for {ticker}: {inner_e}")
        except Exception as e:
            logger.warning(f"Error connecting to Alpaca for {ticker}: {e}")
    
    # Fall back to our predefined mappings
    if ticker in SECTOR_MAPPINGS:
        sector = SECTOR_MAPPINGS[ticker]
        # Cache the result
        SECTOR_CACHE[ticker] = sector
        logger.info(f"Using predefined sector for {ticker}: {sector}")
        # Save the updated cache
        save_sector_cache()
        return sector
    
    # If all else fails, return Unknown
    logger.warning(f"Could not determine sector for {ticker}, using 'Unknown'")
    SECTOR_CACHE[ticker] = "Unknown"
    # Save the updated cache
    save_sector_cache()
    return "Unknown"

def track_sector_exposure(portfolio: Dict) -> Dict[str, float]:
    """
    Calculate current exposure by sector to enforce sector diversification limits.
    
    Args:
        portfolio: Portfolio dictionary containing positions
        
    Returns:
        Dictionary of sector exposures as percentages of portfolio
    """
    sector_exposures = {}
    total_value = 0
    
    # Calculate total portfolio value
    cash = portfolio.get('cash', 0)
    total_value += cash
    
    # Get all positions and their values
    positions = portfolio.get('positions', {})
    position_values = {}
    
    for ticker, position in positions.items():
        # Calculate long position value
        if position.get('long', 0) > 0:
            long_value = position.get('long', 0) * position.get('long_cost_basis', 0)
            position_values[ticker] = position_values.get(ticker, 0) + long_value
            total_value += long_value
            
        # Calculate short position value
        if position.get('short', 0) > 0:
            short_value = position.get('short', 0) * position.get('short_cost_basis', 0)
            position_values[ticker] = position_values.get(ticker, 0) + short_value
            total_value += short_value
    
    # Skip further calculation if portfolio is empty
    if total_value == 0:
        return {'Cash': 100.0}
    
    # Add cash allocation
    sector_exposures['Cash'] = (cash / total_value) * 100
    
    # Calculate exposure by sector
    for ticker, value in position_values.items():
        sector = get_sector_for_ticker(ticker)
        sector_exposures[sector] = sector_exposures.get(sector, 0) + ((value / total_value) * 100)
    
    return sector_exposures

def check_sector_limits(sector_exposures: Dict[str, float], max_sector_pct: float = 25.0) -> List[str]:
    """
    Check for sectors exceeding the maximum allowed exposure.
    
    Args:
        sector_exposures: Dictionary of sector exposures as percentages
        max_sector_pct: Maximum allowed percentage exposure per sector
        
    Returns:
        List of sectors exceeding the limit
    """
    over_exposed = []
    
    for sector, exposure in sector_exposures.items():
        if sector != 'Cash' and exposure > max_sector_pct:
            over_exposed.append(f"{sector} ({exposure:.1f}%)")
            
    return over_exposed

# ------------------- CORRELATION RISK MANAGEMENT -------------------

def calculate_correlation_matrix(tickers: List[str], 
                               lookback_days: int = 90) -> Dict[Tuple[str, str], float]:
    """
    Calculate correlation matrix between tickers based on recent price history.
    
    Args:
        tickers: List of ticker symbols
        lookback_days: Number of days to look back for correlation calculation
        
    Returns:
        Dictionary with ticker pairs as keys and correlation values as values
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for correlation calculation")
        return {}
    
    # Skip if no tickers
    if not tickers:
        return {}
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get price data
        price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr().to_dict()
        
        # Convert to our desired format
        correlations = {}
        for ticker1 in tickers:
            if ticker1 not in corr_matrix:
                continue
                
            for ticker2 in tickers:
                if ticker1 >= ticker2:  # Skip self-correlations and duplicates
                    continue
                    
                if ticker2 in corr_matrix[ticker1]:
                    correlations[(ticker1, ticker2)] = corr_matrix[ticker1][ticker2]
        
        return correlations
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return {}

def check_correlation_risk(portfolio: Dict, 
                         correlation_threshold: float = 0.7) -> List[Dict]:
    """
    Identify highly correlated positions that might create hidden concentration risk.
    
    Args:
        portfolio: Portfolio dictionary containing positions
        correlation_threshold: Threshold above which to flag high correlation
        
    Returns:
        List of high correlation pairs with their details
    """
    # Get tickers with positions
    positions = portfolio.get('positions', {})
    tickers = [ticker for ticker, pos in positions.items() 
              if pos.get('long', 0) > 0 or pos.get('short', 0) > 0]
    
    if len(tickers) < 2:
        return []  # Need at least 2 tickers to check correlation
    
    # Calculate correlation matrix
    correlations = calculate_correlation_matrix(tickers)
    
    # Identify high correlation pairs
    high_correlation_pairs = []
    
    for (ticker1, ticker2), correlation in correlations.items():
        if abs(correlation) > correlation_threshold:
            # Get position details
            pos1 = positions.get(ticker1, {})
            pos2 = positions.get(ticker2, {})
            
            # Only include if both have active positions
            if ((pos1.get('long', 0) > 0 or pos1.get('short', 0) > 0) and
                (pos2.get('long', 0) > 0 or pos2.get('short', 0) > 0)):
                
                high_correlation_pairs.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'correlation': correlation,
                    'direction': 'positive' if correlation > 0 else 'negative',
                    'position1': 'long' if pos1.get('long', 0) > 0 else 'short',
                    'position2': 'long' if pos2.get('long', 0) > 0 else 'short',
                })
    
    return high_correlation_pairs

def save_correlation_data(correlation_data: Dict[Tuple[str, str], float]) -> None:
    """Save correlation data to disk for later analysis"""
    try:
        # Convert tuple keys to strings for JSON serialization
        serializable_data = {f"{t1}_{t2}": value for (t1, t2), value in correlation_data.items()}
        
        # Save to file
        filepath = os.path.join(RISK_DATA_DIR, 'correlations.json')
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f)
            
        logger.info(f"Saved correlation data for {len(correlation_data)} ticker pairs")
    except Exception as e:
        logger.error(f"Error saving correlation data: {e}")

def load_correlation_data() -> Dict[Tuple[str, str], float]:
    """Load previously saved correlation data"""
    try:
        filepath = os.path.join(RISK_DATA_DIR, 'correlations.json')
        if not os.path.exists(filepath):
            return {}
            
        with open(filepath, 'r') as f:
            serialized_data = json.load(f)
            
        # Convert string keys back to tuples
        correlation_data = {}
        for key, value in serialized_data.items():
            ticker1, ticker2 = key.split('_')
            correlation_data[(ticker1, ticker2)] = value
            
        logger.info(f"Loaded correlation data for {len(correlation_data)} ticker pairs")
        return correlation_data
    except Exception as e:
        logger.error(f"Error loading correlation data: {e}")
        return {}

# ------------------- MARKET REGIME DETECTION -------------------

def detect_market_regime(market_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect the current market regime to adjust risk parameters dynamically.
    
    Args:
        market_data: DataFrame of market index data (SPY or similar)
        
    Returns:
        Dictionary with market regime information
    """
    if market_data is None or market_data.empty or len(market_data) < 200:
        return {
            "regime": "Unknown",
            "confidence": 0.0,
            "risk_adjustment": 1.0,
            "reason": "Insufficient market data for regime detection"
        }
    
    try:
        # Calculate key moving averages
        market_data = market_data.copy()
        market_data['ma20'] = market_data['close'].rolling(window=20).mean()
        market_data['ma50'] = market_data['close'].rolling(window=50).mean()
        market_data['ma200'] = market_data['close'].rolling(window=200).mean()
        
        # Calculate volatility (20-day)
        market_data['daily_returns'] = market_data['close'].pct_change()
        volatility_20d = market_data['daily_returns'].rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
        
        # Get latest data point
        latest = market_data.iloc[-1]
        
        # Check moving average relationships
        price_above_ma20 = latest['close'] > latest['ma20']
        price_above_ma50 = latest['close'] > latest['ma50']
        price_above_ma200 = latest['close'] > latest['ma200']
        ma20_above_ma50 = latest['ma20'] > latest['ma50']
        ma50_above_ma200 = latest['ma50'] > latest['ma200']
        
        # Calculate recent trend
        last_20d_change = (latest['close'] / market_data.iloc[-21]['close'] - 1) * 100
        
        # Calculate drawdown from recent high
        recent_high = market_data['close'].rolling(window=60).max().iloc[-1]
        drawdown = (1 - latest['close'] / recent_high) * 100
        
        # Identify market regime based on these factors
        if price_above_ma50 and price_above_ma200 and ma50_above_ma200:
            if volatility_20d < 15 and last_20d_change > 2:
                regime = "Strong Bull"
                confidence = 0.9
                risk_adjustment = 1.0  # Full risk
                reason = f"Strong bull market: price above all MAs, low volatility ({volatility_20d:.1f}%), positive trend (+{last_20d_change:.1f}%)"
            elif volatility_20d < 20:
                regime = "Bull"
                confidence = 0.8
                risk_adjustment = 0.9  # Slightly reduced risk
                reason = f"Bull market: price above major MAs, moderate volatility ({volatility_20d:.1f}%)"
            else:
                regime = "Volatile Bull"
                confidence = 0.7
                risk_adjustment = 0.7  # Moderately reduced risk
                reason = f"Volatile bull market: price above major MAs but high volatility ({volatility_20d:.1f}%)"
        elif price_above_ma200:
            if price_above_ma50:
                regime = "Weakening Bull"
                confidence = 0.6
                risk_adjustment = 0.7
                reason = f"Weakening bull: price above MA200 and MA50, but technical deterioration, volatility: {volatility_20d:.1f}%"
            else:
                regime = "Correction"
                confidence = 0.7
                risk_adjustment = 0.5  # Half risk
                reason = f"Market correction: price below MA50 but above MA200, drawdown: {drawdown:.1f}%"
        elif ma50_above_ma200:
            regime = "Early Bear"
            confidence = 0.6
            risk_adjustment = 0.4  # Significantly reduced risk
            reason = f"Early bear market: price below MA200, MA50 still above MA200, drawdown: {drawdown:.1f}%"
        else:
            if volatility_20d > 30:
                regime = "Crisis"
                confidence = 0.8
                risk_adjustment = 0.2  # Minimal risk
                reason = f"Market crisis: price below all MAs, extreme volatility ({volatility_20d:.1f}%), drawdown: {drawdown:.1f}%"
            else:
                regime = "Bear"
                confidence = 0.8
                risk_adjustment = 0.3  # Very reduced risk
                reason = f"Bear market: price below all MAs, MA50 below MA200, drawdown: {drawdown:.1f}%"
        
        # Create the regime info dictionary
        regime_info = {
            "regime": regime,
            "confidence": confidence,
            "risk_adjustment": risk_adjustment,
            "volatility": volatility_20d,
            "drawdown": drawdown,
            "last_20d_change": last_20d_change,
            "price_above_ma50": price_above_ma50,
            "price_above_ma200": price_above_ma200,
            "ma50_above_ma200": ma50_above_ma200,
            "reason": reason
        }
        
        # Save the regime info for future reference
        save_market_regime(regime_info)
        
        return regime_info
        
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}")
        return {
            "regime": "Unknown",
            "confidence": 0.0,
            "risk_adjustment": 0.5,  # Default to conservative risk
            "reason": f"Error in market regime detection: {str(e)}"
        }

def save_market_regime(regime_info: Dict[str, Any]) -> None:
    """Save market regime data for tracking over time"""
    try:
        # Create a timestamped entry
        timestamped_entry = {
            "timestamp": datetime.now().isoformat(),
            **regime_info
        }
        
        # Load existing data
        filepath = os.path.join(RISK_DATA_DIR, 'market_regimes.json')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                regime_history = json.load(f)
        else:
            regime_history = []
        
        # Add new entry and keep only the last 100 entries
        regime_history.append(timestamped_entry)
        if len(regime_history) > 100:
            regime_history = regime_history[-100:]
        
        # Save back to file
        with open(filepath, 'w') as f:
            json.dump(regime_history, f)
            
    except Exception as e:
        logger.error(f"Error saving market regime data: {e}")

def get_recent_market_regimes(days: int = 30) -> List[Dict[str, Any]]:
    """Get recent market regime data for analysis"""
    try:
        filepath = os.path.join(RISK_DATA_DIR, 'market_regimes.json')
        
        if not os.path.exists(filepath):
            return []
            
        with open(filepath, 'r') as f:
            regime_history = json.load(f)
            
        # Filter to recent entries
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        recent_regimes = [
            regime for regime in regime_history 
            if regime.get('timestamp', '') >= cutoff_date
        ]
        
        return recent_regimes
        
    except Exception as e:
        logger.error(f"Error getting market regime history: {e}")
        return []

# ------------------- PROGRESSIVE DRAWDOWN PROTECTION -------------------

def calculate_position_scale_factor(drawdown_pct: float) -> float:
    """
    Scale position sizes based on drawdown level - progressively reduce
    risk as drawdown increases rather than using a binary circuit breaker.
    
    Args:
        drawdown_pct: Current drawdown percentage (0-100)
        
    Returns:
        Scale factor to apply to position sizes (0.0-1.0)
    """
    # No reduction until 1% drawdown
    if drawdown_pct < 1.0:
        return 1.0
        
    # Progressive scaling between 1% and 10% drawdown
    elif drawdown_pct < 10.0:
        # Linear reduction from 100% to 25% as drawdown increases from 1% to 10%
        scale_factor = 1.0 - ((drawdown_pct - 1.0) / 9.0 * 0.75)
        return max(0.25, min(1.0, scale_factor))  # Ensure between 0.25 and 1.0
        
    # Minimal position sizes beyond 10% drawdown
    else:
        return 0.25  # Maximum 25% of normal position size

def apply_drawdown_protection(portfolio_value: float, highest_value: float,
                            position_limit: float) -> float:
    """
    Apply progressive drawdown protection to position limits.
    
    Args:
        portfolio_value: Current portfolio value
        highest_value: Highest portfolio value reached
        position_limit: Original position limit
        
    Returns:
        Adjusted position limit after drawdown protection
    """
    # Calculate drawdown percentage
    if highest_value <= 0:
        return position_limit
        
    drawdown_pct = (1 - portfolio_value / highest_value) * 100
    
    # Get the scale factor based on drawdown
    scale_factor = calculate_position_scale_factor(drawdown_pct)
    
    # Apply scaling to position limit
    adjusted_limit = position_limit * scale_factor
    
    # Log if significant scaling is applied
    if scale_factor < 0.9:
        logger.info(f"Drawdown protection: {drawdown_pct:.2f}% drawdown detected. Reducing position limits to {scale_factor:.2f}x")
    
    return adjusted_limit

def should_halt_trading(drawdown_pct: float, threshold: float = 15.0) -> bool:
    """
    Determine if trading should be halted due to extreme drawdown.
    This is a last-resort protection beyond the progressive scaling.
    
    Args:
        drawdown_pct: Current drawdown percentage (0-100)
        threshold: Drawdown percentage threshold for halting (default 15%)
        
    Returns:
        Boolean indicating if trading should be halted
    """
    if drawdown_pct >= threshold:
        logger.warning(f"EXTREME DRAWDOWN ALERT: {drawdown_pct:.2f}% drawdown exceeds {threshold:.2f}% threshold")
        return True
    return False

# ------------------- LIQUIDITY RISK ASSESSMENT -------------------

def assess_position_liquidity(ticker: str, position_size: float, avg_daily_volume: float,
                            current_price: float, max_market_impact_pct: float = 10.0) -> Dict[str, Any]:
    """
    Evaluate how many days it would take to exit a position assuming we can
    trade up to a specified percentage of average daily volume without significant market impact.
    
    Args:
        ticker: Stock symbol
        position_size: Position size in dollars
        avg_daily_volume: Average daily trading volume in shares
        current_price: Current price per share
        max_market_impact_pct: Maximum percentage of daily volume to trade
        
    Returns:
        Dictionary with liquidity risk assessment
    """
    if avg_daily_volume <= 0 or current_price <= 0:
        return {
            "risk_level": "Unknown",
            "days_to_exit": None,
            "reason": "Insufficient volume or price data"
        }
    
    # Calculate shares in position
    shares = position_size / current_price
    
    # Calculate daily tradeable volume
    daily_tradeable_shares = avg_daily_volume * (max_market_impact_pct / 100.0)
    
    # Calculate days to exit
    days_to_exit = shares / daily_tradeable_shares
    
    # Determine risk level
    if days_to_exit <= 1:
        risk_level = "Low"
        reason = f"Position can be exited in {days_to_exit:.1f} days (trading {max_market_impact_pct}% of avg volume)"
    elif days_to_exit <= 3:
        risk_level = "Moderate"
        reason = f"Position would take {days_to_exit:.1f} days to exit (trading {max_market_impact_pct}% of avg volume)"
    elif days_to_exit <= 5:
        risk_level = "High"
        reason = f"Position would take {days_to_exit:.1f} days to exit - consider reducing size"
    else:
        risk_level = "Extreme"
        reason = f"Position would take {days_to_exit:.1f} days to exit - significant liquidity risk"
    
    return {
        "risk_level": risk_level,
        "days_to_exit": days_to_exit,
        "max_market_impact_pct": max_market_impact_pct,
        "shares": shares,
        "avg_daily_volume": avg_daily_volume,
        "reason": reason
    }

def get_avg_daily_volume(ticker: str, lookback_days: int = 20) -> float:
    """
    Get the average daily trading volume for a ticker over the specified lookback period.
    
    Args:
        ticker: Stock symbol
        lookback_days: Number of trading days to look back
        
    Returns:
        Average daily volume in shares
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for volume data retrieval")
        return 0.0
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 2)  # Add buffer for weekends/holidays
        
        # Get price data including volume
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Get the last N trading days
        recent_data = data.tail(lookback_days)
        
        # Calculate average volume
        avg_volume = recent_data['Volume'].mean()
        
        return avg_volume
        
    except Exception as e:
        logger.error(f"Error getting average daily volume for {ticker}: {e}")
        return 0.0

def assess_portfolio_liquidity(portfolio: Dict, max_market_impact_pct: float = 10.0) -> Dict[str, Any]:
    """
    Assess the liquidity risk of the entire portfolio.
    
    Args:
        portfolio: Portfolio dictionary containing positions
        max_market_impact_pct: Maximum percentage of daily volume to trade
        
    Returns:
        Dictionary with portfolio liquidity assessment
    """
    positions = portfolio.get('positions', {})
    
    # Skip if no positions
    if not positions:
        return {
            "portfolio_liquidity_risk": "None",
            "liquidity_by_position": {},
            "reason": "No positions in portfolio"
        }
    
    # Assess liquidity for each position
    liquidity_by_position = {}
    high_risk_positions = []
    extreme_risk_positions = []
    
    for ticker, position in positions.items():
        # Calculate position value
        long_value = position.get('long', 0) * position.get('long_cost_basis', 0)
        short_value = position.get('short', 0) * position.get('short_cost_basis', 0)
        total_value = long_value + short_value
        
        if total_value <= 0:
            continue
        
        # Get average daily volume and current price
        avg_volume = get_avg_daily_volume(ticker)
        current_price = position.get('long_cost_basis') if position.get('long', 0) > 0 else position.get('short_cost_basis')
        
        if avg_volume <= 0 or current_price <= 0:
            continue
        
        # Assess liquidity
        liquidity_assessment = assess_position_liquidity(
            ticker, total_value, avg_volume, current_price, max_market_impact_pct
        )
        
        liquidity_by_position[ticker] = liquidity_assessment
        
        # Track high/extreme risk positions
        if liquidity_assessment["risk_level"] == "High":
            high_risk_positions.append(ticker)
        elif liquidity_assessment["risk_level"] == "Extreme":
            extreme_risk_positions.append(ticker)
    
    # Determine overall portfolio liquidity risk
    if len(extreme_risk_positions) > 0:
        portfolio_risk = "Extreme"
        reason = f"Portfolio contains positions with extreme liquidity risk: {', '.join(extreme_risk_positions)}"
    elif len(high_risk_positions) > 0:
        portfolio_risk = "High"
        reason = f"Portfolio contains positions with high liquidity risk: {', '.join(high_risk_positions)}"
    elif len(liquidity_by_position) > 0:
        portfolio_risk = "Moderate"
        reason = "All positions have acceptable liquidity risk"
    else:
        portfolio_risk = "Unknown"
        reason = "Could not assess portfolio liquidity risk"
    
    return {
        "portfolio_liquidity_risk": portfolio_risk,
        "liquidity_by_position": liquidity_by_position,
        "high_risk_positions": high_risk_positions,
        "extreme_risk_positions": extreme_risk_positions,
        "reason": reason
    }

# ------------------- PORTFOLIO BETA MANAGEMENT -------------------

def calculate_ticker_beta(ticker: str, benchmark: str = "SPY", 
                        lookback_days: int = 252) -> float:
    """
    Calculate beta (market risk) for a given ticker compared to a benchmark.
    
    Args:
        ticker: Stock symbol
        benchmark: Benchmark ticker (default SPY)
        lookback_days: Number of trading days to look back
        
    Returns:
        Beta value or None if calculation fails
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for beta calculation")
        return None
    
    try:
        # Calculate date range - use double the lookback to account for weekends/holidays
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 2)
        
        # Get price data for both ticker and benchmark
        data = yf.download([ticker, benchmark], start=start_date, end=end_date)['Adj Close']
        
        # Need at least 30 data points for meaningful calculation
        if len(data) < 30:
            logger.warning(f"Insufficient data for beta calculation for {ticker}")
            return None
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate beta: covariance(stock, market) / variance(market)
        covariance = returns.cov().iloc[0, 1]
        market_variance = returns[benchmark].var()
        
        if market_variance == 0:
            return None
            
        beta = covariance / market_variance
        
        return beta
        
    except Exception as e:
        logger.error(f"Error calculating beta for {ticker}: {e}")
        return None

def calculate_portfolio_beta(portfolio: Dict, benchmark: str = "SPY") -> Dict[str, Any]:
    """
    Calculate the overall portfolio beta to manage market exposure.
    
    Args:
        portfolio: Portfolio dictionary containing positions
        benchmark: Benchmark ticker (default SPY)
        
    Returns:
        Dictionary with portfolio beta information
    """
    positions = portfolio.get('positions', {})
    
    # Skip if no positions
    if not positions:
        return {
            "portfolio_beta": 0.0,
            "beta_by_position": {},
            "beta_weighted_by_value": {},
            "reason": "No positions in portfolio"
        }
    
    # Calculate total portfolio value
    total_portfolio_value = portfolio.get('cash', 0)
    position_values = {}
    
    for ticker, position in positions.items():
        # Calculate position value
        long_value = position.get('long', 0) * position.get('long_cost_basis', 0)
        short_value = position.get('short', 0) * position.get('short_cost_basis', 0)
        position_value = long_value + short_value
        
        if position_value > 0:
            position_values[ticker] = position_value
            total_portfolio_value += position_value
    
    # Skip further calculation if portfolio is empty
    if total_portfolio_value == 0 or not position_values:
        return {
            "portfolio_beta": 0.0,
            "beta_by_position": {},
            "beta_weighted_by_value": {},
            "reason": "Portfolio has no value or positions"
        }
    
    # Calculate beta for each position
    beta_by_position = {}
    beta_weighted_by_value = {}
    
    for ticker, position_value in position_values.items():
        # Get position details
        position = positions[ticker]
        is_short = position.get('short', 0) > 0
        
        # Calculate weight in portfolio
        weight = position_value / total_portfolio_value
        
        # Calculate beta
        beta = calculate_ticker_beta(ticker, benchmark)
        
        # For short positions, beta is negative
        if is_short and beta is not None:
            beta = -beta
        
        if beta is not None:
            beta_by_position[ticker] = beta
            beta_weighted_by_value[ticker] = beta * weight
    
    # Calculate portfolio beta
    portfolio_beta = sum(beta_weighted_by_value.values()) if beta_weighted_by_value else 0.0
    
    # Interpret the beta value
    if portfolio_beta > 1.2:
        beta_interpretation = "High market risk - consider reducing exposure"
    elif portfolio_beta > 0.8:
        beta_interpretation = "Market-like risk profile"
    elif portfolio_beta > 0.0:
        beta_interpretation = "Lower market risk than benchmark"
    elif portfolio_beta < -0.2:
        beta_interpretation = "Net short the market - negative market exposure"
    else:
        beta_interpretation = "Near market-neutral portfolio"
    
    return {
        "portfolio_beta": portfolio_beta,
        "beta_by_position": beta_by_position,
        "beta_weighted_by_value": beta_weighted_by_value,
        "beta_interpretation": beta_interpretation,
        "benchmark": benchmark,
        "reason": f"Portfolio beta: {portfolio_beta:.2f} - {beta_interpretation}"
    }

# ------------------- TIME-BASED RISK RULES -------------------

def get_next_earnings_date(ticker: str) -> Optional[date]:
    """
    Get the next earnings announcement date for a ticker.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Next earnings date or None if not available
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for earnings date retrieval")
        return None
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get next earnings date
        calendar = stock.calendar
        if calendar is not None and hasattr(calendar, 'loc') and 'Earnings Date' in calendar.columns:
            next_earnings = calendar.loc[0, 'Earnings Date']
            if isinstance(next_earnings, pd.Timestamp):
                return next_earnings.date()
        
        # If calendar method doesn't work, try earnings_dates
        earnings_dates = stock.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            # Get the first future earnings date
            future_earnings = earnings_dates[earnings_dates.index > pd.Timestamp.now()]
            if not future_earnings.empty:
                return future_earnings.index[0].date()
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting next earnings date for {ticker}: {e}")
        return None

def check_market_hours() -> Dict[str, Any]:
    """
    Check if market is currently open and get information about market hours.
    
    Returns:
        Dictionary with market hours information
    """
    now = datetime.now()
    today = now.date()
    weekday = now.weekday()
    
    # Check if it's a weekend
    if weekday >= 5:  # 5 = Saturday, 6 = Sunday
        return {
            "is_open": False,
            "status": "Weekend",
            "reason": f"Market is closed on {'Saturday' if weekday == 5 else 'Sunday'}",
            "reduced_risk": False
        }
    
    # Check regular market hours (9:30 AM - 4:00 PM ET)
    # Note: This is a simplified check and doesn't account for holidays
    market_open = datetime(today.year, today.month, today.day, 9, 30, 0)
    market_close = datetime(today.year, today.month, today.day, 16, 0, 0)
    
    # Check for pre-market hours (4:00 AM - 9:30 AM ET)
    pre_market_open = datetime(today.year, today.month, today.day, 4, 0, 0)
    
    # Check for after-hours (4:00 PM - 8:00 PM ET)
    after_hours_close = datetime(today.year, today.month, today.day, 20, 0, 0)
    
    if now < pre_market_open or now > after_hours_close:
        return {
            "is_open": False,
            "status": "Closed",
            "reason": "Outside of all trading hours",
            "reduced_risk": False
        }
    elif now < market_open:
        return {
            "is_open": True,
            "status": "Pre-market",
            "reason": f"Pre-market trading ({pre_market_open.strftime('%H:%M')} - {market_open.strftime('%H:%M')} ET)",
            "reduced_risk": True  # Reduce risk during pre-market due to lower liquidity
        }
    elif now <= market_close:
        # Regular market hours - check for end-of-day risk reduction
        minutes_to_close = (market_close - now).total_seconds() / 60
        
        if minutes_to_close <= 30:
            return {
                "is_open": True,
                "status": "Regular - Closing Soon",
                "reason": f"Market closing in {minutes_to_close:.0f} minutes",
                "reduced_risk": True  # Reduce risk in last 30 minutes of trading
            }
        else:
            return {
                "is_open": True,
                "status": "Regular",
                "reason": "Regular market hours",
                "reduced_risk": False
            }
    else:
        return {
            "is_open": True,
            "status": "After-hours",
            "reason": f"After-hours trading ({market_close.strftime('%H:%M')} - {after_hours_close.strftime('%H:%M')} ET)",
            "reduced_risk": True  # Reduce risk during after-hours due to lower liquidity
        }

def adjust_risk_for_timing(ticker: str, risk_params: Dict) -> Dict:
    """
    Adjust risk parameters based on timing factors including:
    - Proximity to earnings announcements
    - End of day/week (reduce risk before weekends)
    - Market hours (reduce risk during pre/post market)
    
    Args:
        ticker: Stock symbol
        risk_params: Original risk parameters
        
    Returns:
        Adjusted risk parameters
    """
    # Create a copy of risk params to avoid modifying the original
    adjusted_params = risk_params.copy()
    
    # Track reasons for adjustments
    adjustment_reasons = []
    
    # Check if it's Friday afternoon (reduce risk before weekend)
    now = datetime.now()
    is_friday = now.weekday() == 4  # 4 = Friday
    is_afternoon = now.hour >= 14  # After 2 PM
    
    if is_friday and is_afternoon:
        # Reduce position size by 30% before weekend
        weekend_factor = 0.7
        adjusted_params["MAX_POSITION_SIZE_PCT"] *= weekend_factor
        adjustment_reasons.append(f"Friday afternoon (reduced by {(1-weekend_factor)*100:.0f}%)")
    
    # Check market hours
    market_hours = check_market_hours()
    if market_hours["reduced_risk"]:
        # Reduce position size by 40% during non-regular hours
        hours_factor = 0.6
        adjusted_params["MAX_POSITION_SIZE_PCT"] *= hours_factor
        adjustment_reasons.append(f"{market_hours['status']} (reduced by {(1-hours_factor)*100:.0f}%)")
    
    # Check earnings proximity
    next_earnings = get_next_earnings_date(ticker)
    if next_earnings:
        days_to_earnings = (next_earnings - datetime.now().date()).days
        
        # Adjust risk based on earnings proximity
        if 0 <= days_to_earnings <= 1:  # Day of or day before earnings
            earnings_factor = 0.25  # Significant reduction
            adjusted_params["MAX_POSITION_SIZE_PCT"] *= earnings_factor
            adjustment_reasons.append(f"Earnings imminent ({days_to_earnings} days away, reduced by {(1-earnings_factor)*100:.0f}%)")
        elif 2 <= days_to_earnings <= 5:  # In earnings week
            earnings_factor = 0.5  # Moderate reduction
            adjusted_params["MAX_POSITION_SIZE_PCT"] *= earnings_factor
            adjustment_reasons.append(f"Earnings approaching ({days_to_earnings} days away, reduced by {(1-earnings_factor)*100:.0f}%)")
        elif 6 <= days_to_earnings <= 10:  # Near earnings
            earnings_factor = 0.75  # Slight reduction
            adjusted_params["MAX_POSITION_SIZE_PCT"] *= earnings_factor
            adjustment_reasons.append(f"Earnings nearby ({days_to_earnings} days away, reduced by {(1-earnings_factor)*100:.0f}%)")
    
    # Add adjustment reasons to the parameters for reference
    adjusted_params["adjustment_reasons"] = adjustment_reasons
    
    # Log significant adjustments
    if adjustment_reasons:
        original_size = risk_params.get("MAX_POSITION_SIZE_PCT", 0)
        adjusted_size = adjusted_params.get("MAX_POSITION_SIZE_PCT", 0)
        reduction_pct = (1 - adjusted_size / original_size) * 100 if original_size > 0 else 0
        
        if reduction_pct > 20:  # Only log significant adjustments
            logger.info(f"Time-based risk adjustment for {ticker}: -{reduction_pct:.1f}% due to {', '.join(adjustment_reasons)}")
    
    return adjusted_params

# ------------------- VOLATILITY-BASED TRAILING STOPS -------------------

def calculate_atr(prices_df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) for volatility-based position management.
    
    Args:
        prices_df: DataFrame with price data
        period: Period for ATR calculation
        
    Returns:
        ATR value
    """
    try:
        # Create a copy to avoid modifying the original
        df = prices_df.copy()
        
        # Calculate True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate Average True Range
        df['atr'] = df['true_range'].rolling(window=period).mean()
        
        # Get the most recent ATR value
        atr = df['atr'].iloc[-1] if not df.empty and not df['atr'].isna().iloc[-1] else None
        
        return atr
    
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return None

def calculate_dynamic_stop_loss(entry_price: float, current_price: float, atr: float,
                              atr_multiplier: float = 2.5, min_stop_pct: float = 0.03) -> Dict[str, float]:
    """
    Calculate dynamic stop loss levels based on Average True Range (ATR).
    
    Args:
        entry_price: Original entry price
        current_price: Current price
        atr: Average True Range value
        atr_multiplier: Multiplier for ATR to set stop distance
        min_stop_pct: Minimum stop distance as percentage of price
        
    Returns:
        Dictionary with stop loss information
    """
    if atr is None or atr <= 0 or entry_price <= 0 or current_price <= 0:
        # Default to fixed percentage stop if ATR calculation fails
        stop_price = entry_price * 0.95  # 5% below entry price
        return {
            "stop_price": stop_price,
            "stop_distance_pct": 5.0,
            "stop_type": "fixed",
            "reason": "Using fixed 5% stop due to insufficient data for ATR calculation"
        }
    
    # For long positions
    if current_price > entry_price:  # Position in profit
        # ATR-based stop
        atr_stop = current_price - (atr * atr_multiplier)
        
        # Minimum stop based on entry price
        min_stop = entry_price * (1 - min_stop_pct)
        
        # Use the higher of the two stops
        stop_price = max(atr_stop, min_stop)
        
        # Calculate distance as percentage
        stop_distance_pct = (current_price - stop_price) / current_price * 100
        
        stop_type = "trailing_atr"
        reason = f"Trailing ATR-based stop ({atr_multiplier}x ATR = ${atr * atr_multiplier:.2f}), {stop_distance_pct:.1f}% below current price"
        
    else:  # Position in loss or breakeven
        # Fixed percentage stop from entry
        stop_price = entry_price * 0.95  # 5% below entry price
        stop_distance_pct = 5.0
        stop_type = "fixed"
        reason = f"Fixed 5% stop below entry price (${entry_price:.2f})"
    
    return {
        "stop_price": stop_price,
        "stop_distance_pct": stop_distance_pct,
        "stop_type": stop_type,
        "atr": atr,
        "atr_multiplier": atr_multiplier,
        "min_stop_pct": min_stop_pct,
        "reason": reason
    }

def calculate_dynamic_take_profit(entry_price: float, atr: float,
                               reward_risk_ratio: float = 3.0) -> Dict[str, float]:
    """
    Calculate dynamic take profit levels based on ATR and desired reward-to-risk ratio.
    
    Args:
        entry_price: Original entry price
        atr: Average True Range value
        reward_risk_ratio: Desired reward-to-risk ratio
        
    Returns:
        Dictionary with take profit information
    """
    if atr is None or atr <= 0 or entry_price <= 0:
        # Default to fixed percentage take profit if ATR calculation fails
        take_profit = entry_price * 1.2  # 20% above entry price
        return {
            "take_profit_price": take_profit,
            "take_profit_distance_pct": 20.0,
            "take_profit_type": "fixed",
            "reason": "Using fixed 20% take profit due to insufficient data for ATR calculation"
        }
    
    # Calculate stop loss distance in points
    stop_loss_distance = atr * 2.5
    
    # Calculate take profit distance based on reward-risk ratio
    take_profit_distance = stop_loss_distance * reward_risk_ratio
    
    # Calculate take profit price
    take_profit_price = entry_price + take_profit_distance
    
    # Calculate as percentage
    take_profit_distance_pct = (take_profit_price - entry_price) / entry_price * 100
    
    return {
        "take_profit_price": take_profit_price,
        "take_profit_distance_pct": take_profit_distance_pct,
        "take_profit_type": "atr",
        "atr": atr,
        "reward_risk_ratio": reward_risk_ratio,
        "reason": f"ATR-based take profit (R:R = {reward_risk_ratio}:1), {take_profit_distance_pct:.1f}% above entry price"
    }

def calculate_position_risk_parameters(ticker: str, entry_price: float, current_price: float, 
                                     prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive risk parameters for a position, including:
    - Dynamic ATR-based stops
    - Take profit levels
    - Position sizing recommendations
    
    Args:
        ticker: Stock symbol
        entry_price: Original entry price
        current_price: Current price
        prices_df: DataFrame with price data
        
    Returns:
        Dictionary with position risk parameters
    """
    # Calculate ATR
    atr = calculate_atr(prices_df)
    
    # Calculate dynamic stop loss
    stop_loss = calculate_dynamic_stop_loss(entry_price, current_price, atr)
    
    # Calculate take profit
    take_profit = calculate_dynamic_take_profit(entry_price, atr)
    
    # Calculate current R-multiple (how many risk units is the current profit/loss)
    if stop_loss["stop_type"] == "fixed":
        risk_per_share = entry_price * 0.05  # 5% risk
    else:
        risk_per_share = entry_price - stop_loss["stop_price"]
    
    current_pl = current_price - entry_price
    r_multiple = current_pl / risk_per_share if risk_per_share > 0 else 0
    
    # Determine if position should be scaled out partially
    should_scale_out = False
    scale_out_reason = None
    
    if r_multiple >= 2.0:
        should_scale_out = True
        scale_out_reason = f"Position has achieved {r_multiple:.1f}R profit - consider scaling out 1/3"
    
    # Enhance with volatility-based trailing stop
    result = {
        "ticker": ticker,
        "entry_price": entry_price,
        "current_price": current_price,
        "atr": atr,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "r_multiple": r_multiple,
        "should_scale_out": should_scale_out,
        "scale_out_reason": scale_out_reason
    }
    
    return result

# ------------------- RISK PARITY & PORTFOLIO OPTIMIZATION -------------------

def allocate_risk_budget(tickers: List[str], volatilities: List[float], 
                       total_risk_budget: float = 1.0) -> Dict[str, float]:
    """
    Allocate position sizes using risk parity approach - more volatile
    assets get smaller position sizes to equalize risk contribution.
    
    Args:
        tickers: List of ticker symbols
        volatilities: List of volatility values corresponding to tickers
        total_risk_budget: Total risk budget to allocate (1.0 = 100%)
        
    Returns:
        Dictionary mapping tickers to allocation percentages
    """
    if len(tickers) != len(volatilities):
        logger.error(f"Number of tickers ({len(tickers)}) does not match number of volatilities ({len(volatilities)})")
        return {}
    
    # Filter out invalid volatilities
    valid_data = [(ticker, vol) for ticker, vol in zip(tickers, volatilities) if vol is not None and vol > 0]
    
    if not valid_data:
        return {}
    
    valid_tickers, valid_vols = zip(*valid_data)
    
    # Calculate inverse of volatility
    inv_vol = [1/vol for vol in valid_vols]
    total_inv_vol = sum(inv_vol)
    
    # Allocate budget proportional to inverse volatility
    allocations = {}
    for i, ticker in enumerate(valid_tickers):
        allocations[ticker] = (inv_vol[i] / total_inv_vol) * total_risk_budget
    
    return allocations

def calculate_tickers_volatility(tickers: List[str], lookback_days: int = 63) -> Dict[str, float]:
    """
    Calculate annualized volatility for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        lookback_days: Number of trading days to use for calculation
        
    Returns:
        Dictionary mapping tickers to volatility values
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for volatility calculation")
        return {}
    
    volatilities = {}
    
    for ticker in tickers:
        try:
            # Calculate date range with buffer for weekends/holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)
            
            # Get price data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Need at least 30 data points for meaningful calculation
            if len(data) < 30:
                logger.warning(f"Insufficient data for volatility calculation for {ticker}")
                continue
            
            # Calculate daily returns
            returns = data['Adj Close'].pct_change().dropna()
            
            # Use only the requested lookback period
            recent_returns = returns.tail(lookback_days)
            
            # Calculate annualized volatility
            daily_vol = recent_returns.std()
            annualized_vol = daily_vol * np.sqrt(252)  # Assuming 252 trading days per year
            
            volatilities[ticker] = annualized_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {ticker}: {e}")
    
    return volatilities

def optimize_position_sizing(portfolio: Dict, risk_limit_pct: float = 10.0) -> Dict[str, Dict[str, Any]]:
    """
    Optimize position sizing based on volatility and risk parity principles.
    
    Args:
        portfolio: Portfolio dictionary containing positions
        risk_limit_pct: Maximum portfolio percentage for any position
        
    Returns:
        Dictionary with optimized position sizes
    """
    positions = portfolio.get('positions', {})
    
    # Get tickers with positions
    tickers = list(positions.keys())
    
    if not tickers:
        return {}
    
    # Calculate volatilities
    volatilities = calculate_tickers_volatility(tickers)
    
    # Get risk parity allocations
    risk_parity_allocations = allocate_risk_budget(
        list(volatilities.keys()), 
        [volatilities.get(ticker, 0) for ticker in volatilities.keys()]
    )
    
    # Calculate total portfolio value
    total_portfolio_value = portfolio.get('cash', 0)
    for ticker, position in positions.items():
        long_value = position.get('long', 0) * position.get('long_cost_basis', 0)
        short_value = position.get('short', 0) * position.get('short_cost_basis', 0)
        total_portfolio_value += long_value + short_value
    
    # Calculate dollar allocations and compare to current
    optimized_positions = {}
    
    for ticker, allocation_pct in risk_parity_allocations.items():
        current_position = positions.get(ticker, {})
        
        # Calculate current position value
        current_long = current_position.get('long', 0)
        current_short = current_position.get('short', 0)
        long_value = current_long * current_position.get('long_cost_basis', 0)
        short_value = current_short * current_position.get('short_cost_basis', 0)
        current_value = long_value + short_value
        
        # Calculate target dollar allocation
        target_pct = min(allocation_pct, risk_limit_pct / 100)  # Cap at risk limit
        target_value = total_portfolio_value * target_pct
        
        # Calculate difference
        value_difference = target_value - current_value
        pct_difference = (value_difference / current_value * 100) if current_value > 0 else 100
        
        # Get current price for share calculations
        current_price = current_position.get('long_cost_basis', 0)
        if current_price <= 0:
            current_price = current_position.get('short_cost_basis', 0)
        
        # Calculate shares based on target allocation
        target_shares = int(target_value / current_price) if current_price > 0 else 0
        
        # Determine action needed
        if current_value == 0:
            action = "open new position" if target_value > 0 else "none"
        elif value_difference > 0:
            action = "increase position"
        elif value_difference < 0:
            action = "decrease position"
        else:
            action = "maintain position"
        
        # Add to results
        optimized_positions[ticker] = {
            "current_value": current_value,
            "current_pct": (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0,
            "target_value": target_value,
            "target_pct": target_pct * 100,
            "value_difference": value_difference,
            "pct_difference": pct_difference,
            "current_shares": current_long - current_short,  # Net position
            "target_shares": target_shares,
            "shares_difference": target_shares - (current_long - current_short),
            "action": action,
            "volatility": volatilities.get(ticker, 0)
        }
    
    return optimized_positions

# ------------------- UNIFIED RISK DASHBOARD -------------------

def generate_risk_dashboard(portfolio: Dict, tickers: List[str],
                          prices_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Generate a comprehensive risk dashboard for the portfolio,
    combining all risk metrics in one place.
    
    Args:
        portfolio: Portfolio dictionary containing positions
        tickers: List of ticker symbols
        prices_data: Dictionary mapping tickers to price DataFrames
        
    Returns:
        Dictionary with comprehensive risk assessment
    """
    # Market regime assessment
    market_data = prices_data.get("SPY")
    market_regime = detect_market_regime(market_data)
    
    # Sector exposure
    sector_exposures = track_sector_exposure(portfolio)
    over_exposed_sectors = check_sector_limits(sector_exposures)
    
    # Correlation risk
    correlation_risks = check_correlation_risk(portfolio)
    
    # Portfolio beta
    portfolio_beta = calculate_portfolio_beta(portfolio)
    
    # Liquidity risk
    liquidity_risk = assess_portfolio_liquidity(portfolio)
    
    # Position sizing optimization
    position_sizing = optimize_position_sizing(portfolio)
    
    # Calculate portfolio diversity score (0-100)
    diversity_score = calculate_portfolio_diversity_score(
        sector_exposures, correlation_risks, portfolio_beta
    )
    
    # Calculate portfolio robustness score (0-100)
    robustness_score = calculate_portfolio_robustness_score(
        market_regime, liquidity_risk, diversity_score
    )
    
    # Calculate time-based risk adjustment
    time_risk_adjustment = 1.0
    for ticker in tickers:
        adjusted_params = adjust_risk_for_timing(ticker, {"MAX_POSITION_SIZE_PCT": 1.0})
        time_risk_adjustment = min(time_risk_adjustment, adjusted_params.get("MAX_POSITION_SIZE_PCT", 1.0))
    
    # Individual position risks
    position_risks = {}
    for ticker in tickers:
        if ticker not in prices_data:
            continue
            
        position = portfolio.get("positions", {}).get(ticker, {})
        
        # Skip if no position
        if position.get("long", 0) == 0 and position.get("short", 0) == 0:
            continue
            
        entry_price = position.get("long_cost_basis", 0) if position.get("long", 0) > 0 else position.get("short_cost_basis", 0)
        current_price = prices_data[ticker]["close"].iloc[-1] if not prices_data[ticker].empty else entry_price
        
        # Calculate position risk parameters
        risk_params = calculate_position_risk_parameters(
            ticker, entry_price, current_price, prices_data[ticker]
        )
        
        position_risks[ticker] = risk_params
    
    # Combine all risk metrics into dashboard
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "market_regime": market_regime,
        "sector_exposures": sector_exposures,
        "over_exposed_sectors": over_exposed_sectors,
        "correlation_risks": correlation_risks,
        "portfolio_beta": portfolio_beta.get("portfolio_beta"),
        "beta_interpretation": portfolio_beta.get("beta_interpretation"),
        "liquidity_risk": liquidity_risk.get("portfolio_liquidity_risk"),
        "high_liquidity_risk_positions": liquidity_risk.get("high_risk_positions", []),
        "diversity_score": diversity_score,
        "robustness_score": robustness_score,
        "time_risk_adjustment": time_risk_adjustment,
        "position_risks": position_risks,
        "position_sizing": position_sizing
    }
    
    # Save dashboard for future reference
    save_risk_dashboard(dashboard)
    
    return dashboard

def calculate_portfolio_diversity_score(sector_exposures: Dict[str, float],
                                      correlation_risks: List[Dict],
                                      portfolio_beta: Dict) -> float:
    """
    Calculate a portfolio diversity score based on sector exposures,
    correlation risks, and portfolio beta.
    
    Args:
        sector_exposures: Dictionary of sector exposures
        correlation_risks: List of correlation risk information
        portfolio_beta: Portfolio beta information
        
    Returns:
        Diversity score from 0-100
    """
    # Score based on number of sectors with significant allocation (>5%)
    sector_count = sum(1 for sector, exposure in sector_exposures.items() 
                     if sector != "Cash" and exposure > 5.0)
    
    sector_score = min(100, sector_count * 20)  # 20 points per sector, max 100
    
    # Penalty for over-concentrated sectors
    max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0
    if max_sector_exposure > 25:
        sector_score -= (max_sector_exposure - 25) * 2  # 2 point penalty per % over 25%
    
    # Penalty for high correlations
    correlation_penalty = len(correlation_risks) * 10  # 10 point penalty per high correlation pair
    
    # Beta neutrality bonus (higher score for beta closer to 0)
    beta = abs(portfolio_beta.get("portfolio_beta", 1.0))
    beta_score = max(0, 100 - beta * 40)  # 100 points at beta=0, 60 points at beta=1, 0 points at beta=2.5+
    
    # Combine scores with weights
    diversity_score = (
        sector_score * 0.4 +
        max(0, 100 - correlation_penalty) * 0.4 +
        beta_score * 0.2
    )
    
    return max(0, min(100, diversity_score))

def calculate_portfolio_robustness_score(market_regime: Dict,
                                       liquidity_risk: Dict,
                                       diversity_score: float) -> float:
    """
    Calculate a portfolio robustness score based on market regime,
    liquidity risk, and diversity.
    
    Args:
        market_regime: Market regime information
        liquidity_risk: Portfolio liquidity risk information
        diversity_score: Portfolio diversity score
        
    Returns:
        Robustness score from 0-100
    """
    # Base score from diversity
    base_score = diversity_score
    
    # Adjustment based on market regime
    regime_adjustment = 0
    regime = market_regime.get("regime", "Unknown")
    
    if regime in ["Strong Bull", "Bull"]:
        regime_adjustment = 10  # Bonus in favorable regimes
    elif regime in ["Weakening Bull", "Correction"]:
        regime_adjustment = 0  # Neutral
    elif regime in ["Early Bear", "Bear"]:
        regime_adjustment = -20  # Penalty in unfavorable regimes
    elif regime == "Crisis":
        regime_adjustment = -40  # Strong penalty in crisis
    
    # Adjustment based on liquidity risk
    liquidity_adjustment = 0
    liquidity_risk_level = liquidity_risk.get("portfolio_liquidity_risk", "Unknown")
    
    if liquidity_risk_level == "Low":
        liquidity_adjustment = 10
    elif liquidity_risk_level == "Moderate":
        liquidity_adjustment = 0
    elif liquidity_risk_level == "High":
        liquidity_adjustment = -15
    elif liquidity_risk_level == "Extreme":
        liquidity_adjustment = -30
    
    # Calculate final score
    robustness_score = base_score + regime_adjustment + liquidity_adjustment
    
    return max(0, min(100, robustness_score))

def save_risk_dashboard(dashboard: Dict) -> None:
    """Save risk dashboard for tracking over time"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(RISK_DATA_DIR, 'dashboards'), exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(RISK_DATA_DIR, 'dashboards', f'dashboard_{timestamp}.json')
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(dashboard, f, indent=2)
            
        # Also save as latest
        latest_path = os.path.join(RISK_DATA_DIR, 'dashboard_latest.json')
        with open(latest_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
            
        logger.info(f"Saved risk dashboard to {filepath}")
    except Exception as e:
        logger.error(f"Error saving risk dashboard: {e}")

def get_latest_risk_dashboard() -> Dict:
    """Get the most recent risk dashboard"""
    try:
        latest_path = os.path.join(RISK_DATA_DIR, 'dashboard_latest.json')
        if not os.path.exists(latest_path):
            return {}
            
        with open(latest_path, 'r') as f:
            dashboard = json.load(f)
            
        return dashboard
    except Exception as e:
        logger.error(f"Error loading latest risk dashboard: {e}")
        return {}

# ------------------- INTEGRATION WITH EXISTING RISK MANAGER -------------------

def enhance_risk_management(ticker: str, position_limit: float, portfolio: Dict,
                         market_data: pd.DataFrame, prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhance risk management by applying all the advanced risk techniques.
    This function is designed to be called from the existing risk_manager.py.
    
    Args:
        ticker: Stock symbol
        position_limit: Original position limit
        portfolio: Portfolio dictionary
        market_data: DataFrame with market data (SPY)
        prices_df: DataFrame with ticker's price data
        
    Returns:
        Dictionary with enhanced risk information
    """
    try:
        # Step 1: Detect market regime
        market_regime = detect_market_regime(market_data)
        
        # Step 2: Apply market regime adjustment to position limit
        regime_adjusted_limit = position_limit * market_regime["risk_adjustment"]
        
        # Step 3: Apply time-based risk adjustment
        risk_params = {"MAX_POSITION_SIZE_PCT": 1.0}  # Dummy params for time adjustment
        time_adjusted_params = adjust_risk_for_timing(ticker, risk_params)
        time_adjusted_limit = regime_adjusted_limit * time_adjusted_params["MAX_POSITION_SIZE_PCT"]
        
        # Step 4: Apply drawdown protection if applicable
        portfolio_value = portfolio.get('cash', 0)
        highest_value = 0
        
        for pos_ticker, position in portfolio.get('positions', {}).items():
            long_value = position.get('long', 0) * position.get('long_cost_basis', 0)
            short_value = position.get('short', 0) * position.get('short_cost_basis', 0)
            portfolio_value += long_value + short_value
        
        # Use the highest value from our global tracking
        from utils.risk_manager import TODAY_STATE
        highest_value = TODAY_STATE.get("highest_portfolio_value", portfolio_value)
        
        drawdown_adjusted_limit = apply_drawdown_protection(
            portfolio_value, highest_value, time_adjusted_limit
        )
        
        # Step 5: Apply liquidity constraints
        liquidity_assessment = None
        current_price = prices_df['close'].iloc[-1] if not prices_df.empty else 0
        
        if current_price > 0:
            avg_volume = get_avg_daily_volume(ticker)
            position_size = drawdown_adjusted_limit
            
            liquidity_assessment = assess_position_liquidity(
                ticker, position_size, avg_volume, current_price
            )
            
            # Further reduce limit if extreme liquidity risk
            if liquidity_assessment["risk_level"] == "Extreme":
                drawdown_adjusted_limit *= 0.5
            elif liquidity_assessment["risk_level"] == "High":
                drawdown_adjusted_limit *= 0.75
        
        # Step 6: Calculate volatility-based stops for existing position
        position = portfolio.get('positions', {}).get(ticker, {})
        entry_price = position.get('long_cost_basis', 0) if position.get('long', 0) > 0 else position.get('short_cost_basis', 0)
        
        # Only calculate if we have an existing position
        position_risk_params = None
        if entry_price > 0 and current_price > 0:
            position_risk_params = calculate_position_risk_parameters(
                ticker, entry_price, current_price, prices_df
            )
        
        # Compile all enhancements into result
        result = {
            "ticker": ticker,
            "original_limit": position_limit,
            "enhanced_limit": drawdown_adjusted_limit,
            "market_regime": market_regime,
            "regime_adjustment": market_regime["risk_adjustment"],
            "time_adjustment": time_adjusted_params["MAX_POSITION_SIZE_PCT"],
            "adjustment_reasons": time_adjusted_params.get("adjustment_reasons", []),
            "portfolio_value": portfolio_value,
            "highest_value": highest_value,
            "liquidity_assessment": liquidity_assessment,
            "position_risk_params": position_risk_params,
            "halt_trading": should_halt_trading((1 - portfolio_value / highest_value) * 100 if highest_value > 0 else 0)
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in enhanced risk management for {ticker}: {e}")
        return {
            "ticker": ticker,
            "original_limit": position_limit,
            "enhanced_limit": position_limit * 0.5,  # Default to 50% reduction on error
            "error": str(e)
        }
