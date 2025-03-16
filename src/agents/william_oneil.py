import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from typing_extensions import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Local imports
from utils.agent_utils import AgentState, show_agent_reasoning
from utils.progress import progress
from utils.llm import call_llm

# Data utilities
from tools.api import (
    get_prices, 
    prices_to_df, 
    get_financial_metrics, 
    get_company_news,
    get_company_info,
    get_market_cap
)

logger = logging.getLogger(__name__)


class WilliamONeilSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def pull_stock_data(ticker: str, date_range: Dict[str, str]) -> pd.DataFrame:
    """
    Pull historical stock price data for the given ticker and date range.
    
    Args:
        ticker: Stock symbol
        date_range: Dictionary with 'start_date' and 'end_date' keys
        
    Returns:
        DataFrame with historical price data
    """
    try:
        prices = get_prices(ticker=ticker, start_date=date_range['start_date'], end_date=date_range['end_date'])
        if not prices:
            logger.warning(f"No price data found for {ticker}")
            return pd.DataFrame()
            
        prices_df = prices_to_df(prices)
        return prices_df
    except Exception as e:
        logger.error(f"Error pulling stock data for {ticker}: {str(e)}")
        return pd.DataFrame()


def pull_financial_metrics(ticker: str) -> Dict:
    """
    Pull financial metrics for the given ticker.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Dictionary with financial metrics
    """
    try:
        metrics = get_financial_metrics(ticker=ticker)
        if not metrics:
            logger.warning(f"No financial metrics found for {ticker}")
            return {}
            
        return metrics
    except Exception as e:
        logger.error(f"Error pulling financial metrics for {ticker}: {str(e)}")
        return {}


def pull_recent_news(ticker: str) -> List[Dict]:
    """
    Pull recent news for the given ticker.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        List of news items
    """
    try:
        news = get_company_news(ticker=ticker)
        if not news:
            logger.warning(f"No recent news found for {ticker}")
            return []
            
        return news
    except Exception as e:
        logger.error(f"Error pulling recent news for {ticker}: {str(e)}")
        return []


def get_market_cap(ticker: str) -> Optional[float]:
    """
    Get market capitalization for the given ticker.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Market cap as a float, or None if not available
    """
    try:
        info = get_company_info(ticker=ticker)
        if not info or 'marketCap' not in info:
            logger.warning(f"No market cap found for {ticker}")
            return None
            
        return float(info['marketCap'])
    except Exception as e:
        logger.error(f"Error getting market cap for {ticker}: {str(e)}")
        return None


def analyze_current_earnings(metrics: dict) -> dict:
    """
    Analyze 'C' in CANSLIM: Current Quarterly Earnings
    
    O'Neil looks for:
    - Current quarterly earnings up 25% or more compared to the same quarter last year
    - Accelerating earnings growth rates in recent quarters
    
    Returns a rating from 0-100 and analysis details.
    """
    if not metrics or 'quarterly_financials' not in metrics:
        return {
            "rating": None,
            "reason": "Insufficient quarterly financial data available",
            "earnings_growth": None,
            "acceleration": None
        }
    
    try:
        # Extract quarterly EPS data if available
        quarterly_data = metrics.get('quarterly_financials', {})
        quarterly_eps = quarterly_data.get('EPS', [])
        
        if len(quarterly_eps) < 5:  # Need at least 5 quarters for meaningful analysis
            return {
                "rating": None,
                "reason": "Insufficient quarterly EPS history",
                "earnings_growth": None,
                "acceleration": None
            }
        
        # Get the most recent quarter's EPS and the same quarter from last year
        current_quarter_eps = quarterly_eps[0]
        year_ago_quarter_eps = quarterly_eps[4]  # 4 quarters ago
        
        # Calculate year-over-year growth rate
        if year_ago_quarter_eps <= 0:
            # For negative or zero earnings last year, any positive earnings this year is good
            if current_quarter_eps > 0:
                earnings_growth = 100  # Representing significant improvement
            else:
                earnings_growth = (current_quarter_eps - year_ago_quarter_eps) / abs(year_ago_quarter_eps) * 100
        else:
            earnings_growth = ((current_quarter_eps - year_ago_quarter_eps) / year_ago_quarter_eps) * 100
        
        # Check for earnings acceleration by comparing growth rates between quarters
        recent_growth_rates = []
        for i in range(1, 4):  # Look at last 3 quarters of growth
            if i+4 < len(quarterly_eps):
                current = quarterly_eps[i]
                year_ago = quarterly_eps[i+4]
                
                if year_ago <= 0:
                    if current > 0:
                        rate = 100  # Significant improvement
                    else:
                        rate = 0
                else:
                    rate = ((current - year_ago) / year_ago) * 100
                recent_growth_rates.append(rate)
        
        # Determine if growth is accelerating
        acceleration = False
        if len(recent_growth_rates) >= 2:
            # If recent growth rate is higher than previous, it's accelerating
            acceleration = earnings_growth > recent_growth_rates[0]
        
        # Calculate rating based on O'Neil's criteria
        # O'Neil looks for at least 25% EPS growth
        rating = 0
        reason = ""
        
        if earnings_growth >= 75:
            rating = 100
            reason = f"Exceptional quarterly earnings growth: {earnings_growth:.1f}%"
        elif earnings_growth >= 50:
            rating = 90
            reason = f"Very strong quarterly earnings growth: {earnings_growth:.1f}%"
        elif earnings_growth >= 25:
            rating = 80
            reason = f"Strong quarterly earnings growth: {earnings_growth:.1f}%"
        elif earnings_growth >= 15:
            rating = 60
            reason = f"Good quarterly earnings growth: {earnings_growth:.1f}%"
        elif earnings_growth >= 0:
            rating = 40
            reason = f"Positive but modest quarterly earnings growth: {earnings_growth:.1f}%"
        else:
            rating = 20
            reason = f"Declining quarterly earnings: {earnings_growth:.1f}%"
        
        # Bonus for accelerating growth
        if acceleration:
            rating = min(100, rating + 10)
            reason += " with accelerating growth rate"
        
        return {
            "rating": rating,
            "reason": reason,
            "earnings_growth": earnings_growth,
            "acceleration": acceleration
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing current earnings: {str(e)}",
            "earnings_growth": None,
            "acceleration": None
        }


def analyze_annual_earnings(metrics: dict) -> dict:
    """
    Analyze 'A' in CANSLIM: Annual Earnings Growth
    
    O'Neil looks for:
    - Annual EPS growth of at least 25% for each of the last 3 years
    - ROE of at least 17%
    - Stable and increasing profit margins
    
    Returns a rating from 0-100 and analysis details.
    """
    if not metrics or 'annual_financials' not in metrics:
        return {
            "rating": None,
            "reason": "Insufficient annual financial data available",
            "annual_growth": None,
            "consistent_growth": None,
            "roe": None
        }
    
    try:
        # Extract annual financial data
        annual_data = metrics.get('annual_financials', {})
        annual_eps = annual_data.get('EPS', [])
        annual_roe = annual_data.get('ROE', [])
        
        if len(annual_eps) < 3:  # Need at least 3 years for O'Neil's criteria
            return {
                "rating": None,
                "reason": "Insufficient annual EPS history",
                "annual_growth": None,
                "consistent_growth": None,
                "roe": None
            }
        
        # Calculate annual growth rates for last 3 years
        growth_rates = []
        for i in range(len(annual_eps) - 1):
            if i >= 3:  # Only need 3 years of growth rates
                break
                
            current_year = annual_eps[i]
            previous_year = annual_eps[i + 1]
            
            # Handle special cases for negative or zero earnings
            if previous_year <= 0:
                if current_year > 0:
                    rate = 100  # Significant improvement
                else:
                    rate = 0
            else:
                rate = ((current_year - previous_year) / previous_year) * 100
                
            growth_rates.append(rate)
        
        # Calculate average annual growth
        avg_annual_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
        
        # Check if growth is consistent (all years positive growth)
        consistent_growth = all(rate > 0 for rate in growth_rates)
        
        # Get the most recent ROE
        current_roe = annual_roe[0] if annual_roe else None
        
        # Calculate rating based on O'Neil's criteria
        rating = 0
        reason = ""
        
        # Evaluate annual growth - O'Neil seeks 25%+ annual growth
        if avg_annual_growth >= 35:
            growth_rating = a_score = 50
            growth_reason = f"Exceptional annual EPS growth: {avg_annual_growth:.1f}%"
        elif avg_annual_growth >= 25:
            growth_rating = 45
            growth_reason = f"Strong annual EPS growth: {avg_annual_growth:.1f}%"
        elif avg_annual_growth >= 15:
            growth_rating = 35
            growth_reason = f"Good annual EPS growth: {avg_annual_growth:.1f}%"
        elif avg_annual_growth >= 5:
            growth_rating = 25
            growth_reason = f"Moderate annual EPS growth: {avg_annual_growth:.1f}%"
        elif avg_annual_growth >= 0:
            growth_rating = 15
            growth_reason = f"Minimal annual EPS growth: {avg_annual_growth:.1f}%"
        else:
            growth_rating = 5
            growth_reason = f"Declining annual EPS: {avg_annual_growth:.1f}%"
        
        # Evaluate ROE - O'Neil seeks 17%+ ROE
        if current_roe is not None:
            if current_roe >= 25:
                roe_rating = 50
                roe_reason = f"Exceptional ROE: {current_roe:.1f}%"
            elif current_roe >= 17:
                roe_rating = 40
                roe_reason = f"Strong ROE: {current_roe:.1f}%"
            elif current_roe >= 10:
                roe_rating = 30
                roe_reason = f"Good ROE: {current_roe:.1f}%"
            elif current_roe >= 5:
                roe_rating = 20
                roe_reason = f"Moderate ROE: {current_roe:.1f}%"
            elif current_roe > 0:
                roe_rating = 10
                roe_reason = f"Low ROE: {current_roe:.1f}%"
            else:
                roe_rating = 0
                roe_reason = f"Negative ROE: {current_roe:.1f}%"
        else:
            roe_rating = 0
            roe_reason = "ROE data unavailable"
        
        # Combine ratings and reasons
        rating = growth_rating + roe_rating
        reason = f"{growth_reason}. {roe_reason}."
        
        # Adjust for consistency
        if consistent_growth:
            rating = min(100, rating + 10)
            reason += " Growth has been consistent across years."
        
        return {
            "rating": rating,
            "reason": reason,
            "annual_growth": avg_annual_growth,
            "consistent_growth": consistent_growth,
            "roe": current_roe
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing annual earnings: {str(e)}",
            "annual_growth": None,
            "consistent_growth": None,
            "roe": None
        }


def analyze_new_factors(prices_df: pd.DataFrame, news: list) -> dict:
    """
    Analyze 'N' in CANSLIM: New Products, Management, or Price Highs
    
    O'Neil looks for:
    - Stock making new 52-week highs or near all-time highs
    - New products, services or management changes that could drive future growth
    - Recent positive news developments
    
    Returns a rating from 0-100 and analysis details.
    """
    if prices_df is None or prices_df.empty:
        return {
            "rating": None,
            "reason": "Insufficient price data available",
            "near_high": None,
            "has_news": None
        }
    
    try:
        # Analyze price for new highs
        # Calculate the 52-week (or available data) high
        max_days = min(252, len(prices_df))  # Use available data, up to ~252 trading days (1 year)
        fifty_two_week_high = prices_df['high'][:max_days].max()
        current_price = prices_df['close'].iloc[0]
        
        # Calculate how close the current price is to the 52-week high
        if fifty_two_week_high > 0:
            percent_from_high = ((fifty_two_week_high - current_price) / fifty_two_week_high) * 100
        else:
            percent_from_high = 100
        
        # Determine if near 52-week high
        # O'Neil favors stocks within 15% of their highs
        near_high = percent_from_high <= 15
        
        # Check if price recently made a new 52-week high
        recent_high = False
        if max_days > 20:  # Need enough data to determine a "recent" high
            recent_max = prices_df['high'][:20].max()  # Last ~month of trading
            recent_high = (recent_max == fifty_two_week_high)
        
        # Analyze news
        has_recent_news = len(news) > 0
        
        # Score new price highs
        if percent_from_high <= 1:
            price_rating = a_score = 50
            price_reason = "At or extremely close to 52-week high"
        elif percent_from_high <= 5:
            price_rating = 45
            price_reason = f"Very close to 52-week high (within {percent_from_high:.1f}%)"
        elif percent_from_high <= 10:
            price_rating = 40
            price_reason = f"Close to 52-week high (within {percent_from_high:.1f}%)"
        elif percent_from_high <= 15:
            price_rating = 35
            price_reason = f"Relatively close to 52-week high (within {percent_from_high:.1f}%)"
        elif percent_from_high <= 25:
            price_rating = 20
            price_reason = f"Somewhat below 52-week high (within {percent_from_high:.1f}%)"
        else:
            price_rating = 10
            price_reason = f"Significantly below 52-week high ({percent_from_high:.1f}% away)"
        
        # Score news factor
        if has_recent_news:
            news_rating = 30
            news_reason = "Recent news available that could impact company prospects"
        else:
            news_rating = 0
            news_reason = "No recent news found"
        
        # Extra points for recent new high
        recent_high_bonus = 20 if recent_high else 0
        
        # Combine scores
        rating = price_rating + news_rating + recent_high_bonus
        rating = min(100, rating)  # Cap at 100
        
        # Combine reasons
        reason = f"{price_reason}. {news_reason}."
        if recent_high:
            reason += " Stock recently made a new 52-week high, very positive sign."
        
        return {
            "rating": rating,
            "reason": reason,
            "near_high": near_high,
            "has_news": has_recent_news,
            "percent_from_high": percent_from_high,
            "recent_high": recent_high
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing new factors: {str(e)}",
            "near_high": None,
            "has_news": None
        }


def analyze_supply_demand(market_cap: float, prices_df: pd.DataFrame) -> dict:
    """
    Analyze 'S' in CANSLIM: Supply and Demand
    
    O'Neil looks for:
    - Lower supply (small float, share buybacks)
    - Increasing demand (higher volume on up days)
    - Institutional ownership but not excessive
    
    Returns a rating from 0-100 and analysis details.
    """
    if prices_df is None or prices_df.empty:
        return {
            "rating": None,
            "reason": "Insufficient price and volume data available",
            "volume_trend": None,
            "has_accumulation": None
        }
    
    try:
        # Analyze market cap (supply component)
        # O'Neil typically favors smaller to mid-sized companies where big players can't easily take large positions
        small_cap_cutoff = 2e9  # $2 billion
        mid_cap_cutoff = 10e9   # $10 billion
        
        # Analyze volume patterns (demand component)
        # Get at least 30 days of data if available
        analysis_period = min(30, len(prices_df))
        recent_data = prices_df.head(analysis_period)
        
        # Calculate average volume
        avg_volume = recent_data['volume'].mean()
        
        # Identify up days and down days
        recent_data['price_change'] = recent_data['close'].diff(-1) * -1  # Reverse order to get positive for up days
        up_days = recent_data[recent_data['price_change'] > 0]
        down_days = recent_data[recent_data['price_change'] < 0]
        
        # Calculate average volume on up days vs down days
        avg_volume_up = up_days['volume'].mean() if len(up_days) > 0 else 0
        avg_volume_down = down_days['volume'].mean() if len(down_days) > 0 else float('inf')
        
        # O'Neil looks for higher volume on up days (accumulation)
        has_accumulation = avg_volume_up > avg_volume_down
        
        # Calculate volume trend (increasing or decreasing)
        # Use simple linear regression on volume data
        if len(recent_data) >= 10:
            x = np.arange(len(recent_data))
            y = recent_data['volume'].values
            slope = np.polyfit(x, y, 1)[0]
            volume_trend = slope > 0
        else:
            volume_trend = None
        
        # Score market cap factor
        if market_cap is None:
            market_cap_rating = 0
            market_cap_reason = "Market cap data unavailable"
        elif market_cap < small_cap_cutoff:
            market_cap_rating = 40
            market_cap_reason = f"Small cap stock (${market_cap/1e9:.1f}B), favorable for institutional accumulation"
        elif market_cap < mid_cap_cutoff:
            market_cap_rating = 35
            market_cap_reason = f"Mid cap stock (${market_cap/1e9:.1f}B), still room for institutional buying"
        else:
            market_cap_rating = 25
            market_cap_reason = f"Large cap stock (${market_cap/1e9:.1f}B), may limit exceptional growth"
        
        # Score volume factors
        volume_rating = 0
        volume_reason = ""
        
        if has_accumulation:
            volume_rating += 30
            volume_reason = "Higher volume on up days indicates institutional accumulation"
        else:
            volume_rating += 10
            volume_reason = "Lower volume on up days suggests lack of institutional demand"
        
        if volume_trend:
            volume_rating += 20
            volume_reason += ". Increasing volume trend shows growing interest"
        elif volume_trend is not None:
            volume_rating += 0
            volume_reason += ". Decreasing volume trend indicates diminishing interest"
        
        # Combine scores
        rating = market_cap_rating + volume_rating
        rating = min(100, rating)  # Cap at 100
        
        # Combine reasons
        reason = f"{market_cap_reason}. {volume_reason}."
        
        return {
            "rating": rating,
            "reason": reason,
            "market_cap": market_cap,
            "volume_trend": volume_trend,
            "has_accumulation": has_accumulation,
            "avg_volume_up": avg_volume_up,
            "avg_volume_down": avg_volume_down
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing supply and demand: {str(e)}",
            "volume_trend": None,
            "has_accumulation": None
        }


def analyze_leader_or_laggard(prices_df: pd.DataFrame, market_cap: float) -> dict:
    """
    Analyze 'L' in CANSLIM: Leader or Laggard
    
    O'Neil looks for:
    - Stocks with strong relative strength compared to the market
    - Leading stocks in leading industry groups
    - Avoid laggards in a sector even if they seem cheap
    
    Returns a rating from 0-100 and analysis details.
    """
    if prices_df is None or prices_df.empty:
        return {
            "rating": None,
            "reason": "Insufficient price data available",
            "relative_strength": None,
            "is_leader": None
        }
    
    try:
        # Calculate relative strength against market
        # Ideally, we'd compare against industry peers but we don't have that data here
        # Instead, we'll calculate a simple relative strength index (RSI)
        
        # Calculate 14-day RSI if we have enough data
        if len(prices_df) < 15:
            return {
                "rating": None,
                "reason": "Insufficient price history for relative strength calculation",
                "relative_strength": None,
                "is_leader": None
            }
        
        # Calculate price changes
        delta = prices_df['close'].diff()
        
        # Create arrays for gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate 14-day averages
        avg_gain = gains.rolling(window=14).mean()
        avg_loss = losses.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Get the most recent RSI value
        current_rsi = rsi.iloc[0]
        
        # Evaluate if this is a leader based on RSI
        # O'Neil typically likes stocks with strong momentum (high RSI)
        is_leader = current_rsi > 70
        strong_momentum = current_rsi > 60
        
        # Calculate score based on RSI values
        if current_rsi > 80:
            rating = 100
            reason = f"Exceptional relative strength (RSI: {current_rsi:.1f}), clear market leader"
        elif current_rsi > 70:
            rating = 90
            reason = f"Very strong relative strength (RSI: {current_rsi:.1f}), likely market leader"
        elif current_rsi > 60:
            rating = 75
            reason = f"Strong relative strength (RSI: {current_rsi:.1f}), showing leadership characteristics"
        elif current_rsi > 50:
            rating = 50
            reason = f"Above average relative strength (RSI: {current_rsi:.1f}), potential emerging leader"
        elif current_rsi > 40:
            rating = 30
            reason = f"Average relative strength (RSI: {current_rsi:.1f}), not showing leadership yet"
        else:
            rating = 10
            reason = f"Poor relative strength (RSI: {current_rsi:.1f}), lagging the market"
        
        # Incorporate market cap as a factor in leadership potential
        # O'Neil often favors leaders in emerging small to mid-cap stocks with high growth potential
        if market_cap is not None:
            if market_cap < 5e9 and current_rsi > 60:  # Small cap with strong momentum
                rating = min(100, rating + 10)
                reason += ". Small cap with strong momentum, potential for significant leadership."
        
        return {
            "rating": rating,
            "reason": reason,
            "relative_strength": current_rsi,
            "is_leader": is_leader,
            "strong_momentum": strong_momentum
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing leader/laggard status: {str(e)}",
            "relative_strength": None,
            "is_leader": None
        }


def analyze_institutional_sponsorship(market_cap: float, prices_df: pd.DataFrame) -> dict:
    """
    Analyze 'I' in CANSLIM: Institutional Sponsorship
    
    O'Neil looks for:
    - Some institutional ownership (but not too much)
    - Increasing institutional ownership
    - Ownership by high-quality funds with good track records
    
    Returns a rating from 0-100 and analysis details.
    """
    if prices_df is None or prices_df.empty:
        return {
            "rating": None,
            "reason": "Insufficient price and volume data available",
            "institutional_interest": None
        }
    
    try:
        # Since we don't have direct institutional ownership data, we'll infer it from:
        # 1. Market cap (institutions generally prefer larger caps)
        # 2. Average daily volume (institutions need liquidity)
        # 3. Price patterns that suggest institutional accumulation
        
        # Calculate average daily volume
        avg_daily_volume = prices_df['volume'].mean()
        
        # Calculate average dollar volume (liquidity)
        avg_price = prices_df['close'].mean()
        avg_dollar_volume = avg_daily_volume * avg_price
        
        # Analyze volume patterns for institutional activity
        # Get shorter-term data for recent activity
        recent_days = min(20, len(prices_df))
        recent_data = prices_df.head(recent_days)
        
        # Look for accumulation days (price up on higher than average volume)
        recent_data['price_change'] = recent_data['close'].diff(-1) * -1  # Positive = up day
        recent_data['volume_ratio'] = recent_data['volume'] / avg_daily_volume
        
        # Count accumulation days (up days with above-average volume)
        accumulation_days = recent_data[(recent_data['price_change'] > 0) & 
                                        (recent_data['volume_ratio'] > 1.2)].shape[0]
        
        # Count distribution days (down days with above-average volume)
        distribution_days = recent_data[(recent_data['price_change'] < 0) & 
                                       (recent_data['volume_ratio'] > 1.2)].shape[0]
        
        # Calculate institutional interest based on market cap
        inst_interest_by_cap = 0
        if market_cap is not None:
            if market_cap > 50e9:  # Large cap
                inst_interest_by_cap = 90  # Likely high institutional ownership
            elif market_cap > 10e9:  # Mid-large cap
                inst_interest_by_cap = 80  # Significant institutional ownership
            elif market_cap > 2e9:  # Mid cap
                inst_interest_by_cap = 70  # Moderate institutional ownership
            elif market_cap > 500e6:  # Small-mid cap
                inst_interest_by_cap = 50  # Some institutional ownership
            elif market_cap > 100e6:  # Small cap
                inst_interest_by_cap = 30  # Limited institutional ownership
            else:  # Micro cap
                inst_interest_by_cap = 10  # Minimal institutional ownership
        
        # Calculate institutional interest based on liquidity (daily dollar volume)
        inst_interest_by_volume = 0
        if avg_dollar_volume > 50e6:  # $50M+ daily volume
            inst_interest_by_volume = 90  # High institutional interest possible
        elif avg_dollar_volume > 20e6:  # $20M+ daily volume
            inst_interest_by_volume = 80  # Good institutional interest possible
        elif avg_dollar_volume > 10e6:  # $10M+ daily volume
            inst_interest_by_volume = 70  # Moderate institutional interest possible
        elif avg_dollar_volume > 5e6:  # $5M+ daily volume
            inst_interest_by_volume = 50  # Some institutional interest possible
        elif avg_dollar_volume > 1e6:  # $1M+ daily volume
            inst_interest_by_volume = 30  # Limited institutional interest likely
        else:  # Under $1M daily volume
            inst_interest_by_volume = 10  # Too illiquid for most institutions
        
        # O'Neil actually prefers stocks with some but not excessive institutional ownership
        # Too much = overowned, too little = undiscovered or problematic
        # Adjust score based on O'Neil's preference for "right amount" of ownership
        if inst_interest_by_cap > 90:
            inst_interest_by_cap = 70  # Penalty for likely overownership
        
        # Accumulation vs distribution indicates recent institutional behavior
        recent_behavior = 0
        if accumulation_days > distribution_days + 2:
            recent_behavior = 30  # Strong recent accumulation
            behavior_desc = f"Strong recent accumulation ({accumulation_days} accumulation vs {distribution_days} distribution days)"
        elif accumulation_days > distribution_days:
            recent_behavior = 20  # Modest accumulation
            behavior_desc = f"Recent accumulation ({accumulation_days} accumulation vs {distribution_days} distribution days)"
        elif accumulation_days == distribution_days:
            recent_behavior = 10  # Neutral
            behavior_desc = f"Neutral institutional activity ({accumulation_days} accumulation vs {distribution_days} distribution days)"
        else:
            recent_behavior = 0  # Distribution
            behavior_desc = f"Recent distribution ({accumulation_days} accumulation vs {distribution_days} distribution days)"
        
        # Calculate final rating - more weight on recent behavior
        base_inst_score = (inst_interest_by_cap + inst_interest_by_volume) / 2
        rating = (base_inst_score * 0.7) + (recent_behavior * 0.3)
        
        # Cap at 100
        rating = min(100, rating)
        
        # Construct reason
        if market_cap is not None:
            cap_reason = f"Market cap (${market_cap/1e9:.1f}B) suggests {'high' if inst_interest_by_cap > 70 else 'moderate' if inst_interest_by_cap > 40 else 'limited'} institutional ownership"
        else:
            cap_reason = "Market cap data unavailable for institutional ownership assessment"
            
        volume_reason = f"Average daily dollar volume (${avg_dollar_volume/1e6:.1f}M) indicates {'high' if inst_interest_by_volume > 70 else 'moderate' if inst_interest_by_volume > 40 else 'limited'} liquidity for institutional trading"
        
        reason = f"{cap_reason}. {volume_reason}. {behavior_desc}."
        
        # O'Neil notes - add special comment if in sweet spot
        if 40 <= rating <= 80:
            reason += " This stock is in O'Neil's institutional ownership sweet spot - enough institutional interest to provide support but not overowned."
        
        return {
            "rating": rating,
            "reason": reason,
            "institutional_interest": rating / 100,  # Scale 0-1
            "accumulation_days": accumulation_days,
            "distribution_days": distribution_days,
            "avg_dollar_volume": avg_dollar_volume
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing institutional sponsorship: {str(e)}",
            "institutional_interest": None
        }


def analyze_market_direction(market_data: pd.DataFrame) -> dict:
    """
    Analyze 'M' in CANSLIM: Market Direction
    
    O'Neil believes in following the overall market trend, as:
    - 3 out of 4 stocks follow the general market direction
    - It's risky to go against the prevailing market trend
    
    Returns a rating from 0-100 and analysis details.
    """
    if market_data is None or market_data.empty or len(market_data) < 200:
        return {
            "rating": None,
            "reason": "Insufficient market data available",
            "market_trend": None
        }
    
    try:
        # Calculate key moving averages for index
        market_data['MA50'] = market_data['close'].rolling(window=50).mean()
        market_data['MA200'] = market_data['close'].rolling(window=200).mean()
        
        # Get most recent data point
        latest = market_data.iloc[0]
        
        # Determine if price is above key moving averages
        above_ma50 = latest['close'] > latest['MA50']
        above_ma200 = latest['close'] > latest['MA200']
        
        # Calculate distance from moving averages as percentage
        dist_from_ma50 = ((latest['close'] / latest['MA50']) - 1) * 100
        dist_from_ma200 = ((latest['close'] / latest['MA200']) - 1) * 100
        
        # Calculate recent market trend (last 10 days vs previous 10 days)
        if len(market_data) >= 20:
            recent_avg = market_data.iloc[0:10]['close'].mean()
            previous_avg = market_data.iloc[10:20]['close'].mean()
            recent_trend_pct = ((recent_avg / previous_avg) - 1) * 100
        else:
            recent_trend_pct = 0
            
        # Calculate distribution days in last 25 sessions
        if len(market_data) >= 25:
            recent_market = market_data.head(25)
            recent_market['price_change'] = recent_market['close'].diff(-1) * -1
            recent_market['volume_ratio'] = recent_market['volume'] / market_data['volume'].mean()
            
            # Count distribution days (down days on higher volume)
            distribution_days = recent_market[(recent_market['price_change'] < 0) & 
                                             (recent_market['volume_ratio'] > 1.2)].shape[0]
        else:
            distribution_days = 0
        
        # Determine market trend based on moving averages and distribution days
        if above_ma50 and above_ma200 and distribution_days <= 3:
            if dist_from_ma50 > 5 and recent_trend_pct > 3:
                market_trend = "Strong Uptrend"
                rating = 100
                reason = f"Market in strong uptrend: {dist_from_ma50:.1f}% above 50-day MA, {dist_from_ma200:.1f}% above 200-day MA, recent trend +{recent_trend_pct:.1f}%, only {distribution_days} distribution days."
            elif dist_from_ma50 > 0:
                market_trend = "Confirmed Uptrend"
                rating = 90
                reason = f"Market in confirmed uptrend: {dist_from_ma50:.1f}% above 50-day MA, {dist_from_ma200:.1f}% above 200-day MA, recent trend +{recent_trend_pct:.1f}%, with {distribution_days} distribution days."
            else:
                market_trend = "Modest Uptrend"
                rating = 75
                reason = f"Market in modest uptrend: above 200-day MA but only {dist_from_ma50:.1f}% from 50-day MA, recent trend +{recent_trend_pct:.1f}%, with {distribution_days} distribution days."
        elif above_ma200 and distribution_days <= 5:
            market_trend = "Healthy Market"
            rating = 60
            reason = f"Healthy market: above 200-day MA but {'below' if not above_ma50 else 'near'} 50-day MA, recent trend {recent_trend_pct:.1f}%, with {distribution_days} distribution days."
        elif above_ma200:
            market_trend = "Market Under Pressure"
            rating = 40
            reason = f"Market under pressure: above 200-day MA but below 50-day MA, recent trend {recent_trend_pct:.1f}%, with {distribution_days} distribution days."
        elif distribution_days >= 7:
            market_trend = "Market Correction"
            rating = 20
            reason = f"Market in correction: below 50-day and 200-day MAs, recent trend {recent_trend_pct:.1f}%, with {distribution_days} distribution days."
        else:
            market_trend = "Bear Market"
            rating = 10
            reason = f"Bear market conditions: below 50-day and 200-day MAs, recent trend {recent_trend_pct:.1f}%, with {distribution_days} distribution days."
        
        # Add O'Neil's famous advice
        if rating <= 40:
            reason += " O'Neil advises increased caution during market corrections, keeping high cash positions and avoiding new purchases."
        elif rating >= 75:
            reason += " According to O'Neil, this is an ideal time to be invested with the market in a confirmed uptrend."
        
        return {
            "rating": rating,
            "reason": reason,
            "market_trend": market_trend,
            "above_ma50": above_ma50,
            "above_ma200": above_ma200,
            "distribution_days": distribution_days
        }
        
    except Exception as e:
        return {
            "rating": None,
            "reason": f"Error analyzing market direction: {str(e)}",
            "market_trend": None
        }


def william_oneil_agent(state: AgentState):
    """
    The William O'Neil agent identifies growth stocks using the CANSLIM methodology.
    
    CANSLIM is an acronym for:
    C - Current quarterly earnings per share (EPS)
    A - Annual earnings growth
    N - New products, management, or price highs
    S - Supply and demand (small float, institutional ownership)
    L - Leader vs. laggard in industry group
    I - Institutional sponsorship
    M - Market direction
    
    O'Neil looks for strong growth stocks with exceptional earnings, emerging products,
    limited supply, institutional accumulation, industry leadership, and supportive market conditions.
    """
    # Get the ticker and date range from the state
    tickers = state.tickers
    date_range = state.date_range
    
    # Get market data (using SPY as proxy for market)
    market_data = pull_stock_data("SPY", date_range)
    market_direction = analyze_market_direction(market_data)
    
    # Initialize signals dictionary
    signals = {}
    
    for ticker in tickers:
        # Get the data for the ticker
        prices_df = pull_stock_data(ticker, date_range)
        metrics = pull_financial_metrics(ticker)
        news = pull_recent_news(ticker)
        market_cap = get_market_cap(ticker)
        
        # Analyze each CANSLIM component
        c_analysis = analyze_current_earnings(metrics)
        a_analysis = analyze_annual_earnings(metrics)
        n_analysis = analyze_new_factors(prices_df, news)
        s_analysis = analyze_supply_demand(market_cap, prices_df)
        l_analysis = analyze_leader_or_laggard(prices_df, market_cap)
        i_analysis = analyze_institutional_sponsorship(market_cap, prices_df)
        
        # Combine all analyses into a comprehensive CANSLIM score
        analyses = {
            "C": c_analysis,
            "A": a_analysis,
            "N": n_analysis,
            "S": s_analysis,
            "L": l_analysis,
            "I": i_analysis,
            "M": market_direction
        }
        
        # Calculate weighted CANSLIM score - O'Neil emphasizes earnings and market direction
        components_rating = {}
        total_weight = 0
        weighted_sum = 0
        
        # Weights for each component
        weights = {
            "C": 25,  # Current Earnings - critical
            "A": 20,  # Annual Earnings Growth - very important
            "N": 15,  # New Products/Highs - important
            "S": 10,  # Supply/Demand - moderately important
            "L": 15,  # Leadership Status - important
            "I": 10,  # Institutional Sponsorship - moderately important
            "M": 5    # Market Direction - applies to all stocks but important context
        }
        
        # Calculate weighted sum
        for component, analysis in analyses.items():
            if analysis is not None and analysis.get("rating") is not None:
                rating = analysis.get("rating")
                weight = weights[component]
                
                components_rating[component] = rating
                weighted_sum += rating * weight
                total_weight += weight
        
        # Calculate final score (0-100 scale)
        if total_weight > 0:
            canslim_score = weighted_sum / total_weight
        else:
            canslim_score = 50  # Neutral if no data
            
        # Determine O'Neil's signal based on CANSLIM score
        if canslim_score >= 80:
            signal = "strong_buy"
            confidence = 0.90
            reasoning = f"Strong CANSLIM candidate (score: {canslim_score:.1f}/100). O'Neil would consider this a high-conviction growth stock with exceptional earnings, price strength, and institutional support."
        elif canslim_score >= 70:
            signal = "buy"
            confidence = 0.75
            reasoning = f"Solid CANSLIM candidate (score: {canslim_score:.1f}/100). Displays many growth characteristics that O'Neil looks for, though some criteria may not be optimal."
        elif canslim_score >= 60:
            signal = "weak_buy"
            confidence = 0.60
            reasoning = f"Moderate CANSLIM candidate (score: {canslim_score:.1f}/100). Has some positive growth factors but falls short on key O'Neil criteria."
        elif canslim_score >= 45:
            signal = "neutral"
            confidence = 0.50
            reasoning = f"Not a strong CANSLIM candidate (score: {canslim_score:.1f}/100). Missing several important growth criteria that O'Neil requires."
        elif canslim_score >= 30:
            signal = "weak_sell"
            confidence = 0.60
            reasoning = f"Poor CANSLIM candidate (score: {canslim_score:.1f}/100). Lacks fundamental and technical strength that O'Neil emphasizes."
        else:
            signal = "sell"
            confidence = 0.80
            reasoning = f"Avoid according to CANSLIM methodology (score: {canslim_score:.1f}/100). Fundamentally and technically weak, contrary to O'Neil's growth criteria."
        
        # Add component breakdown to reasoning
        reasoning += "\n\nCANSLIM Component Analysis:"
        
        # Add details for each component with weighting
        for component, name in [
            ("C", "Current Quarterly Earnings"),
            ("A", "Annual Earnings Growth"),
            ("N", "New Products/Highs"),
            ("S", "Supply & Demand"),
            ("L", "Leadership Status"),
            ("I", "Institutional Sponsorship"),
            ("M", "Market Direction")
        ]:
            analysis = analyses[component]
            if analysis is not None and analysis.get("rating") is not None:
                score = analysis.get("rating")
                weight = weights[component]
                contribution = (score * weight) / 100
                
                reasoning += f"\n• {component}: {name} - {score:.1f}/100 "
                reasoning += f"(Weight: {weight}%, Contribution: {contribution:.1f})"
                reasoning += f"\n  {analysis.get('reason')}"
        
        # Create WilliamONeilSignal named tuple
        oneil_signal = WilliamONeilSignal(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Add to signals dictionary
        signals[ticker] = oneil_signal
    
    # Update the agent state with the signals
    state.signals = signals
    
    # Create a message to return to the system
    message = HumanMessage(
        content=f"William O'Neil has analyzed {len(tickers)} stocks using his CANSLIM methodology."
    )
    
    return message 