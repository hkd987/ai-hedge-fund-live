from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
from typing_extensions import Literal
from tools.api import get_prices, prices_to_df, get_company_news, get_financial_metrics
from utils.llm import call_llm
from utils.progress import progress
from datetime import datetime, timedelta
from utils.caching import cached_analyst


class LindaRaschkeSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


@cached_analyst()
def linda_raschke_agent(state: AgentState):
    """
    Linda Raschke's agent for short-term technical trading strategies.
    
    Implements multiple Raschke strategies including:
    - The "Holy Grail" setup (trend, momentum, support/resistance)
    - 2-period ROC for momentum signals
    - 80-20 trading strategy for mean reversion
    - 3-10 oscillator for short-term momentum trading
    """
    progress.update_status("linda_raschke_agent", None, "Analyzing market data")

    # Get tickers and date range from state
    tickers = state["data"]["tickers"]
    start_date = state["data"]["start_date"]
    end_date = state["data"]["end_date"]

    # For Raschke strategies, we need more historical data
    # Her strategies often use 20+ days of data, so we'll fetch extra days
    extended_start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")

    # Initialize signals
    signals = {}

    for ticker in tickers:
        progress.update_status("linda_raschke_agent", ticker, "Fetching price data")

        # Fetch extended price data
        prices = get_prices(ticker=ticker, start_date=extended_start_date, end_date=end_date)
        if not prices:
            progress.update_status("linda_raschke_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)
        
        # Get recent news for sentiment context
        progress.update_status("linda_raschke_agent", ticker, "Fetching news and context")
        news = get_company_news(ticker=ticker, end_date=end_date)
        
        # Get financial metrics for additional context
        progress.update_status("linda_raschke_agent", ticker, "Analyzing fundamentals")
        metrics = get_financial_metrics(ticker=ticker, end_date=end_date)

        # Analyze with Raschke's strategies
        progress.update_status("linda_raschke_agent", ticker, "Analyzing with Holy Grail setup")
        holy_grail = analyze_holy_grail(prices_df)
        
        progress.update_status("linda_raschke_agent", ticker, "Analyzing with 2-period ROC")
        roc_signals = analyze_roc(prices_df)
        
        progress.update_status("linda_raschke_agent", ticker, "Analyzing with 80-20 strategy")
        mean_reversion = analyze_80_20_strategy(prices_df)
        
        progress.update_status("linda_raschke_agent", ticker, "Analyzing with 3-10 oscillator")
        oscillator_signals = analyze_3_10_oscillator(prices_df)
        
        progress.update_status("linda_raschke_agent", ticker, "Analyzing volume patterns")
        volume_signals = analyze_volume_patterns(prices_df)

        # Combine all analysis into a comprehensive data dictionary
        analysis_data = {
            "holy_grail": holy_grail,
            "roc_signals": roc_signals,
            "mean_reversion": mean_reversion,
            "oscillator_signals": oscillator_signals,
            "volume_signals": volume_signals,
            "current_price": prices_df['close'].iloc[-1],
            "recent_volatility": prices_df['close'].pct_change().std() * 100,  # Volatility as percentage
            "news_available": len(news) > 0,
            "metrics_available": metrics is not None,
        }

        # Generate trading signal with LLM reasoning
        progress.update_status("linda_raschke_agent", ticker, "Generating trading signal")
        signal = generate_raschke_output(
            ticker, 
            analysis_data, 
            state["metadata"]["model_name"], 
            state["metadata"]["model_provider"]
        )

        signals[ticker] = signal
        
        # Model dump converts to dict for JSON serialization
        progress.update_status("linda_raschke_agent", ticker, "Done")

    # Create message
    message = HumanMessage(
        content=json.dumps({ticker: signal.model_dump() for ticker, signal in signals.items()}),
        name="linda_raschke_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: signal.model_dump() for ticker, signal in signals.items()}, "Linda Raschke Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["linda_raschke_agent"] = signals

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def analyze_holy_grail(prices_df: pd.DataFrame) -> dict:
    """
    Implement Linda Raschke's "Holy Grail" setup which combines:
    1. Trend identification (20 EMA)
    2. Momentum (MACD)
    3. Support/resistance (Keltner Channels)
    
    This is one of her signature strategies.
    """
    # Ensure we have enough data
    if len(prices_df) < 30:
        return {"setup_detected": False, "reason": "Insufficient data for Holy Grail setup"}
    
    # Calculate 20-day EMA for trend direction
    prices_df['ema20'] = prices_df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate MACD for momentum
    prices_df['ema12'] = prices_df['close'].ewm(span=12, adjust=False).mean()
    prices_df['ema26'] = prices_df['close'].ewm(span=26, adjust=False).mean()
    prices_df['macd'] = prices_df['ema12'] - prices_df['ema26']
    prices_df['macd_signal'] = prices_df['macd'].ewm(span=9, adjust=False).mean()
    prices_df['macd_hist'] = prices_df['macd'] - prices_df['macd_signal']
    
    # Calculate Keltner Channels for support/resistance
    typical_price = (prices_df['high'] + prices_df['low'] + prices_df['close']) / 3
    atr_period = 10
    
    # Calculate ATR
    prices_df['tr1'] = abs(prices_df['high'] - prices_df['low'])
    prices_df['tr2'] = abs(prices_df['high'] - prices_df['close'].shift(1))
    prices_df['tr3'] = abs(prices_df['low'] - prices_df['close'].shift(1))
    prices_df['tr'] = prices_df[['tr1', 'tr2', 'tr3']].max(axis=1)
    prices_df['atr'] = prices_df['tr'].rolling(window=atr_period).mean()
    
    # Calculate Keltner Channels
    prices_df['kc_middle'] = typical_price.rolling(window=20).mean()
    prices_df['kc_upper'] = prices_df['kc_middle'] + 2 * prices_df['atr']
    prices_df['kc_lower'] = prices_df['kc_middle'] - 2 * prices_df['atr']
    
    # Get recent values
    current_close = prices_df['close'].iloc[-1]
    current_ema20 = prices_df['ema20'].iloc[-1]
    current_macd = prices_df['macd'].iloc[-1]
    current_macd_signal = prices_df['macd_signal'].iloc[-1]
    current_kc_upper = prices_df['kc_upper'].iloc[-1]
    current_kc_lower = prices_df['kc_lower'].iloc[-1]
    
    # Analyze for Holy Grail setup
    trend_up = current_close > current_ema20
    momentum_positive = current_macd > current_macd_signal
    near_support = current_close < (current_kc_lower * 1.02)  # Within 2% of lower channel
    near_resistance = current_close > (current_kc_upper * 0.98)  # Within 2% of upper channel
    
    # Look for bullish Holy Grail setup: uptrend + positive momentum + near support
    bullish_setup = trend_up and momentum_positive and near_support
    
    # Look for bearish Holy Grail setup: downtrend + negative momentum + near resistance
    bearish_setup = not trend_up and not momentum_positive and near_resistance
    
    # Determine setup strength (0-100)
    setup_strength = 0
    signal_type = "neutral"
    
    if bullish_setup:
        setup_strength = min(80, 50 + 
                          (10 if trend_up else 0) + 
                          (10 if momentum_positive else 0) + 
                          (10 if near_support else 0))
        signal_type = "bullish"
    elif bearish_setup:
        setup_strength = min(80, 50 + 
                          (10 if not trend_up else 0) + 
                          (10 if not momentum_positive else 0) + 
                          (10 if near_resistance else 0))
        signal_type = "bearish"
    
    # Check for recent MACD crossover for additional confirmation
    macd_crossover = False
    if len(prices_df) >= 3:
        prev_macd_above_signal = prices_df['macd'].iloc[-2] > prices_df['macd_signal'].iloc[-2]
        current_macd_above_signal = current_macd > current_macd_signal
        
        # Detect crossover
        macd_crossover = prev_macd_above_signal != current_macd_above_signal
        
        # If we have a bullish setup and a bullish crossover, increase strength
        if bullish_setup and current_macd_above_signal and macd_crossover:
            setup_strength += 20
        # If we have a bearish setup and a bearish crossover, increase strength
        elif bearish_setup and not current_macd_above_signal and macd_crossover:
            setup_strength += 20
    
    return {
        "setup_detected": bullish_setup or bearish_setup,
        "signal_type": signal_type,
        "setup_strength": setup_strength,
        "trend_up": trend_up,
        "momentum_positive": momentum_positive,
        "near_support": near_support,
        "near_resistance": near_resistance,
        "macd_crossover": macd_crossover,
        "current_close": current_close,
        "ema20": current_ema20,
        "kc_upper": current_kc_upper,
        "kc_lower": current_kc_lower
    }


def analyze_roc(prices_df: pd.DataFrame) -> dict:
    """
    Implement Linda Raschke's 2-period ROC (Rate of Change) strategy.
    
    This focuses on short-term momentum reversals and is one of her favorite
    indicators for identifying potential short-term turning points.
    """
    # Ensure we have enough data
    if len(prices_df) < 10:
        return {"setup_detected": False, "reason": "Insufficient data for ROC analysis"}
    
    # Calculate 2-period ROC
    prices_df['roc_2'] = prices_df['close'].pct_change(periods=2) * 100
    
    # Calculate 5-day average of 2-period ROC for smoothing
    prices_df['roc_2_avg'] = prices_df['roc_2'].rolling(window=5).mean()
    
    # Get current and recent values
    current_roc = prices_df['roc_2'].iloc[-1]
    current_roc_avg = prices_df['roc_2_avg'].iloc[-1]
    
    # Check for overbought/oversold conditions
    oversold = current_roc < -4.0  # Raschke often used -4% as an oversold threshold
    overbought = current_roc > 4.0  # And +4% as an overbought threshold
    
    # Check for ROC divergence (price making new low but ROC making higher low)
    divergence_bullish = False
    divergence_bearish = False
    
    if len(prices_df) >= 10:
        # Look for recent low in price using argmin instead of idxmin
        recent_min_idx = prices_df['low'][-10:].argmin()
        recent_min_iloc = len(prices_df) - 10 + recent_min_idx  # Convert to proper iloc index
        if recent_min_iloc != len(prices_df) - 1:  # If recent low isn't the most recent bar
            price_made_lower_low = prices_df['low'].iloc[-1] < prices_df['low'].iloc[recent_min_iloc]
            roc_made_higher_low = prices_df['roc_2'].iloc[-1] > prices_df['roc_2'].iloc[recent_min_iloc]
            
            divergence_bullish = price_made_lower_low and roc_made_higher_low
        
        # Look for recent high in price using argmax instead of idxmax
        recent_max_idx = prices_df['high'][-10:].argmax()
        recent_max_iloc = len(prices_df) - 10 + recent_max_idx  # Convert to proper iloc index
        if recent_max_iloc != len(prices_df) - 1:  # If recent high isn't the most recent bar
            price_made_higher_high = prices_df['high'].iloc[-1] > prices_df['high'].iloc[recent_max_iloc]
            roc_made_lower_high = prices_df['roc_2'].iloc[-1] < prices_df['roc_2'].iloc[recent_max_iloc]
            
            divergence_bearish = price_made_higher_high and roc_made_lower_high
    
    # Determine signal type and strength
    signal_type = "neutral"
    signal_strength = 0
    
    # Bullish signal: oversold condition or bullish divergence
    if oversold or divergence_bullish:
        signal_type = "bullish"
        signal_strength = 60 + (20 if oversold else 0) + (20 if divergence_bullish else 0)
    
    # Bearish signal: overbought condition or bearish divergence
    elif overbought or divergence_bearish:
        signal_type = "bearish"
        signal_strength = 60 + (20 if overbought else 0) + (20 if divergence_bearish else 0)
    
    return {
        "setup_detected": signal_type != "neutral",
        "signal_type": signal_type,
        "signal_strength": signal_strength,
        "current_roc": current_roc,
        "current_roc_avg": current_roc_avg,
        "oversold": oversold,
        "overbought": overbought,
        "divergence_bullish": divergence_bullish,
        "divergence_bearish": divergence_bearish
    }


def analyze_80_20_strategy(prices_df: pd.DataFrame) -> dict:
    """
    Implement Linda Raschke's 80-20 strategy for mean reversion.
    
    This strategy looks for situations where price closes in the upper or lower
    20% of its range, and then seeks to fade that move the following day.
    """
    # Ensure we have enough data
    if len(prices_df) < 5:
        return {"setup_detected": False, "reason": "Insufficient data for 80-20 strategy"}
    
    # Calculate daily range
    prices_df['day_range'] = prices_df['high'] - prices_df['low']
    
    # Calculate where the close is within the range (0% = at low, 100% = at high)
    prices_df['close_in_range_pct'] = (prices_df['close'] - prices_df['low']) / prices_df['day_range'] * 100
    
    # Get most recent values
    current_close_pct = prices_df['close_in_range_pct'].iloc[-1]
    prev_close_pct = prices_df['close_in_range_pct'].iloc[-2] if len(prices_df) > 2 else 50
    
    # Check if price closed in the upper or lower 20% of the range
    upper_20_pct = current_close_pct >= 80
    lower_20_pct = current_close_pct <= 20
    
    # Raschke would look for:
    # 1. Close in upper 20% of range -> potential short next day
    # 2. Close in lower 20% of range -> potential long next day
    
    # Determine if we have a mean reversion setup
    bearish_setup = upper_20_pct
    bullish_setup = lower_20_pct
    
    # Check for NLP (Narrow Range, Large Range, Position) setup
    narrow_range = False
    wide_range = False
    
    if len(prices_df) >= 5:
        # Get the median range over the last 5 days
        median_range = prices_df['day_range'][-5:].median()
        
        # Check if yesterday's range was narrow (< 80% of median)
        prev_range = prices_df['day_range'].iloc[-2] if len(prices_df) > 2 else median_range
        narrow_range = prev_range < (median_range * 0.8)
        
        # Check if today's range was wide (> 120% of median)
        current_range = prices_df['day_range'].iloc[-1]
        wide_range = current_range > (median_range * 1.2)
    
    # NLP involves: Narrow range day followed by wide range day with close in upper/lower 20%
    nlp_bullish = narrow_range and wide_range and lower_20_pct
    nlp_bearish = narrow_range and wide_range and upper_20_pct
    
    # Determine signal type and strength
    signal_type = "neutral"
    signal_strength = 0
    
    if nlp_bullish:
        signal_type = "bullish"
        signal_strength = 80  # Strongest signal when we have NLP confirmation
    elif bullish_setup:
        signal_type = "bullish"
        signal_strength = 60
    elif nlp_bearish:
        signal_type = "bearish"
        signal_strength = 80  # Strongest signal when we have NLP confirmation
    elif bearish_setup:
        signal_type = "bearish"
        signal_strength = 60
    
    return {
        "setup_detected": bullish_setup or bearish_setup,
        "signal_type": signal_type,
        "signal_strength": signal_strength,
        "current_close_pct": current_close_pct,
        "prev_close_pct": prev_close_pct,
        "upper_20_pct": upper_20_pct,
        "lower_20_pct": lower_20_pct,
        "narrow_range": narrow_range,
        "wide_range": wide_range,
        "nlp_bullish": nlp_bullish,
        "nlp_bearish": nlp_bearish
    }


def analyze_3_10_oscillator(prices_df: pd.DataFrame) -> dict:
    """
    Implement Linda Raschke's 3-10 Oscillator strategy.
    
    This strategy compares a 3-period and 10-period simple moving average of closing prices
    to identify momentum and potential turning points.
    """
    # Ensure we have enough data
    if len(prices_df) < 15:
        return {"setup_detected": False, "reason": "Insufficient data for 3-10 oscillator"}
    
    # Calculate the 3 and 10 period simple moving averages
    prices_df['sma3'] = prices_df['close'].rolling(window=3).mean()
    prices_df['sma10'] = prices_df['close'].rolling(window=10).mean()
    
    # Calculate the 3-10 oscillator (difference between the two SMAs)
    prices_df['osc_3_10'] = prices_df['sma3'] - prices_df['sma10']
    
    # Normalize the oscillator as a percentage of price for better comparison across stocks
    prices_df['osc_3_10_pct'] = (prices_df['osc_3_10'] / prices_df['close']) * 100
    
    # Calculate a 16-period simple moving average of the oscillator for the signal line
    prices_df['osc_3_10_signal'] = prices_df['osc_3_10_pct'].rolling(window=16).mean()
    
    # Get current values
    current_osc = prices_df['osc_3_10_pct'].iloc[-1]
    current_signal = prices_df['osc_3_10_signal'].iloc[-1]
    
    # Check for crossovers
    crossover_bullish = False
    crossover_bearish = False
    
    if len(prices_df) >= 3:
        prev_osc = prices_df['osc_3_10_pct'].iloc[-2]
        prev_signal = prices_df['osc_3_10_signal'].iloc[-2]
        
        # Bullish crossover: oscillator crosses above signal line
        crossover_bullish = prev_osc <= prev_signal and current_osc > current_signal
        
        # Bearish crossover: oscillator crosses below signal line
        crossover_bearish = prev_osc >= prev_signal and current_osc < current_signal
    
    # Check for extreme readings (overbought/oversold)
    # Raschke often used +1.5% and -1.5% as thresholds
    overbought = current_osc > 1.5
    oversold = current_osc < -1.5
    
    # Determine signal type and strength
    signal_type = "neutral"
    signal_strength = 0
    
    if crossover_bullish:
        signal_type = "bullish"
        signal_strength = 70
    elif crossover_bearish:
        signal_type = "bearish"
        signal_strength = 70
    elif oversold and current_osc > current_signal:  # Bullish divergence from oversold
        signal_type = "bullish"
        signal_strength = 60
    elif overbought and current_osc < current_signal:  # Bearish divergence from overbought
        signal_type = "bearish"
        signal_strength = 60
    
    return {
        "setup_detected": crossover_bullish or crossover_bearish or overbought or oversold,
        "signal_type": signal_type,
        "signal_strength": signal_strength,
        "current_osc": current_osc,
        "current_signal": current_signal,
        "crossover_bullish": crossover_bullish,
        "crossover_bearish": crossover_bearish,
        "overbought": overbought,
        "oversold": oversold
    }


def analyze_volume_patterns(prices_df: pd.DataFrame) -> dict:
    """
    Analyze volume patterns using Linda Raschke's volume principles.
    
    Raschke pays close attention to:
    1. Above-average volume on key price movements
    2. Volume climax patterns
    3. Volume confirmation of price moves
    """
    # Ensure we have enough data
    if len(prices_df) < 20:
        return {"setup_detected": False, "reason": "Insufficient data for volume analysis"}
    
    # Calculate 20-day average volume
    prices_df['volume_sma20'] = prices_df['volume'].rolling(window=20).mean()
    
    # Calculate volume ratio (current volume / average volume)
    prices_df['volume_ratio'] = prices_df['volume'] / prices_df['volume_sma20']
    
    # Identify high volume days (> 1.5x average)
    prices_df['high_volume'] = prices_df['volume_ratio'] > 1.5
    
    # Identify very high volume days (> 2x average) for climax detection
    prices_df['climax_volume'] = prices_df['volume_ratio'] > 2.0
    
    # Get current values
    current_volume = prices_df['volume'].iloc[-1]
    current_volume_ratio = prices_df['volume_ratio'].iloc[-1]
    is_high_volume = prices_df['high_volume'].iloc[-1]
    is_climax_volume = prices_df['climax_volume'].iloc[-1]
    
    # Calculate price change
    current_close = prices_df['close'].iloc[-1]
    prev_close = prices_df['close'].iloc[-2] if len(prices_df) > 2 else current_close
    price_change = ((current_close / prev_close) - 1) * 100  # as percentage
    
    # Check for volume patterns
    volume_climax = False
    volume_confirmation = False
    volume_divergence = False
    
    # Volume climax: Extremely high volume with a significant price move
    if is_climax_volume and abs(price_change) > 2.0:
        volume_climax = True
    
    # Volume confirmation: High volume in the direction of the trend
    if is_high_volume and ((price_change > 1.0) or (price_change < -1.0)):
        volume_confirmation = True
    
    # Volume divergence: Price makes new high/low but volume doesn't confirm
    if len(prices_df) >= 5:
        # For bullish divergence: price made new low but volume decreased
        if prices_df['low'].iloc[-1] < prices_df['low'].iloc[-5:].min() and prices_df['volume'].iloc[-1] < prices_df['volume'].iloc[-2]:
            volume_divergence = True
            
        # For bearish divergence: price made new high but volume decreased
        if prices_df['high'].iloc[-1] > prices_df['high'].iloc[-5:].max() and prices_df['volume'].iloc[-1] < prices_df['volume'].iloc[-2]:
            volume_divergence = True
    
    # Raschke's interpretation of volume patterns
    signal_type = "neutral"
    signal_strength = 0
    
    # Volume climax can indicate exhaustion and potential reversal
    if volume_climax:
        if price_change > 0:
            signal_type = "bearish"  # Potential buying climax
            signal_strength = 70
        else:
            signal_type = "bullish"  # Potential selling climax
            signal_strength = 70
    
    # Volume confirmation strengthens existing trend
    elif volume_confirmation:
        if price_change > 0:
            signal_type = "bullish"
            signal_strength = 60
        else:
            signal_type = "bearish"
            signal_strength = 60
    
    # Volume divergence can indicate potential reversals
    elif volume_divergence:
        if price_change > 0:
            signal_type = "bearish"  # New high with decreasing volume
            signal_strength = 50
        else:
            signal_type = "bullish"  # New low with decreasing volume
            signal_strength = 50
    
    return {
        "setup_detected": volume_climax or volume_confirmation or volume_divergence,
        "signal_type": signal_type,
        "signal_strength": signal_strength,
        "current_volume": current_volume,
        "current_volume_ratio": current_volume_ratio,
        "is_high_volume": is_high_volume,
        "is_climax_volume": is_climax_volume,
        "price_change": price_change,
        "volume_climax": volume_climax,
        "volume_confirmation": volume_confirmation,
        "volume_divergence": volume_divergence
    }


def convert_to_serializable(obj):
    """
    Convert NumPy types to standard Python types for JSON serialization.
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def generate_raschke_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> LindaRaschkeSignal:
    """
    Generate a trading signal based on Linda Raschke's analysis methods.
    Uses LLM to synthesize and interpret multiple technical signals.
    """
    # Convert analysis data to serializable format
    serializable_data = convert_to_serializable(analysis_data)
    
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a trading expert following Linda Raschke's trading strategies.
                
                Linda Raschke is known for:
                1. The Holy Grail setup combining trend, momentum, and support/resistance
                2. The 2-period ROC for momentum trading
                3. 80-20 strategy for mean reversion
                4. 3-10 oscillator for short-term momentum detection
                5. Strong focus on volume analysis and confirmation
                
                Your job is to analyze technical signals and generate a trading decision.
                Return either "bullish", "bearish", or "neutral" with a confidence level (0-100).
                
                Focus on Raschke's key principles:
                - Emphasis on short-term trading (1-5 days)
                - Follow the path of least resistance
                - Look for a convergence of multiple signals
                - Focus on risk management; take partial profits quickly
                - Never add to losing positions
                - Pay attention to volume as confirmation
                
                Provide detailed reasoning in a professional, concise manner.
                """,
            ),
            (
                "human",
                """Analyze the following technical data for ${ticker} and provide a trading signal:
                
                Holy Grail Setup:
                ${holy_grail}
                
                2-Period ROC Analysis:
                ${roc_signals}
                
                80-20 Strategy:
                ${mean_reversion}
                
                3-10 Oscillator:
                ${oscillator_signals}
                
                Volume Analysis:
                ${volume_signals}
                
                Current price: ${current_price}
                Recent volatility: ${recent_volatility}%
                
                Based on Linda Raschke's trading methodology, what's your trading signal for ${ticker}?
                Return your answer as a JSON with these fields:
                {{
                  "signal": "bullish" or "bearish" or "neutral",
                  "confidence": a float value between 0 and 100,
                  "reasoning": "Your detailed reasoning using Raschke's principles"
                }}
                """,
            ),
        ]
    )
    
    # Create a simpler, default response in case the LLM call fails
    def create_default_raschke_signal():
        # Count the signals by type
        signal_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        # Process Holy Grail signal
        if serializable_data["holy_grail"]["setup_detected"]:
            signal_counts[serializable_data["holy_grail"]["signal_type"]] += 1
            
        # Process 2-period ROC signal
        if serializable_data["roc_signals"]["setup_detected"]:
            signal_counts[serializable_data["roc_signals"]["signal_type"]] += 1
            
        # Process 80-20 strategy signal
        if serializable_data["mean_reversion"]["setup_detected"]:
            signal_counts[serializable_data["mean_reversion"]["signal_type"]] += 1
            
        # Process 3-10 oscillator signal
        if serializable_data["oscillator_signals"]["setup_detected"]:
            signal_counts[serializable_data["oscillator_signals"]["signal_type"]] += 1
            
        # Process volume signals
        if serializable_data["volume_signals"]["setup_detected"]:
            signal_counts[serializable_data["volume_signals"]["signal_type"]] += 1
            
        # Determine the most common signal type
        if signal_counts["bullish"] > signal_counts["bearish"]:
            signal_type = "bullish"
            confidence = min(60 + (signal_counts["bullish"] * 10), 90)
        elif signal_counts["bearish"] > signal_counts["bullish"]:
            signal_type = "bearish"
            confidence = min(60 + (signal_counts["bearish"] * 10), 90)
        else:
            signal_type = "neutral"
            confidence = 50
            
        return LindaRaschkeSignal(
            signal=signal_type, 
            confidence=confidence,
            reasoning=f"Based on {signal_counts['bullish']} bullish and {signal_counts['bearish']} bearish signals across Linda Raschke's strategies."
        )
    
    # Generate the prompt with the analysis data
    prompt = template.invoke(
        {
            "ticker": ticker,
            "holy_grail": json.dumps(serializable_data["holy_grail"], indent=2),
            "roc_signals": json.dumps(serializable_data["roc_signals"], indent=2),
            "mean_reversion": json.dumps(serializable_data["mean_reversion"], indent=2),
            "oscillator_signals": json.dumps(serializable_data["oscillator_signals"], indent=2),
            "volume_signals": json.dumps(serializable_data["volume_signals"], indent=2),
            "current_price": serializable_data["current_price"],
            "recent_volatility": serializable_data["recent_volatility"],
        }
    )
    
    # Call the LLM to generate the trading signal
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=LindaRaschkeSignal,
        agent_name="linda_raschke_agent",
        default_factory=create_default_raschke_signal,
    )
