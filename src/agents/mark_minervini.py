from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import numpy as np
import pandas as pd
from typing_extensions import Literal
from tools.api import get_prices, prices_to_df, get_financial_metrics, get_market_cap
from utils.llm import call_llm
from utils.progress import progress


class MarkMinerviniSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(..., alias="trading_signal")
    confidence: float
    reasoning: str


def mark_minervini_agent(state: AgentState):
    """
    Mark Minervini agent that analyzes stocks based on his SEPA (Specific Entry Point Analysis) methodology.
    
    Minervini is known for his growth stock momentum strategy, focusing on:
    - Strong earnings growth (>25% quarterly)
    - Relative strength and price performance
    - Tight price consolidation followed by breakouts
    - Volume confirmation on breakouts
    - Risk management with tight stop losses
    """
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    minervini_analysis = {}

    # Process each ticker
    for ticker in tickers:
        progress.update_status("mark_minervini_agent", ticker, "Analyzing price data")

        # Get price data
        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("mark_minervini_agent", ticker, "Failed: No price data")
            continue

        prices_df = prices_to_df(prices)

        # Get financial metrics for earnings growth analysis
        progress.update_status("mark_minervini_agent", ticker, "Analyzing fundamentals")
        metrics = get_financial_metrics(ticker, end_date=data["end_date"])
        
        # Get market cap for stock classification
        market_cap = get_market_cap(ticker, end_date=data["end_date"])

        # Run technical analysis specifically tailored to Minervini's SEPA methodology
        progress.update_status("mark_minervini_agent", ticker, "Performing SEPA analysis")
        
        # Calculate Minervini's technical indicators
        sepa_analysis = analyze_sepa_criteria(prices_df)
        
        # Analyze earnings growth
        earnings_analysis = analyze_earnings_growth(metrics)
        
        # Check for sector leaders (relative strength)
        relative_strength = analyze_relative_strength(prices_df)
        
        # Identify consolidation patterns and breakouts
        pattern_analysis = analyze_price_patterns(prices_df)
        
        # Analyze volume characteristics
        volume_analysis = analyze_volume(prices_df)
        
        # Risk assessment (potential stop loss levels)
        risk_analysis = calculate_risk_levels(prices_df)

        # Combine all analysis components
        analysis_data = {
            "ticker": ticker,
            "current_price": float(prices_df["close"].iloc[-1]),
            "market_cap": market_cap,
            "sepa_criteria": sepa_analysis,
            "earnings_growth": earnings_analysis,
            "relative_strength": relative_strength,
            "pattern_analysis": pattern_analysis,
            "volume_analysis": volume_analysis,
            "risk_analysis": risk_analysis,
        }

        # Generate the final output
        progress.update_status("mark_minervini_agent", ticker, "Generating recommendation")
        minervini_signal = generate_minervini_output(
            ticker,
            analysis_data,
            state["metadata"]["model_name"],
            state["metadata"]["model_provider"],
        )

        minervini_analysis[ticker] = {
            "signal": minervini_signal.signal,
            "confidence": minervini_signal.confidence,
            "reasoning": minervini_signal.reasoning,
        }

        progress.update_status("mark_minervini_agent", ticker, "Done")

    # Format the message for the next agent
    message = HumanMessage(
        content=json.dumps(minervini_analysis),
        name="mark_minervini_agent",
    )

    # Show reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(minervini_analysis, "Mark Minervini Agent")

    # Add the signal to the analyst_signals
    state["data"]["analyst_signals"]["mark_minervini_agent"] = minervini_analysis

    # Return updated state
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def analyze_sepa_criteria(prices_df: pd.DataFrame) -> dict:
    """
    Analyze stock based on Minervini's SEPA criteria:
    1. Stock is in uptrend (price above 150 and 200-day MA)
    2. 150-day MA is above 200-day MA
    3. 50-day MA is above 150-day and 200-day MA
    4. Current price is above 50-day MA
    5. Current price is at least 30% above 52-week low
    6. Current price is within 25% of 52-week high
    """
    # Calculate moving averages
    prices_df['MA50'] = prices_df['close'].rolling(window=50).mean()
    prices_df['MA150'] = prices_df['close'].rolling(window=150).mean()
    prices_df['MA200'] = prices_df['close'].rolling(window=200).mean()
    
    # Get the most recent values
    latest = prices_df.iloc[-1]
    
    # Calculate 52-week high and low
    year_data = prices_df.iloc[-252:] if len(prices_df) >= 252 else prices_df
    high_52_week = year_data['high'].max()
    low_52_week = year_data['low'].min()
    
    # Check SEPA criteria
    current_price = latest['close']
    
    # Individual criteria checks
    criteria = {
        "price_above_ma150": bool(current_price > latest['MA150']),
        "price_above_ma200": bool(current_price > latest['MA200']),
        "ma150_above_ma200": bool(latest['MA150'] > latest['MA200']),
        "ma50_above_ma150": bool(latest['MA50'] > latest['MA150']),
        "ma50_above_ma200": bool(latest['MA50'] > latest['MA200']),
        "price_above_ma50": bool(current_price > latest['MA50']),
        "price_30pct_above_low": bool(current_price >= low_52_week * 1.3),
        "price_within_25pct_high": bool(current_price >= high_52_week * 0.75),
    }
    
    # Count how many criteria are met
    criteria_met_count = sum(1 for value in criteria.values() if value)
    
    # Calculate percentage of criteria met
    criteria_met_percentage = (criteria_met_count / len(criteria)) * 100
    
    # Overall assessment
    criteria["all_uptrend_criteria_met"] = (
        criteria["price_above_ma150"] and 
        criteria["price_above_ma200"] and 
        criteria["ma150_above_ma200"] and 
        criteria["ma50_above_ma150"] and 
        criteria["ma50_above_ma200"] and 
        criteria["price_above_ma50"]
    )
    
    criteria["all_price_position_criteria_met"] = (
        criteria["price_30pct_above_low"] and 
        criteria["price_within_25pct_high"]
    )
    
    criteria["all_criteria_met"] = (
        criteria["all_uptrend_criteria_met"] and 
        criteria["all_price_position_criteria_met"]
    )
    
    # Additional context
    context = {
        "current_price": float(current_price),
        "ma50": float(latest['MA50']),
        "ma150": float(latest['MA150']),
        "ma200": float(latest['MA200']),
        "high_52_week": float(high_52_week),
        "low_52_week": float(low_52_week),
        "pct_off_high": float(((high_52_week - current_price) / high_52_week) * 100),
        "pct_above_low": float(((current_price - low_52_week) / low_52_week) * 100),
        "criteria_met_percentage": float(criteria_met_percentage)
    }
    
    return {
        "criteria": criteria,
        "context": context,
        "overall_rating": "excellent" if criteria_met_percentage >= 90 else 
                         "good" if criteria_met_percentage >= 75 else
                         "fair" if criteria_met_percentage >= 60 else
                         "poor"
    }


def analyze_earnings_growth(metrics: dict) -> dict:
    """
    Analyze the earnings growth according to Minervini's criteria.
    Minervini looks for accelerating earnings growth, particularly >25% quarterly.
    """
    if not metrics or not isinstance(metrics, dict):
        return {
            "has_strong_growth": False,
            "has_accelerating_growth": False,
            "reason": "Insufficient financial data"
        }
    
    # Extract relevant earnings metrics if available
    quarterly_earnings = metrics.get("quarterlyEarnings", [])
    eps_growth_rates = []
    
    # Calculate quarter-over-quarter EPS growth rates
    if len(quarterly_earnings) >= 2:
        for i in range(1, min(5, len(quarterly_earnings))):
            current_eps = quarterly_earnings[i-1].get("reportedEPS", 0)
            previous_eps = quarterly_earnings[i].get("reportedEPS", 0)
            
            # Avoid division by zero or negative EPS in denominator
            if previous_eps and previous_eps > 0:
                growth_rate = ((current_eps - previous_eps) / previous_eps) * 100
                eps_growth_rates.append(growth_rate)
    
    # Analyze growth patterns
    has_strong_growth = False
    has_accelerating_growth = False
    latest_growth_rate = None
    avg_growth_rate = None
    
    if eps_growth_rates:
        latest_growth_rate = eps_growth_rates[0]
        avg_growth_rate = sum(eps_growth_rates) / len(eps_growth_rates)
        
        # Minervini looks for at least 25% quarterly growth
        has_strong_growth = latest_growth_rate >= 25
        
        # Check for acceleration (current quarter growth > average of previous quarters)
        has_accelerating_growth = latest_growth_rate > avg_growth_rate
    
    return {
        "has_strong_growth": has_strong_growth,
        "has_accelerating_growth": has_accelerating_growth,
        "latest_growth_rate": latest_growth_rate,
        "avg_growth_rate": avg_growth_rate,
        "growth_rates": eps_growth_rates,
        "reason": f"Latest quarterly growth: {latest_growth_rate:.1f}%" if latest_growth_rate is not None else "Insufficient earnings data"
    }


def analyze_relative_strength(prices_df: pd.DataFrame) -> dict:
    """
    Calculate relative strength metrics according to Minervini's approach.
    Minervini focuses on stocks showing strong relative strength vs. the market.
    """
    # Calculate returns for different timeframes
    if len(prices_df) < 252:  # Less than a year of data
        return {"has_strong_rs": False, "reason": "Insufficient price history"}
    
    # Calculate returns for different timeframes
    current_price = prices_df['close'].iloc[-1]
    price_3m_ago = prices_df['close'].iloc[-63] if len(prices_df) >= 63 else prices_df['close'].iloc[0]
    price_6m_ago = prices_df['close'].iloc[-126] if len(prices_df) >= 126 else prices_df['close'].iloc[0]
    price_12m_ago = prices_df['close'].iloc[-252] if len(prices_df) >= 252 else prices_df['close'].iloc[0]
    
    returns = {
        "3_month": ((current_price / price_3m_ago) - 1) * 100,
        "6_month": ((current_price / price_6m_ago) - 1) * 100,
        "12_month": ((current_price / price_12m_ago) - 1) * 100
    }
    
    # Calculate RS Rating (simplified version of Minervini's approach)
    # In a real implementation, this would compare against sector and market benchmarks
    # For this example, we'll use fixed thresholds
    rs_score = 0
    
    # 3-month return contribution
    if returns["3_month"] > 25:
        rs_score += 40
    elif returns["3_month"] > 15:
        rs_score += 30
    elif returns["3_month"] > 5:
        rs_score += 20
    elif returns["3_month"] > 0:
        rs_score += 10
    
    # 6-month return contribution
    if returns["6_month"] > 50:
        rs_score += 30
    elif returns["6_month"] > 30:
        rs_score += 20
    elif returns["6_month"] > 10:
        rs_score += 10
    
    # 12-month return contribution
    if returns["12_month"] > 100:
        rs_score += 30
    elif returns["12_month"] > 50:
        rs_score += 20
    elif returns["12_month"] > 20:
        rs_score += 10
    
    # Minervini typically focuses on stocks with RS Rating above 80
    has_strong_rs = rs_score >= 80
    
    return {
        "has_strong_rs": has_strong_rs,
        "rs_score": rs_score,
        "returns": returns,
        "reason": f"RS Score: {rs_score}/100" + (" - Strong relative strength" if has_strong_rs else " - Insufficient relative strength")
    }


def analyze_price_patterns(prices_df: pd.DataFrame) -> dict:
    """
    Analyze price consolidation and breakout patterns according to Minervini's strategy.
    Minervini looks for tight price consolidation (reduced volatility) followed by breakouts.
    """
    # We need at least 50 days of data for meaningful analysis
    if len(prices_df) < 50:
        return {"has_favorable_pattern": False, "reason": "Insufficient price history"}
    
    # Get recent price action (last 20 trading days)
    recent_prices = prices_df.iloc[-20:]
    
    # Calculate volatility (standard deviation of daily returns)
    daily_returns = prices_df['close'].pct_change().dropna()
    recent_volatility = daily_returns.iloc[-20:].std() * 100  # As percentage
    
    # Check for tight consolidation (low volatility relative to previous periods)
    prev_volatility = daily_returns.iloc[-40:-20].std() * 100 if len(daily_returns) >= 40 else None
    is_consolidating = prev_volatility is not None and recent_volatility < prev_volatility * 0.8
    
    # Check for a potential breakout
    # Minervini often looks for breakouts above resistance on increased volume
    recent_high = recent_prices['high'].max()
    current_price = prices_df['close'].iloc[-1]
    recent_volume = prices_df['volume'].iloc[-5:].mean()
    previous_volume = prices_df['volume'].iloc[-20:-5].mean()
    volume_increase = recent_volume > previous_volume * 1.5
    
    # Define breakout as current price being within 2% of recent high
    is_near_high = current_price >= recent_high * 0.98
    
    # Check for breakout above resistance with increased volume
    is_breakout = is_near_high and volume_increase
    
    # Minervini also looks for cup and handle patterns, but that's more complex
    # For simplicity, we're focusing on consolidation and breakout patterns
    
    # Determine if the pattern is favorable
    has_favorable_pattern = (is_consolidating and is_breakout) or is_breakout
    
    return {
        "has_favorable_pattern": has_favorable_pattern,
        "is_consolidating": is_consolidating,
        "is_breakout": is_breakout,
        "recent_volatility": float(recent_volatility),
        "previous_volatility": float(prev_volatility) if prev_volatility is not None else None,
        "volume_increase": volume_increase,
        "reason": "Favorable pattern: Consolidation followed by breakout" if has_favorable_pattern else
                 "Breakout detected" if is_breakout else
                 "Consolidation detected" if is_consolidating else
                 "No favorable pattern detected"
    }


def analyze_volume(prices_df: pd.DataFrame) -> dict:
    """
    Analyze volume characteristics according to Minervini's methods.
    Minervini emphasizes volume confirmation on breakouts and accumulation patterns.
    """
    if len(prices_df) < 50:
        return {"has_strong_volume": False, "reason": "Insufficient data"}
    
    # Calculate recent average volume (last 5 days)
    recent_avg_volume = prices_df['volume'].iloc[-5:].mean()
    
    # Calculate average volume over the last 50 days
    avg_volume_50d = prices_df['volume'].iloc[-50:].mean()
    
    # Check for volume surge (recent volume > 50-day average)
    has_volume_surge = recent_avg_volume > avg_volume_50d * 1.5
    
    # Check for up days on higher volume
    recent_days = prices_df.iloc[-10:]
    up_days = recent_days[recent_days['close'] > recent_days['open']]
    down_days = recent_days[recent_days['close'] < recent_days['open']]
    
    # Calculate average volume on up days and down days
    avg_volume_up = up_days['volume'].mean() if not up_days.empty else 0
    avg_volume_down = down_days['volume'].mean() if not down_days.empty else float('inf')
    
    # Minervini looks for higher volume on up days than down days
    higher_volume_on_up_days = avg_volume_up > avg_volume_down
    
    # Check for accumulation (rising prices with increasing volume)
    price_trend = prices_df['close'].iloc[-20:].pct_change().mean() > 0
    volume_trend = prices_df['volume'].iloc[-20:].pct_change().mean() > 0
    has_accumulation = price_trend and volume_trend
    
    # Overall volume assessment
    has_strong_volume = has_volume_surge or (higher_volume_on_up_days and has_accumulation)
    
    return {
        "has_strong_volume": has_strong_volume,
        "has_volume_surge": has_volume_surge,
        "higher_volume_on_up_days": higher_volume_on_up_days,
        "has_accumulation": has_accumulation,
        "recent_avg_volume": float(recent_avg_volume),
        "avg_volume_50d": float(avg_volume_50d),
        "avg_volume_up": float(avg_volume_up) if avg_volume_up else None,
        "avg_volume_down": float(avg_volume_down) if avg_volume_down < float('inf') else None,
        "reason": "Strong volume characteristics" if has_strong_volume else "Weak volume characteristics"
    }


def calculate_risk_levels(prices_df: pd.DataFrame) -> dict:
    """
    Calculate risk levels according to Minervini's risk management rules.
    Minervini is known for using tight stop losses, often 7-8% below purchase price.
    """
    if len(prices_df) < 20:
        return {"stop_loss_defined": False, "reason": "Insufficient price history"}
    
    current_price = prices_df['close'].iloc[-1]
    
    # Calculate volatility-based stop loss (Minervini uses volatility-adjusted stops)
    atr_period = 14
    high_low = prices_df['high'] - prices_df['low']
    high_close = abs(prices_df['high'] - prices_df['close'].shift())
    low_close = abs(prices_df['low'] - prices_df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    # Minervini typically uses 2-3 ATR for stop loss
    stop_loss_price = current_price - (2.5 * atr)
    stop_loss_percentage = ((current_price - stop_loss_price) / current_price) * 100
    
    # Minervini also follows the 7-8% rule for maximum loss
    fixed_stop_price = current_price * 0.925  # 7.5% below current price
    
    # Use the more conservative (higher) of the two stop prices
    final_stop_price = max(stop_loss_price, fixed_stop_price)
    final_stop_percentage = ((current_price - final_stop_price) / current_price) * 100
    
    # Calculate potential reward (next resistance level)
    # For simplicity, we'll use the 52-week high as our target
    year_data = prices_df.iloc[-252:] if len(prices_df) >= 252 else prices_df
    target_price = year_data['high'].max() * 1.05  # 5% above 52-week high
    
    # If current price is already near the 52-week high, project a new target
    if current_price > year_data['high'].max() * 0.95:
        target_price = current_price * 1.15  # 15% above current price
    
    reward_percentage = ((target_price - current_price) / current_price) * 100
    
    # Calculate risk/reward ratio (Minervini looks for at least 3:1)
    risk_reward_ratio = reward_percentage / final_stop_percentage if final_stop_percentage > 0 else 0
    
    return {
        "stop_loss_defined": True,
        "stop_loss_price": float(final_stop_price),
        "stop_loss_percentage": float(final_stop_percentage),
        "target_price": float(target_price),
        "reward_percentage": float(reward_percentage),
        "risk_reward_ratio": float(risk_reward_ratio),
        "favorable_risk_reward": risk_reward_ratio >= 3.0,
        "reason": f"Risk/Reward ratio: {risk_reward_ratio:.1f}" + (
            " - Favorable" if risk_reward_ratio >= 3.0 else " - Unfavorable"
        )
    }


def convert_to_serializable(obj):
    """
    Convert NumPy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def generate_minervini_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> MarkMinerviniSignal:
    """Generate the final output using the LLM."""
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Mark Minervini, a legendary growth stock trader known for your SEPA (Specific Entry Point Analysis) method. 
                Your trading approach focuses on:
                
                1. Strong uptrends with price above key moving averages (50-day, 150-day, and 200-day)
                2. Exceptional earnings growth (25%+ quarterly growth)
                3. Strong relative strength compared to the market
                4. Tight price consolidation followed by breakouts on increasing volume
                5. Strict risk management with 7-8% stop losses and 3:1 reward-to-risk minimum
                
                You are extremely selective and only invest in stocks that meet all or most of your criteria.
                You're looking for the next potential 100-300% winners, not value plays or turnarounds.
                
                Return your analysis in the following JSON format:
                {
                  "signal": "bullish" or "bearish" or "neutral",
                  "confidence": a float value between 0 and 100,
                  "reasoning": "Your detailed reasoning here"
                }
                """,
            ),
            (
                "human",
                """Analyze this stock using your SEPA method:
                
                Ticker: {ticker}
                Current Price: ${current_price}
                Market Cap: {market_cap}
                
                SEPA Technical Criteria Analysis:
                {sepa_criteria}
                
                Earnings Growth Analysis:
                {earnings_growth}
                
                Relative Strength Analysis:
                {relative_strength}
                
                Price Pattern Analysis:
                {pattern_analysis}
                
                Volume Analysis:
                {volume_analysis}
                
                Risk Analysis:
                {risk_analysis}
                
                Based on your analysis, provide your trading signal (bullish, bearish, or neutral) with confidence level (0-100) and detailed reasoning explaining why the stock does or doesn't meet your criteria.
                
                Remember to format your response as a JSON object with these keys exactly:
                - "signal": A string that must be one of "bullish", "bearish", or "neutral"
                - "confidence": A number between 0 and 100
                - "reasoning": A string with your detailed analysis
                """,
            ),
        ]
    )
    
    # Convert analysis data to serializable format
    serializable_data = convert_to_serializable(analysis_data)
    
    # Format the prompt with the analysis data
    prompt = template.invoke(
        {
            "ticker": ticker,
            "current_price": serializable_data["current_price"],
            "market_cap": serializable_data["market_cap"],
            "sepa_criteria": json.dumps(serializable_data["sepa_criteria"], indent=2),
            "earnings_growth": json.dumps(serializable_data["earnings_growth"], indent=2),
            "relative_strength": json.dumps(serializable_data["relative_strength"], indent=2),
            "pattern_analysis": json.dumps(serializable_data["pattern_analysis"], indent=2),
            "volume_analysis": json.dumps(serializable_data["volume_analysis"], indent=2),
            "risk_analysis": json.dumps(serializable_data["risk_analysis"], indent=2),
        }
    )
    
    # Create default factory for MarkMinerviniSignal
    def create_default_minervini_signal():
        return MarkMinerviniSignal(
            signal="neutral",
            confidence=0.0,
            reasoning=f"Could not analyze {ticker} due to an error. Mark Minervini would typically avoid stocks with incomplete data.",
        )
    
    # Call the LLM
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=MarkMinerviniSignal,
        agent_name="mark_minervini_agent",
        default_factory=create_default_minervini_signal,
    ) 