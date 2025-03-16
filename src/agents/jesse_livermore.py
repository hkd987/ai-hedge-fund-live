from graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
from typing_extensions import Literal
from tools.api import get_prices, prices_to_df, get_financial_metrics, get_market_cap
from utils.llm import call_llm
from utils.progress import progress
from utils.caching import cached_analyst


class JesseLivermoreSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


@cached_analyst()
def jesse_livermore_agent(state: AgentState):
    """
    Jesse Livermore agent that analyzes stocks based on his speculative trading methodology.
    
    Livermore was known for his short-term trading approach, focusing on:
    - Market momentum and trend following
    - Pivot points for entries and exits
    - Trading the path of least resistance 
    - Psychology of trading (cutting losses, letting winners run)
    - Trading in line with the overall market
    - Sizing up in winning positions
    """
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    livermore_analysis = {}

    # Process each ticker
    for ticker in tickers:
        progress.update_status("jesse_livermore_agent", ticker, "Analyzing price action")

        # Get price data
        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("jesse_livermore_agent", ticker, "Failed: No price data")
            continue

        prices_df = prices_to_df(prices)
        
        # Get market cap for liquidity assessment (Livermore focused on liquid stocks)
        progress.update_status("jesse_livermore_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date=data["end_date"])

        # Run analysis specifically tailored to Livermore's methodology
        progress.update_status("jesse_livermore_agent", ticker, "Identifying pivot points")
        
        # Find pivot points and market structure
        pivots_analysis = identify_pivot_points(prices_df)
        
        # Analyze trend and momentum
        progress.update_status("jesse_livermore_agent", ticker, "Analyzing momentum")
        trend_analysis = analyze_trend_and_momentum(prices_df)
        
        # Check for Livermore's price pattern recognition
        progress.update_status("jesse_livermore_agent", ticker, "Analyzing price patterns")
        pattern_analysis = analyze_price_patterns(prices_df)
        
        # Analyze volume characteristics (Livermore heavily relied on volume)
        progress.update_status("jesse_livermore_agent", ticker, "Analyzing volume")
        volume_analysis = analyze_volume_patterns(prices_df)
        
        # Calculate money management parameters
        progress.update_status("jesse_livermore_agent", ticker, "Calculating position parameters")
        money_management = calculate_position_parameters(prices_df)

        # Combine all analysis components
        analysis_data = {
            "ticker": ticker,
            "current_price": float(prices_df["close"].iloc[-1]),
            "market_cap": market_cap,
            "pivots": pivots_analysis,
            "trend": trend_analysis,
            "patterns": pattern_analysis,
            "volume": volume_analysis,
            "money_management": money_management,
        }

        # Generate the final output
        progress.update_status("jesse_livermore_agent", ticker, "Generating recommendation")
        livermore_signal = generate_livermore_output(
            ticker,
            analysis_data,
            state["metadata"]["model_name"],
            state["metadata"]["model_provider"],
        )

        livermore_analysis[ticker] = {
            "signal": livermore_signal.signal,
            "confidence": livermore_signal.confidence,
            "reasoning": livermore_signal.reasoning,
        }

        progress.update_status("jesse_livermore_agent", ticker, "Done")

    # Format the message for the next agent
    message = HumanMessage(
        content=json.dumps(livermore_analysis),
        name="jesse_livermore_agent",
    )

    # Show reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(livermore_analysis, "Jesse Livermore Agent")

    # Add the signal to the analyst_signals
    state["data"]["analyst_signals"]["jesse_livermore_agent"] = livermore_analysis

    # Return updated state
    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def identify_pivot_points(prices_df: pd.DataFrame) -> dict:
    """
    Identify significant pivot points in price action.
    Livermore was known for his recognition of "pivotal points" that signaled a change in trend.
    """
    if len(prices_df) < 60:  # Need at least 60 days for meaningful analysis
        return {"pivots_identified": False, "reason": "Insufficient price history"}
    
    # Calculate local minima and maxima using a simple algorithm
    window = 10  # For detecting local min/max 
    prices_df['local_max'] = prices_df['high'].rolling(window=window, center=True).max() == prices_df['high']
    prices_df['local_min'] = prices_df['low'].rolling(window=window, center=True).min() == prices_df['low']
    
    # Filter out less significant pivot points
    # Livermore would focus on significant pivots that emerged during trend changes
    significant_pivots = pd.DataFrame()
    
    # Find the last 5 significant pivot highs
    highs = prices_df[prices_df['local_max'] == True].sort_values('high', ascending=False).head(5)
    if not highs.empty:
        significant_pivots = pd.concat([significant_pivots, highs])
    
    # Find the last 5 significant pivot lows
    lows = prices_df[prices_df['local_min'] == True].sort_values('low', ascending=True).head(5)
    if not lows.empty:
        significant_pivots = pd.concat([significant_pivots, lows])
    
    # Sort by date to sequence them properly
    significant_pivots = significant_pivots.sort_index()
    
    # Current price and last pivot points
    current_price = prices_df['close'].iloc[-1]
    
    # Get the most recent significant pivot high and low
    recent_pivot_high = significant_pivots[significant_pivots['local_max'] == True]['high'].iloc[-1] if not highs.empty else None
    recent_pivot_low = significant_pivots[significant_pivots['local_min'] == True]['low'].iloc[-1] if not lows.empty else None
    
    # Determine if price has broken above recent pivot high or below recent pivot low
    broken_above_resistance = bool(recent_pivot_high is not None and current_price > recent_pivot_high)
    broken_below_support = bool(recent_pivot_low is not None and current_price < recent_pivot_low)
    
    # Livermore's key pivot point signals
    # A break above a key pivot high was a buying signal
    # A break below a key pivot low was a selling signal
    pivotal_buy_signal = broken_above_resistance
    pivotal_sell_signal = broken_below_support
    
    # Natural rally/reaction points - Livermore's 3 phases of market movement
    # Phase 1: Initial movement
    # Phase 2: Technical correction (reaction)
    # Phase 3: Movement in the direction of the primary trend
    
    # Calculate the ranges between pivots to identify potential phase
    price_range = 0
    potential_phase = 0
    if recent_pivot_high is not None and recent_pivot_low is not None:
        price_range = recent_pivot_high - recent_pivot_low
        
        # Simplistic determination of phase - in reality would require more sophisticated analysis
        if broken_above_resistance:
            potential_phase = 3  # Possibly in Phase 3 upward movement
        elif broken_below_support:
            potential_phase = 3  # Possibly in Phase 3 downward movement
        else:
            distance_from_high = (recent_pivot_high - current_price) / price_range if price_range > 0 else 0
            if distance_from_high < 0.3:
                potential_phase = 2  # Possibly in Phase 2 (reaction/correction) near a high
            else:
                potential_phase = 1  # Possibly in Phase 1 (initial movement)
    
    return {
        "pivots_identified": True,
        "recent_pivot_high": float(recent_pivot_high) if recent_pivot_high is not None else None,
        "recent_pivot_low": float(recent_pivot_low) if recent_pivot_low is not None else None,
        "broken_above_resistance": broken_above_resistance,
        "broken_below_support": broken_below_support,
        "pivotal_buy_signal": pivotal_buy_signal,
        "pivotal_sell_signal": pivotal_sell_signal,
        "price_range": float(price_range) if price_range else None,
        "potential_market_phase": int(potential_phase) if potential_phase else None,
        "current_price": float(current_price),
        "reason": "Significant price pivots detected and analyzed"
    }


def analyze_trend_and_momentum(prices_df: pd.DataFrame) -> dict:
    """
    Analyze market trend and momentum.
    Livermore believed in trading in the direction of the main trend and "the line of least resistance".
    """
    if len(prices_df) < 60:
        return {"trend_identified": False, "reason": "Insufficient price history"}
    
    # Calculate short and long-term trends
    prices_df['ma10'] = prices_df['close'].rolling(window=10).mean()
    prices_df['ma20'] = prices_df['close'].rolling(window=20).mean()
    prices_df['ma50'] = prices_df['close'].rolling(window=50).mean()
    
    # Calculate momentum indicators
    # 1. Rate of change (Price Velocity) - Livermore watched for accelerating price moves
    prices_df['roc_10'] = prices_df['close'].pct_change(periods=10) * 100
    
    # 2. Moving average convergence/divergence
    prices_df['ema12'] = prices_df['close'].ewm(span=12).mean()
    prices_df['ema26'] = prices_df['close'].ewm(span=26).mean()
    prices_df['macd'] = prices_df['ema12'] - prices_df['ema26']
    prices_df['macd_signal'] = prices_df['macd'].ewm(span=9).mean()
    prices_df['macd_histogram'] = prices_df['macd'] - prices_df['macd_signal']
    
    # 3. Directional Movement (ADX calculation simplified)
    # Livermore didn't use ADX as it didn't exist, but he observed trend strength
    high_low = prices_df['high'] - prices_df['low']
    high_close = np.abs(prices_df['high'] - prices_df['close'].shift())
    low_close = np.abs(prices_df['low'] - prices_df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    prices_df['atr_14'] = atr_14
    
    # Get the most recent values
    current_price = prices_df['close'].iloc[-1]
    current_ma10 = prices_df['ma10'].iloc[-1]
    current_ma20 = prices_df['ma20'].iloc[-1]
    current_ma50 = prices_df['ma50'].iloc[-1]
    
    # Check price position relative to moving averages
    price_above_ma10 = current_price > current_ma10
    price_above_ma20 = current_price > current_ma20
    price_above_ma50 = current_price > current_ma50
    
    # Check moving average alignment (indicates trend direction)
    ma10_above_ma20 = current_ma10 > current_ma20
    ma20_above_ma50 = current_ma20 > current_ma50
    
    # Trend strength indicators
    recent_roc = prices_df['roc_10'].iloc[-1]  # Recent rate of change
    recent_macd = prices_df['macd'].iloc[-1]
    recent_macd_signal = prices_df['macd_signal'].iloc[-1]
    recent_macd_histogram = prices_df['macd_histogram'].iloc[-1]
    
    # Directional movement (simplified trend strength indicator)
    recent_atr = prices_df['atr_14'].iloc[-1]
    atr_percentage = (recent_atr / current_price) * 100  # ATR as percentage of price
    
    # Livermore's "line of least resistance" concept
    # Strong uptrend: Price above all MAs, MAs aligned upward, positive momentum
    # Strong downtrend: Price below all MAs, MAs aligned downward, negative momentum
    uptrend = price_above_ma10 and price_above_ma20 and price_above_ma50 and ma10_above_ma20 and ma20_above_ma50
    downtrend = not price_above_ma10 and not price_above_ma20 and not price_above_ma50 and not ma10_above_ma20 and not ma20_above_ma50
    
    # Livermore paid attention to accelerating trends
    accelerating_uptrend = uptrend and recent_roc > 0 and recent_macd > recent_macd_signal and recent_macd_histogram > 0
    accelerating_downtrend = downtrend and recent_roc < 0 and recent_macd < recent_macd_signal and recent_macd_histogram < 0
    
    # Determine overall trend direction
    if accelerating_uptrend:
        trend_direction = "strongly_bullish"
    elif uptrend:
        trend_direction = "bullish"
    elif accelerating_downtrend:
        trend_direction = "strongly_bearish"
    elif downtrend:
        trend_direction = "bearish"
    else:
        trend_direction = "neutral"
    
    # Livermore was interested in the persistence and consistency of trends
    trend_consistency = 0
    if len(prices_df) >= 20:
        # Count how many of the last 20 days followed the trend
        last_20_days = prices_df.iloc[-20:].copy()
        if trend_direction in ["bullish", "strongly_bullish"]:
            trend_consistency = sum(last_20_days['close'] > last_20_days['close'].shift(1)) / 20 * 100
        elif trend_direction in ["bearish", "strongly_bearish"]:
            trend_consistency = sum(last_20_days['close'] < last_20_days['close'].shift(1)) / 20 * 100
            
    return {
        "trend_identified": True,
        "trend_direction": trend_direction,
        "price_above_ma10": bool(price_above_ma10),
        "price_above_ma20": bool(price_above_ma20),
        "price_above_ma50": bool(price_above_ma50),
        "ma10_above_ma20": bool(ma10_above_ma20),
        "ma20_above_ma50": bool(ma20_above_ma50),
        "accelerating_uptrend": bool(accelerating_uptrend),
        "accelerating_downtrend": bool(accelerating_downtrend),
        "rate_of_change": float(recent_roc),
        "macd_histogram": float(recent_macd_histogram),
        "trend_consistency": float(trend_consistency),
        "volatility_atr_pct": float(atr_percentage),
        "reason": f"Trend direction: {trend_direction.replace('_', ' ')}, "
                 f"Consistency: {trend_consistency:.1f}%, "
                 f"Momentum: {'Accelerating' if accelerating_uptrend or accelerating_downtrend else 'Steady'}"
    }


def analyze_price_patterns(prices_df: pd.DataFrame) -> dict:
    """
    Analyze price patterns that Livermore paid attention to.
    Livermore was particularly attuned to price behavior at key levels and during consolidation.
    """
    if len(prices_df) < 60:
        return {"patterns_identified": False, "reason": "Insufficient price history"}
    
    # Get recent price action
    recent_prices = prices_df.iloc[-30:]
    current_price = prices_df['close'].iloc[-1]
    
    # Calculate price range and volatility
    recent_high = recent_prices['high'].max()
    recent_low = recent_prices['low'].min()
    price_range = recent_high - recent_low
    
    # Livermore paid attention to price contraction and expansion
    # Calculate recent range vs previous range to detect contraction/expansion
    current_period_range = prices_df['high'].iloc[-10:].max() - prices_df['low'].iloc[-10:].min()
    previous_period_range = prices_df['high'].iloc[-20:-10].max() - prices_df['low'].iloc[-20:-10].min()
    
    # Range expansion/contraction
    range_expansion = current_period_range > previous_period_range * 1.2
    range_contraction = current_period_range < previous_period_range * 0.8
    
    # Livermore watched for "crawling along the line" - price hovering near support/resistance
    # This often preceded major moves
    crawling_pattern = False
    if len(prices_df) >= 15:
        # Check if price has been consolidating near the high or low
        near_high_count = sum(prices_df['high'].iloc[-15:] >= recent_high * 0.98)
        near_low_count = sum(prices_df['low'].iloc[-15:] <= recent_low * 1.02)
        
        # Livermore considered "crawling" when price repeatedly tested a level without breaking it
        crawling_pattern = near_high_count >= 5 or near_low_count >= 5
    
    # Livermore's concept of "testing" - the market retracing and then continuing the trend
    successful_test = False
    failed_test = False
    
    if len(prices_df) >= 20:
        # Check for a pullback followed by a continuation
        pullback_threshold = 0.03  # 3% pullback
        
        # For uptrend testing
        if current_price > prices_df['close'].iloc[-20]:  # Overall uptrend
            # Find the minimum during the period
            min_during_period = prices_df['low'].iloc[-19:].min()
            max_before_min = prices_df['high'].iloc[:-19].iloc[-5:].max()
            
            # Calculate pullback size
            pullback = (max_before_min - min_during_period) / max_before_min
            
            # A successful test is when price pulled back but then continued higher
            if pullback > pullback_threshold and current_price > max_before_min:
                successful_test = True
            elif pullback > pullback_threshold and current_price < min_during_period:
                failed_test = True
                
        # For downtrend testing
        elif current_price < prices_df['close'].iloc[-20]:  # Overall downtrend
            # Find the maximum during the period
            max_during_period = prices_df['high'].iloc[-19:].max()
            min_before_max = prices_df['low'].iloc[:-19].iloc[-5:].min()
            
            # Calculate bounce size
            bounce = (max_during_period - min_before_max) / min_before_max
            
            # A successful test is when price bounced but then continued lower
            if bounce > pullback_threshold and current_price < min_before_max:
                successful_test = True
            elif bounce > pullback_threshold and current_price > max_during_period:
                failed_test = True
    
    # Livermore's line break concept - when price breaks a consolidation zone
    consolidation_break = False
    if range_contraction and (current_price > recent_high or current_price < recent_low):
        consolidation_break = True
    
    # Overall pattern assessment
    pattern_detected = crawling_pattern or successful_test or failed_test or consolidation_break
    
    if pattern_detected:
        if consolidation_break and current_price > recent_high:
            pattern_name = "Upside Consolidation Break"
        elif consolidation_break and current_price < recent_low:
            pattern_name = "Downside Consolidation Break"
        elif crawling_pattern and current_price > recent_prices['close'].mean():
            pattern_name = "Crawling Along Resistance"
        elif crawling_pattern and current_price < recent_prices['close'].mean():
            pattern_name = "Crawling Along Support"
        elif successful_test and current_price > prices_df['close'].iloc[-20]:
            pattern_name = "Successful Test of Support"
        elif successful_test and current_price < prices_df['close'].iloc[-20]:
            pattern_name = "Successful Test of Resistance"
        elif failed_test and current_price > prices_df['close'].iloc[-20]:
            pattern_name = "Failed Test of Resistance"
        elif failed_test and current_price < prices_df['close'].iloc[-20]:
            pattern_name = "Failed Test of Support"
        else:
            pattern_name = "Complex Pattern"
    else:
        pattern_name = "No Significant Pattern"
    
    return {
        "patterns_identified": pattern_detected,
        "pattern_name": pattern_name,
        "range_expansion": bool(range_expansion),
        "range_contraction": bool(range_contraction),
        "crawling_pattern": bool(crawling_pattern),
        "successful_test": bool(successful_test),
        "failed_test": bool(failed_test),
        "consolidation_break": bool(consolidation_break),
        "recent_high": float(recent_high),
        "recent_low": float(recent_low),
        "price_range": float(price_range),
        "reason": f"Pattern identified: {pattern_name}"
    }


def analyze_volume_patterns(prices_df: pd.DataFrame) -> dict:
    """
    Analyze volume patterns that Livermore considered significant.
    Livermore believed volume confirmed price action and often preceded price movement.
    """
    if len(prices_df) < 60 or 'volume' not in prices_df.columns:
        return {"volume_analysis_complete": False, "reason": "Insufficient volume data"}
    
    # Calculate recent average volume
    recent_volume = prices_df['volume'].iloc[-10:].mean()
    longer_term_volume = prices_df['volume'].iloc[-30:].mean()
    
    # Volume increase/decrease
    volume_increasing = recent_volume > longer_term_volume * 1.2
    volume_decreasing = recent_volume < longer_term_volume * 0.8
    
    # Livermore watched for volume expansion/contraction
    volume_expansion = prices_df['volume'].iloc[-5:].mean() > prices_df['volume'].iloc[-10:-5].mean() * 1.5
    volume_contraction = prices_df['volume'].iloc[-5:].mean() < prices_df['volume'].iloc[-10:-5].mean() * 0.5
    
    # Livermore paid attention to effort vs. result
    # High volume with little price movement often indicated distribution or accumulation
    recent_price_range = (prices_df['high'].iloc[-5:] - prices_df['low'].iloc[-5:]).mean()
    recent_volume_per_range = recent_volume / recent_price_range if recent_price_range > 0 else 0
    
    previous_price_range = (prices_df['high'].iloc[-10:-5] - prices_df['low'].iloc[-10:-5]).mean()
    previous_volume_per_range = longer_term_volume / previous_price_range if previous_price_range > 0 else 0
    
    # Increasing effort for diminishing results (potential trend exhaustion)
    increasing_effort_diminishing_result = recent_volume_per_range > previous_volume_per_range * 1.5
    
    # Volume on up days vs down days
    # Livermore noted when volume was higher on up days (accumulation) vs down days (distribution)
    recent_up_days = prices_df.iloc[-20:][prices_df['close'].iloc[-20:] > prices_df['open'].iloc[-20:]]
    recent_down_days = prices_df.iloc[-20:][prices_df['close'].iloc[-20:] < prices_df['open'].iloc[-20:]]
    
    avg_up_volume = recent_up_days['volume'].mean() if not recent_up_days.empty else 0
    avg_down_volume = recent_down_days['volume'].mean() if not recent_down_days.empty else float('inf')
    
    # Higher volume on up days indicates accumulation
    higher_volume_on_up_days = avg_up_volume > avg_down_volume if avg_down_volume != float('inf') else False
    
    # Higher volume on down days indicates distribution
    higher_volume_on_down_days = avg_down_volume > avg_up_volume if avg_up_volume > 0 else False
    
    # Livermore's concept of stopping volume (climactic volume)
    # A spike in volume often marked the end of a move
    recent_max_volume = prices_df['volume'].iloc[-10:].max()
    stopping_volume = recent_max_volume > prices_df['volume'].iloc[-30:-10].mean() * 3
    
    # Climactic volume day location
    climactic_volume_day_index = prices_df['volume'].iloc[-10:].idxmax()
    days_since_climax = len(prices_df) - 1 - prices_df.index.get_loc(climactic_volume_day_index) if stopping_volume else None
    
    # Determine overall volume pattern
    if stopping_volume and days_since_climax is not None and days_since_climax < 3:
        volume_pattern = "Recent Climactic Volume"
    elif volume_expansion and higher_volume_on_up_days:
        volume_pattern = "Bullish Volume Expansion"
    elif volume_expansion and higher_volume_on_down_days:
        volume_pattern = "Bearish Volume Expansion"
    elif volume_contraction:
        volume_pattern = "Volume Contraction"
    elif increasing_effort_diminishing_result:
        volume_pattern = "Effort vs Result Divergence"
    elif higher_volume_on_up_days:
        volume_pattern = "Accumulation"
    elif higher_volume_on_down_days:
        volume_pattern = "Distribution"
    else:
        volume_pattern = "Neutral Volume"
    
    return {
        "volume_analysis_complete": True,
        "volume_pattern": volume_pattern,
        "volume_increasing": bool(volume_increasing),
        "volume_decreasing": bool(volume_decreasing),
        "volume_expansion": bool(volume_expansion),
        "volume_contraction": bool(volume_contraction),
        "higher_volume_on_up_days": bool(higher_volume_on_up_days),
        "higher_volume_on_down_days": bool(higher_volume_on_down_days),
        "stopping_volume": bool(stopping_volume),
        "effort_result_divergence": bool(increasing_effort_diminishing_result),
        "days_since_climax": days_since_climax,
        "avg_up_volume": float(avg_up_volume) if avg_up_volume else None,
        "avg_down_volume": float(avg_down_volume) if avg_down_volume != float('inf') else None,
        "reason": f"Volume pattern: {volume_pattern}"
    }


def calculate_position_parameters(prices_df: pd.DataFrame) -> dict:
    """
    Calculate position sizing and risk parameters based on Livermore's methods.
    Livermore was known for scaling into winning positions and cutting losses quickly.
    """
    if len(prices_df) < 30:
        return {"position_sizing_complete": False, "reason": "Insufficient price history"}
    
    # Current price and recent volatility
    current_price = prices_df['close'].iloc[-1]
    
    # Livermore would use natural market points for stop losses
    # For simplicity, we'll use a volatility-based approach
    atr_period = 14
    high_low = prices_df['high'] - prices_df['low']
    high_close = abs(prices_df['high'] - prices_df['close'].shift())
    low_close = abs(prices_df['low'] - prices_df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]
    
    # Livermore was known for tight stop losses
    initial_stop_loss = current_price - (2 * atr)
    stop_loss_percentage = ((current_price - initial_stop_loss) / current_price) * 100
    
    # Livermore scaled into positions at key points
    # For simplicity, we'll define potential entry/scaling points
    entry_point = current_price
    scale_in_point_1 = current_price * 1.03  # 3% higher
    scale_in_point_2 = current_price * 1.07  # 7% higher
    
    # Potential profit targets
    # Livermore didn't use fixed profit targets but traded the trend
    # These are simplified approximations
    pivot_points = identify_pivot_points(prices_df)
    recent_pivot_high = pivot_points.get("recent_pivot_high")
    recent_pivot_low = pivot_points.get("recent_pivot_low")
    
    # Define potential profit targets based on pivot points
    if recent_pivot_high is not None and recent_pivot_low is not None:
        price_range = recent_pivot_high - recent_pivot_low
        profit_target_1 = recent_pivot_high + (price_range * 0.5)
        profit_target_2 = recent_pivot_high + price_range
    else:
        # Fallback if pivot points not available
        profit_target_1 = current_price * 1.15  # 15% higher
        profit_target_2 = current_price * 1.30  # 30% higher
    
    # Livermore's concept of "testing the waters" - starting with a small position
    # Then adding as the trend confirms
    suggested_initial_position = 0.25  # 25% of intended full position
    suggested_scaling_increment = 0.25  # Add 25% at each scale point
    
    # Calculate risk-reward ratio
    risk = current_price - initial_stop_loss
    reward_target_1 = profit_target_1 - current_price
    risk_reward_ratio = reward_target_1 / risk if risk > 0 else 0
    
    return {
        "position_sizing_complete": True,
        "entry_point": float(entry_point),
        "initial_stop_loss": float(initial_stop_loss),
        "stop_loss_percentage": float(stop_loss_percentage),
        "scale_in_point_1": float(scale_in_point_1),
        "scale_in_point_2": float(scale_in_point_2),
        "profit_target_1": float(profit_target_1),
        "profit_target_2": float(profit_target_2),
        "suggested_initial_position": float(suggested_initial_position),
        "suggested_scaling_increment": float(suggested_scaling_increment),
        "risk_reward_ratio": float(risk_reward_ratio),
        "reason": f"Risk/Reward ratio: {risk_reward_ratio:.2f}, "
                 f"Stop loss: {stop_loss_percentage:.1f}%, "
                 f"Suggested initial position: {suggested_initial_position*100}% of capital"
    }


def generate_livermore_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> JesseLivermoreSignal:
    """Generate the final output using the LLM."""
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Jesse Livermore, a legendary speculative trader from the early 20th century known as "The Boy Plunger."
                
                Your trading approach focuses on:
                
                1. Trading with the path of least resistance (following the market's dominant trend)
                2. Identifying pivotal points where major trends begin or end
                3. Watching for confirming price action - price must prove itself before you act
                4. Cutting losses quickly without hesitation
                5. Scaling into winning positions as they prove themselves
                6. Trading based on price action, not opinions or fundamentals
                7. Recognizing when a stock is "acting right" versus showing concerning behavior
                
                You famously said:
                - "Markets are never wrong – opinions often are."
                - "The big money is made by sitting, not thinking."
                - "It was never my thinking that made the big money for me. It was always my sitting."
                
                As Livermore, provide decisive trading signals based on your analysis of price action, momentum, and volume behavior.
                
                IMPORTANT: Your response must be a JSON object with these exact fields:
                {{
                    "signal": "bullish" or "bearish" or "neutral", 
                    "confidence": a float value between 0.0 and 100.0,
                    "reasoning": "detailed explanation in Livermore's style"
                }}
                """,
            ),
            (
                "human",
                """Analyze this stock using your speculative trading methodology:
                
                Ticker: {ticker}
                Current Price: ${current_price}
                Market Cap: {market_cap}
                
                Pivot Point Analysis:
                {pivots}
                
                Trend and Momentum Analysis:
                {trend}
                
                Price Pattern Analysis:
                {patterns}
                
                Volume Analysis:
                {volume}
                
                Position Parameters:
                {money_management}
                
                Based on your analysis, provide your trading signal with confidence level (0-100) and detailed reasoning in your authentic voice. Explain why this stock is either worth speculating on or should be avoided based on your proven methodology.
                
                REMEMBER: Your response must be a valid JSON with exactly these three fields:
                {{
                    "signal": "bullish" or "bearish" or "neutral",
                    "confidence": a number between 0 and 100,
                    "reasoning": "your detailed explanation"
                }}
                """,
            ),
        ]
    )
    
    # Format the prompt with the analysis data
    prompt = template.invoke(
        {
            "ticker": ticker,
            "current_price": analysis_data["current_price"],
            "market_cap": analysis_data["market_cap"],
            "pivots": json.dumps(analysis_data["pivots"], indent=2),
            "trend": json.dumps(analysis_data["trend"], indent=2),
            "patterns": json.dumps(analysis_data["patterns"], indent=2),
            "volume": json.dumps(analysis_data["volume"], indent=2),
            "money_management": json.dumps(analysis_data["money_management"], indent=2),
        }
    )
    
    # Create default factory for JesseLivermoreSignal
    def create_default_livermore_signal():
        return JesseLivermoreSignal(
            signal="neutral",
            confidence=0.0,
            reasoning=f"Could not analyze {ticker} due to an error. As Livermore would say, when in doubt, stay out.",
        )
    
    # Call the LLM
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=JesseLivermoreSignal,
        agent_name="jesse_livermore_agent",
        default_factory=create_default_livermore_signal,
    ) 