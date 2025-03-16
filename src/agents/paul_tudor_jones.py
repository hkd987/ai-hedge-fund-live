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
from datetime import datetime, timedelta
from utils.caching import cached_analyst


class PaulTudorJonesSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


@cached_analyst()
def paul_tudor_jones_agent(state: AgentState):
    """
    Analyzes stocks using Tudor Jones' technical and macro principles.
    
    Paul Tudor Jones is known for:
    1. Technical analysis and price action
    2. Macro trend identification
    3. Risk management (never risking more than 1-2% on a single trade)
    4. Contrarian positioning at market extremes
    5. Looking for 5:1 reward-to-risk ratio trades
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis for LLM reasoning
    analysis_data = {}
    tudor_jones_analysis = {}

    for ticker in tickers:
        progress.update_status("paul_tudor_jones_agent", ticker, "Fetching price data")
        # Get price data for technical analysis
        # If end_date is None, use today's date
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Calculate start_date by subtracting 180 days from end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - timedelta(days=180)
        start_date = start_date_obj.strftime("%Y-%m-%d")
        
        # Call get_prices with start_date and end_date instead of days parameter
        prices = get_prices(ticker, start_date=start_date, end_date=end_date)  # 6 months of data
        
        if not prices:
            progress.update_status("paul_tudor_jones_agent", ticker, "Failed: No price data found")
            continue
            
        prices_df = prices_to_df(prices)

        progress.update_status("paul_tudor_jones_agent", ticker, "Analyzing technical indicators")
        # Calculate technical indicators
        tech_analysis = analyze_technicals(prices_df)
        
        progress.update_status("paul_tudor_jones_agent", ticker, "Analyzing price momentum")
        # Analyze price momentum and trend
        momentum_analysis = analyze_momentum(prices_df)
        
        progress.update_status("paul_tudor_jones_agent", ticker, "Analyzing volatility")
        # Analyze volatility patterns
        volatility_analysis = analyze_volatility(prices_df)
        
        progress.update_status("paul_tudor_jones_agent", ticker, "Evaluating risk-reward ratio")
        # Calculate potential risk-reward ratio
        risk_reward = calculate_risk_reward(prices_df)

        # Get market cap to determine if it's a liquid enough stock
        progress.update_status("paul_tudor_jones_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)
        
        # Combine all analysis
        analysis_data[ticker] = {
            "technical_analysis": tech_analysis,
            "momentum_analysis": momentum_analysis,
            "volatility_analysis": volatility_analysis,
            "risk_reward_ratio": risk_reward,
            "market_cap": market_cap,
            "current_price": float(prices_df["close"].iloc[-1]) if not prices_df.empty else None,
            "price_history": {
                "last_week": float(prices_df["close"].iloc[-5]) if len(prices_df) >= 5 else None,
                "last_month": float(prices_df["close"].iloc[-20]) if len(prices_df) >= 20 else None,
                "last_quarter": float(prices_df["close"].iloc[-60]) if len(prices_df) >= 60 else None,
            }
        }

        # Generate Tudor Jones' trading signal using LLM
        progress.update_status("paul_tudor_jones_agent", ticker, "Generating Tudor Jones' analysis")
        tudor_jones_analysis[ticker] = generate_tudor_jones_output(
            ticker,
            analysis_data[ticker],
            state["metadata"]["model_name"],
            state["metadata"]["model_provider"],
        )

        progress.update_status("paul_tudor_jones_agent", ticker, "Done")

    # Create and return message
    message = HumanMessage(
        content=json.dumps({ticker: signal.model_dump() for ticker, signal in tudor_jones_analysis.items()}),
        name="paul_tudor_jones_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: signal.model_dump() for ticker, signal in tudor_jones_analysis.items()}, "Paul Tudor Jones Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["paul_tudor_jones_agent"] = tudor_jones_analysis

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def analyze_technicals(prices_df: pd.DataFrame) -> dict:
    """Calculates key technical indicators that Paul Tudor Jones might use."""
    if prices_df.empty:
        return {"error": "No price data available"}
    
    # Make a copy to avoid modifying the original
    df = prices_df.copy()
    
    # Calculate moving averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_StdDev'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
    
    # Calculate Average True Range (ATR) for volatility
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Get the last values for analysis
    last_values = df.iloc[-1].to_dict()
    
    # Check for golden cross / death cross (SMA 50 crossing SMA 200)
    golden_cross = (df['SMA_50'].iloc[-2] <= df['SMA_200'].iloc[-2]) and (df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1])
    death_cross = (df['SMA_50'].iloc[-2] >= df['SMA_200'].iloc[-2]) and (df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1])
    
    # Determine if price is above or below key moving averages
    price_above_sma_20 = df['close'].iloc[-1] > df['SMA_20'].iloc[-1]
    price_above_sma_50 = df['close'].iloc[-1] > df['SMA_50'].iloc[-1]
    price_above_sma_200 = df['close'].iloc[-1] > df['SMA_200'].iloc[-1]
    
    # Check for bullish MACD crossover
    macd_bullish_crossover = (df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]) and (df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1])
    macd_bearish_crossover = (df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]) and (df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1])
    
    # Check Bollinger Band signals
    bb_upper_touch = df['high'].iloc[-1] >= df['BB_Upper'].iloc[-1]
    bb_lower_touch = df['low'].iloc[-1] <= df['BB_Lower'].iloc[-1]
    
    # Compile the analysis
    return {
        "moving_averages": {
            "sma_20": float(last_values.get('SMA_20', 0)),
            "sma_50": float(last_values.get('SMA_50', 0)),
            "sma_200": float(last_values.get('SMA_200', 0)),
            "price_above_sma_20": price_above_sma_20,
            "price_above_sma_50": price_above_sma_50,
            "price_above_sma_200": price_above_sma_200,
            "golden_cross": golden_cross,
            "death_cross": death_cross
        },
        "oscillators": {
            "rsi": float(last_values.get('RSI', 0)),
            "rsi_overbought": last_values.get('RSI', 0) > 70,
            "rsi_oversold": last_values.get('RSI', 0) < 30
        },
        "macd": {
            "macd_value": float(last_values.get('MACD', 0)),
            "macd_signal": float(last_values.get('MACD_Signal', 0)),
            "macd_histogram": float(last_values.get('MACD_Histogram', 0)),
            "bullish_crossover": macd_bullish_crossover,
            "bearish_crossover": macd_bearish_crossover
        },
        "bollinger_bands": {
            "upper": float(last_values.get('BB_Upper', 0)),
            "middle": float(last_values.get('BB_Middle', 0)),
            "lower": float(last_values.get('BB_Lower', 0)),
            "touch_upper": bb_upper_touch,
            "touch_lower": bb_lower_touch
        },
        "volatility": {
            "atr": float(last_values.get('ATR', 0)),
            "atr_percent": float(last_values.get('ATR', 0) / df['close'].iloc[-1] * 100)
        }
    }


def analyze_momentum(prices_df: pd.DataFrame) -> dict:
    """Analyzes price momentum and trend strength, key aspects of Tudor Jones' strategy."""
    if prices_df.empty:
        return {"error": "No price data available"}
    
    df = prices_df.copy()
    
    # Calculate price changes over different periods
    df['1d_change'] = df['close'].pct_change(1)
    df['5d_change'] = df['close'].pct_change(5)
    df['20d_change'] = df['close'].pct_change(20)
    df['60d_change'] = df['close'].pct_change(60)
    
    # Calculate rate of change (ROC)
    df['ROC_10'] = df['close'].pct_change(10) * 100
    
    # Calculate Average Directional Index (ADX) for trend strength
    # First, calculate True Range
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate +DM and -DM
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
    minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
    
    # Calculate smoothed TR, +DM and -DM
    smoothing_period = 14
    smoothed_tr = tr.rolling(smoothing_period).sum()
    smoothed_plus_dm = plus_dm.rolling(smoothing_period).sum()
    smoothed_minus_dm = minus_dm.rolling(smoothing_period).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
    
    # Calculate directional movement index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    df['ADX'] = dx.rolling(smoothing_period).mean()
    
    # Determine if we're in a strong trend (ADX > 25)
    strong_trend = df['ADX'].iloc[-1] > 25 if not pd.isna(df['ADX'].iloc[-1]) else False
    
    # Determine if momentum is increasing or decreasing
    momentum_increasing = df['ROC_10'].iloc[-1] > df['ROC_10'].iloc[-5] if len(df) >= 5 else False
    
    # Create volume-weighted price trends
    df['volume_price'] = df['close'] * df['volume']
    df['vwap_5d'] = df['volume_price'].rolling(5).sum() / df['volume'].rolling(5).sum()
    
    # Check if price is above VWAP (bullish) or below (bearish)
    above_vwap = df['close'].iloc[-1] > df['vwap_5d'].iloc[-1] if not pd.isna(df['vwap_5d'].iloc[-1]) else False
    
    # Get the last values
    last_values = df.iloc[-1].to_dict()
    
    return {
        "price_changes": {
            "daily": float(last_values.get('1d_change', 0)) * 100,
            "weekly": float(last_values.get('5d_change', 0)) * 100,
            "monthly": float(last_values.get('20d_change', 0)) * 100,
            "quarterly": float(last_values.get('60d_change', 0)) * 100
        },
        "rate_of_change": {
            "roc_10": float(last_values.get('ROC_10', 0)),
            "momentum_increasing": momentum_increasing
        },
        "trend_strength": {
            "adx": float(last_values.get('ADX', 0)) if not pd.isna(last_values.get('ADX', 0)) else 0,
            "strong_trend": strong_trend
        },
        "volume_trends": {
            "above_vwap": above_vwap,
            "volume_momentum": df['volume'].iloc[-5:].mean() > df['volume'].iloc[-10:-5].mean() if len(df) >= 10 else False
        },
        "swing_trade_setup": {
            "trend_aligned_pullback": strong_trend and ((df['ROC_10'].iloc[-1] < 0 and df['20d_change'].iloc[-1] > 0) or 
                                                       (df['ROC_10'].iloc[-1] > 0 and df['20d_change'].iloc[-1] < 0)),
            "momentum_divergence": (df['close'].iloc[-1] > df['close'].iloc[-5] and df['ROC_10'].iloc[-1] < df['ROC_10'].iloc[-5]) or
                                 (df['close'].iloc[-1] < df['close'].iloc[-5] and df['ROC_10'].iloc[-1] > df['ROC_10'].iloc[-5])
        }
    }


def analyze_volatility(prices_df: pd.DataFrame) -> dict:
    """
    Analyzes volatility patterns to identify potential market turning points.
    Tudor Jones is known for identifying opportunities during volatile periods.
    """
    if prices_df.empty:
        return {"error": "No price data available"}
    
    df = prices_df.copy()
    
    # Calculate historical volatility (20-day standard deviation of returns)
    df['returns'] = df['close'].pct_change()
    df['20d_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100  # Annualized
    
    # Calculate volatility ratio (current vs historical)
    current_volatility = df['20d_volatility'].iloc[-1]
    historical_volatility = df['20d_volatility'].iloc[-60:-20].mean() if len(df) >= 60 else df['20d_volatility'].mean()
    volatility_ratio = current_volatility / historical_volatility if historical_volatility else 1.0
    
    # Calculate volatility trend
    volatility_increasing = df['20d_volatility'].iloc[-1] > df['20d_volatility'].iloc[-10] if len(df) >= 10 else False
    
    # Bollinger Band width as volatility indicator
    has_bb_columns = all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle'])
    
    if has_bb_columns:
        # Calculate BB width only if all necessary columns exist
        df['BB_width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_width'] = df['BB_width'].replace([np.inf, -np.inf], np.nan)  # Handle division by zero
        
        # Check if we have valid BB width values before comparing
        if len(df) >= 5 and not df['BB_width'].iloc[-1] is None and not df['BB_width'].iloc[-5] is None:
            bb_width_expanding = df['BB_width'].iloc[-1] > df['BB_width'].iloc[-5]
        else:
            bb_width_expanding = False
            
        # Only calculate percentiles if we have valid data
        if not df['BB_width'].dropna().empty:
            bb_width_percentile = np.percentile(df['BB_width'].dropna(), [25, 50, 75, 90])
            current_bb_width = df['BB_width'].iloc[-1]
            if current_bb_width is not None:
                bb_width_rank = sum(current_bb_width > p for p in bb_width_percentile) / 4
            else:
                bb_width_rank = 0.5  # Neutral value if we can't calculate
        else:
            bb_width_rank = 0.5  # Neutral value if no data
    else:
        bb_width_expanding = False
        bb_width_rank = 0.5  # Neutral value
    
    # Calculate max drawdown in the period
    df['rolling_max'] = df['close'].rolling(window=60, min_periods=1).max()
    df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max']
    max_drawdown = df['drawdown'].min() * 100
    
    # Identify volatility breakouts
    if 'ATR' in df:
        atr_percentile = np.percentile(df['ATR'].dropna(), [50, 75, 90])
        atr_breakout = df['ATR'].iloc[-1] > atr_percentile[1]  # Above 75th percentile
    else:
        atr_breakout = None
    
    return {
        "current_volatility": float(current_volatility) if not pd.isna(current_volatility) else 0,
        "historical_volatility": float(historical_volatility) if not pd.isna(historical_volatility) else 0,
        "volatility_ratio": float(volatility_ratio) if not pd.isna(volatility_ratio) else 1.0,
        "volatility_trend": {
            "increasing": volatility_increasing,
            "bollinger_band_expanding": bb_width_expanding,
            "bb_width_percentile": float(bb_width_rank) if bb_width_rank is not None else None
        },
        "market_stress": {
            "max_drawdown": float(max_drawdown),
            "high_volatility_environment": volatility_ratio > 1.5,
            "volatility_breakout": atr_breakout
        },
        "contrarian_signals": {
            "extreme_volatility": volatility_ratio > 2.0,
            "volatility_collapse": volatility_ratio < 0.5,
            "extreme_drawdown": max_drawdown < -15
        }
    }


def calculate_risk_reward(prices_df: pd.DataFrame) -> dict:
    """
    Calculates potential risk-reward scenarios for swing trades.
    Tudor Jones is known for seeking 5:1 reward-to-risk opportunities.
    """
    if prices_df.empty:
        return {"error": "No price data available"}
    
    df = prices_df.copy()
    current_price = df['close'].iloc[-1]
    
    # Use ATR to determine potential stop loss level (1.5 * ATR)
    if 'ATR' in df and not pd.isna(df['ATR'].iloc[-1]):
        atr = df['ATR'].iloc[-1]
        stop_loss_distance = 1.5 * atr
    else:
        # If ATR is not available, use 2% of current price
        stop_loss_distance = current_price * 0.02
    
    # Determine potential stop loss prices
    stop_loss_price_long = current_price - stop_loss_distance
    stop_loss_price_short = current_price + stop_loss_distance
    
    # Use technical levels for potential targets
    # For a long trade, consider recent highs or resistance
    recent_high = df['high'].iloc[-20:].max() if len(df) >= 20 else df['high'].max()
    recent_low = df['low'].iloc[-20:].min() if len(df) >= 20 else df['low'].min()
    
    # For long trade: target recent high or 3 * ATR, whichever is higher
    target_long = max(recent_high, current_price + 3 * stop_loss_distance)
    
    # For short trade: target recent low or 3 * ATR, whichever is lower
    target_short = min(recent_low, current_price - 3 * stop_loss_distance)
    
    # Calculate reward-to-risk ratios
    risk_long = current_price - stop_loss_price_long
    reward_long = target_long - current_price
    ratio_long = reward_long / risk_long if risk_long > 0 else 0
    
    risk_short = stop_loss_price_short - current_price
    reward_short = current_price - target_short
    ratio_short = reward_short / risk_short if risk_short > 0 else 0
    
    # Tudor Jones looks for 5:1 opportunities
    ideal_ratio = 5.0
    
    return {
        "long_trade": {
            "entry": float(current_price),
            "stop_loss": float(stop_loss_price_long),
            "target": float(target_long),
            "risk_pct": float((current_price - stop_loss_price_long) / current_price * 100),
            "reward_pct": float((target_long - current_price) / current_price * 100),
            "risk_reward_ratio": float(ratio_long),
            "meets_tudor_jones_criteria": ratio_long >= ideal_ratio
        },
        "short_trade": {
            "entry": float(current_price),
            "stop_loss": float(stop_loss_price_short),
            "target": float(target_short),
            "risk_pct": float((stop_loss_price_short - current_price) / current_price * 100),
            "reward_pct": float((current_price - target_short) / current_price * 100),
            "risk_reward_ratio": float(ratio_short),
            "meets_tudor_jones_criteria": ratio_short >= ideal_ratio
        },
        "ideal_ratio": ideal_ratio,
        "trade_preference": "long" if ratio_long > ratio_short else "short"
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


def generate_tudor_jones_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> PaulTudorJonesSignal:
    """Generate the final output for the Paul Tudor Jones agent."""
    
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Paul Tudor Jones, a legendary global macro trader known for your ability to identify major market shifts and for your risk management approach.
                
                Your key trading principles include:
                1. Capital preservation is your #1 priority
                2. Cut losses quickly when wrong
                3. Ride winners with trailing stops
                4. Look for 5:1 risk-reward opportunities
                5. Pay close attention to market psychology and sentiment
                6. Trade with the trend but look for important pivot points
                7. Monitor the broader economic environment for major shifts
                
                You're particularly known for:
                - Your ability to identify major market tops and bottoms
                - Your emphasis on risk management
                - Your attention to price action and technical analysis
                - Your integration of macro trends with technical indicators
                
                Based on the analysis provided, give your trading signal (bullish, bearish, or neutral) with a confidence level (0-100).
                
                Provide a detailed explanation for your reasoning. Be authentic to your style - focused on price action, sentiment, and risk management.
                """
            ),
            (
                "human",
                """Analyze this stock for trading potential:
                
                Ticker: {ticker}
                
                Technical Analysis:
                {tech_analysis}
                
                Momentum Analysis:
                {momentum_analysis}
                
                Volatility Analysis:
                {volatility_analysis}
                
                Risk-Reward Analysis:
                {risk_reward}
                
                Current Price: ${current_price}
                Market Cap: ${market_cap}
                
                Price History:
                - 1 Week Ago: ${price_week_ago}
                - 1 Month Ago: ${price_month_ago}
                - 3 Months Ago: ${price_quarter_ago}
                
                Provide your signal as a JSON object with the following structure:
                {{
                    "signal": "bullish" or "bearish" or "neutral",
                    "confidence": float between 0 and 100,
                    "reasoning": "detailed explanation"
                }}
                """
            ),
        ]
    )
    
    # Convert data to serializable format
    serializable_data = convert_to_serializable(analysis_data)
    
    # Filter out None values and format
    tech_analysis = json.dumps(serializable_data.get("technical_analysis", {}), indent=2)
    momentum_analysis = json.dumps(serializable_data.get("momentum_analysis", {}), indent=2)
    volatility_analysis = json.dumps(serializable_data.get("volatility_analysis", {}), indent=2)
    risk_reward = json.dumps(serializable_data.get("risk_reward_ratio", {}), indent=2)
    current_price = serializable_data.get("current_price", "N/A")
    market_cap = serializable_data.get("market_cap", "N/A")
    
    price_history = serializable_data.get("price_history", {})
    price_week_ago = price_history.get("last_week", "N/A")
    price_month_ago = price_history.get("last_month", "N/A") 
    price_quarter_ago = price_history.get("last_quarter", "N/A")
    
    prompt = template.invoke(
        {
            "ticker": ticker,
            "tech_analysis": tech_analysis,
            "momentum_analysis": momentum_analysis,
            "volatility_analysis": volatility_analysis,
            "risk_reward": risk_reward,
            "current_price": current_price,
            "market_cap": market_cap,
            "price_week_ago": price_week_ago,
            "price_month_ago": price_month_ago,
            "price_quarter_ago": price_quarter_ago,
        }
    )
    
    def create_default_tudor_jones_signal():
        return PaulTudorJonesSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Insufficient data to make a determination. Paul Tudor Jones would wait for a clearer setup."
        )
    
    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=PaulTudorJonesSignal,
        agent_name="paul_tudor_jones_agent",
        default_factory=create_default_tudor_jones_signal,
    ) 