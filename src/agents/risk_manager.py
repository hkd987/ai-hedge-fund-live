from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json
import os
import logging

# Import the risk management utility functions
from utils.risk_manager import (
    RISK_PARAMS,
    check_position_size_limit,
    should_adjust_for_volatility,
    update_portfolio_value,
    TODAY_STATE
)

# Try to import VIX data function for volatility adjustment
try:
    from utils.market_data import get_current_vix, ALPACA_AVAILABLE as MARKET_DATA_AVAILABLE
except ImportError:
    MARKET_DATA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('risk_manager_agent')

##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls

    # Get current VIX value for volatility adjustments if available
    vix_value = None
    if MARKET_DATA_AVAILABLE:
        try:
            vix_value = get_current_vix()
            if vix_value:
                logger.info(f"Current VIX value: {vix_value}")
        except Exception as e:
            logger.warning(f"Failed to get VIX data: {e}")

    # Get volatility adjustment factor
    is_high_volatility, volatility_adjustment = should_adjust_for_volatility(vix_value)

    # Calculate total portfolio value
    total_portfolio_value = portfolio.get("cash", 0)
    for ticker in portfolio.get("positions", {}):
        position = portfolio["positions"][ticker]
        if "long" in position and position["long"] > 0:
            # Add long position value
            long_value = position["long"] * position.get("long_cost_basis", 0)
            total_portfolio_value += long_value
        
        if "short" in position and position["short"] > 0:
            # Add short position value (assuming we're tracking the value correctly)
            short_value = position["short"] * position.get("short_cost_basis", 0)
            total_portfolio_value += short_value

    # Update the risk management system with current portfolio value
    update_portfolio_value(total_portfolio_value)

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")

        # Calculate portfolio value
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price  # Store the current price

        # Calculate current position value for this ticker
        current_position = portfolio.get("positions", {}).get(ticker, {})
        long_position_value = current_position.get("long", 0) * current_position.get("long_cost_basis", 0)
        short_position_value = current_position.get("short", 0) * current_position.get("short_cost_basis", 0)
        current_position_value = long_position_value + short_position_value

        # Calculate basic position limit based on max position size parameter
        # Use the advanced position sizing logic from utils/risk_manager.py
        passes_size_limit, max_position_size = check_position_size_limit(
            ticker, 
            total_portfolio_value * RISK_PARAMS["MAX_POSITION_SIZE_PCT"], 
            total_portfolio_value
        )

        # Apply volatility adjustment if needed
        if is_high_volatility:
            max_position_size *= volatility_adjustment
            logger.info(f"Adjusting position size for {ticker} due to high volatility (VIX: {vix_value})")

        # For existing positions, subtract current position value from limit
        remaining_position_limit = max_position_size - current_position_value

        # Ensure we don't exceed available cash for new purchases
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        # Calculate maximum shares based on current price
        max_shares = int(max_position_size / current_price) if current_price > 0 else 0

        # Check if circuit breaker is active
        circuit_breaker_active = TODAY_STATE.get("circuit_breaker_triggered", False)
        if circuit_breaker_active:
            logger.warning(f"Circuit breaker active - no new trades allowed for {ticker}")
            max_position_size = 0
            max_shares = 0

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "max_shares": max_shares,
            "circuit_breaker_active": circuit_breaker_active,
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "max_position_size_pct": float(RISK_PARAMS["MAX_POSITION_SIZE_PCT"]),
                "position_limit": float(max_position_size),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "vix_value": vix_value,
                "high_volatility": is_high_volatility,
                "volatility_adjustment": volatility_adjustment if is_high_volatility else 1.0,
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
