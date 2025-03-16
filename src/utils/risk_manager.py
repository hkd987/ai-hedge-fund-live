"""
Risk Management Module for AI Hedge Fund
----------------------------------------
This module provides risk management functionality to protect the portfolio
and limit losses during live trading.
"""

import logging
import os
from datetime import datetime, date, timedelta
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('risk_manager')

# Default risk parameters that can be overridden by environment variables
DEFAULT_RISK_PARAMS = {
    # Stop loss and take profit settings
    "STOP_LOSS_PCT": 0.05,            # 5% stop loss below purchase price
    "TAKE_PROFIT_PCT": 0.20,          # 20% take profit above purchase price
    "TRAILING_STOP_PCT": 0.03,        # 3% trailing stop when in profit
    
    # Position sizing constraints
    "MAX_POSITION_SIZE_PCT": 0.10,    # No position can be > 10% of portfolio
    "MAX_SECTOR_EXPOSURE_PCT": 0.25,  # No sector can be > 25% of portfolio
    
    # Daily trading limits
    "MAX_DAILY_TRADES": 10,           # Maximum number of trades per day
    "MAX_DAILY_CAPITAL_PCT": 0.20,    # Maximum capital to deploy in a day (20%)
    
    # Portfolio-wide risk controls
    "MAX_DRAWDOWN_PCT": 0.10,         # Maximum portfolio drawdown allowed (10%)
    "DAILY_LOSS_CIRCUIT_BREAKER_PCT": 0.03,  # Stop trading if portfolio loses 3% in a day
    
    # Market condition adjustments
    "HIGH_VOLATILITY_REDUCTION_PCT": 0.50,  # Reduce position sizes by 50% during high volatility
    "HIGH_VOLATILITY_VIX_THRESHOLD": 25,    # VIX threshold for high volatility
}

# Load custom risk parameters from environment variables
RISK_PARAMS = {}
for key, default_value in DEFAULT_RISK_PARAMS.items():
    env_value = os.getenv(key)
    if env_value is not None:
        try:
            # Convert the environment variable to the same type as the default value
            if isinstance(default_value, float):
                RISK_PARAMS[key] = float(env_value)
            elif isinstance(default_value, int):
                RISK_PARAMS[key] = int(env_value)
            else:
                RISK_PARAMS[key] = env_value
        except ValueError:
            logger.warning(f"Invalid value for {key} in environment variables. Using default: {default_value}")
            RISK_PARAMS[key] = default_value
    else:
        RISK_PARAMS[key] = default_value

# Data storage for tracking today's activities
TODAY_STATE = {
    "date": date.today(),
    "starting_portfolio_value": None,
    "current_portfolio_value": None, 
    "trades_executed": 0,
    "capital_deployed": 0.0,
    "highest_portfolio_value": 0.0,
    "transactions": [],
    "circuit_breaker_triggered": False
}

def reset_daily_state(portfolio_value: float) -> None:
    """Reset the daily state tracking with a new starting portfolio value."""
    global TODAY_STATE
    
    TODAY_STATE = {
        "date": date.today(),
        "starting_portfolio_value": portfolio_value,
        "current_portfolio_value": portfolio_value,
        "highest_portfolio_value": portfolio_value,
        "trades_executed": 0,
        "capital_deployed": 0.0,
        "transactions": [],
        "circuit_breaker_triggered": False
    }
    logger.info(f"Daily risk state reset with starting portfolio value: ${portfolio_value:.2f}")
    
    # Save the state to a file for persistence across runs
    save_risk_state()

def update_portfolio_value(current_value: float) -> None:
    """Update the current and highest portfolio values."""
    global TODAY_STATE
    
    # Check if we need to reset for a new day
    if TODAY_STATE["date"] != date.today():
        reset_daily_state(current_value)
        return
    
    TODAY_STATE["current_portfolio_value"] = current_value
    
    # Update highest value if current value is higher
    if current_value > TODAY_STATE["highest_portfolio_value"]:
        TODAY_STATE["highest_portfolio_value"] = current_value
    
    # Save the updated state
    save_risk_state()

def save_risk_state() -> None:
    """Save the current risk state to a file."""
    state_copy = TODAY_STATE.copy()
    state_copy["date"] = state_copy["date"].isoformat()
    
    try:
        # Ensure the directory exists
        os.makedirs('data/risk', exist_ok=True)
        
        # Save the state
        with open('data/risk/daily_state.json', 'w') as f:
            json.dump(state_copy, f)
    except Exception as e:
        logger.error(f"Failed to save risk state: {e}")

def load_risk_state() -> None:
    """Load the risk state from a file."""
    global TODAY_STATE
    
    try:
        # Check if the file exists
        if not os.path.exists('data/risk/daily_state.json'):
            # If the file doesn't exist, we'll just use the default state
            return
        
        with open('data/risk/daily_state.json', 'r') as f:
            loaded_state = json.load(f)
            
        # Convert date string back to date object
        loaded_state["date"] = datetime.fromisoformat(loaded_state["date"]).date()
        
        # Only use the loaded state if it's from today
        if loaded_state["date"] == date.today():
            TODAY_STATE = loaded_state
            logger.info(f"Loaded risk state from file. Trades executed today: {TODAY_STATE['trades_executed']}")
        else:
            logger.info("Loaded risk state is from a different day. Using default state.")
    except Exception as e:
        logger.error(f"Failed to load risk state: {e}")

def check_position_size_limit(ticker: str, proposed_value: float, portfolio_value: float) -> Tuple[bool, float]:
    """
    Check if a proposed position size exceeds the maximum allowed percentage of the portfolio.
    Returns (is_allowed, adjusted_value)
    """
    max_position_size = portfolio_value * RISK_PARAMS["MAX_POSITION_SIZE_PCT"]
    
    if proposed_value > max_position_size:
        logger.warning(f"Position size for {ticker} (${proposed_value:.2f}) exceeds maximum allowed (${max_position_size:.2f})")
        return False, max_position_size
    
    return True, proposed_value

def check_daily_trading_limits(proposed_trade_value: float, portfolio_value: float) -> bool:
    """Check if the proposed trade violates daily trading limits."""
    # Check number of trades limit
    if TODAY_STATE["trades_executed"] >= RISK_PARAMS["MAX_DAILY_TRADES"]:
        logger.warning(f"Daily trade limit ({RISK_PARAMS['MAX_DAILY_TRADES']}) reached. Trade rejected.")
        return False
    
    # Check daily capital deployment limit
    max_daily_capital = portfolio_value * RISK_PARAMS["MAX_DAILY_CAPITAL_PCT"]
    if TODAY_STATE["capital_deployed"] + proposed_trade_value > max_daily_capital:
        logger.warning(f"Daily capital deployment limit (${max_daily_capital:.2f}) would be exceeded. Trade rejected.")
        return False
    
    return True

def record_trade_execution(ticker: str, action: str, quantity: int, price: float, total_value: float) -> None:
    """Record a trade execution for daily tracking."""
    global TODAY_STATE
    
    TODAY_STATE["trades_executed"] += 1
    
    # For buys and shorts, we increase the capital deployed
    if action in ["buy", "short"]:
        TODAY_STATE["capital_deployed"] += total_value
    
    # Record the transaction
    transaction = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "price": price,
        "total_value": total_value
    }
    TODAY_STATE["transactions"].append(transaction)
    
    # Save the updated state
    save_risk_state()
    
    logger.info(f"Trade recorded: {action} {quantity} shares of {ticker} at ${price:.2f} (${total_value:.2f})")
    logger.info(f"Daily stats: Trades={TODAY_STATE['trades_executed']}, Capital Deployed=${TODAY_STATE['capital_deployed']:.2f}")

def check_drawdown_limits(portfolio_value: float) -> bool:
    """
    Check if the portfolio has exceeded the maximum allowed drawdown.
    Returns True if trading should continue, False if trading should stop.
    """
    if TODAY_STATE["starting_portfolio_value"] is None:
        # Can't calculate drawdown if we don't have a starting value
        return True
    
    # Calculate the daily drawdown
    if TODAY_STATE["highest_portfolio_value"] > 0:
        current_drawdown = (TODAY_STATE["highest_portfolio_value"] - portfolio_value) / TODAY_STATE["highest_portfolio_value"]
        
        # Check against the daily loss circuit breaker
        if current_drawdown > RISK_PARAMS["DAILY_LOSS_CIRCUIT_BREAKER_PCT"]:
            logger.warning(f"CIRCUIT BREAKER TRIGGERED: Portfolio down {current_drawdown*100:.2f}% from today's high")
            TODAY_STATE["circuit_breaker_triggered"] = True
            save_risk_state()
            return False
    
    # Update the current portfolio value
    update_portfolio_value(portfolio_value)
    return True

def should_adjust_for_volatility(vix_value: Optional[float] = None) -> Tuple[bool, float]:
    """
    Determine if position sizes should be adjusted based on market volatility.
    Returns (should_adjust, adjustment_factor)
    """
    # If VIX value is not provided, we default to not adjusting
    if vix_value is None:
        return False, 1.0
    
    if vix_value > RISK_PARAMS["HIGH_VOLATILITY_VIX_THRESHOLD"]:
        adjustment_factor = 1.0 - RISK_PARAMS["HIGH_VOLATILITY_REDUCTION_PCT"]
        logger.info(f"High volatility detected (VIX: {vix_value:.2f}). Reducing position sizes by {RISK_PARAMS['HIGH_VOLATILITY_REDUCTION_PCT']*100:.0f}%")
        return True, adjustment_factor
    
    return False, 1.0

def can_execute_trade(ticker: str, action: str, quantity: int, price: float) -> bool:
    """
    Primary risk check function that combines all risk management rules.
    Returns True if the trade can be executed, False otherwise.
    """
    # Skip if the circuit breaker has been triggered
    if TODAY_STATE.get("circuit_breaker_triggered", False):
        logger.warning("Circuit breaker active. All trading is suspended for today.")
        return False
    
    # Calculate the total value of the trade
    total_value = quantity * price
    
    # Skip if there's no starting portfolio value set yet
    if TODAY_STATE["starting_portfolio_value"] is None:
        logger.warning("No starting portfolio value set. Can't perform risk checks.")
        return False
    
    portfolio_value = TODAY_STATE["current_portfolio_value"] or TODAY_STATE["starting_portfolio_value"]
    
    # Check position size limits
    passes_size_limit, adjusted_value = check_position_size_limit(ticker, total_value, portfolio_value)
    if not passes_size_limit:
        return False
    
    # Check daily trading limits
    if not check_daily_trading_limits(total_value, portfolio_value):
        return False
    
    # Check drawdown limits
    if not check_drawdown_limits(portfolio_value):
        return False
    
    # If we passed all checks, the trade can be executed
    return True

# Initialize by loading the previous state
load_risk_state() 