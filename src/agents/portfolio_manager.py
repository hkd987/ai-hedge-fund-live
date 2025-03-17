import json
import os
import logging
import traceback
import time
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, time as datetime_time, timedelta, date
from pydantic import BaseModel, Field
from typing_extensions import Literal

from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from utils.caching import cached_analyst
from utils.llm import call_llm

# Import risk management module
from utils.risk_manager import (
    can_execute_trade, 
    record_trade_execution, 
    reset_daily_state,
    update_portfolio_value,
    RISK_PARAMS
)

# Import enhanced risk management functions
from utils.enhanced_risk import (
    calculate_position_risk_parameters,
    check_market_hours,
    adjust_risk_for_timing,
    generate_risk_dashboard,
    track_sector_exposure,
    check_sector_limits,
    check_correlation_risk,
    detect_market_regime,
    assess_portfolio_liquidity,
    calculate_portfolio_beta,
    get_next_earnings_date
)

# Import alpaca-py for live trading
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, 
        LimitOrderRequest, 
        StopOrderRequest,
        StopLimitOrderRequest,
        TrailingStopOrderRequest,
        BracketOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderStatus
    
    # Import Alpaca data client for price retrieval
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('portfolio_manager')

# Check for live trading environment variable
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Load risk parameters from environment
RISK_PARAMS = {
    "STOP_LOSS_PCT": float(os.getenv("STOP_LOSS_PCT", "0.05")),
    "TAKE_PROFIT_PCT": float(os.getenv("TAKE_PROFIT_PCT", "0.20")),
    "TRAILING_STOP_PCT": float(os.getenv("TRAILING_STOP_PCT", "0.03")),
}

class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")
    cancel_existing_orders: bool = Field(default=False, description="Whether to cancel existing orders for this ticker")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


def get_alpaca_client():
    """Initialize and return Alpaca trading client if credentials are available"""
    if not ALPACA_AVAILABLE:
        logger.warning("Alpaca SDK not installed. Run 'pip install alpaca-py' to enable live trading.")
        return None

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found in environment variables.")
        return None
    
    try:
        # Check if live trading is enabled in the environment
        live_trading = os.getenv("LIVE_TRADING", "false").lower() == "true"
        
        # Use paper trading unless LIVE_TRADING is explicitly set to true
        is_paper = not live_trading
        client = TradingClient(api_key, api_secret, paper=is_paper)
        
        # Log which environment we're using
        env_type = "paper trading" if is_paper else "live trading"
        logger.info(f"Initialized Alpaca client for {env_type}")
        
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return None


def get_alpaca_open_orders(client, ticker=None):
    """
    Get open orders from Alpaca
    
    Args:
        client: Alpaca client
        ticker: Optional ticker to filter orders
        
    Returns:
        Dictionary of ticker to list of orders
    """
    import logging
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus

    if not client:
        logging.warning("No Alpaca client provided for get_alpaca_open_orders")
        return {}
        
    logging.info("Attempting to get open orders from Alpaca")
    
    try:
        # Create an orders_by_ticker dictionary
        orders_by_ticker = {}
        
        # Try multiple order statuses that represent "open" orders
        statuses_to_try = ["open", "new", "partially_filled", "pending_new", "accepted"]
        
        for status_name in statuses_to_try:
            try:
                # Try to get the proper enum value if possible
                try:
                    status_enum = getattr(OrderStatus, status_name.upper())
                    request_params = GetOrdersRequest(status=status_enum)
                except AttributeError:
                    # If the enum doesn't exist, try the string directly
                    request_params = GetOrdersRequest(status=status_name)
                
                # Get the orders using the request params object
                status_orders = client.get_orders(request_params)
                
                if status_orders:
                    logging.info(f"Found {len(status_orders)} orders with status '{status_name}'")
                    
                    for order in status_orders:
                        # If a specific ticker was requested, skip others
                        if ticker and order.symbol != ticker:
                            continue
                            
                        if order.symbol not in orders_by_ticker:
                            orders_by_ticker[order.symbol] = []
                            
                        # Check if order is already in the list to avoid duplicates
                        if not any(o.id == order.id for o in orders_by_ticker[order.symbol]):
                            orders_by_ticker[order.symbol].append(order)
            except Exception as e:
                logging.warning(f"Error getting orders with status '{status_name}': {e}")
        
        # If no orders found with the above statuses, try getting all orders as a fallback
        if not orders_by_ticker:
            logging.info("No orders found with specific statuses, trying to get all orders")
            try:
                all_orders = client.get_orders()
                
                if all_orders:
                    for order in all_orders:
                        # Filter to include only open order statuses
                        if hasattr(order, 'status') and order.status in ['open', 'new', 'partially_filled', 'pending_new', 'accepted']:
                            # If a specific ticker was requested, skip others
                            if ticker and order.symbol != ticker:
                                continue
                                
                            if order.symbol not in orders_by_ticker:
                                orders_by_ticker[order.symbol] = []
                                
                            orders_by_ticker[order.symbol].append(order)
            except Exception as e:
                logging.warning(f"Error getting all orders: {e}")
        
        # Log summary of found orders
        total_orders = sum(len(orders) for orders in orders_by_ticker.values())
        logging.info(f"Found a total of {total_orders} open orders across {len(orders_by_ticker)} tickers")
        
        # Log details for specific tickers (especially AAPL and VOO which seem to be missing)
        for symbol, orders in orders_by_ticker.items():
            logging.info(f"Found {len(orders)} orders for {symbol}")
            
        return orders_by_ticker
        
    except Exception as e:
        logging.error(f"Error getting open orders: {e}")
        return {}


def execute_alpaca_trade(ticker, action, quantity, current_price, prices_df=None):
    """
    Execute a trade with Alpaca, applying risk management rules
    
    Args:
        ticker: Ticker symbol
        action: Trade action (buy, sell, short, cover)
        quantity: Number of shares
        current_price: Current price of the security
        prices_df: Optional price dataframe for dynamic stop calculation
        
    Returns:
        dict: Results of the trade execution including status and details
    """
    logger.info(f"=== EXECUTE TRADE START: {ticker} {action} {quantity} at ${current_price} ===")
    
    # Check if the trade is allowed by our risk management system
    if not can_execute_trade(ticker, action, quantity, current_price):
        logger.warning(f"Trade rejected by risk management: {action} {quantity} shares of {ticker}")
        return {
            "success": False,
            "reason": "Trade rejected by risk management",
            "ticker": ticker,
            "action": action,
            "quantity": quantity
        }
    
    # First, check market hours to see if this is an appropriate time to trade
    if prices_df is not None:
        market_hours_check = check_market_hours()
        if market_hours_check["high_risk_period"]:
            logger.warning(f"High risk trading period detected: {market_hours_check['reason']}")
            # Reduce order size by 50% during high risk periods
            quantity = max(1, int(quantity * 0.5))
            logger.info(f"Reduced order size to {quantity} shares due to market timing risk")
    
    # Get Alpaca client
    client = get_alpaca_client()
    
    # Log Alpaca client info
    if client:
        try:
            account = client.get_account()
            env_type = "paper trading" if getattr(account, 'is_paper', True) else "live trading"
            logger.info(f"Alpaca client initialized for {env_type}, account ID: {account.id}")
            logger.info(f"Account status: {account.status}, buying power: ${float(account.buying_power):.2f}")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
    else:
        logger.error("Alpaca client not available")
        return {
            "success": False, 
            "reason": "Alpaca client not available", 
            "ticker": ticker
        }
    
    # Check for existing open orders for this ticker
    open_orders = get_alpaca_open_orders(client, ticker)
    if ticker in open_orders and open_orders[ticker]:
        existing_orders = open_orders[ticker]
        logger.info(f"Found {len(existing_orders)} existing orders for {ticker}")
        
        # Log details about each existing order
        for i, order in enumerate(existing_orders):
            logger.info(f"  Order {i+1}: ID={order.id}, Status={order.status}, Side={order.side}, Qty={order.qty}")
        
        # Check if there's already an order with the same action
        for order in existing_orders:
            order_side = order.side
            is_same_action = False
            
            # Map Alpaca order side to our action
            if order_side == "buy" and action in ["buy", "cover"]:
                is_same_action = True
            elif order_side == "sell" and action in ["sell", "short"]:
                is_same_action = True
                
            if is_same_action:
                logger.warning(f"There is already an open {order_side} order for {ticker}. Skipping duplicate order.")
                return {
                    "success": False,
                    "reason": f"Duplicate {order_side} order exists",
                    "ticker": ticker,
                    "action": action
                }
    else:
        logger.info(f"No existing orders found for {ticker}")
    
    # Map our action to Alpaca's OrderSide and determine if it's a short order
    side_mapping = {
        "buy": OrderSide.BUY,
        "sell": OrderSide.SELL,
        "short": OrderSide.SELL,
        "cover": OrderSide.BUY
    }
    
    if action not in side_mapping:
        logger.error(f"Unsupported action: {action}")
        return {
            "success": False,
            "reason": f"Unsupported action: {action}",
            "ticker": ticker
        }
    
    side = side_mapping[action]
    logger.info(f"Mapped action '{action}' to OrderSide '{side}'")
    
    try:
        # For short and cover, we need to check the current position
        if action in ["short", "cover"]:
            # Get the current position
            try:
                position = client.get_position(ticker)
                current_qty = int(position.qty)
                logger.info(f"Found existing position for {ticker}: {current_qty} shares")
                
                # Handle cover (closing a short position)
                if action == "cover":
                    if current_qty >= 0:  # Not a short position
                        logger.warning(f"Cannot cover {ticker}: No short position exists")
                        return {
                            "success": False,
                            "reason": "No short position exists",
                            "ticker": ticker,
                            "action": action
                        }
                    
                    # Limit quantity to current short position
                    if abs(current_qty) < quantity:
                        logger.info(f"Adjusting cover quantity from {quantity} to {abs(current_qty)} for {ticker}")
                        quantity = abs(current_qty)
                
                # Handle short (creating a short position or adding to existing short)
                elif action == "short":
                    if current_qty < 0:  # Already have a short position
                        # Check if this would exceed our maximum short
                        # This is a safety check to prevent doubling up on shorts
                        logger.info(f"Already have a short position of {abs(current_qty)} shares for {ticker}")
                        # You can add additional logic here if needed
            except Exception as e:
                # Position doesn't exist
                logger.info(f"No position found for {ticker}: {str(e)}")
                if action == "cover":
                    logger.warning(f"Cannot cover {ticker}: No position exists")
                    return {
                        "success": False,
                        "reason": "No position exists to cover",
                        "ticker": ticker,
                        "action": action,
                        "error": str(e)
                    }
                elif action == "short":
                    # This is a new short position, no special handling needed
                    logger.info(f"Creating new short position for {ticker}")
        
        # For buy and sell, verify the current position
        elif action in ["buy", "sell"]:
            try:
                position = client.get_position(ticker)
                current_qty = int(position.qty)
                logger.info(f"Found existing position for {ticker}: {current_qty} shares")
                
                # Handle sell (closing a long position)
                if action == "sell":
                    if current_qty <= 0:  # Not a long position
                        logger.warning(f"Cannot sell {ticker}: No long position exists")
                        return {
                            "success": False,
                            "reason": "No long position exists",
                            "ticker": ticker,
                            "action": action
                        }
                    
                    # Limit quantity to current long position
                    if current_qty < quantity:
                        logger.info(f"Adjusting sell quantity from {quantity} to {current_qty} for {ticker}")
                        quantity = current_qty
                
                # Handle buy (creating or adding to a long position)
                elif action == "buy":
                    if current_qty > 0:
                        # We already have a long position, just informational
                        logger.info(f"Adding to existing long position of {current_qty} shares for {ticker}")
            except Exception as e:
                # Position doesn't exist
                logger.info(f"No position found for {ticker}: {str(e)}")
                if action == "sell":
                    logger.warning(f"Cannot sell {ticker}: No position exists")
                    return {
                        "success": False,
                        "reason": "No position exists to sell",
                        "ticker": ticker,
                        "action": action,
                        "error": str(e)
                    }
                elif action == "buy":
                    # This is a new buy position, no special handling needed
                    logger.info(f"Creating new buy position for {ticker}")
        
        # Calculate stop loss and take profit prices using dynamic ATR-based values if available
        stop_loss_price = None
        take_profit_price = None
        
        # If price dataframe is available, calculate dynamic stops based on volatility
        if prices_df is not None and action in ["buy", "short"]:
            # Get position risk parameters from enhanced risk system
            position_risk_params = calculate_position_risk_parameters(
                ticker, current_price, current_price, prices_df
            )
            
            if position_risk_params:
                if action == "buy":
                    stop_loss_price = position_risk_params['stop_loss_price']
                    take_profit_price = position_risk_params['take_profit_price']
                    logger.info(f"Using dynamic ATR-based stops for {ticker}: Stop loss at ${stop_loss_price:.2f}, Take profit at ${take_profit_price:.2f}")
                elif action == "short":
                    # For shorts, the stop is above and take profit is below
                    stop_loss_price = position_risk_params['short_stop_price']
                    take_profit_price = position_risk_params['short_take_profit_price']
                    logger.info(f"Using dynamic ATR-based stops for short {ticker}: Stop loss at ${stop_loss_price:.2f}, Take profit at ${take_profit_price:.2f}")
        
        # Fall back to fixed percentages if dynamic calculation fails
        if stop_loss_price is None or take_profit_price is None:
            if action == "buy":
                # Calculate stop loss (5% below purchase price by default)
                stop_loss_price = current_price * (1 - RISK_PARAMS["STOP_LOSS_PCT"])
                # Calculate take profit (20% above purchase price by default)
                take_profit_price = current_price * (1 + RISK_PARAMS["TAKE_PROFIT_PCT"])
            elif action == "short":
                # For short, the stop loss is above the entry price
                stop_loss_price = current_price * (1 + RISK_PARAMS["STOP_LOSS_PCT"])
                # For short, the take profit is below the entry price
                take_profit_price = current_price * (1 - RISK_PARAMS["TAKE_PROFIT_PCT"])
            
            logger.info(f"Using fixed percentage stops for {ticker} {action}: Stop loss at ${stop_loss_price:.2f}, Take profit at ${take_profit_price:.2f}")
        
        if action == "buy":
            # Use bracket order for buy orders to include stop loss and take profit
            order_data = BracketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                take_profit=LimitOrderRequest(
                    limit_price=take_profit_price,
                    time_in_force=TimeInForce.GTC
                ),
                stop_loss=StopOrderRequest(
                    stop_price=stop_loss_price, 
                    time_in_force=TimeInForce.GTC
                )
            )
            logger.info(f"Creating bracket order for {ticker}: Stop loss at ${stop_loss_price:.2f}, Take profit at ${take_profit_price:.2f}")
            
        elif action == "short":
            # Use bracket order for short orders as well
            order_data = BracketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                take_profit=LimitOrderRequest(
                    limit_price=take_profit_price,
                    time_in_force=TimeInForce.GTC
                ),
                stop_loss=StopOrderRequest(
                    stop_price=stop_loss_price,
                    time_in_force=TimeInForce.GTC
                )
            )
            logger.info(f"Creating bracket order for short {ticker}: Stop loss at ${stop_loss_price:.2f}, Take profit at ${take_profit_price:.2f}")
            
        else:
            # For sell and cover orders, use simple market orders
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            logger.info(f"Creating market order for {action} {ticker}: {quantity} shares")
        
        # Log the order request details
        logger.info(f"Submitting order: {order_data}")
        
        # Submit the order
        order = client.submit_order(order_data)
        logger.info(f"Successfully submitted {action} order for {quantity} shares of {ticker}: Order ID {order.id}")
        
        # Record the successful trade in our risk management system
        record_trade_execution(ticker, action, quantity, current_price, quantity * current_price)
        
        logger.info(f"=== EXECUTE TRADE SUCCESS: {ticker} {action} {quantity} ===")
        
        return {
            "success": True,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": current_price,
            "order_id": order.id,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price
        }
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Failed to submit {action} order for {ticker}: {error_message}")
        logger.exception("Detailed error traceback:")
        
        logger.info(f"=== EXECUTE TRADE FAILED: {ticker} {action} {quantity} ===")
        
        return {
            "success": False,
            "reason": f"Order submission failed: {error_message}",
            "ticker": ticker,
            "action": action,
            "quantity": quantity
        }


def get_alpaca_portfolio_state(client, tickers):
    """Get current portfolio state from Alpaca API"""
    if not client:
        return None
    
    try:
        # Get account information
        account = client.get_account()
        
        # Initialize portfolio structure
        portfolio = {
            "cash": float(account.cash),
            "positions": {},
            "margin_requirement": 0.0,  # We'll calculate this based on positions
            "open_orders": {},  # We'll store open orders here
        }
        
        # Get all positions
        all_positions = {}
        try:
            positions = client.get_all_positions()
            for position in positions:
                all_positions[position.symbol] = position
        except Exception as e:
            logger.warning(f"Failed to get positions from Alpaca: {e}")
        
        # Get all open orders
        try:
            open_orders_by_ticker = get_alpaca_open_orders(client)
            portfolio["open_orders"] = open_orders_by_ticker
            logger.info(f"Found open orders for {len(open_orders_by_ticker)} tickers")
        except Exception as e:
            logger.warning(f"Failed to get open orders from Alpaca: {e}")
        
        # Organize positions by ticker
        for ticker in tickers:
            position = all_positions.get(ticker)
            
            if position:
                qty = int(position.qty)
                market_value = float(position.market_value)
                cost_basis = float(position.cost_basis)
                
                # Determine if long or short position
                if qty > 0:  # Long position
                    portfolio["positions"][ticker] = {
                        "long": qty,
                        "short": 0,
                        "long_cost_basis": cost_basis / qty if qty > 0 else 0.0,
                        "short_cost_basis": 0.0,
                        "short_margin_used": 0.0
                    }
                elif qty < 0:  # Short position
                    # For shorts, qty is negative
                    short_qty = abs(qty)
                    # Estimate margin requirement at 50% of position value
                    margin_used = market_value * 0.5
                    
                    portfolio["positions"][ticker] = {
                        "long": 0,
                        "short": short_qty,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": cost_basis / short_qty if short_qty > 0 else 0.0,
                        "short_margin_used": margin_used
                    }
                    
                    # Add to total margin requirement
                    portfolio["margin_requirement"] += margin_used
            else:
                # No position for this ticker
                portfolio["positions"][ticker] = {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0
                }
        
        # Calculate total portfolio value including both long and short positions
        portfolio_value = float(account.equity)
        
        # Update the risk management system with the current portfolio value
        update_portfolio_value(portfolio_value)
        
        return portfolio
    
    except Exception as e:
        logger.error(f"Failed to get portfolio state from Alpaca: {e}")
        return None


def apply_sector_correlation_adjustments(ticker, decision, quantity, portfolio):
    """Apply position size adjustments based on sector exposure and correlation risk"""
    # Original quantity
    original_quantity = quantity
    adjusted_quantity = quantity
    adjustment_reasons = []
    
    # Check sector exposure
    try:
        sector_exposures = track_sector_exposure(portfolio)
        over_exposed_sectors = check_sector_limits(sector_exposures)
        
        # Get the ticker's sector
        ticker_sector = None
        for sector, tickers in sector_exposures.items():
            if ticker in tickers:
                ticker_sector = sector
                break
        
        # If ticker is in an over-exposed sector, reduce position size
        if ticker_sector and ticker_sector in over_exposed_sectors:
            current_exposure = over_exposed_sectors[ticker_sector]["current_exposure"]
            max_allowed = over_exposed_sectors[ticker_sector]["max_allowed"]
            overexposure_ratio = current_exposure / max_allowed if max_allowed > 0 else 2.0
            
            # Apply a graduated reduction based on how overexposed the sector is
            if overexposure_ratio >= 2.0:
                # Extremely overexposed - reduce by 70%
                sector_adjusted = int(adjusted_quantity * 0.3)
                adjustment_reasons.append(f"Sector {ticker_sector} extremely overexposed ({current_exposure:.1f}% vs {max_allowed:.1f}% limit)")
            elif overexposure_ratio >= 1.5:
                # Significantly overexposed - reduce by 50%
                sector_adjusted = int(adjusted_quantity * 0.5)
                adjustment_reasons.append(f"Sector {ticker_sector} significantly overexposed ({current_exposure:.1f}% vs {max_allowed:.1f}% limit)")
            else:
                # Moderately overexposed - reduce by 30%
                sector_adjusted = int(adjusted_quantity * 0.7)
                adjustment_reasons.append(f"Sector {ticker_sector} moderately overexposed ({current_exposure:.1f}% vs {max_allowed:.1f}% limit)")
            
            adjusted_quantity = max(1, sector_adjusted)
    except Exception as e:
        logger.warning(f"Error checking sector exposure for {ticker}: {e}")
    
    # Check correlation risk
    try:
        correlation_risks = check_correlation_risk(portfolio)
        high_correlation_tickers = []
        
        # Find correlated assets
        for corr_pair in correlation_risks:
            if ticker in corr_pair["tickers"]:
                # Extract the other ticker from the pair
                other_ticker = corr_pair["tickers"][0] if corr_pair["tickers"][1] == ticker else corr_pair["tickers"][1]
                high_correlation_tickers.append((other_ticker, corr_pair["correlation"]))
        
        # If we have high correlations, adjust position size
        if high_correlation_tickers:
            # Sort by correlation strength (highest first)
            high_correlation_tickers.sort(key=lambda x: x[1], reverse=True)
            
            # Find highest correlation value
            highest_corr_ticker, highest_corr = high_correlation_tickers[0]
            
            # Apply adjustment based on correlation strength
            if highest_corr >= 0.9:
                # Extremely high correlation - reduce by 60%
                corr_adjusted = int(adjusted_quantity * 0.4)
                adjustment_reasons.append(f"Extremely high correlation ({highest_corr:.2f}) with {highest_corr_ticker}")
            elif highest_corr >= 0.8:
                # High correlation - reduce by 40%
                corr_adjusted = int(adjusted_quantity * 0.6)
                adjustment_reasons.append(f"High correlation ({highest_corr:.2f}) with {highest_corr_ticker}")
            elif highest_corr >= 0.7:
                # Moderate correlation - reduce by 20%
                corr_adjusted = int(adjusted_quantity * 0.8)
                adjustment_reasons.append(f"Moderate correlation ({highest_corr:.2f}) with {highest_corr_ticker}")
            else:
                corr_adjusted = adjusted_quantity
            
            adjusted_quantity = max(1, corr_adjusted)
    except Exception as e:
        logger.warning(f"Error checking correlation risk for {ticker}: {e}")
    
    # Return the adjusted quantity and reasons
    return adjusted_quantity, adjustment_reasons


def serialize_for_json(obj):
    """
    Recursively serialize objects for JSON.
    Handles Pydantic models and custom types with robust error handling.
    """
    try:
        # Handle None values
        if obj is None:
            return None
            
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            try:
                return obj.model_dump()
            except Exception as e:
                logger.warning(f"Error in model_dump: {e}")
                # Try to extract attributes manually
                return {k: serialize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
                
        elif hasattr(obj, 'dict') and callable(obj.dict):
            try:
                return obj.dict()  # Legacy Pydantic v1
            except Exception as e:
                logger.warning(f"Error in dict() call: {e}")
                # Try to extract attributes manually
                return {k: serialize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
                
        # Handle dictionaries
        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                try:
                    # Skip None values and entries with None keys
                    if k is None or v is None:
                        continue
                    result[str(k)] = serialize_for_json(v)
                except Exception as e:
                    logger.warning(f"Error serializing dictionary entry {k}: {e}")
            return result
            
        # Handle lists
        elif isinstance(obj, list):
            result = []
            for item in obj:
                try:
                    if item is not None:
                        result.append(serialize_for_json(item))
                except Exception as e:
                    logger.warning(f"Error serializing list item: {e}")
            return result
            
        # Handle tuples
        elif isinstance(obj, tuple):
            result = []
            for item in obj:
                try:
                    if item is not None:
                        result.append(serialize_for_json(item))
                except Exception as e:
                    logger.warning(f"Error serializing tuple item: {e}")
            return result
            
        # Handle string-formatted signals (for cached Paul Tudor Jones signals)
        elif isinstance(obj, str) and "signal=" in obj and "confidence=" in obj and "reasoning=" in obj:
            try:
                # Extract signal, confidence, and reasoning using regex
                import re
                signal_match = re.search(r"signal=['\"]([^'\"]+)['\"]", obj)
                confidence_match = re.search(r"confidence=([0-9.]+)", obj)
                reasoning_match = re.search(r"reasoning=['\"]([^$]*?)['\"](?:\s|$)", obj)
                
                signal = signal_match.group(1) if signal_match else "neutral"
                confidence = float(confidence_match.group(1)) if confidence_match else 50.0
                reasoning = reasoning_match.group(1) if reasoning_match else ""
                
                logger.info(f"Extracted signal={signal}, confidence={confidence} from string")
                
                return {
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            except Exception as e:
                logger.warning(f"Error parsing string-formatted signal: {e}")
                return {
                    "signal": "neutral",
                    "confidence": 50.0,
                    "reasoning": "Error parsing signal string"
                }
            
        # Special handling for William O'Neil signals
        elif obj.__class__.__name__ == 'WilliamONeilSignal':
            try:
                return {
                    "signal": getattr(obj, "signal", "neutral"),
                    "confidence": float(getattr(obj, "confidence", 50.0)),
                    "reasoning": str(getattr(obj, "reasoning", ""))
                }
            except Exception as e:
                logger.warning(f"Error serializing WilliamONeilSignal: {e}")
                return {
                    "signal": "neutral",
                    "confidence": 50.0,
                    "reasoning": "Error serializing signal"
                }
                
        # Special handling for other signal objects
        elif hasattr(obj, 'signal') and hasattr(obj, 'confidence'):
            try:
                # Fix 0 confidence
                confidence = getattr(obj, "confidence", 50.0)
                
                # Convert from decimal (0-1) to percentage (0-100) if needed
                if confidence > 0 and confidence <= 1.0:
                    logger.info(f"Converting decimal confidence {confidence} to percentage for {obj.__class__.__name__}")
                    confidence = confidence * 100.0
                elif confidence == 0:
                    confidence = 50.0  # Assign neutral confidence to 0 values
                    
                return {
                    "signal": str(getattr(obj, "signal", "neutral")),
                    "confidence": float(confidence),
                    "reasoning": str(getattr(obj, "reasoning", "")) if hasattr(obj, "reasoning") else ""
                }
            except Exception as e:
                logger.warning(f"Error serializing signal object: {e}")
                return {
                    "signal": "neutral",
                    "confidence": 50.0,
                    "reasoning": "Error serializing signal"
                }
        
        # Other primitive types
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            # Convert other objects to string
            return str(obj)
            
    except Exception as e:
        logger.error(f"Unhandled error in serialize_for_json: {e}")
        return "Error during serialization"


##### Portfolio Management Agent #####
@cached_analyst()
def portfolio_management_agent(state: AgentState):
    """Portfolio Management Agent that generates trading decisions."""
    data = state.get("data", {})
    metadata = state.get("metadata", {})
    
    # Extract data needed for trading decisions
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {})
    analyst_signals = data.get("analyst_signals", {})
    
    # Get model information
    model_name = metadata.get("model_name", "gpt-4o")
    model_provider = metadata.get("model_provider", "OpenAI")
    
    # Create a progress indicator
    progress.update_status("portfolio_management_agent", None, "Generating trading decisions")
    
    try:
        # Calculate maximum shares to trade for each ticker
        max_shares = calculate_max_shares_per_ticker(portfolio, tickers)
        
        # Get current prices
        current_prices = portfolio.get("current_prices", {})
        
        # If we're missing current prices, try to get them from positions
        if not current_prices and "positions" in portfolio:
            for ticker, position in portfolio["positions"].items():
                if "current_price" in position:
                    current_prices[ticker] = position["current_price"]
        
        # Make sure we have current prices for all tickers
        missing_prices = [ticker for ticker in tickers if ticker not in current_prices]
        if missing_prices:
            logger.warning(f"Missing prices for tickers: {missing_prices}")
            # Try to get current prices from Alpaca
            try:
                client = get_alpaca_client()
                if client:
                    for ticker in missing_prices:
                        try:
                            # Get the latest bar
                            from alpaca.data.historical import StockHistoricalDataClient
                            from alpaca.data.requests import StockBarsRequest
                            from alpaca.data.timeframe import TimeFrame
                            
                            api_key = os.getenv("ALPACA_API_KEY")
                            api_secret = os.getenv("ALPACA_API_SECRET")
                            
                            if api_key and api_secret:
                                data_client = StockHistoricalDataClient(api_key, api_secret)
                                request_params = StockBarsRequest(
                                    symbol_or_symbols=[ticker],
                                    timeframe=TimeFrame.Day,
                                    start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                                    end=datetime.now().strftime("%Y-%m-%d")
                                )
                                bars = data_client.get_stock_bars(request_params)
                                if bars and ticker in bars:
                                    df = bars[ticker].df
                                    if not df.empty:
                                        # Get the most recent close price
                                        current_prices[ticker] = df['close'].iloc[-1]
                                        logger.info(f"Got current price for {ticker} from Alpaca: ${current_prices[ticker]:.2f}")
                        except Exception as e:
                            logger.warning(f"Error getting current price for {ticker}: {e}")
            except Exception as e:
                logger.warning(f"Error getting current prices from Alpaca: {e}")
                
        # Generate the trading decisions
        logger.info(f"Generating trading decisions for tickers: {tickers}")
        logger.info(f"Portfolio cash: ${portfolio.get('cash', 0):.2f}")
        
        # We need to gather analyst signals in the format required by generate_trading_decision
        signals_by_ticker = {}
        for ticker in tickers:
            signals_by_ticker[ticker] = {}
            
            for analyst, signals in analyst_signals.items():
                if ticker in signals:
                    signals_by_ticker[ticker][analyst] = signals[ticker]
        
        # Generate a comprehensive risk dashboard
        from utils.enhanced_risk import generate_risk_dashboard
        risk_dashboard = generate_risk_dashboard(
            portfolio=portfolio,
            tickers=tickers,
            prices_data=data.get("price_data", {})
        )
        
        logger.info("Calling generate_trading_decision")
        # Call the function to generate trading decisions
        trading_decisions = generate_trading_decision(
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            max_shares=max_shares,
            portfolio=portfolio,
            risk_dashboard=risk_dashboard,
            model_name=model_name,
            model_provider=model_provider,
        )
        
        # Process the decisions
        decisions_to_execute = {}
        for ticker, decision in trading_decisions.decisions.items():
            # Process the decision (e.g., apply risk management rules)
            # For now, we'll just copy the decision as-is
            decisions_to_execute[ticker] = decision
            
            # Check if we need to cancel open orders for hold decisions
            if decision.action == "hold" and "open_orders" in portfolio:
                open_orders = portfolio["open_orders"]
                if ticker in open_orders and open_orders[ticker]:
                    # There are open orders for this ticker
                    decision.cancel_existing_orders = True
                    decision.reasoning += f"\n\nExisting open orders for {ticker} will be canceled based on the 'hold' recommendation."
                    logger.info(f"Marking open orders for {ticker} to be canceled due to 'hold' recommendation")
                    
            # Check if we need to cancel existing orders for other decisions that change direction
            if decision.action in ["buy", "short"] and "open_orders" in portfolio:
                open_orders = portfolio["open_orders"]
                if ticker in open_orders and open_orders[ticker]:
                    # Check if existing orders conflict with the new direction
                    conflicting_orders = False
                    for order in open_orders[ticker]:
                        if (decision.action == "buy" and order.side == "sell") or \
                           (decision.action == "short" and order.side == "buy"):
                            conflicting_orders = True
                            break
                    
                    if conflicting_orders:
                        decision.cancel_existing_orders = True
                        decision.reasoning += f"\n\nExisting orders for {ticker} will be canceled due to change in direction."
                        logger.info(f"Marking open orders for {ticker} to be canceled due to change in direction")
        
        # Execute the decisions regardless of live trading mode
        # This ensures orders are processed in both paper trading and live modes
        client = get_alpaca_client()
        if client:
            # Log whether we're in live or paper mode
            env_type = "live trading" if LIVE_TRADING_ENABLED else "paper trading"
            logger.info(f"Executing trading decisions with Alpaca ({env_type})")
            
            # Add enhanced debugging for account status
            try:
                account = client.get_account()
                is_paper_account = getattr(account, 'is_paper', True)
                env_type = "paper trading" if is_paper_account else "live trading"
                logger.info(f"===== ORDER PROCESSING STARTED IN {env_type.upper()} MODE =====")
                logger.info(f"LIVE_TRADING_ENABLED = {LIVE_TRADING_ENABLED}")
                
                # Print all decisions for debugging
                for ticker, decision in decisions_to_execute.items():
                    logger.info(f"Decision for {ticker}: action={decision.action}, cancel_orders={decision.cancel_existing_orders}")
            except Exception as e:
                logger.error(f"Error checking account details: {e}")
                logger.info(f"===== ORDER PROCESSING STARTED IN UNKNOWN MODE =====")
                logger.info(f"LIVE_TRADING_ENABLED = {LIVE_TRADING_ENABLED}")
            
            # First, process all cancellations
            cancel_results = {}
            try:
                for ticker, decision in decisions_to_execute.items():
                    if decision.cancel_existing_orders:
                        logger.info(f"Step 1: Canceling existing orders for {ticker}")
                        # First get the existing orders to see what we're canceling
                        ticker_orders = get_alpaca_open_orders(client, ticker)
                        if ticker in ticker_orders and ticker_orders[ticker]:
                            logger.info(f"Found {len(ticker_orders[ticker])} orders to cancel for {ticker}")
                            for i, order in enumerate(ticker_orders[ticker]):
                                logger.info(f"Order {i+1} to cancel: ID={order.id}, Status={order.status}, Side={order.side}, Qty={order.qty}")
                        else:
                            logger.warning(f"No open orders found for {ticker} even though cancellation was requested")
                        
                        # Now proceed with cancellation
                        cancel_result = cancel_open_orders_for_ticker(client, ticker)
                        cancel_results[ticker] = cancel_result
                        logger.info(f"Cancellation result for {ticker}: {cancel_result}")
            except Exception as e:
                logger.error(f"Error during order cancellation phase: {e}")
                logger.exception("Detailed cancellation error traceback:")
                
            # Wait a bit longer for cancellations to be fully processed
            if cancel_results:
                logger.info("Waiting for cancellations to be fully processed...")
                time.sleep(3)  # Increased wait time from 2 to 3 seconds
                
                # Verify cancellations by checking again
                try:
                    for ticker in cancel_results:
                        post_cancel_orders = get_alpaca_open_orders(client, ticker)
                        if ticker in post_cancel_orders and post_cancel_orders[ticker]:
                            logger.warning(f"STILL FOUND {len(post_cancel_orders[ticker])} ORDERS for {ticker} AFTER CANCELLATION!")
                            for i, order in enumerate(post_cancel_orders[ticker]):
                                logger.warning(f"Remaining order {i+1}: ID={order.id}, Status={order.status}, Side={order.side}, Qty={order.qty}")
                        else:
                            logger.info(f"✓ Successfully canceled all orders for {ticker}")
                except Exception as e:
                    logger.error(f"Error during post-cancellation verification: {e}")
                
            # Then, execute all new trades
            execution_results = {}
            try:
                for ticker, decision in decisions_to_execute.items():
                    # Only execute if it's not a hold action and quantity > 0
                    if decision.action != "hold" and decision.quantity > 0:
                        # Double-check that the order won't conflict with existing orders
                        proceed_with_execution = True
                        
                        # Re-check for open orders after cancellations
                        open_orders = get_alpaca_open_orders(client, ticker)
                        if ticker in open_orders and open_orders[ticker]:
                            logger.warning(f"Still found {len(open_orders[ticker])} open orders for {ticker} after cancellation")
                            
                            # If we tried to cancel but failed, we'll log it and proceed anyway
                            if ticker in cancel_results and cancel_results[ticker].get("success", False):
                                logger.warning(f"Cancellation was reported successful but orders still exist! Attempting execution anyway.")
                            else:
                                proceed_with_execution = False
                                logger.error(f"Cannot execute {decision.action} for {ticker} due to existing orders that couldn't be canceled")
                        
                        if proceed_with_execution:
                            logger.info(f"Step 2: Executing {decision.action} for {ticker} with quantity {decision.quantity}")
                            current_price = current_prices.get(ticker, 100.0)  # Default price if not available
                            execution_result = execute_alpaca_trade(ticker, decision.action, decision.quantity, current_price)
                            execution_results[ticker] = execution_result
                            logger.info(f"Execution result for {ticker} {decision.action} {decision.quantity}: {execution_result}")
            except Exception as e:
                logger.error(f"Error during order execution phase: {e}")
                logger.exception("Detailed execution error traceback:")
        else:
            logger.warning("Alpaca client not available - cannot execute trades or cancel orders")
        
        progress.update_status("portfolio_management_agent", None, "Done")
        
        # Create a human message with the decisions
        message = HumanMessage(
            content=json.dumps({"decisions": serialize_for_json(trading_decisions.decisions)}),
            name="portfolio_management_agent",
        )
        
        # Print the reasoning if the flag is set
        if metadata.get("show_reasoning", False):
            show_agent_reasoning(trading_decisions, "Portfolio Management Agent")
        
        # Add the decisions to the data
        data["portfolio_decisions"] = trading_decisions
        
        return {
            "messages": [message],
            "data": data,
        }
        
    except Exception as e:
        logger.error(f"Error in portfolio_management_agent: {e}")
        
        # Create a default output in case of error
        default_output = create_default_portfolio_output(tickers)
        
        # Create a message with the default output
        message = HumanMessage(
            content=json.dumps({"decisions": serialize_for_json(default_output.decisions)}),
            name="portfolio_management_agent",
        )
        
        # Add the default decisions to the data
        data["portfolio_decisions"] = default_output
        
        return {
            "messages": [message],
            "data": data,
        }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    risk_dashboard: dict,
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    """
    Generates trading decisions for multiple tickers using an LLM.
    
    Args:
        tickers: List of tickers to generate decisions for
        signals_by_ticker: Dictionary of analyst signals for each ticker
        current_prices: Dictionary of current prices for each ticker
        max_shares: Dictionary of maximum shares allowed for each ticker
        portfolio: Portfolio state including cash and positions
        risk_dashboard: Risk management dashboard with market regime and other risk factors
        model_name: Name of the model to use
        model_provider: Provider of the model
        
    Returns:
        PortfolioManagerOutput: Trading decisions for each ticker
    """
    
    # Get the portfolio cash and positions
    portfolio_cash = portfolio.get("cash", 0)
    portfolio_positions = portfolio.get("positions", {})
    margin_requirement = portfolio.get("margin_requirement", 0)
    
    # Log the portfolio cash value to verify it's being correctly retrieved
    logger.info(f"Portfolio cash available: ${portfolio_cash}")
    
    # If the cash value is zero or unusually low for a paper trading account, use a reasonable default
    if portfolio_cash < 100 and LIVE_TRADING_ENABLED and os.getenv("LIVE_TRADING", "false").lower() != "true":
        logger.warning(f"Unusually low cash value detected (${portfolio_cash}), using default paper trading value")
        portfolio_cash = 100000.0  # Default paper trading cash
        portfolio["cash"] = portfolio_cash
        
    logger.info(f"Final portfolio cash for decision making: ${portfolio_cash}")
    
    # Get open orders if we have a live portfolio
    open_orders_by_ticker = {}
    if LIVE_TRADING_ENABLED:
        try:
            alpaca_client = get_alpaca_client()
            if alpaca_client:
                open_orders_by_ticker = get_alpaca_open_orders(alpaca_client)
                logger.info(f"Found open orders for tickers: {list(open_orders_by_ticker.keys())}")
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")

    # Format open orders information for the prompt
    open_orders_info = "Open orders:\n"
    if open_orders_by_ticker:
        for ticker, orders in open_orders_by_ticker.items():
            if orders:
                open_orders_info += f"{ticker}: {len(orders)} order(s) - "
                for order in orders:
                    open_orders_info += f"{order.get('side', 'unknown')} {order.get('qty', '?')} @ {order.get('limit_price', 'market')} "
                open_orders_info += "\n"
    else:
        open_orders_info += "No open orders.\n"

    # Load the prompt template
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "portfolio_manager_prompt.txt")
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
    except Exception as e:
        logger.error(f"Error loading prompt template: {e}")
        prompt_template = """
        You are a portfolio manager making final trading decisions based on multiple tickers.
        
        Trading Rules:
        - For long positions:
          * Only buy if you have available cash
          * Only sell if you currently hold long shares of that ticker
          * Sell quantity must be ≤ current long position shares
          * Buy quantity must be ≤ max_shares for that ticker
        
        - For short positions:
          * Only short if you have available margin (50% of position value required)
          * Only cover if you currently have short shares of that ticker
          * Cover quantity must be ≤ current short position shares
          * Short quantity must respect margin requirements
        
        - Position validation requirements:
          * CRITICALLY IMPORTANT: DO NOT double sell positions
          * CRITICALLY IMPORTANT: DO NOT double short positions
          * If the current position already matches your intended action (e.g., already sold or already shorted), use "hold" instead
          * Verify existing positions before making sell/cover decisions
          * Only suggest actionable trades
          
        - Open Orders:
          * If there are already open orders for a ticker, default to "hold" for that ticker
          * Do not place new orders for tickers that already have open orders of the same type
        
        Available Actions:
        - "buy": Open or add to long position
        - "sell": Close or reduce long position
        - "short": Open or add to short position
        - "cover": Close or reduce short position
        - "hold": No action
        
        Based on the team's analysis, make your trading decisions for each ticker.
        """
    
    # Prepare signals for the prompt by converting them to JSON serializable format
    try:
        # Filter out potentially problematic signals
        filtered_signals = {}
        
        # Log all signals for debugging
        logger.info(f"Processing signals for {len(tickers)} tickers from {len(signals_by_ticker)} analysts")
        
        for analyst, signals in signals_by_ticker.items():
            if isinstance(signals, dict):
                # Check if signals contain only ticker data
                filtered_analyst_signals = {}
                
                # Only include signals for requested tickers
                for ticker in tickers:
                    if ticker in signals:
                        ticker_signal = signals[ticker]
                        
                        # Check if the signal looks valid
                        if ticker_signal is not None:
                            # Skip signals with 0% confidence
                            if hasattr(ticker_signal, 'confidence') and ticker_signal.confidence == 0:
                                logger.warning(f"Skipping {analyst} signal for {ticker} with 0% confidence")
                                continue
                                
                            # Add valid signal
                            filtered_analyst_signals[ticker] = ticker_signal
                
                # Only add analyst if there are signals
                if filtered_analyst_signals:
                    filtered_signals[analyst] = filtered_analyst_signals
            else:
                # Skip analysts without ticker-specific signals
                logger.warning(f"Skipping analyst {analyst} - signal format not recognized")
                
        # Use the custom serializer to handle Pydantic models and complex objects
        serialized_signals = serialize_for_json(filtered_signals)
        
        # Check if serialization was successful
        if not serialized_signals:
            logger.warning("No valid signals after filtering and serialization")
        
        # Create simplified analyst recommendations for easier LLM consumption
        simplified_recommendations = {}
        for ticker in tickers:
            ticker_recommendations = []
            for analyst, signals in filtered_signals.items():
                if ticker in signals:
                    signal = signals[ticker]
                    if hasattr(signal, 'signal') and hasattr(signal, 'confidence'):
                        ticker_recommendations.append({
                            "analyst": analyst.replace("_agent", "").replace("_", " ").title(),
                            "signal": getattr(signal, 'signal', 'neutral'),
                            "confidence": getattr(signal, 'confidence', 50.0)
                        })
            simplified_recommendations[ticker] = ticker_recommendations
            
        # Format prompt with simplified recommendations
        formatted_prompt = prompt_template.format(
            signals_by_ticker=json.dumps(serialized_signals, indent=2),
            current_prices=json.dumps(current_prices, indent=2),
            max_shares=json.dumps(max_shares, indent=2),
            portfolio_cash=portfolio_cash,
            portfolio_positions=json.dumps(portfolio_positions, indent=2),
            margin_requirement=margin_requirement,
            risk_dashboard=json.dumps(risk_dashboard if risk_dashboard else {}, indent=2)
        )
        
        # Ensure cash amount is properly formatted in the template
        formatted_prompt = formatted_prompt.replace("${portfolio_cash:.2f}", f"${portfolio_cash:.2f}")
        
        # Add open orders information
        formatted_prompt += "\n\n" + open_orders_info
        
        # Add simplified recommendations section
        formatted_prompt += "\n\nSIMPLIFIED RECOMMENDATIONS:\n"
        formatted_prompt += json.dumps(simplified_recommendations, indent=2)
        
    except Exception as e:
        logger.error(f"Error formatting prompt: {e}", exc_info=True)
        # Fallback to a simplified prompt without the problematic parts
        formatted_prompt = """
        You are a portfolio manager making final trading decisions based on available data.
        
        I need you to make trading decisions for the following tickers:
        """
        
        # Add ticker list
        for ticker in tickers:
            formatted_prompt += f"\n- {ticker}"
            
        # Add portfolio positions
        formatted_prompt += "\n\nCurrent positions:"
        for ticker in tickers:
            if ticker in portfolio_positions:
                pos = portfolio_positions[ticker]
                long_pos = pos.get("long", 0)
                short_pos = pos.get("short", 0)
                if long_pos > 0:
                    formatted_prompt += f"\n- {ticker}: LONG {long_pos} shares"
                elif short_pos > 0:
                    formatted_prompt += f"\n- {ticker}: SHORT {short_pos} shares"
                else:
                    formatted_prompt += f"\n- {ticker}: No position"
        
        formatted_prompt += f"\n\nCurrent cash available: ${portfolio_cash}"
        
        # Add current prices
        formatted_prompt += "\n\nCurrent prices:"
        for ticker, price in current_prices.items():
            formatted_prompt += f"\n- {ticker}: ${price}"
            
        formatted_prompt += """
        
        Please provide trading decisions for each ticker with a focus on managing risk.
        
        Valid actions are:
        - "buy": Open or add to long position
        - "sell": Close or reduce long position
        - "short": Open or add to short position
        - "cover": Close or reduce short position
        - "hold": No action
        
        IMPORTANT: 
        - Only sell if you have a long position
        - Only cover if you have a short position
        - Be sure to consider the current cash and position limits
        
        Your output MUST follow this JSON format:
        {
          "decisions": {
            "TICKER1": {
              "action": "buy/sell/short/cover/hold",
              "quantity": 0,
              "confidence": 50.0,
              "reasoning": "Reasoning for decision"
            },
            "TICKER2": { ... },
            ...
          }
        }
        """
    
    # Create messages for the LLM call
    try:
        from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # Create messages manually
        system_message = SystemMessage(content=formatted_prompt)
        # Add "json" to the human message to comply with OpenAI's JSON mode requirements
        human_message = HumanMessage(content="""Based on the data provided, generate trading decisions for each ticker.

YOU MUST return your response as a JSON object with a structure exactly like this:
{
  "decisions": {
    "AAPL": {
      "action": "hold",
      "quantity": 0,
      "confidence": 65.0,
      "reasoning": "Reasoning for this decision"
    },
    "MSFT": {
      "action": "buy",
      "quantity": 10,
      "confidence": 75.0,
      "reasoning": "Reasoning for this decision"
    }
  }
}

The "decisions" key must contain an object with ticker symbols as keys, and each ticker must have action, quantity, confidence, and reasoning fields.
All fields are required.
""")
        messages = [system_message, human_message]
        
        logger.info(f"Created messages for LLM call: {len(messages)} messages")
    except Exception as e:
        logger.error(f"Error creating messages: {e}")
        # Fallback to a simple string prompt
        messages = f"""{formatted_prompt}

Based on the data provided, generate trading decisions for each ticker.

YOU MUST return your response as a JSON object with a structure exactly like this:
{{
  "decisions": {{
    "AAPL": {{
      "action": "hold",
      "quantity": 0,
      "confidence": 65.0,
      "reasoning": "Reasoning for this decision"
    }},
    "MSFT": {{
      "action": "buy",
      "quantity": 10,
      "confidence": 75.0,
      "reasoning": "Reasoning for this decision"
    }}
  }}
}}

The "decisions" key must contain an object with ticker symbols as keys, and each ticker must have action, quantity, confidence, and reasoning fields.
All fields are required.
"""
        logger.info("Using fallback string prompt for LLM call")
    
    # Check for open orders and create a default decision for tickers with open orders
    decisions_for_open_orders = {}
    for ticker in tickers:
        if ticker in open_orders_by_ticker and open_orders_by_ticker[ticker]:
            # If ticker has open orders, create a hold decision
            decisions_for_open_orders[ticker] = PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=50.0,
                reasoning=f"Hold due to {len(open_orders_by_ticker[ticker])} existing open order(s) for {ticker}"
            )
    
    # If all tickers have open orders, just return the hold decisions
    if len(decisions_for_open_orders) == len(tickers):
        logger.info("All tickers have open orders, returning hold decisions for all")
        return PortfolioManagerOutput(decisions=decisions_for_open_orders)
    
    # Call the LLM with the prompt
    try:
        # Log the prompt we're sending to the LLM
        logger.info(f"Calling LLM for portfolio decisions with model: {model_name}, provider: {model_provider}")
        
        # Log summary of signals being sent
        signal_summary = {}
        for analyst, signals in signals_by_ticker.items():
            if isinstance(signals, dict):
                signal_count = len(signals)
                signal_summary[analyst] = f"{signal_count} signals"
            else:
                signal_summary[analyst] = str(type(signals))
        logger.info(f"Signal summary: {signal_summary}")
        
        # Log tickers being analyzed
        logger.info(f"Generating decisions for tickers: {tickers}")
        
        # Define a custom fallback function for when the structured output fails
        def direct_parsing_fallback():
            """Fallback parser for when the LLM doesn't return proper JSON or pydantic format."""
            logger.warning("Using direct parsing fallback for portfolio manager")
            fallback_decisions = {}
            
            # First try to parse the response as JSON
            try:
                output_string = ""
                
                # First, attempt to handle the case where the LLM returns a string
                if isinstance(messages, list):
                    # If we have a list of messages, the LLM's response will be available
                    # Check if any tools are available to access the responses
                    logger.info("Attempting to extract LLM response from messages")
                    
                    # Try to use the tools available in the current context
                    try:
                        # This is for when we receive a raw response
                        last_message = messages[-1]
                        
                        # Try to extract any JSON from the content
                        if hasattr(last_message, 'content') and last_message.content:
                            output_string = last_message.content
                            logger.info(f"Extracted content from last message (length: {len(output_string)})")
                    except Exception as e:
                        logger.error(f"Error extracting response from messages: {e}")
                
                if output_string:
                    # Look for JSON in the output string
                    logger.info("Attempting to extract JSON from output string")
                    
                    # Look for any JSON objects in the output string
                    import re
                    json_pattern = r'(\{[\s\S]*\})'
                    json_matches = re.findall(json_pattern, output_string)
                    
                    for json_str in json_matches:
                        try:
                            json_data = json.loads(json_str.strip())
                            logger.info(f"Successfully extracted JSON: {list(json_data.keys())}")
                            
                            # Check if this JSON has the expected structure
                            if "decisions" in json_data:
                                logger.info("Found decisions key in JSON")
                                decisions_dict = json_data["decisions"]
                                
                                # Convert decisions to PortfolioDecision objects
                                for ticker, decision in decisions_dict.items():
                                    if isinstance(decision, dict):
                                        try:
                                            fallback_decisions[ticker] = PortfolioDecision(**decision)
                                            logger.info(f"Successfully created PortfolioDecision for {ticker}")
                                        except Exception as e:
                                            logger.error(f"Error creating PortfolioDecision for {ticker}: {e}")
                                            fallback_decisions[ticker] = create_default_decision(ticker)
                                    else:
                                        logger.warning(f"Decision for {ticker} is not a dictionary: {type(decision)}")
                                        fallback_decisions[ticker] = create_default_decision(ticker)
                                            
                                if fallback_decisions:
                                    logger.info(f"Successfully created {len(fallback_decisions)} decisions from JSON")
                                    return PortfolioManagerOutput(decisions=fallback_decisions)
                            
                            # Check if the JSON doesn't have a decisions key but has ticker keys directly
                            elif any(ticker in json_data for ticker in tickers):
                                logger.info("Found ticker keys directly in JSON")
                                
                                # Process each ticker decision
                                for ticker in tickers:
                                    if ticker in json_data:
                                        ticker_decision = json_data[ticker]
                                        
                                        # Check if the decision is a string (explanation) or a dict
                                        if isinstance(ticker_decision, str):
                                            # Try to create a decision from the explanation
                                            logger.info(f"Decision for {ticker} is a string, creating from explanation")
                                            fallback_decisions[ticker] = PortfolioDecision(
                                                action="hold",  # Default to hold when we only have text
                                                quantity=0,
                                                confidence=50.0,
                                                reasoning=ticker_decision
                                            )
                                        elif isinstance(ticker_decision, dict):
                                            # Try to create a PortfolioDecision from the dict
                                            try:
                                                fallback_decisions[ticker] = PortfolioDecision(**ticker_decision)
                                                logger.info(f"Created PortfolioDecision for {ticker} from direct ticker JSON")
                                            except Exception as e:
                                                logger.error(f"Error creating PortfolioDecision from ticker JSON for {ticker}: {e}")
                                                fallback_decisions[ticker] = create_default_decision(ticker)
                                
                                if fallback_decisions:
                                    logger.info(f"Successfully created {len(fallback_decisions)} decisions from direct ticker JSON")
                                    return PortfolioManagerOutput(decisions=fallback_decisions)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from: {json_str[:100]}...")
                            continue
                
                # If we couldn't extract JSON from the response or no decisions were created,
                # try to execute one LLM call per ticker to get decisions
                if len(fallback_decisions) < len(tickers):
                    logger.info("Generating individual decisions for each ticker")
                    
                    for ticker in tickers:
                        if ticker not in fallback_decisions:
                            logger.info(f"Generating fallback decision for {ticker}")
                            
                            # Generate a simplified prompt specifically for this ticker
                            fallback_prompt = f"""
Based on the following data for {ticker}, please generate a trading decision.
Current price: ${current_prices.get(ticker, 0):.2f}
Signals summary: {json.dumps(signals_by_ticker.get(ticker, {}), default=serialize_for_json)}

Return your response in JSON format with exactly this structure:
{{
  "action": "buy", "sell", "short", "cover", or "hold",
  "quantity": number of shares as an integer,
  "confidence": a number between 1 and 100,
  "reasoning": "A brief explanation of your decision"
}}

Remember to include the word "json" in your response.
                            """
                            
                            # Make an individual LLM call for this ticker
                            try:
                                single_output = call_llm(
                                    prompt=fallback_prompt,
                                    model_name=model_name,
                                    model_provider=model_provider,
                                    pydantic_model=PortfolioDecision,
                                    agent_name=f"portfolio_fallback_{ticker}",
                                )
                                
                                if single_output:
                                    fallback_decisions[ticker] = single_output
                                    logger.info(f"Successfully generated fallback decision for {ticker}")
                                else:
                                    logger.warning(f"Failed to generate fallback decision for {ticker}, using default")
                                    fallback_decisions[ticker] = create_default_decision(ticker)
                            except Exception as e:
                                logger.error(f"Error generating fallback decision for {ticker}: {e}")
                                fallback_decisions[ticker] = create_default_decision(ticker)
                
                # If we were able to generate decisions for all tickers, return them
                if len(fallback_decisions) == len(tickers):
                    logger.info(f"Successfully generated fallback decisions for all {len(tickers)} tickers")
                    return PortfolioManagerOutput(decisions=fallback_decisions)
                
                # If we still don't have decisions for all tickers, create defaults for missing ones
                for ticker in tickers:
                    if ticker not in fallback_decisions:
                        fallback_decisions[ticker] = create_default_decision(ticker)
                
                return PortfolioManagerOutput(decisions=fallback_decisions)
            except Exception as e:
                logger.error(f"Error in direct parsing fallback: {e}")
                
                # Last resort: create a default response for all tickers
                default_decisions = {ticker: create_default_decision(ticker) for ticker in tickers}
                return PortfolioManagerOutput(decisions=default_decisions)
            
        # Helper function to create a default decision for a ticker
        def create_default_decision(ticker):
            return PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=50.0,
                reasoning=f"Could not generate a valid decision for {ticker}",
                cancel_existing_orders=False
            )
        
        # Create a ChatPromptTemplate manually from our messages if needed
        prompt_for_llm = None
        if isinstance(messages, list):
            # If it's a list of messages, use it directly but ensure JSON is mentioned
            prompt_for_llm = messages
            
            # Ensure the human message contains the word 'json' for OpenAI
            if model_provider.lower() == "openai":
                # Log that we're ensuring JSON format for OpenAI
                logger.info("Ensuring JSON format is specified for OpenAI call")
                # Check the human message for 'json'
                if len(messages) > 1 and isinstance(messages[-1], HumanMessage):
                    human_message = messages[-1]
                    if "json" not in human_message.content.lower():
                        # Update the content to explicitly mention JSON
                        human_message.content += "\n\nReturn your response in JSON format."
        else:
            # If it's already a string, use it directly
            prompt_for_llm = messages

        # Call the LLM with detailed metrics
        trading_output = call_llm(
            prompt=prompt_for_llm,
            model_name=model_name,
            model_provider=model_provider,
            pydantic_model=PortfolioManagerOutput,
            agent_name="portfolio_management_agent",
            default_factory=direct_parsing_fallback
        )
        
        # Log the result structure to verify what we got back
        logger.info(f"LLM call complete. Result type: {type(trading_output)}")
        
        # Check if we got a proper output with decisions
        has_decisions = hasattr(trading_output, 'decisions') and trading_output.decisions
        
        # If we didn't get valid decisions, try to handle the response ourselves
        if not has_decisions:
            logger.warning("LLM call returned no decisions, trying to extract from raw response")
            
            try:
                # Try to access the raw output
                raw_output = None
                
                # Check if the output has a raw attribute
                if hasattr(trading_output, 'raw'):
                    raw_output = trading_output.raw
                    
                # Check if we can directly get content
                elif hasattr(trading_output, 'content'):
                    raw_output = trading_output.content
                
                # If we still don't have raw output, try to convert to string
                if not raw_output and hasattr(trading_output, '__str__'):
                    raw_output = str(trading_output)
                
                if raw_output:
                    logger.info(f"Found raw output: {raw_output[:100]}...")
                    
                    # Try to parse as JSON first
                    try:
                        # See if it's a JSON string
                        if isinstance(raw_output, str):
                            import re
                            json_match = re.search(r'(\{[\s\S]*\})', raw_output, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                response_dict = json.loads(json_str)
                                
                                # Check for different response formats
                                decisions_dict = {}
                                
                                # Check if we have a "decisions" key
                                if "decisions" in response_dict:
                                    decisions_dict = response_dict["decisions"]
                                # Check if we have ticker keys directly
                                elif any(ticker in response_dict for ticker in tickers):
                                    # We have ticker keys directly, create hold decisions
                                    for ticker in tickers:
                                        if ticker in response_dict:
                                            explanation = response_dict[ticker]
                                            decisions_dict[ticker] = {
                                                "action": "hold",
                                                "quantity": 0,
                                                "confidence": 50.0,
                                                "reasoning": explanation
                                            }
                                
                                # Create PortfolioDecisions from the dictionary
                                if decisions_dict:
                                    portfolio_decisions = {}
                                    for ticker, decision in decisions_dict.items():
                                        # Only process tickers that were requested
                                        if ticker in tickers:
                                            # Extract decision data
                                            action = decision.get("action", "hold").lower()
                                            
                                            # Validate action
                                            if action not in ["buy", "sell", "short", "cover", "hold"]:
                                                action = "hold"
                                                
                                            # Get quantity
                                            try:
                                                quantity = int(decision.get("quantity", 0))
                                            except (ValueError, TypeError):
                                                quantity = 0
                                                
                                            # Get confidence
                                            try:
                                                confidence = float(decision.get("confidence", 50.0))
                                                # Ensure confidence is within bounds
                                                confidence = max(1.0, min(100.0, confidence))
                                            except (ValueError, TypeError):
                                                confidence = 50.0
                                                
                                            # Get reasoning
                                            reasoning = str(decision.get("reasoning", "Extracted from raw response"))
                                            
                                            # Create decision
                                            portfolio_decisions[ticker] = PortfolioDecision(
                                                action=action,
                                                quantity=quantity,
                                                confidence=confidence,
                                                reasoning=reasoning
                                            )
                                    
                                    # If we created any decisions, use them
                                    if portfolio_decisions:
                                        logger.info(f"Created {len(portfolio_decisions)} decisions from raw response")
                                        trading_output = PortfolioManagerOutput(decisions=portfolio_decisions)
                                        has_decisions = True
                    except Exception as e:
                        logger.error(f"Error parsing raw output: {e}")
                        
                # If we still don't have decisions, check if it's a text response with ticker explanations
                if not has_decisions and raw_output and isinstance(raw_output, str):
                    # Let's look for ticker mentions and create hold decisions
                    portfolio_decisions = {}
                    
                    for ticker in tickers:
                        # Look for sections that talk about this ticker
                        if ticker in raw_output:
                            # Try to extract reasoning for this ticker
                            import re
                            ticker_pattern = fr'(?:["\s]|^){ticker}(?:["\s]|$)[^\n]*\n([^\n]+)'
                            ticker_match = re.search(ticker_pattern, raw_output)
                            
                            reasoning = f"Hold decision generated from LLM response mentioning {ticker}"
                            if ticker_match and ticker_match.group(1):
                                reasoning = ticker_match.group(1).strip()
                            
                            # Create a hold decision
                            portfolio_decisions[ticker] = PortfolioDecision(
                                action="hold",
                                quantity=0,
                                confidence=50.0,
                                reasoning=reasoning
                            )
                    
                    # If we found decisions for all tickers, use them
                    if len(portfolio_decisions) == len(tickers):
                        logger.info(f"Created decisions for all {len(tickers)} tickers from text analysis")
                        trading_output = PortfolioManagerOutput(decisions=portfolio_decisions)
                        has_decisions = True
                    
            except Exception as e:
                logger.error(f"Failed to extract decisions from raw output: {e}")
        
        if hasattr(trading_output, 'decisions'):
            decision_count = len(trading_output.decisions)
            logger.info(f"Received {decision_count} decisions from LLM")
            
            # Add any missing tickers with default decisions
            missing_tickers = set(tickers) - set(trading_output.decisions.keys())
            if missing_tickers:
                logger.warning(f"Missing decisions for tickers: {missing_tickers}")
                for ticker in missing_tickers:
                    # Create default hold decision for missing ticker
                    trading_output.decisions[ticker] = PortfolioDecision(
                        action="hold",
                        quantity=0,
                        confidence=50.0,
                        reasoning=f"No decision generated for {ticker} - using default hold action"
                    )
            
            # Log a summary of the decisions
            action_summary = {}
            for ticker, decision in trading_output.decisions.items():
                if ticker not in action_summary:
                    action_summary[ticker] = decision.action
            logger.info(f"Decision summary: {action_summary}")
        else:
            logger.error("No decisions attribute in LLM output")
        
        # Merge the LLM decisions with our open orders decisions
        if decisions_for_open_orders:
            # Get the decisions from the LLM output
            llm_decisions = trading_output.decisions
            
            # Merge with our open orders decisions (open orders take precedence)
            merged_decisions = {**llm_decisions, **decisions_for_open_orders}
            
            # Create a new output with the merged decisions
            trading_output = PortfolioManagerOutput(decisions=merged_decisions)
            logger.info(f"Merged {len(decisions_for_open_orders)} open order decisions with LLM output")
        
        return trading_output
    except Exception as e:
        logger.error(f"Error calling LLM for portfolio decisions: {e}", exc_info=True)
        # Merge any open orders decisions with the default ones
        default_output = create_default_portfolio_output(tickers)
        if decisions_for_open_orders:
            merged_decisions = {**default_output.decisions, **decisions_for_open_orders}
            default_output = PortfolioManagerOutput(decisions=merged_decisions)
        return default_output


# Create default factory for PortfolioManagerOutput
def create_default_portfolio_output(tickers=None):
    """
    Create default hold decisions when the portfolio manager fails.
    This acts as a safety fallback.
    
    Args:
        tickers: List of tickers to create decisions for. If None, returns empty decisions.
        
    Returns:
        PortfolioManagerOutput: Default portfolio decisions with "hold" action for each ticker
    """
    decisions = {}
    
    # Default to hold for all tickers
    if tickers:
        logger.info(f"Creating default 'hold' decisions for {len(tickers)} tickers")
        
        # Get the current stack trace to include in reasoning to help with debugging
        import traceback
        stack_trace = traceback.format_stack()
        stack_summary = "".join(stack_trace[-3:-1])  # Get the last few frames
        
        for ticker in tickers:
            decisions[ticker] = PortfolioDecision(
                action="hold", 
                quantity=0, 
                confidence=50.0, 
                reasoning=(
                    "Safety fallback: Using default hold action due to error in portfolio manager analysis. "
                    "The system encountered an issue while trying to generate a recommendation. "
                    "This is a temporary defensive action to protect your portfolio. "
                    "No trades will be executed until the system can properly analyze this ticker."
                ),
                cancel_existing_orders=False
            )
    else:
        logger.warning("No tickers provided to create_default_portfolio_output, returning empty decisions")
    
    return PortfolioManagerOutput(decisions=decisions)


def diagnose_alpaca_orders(client):
    """
    Comprehensive diagnostic function to check all orders in Alpaca account
    
    Args:
        client: Alpaca client
        
    Returns:
        Dictionary with diagnostic information
    """
    import logging
    import os
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus
    
    if not client:
        logging.error("No Alpaca client provided for diagnose_alpaca_orders")
        return {
            "error": "No Alpaca client provided",
            "orders": [],
            "order_count": 0
        }
    
    logging.info("========== ALPACA ORDER DIAGNOSTICS ==========")
    
    # Check if using paper trading
    paper_trading = os.getenv("LIVE_TRADING", "false").lower() != "true"
    environment = "paper trading" if paper_trading else "live trading"
    logging.info(f"Using {environment} environment")
    
    # Get account details
    try:
        account = client.get_account()
        logging.info(f"Account details: ID={account.id}, Status={account.status}")
    except Exception as e:
        logging.error(f"Error getting account details: {e}")
    
    # Try to get all orders regardless of status
    statuses_to_try = ["all", "open", "closed", "new", "partially_filled", "filled", 
                        "canceled", "expired", "pending_new", "accepted", "pending", 
                        "accepted_for_bidding", "stopped", "rejected", "suspended", 
                        "calculated", "done_for_day"]
    
    all_orders = []
    status_breakdown = {}
    
    # Try getting orders with GetOrdersRequest for all standard statuses
    for status_name in statuses_to_try:
        try:
            if status_name == "all":
                # For "all" status, don't specify a status
                orders = client.get_orders()
            else:
                # Try to get the proper enum value if possible
                try:
                    status_enum = getattr(OrderStatus, status_name.upper())
                    request_params = GetOrdersRequest(status=status_enum)
                    orders = client.get_orders(request_params)
                except AttributeError:
                    # If the enum doesn't exist, try the string directly
                    request_params = GetOrdersRequest(status=status_name)
                    orders = client.get_orders(request_params)
                
            if orders:
                logging.info(f"Status '{status_name}' returned {len(orders)} orders")
                status_breakdown[status_name] = len(orders)
                
                # Only add to all_orders if not already there (avoid duplicates)
                for order in orders:
                    if not any(o.id == order.id for o in all_orders):
                        all_orders.append(order)
        except Exception as e:
            logging.info(f"Status '{status_name}' not supported or error: {e}")
    
    # Check specifically for AAPL and VOO orders
    aapl_orders = [o for o in all_orders if o.symbol == "AAPL"]
    voo_orders = [o for o in all_orders if o.symbol == "VOO"]
    
    logging.info(f"Found {len(aapl_orders)} orders for AAPL")
    if aapl_orders:
        for i, order in enumerate(aapl_orders[:3]):  # Log up to 3 orders
            logging.info(f"AAPL Order {i+1}: ID={order.id}, Status={order.status}, Side={order.side}, Qty={order.qty}")
    
    logging.info(f"Found {len(voo_orders)} orders for VOO")
    if voo_orders:
        for i, order in enumerate(voo_orders[:3]):  # Log up to 3 orders
            logging.info(f"VOO Order {i+1}: ID={order.id}, Status={order.status}, Side={order.side}, Qty={order.qty}")
    
    logging.info(f"Retrieved total of {len(all_orders)} orders across all status types")
    logging.info("========== END ALPACA ORDER DIAGNOSTICS ==========")
    
    return {
        "environment": environment,
        "orders": all_orders,
        "order_count": len(all_orders),
        "status_breakdown": status_breakdown,
        "aapl_orders": len(aapl_orders),
        "voo_orders": len(voo_orders)
    }


def cancel_open_orders_for_ticker(client, ticker):
    """
    Cancel all open orders for a specific ticker
    
    Args:
        client: Alpaca client
        ticker: Ticker symbol to cancel orders for
        
    Returns:
        dict: Results of the cancellation operation including count and details
    """
    logger.info(f"=== CANCEL ORDERS START: {ticker} ===")
    
    if not client:
        logger.warning("No Alpaca client provided for canceling orders")
        return {"success": False, "reason": "No Alpaca client provided", "count": 0, "ticker": ticker}
        
    logger.info(f"Attempting to cancel all open orders for {ticker}")
    
    try:
        # First, check if we're in paper trading mode
        try:
            account = client.get_account()
            is_paper = getattr(account, 'is_paper', True)
            logger.info(f"Canceling orders in {'paper' if is_paper else 'live'} trading mode")
        except Exception as e:
            logger.warning(f"Could not determine trading mode: {e}")
            is_paper = True  # Default to paper if we can't detect
            
        # Get open orders for this ticker
        open_orders = get_alpaca_open_orders(client, ticker)
        
        if ticker not in open_orders or not open_orders[ticker]:
            logger.info(f"No open orders found for {ticker}")
            return {"success": False, "reason": "No open orders found", "count": 0, "ticker": ticker}
            
        # Log the orders we found
        logger.info(f"Found {len(open_orders[ticker])} orders to cancel for {ticker}")
        for i, order in enumerate(open_orders[ticker]):
            logger.info(f"  Order {i+1}: ID={order.id}, Status={order.status}, Side={order.side}, Qty={order.qty}")
            
        # Cancel each order
        cancel_count = 0
        canceled_order_ids = []
        failed_order_ids = []
        
        for order in open_orders[ticker]:
            try:
                order_id = order.id
                logger.info(f"Attempting to cancel order {order_id} for {ticker}")
                
                # Try a direct cancel with enhanced error handling
                try:
                    # Cancel the order
                    logger.info(f"Sending cancel request for order {order_id}")
                    client.cancel_order_by_id(order_id)
                    logger.info(f"Cancel request sent successfully for order {order_id}")
                except Exception as e:
                    error_message = str(e).lower()
                    # Check if the error indicates the order is already canceled or doesn't exist
                    if "order not found" in error_message or "already canceled" in error_message:
                        logger.info(f"Order {order_id} already canceled or not found: {e}")
                        cancel_count += 1
                        canceled_order_ids.append(order_id)
                        continue
                    else:
                        logger.error(f"Error canceling order {order_id}: {e}")
                        failed_order_ids.append(order_id)
                        continue
                
                # Verify the order is actually canceled
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Get the order to check if it's canceled
                        updated_order = client.get_order_by_id(order_id)
                        order_status = getattr(updated_order, 'status', '').lower()
                        
                        if order_status == "canceled":
                            logger.info(f"Confirmed order {order_id} for {ticker} is canceled (status: {order_status})")
                            cancel_count += 1
                            canceled_order_ids.append(order_id)
                            break
                        else:
                            logger.warning(f"Order {order_id} for {ticker} not yet canceled (status: {order_status}), retry {retry+1}/{max_retries}")
                            if retry < max_retries - 1:
                                logger.info(f"Waiting 1 second before retry {retry+2}...")
                                time.sleep(1)  # Wait a second before retrying
                    except Exception as e:
                        error_message = str(e).lower()
                        # If we get an error like "order not found", it might be because it's canceled
                        if "order not found" in error_message:
                            logger.info(f"Order {order_id} for {ticker} not found - assuming it's canceled")
                            cancel_count += 1
                            canceled_order_ids.append(order_id)
                            break
                        else:
                            logger.warning(f"Error checking order {order_id} status: {e}")
                            if retry < max_retries - 1:
                                logger.info(f"Waiting 1 second before retry {retry+2}...")
                                time.sleep(1)  # Wait a second before retrying
                else:
                    # If we get here, all retries failed
                    logger.error(f"Failed to confirm cancellation of order {order_id} for {ticker} after {max_retries} retries")
                    failed_order_ids.append(order_id)
            except Exception as e:
                logger.error(f"Error in cancel process for order {order.id} for {ticker}: {e}")
                logger.exception("Detailed error traceback:")
                failed_order_ids.append(order.id)
                
        # Do a final check to see if there are any remaining orders
        try:
            remaining_orders = get_alpaca_open_orders(client, ticker)
            if ticker in remaining_orders and remaining_orders[ticker]:
                remaining_count = len(remaining_orders[ticker])
                logger.warning(f"After cancellation process, still found {remaining_count} orders for {ticker}")
                
                # Log the remaining orders
                for i, order in enumerate(remaining_orders[ticker]):
                    order_id = getattr(order, 'id', 'unknown')
                    logger.warning(f"  Remaining order {i+1}: ID={order_id}")
                    
                    # If this was supposed to be canceled but still exists, add it to failed list
                    if order_id in canceled_order_ids:
                        logger.error(f"Order {order_id} was reported as canceled but still exists!")
                        # Move from canceled to failed
                        canceled_order_ids.remove(order_id)
                        if order_id not in failed_order_ids:
                            failed_order_ids.append(order_id)
                        cancel_count -= 1
        except Exception as e:
            logger.error(f"Error in final verification of cancellations: {e}")
                
        logger.info(f"Canceled {cancel_count} orders for {ticker}, {len(failed_order_ids)} failed to cancel")
        
        result = {
            "success": cancel_count > 0,
            "count": cancel_count,
            "ticker": ticker,
            "total_orders": len(open_orders[ticker]),
            "canceled_order_ids": canceled_order_ids,
            "failed_order_ids": failed_order_ids
        }
        
        logger.info(f"=== CANCEL ORDERS RESULT: {ticker} - {cancel_count}/{len(open_orders[ticker])} canceled ===")
        return result
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error canceling orders for {ticker}: {error_message}")
        logger.exception("Detailed error traceback:")
        
        logger.info(f"=== CANCEL ORDERS FAILED: {ticker} ===")
        return {"success": False, "reason": error_message, "count": 0, "ticker": ticker}


def calculate_max_shares_per_ticker(portfolio, tickers):
    """
    Calculate the maximum number of shares to trade for each ticker.
    
    Args:
        portfolio: Portfolio dictionary containing cash, positions, etc.
        tickers: List of tickers to calculate max shares for
        
    Returns:
        Dict mapping ticker to maximum shares that can be traded
    """
    max_shares = {}
    
    # Get portfolio cash and total value
    portfolio_cash = portfolio.get("cash", 100000.0)
    portfolio_value = portfolio.get("portfolio_value", portfolio_cash)
    
    # Default risk limit - don't put more than 5% of portfolio in any one position
    risk_limit_pct = 0.05
    
    # Check if we have a risk dashboard with more specific limits
    risk_dashboard = portfolio.get("risk_dashboard", {})
    if risk_dashboard and "position_limits" in risk_dashboard:
        position_limits = risk_dashboard["position_limits"]
        if isinstance(position_limits, dict):
            # See if we have ticker-specific limits
            for ticker in tickers:
                if ticker in position_limits:
                    ticker_limit = position_limits[ticker]
                    max_position_value = portfolio_value * ticker_limit
                    
                    # Get current price (default to $100 if not available)
                    current_price = 100.0
                    if "current_prices" in portfolio and ticker in portfolio["current_prices"]:
                        current_price = portfolio["current_prices"][ticker]
                    elif "positions" in portfolio and ticker in portfolio["positions"]:
                        position = portfolio["positions"][ticker]
                        if "current_price" in position:
                            current_price = position["current_price"]
                    
                    # Calculate max shares based on position limit
                    max_shares_by_limit = int(max_position_value / current_price)
                    
                    # Limit by available cash as well
                    max_shares_by_cash = int(portfolio_cash / current_price * 0.95)  # Use 95% of cash at most
                    
                    # Use the smaller of the two limits
                    max_shares[ticker] = min(max_shares_by_limit, max_shares_by_cash)
                    
                    # Ensure we have at least a minimum number of shares
                    max_shares[ticker] = max(max_shares[ticker], 5)
    
    # For any tickers that didn't have specific limits
    for ticker in tickers:
        if ticker not in max_shares:
            # Get current price (default to $100 if not available)
            current_price = 100.0
            if "current_prices" in portfolio and ticker in portfolio["current_prices"]:
                current_price = portfolio["current_prices"][ticker]
            elif "positions" in portfolio and ticker in portfolio["positions"]:
                position = portfolio["positions"][ticker]
                if "current_price" in position:
                    current_price = position["current_price"]
            
            # Calculate max shares based on default risk limit
            max_position_value = portfolio_value * risk_limit_pct
            max_shares_by_limit = int(max_position_value / current_price)
            
            # Limit by available cash as well
            max_shares_by_cash = int(portfolio_cash / current_price * 0.95)  # Use 95% of cash at most
            
            # Use the smaller of the two limits
            max_shares[ticker] = min(max_shares_by_limit, max_shares_by_cash)
            
            # Ensure we have at least a minimum number of shares
            max_shares[ticker] = max(max_shares[ticker], 5)
    
    # Log the calculated values
    logging.info(f"Calculated max shares: {max_shares}")
    
    return max_shares
