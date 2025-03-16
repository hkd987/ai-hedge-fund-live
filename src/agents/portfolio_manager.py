import json
import os
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import logging

from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
from utils.caching import cached_analyst

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
        BracketOrderRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
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
    Get all open orders from Alpaca, optionally filtered by ticker.
    
    Args:
        client: Initialized Alpaca client
        ticker: Optional ticker symbol to filter by
        
    Returns:
        dict: Dictionary of ticker to list of open orders
    """
    if not client:
        logger.warning("No Alpaca client available. Cannot get open orders.")
        return {}
    
    try:
        # Get all open orders
        if ticker:
            orders = client.get_orders(status="open", symbols=ticker)
        else:
            orders = client.get_orders(status="open")
        
        # Organize by ticker
        orders_by_ticker = {}
        for order in orders:
            symbol = order.symbol
            if symbol not in orders_by_ticker:
                orders_by_ticker[symbol] = []
            orders_by_ticker[symbol].append(order)
        
        # Log order information
        for ticker, ticker_orders in orders_by_ticker.items():
            logger.info(f"Found {len(ticker_orders)} open orders for {ticker}")
            for order in ticker_orders:
                logger.info(f"  Order {order.id}: {order.side} {order.qty} shares at ${order.limit_price if hasattr(order, 'limit_price') and order.limit_price else 'market'}")
        
        return orders_by_ticker
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}")
        return {}


def execute_alpaca_trade(ticker, action, quantity, current_price, prices_df=None):
    """Execute a trade with Alpaca, applying risk management rules"""
    # Check if the trade is allowed by our risk management system
    if not can_execute_trade(ticker, action, quantity, current_price):
        logger.warning(f"Trade rejected by risk management: {action} {quantity} shares of {ticker}")
        return False
    
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
    if not client:
        logger.error("Alpaca client not available")
        return False
    
    # Check for existing open orders for this ticker
    open_orders = get_alpaca_open_orders(client, ticker)
    if ticker in open_orders and open_orders[ticker]:
        existing_orders = open_orders[ticker]
        
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
                return False
    
    # Map our action to Alpaca's OrderSide and determine if it's a short order
    side_mapping = {
        "buy": OrderSide.BUY,
        "sell": OrderSide.SELL,
        "short": OrderSide.SELL,
        "cover": OrderSide.BUY
    }
    
    if action not in side_mapping:
        logger.error(f"Unsupported action: {action}")
        return False
    
    side = side_mapping[action]
    
    try:
        # For short and cover, we need to check the current position
        if action in ["short", "cover"]:
            # Get the current position
            try:
                position = client.get_position(ticker)
                current_qty = int(position.qty)
                
                # Handle cover (closing a short position)
                if action == "cover":
                    if current_qty >= 0:  # Not a short position
                        logger.warning(f"Cannot cover {ticker}: No short position exists")
                        return False
                    
                    # Limit quantity to current short position
                    if abs(current_qty) < quantity:
                        logger.info(f"Adjusting cover quantity from {quantity} to {abs(current_qty)} for {ticker}")
                        quantity = abs(current_qty)
                
                # Handle short (creating a short position or adding to existing short)
                elif action == "short":
                    if current_qty < 0:  # Already have a short position
                        # Check if this would exceed our maximum short target
                        # This is a safety check to prevent doubling up on shorts
                        logger.info(f"Already have a short position of {abs(current_qty)} shares for {ticker}")
                        # You can add additional logic here if needed
            except Exception as e:
                # Position doesn't exist
                if action == "cover":
                    logger.warning(f"Cannot cover {ticker}: No position exists")
                    return False
                elif action == "short":
                    # This is a new short position, no special handling needed
                    pass
        
        # For buy and sell, verify the current position
        elif action in ["buy", "sell"]:
            try:
                position = client.get_position(ticker)
                current_qty = int(position.qty)
                
                # Handle sell (closing a long position)
                if action == "sell":
                    if current_qty <= 0:  # Not a long position
                        logger.warning(f"Cannot sell {ticker}: No long position exists")
                        return False
                    
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
                if action == "sell":
                    logger.warning(f"Cannot sell {ticker}: No position exists")
                    return False
                elif action == "buy":
                    # This is a new buy position, no special handling needed
                    pass
        
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
        
        # Submit the order
        order = client.submit_order(order_data)
        logger.info(f"Submitted {action} order for {quantity} shares of {ticker}: Order ID {order.id}")
        
        # Record the successful trade in our risk management system
        record_trade_execution(ticker, action, quantity, current_price, quantity * current_price)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to submit {action} order for {ticker}: {e}")
        return False


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
                if confidence == 0:
                    confidence = 50.0
                    
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
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Ensure we're working with a dictionary, not the Alpaca client
    # Extract state data safely
    if not isinstance(state, dict):
        # If state is a class instance with a dict() method
        if hasattr(state, 'dict') and callable(getattr(state, 'dict')):
            state_dict = state.dict()
        # If state is a class instance with a get() method
        elif hasattr(state, 'get') and callable(getattr(state, 'get')):
            state_dict = {'data': state.get('data', {})}
        else:
            # Last resort, try to extract data from state's attributes
            state_dict = {'data': getattr(state, 'data', {})}
    else:
        state_dict = state
    
    # Now safely extract portfolio and other data
    data = state_dict.get('data', {})
    portfolio = data.get('portfolio', {})
    analyst_signals = data.get('analyst_signals', {})
    tickers = data.get('tickers', [])
    price_data = data.get('price_data', {})
    
    # Ensure portfolio is a dictionary
    if not isinstance(portfolio, dict):
        logger.error(f"Portfolio is not a dictionary: {type(portfolio)}")
        portfolio = {}
        
    # Ensure portfolio has a reasonable cash value
    if not portfolio or "cash" not in portfolio or portfolio.get("cash", 0) < 1000:
        logger.warning("Portfolio has no cash or unreasonably low cash, initializing with default")
        if not portfolio:
            portfolio = {}
        portfolio["cash"] = 100000.0
        if isinstance(state_dict, dict) and isinstance(state_dict.get('data'), dict):
            state_dict["data"]["portfolio"] = portfolio

    logger.info(f"Initial portfolio cash: ${portfolio.get('cash', 0)}")

    # If live trading is enabled, try to get current portfolio state from Alpaca
    alpaca_client = None
    open_orders_by_ticker = {}
    if LIVE_TRADING_ENABLED:
        progress.update_status("portfolio_management_agent", None, "Getting current portfolio state from Alpaca")
        
        alpaca_client = get_alpaca_client()
        if alpaca_client:
            # Get open orders
            open_orders_by_ticker = get_alpaca_open_orders(alpaca_client)
            
            # Get portfolio state
            alpaca_portfolio = get_alpaca_portfolio_state(alpaca_client, tickers)
            if alpaca_portfolio:
                logger.info("Using portfolio state from Alpaca")
                portfolio = alpaca_portfolio
                # Update the portfolio in the state data for other agents to use
                state_dict["data"]["portfolio"] = portfolio
                
                # Initialize risk management with portfolio value if not already set
                portfolio_value = float(alpaca_client.get_account().equity)
                update_portfolio_value(portfolio_value)
            else:
                logger.warning("Could not get portfolio state from Alpaca, using existing portfolio data")
        else:
            logger.warning("Alpaca client not available, using existing portfolio data")

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    # Generate comprehensive risk dashboard if we have price data
    risk_dashboard = None
    market_data = price_data.get('SPY', None)
    prices_by_ticker = {}
    
    # Extract price dataframes for each ticker
    for ticker in tickers:
        if ticker in price_data:
            prices_by_ticker[ticker] = price_data[ticker]
    
    # If we have market data, generate a risk dashboard
    if len(prices_by_ticker) > 0 and market_data is not None:
        try:
            risk_dashboard = generate_risk_dashboard(portfolio, tickers, prices_by_ticker)
            logger.info("Generated comprehensive risk dashboard")
        except Exception as e:
            logger.error(f"Failed to generate risk dashboard: {e}")
    
    # Get current prices for each ticker from the most recent price data
    current_prices = {}
    for ticker in tickers:
        if ticker in price_data:
            # Get the last closing price from the price data
            current_prices[ticker] = price_data[ticker]['close'].iloc[-1]
    
    # Get maximum shares per ticker from the risk management agent
    max_shares = {}
    if "risk_management_agent" in analyst_signals:
        risk_signals = analyst_signals["risk_management_agent"]
        
        # Log the structure of risk signals to help with debugging
        logger.info(f"Risk signals type: {type(risk_signals)}")
        if isinstance(risk_signals, dict):
            logger.info(f"Risk signals keys: {list(risk_signals.keys())}")
        
        # Extract max shares for each ticker
        for ticker in tickers:
            if ticker in risk_signals:
                try:
                    ticker_risk = risk_signals[ticker]
                    
                    # Check if the ticker_risk object is valid
                    if ticker_risk is None:
                        logger.warning(f"Risk signal for {ticker} is None, using default")
                        max_shares[ticker] = 100  # Default value
                        continue
                    
                    # Try to get max_shares directly from the dictionary or attribute
                    ticker_max_shares = None
                    
                    # First check if it's a dictionary
                    if isinstance(ticker_risk, dict):
                        ticker_max_shares = ticker_risk.get("max_shares")
                        
                        # Log confidence score for debugging if available
                        if "confidence" in ticker_risk:
                            logger.info(f"Risk confidence for {ticker}: {ticker_risk['confidence']}")
                            
                    # Then try attribute access (for objects)
                    elif hasattr(ticker_risk, "max_shares"):
                        ticker_max_shares = ticker_risk.max_shares
                        
                        # Log confidence score for debugging if available
                        if hasattr(ticker_risk, "confidence"):
                            logger.info(f"Risk confidence for {ticker}: {ticker_risk.confidence}")
                    
                    # Validate the max_shares value
                    if ticker_max_shares is not None and ticker_max_shares > 0:
                        max_shares[ticker] = ticker_max_shares
                        logger.info(f"Using max_shares={ticker_max_shares} for {ticker} from risk manager")
                    else:
                        logger.warning(f"Invalid max_shares in risk signal for {ticker}, using default")
                        max_shares[ticker] = 100  # Default value
                        
                except Exception as e:
                    logger.error(f"Error processing risk signal for {ticker}: {e}")
                    max_shares[ticker] = 100  # Default value
            else:
                logger.warning(f"No risk signal found for {ticker}, using default max_shares")
                max_shares[ticker] = 100  # Default
    else:
        logger.warning("No risk_management_agent found in analyst_signals, using default max_shares for all tickers")
        for ticker in tickers:
            max_shares[ticker] = 100  # Default
            
    # Log the final max_shares values
    logger.info(f"Final max_shares values: {max_shares}")

    # Create a simplified version of analyst signals for the trading decision function
    simplified_signals = {}
    
    for analyst_name, signals in analyst_signals.items():
        if isinstance(signals, dict):
            simplified_analyst_signals = {}
            
            for ticker in tickers:
                if ticker in signals:
                    signal = signals[ticker]
                    
                    # Skip None signals
                    if signal is None:
                        logger.warning(f"Skipping None signal from {analyst_name} for {ticker}")
                        continue
                        
                    # Fix signals with 0% confidence
                    if hasattr(signal, 'confidence') and signal.confidence == 0:
                        logger.warning(f"Fixing 0% confidence signal from {analyst_name} for {ticker}")
                        if hasattr(signal, 'confidence'):
                            signal.confidence = 50.0  # Set to neutral confidence
                    
                    simplified_analyst_signals[ticker] = signal
            
            if simplified_analyst_signals:
                simplified_signals[analyst_name] = simplified_analyst_signals
    
    # Generate trading decisions using LLM
    progress.update_status("portfolio_management_agent", None, "Generating trading decisions")
    trading_output = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=simplified_signals,  # Use simplified signals
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        risk_dashboard=risk_dashboard,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )
    
    # Format the message for the output
    reasoning = {}
    decisions = {}
    
    # Extract the decisions from the trading output
    for ticker, decision in trading_output.decisions.items():
        if ticker in tickers:
            # Only include tickers that were requested
            decisions[ticker] = decision
            reasoning[ticker] = decision.reasoning
    
    message_text = json.dumps({"decisions": decisions}, default=lambda x: x.model_dump())
    message = HumanMessage(
        content=message_text,
        name="portfolio_management_agent",
    )
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(reasoning, "Portfolio Management Agent")
    
    # Execute trades if live trading is enabled
    if LIVE_TRADING_ENABLED:
        progress.update_status("portfolio_management_agent", None, "Executing trades")
        for ticker, decision in decisions.items():
            if decision.action != "hold" and decision.quantity > 0:
                # Check if there are already open orders for this ticker
                ticker_has_open_orders = ticker in open_orders_by_ticker and open_orders_by_ticker[ticker]
                if ticker_has_open_orders:
                    open_order_sides = [order.side for order in open_orders_by_ticker[ticker]]
                    logger.warning(f"Skipping {decision.action} for {ticker} - there are already open orders: {open_order_sides}")
                    continue
                
                # Get the current position for this ticker
                position_data = portfolio.get("positions", {}).get(ticker, {})
                has_long = position_data.get("long", 0) > 0
                has_short = position_data.get("short", 0) > 0
                
                # Validate the decision based on current positions
                if decision.action == "sell" and not has_long:
                    logger.warning(f"Cannot sell {ticker}: No long position exists")
                    continue
                elif decision.action == "cover" and not has_short:
                    logger.warning(f"Cannot cover {ticker}: No short position exists")
                    continue
                    
                # Apply additional risk adjustments
                original_quantity = decision.quantity
                adjusted_quantity = decision.quantity
                adjustment_reasons = []
                
                # Check for market regime-based adjustments
                if risk_dashboard and "market_regime" in risk_dashboard:
                    if risk_dashboard["market_regime"] == "bear":
                        adjusted_quantity = int(adjusted_quantity * 0.5)  # Reduce by 50% in bear markets
                        adjustment_reasons.append("bear market detected")
                    elif risk_dashboard["market_regime"] == "high_volatility":
                        adjusted_quantity = int(adjusted_quantity * 0.7)  # Reduce by 30% in high volatility
                        adjustment_reasons.append("high market volatility")
                
                # Check for pre-earnings announcements
                ticker_risk = risk_signals.get(ticker, {})
                if "position_risk_params" in ticker_risk and ticker_risk["position_risk_params"]:
                    risk_params = ticker_risk["position_risk_params"]
                    if risk_params.get("upcoming_earnings", False):
                        adjusted_quantity = int(adjusted_quantity * 0.5)  # Reduce by 50% before earnings
                        adjustment_reasons.append("upcoming earnings announcement")
                
                # Update the decision quantity
                decision.quantity = adjusted_quantity
                
                # Log any adjustments
                if original_quantity != adjusted_quantity:
                    logger.warning(f"{ticker}: {', '.join(adjustment_reasons)}. Reducing order from {original_quantity} to {adjusted_quantity} shares")
                
                # Check for existing open orders if using Alpaca
                if LIVE_TRADING_ENABLED and ticker in open_orders_by_ticker and open_orders_by_ticker[ticker]:
                    existing_orders = open_orders_by_ticker[ticker]
                    logger.info(f"{ticker} already has {len(existing_orders)} open orders. Checking compatibility...")
                    
                    # Check if any orders conflict with the current decision
                    conflicting_order = False
                    for order in existing_orders:
                        order_side = order.side
                        
                        # Check if existing order conflicts with our action
                        if (decision.action in ["buy", "cover"] and order_side == "buy") or \
                           (decision.action in ["sell", "short"] and order_side == "sell"):
                            logger.warning(f"There is already an open {order_side} order for {ticker}. Canceling new {decision.action} order.")
                            decision.quantity = 0  # Zero out quantity to skip this order
                            conflicting_order = True
                            break
                    
                    if conflicting_order:
                        decision.reasoning += f"\nOrder canceled due to existing open {order_side} orders for {ticker}."
                        continue
                
                # Execute the trade if quantity is still positive after all risk adjustments
                if decision.quantity > 0:
                    progress.update_status(
                        "portfolio_management_agent", 
                        ticker, 
                        f"Executing {decision.action} order for {decision.quantity} shares"
                    )
                    
                    # Pass price data to execute_alpaca_trade for dynamic risk calculations
                    ticker_prices = prices_by_ticker.get(ticker)
                    success = execute_alpaca_trade(
                        ticker, 
                        decision.action, 
                        decision.quantity, 
                        current_prices[ticker],
                        prices_df=ticker_prices
                    )
                    
                    if success:
                        logger.info(f"Successfully executed {decision.action} order for {decision.quantity} shares of {ticker}")
                    else:
                        logger.warning(f"Failed to execute {decision.action} order for {decision.quantity} shares of {ticker}")
                else:
                    logger.warning(f"Order for {ticker} canceled due to risk management: Quantity reduced to 0 after all risk adjustments")
    else:
        progress.update_status("portfolio_management_agent", None, "Live trading disabled (simulation only)")

    progress.update_status("portfolio_management_agent", None, "Done")

    # Prepare the return message
    message = HumanMessage(content=f"Portfolio management analysis complete. Generated {len(decisions)} decisions.")
    
    # Important: Store the decisions in the data for other agents to use
    if isinstance(state_dict, dict) and isinstance(state_dict.get('data'), dict):
        # Store the raw decisions
        state_dict["data"]["portfolio_decisions"] = decisions
        
        # Also store in analyst_signals to make it easier to find
        if "analyst_signals" not in state_dict["data"]:
            state_dict["data"]["analyst_signals"] = {}
        state_dict["data"]["analyst_signals"]["portfolio_management_agent"] = {"decisions": decisions}
            
    # Log the decisions before returning
    logger.info(f"Portfolio manager generated {len(decisions)} decisions: {list(decisions.keys() if isinstance(decisions, dict) else [])}")

    return {
        "messages": state["messages"] + [message],
        "data": state_dict["data"],
        "decisions": decisions,  # Include decisions directly in the returned state
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
                reasoning=f"Could not generate a valid decision for {ticker}"
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
                )
            )
    else:
        logger.warning("No tickers provided to create_default_portfolio_output, returning empty decisions")
    
    return PortfolioManagerOutput(decisions=decisions)
