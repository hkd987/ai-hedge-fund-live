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


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]
    price_data = state["data"].get("price_data", {})

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
                state["data"]["portfolio"] = portfolio
                
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
        
        # Extract max shares for each ticker
        for ticker in tickers:
            if ticker in risk_signals:
                ticker_risk = risk_signals[ticker]
                max_shares[ticker] = ticker_risk.get("max_shares", 0)
    
    # Create default max shares if needed
    for ticker in tickers:
        if ticker not in max_shares:
            # Default to 100 shares as a fallback
            max_shares[ticker] = 100

    # Generate trading decisions using LLM
    progress.update_status("portfolio_management_agent", None, "Generating trading decisions")
    trading_output = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=analyst_signals,
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

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
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
    Generate trading decisions for multiple tickers based on analyst signals.
    
    Args:
        tickers: List of ticker symbols.
        signals_by_ticker: Dictionary of ticker to analyst signals.
        current_prices: Dictionary of ticker to current price.
        max_shares: Dictionary of ticker to maximum shares allowed by risk manager.
        portfolio: Portfolio data including positions and cash.
        risk_dashboard: Risk dashboard data from risk manager.
        model_name: Name of the LLM model to use.
        model_provider: Provider of the LLM model to use.
        
    Returns:
        PortfolioManagerOutput: Trading decisions for each ticker.
    """
    # Get open orders from portfolio if available
    open_orders = portfolio.get("open_orders", {})
    
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
              "system",
              """You are a portfolio manager making final trading decisions based on multiple tickers.

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

              - The max_shares values are pre-calculated to respect position limits
              - Consider both long and short opportunities based on signals
              - Maintain appropriate risk management with both long and short exposure
              - Trades will have automated stop losses to protect against significant losses
              
              Enhanced Risk Guidelines:
              - Sector Exposure: Reduce position sizes in sectors that are already heavily weighted in the portfolio
              - Correlation Risk: Avoid concentrated positions in highly correlated assets
              - Market Regime: Adjust risk appetite based on the current market regime (bull, bear, or correction)
              - Liquidity Constraints: Be more cautious with less liquid assets
              - Volatility-Based Risk: Higher volatility assets should have smaller position sizes
              - Time-Based Risk: Consider earnings announcements and market hours in your decisions
              - Portfolio Diversification: Maintain appropriate diversification across sectors and asset types
              - Circuit Breaker Awareness: Respect circuit breaker status in risk decisions

              Available Actions:
              - "buy": Open or add to long position
              - "sell": Close or reduce long position
              - "short": Open or add to short position
              - "cover": Close or reduce short position
              - "hold": No action

              Important Notes on Open Orders:
              - Some tickers may already have open orders that have not yet executed
              - Do NOT place new orders for tickers that already have open orders of the same type
              - If a ticker has an open BUY order, do not place another BUY order
              - If a ticker has an open SELL order, do not place another SELL order
              - For tickers with open orders, use "hold" action with confidence 50.0 and explain
              """,
            ),
            (
              "human",
              """Based on the team's analysis, make your trading decisions for each ticker.

              Here are the signals by ticker:
              {signals_by_ticker}

              Current Prices:
              {current_prices}

              Maximum Shares Allowed For Purchases:
              {max_shares}

              Portfolio Cash: {portfolio_cash}
              Current Positions: {portfolio_positions}
              Current Margin Requirement: {margin_requirement}
              Risk Dashboard: {risk_dashboard}
              
              Open Orders Information:
              {open_orders_info}

              Output strictly in JSON with the following structure:
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": integer,
                    "confidence": float between 0 and 100,
                    "reasoning": "string"
                  }},
                  "TICKER2": {{
                    ...
                  }},
                  ...
                }}
              }}
              """,
            ),
        ]
    )
    
    # Format open orders information for the prompt
    open_orders_info = {}
    for ticker, orders in open_orders.items():
        if ticker in tickers:
            order_details = []
            for order in orders:
                order_details.append({
                    "side": order.side,
                    "qty": order.qty,
                    "type": order.type,
                    "status": order.status
                })
            open_orders_info[ticker] = order_details
    
    # Generate the prompt
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
            "risk_dashboard": json.dumps(risk_dashboard, indent=2) if risk_dashboard else "{}",
            "open_orders_info": json.dumps(open_orders_info, indent=2)
        }
    )
    
    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        # Create default hold decisions, respecting open orders
        decisions = {}
        for ticker in tickers:
            reasoning = "Error in portfolio management, defaulting to hold"
            if ticker in open_orders and open_orders[ticker]:
                reasoning = f"There are already open orders for {ticker}. Waiting for them to execute."
            
            decisions[ticker] = PortfolioDecision(
                action="hold", 
                quantity=0, 
                confidence=0.0, 
                reasoning=reasoning
            )
        return PortfolioManagerOutput(decisions=decisions)
    
    # Call the LLM to generate trading decisions
    result = call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=PortfolioManagerOutput, 
        agent_name="portfolio_management_agent", 
        default_factory=create_default_portfolio_output
    )
    
    # Post-process the decisions to enforce open order constraints
    if hasattr(result, 'decisions'):
        for ticker, decision in result.decisions.items():
            # If there are open orders for this ticker, override with hold
            if ticker in open_orders and open_orders[ticker]:
                open_order_sides = [order.side for order in open_orders[ticker]]
                decision.action = "hold"
                decision.quantity = 0
                decision.confidence = 50.0
                decision.reasoning = f"There are already open {', '.join(open_order_sides)} orders for {ticker}. Waiting for them to execute."
                
                # Log the override
                logger.info(f"Overriding decision for {ticker} due to existing open orders: {open_order_sides}")
    
    return result
