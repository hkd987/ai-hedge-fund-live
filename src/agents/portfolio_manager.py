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
                
            except Exception as e:
                # Position doesn't exist
                if action == "cover":
                    logger.warning(f"Cannot cover {ticker}: No position exists")
                    return False
                elif action == "short":
                    # This is a new short position, no special handling needed
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
        }
        
        # Get all positions
        all_positions = {}
        try:
            positions = client.get_all_positions()
            for position in positions:
                all_positions[position.symbol] = position
        except Exception as e:
            logger.warning(f"Failed to get positions from Alpaca: {e}")
        
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
    if LIVE_TRADING_ENABLED:
        progress.update_status("portfolio_management_agent", None, "Getting current portfolio state from Alpaca")
        
        alpaca_client = get_alpaca_client()
        if alpaca_client:
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
    if market_data is not None:
        try:
            progress.update_status("portfolio_management_agent", None, "Generating risk dashboard")
            risk_dashboard = generate_risk_dashboard(portfolio, tickers, price_data)
            logger.info("Successfully generated comprehensive risk dashboard")
        except Exception as e:
            logger.warning(f"Failed to generate risk dashboard: {e}")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    prices_by_ticker = {}
    
    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)
        
        # Store price data for use in trade execution
        prices_by_ticker[ticker] = price_data.get(ticker)
        
        # Get max shares directly from risk manager if available, otherwise calculate it
        if "max_shares" in risk_data:
            max_shares[ticker] = risk_data.get("max_shares", 0)
        else:
            # Calculate maximum shares allowed based on position limit and price
            if current_prices[ticker] > 0:
                max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
            else:
                max_shares[ticker] = 0
                
        # Check if circuit breaker is active
        if risk_data.get("circuit_breaker_active", False):
            # No trades allowed when circuit breaker is active
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                # Handle both dictionary-style and object-style signals (Pydantic models)
                signal_obj = signals[ticker]
                if hasattr(signal_obj, 'model_dump'):  # It's a Pydantic model
                    signal_dict = signal_obj.model_dump()
                    ticker_signals[agent] = {
                        "signal": signal_dict.get("signal", "neutral"),
                        "confidence": signal_dict.get("confidence", 0.0)
                    }
                elif isinstance(signal_obj, dict):  # It's already a dictionary
                    ticker_signals[agent] = {
                        "signal": signal_obj.get("signal", "neutral"),
                        "confidence": signal_obj.get("confidence", 0.0)
                    }
                else:  # It's a Pydantic model but doesn't have model_dump (older version)
                    ticker_signals[agent] = {
                        "signal": getattr(signal_obj, "signal", "neutral"),
                        "confidence": getattr(signal_obj, "confidence", 0.0)
                    }
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_management_agent", None, "Making trading decisions")

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        risk_dashboard=risk_dashboard,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Management Agent")

    # Execute trades with Alpaca if live trading is enabled
    if LIVE_TRADING_ENABLED:
        progress.update_status("portfolio_management_agent", None, "Executing live trades on Alpaca")
        
        # Check if we've hit a circuit breaker
        from utils.risk_manager import TODAY_STATE
        if TODAY_STATE.get("circuit_breaker_triggered", False):
            progress.update_status("portfolio_management_agent", None, "CIRCUIT BREAKER ACTIVE: Trading suspended")
            logger.warning("CIRCUIT BREAKER ACTIVE: All trading is suspended for today.")
        else:
            # Execute trades if not in circuit breaker mode
            for ticker, decision in result.decisions.items():
                if decision.action != "hold" and decision.quantity > 0:
                    # First perform time-based risk check for earnings and market hours
                    risk_adjustment_needed = False
                    risk_reasons = []
                    
                    # Check market hours for high risk periods
                    market_hours = check_market_hours()
                    if market_hours["high_risk_period"]:
                        risk_adjustment_needed = True
                        risk_reasons.append(f"Market timing risk: {market_hours['reason']}")
                    
                    # Check for upcoming earnings announcements 
                    try:
                        next_earnings = get_next_earnings_date(ticker)
                        if next_earnings and next_earnings.get("days_until_earnings", 100) <= 5:
                            risk_adjustment_needed = True
                            risk_reasons.append(f"Earnings announcement in {next_earnings['days_until_earnings']} days")
                    except Exception as e:
                        logger.warning(f"Error checking earnings for {ticker}: {e}")
                    
                    # Apply time-based risk adjustments if needed
                    original_quantity = decision.quantity
                    if risk_adjustment_needed:
                        # Reduce position size by 40% for high risk periods
                        adjusted_quantity = max(1, int(decision.quantity * 0.6))
                        decision.quantity = adjusted_quantity
                        logger.warning(f"{ticker}: Time-based risk factors: {', '.join(risk_reasons)}. Reducing order from {original_quantity} to {adjusted_quantity} shares")
                    
                    # Apply sector and correlation adjustments
                    progress.update_status("portfolio_management_agent", ticker, "Checking sector and correlation risk")
                    adjusted_quantity, adjustment_reasons = apply_sector_correlation_adjustments(
                        ticker, decision.action, decision.quantity, portfolio
                    )
                    
                    if adjusted_quantity != decision.quantity:
                        original_quantity = decision.quantity
                        decision.quantity = adjusted_quantity
                        logger.warning(f"{ticker}: {', '.join(adjustment_reasons)}. Reducing order from {original_quantity} to {adjusted_quantity} shares")
                    
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
    """Attempts to get a decision from the LLM with retry logic"""
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

              Inputs:
              - signals_by_ticker: dictionary of ticker → signals
              - max_shares: maximum shares allowed per ticker
              - portfolio_cash: current cash in portfolio
              - portfolio_positions: current positions (both long and short)
              - current_prices: current prices for each ticker
              - margin_requirement: current margin requirement for short positions
              - risk_dashboard: comprehensive risk dashboard for the portfolio
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
        }
    )

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error in portfolio management, defaulting to hold") for ticker in tickers})

    return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=PortfolioManagerOutput, agent_name="portfolio_management_agent", default_factory=create_default_portfolio_output)
