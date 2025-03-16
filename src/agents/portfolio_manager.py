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
        return TradingClient(api_key, api_secret, paper=True)
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return None


def execute_alpaca_trade(ticker, action, quantity, current_price):
    """Execute a trade using Alpaca API with risk management controls"""
    if quantity <= 0:
        logger.info(f"Skipping {action} order for {ticker}: quantity must be > 0")
        return False
    
    # Apply risk management checks
    if not can_execute_trade(ticker, action, quantity, current_price):
        logger.warning(f"Risk management rejected {action} order for {ticker}")
        return False
    
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
        
        # Calculate stop loss and take profit prices
        stop_loss_price = None
        take_profit_price = None
        
        if action == "buy":
            # Calculate stop loss (5% below purchase price by default)
            stop_loss_price = current_price * (1 - RISK_PARAMS["STOP_LOSS_PCT"])
            # Calculate take profit (20% above purchase price by default)
            take_profit_price = current_price * (1 + RISK_PARAMS["TAKE_PROFIT_PCT"])
            
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
            # For short, the stop loss is above the entry price
            stop_loss_price = current_price * (1 + RISK_PARAMS["STOP_LOSS_PCT"])
            # For short, the take profit is below the entry price
            take_profit_price = current_price * (1 - RISK_PARAMS["TAKE_PROFIT_PCT"])
            
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


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

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
                reset_daily_state(portfolio_value)
            else:
                logger.warning("Could not get portfolio state from Alpaca, using existing portfolio data")
        else:
            logger.warning("Alpaca client not available, using existing portfolio data")

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)
        
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
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_management_agent", None, "Making trading decisions")

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
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
                    progress.update_status(
                        "portfolio_management_agent", 
                        ticker, 
                        f"Executing {decision.action} order for {decision.quantity} shares"
                    )
                    
                    success = execute_alpaca_trade(ticker, decision.action, decision.quantity, current_prices[ticker])
                    
                    if success:
                        logger.info(f"Successfully executed {decision.action} order for {decision.quantity} shares of {ticker}")
                    else:
                        logger.warning(f"Failed to execute {decision.action} order for {decision.quantity} shares of {ticker}")
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
        }
    )

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error in portfolio management, defaulting to hold") for ticker in tickers})

    return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=PortfolioManagerOutput, agent_name="portfolio_management_agent", default_factory=create_default_portfolio_output)
