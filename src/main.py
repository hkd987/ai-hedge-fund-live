import sys
import os
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Back, Style, init
import questionary
from agents.ben_graham import ben_graham_agent
from agents.bill_ackman import bill_ackman_agent
from agents.fundamentals import fundamentals_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.warren_buffett import warren_buffett_agent
from graph.state import AgentState
from agents.valuation import valuation_agent
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
from utils.market_data import get_current_vix, get_market_status, ALPACA_AVAILABLE as MARKET_DATA_AVAILABLE

import argparse
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
from utils.visualize import save_graph_as_png
import json
import logging

# For Alpaca holdings integration
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.trading.requests import MarketOrderRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')

# Check for live trading environment variable
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Display a prominent warning if live trading is enabled
if LIVE_TRADING_ENABLED:
    warning_message = (
        f"\n{'!'*80}\n"
        f"{'!'*20} WARNING: LIVE TRADING IS ENABLED {'!'*20}\n"
        f"{'!'*20} REAL MONEY WILL BE USED FOR TRADES {'!'*20}\n"
        f"{'!'*80}\n"
    )
    logger.warning(warning_message)
    print(f"{Fore.RED}{Style.BRIGHT}{warning_message}{Style.RESET_ALL}")


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


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
        # Use paper trading unless LIVE_TRADING is explicitly set to true
        is_paper = not LIVE_TRADING_ENABLED
        client = TradingClient(api_key, api_secret, paper=is_paper)
        
        # Log which environment we're using
        env_type = "paper trading" if is_paper else "live trading"
        logger.info(f"Initialized Alpaca client for {env_type}")
        
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca client: {e}")
        return None


def get_alpaca_holdings():
    """Fetch all current holdings from Alpaca account"""
    if not ALPACA_AVAILABLE:
        logger.warning("Alpaca SDK not installed. Run 'pip install alpaca-py' to include Alpaca holdings.")
        return []

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Alpaca API credentials not found in environment variables.")
        return []
    
    try:
        # Initialize Alpaca client with the appropriate environment
        is_paper = not LIVE_TRADING_ENABLED
        client = TradingClient(api_key, api_secret, paper=is_paper)
        
        # Get all positions
        positions = client.get_all_positions()
        
        # Extract tickers from positions
        holdings = [position.symbol for position in positions]
        
        # Log which environment we're using
        env_type = "paper trading" if is_paper else "live trading"
        if holdings:
            logger.info(f"Found {len(holdings)} holdings in Alpaca account ({env_type}): {', '.join(holdings)}")
        else:
            logger.info(f"No holdings found in Alpaca account ({env_type})")
            
        return holdings
        
    except Exception as e:
        logger.error(f"Failed to get holdings from Alpaca: {e}")
        return []


def initialize_portfolio_from_alpaca(tickers):
    """
    Initialize the portfolio with data from Alpaca in live trading mode.
    This ensures we start with actual cash balances and margin requirements.
    """
    if not LIVE_TRADING_ENABLED or not ALPACA_AVAILABLE:
        logger.info("Not using Alpaca for portfolio initialization (either live trading is disabled or Alpaca SDK not available)")
        return None
    
    client = get_alpaca_client()
    if not client:
        logger.warning("Could not initialize Alpaca client for portfolio initialization")
        return None
    
    try:
        # Get account information
        account = client.get_account()
        
        # Initialize portfolio structure
        portfolio = {
            "cash": float(account.cash),
            "positions": {},
            "margin_requirement": 0.0,  # We'll calculate this based on positions
            "realized_gains": {}
        }
        
        logger.info(f"Initialized portfolio with ${portfolio['cash']:.2f} cash from Alpaca account")
        
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
            portfolio["realized_gains"][ticker] = {
                "long": 0.0,
                "short": 0.0,
            }
            
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
                    logger.info(f"Found long position for {ticker}: {qty} shares at ${cost_basis / qty:.2f} cost basis")
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
                    logger.info(f"Found short position for {ticker}: {short_qty} shares at ${cost_basis / short_qty:.2f} cost basis")
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
        
        # Initialize risk management with portfolio value
        from utils.risk_manager import reset_daily_state
        reset_daily_state(portfolio_value)
        
        logger.info(f"Successfully initialized portfolio from Alpaca with total value: ${portfolio_value:.2f}")
        logger.info(f"Using margin requirement: ${portfolio['margin_requirement']:.2f}")
        
        return portfolio
    
    except Exception as e:
        logger.error(f"Failed to initialize portfolio from Alpaca: {e}")
        return None


def emergency_liquidate_all_positions():
    """
    Emergency function to liquidate all positions in the portfolio.
    This is the "panic button" to be used in emergency situations.
    """
    if not LIVE_TRADING_ENABLED:
        logger.warning("Live trading is not enabled. Cannot liquidate positions.")
        return False
    
    client = get_alpaca_client()
    if not client:
        logger.error("Alpaca client not available. Cannot liquidate positions.")
        return False
    
    try:
        # Get account info to determine environment
        account = client.get_account()
        is_paper = bool(getattr(account, 'is_paper', True))  # Default to True if attribute doesn't exist
        env_type = "paper trading" if is_paper else "LIVE TRADING"
        
        # Get all positions
        positions = client.get_all_positions()
        
        if not positions:
            logger.info(f"No positions to liquidate in {env_type} account.")
            return True
        
        logger.warning(f"EMERGENCY LIQUIDATION: Attempting to liquidate {len(positions)} positions in {env_type} account")
        
        # Liquidate each position
        for position in positions:
            ticker = position.symbol
            qty = abs(int(position.qty))  # Absolute quantity
            
            if qty <= 0:
                continue
                
            # Determine if this is a long or short position
            if int(position.qty) > 0:  # Long position
                action = "sell"
                side = OrderSide.SELL
            else:  # Short position
                action = "cover"
                side = OrderSide.BUY
            
            # Create market order to liquidate
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit the order
            order = client.submit_order(order_data)
            logger.warning(f"EMERGENCY: Submitted {action} order for {qty} shares of {ticker} in {env_type} account: Order ID {order.id}")
        
        logger.warning(f"EMERGENCY LIQUIDATION COMPLETE: All positions have been liquidated in {env_type} account")
        return True
        
    except Exception as e:
        logger.error(f"Failed to liquidate positions: {e}")
        return False


def check_market_conditions():
    """
    Check current market conditions and return relevant data for risk assessment.
    """
    conditions = {
        "vix": None,
        "market_open": None,
        "high_volatility": False
    }
    
    # Skip if Alpaca market data is not available
    if not MARKET_DATA_AVAILABLE:
        logger.warning("Alpaca market data is not available. Skipping market condition checks.")
        return conditions
    
    # Get VIX data
    vix_value = get_current_vix()
    
    # Get market status
    market_status = get_market_status()
    
    conditions = {
        "vix": vix_value,
        "market_open": market_status.get("is_open"),
        "high_volatility": vix_value > float(os.getenv("HIGH_VOLATILITY_VIX_THRESHOLD", 25)) if vix_value else False
    }
    
    # Log market conditions
    logger.info(f"Market conditions from Alpaca: VIX={vix_value}, Market Open={market_status.get('is_open')}")
    
    if conditions["high_volatility"]:
        logger.warning(f"HIGH VOLATILITY DETECTED: VIX at {vix_value}")
    
    return conditions


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


# Create the default workflow and compile it
def create_default_workflow():
    """Create the default workflow with all analysts."""
    return create_workflow()

# Compile the default workflow
app = create_default_workflow().compile()


def parse_args():
    parser = argparse.ArgumentParser(description="AI Hedge Fund")
    parser.add_argument(
        "--ticker",
        type=str,
        help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show reasoning behind the decisions",
    )
    parser.add_argument(
        "--selected-analysts",
        type=str,
        help="Comma-separated list of analysts to use (e.g. warren_buffett,bill_ackman)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name to use (e.g., gpt-4o, claude-3-5-sonnet)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="OpenAI",
        help="Model provider (e.g., OpenAI, ANTHROPIC, GROQ)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (default: 3 months ago)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--emergency-liquidate",
        action="store_true",
        help="Emergency liquidate all positions",
    )
    parser.add_argument(
        "--include-alpaca-holdings",
        action="store_true",
        help="Include current holdings from Alpaca account",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash in the portfolio (default: 100000.0)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement (default: 0.0)",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save the agent workflow graph as a PNG file",
    )
    parser.add_argument(
        "--schedule",
        "-s",
        action="store_true",
        help="Run the hedge fund every 60 minutes automatically",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval in minutes between scheduled runs (default: 60)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=20,
        help="Interval in minutes to check portfolio between full runs",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading with Alpaca",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=3,
        help="Maximum number of concurrent positions",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run the API server",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual trades",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run with paper trading",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode",
    )
    return parser.parse_args()


def update_portfolio_status(alpaca_client, portfolio):
    """
    Update portfolio status between full analysis runs.
    Gets latest position information and prices from Alpaca.
    
    Args:
        alpaca_client: Initialized Alpaca client
        portfolio: Current portfolio state
        
    Returns:
        tuple: (alpaca_client, updated_portfolio)
    """
    if not alpaca_client:
        logging.warning("No Alpaca client available. Cannot update portfolio status.")
        return alpaca_client, portfolio
    
    try:
        # Get account information for cash balance
        account = alpaca_client.get_account()
        portfolio["cash"] = float(account.cash)
        
        # Get all current positions
        positions = alpaca_client.get_all_positions()
        current_positions = {position.symbol: position for position in positions}
        
        # Update position data for each ticker in the portfolio
        for ticker in portfolio["positions"]:
            if ticker in current_positions:
                position = current_positions[ticker]
                qty = int(position.qty)
                
                # Update position details based on current data
                if qty > 0:  # Long position
                    portfolio["positions"][ticker]["long"] = qty
                    portfolio["positions"][ticker]["long_cost_basis"] = float(position.cost_basis) / qty
                    portfolio["positions"][ticker]["short"] = 0
                    portfolio["positions"][ticker]["short_cost_basis"] = 0.0
                elif qty < 0:  # Short position
                    short_qty = abs(qty)
                    portfolio["positions"][ticker]["short"] = short_qty
                    portfolio["positions"][ticker]["short_cost_basis"] = float(position.cost_basis) / short_qty
                    portfolio["positions"][ticker]["long"] = 0
                    portfolio["positions"][ticker]["long_cost_basis"] = 0.0
                    # Update margin used
                    portfolio["positions"][ticker]["short_margin_used"] = float(position.market_value) * 0.5
            else:
                # No current position for this ticker
                portfolio["positions"][ticker]["long"] = 0
                portfolio["positions"][ticker]["short"] = 0
        
        # Calculate total portfolio value and update
        portfolio_value = float(account.equity)
        portfolio["portfolio_value"] = portfolio_value
        
        # Calculate current prices for all tickers in the portfolio
        current_prices = {}
        for ticker in portfolio["positions"]:
            # Try to get current price from existing position
            if ticker in current_positions:
                position = current_positions[ticker]
                qty = abs(int(position.qty))
                if qty > 0:
                    current_prices[ticker] = float(position.market_value) / qty
        
        # Store current prices in the portfolio
        portfolio["current_prices"] = current_prices
        
        logging.info(f"Portfolio updated: ${portfolio_value:.2f} total value, ${portfolio['cash']:.2f} cash")
        return alpaca_client, portfolio
        
    except Exception as e:
        logging.error(f"Error updating portfolio status: {e}")
        return alpaca_client, portfolio


def init_portfolio():
    """
    Initialize portfolio and Alpaca client for live trading.
    
    Returns:
        tuple: (alpaca_client, portfolio)
    """
    alpaca_client = get_alpaca_client()
    portfolio = {}
    
    if alpaca_client:
        try:
            # Get account info
            account = alpaca_client.get_account()
            
            # Initialize basic portfolio structure
            portfolio = {
                "cash": float(account.cash),
                "portfolio_value": float(account.equity),
                "positions": {},
                "realized_gains": {},
                "current_prices": {}
            }
            
            # Get all positions
            positions = alpaca_client.get_all_positions()
            
            # Initialize empty positions and realized gains for all positions
            for position in positions:
                ticker = position.symbol
                portfolio["positions"][ticker] = {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0
                }
                portfolio["realized_gains"][ticker] = {
                    "long": 0.0,
                    "short": 0.0
                }
            
            # Update with actual position data
            alpaca_client, portfolio = update_portfolio_status(alpaca_client, portfolio)
            
        except Exception as e:
            logging.error(f"Error initializing portfolio: {e}")
            portfolio = {
                "cash": 10000.0,  # Default cash value
                "portfolio_value": 10000.0,
                "positions": {},
                "realized_gains": {},
                "current_prices": {}
            }
    else:
        # No Alpaca client, use default portfolio
        portfolio = {
            "cash": 10000.0,
            "portfolio_value": 10000.0,
            "positions": {},
            "realized_gains": {},
            "current_prices": {}
        }
    
    return alpaca_client, portfolio


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle emergency liquidation first if requested
    if args.emergency_liquidate:
        confirmed = questionary.confirm("Are you sure you want to EMERGENCY LIQUIDATE ALL POSITIONS?").ask()
        if confirmed:
            success = emergency_liquidate_all_positions()
            if success:
                print(f"{Fore.GREEN}Emergency liquidation completed.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Emergency liquidation failed.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Emergency liquidation cancelled.{Style.RESET_ALL}")
        return

    if not args.ticker:
        print(f"{Fore.RED}No ticker provided. Use --ticker SYMBOL,SYMBOL2,... to specify tickers{Style.RESET_ALL}")
        return

    tickers = args.ticker.split(',')
    
    # Select analysts if not specified in command line
    selected_analysts = []
    if args.selected_analysts:
        selected_analysts = args.selected_analysts.split(',')
    else:
        print(f"{Fore.CYAN}Select analysts to include in the analysis (Space to select, Enter to confirm):{Style.RESET_ALL}")
        choices = questionary.checkbox(
            "Select analysts:",
            choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
            style=questionary.Style([
                ("selected", "bg:blue fg:white"),
                ("checkbox", "bg:blue fg:white"),
            ]),
        ).ask()
        
        if choices:
            selected_analysts = choices
            print(f"{Fore.GREEN}Selected analysts: {', '.join(selected_analysts)}{Style.RESET_ALL}")
        else:
            # If no analysts selected, use all
            selected_analysts = [value for _, value in ANALYST_ORDER]
            print(f"{Fore.YELLOW}No analysts selected, using all analysts.{Style.RESET_ALL}")
    
    # Select model if not specified
    model_name = args.model
    model_provider = args.provider
    
    if args.api:
        # ... existing API server code ...
        return
    
    # Initialize portfolio for live trading
    if args.live:
        alpaca, portfolio = init_portfolio()
    else:
        portfolio = {}
    
    # Set default start and end dates if not provided
    end_date = args.end_date
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    start_date = args.start_date
    if not start_date:
        # Default to 3 months before end date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - timedelta(days=90)).strftime("%Y-%m-%d")
    
    # If schedule flag is set, run the hedge fund at regular intervals
    if args.schedule:
        logging.info(f"Starting scheduled hedge fund runs every {args.interval} minutes")
        logging.info(f"Portfolio checks every {args.check_interval} minutes")
        
        while True:
            current_time = datetime.now()
            logging.info(f"Running hedge fund analysis at {current_time}")
            
            # Update end date to current time for each run in scheduled mode
            end_date = current_time.strftime("%Y-%m-%d")
            
            # Run the hedge fund with current settings
            result = run_hedge_fund(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                portfolio=portfolio,
                show_reasoning=args.show_reasoning,
                selected_analysts=selected_analysts,
                model_name=model_name,
                model_provider=model_provider,
            )
            
            # Print the trading output
            print_trading_output(result)
            
            # Wait for the specified interval
            logging.info(f"Next full analysis scheduled for {current_time + timedelta(minutes=args.interval)}")
            
            # Set up intermediate portfolio checks
            check_count = args.interval // args.check_interval
            
            for i in range(check_count):
                # Sleep until next check
                time.sleep(args.check_interval * 60)
                
                if args.live and not args.dry_run:
                    check_time = datetime.now()
                    logging.info(f"Performing portfolio check at {check_time}")
                    
                    # Update portfolio status and check for any necessary adjustments
                    # This is a lightweight check compared to the full analysis
                    try:
                        alpaca, portfolio = update_portfolio_status(alpaca, portfolio)
                        # Optionally perform a quick analysis to see if any positions need adjustment
                    except Exception as e:
                        logging.error(f"Error during portfolio check: {e}")
            
            # If we didn't use all the time with checks, sleep for the remainder
            remaining_time = args.interval - (check_count * args.check_interval)
            if remaining_time > 0:
                time.sleep(remaining_time * 60)
    else:
        # Run once (original behavior)
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=args.show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_name,
            model_provider=model_provider,
        )
        
        # Print the trading output
        print_trading_output(result)


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            # Use the default compiled app
            agent = app
            
        # Check market conditions if in live trading mode
        if LIVE_TRADING_ENABLED:
            market_conditions = check_market_conditions()
            
            # Add market conditions to the state data
            portfolio["market_conditions"] = market_conditions

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


if __name__ == "__main__":
    main()
