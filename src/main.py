import sys
import os
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
import copy

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

# Import our new run cache module
from utils.run_cache import should_use_cached_data, clear_cache, save_run_history

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
    """
    Parses a JSON string and returns a dictionary with a consistent structure.
    Ensures 'decisions' is properly formatted as a ticker-to-decision mapping.
    """
    logger = logging.getLogger('hedge_fund_parser')
    
    try:
        # Parse the JSON string
        parsed_data = json.loads(response)
        
        # Log the parsed data structure for debugging
        logger.info(f"Parsed data type: {type(parsed_data)}")
        if isinstance(parsed_data, dict):
            logger.info(f"Parsed data keys: {list(parsed_data.keys())}")
        
        # Case 1: Already properly structured with 'decisions' key
        if isinstance(parsed_data, dict) and 'decisions' in parsed_data:
            decisions = parsed_data['decisions']
            
            # Check if decisions is a proper ticker-to-decision mapping
            # or if it's a flattened decision object
            if isinstance(decisions, dict):
                # Check if it looks like a decision object instead of ticker mapping
                if all(key in ["action", "quantity", "confidence", "reasoning"] for key in decisions.keys()):
                    # It's a single decision without ticker - wrap it with a default ticker
                    ticker = parsed_data.get("ticker", "UNKNOWN")
                    parsed_data['decisions'] = {ticker: decisions}
            return parsed_data
            
        # Case 2: Response is a dict that looks like a decision object
        elif isinstance(parsed_data, dict) and any(key in ["action", "quantity", "confidence"] for key in parsed_data.keys()):
            # It's a direct decision object, wrap it in the proper structure
            ticker = "UNKNOWN"  # Default ticker if none is available
            if 'ticker' in parsed_data:
                ticker = parsed_data.pop('ticker')
            return {'decisions': {ticker: parsed_data}}
            
        # Case 3: Response is already a ticker-to-decision mapping
        elif isinstance(parsed_data, dict) and all(isinstance(v, dict) for v in parsed_data.values()):
            # Wrap it in a decisions dictionary
            return {'decisions': parsed_data}
            
        # Case 4: Any other structure - wrap in decisions
        else:
            logger.warning(f"Response structure not recognized, using generic wrapper: {type(parsed_data)}")
            return {'decisions': parsed_data}
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return {'decisions': {}}
    except TypeError as e:
        logger.error(f"Type error during parsing: {e}\nResponse: {repr(response)}")
        return {'decisions': {}}
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {e}\nResponse: {repr(response)}")
        return {'decisions': {}}


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
    Initialize portfolio from Alpaca for live trading.
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Portfolio dictionary with Alpaca data
    """
    alpaca_client = get_alpaca_client()
    
    if not alpaca_client:
        # Create default portfolio
        portfolio = {
            "cash": 100000.0,
            "portfolio_value": 100000.0,
            "positions": {},
            "realized_gains": 0.0,
            "current_prices": {},
            "open_orders": {}  # Initialize empty open orders dict
        }
        return portfolio
        
    try:
        # Get account information
        account = alpaca_client.get_account()
        
        # Create initial portfolio structure
        portfolio = {
            "cash": float(account.cash),
            "portfolio_value": float(account.equity),
            "positions": {},
            "realized_gains": 0.0,
            "current_prices": {},
            "open_orders": {}  # Initialize empty open orders dict
        }
        
        # Get all positions and add to portfolio
        try:
            positions = alpaca_client.get_all_positions()
            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                avg_entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                
                portfolio["positions"][symbol] = {
                    "quantity": qty,
                    "avg_price": avg_entry_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl
                }
                
                # Also add to current prices
                portfolio["current_prices"][symbol] = current_price
        except Exception as e:
            logging.error(f"Error getting positions from Alpaca: {e}")
            
        # Get open orders and add to portfolio
        from agents.portfolio_manager import get_alpaca_open_orders
        try:
            open_orders = get_alpaca_open_orders(alpaca_client)
            portfolio["open_orders"] = open_orders
            if open_orders:
                ticker_list = list(open_orders.keys())
                order_count = sum(len(orders) for orders in open_orders.values())
                logging.info(f"Added {order_count} open orders for {len(ticker_list)} tickers to initial portfolio")
        except Exception as e:
            logging.error(f"Error getting open orders for portfolio initialization: {e}")
            portfolio["open_orders"] = {}
            
        logging.info(f"Initialized portfolio from Alpaca: ${portfolio['portfolio_value']:,.2f} total value, ${portfolio['cash']:,.2f} cash")
        return portfolio
        
    except Exception as e:
        logging.error(f"Error initializing portfolio from Alpaca: {e}")
        # Fall back to default portfolio
        portfolio = {
            "cash": 100000.0,
            "portfolio_value": 100000.0,
            "positions": {},
            "realized_gains": 0.0,
            "current_prices": {},
            "open_orders": {}  # Initialize empty open orders dict
        }
        return portfolio


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
        "--custom",
        action="store_true",
        help="Use custom analyst selection without prompting",
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
        "--live",
        action="store_true",
        help="Use live trading mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute actual trades in live mode",
    )
    parser.add_argument(
        "--emergency-liquidate",
        action="store_true",
        help="Emergency liquidate all positions (live mode only)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the agent workflow as a graph",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging",
    )
    parser.add_argument(
        "--schedule", 
        action="store_true",
        help="Run the hedge fund at regular intervals"
    )
    parser.add_argument(
        "--interval", 
        type=int,
        default=60,
        help="Minutes between full analysis runs when using --schedule"
    )
    parser.add_argument(
        "--check-interval", 
        type=int,
        default=15,
        help="Minutes between portfolio checks when using --schedule"
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Force a fresh run (clear cache before running)",
    )
    
    return parser.parse_args()


def update_portfolio_status(alpaca_client, portfolio):
    """
    Update the portfolio status with the latest data from Alpaca
    
    Args:
        alpaca_client: Alpaca client
        portfolio: Portfolio dictionary
        
    Returns:
        Updated portfolio dictionary
    """
    if not alpaca_client:
        logging.warning("No Alpaca client available for updating portfolio status")
        return portfolio
        
    try:
        # Get account data
        account = alpaca_client.get_account()
        portfolio["portfolio_value"] = float(account.equity)
        portfolio["cash"] = float(account.cash)
        
        # Get current positions
        try:
            positions = alpaca_client.get_all_positions()
            if positions:
                for position in positions:
                    symbol = position.symbol
                    qty = float(position.qty)
                    avg_entry_price = float(position.avg_entry_price)
                    current_price = float(position.current_price)
                    market_value = float(position.market_value)
                    unrealized_pl = float(position.unrealized_pl)
                    
                    if symbol not in portfolio["positions"]:
                        portfolio["positions"][symbol] = {
                            "quantity": 0,
                            "avg_price": 0,
                            "current_price": 0,
                            "market_value": 0,
                            "unrealized_pl": 0
                        }
                    
                    # Update position data
                    portfolio["positions"][symbol]["quantity"] = qty
                    portfolio["positions"][symbol]["avg_price"] = avg_entry_price
                    portfolio["positions"][symbol]["current_price"] = current_price
                    portfolio["positions"][symbol]["market_value"] = market_value
                    portfolio["positions"][symbol]["unrealized_pl"] = unrealized_pl
                    
                    # Also update current price in the global price dict
                    portfolio["current_prices"][symbol] = current_price
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            
        # Get all open orders and include them in the portfolio
        logging.info("Attempting to retrieve open orders in update_portfolio_status")
        
        # Run diagnostics first to provide complete information
        from agents.portfolio_manager import diagnose_alpaca_orders
        diagnostics = diagnose_alpaca_orders(alpaca_client)
        portfolio["_order_diagnostics"] = diagnostics
        
        # Then get the actual open orders using the fixed function
        from agents.portfolio_manager import get_alpaca_open_orders
        try:
            open_orders = get_alpaca_open_orders(alpaca_client)
            portfolio["open_orders"] = open_orders
            if open_orders:
                ticker_list = list(open_orders.keys())
                order_count = sum(len(orders) for orders in open_orders.values())
                logging.info(f"Added {order_count} open orders for {len(ticker_list)} tickers to portfolio")
            else:
                logging.info("No open orders returned from Alpaca (empty dictionary)")
        except Exception as e:
            logging.error(f"Error getting open orders: {e}")
            portfolio["open_orders"] = {}
            
        # Calculate total position value
        position_value = sum(pos["market_value"] for pos in portfolio["positions"].values())
        logging.info(f"Portfolio updated: ${portfolio['portfolio_value']:.2f} total value, ${portfolio['cash']:.2f} cash")
        
        return portfolio
        
    except Exception as e:
        logging.error(f"Error updating portfolio: {e}")
        return portfolio


def init_portfolio(args):
    """
    Initialize the portfolio with cash or restore from saved state.
    
    Args:
        args: Command line arguments
        
    Returns:
        Portfolio dictionary with initial state
    """
    # Check if we have Alpaca credentials for live trading
    alpaca_client = get_alpaca_client()
    
    # Initialize portfolio
    if alpaca_client:
        try:
            # Get account information
            account = alpaca_client.get_account()
            
            # Create initial portfolio structure
            portfolio = {
                "cash": float(account.cash),
                "portfolio_value": float(account.equity),
                "positions": {},
                "realized_gains": 0.0,
                "current_prices": {},
                "open_orders": {}  # Initialize empty open orders dict
            }
            
            # Get all positions and add to portfolio
            positions = alpaca_client.get_all_positions()
            for position in positions:
                symbol = position.symbol
                qty = float(position.qty)
                avg_entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                
                portfolio["positions"][symbol] = {
                    "quantity": qty,
                    "avg_price": avg_entry_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl
                }
                
                # Also add to current prices
                portfolio["current_prices"][symbol] = current_price
                
            # Get open orders and add to portfolio
            from agents.portfolio_manager import get_alpaca_open_orders
            try:
                open_orders = get_alpaca_open_orders(alpaca_client)
                portfolio["open_orders"] = open_orders
                if open_orders:
                    ticker_list = list(open_orders.keys())
                    order_count = sum(len(orders) for orders in open_orders.values())
                    logging.info(f"Added {order_count} open orders for {len(ticker_list)} tickers to initial portfolio")
            except Exception as e:
                logging.error(f"Error getting open orders for portfolio initialization: {e}")
                portfolio["open_orders"] = {}
                
            logging.info(f"Initialized portfolio from Alpaca: ${portfolio['portfolio_value']:,.2f} total value, ${portfolio['cash']:,.2f} cash")
            return portfolio
            
        except Exception as e:
            logging.error(f"Error initializing portfolio from Alpaca: {e}")
            # Fall back to default portfolio
            
    # Default portfolio if Alpaca is not available
    # Use default values - don't try to access args.initial_capital which may not exist
    default_capital = 100000.0
    portfolio = {
        "cash": default_capital,
        "portfolio_value": default_capital,
        "positions": {},
        "realized_gains": 0.0,
        "current_prices": {},
        "open_orders": {}  # Initialize empty open orders dict
    }
    
    logging.info(f"Initialized default portfolio: ${portfolio['portfolio_value']:,.2f}")
    return portfolio


# Initialize a global variable to track shutdown status
is_shutting_down = False

# Set up signal handlers for graceful shutdown
def handle_shutdown_signal(signum, frame):
    """
    Handle shutdown signals (SIGINT, SIGTERM) gracefully.
    This allows for a clean exit when Ctrl+C is pressed or the process is terminated.
    """
    global is_shutting_down
    
    # Mark as shutting down
    is_shutting_down = True
    
    # Clear line and display message
    print(f"\n{Fore.CYAN}Intercepted shutdown signal. Cleaning up...{Style.RESET_ALL}")
    
    # Make sure progress tracker is stopped
    progress.stop()
    
    # Print shutdown message
    print(f"\n{Fore.GREEN}AI Hedge Fund application gracefully shut down.{Style.RESET_ALL}")
    
    # Exit with success code
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_shutdown_signal)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_shutdown_signal) # Termination signal


def main():
    """Main entry point for the application."""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    args = parse_args()
    
    # Check if user wants to use fresh data (ignore cache)
    if args.fresh_run:
        print("Fresh run requested - clearing cache")
        clear_cache()
    
    # Load environment variables
    load_dotenv()
    
    # Get the current date as the default end date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Set default start and end dates
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = today
    
    # Check for emergency liquidation flag
    if args.emergency_liquidate:
        emergency_liquidate_all_positions()
        return
        
    # Get the ticker(s) from command line arguments
    tickers = []
    if args.ticker:
        # Split multiple tickers if provided as comma-separated list
        if "," in args.ticker:
            tickers = [ticker.strip() for ticker in args.ticker.split(",")]
        else:
            tickers = [args.ticker]
    
    # If no tickers provided, ask for ticker input
    if not tickers:
        ticker_input = questionary.text("Enter ticker symbol(s) (comma-separated for multiple):").ask()
        if ticker_input:
            tickers = [ticker.strip() for ticker in ticker_input.split(",")]
        else:
            print("No ticker provided. Exiting...")
            return
            
    # Convert all tickers to uppercase
    tickers = [ticker.upper() for ticker in tickers]
    
    # Ask if the user wants to select specific analysts
    custom_analysts = args.selected_analysts
    selected_analysts = []
    
    # Create mapping dictionaries for analyst names
    display_name_to_key = {display_name: key for display_name, key in ANALYST_ORDER}
    key_to_display_name = {key: display_name for display_name, key in ANALYST_ORDER}
    
    if custom_analysts:
        # Handle comma-separated analyst names
        if isinstance(custom_analysts, str) and "," in custom_analysts:
            analyst_names = [name.strip() for name in custom_analysts.split(",")]
        else:
            analyst_names = [custom_analysts]
        
        # Convert display names to agent keys
        for name in analyst_names:
            # Check if the name is a display name (like "Warren Buffett")
            if name in display_name_to_key:
                selected_analysts.append(display_name_to_key[name])
            # Check if the name is a key (like "warren_buffett")
            elif name in key_to_display_name:
                selected_analysts.append(name)
            else:
                print(f"Warning: Analyst '{name}' not recognized. Skipping.")
        
        if not selected_analysts:
            print("No valid analysts selected. Using all analysts.")
    else:
        # Ask if user wants to select specific analysts
        use_custom = args.custom or questionary.confirm("Do you want to select specific analysts?").ask()
        
        if use_custom:
            # Create checkbox options for analysts
            analyst_options = [display_name for display_name, _ in ANALYST_ORDER]
            selected_options = questionary.checkbox("Select analysts to use:", choices=analyst_options).ask()
            
            # Map selected display names to agent keys
            for display_name in selected_options:
                key = display_name_to_key.get(display_name)
                if key:
                    selected_analysts.append(key)
    
    # Initialize the portfolio
    portfolio = init_portfolio(args)
    
    # If live trading is enabled, get current holdings from Alpaca
    if LIVE_TRADING_ENABLED:
        print("Live trading is enabled. Checking Alpaca account...")
        
        # Get Alpaca holdings
        alpaca_holdings = get_alpaca_holdings()
        if alpaca_holdings:
            # Initialize portfolio with Alpaca data
            alpaca_portfolio, alpaca_client = initialize_portfolio_from_alpaca(tickers)
            if alpaca_portfolio:
                portfolio = alpaca_portfolio
                print(f"Initialized portfolio from Alpaca with ${portfolio.get('cash', 0):.2f} cash")
            else:
                print("Could not initialize portfolio from Alpaca. Using default portfolio.")
    
    # Get the model name and provider from arguments or environment
    model_name = args.model or os.getenv("DEFAULT_MODEL", "gpt-4o")
    model_provider = args.provider or os.getenv("DEFAULT_PROVIDER", "OpenAI")
    
    # Print configuration information
    print(f"\nRunning hedge fund with the following configuration:")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Model: {model_name} ({model_provider})")
    if selected_analysts:
        print(f"  Selected analysts: {', '.join(selected_analysts)}")
    else:
        print(f"  Using all analysts")
    print(f"  Start date: {start_date}")
    print(f"  End date: {end_date}")
    print(f"  Portfolio cash: ${portfolio.get('cash', 0):.2f}")
    if args.schedule:
        print(f"  Running in scheduled mode with {args.interval} minute intervals")
        print(f"  Portfolio checks every {args.check_interval} minutes")
    print()
    
    # Check market conditions
    check_market_conditions()
    
    # Check if we should run in scheduled mode
    if args.schedule:
        # Get intervals in seconds
        run_interval_seconds = args.interval * 60
        check_interval_seconds = args.check_interval * 60
        
        # Print start message
        print(f"\n{Fore.CYAN}Starting scheduled runs. Press Ctrl+C to exit.{Style.RESET_ALL}")
        print(f"Full analysis runs every {args.interval} minutes")
        print(f"Portfolio checks every {args.check_interval} minutes")
        
        # Track last run times
        last_full_run = datetime.now()
        last_check = datetime.now()
        
        # Perform the initial full run before entering the waiting loop
        print(f"\n{Fore.GREEN}Performing initial full analysis at {last_full_run.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        
        # Run full analysis
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=args.show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_name,
            model_provider=model_provider,
            args=args,
        )
        
        # Check for errors
        if "error" in result:
            print(f"Error running hedge fund: {result['error']}")
        else:
            # Print the trading output
            print_trading_output(result)
            
            # Update portfolio
            if LIVE_TRADING_ENABLED:
                alpaca_client, portfolio = update_portfolio_status(alpaca_client, portfolio)
        
        # Update last full run time after initial run
        last_full_run = datetime.now()
        last_check = datetime.now()
        
        print(f"\n{Fore.CYAN}Initial run complete. Entering scheduled mode.{Style.RESET_ALL}")
        print(f"Next full run will be in {args.interval} minutes")
        
        # Run until interrupted
        try:
            while True:
                current_time = datetime.now()
                
                # Calculate time since last runs
                time_since_full_run = (current_time - last_full_run).total_seconds()
                time_since_check = (current_time - last_check).total_seconds()
                
                # Check if it's time for a full run
                if time_since_full_run >= run_interval_seconds:
                    print(f"\n{Fore.GREEN}Running full analysis at {current_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
                    
                    # Run full analysis
                    result = run_hedge_fund(
                        tickers=tickers,
                        start_date=start_date,
                        end_date=end_date,
                        portfolio=portfolio,
                        show_reasoning=args.show_reasoning,
                        selected_analysts=selected_analysts,
                        model_name=model_name,
                        model_provider=model_provider,
                        args=args,
                    )
                    
                    # Check for errors
                    if "error" in result:
                        print(f"Error running hedge fund: {result['error']}")
                    else:
                        # Print the trading output
                        print_trading_output(result)
                        
                        # Update portfolio
                        if LIVE_TRADING_ENABLED:
                            alpaca_client, portfolio = update_portfolio_status(alpaca_client, portfolio)
                    
                    # Update last full run time
                    last_full_run = datetime.now()
                    last_check = datetime.now()
                
                # Check if it's time for a portfolio check
                elif time_since_check >= check_interval_seconds:
                    print(f"\n{Fore.YELLOW}Checking portfolio at {current_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
                    
                    # Update portfolio status
                    if LIVE_TRADING_ENABLED:
                        alpaca_client, portfolio = update_portfolio_status(alpaca_client, portfolio)
                        
                        # Print current portfolio value
                        print(f"Current portfolio value: ${portfolio.get('portfolio_value', 0):.2f}")
                        print(f"Cash: ${portfolio.get('cash', 0):.2f}")
                        
                        # Print positions
                        if portfolio.get("positions"):
                            print("Current positions:")
                            for ticker, position in portfolio["positions"].items():
                                if position.get("long", 0) > 0:
                                    print(f"  {ticker}: {position['long']} shares (LONG)")
                                elif position.get("short", 0) > 0:
                                    print(f"  {ticker}: {position['short']} shares (SHORT)")
                    else:
                        print("Live trading disabled. Cannot check portfolio status.")
                    
                    # Update last check time
                    last_check = datetime.now()
                
                # Sleep to avoid high CPU usage
                time.sleep(10)
                
                # Calculate and print time until next runs
                current_time = datetime.now()
                time_until_full_run = run_interval_seconds - (current_time - last_full_run).total_seconds()
                time_until_check = check_interval_seconds - (current_time - last_check).total_seconds()
                
                # Only print status every 30 seconds to avoid excessive output
                if int(time.time()) % 30 == 0:
                    print(f"\rNext full run in {int(time_until_full_run/60)} minutes, next check in {int(time_until_check/60)} minutes    ", end="")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}Scheduled mode terminated by user.{Style.RESET_ALL}")
            return
    else:
        # Run single analysis
        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=args.show_reasoning,
            selected_analysts=selected_analysts,
            model_name=model_name,
            model_provider=model_provider,
            args=args,
        )
        
        # Check for errors
        if "error" in result:
            print(f"Error running hedge fund: {result['error']}")
            return
        
        # Print the trading output
        print_trading_output(result)
        
        # If live trading is enabled, update the portfolio status
        if LIVE_TRADING_ENABLED:
            alpaca_client = get_alpaca_client()
            if alpaca_client:
                alpaca_client, portfolio = update_portfolio_status(alpaca_client, portfolio)


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
    args = None,
):
    """
    Run the hedge fund for the given tickers and dates.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for the analysis
        end_date: End date for the analysis
        portfolio: Portfolio dictionary
        show_reasoning: Whether to show detailed reasoning from analysts
        selected_analysts: List of analysts to include
        model_name: Name of the LLM model to use
        model_provider: Provider of the LLM model
        args: Command line arguments
        
    Returns:
        Result dictionary containing decisions and analyst signals
    """
    # Add a timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    portfolio_copy = copy.deepcopy(portfolio)
    
    # Get the alpaca client for updating portfolio
    alpaca_client = get_alpaca_client()
    
    # Update portfolio with latest data
    if alpaca_client:
        portfolio_copy = update_portfolio_status(alpaca_client, portfolio_copy)
    
    # Set up the state for the workflow
    state = AgentState(
        data={
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "portfolio": portfolio_copy,
            "price_data": {},
            "analyst_signals": {},
        },
        metadata={
            "timestamp": timestamp,
            "show_reasoning": show_reasoning,
            "model_name": model_name,
            "model_provider": model_provider,
            "selected_analysts": selected_analysts,
        },
    )
    
    # Get the appropriate workflow and compile it
    if not selected_analysts:
        workflow = create_default_workflow().compile()
    else:
        workflow = create_workflow(selected_analysts).compile()
    
    # Run the workflow using the correct method for compiled workflows
    result = workflow.invoke(state)
    
    return result


if __name__ == "__main__":
    sys.exit(main())
