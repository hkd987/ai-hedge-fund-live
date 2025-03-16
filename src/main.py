import sys
import os

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
from datetime import datetime
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
        return TradingClient(api_key, api_secret, paper=True)
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
        # Initialize Alpaca client
        client = TradingClient(api_key, api_secret, paper=True)
        
        # Get all positions
        positions = client.get_all_positions()
        
        # Extract tickers from positions
        holdings = [position.symbol for position in positions]
        
        if holdings:
            logger.info(f"Found {len(holdings)} holdings in Alpaca account: {', '.join(holdings)}")
        else:
            logger.info("No holdings found in Alpaca account")
            
        return holdings
        
    except Exception as e:
        logger.error(f"Failed to get holdings from Alpaca: {e}")
        return []


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
        # Get all positions
        positions = client.get_all_positions()
        
        if not positions:
            logger.info("No positions to liquidate.")
            return True
        
        logger.warning(f"EMERGENCY LIQUIDATION: Attempting to liquidate {len(positions)} positions")
        
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
            logger.warning(f"EMERGENCY: Submitted {action} order for {qty} shares of {ticker}: Order ID {order.id}")
        
        logger.warning("EMERGENCY LIQUIDATION COMPLETE: All positions have been liquidated")
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


##### Run the Hedge Fund #####
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position. Defaults to 100000.0)"
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement. Defaults to 0.0"
    )
    parser.add_argument("--tickers", type=str, required=False, help="Comma-separated list of stock ticker symbols")
    parser.add_argument(
        "--include-alpaca-holdings",
        action="store_true",
        help="Include all current holdings from Alpaca account in addition to tickers specified"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    parser.add_argument(
        "--show-agent-graph", action="store_true", help="Show the agent graph"
    )
    parser.add_argument(
        "--emergency-liquidate",
        action="store_true",
        help="Emergency liquidate all positions and exit. This is the 'panic button'."
    )

    args = parser.parse_args()
    
    # Handle emergency liquidation first if requested
    if args.emergency_liquidate:
        if questionary.confirm(
            "⚠️ EMERGENCY LIQUIDATION ⚠️\nThis will sell ALL positions immediately at market price.\nAre you absolutely sure?",
            default=False
        ).ask():
            print(f"{Fore.RED}{Style.BRIGHT}EXECUTING EMERGENCY LIQUIDATION{Style.RESET_ALL}")
            if emergency_liquidate_all_positions():
                print(f"{Fore.GREEN}Emergency liquidation complete.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to complete emergency liquidation.{Style.RESET_ALL}")
        else:
            print("Emergency liquidation cancelled.")
        sys.exit(0)

    # Get tickers from various sources
    all_tickers = []
    
    # Add manually specified tickers
    if args.tickers:
        manual_tickers = [ticker.strip() for ticker in args.tickers.split(",")]
        all_tickers.extend(manual_tickers)
        print(f"Using manually specified tickers: {', '.join(manual_tickers)}")
    
    # Add Alpaca holdings if flag is set
    if args.include_alpaca_holdings:
        alpaca_holdings = get_alpaca_holdings()
        # Only add unique tickers not already in the list
        for ticker in alpaca_holdings:
            if ticker not in all_tickers:
                all_tickers.append(ticker)
        
        if alpaca_holdings:
            print(f"Added {len(alpaca_holdings)} tickers from Alpaca holdings")
    
    # Ensure we have at least one ticker to analyze
    if not all_tickers:
        print("Error: No tickers specified. Use --tickers or --include-alpaca-holdings to specify tickers.")
        sys.exit(1)
    
    print(f"\nAnalyzing the following tickers: {', '.join(all_tickers)}")

    # Select analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

    # Select LLM model
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()

    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        # Get model info using the helper function
        model_info = get_model_info(model_choice)
        if model_info:
            model_provider = model_info.provider.value
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": args.initial_cash,  # Initial cash amount
        "margin_requirement": args.margin_requirement,  # Initial margin requirement
        "positions": {
            ticker: {
                "long": 0,  # Number of shares held long
                "short": 0,  # Number of shares held short
                "long_cost_basis": 0.0,  # Average cost basis for long positions
                "short_cost_basis": 0.0,  # Average price at which shares were sold short
            } for ticker in all_tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # Realized gains from long positions
                "short": 0.0,  # Realized gains from short positions
            } for ticker in all_tickers
        }
    }

    # Run the hedge fund
    result = run_hedge_fund(
        tickers=all_tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_choice,
        model_provider=model_provider,
    )
    print_trading_output(result)
