from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
import os
import logging


def sort_analyst_signals(signals):
    """Sort analyst signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    Print formatted trading results with colored tables for multiple tickers.

    Args:
        result: The result dictionary from the hedge fund run
    """
    data = result.get("data", {})
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {})
    decisions = data.get("portfolio_decisions", {})
    analyst_signals = data.get("analyst_signals", {})
    
    logging.info(f"Tickers to display: {tickers}")
    
    # Handle decisions
    logging.info(f"Decisions structure: {type(decisions)}")
    if hasattr(decisions, "decisions"):
        logging.info(f"Decision keys: {list(decisions.decisions.keys())}")
        original_decisions = decisions
        decisions = decisions.decisions
        logging.info(f"Original decisions structure: {type(original_decisions)}")
        logging.info(f"Original decisions keys: {list(original_decisions.decisions.keys())}")
    else:
        logging.info(f"Decision keys: {list(decisions.keys())}")
        
    final_decisions = {}
    for ticker in tickers:
        if ticker in decisions:
            decision = decisions[ticker]
            logging.info(f"Decision for {ticker} is of type: {type(decision)}")
            
            # No need to modify if already in the right format
            if hasattr(decision, 'action') and hasattr(decision, 'quantity') and hasattr(decision, 'confidence'):
                final_decisions[ticker] = decision
            elif isinstance(decision, dict) and all(k in decision for k in ['action', 'quantity', 'confidence']):
                # Convert from dict to pydantic model if needed
                from agents.portfolio_manager import PortfolioDecision
                final_decisions[ticker] = PortfolioDecision(
                    action=decision['action'],
                    quantity=decision['quantity'],
                    confidence=decision['confidence'],
                    reasoning=decision.get('reasoning', "No reasoning provided")
                )
            else:
                logging.warning(f"Unexpected decision format for {ticker}: {decision}")
    
    logging.info(f"Final decisions structure has keys: {list(final_decisions.keys())}")
    logging.info(f"Expected tickers: {tickers}")
    
    for ticker in tickers:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Analysis for {ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}=================================================={Style.RESET_ALL}")
        
        # Display analyst signals for this ticker
        print(f"\n{Fore.WHITE}{Style.BRIGHT}ANALYST SIGNALS: [{ticker}]{Style.RESET_ALL}")
        signals = []
        
        for analyst, output in analyst_signals.items():
            if analyst == "portfolio_management_agent":
                continue
                
            # Skip if the analyst doesn't have a signal for this ticker
            if ticker not in output:
                continue
                
            signal_output = output[ticker]
            
            # Determine display name
            display_name = analyst.replace("_agent", "").replace("_", " ").title()
            
            # For William O'Neil's special signal format
            if "signal" in signal_output and signal_output["signal"] in ["strong_buy", "buy", "weak_buy", "weak_sell", "sell", "strong_sell"]:
                signal_value = signal_output["signal"].upper()
            # For most standard signals
            elif "signal" in signal_output:
                signal_value = signal_output["signal"].upper()
            # For some legacy formats
            elif "trading_signal" in signal_output:
                signal_value = signal_output["trading_signal"].upper()
            else:
                signal_value = "UNKNOWN"
                
            # Get confidence
            confidence = signal_output.get("confidence", 0)
            
            signals.append((display_name, signal_value, confidence))
            
        # Sort signals by analyst name using predefined order
        sorted_signals = sort_analyst_signals(signals)
        
        # Create the signals table
        signals_table = []
        for name, signal, confidence in sorted_signals:
            # Format signal with color
            if signal in ["BULLISH", "STRONG_BUY", "BUY"]:
                signal_formatted = f"{Fore.GREEN}{signal}{Style.RESET_ALL}"
            elif signal in ["BEARISH", "STRONG_SELL", "SELL"]:
                signal_formatted = f"{Fore.RED}{signal}{Style.RESET_ALL}"
            elif signal in ["WEAK_BUY"]:
                signal_formatted = f"{Fore.CYAN}{signal}{Style.RESET_ALL}"
            elif signal in ["WEAK_SELL"]:
                signal_formatted = f"{Fore.MAGENTA}{signal}{Style.RESET_ALL}"
            else:
                signal_formatted = f"{Fore.YELLOW}{signal}{Style.RESET_ALL}"
                
            signals_table.append([name, signal_formatted, f"{confidence}%"])
            
        print(tabulate(
            signals_table,
            headers=["Analyst", "Signal", "Confidence"],
            tablefmt="grid"
        ))
        
        # Display trading decision for this ticker
        print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION: [{ticker}]{Style.RESET_ALL}")
        
        if ticker in final_decisions:
            decision = final_decisions[ticker]
            action = decision.action.upper()
            
            # Color code the action
            if action in ["BUY", "COVER"]:
                action_color = Fore.GREEN
            elif action in ["SELL", "SHORT"]:
                action_color = Fore.RED
            else:
                action_color = Fore.YELLOW
                
            # Format quantity
            quantity = decision.quantity
            
            # Format confidence
            confidence = decision.confidence
            
            decision_table = [
                ["Action", f"{action_color}{action}{Style.RESET_ALL}"],
                ["Quantity", quantity],
                ["Confidence", f"{confidence}%"]
            ]
            
            print(tabulate(decision_table, tablefmt="grid"))
            
            # Print reasoning if available
            if hasattr(decision, 'reasoning') and decision.reasoning:
                print(f"\n{Fore.WHITE}{Style.BRIGHT}REASONING:{Style.RESET_ALL}")
                print(decision.reasoning)
        else:
            print(f"{Fore.YELLOW}No trading decision available for {ticker}{Style.RESET_ALL}")
    
    # Print portfolio summary if available
    if portfolio:
        print(f"\n{Fore.WHITE}{Style.BRIGHT}=================================================={Style.RESET_ALL}")
        logging.info(f"Portfolio structure: {list(portfolio.keys())}")
        
        # Portfolio Summary
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
        print(f"Cash Balance: ${portfolio.get('cash', 0):,.2f}")
        
        # Calculate total position value
        total_position_value = 0
        if 'positions' in portfolio:
            for pos in portfolio['positions'].values():
                if 'market_value' in pos:
                    total_position_value += pos['market_value']
                elif 'quantity' in pos and 'current_price' in pos:
                    total_position_value += pos['quantity'] * pos['current_price']
                    
        print(f"Total Position Value: ${total_position_value:,.2f}")
        print(f"Total Portfolio Value: ${portfolio.get('portfolio_value', 0):,.2f}")
        
        # Current Holdings
        print(f"\n{Fore.WHITE}{Style.BRIGHT}CURRENT HOLDINGS:{Style.RESET_ALL}")
        if 'positions' in portfolio and portfolio['positions']:
            holdings_table = []
            for ticker, position in portfolio['positions'].items():
                if 'quantity' in position and position['quantity'] != 0:
                    qty = position['quantity']
                    avg_price = position.get('avg_price', 0)
                    current_price = position.get('current_price', 0)
                    market_value = position.get('market_value', 0)
                    unrealized_pl = position.get('unrealized_pl', 0)
                    
                    # Calculate percent return
                    if qty != 0 and avg_price != 0:
                        pct_return = (current_price - avg_price) / avg_price * 100
                    else:
                        pct_return = 0
                        
                    holdings_table.append([
                        ticker,
                        qty,
                        f"${avg_price:.2f}",
                        f"${current_price:.2f}",
                        f"${market_value:.2f}",
                        f"${unrealized_pl:.2f}",
                        f"{pct_return:.2f}%"
                    ])
                    
            if holdings_table:
                print(tabulate(
                    holdings_table,
                    headers=["Ticker", "Quantity", "Avg Price", "Current Price", "Market Value", "Unrealized P/L", "% Return"],
                    tablefmt="simple"
                ))
            else:
                print(f"{Fore.YELLOW}No current holdings{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No current holdings{Style.RESET_ALL}")
        
        # Open Orders
        print(f"\n{Fore.WHITE}{Style.BRIGHT}OPEN ORDERS:{Style.RESET_ALL}")
        if 'open_orders' in portfolio:
            open_orders = portfolio['open_orders']
            logging.info(f"Open orders found in portfolio: {type(open_orders)}")
            
            if isinstance(open_orders, dict):
                logging.info(f"Open orders keys: {list(open_orders.keys())}")
                order_count = sum(len(orders) for orders in open_orders.values())
                logging.info(f"Total order count: {order_count}")
                
                if open_orders:
                    orders_table = []
                    
                    for ticker, ticker_orders in open_orders.items():
                        for order in ticker_orders:
                            try:
                                # Try to get order details - handle different attribute access methods
                                side = getattr(order, 'side', None)
                                if side is None and hasattr(order, '__getitem__'):
                                    side = order.get('side', 'unknown')
                                
                                qty = getattr(order, 'qty', None)
                                if qty is None and hasattr(order, '__getitem__'):
                                    qty = order.get('qty', 'unknown')
                                
                                status = getattr(order, 'status', None)
                                if status is None and hasattr(order, '__getitem__'):
                                    status = order.get('status', 'unknown')
                                
                                # Get order type
                                order_type = getattr(order, 'type', None)
                                if order_type is None and hasattr(order, '__getitem__'):
                                    order_type = order.get('type', 'unknown')
                                
                                # Try to get price information
                                price = None
                                # First try limit_price
                                if hasattr(order, 'limit_price'):
                                    price = order.limit_price
                                elif hasattr(order, '__getitem__') and 'limit_price' in order:
                                    price = order['limit_price']
                                
                                # Then try price
                                if price is None:
                                    if hasattr(order, 'price'):
                                        price = order.price
                                    elif hasattr(order, '__getitem__') and 'price' in order:
                                        price = order['price']
                                
                                # Format price display
                                if order_type and order_type.lower() == 'market':
                                    price_display = "Market"
                                elif price:
                                    price_display = f"${float(price):.2f}"
                                else:
                                    price_display = "N/A"
                                
                                # Format side with status
                                side_display = f"{side} ({status})" if status and status.lower() != 'open' else side
                                
                                # Add row to table
                                orders_table.append([ticker, side_display, qty, price_display])
                            except Exception as e:
                                logging.error(f"Error processing order: {e}")
                    
                    if orders_table:
                        print(tabulate(
                            orders_table,
                            headers=["Ticker", "Side", "Quantity", "Price"],
                            tablefmt="simple"
                        ))
                    else:
                        print(f"{Fore.YELLOW}No open orders{Style.RESET_ALL}")
                        
                        # Display diagnostics if available and no orders are shown
                        if '_order_diagnostics' in portfolio:
                            diagnostics = portfolio['_order_diagnostics']
                            if diagnostics and 'order_count' in diagnostics and diagnostics['order_count'] > 0:
                                print(f"\n{Fore.YELLOW}DIAGNOSTICS: Orders found but not displayed{Style.RESET_ALL}")
                                print(f"Total orders detected: {diagnostics['order_count']}")
                                if 'status_breakdown' in diagnostics:
                                    print(f"Status breakdown: {diagnostics['status_breakdown']}")
                                if 'aapl_orders' in diagnostics and diagnostics['aapl_orders'] > 0:
                                    print(f"AAPL orders: {diagnostics['aapl_orders']} (check logs for details)")
                                if 'voo_orders' in diagnostics and diagnostics['voo_orders'] > 0:
                                    print(f"VOO orders: {diagnostics['voo_orders']} (check logs for details)")
                                print(f"Environment: {diagnostics.get('environment', 'unknown')}")
                else:
                    print(f"{Fore.YELLOW}No open orders{Style.RESET_ALL}")
                    
                    # Display diagnostics if available
                    if '_order_diagnostics' in portfolio:
                        diagnostics = portfolio['_order_diagnostics']
                        if diagnostics and 'order_count' in diagnostics and diagnostics['order_count'] > 0:
                            print(f"\n{Fore.YELLOW}DIAGNOSTICS: Orders found but not displayed{Style.RESET_ALL}")
                            print(f"Total orders detected: {diagnostics['order_count']}")
                            if 'status_breakdown' in diagnostics:
                                print(f"Status breakdown: {diagnostics['status_breakdown']}")
                            if 'aapl_orders' in diagnostics and diagnostics['aapl_orders'] > 0:
                                print(f"AAPL orders: {diagnostics['aapl_orders']} (check logs for details)")
                            if 'voo_orders' in diagnostics and diagnostics['voo_orders'] > 0:
                                print(f"VOO orders: {diagnostics['voo_orders']} (check logs for details)")
                            print(f"Environment: {diagnostics.get('environment', 'unknown')}")
            else:
                print(f"{Fore.YELLOW}No open orders (unexpected format){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No open orders information available{Style.RESET_ALL}")
    
    print(f"\n{Fore.WHITE}{Style.BRIGHT}=================================================={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Analysis complete.{Style.RESET_ALL}")


def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table"""
    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # Split rows into ticker rows and summary rows
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    
    # Display latest portfolio summary
    if summary_rows:
        latest_summary = summary_rows[-1]
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")

        # Extract values and remove commas before converting to float
        cash_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        position_str = latest_summary[6].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Return: {latest_summary[9]}")
        
        # Display performance metrics if available
        if latest_summary[10]:  # Sharpe ratio
            print(f"Sharpe Ratio: {latest_summary[10]}")
        if latest_summary[11]:  # Sortino ratio
            print(f"Sortino Ratio: {latest_summary[11]}")
        if latest_summary[12]:  # Max drawdown
            print(f"Max Drawdown: {latest_summary[12]}")

    # Add vertical spacing
    print("\n" * 2)

    # Print the table with just ticker rows
    print(
        tabulate(
            ticker_rows,
            headers=[
                "Date",
                "Ticker",
                "Action",
                "Quantity",
                "Price",
                "Shares",
                "Position Value",
                "Bullish",
                "Bearish",
                "Neutral",
            ],
            tablefmt="grid",
            colalign=(
                "left",  # Date
                "left",  # Ticker
                "center",  # Action
                "right",  # Quantity
                "right",  # Price
                "right",  # Shares
                "right",  # Position Value
                "right",  # Bullish
                "right",  # Bearish
                "right",  # Neutral
            ),
        )
    )

    # Add vertical spacing
    print("\n" * 4)


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    shares_owned: float,
    position_value: float,
    bullish_count: int,
    bearish_count: int,
    neutral_count: int,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
) -> list[any]:
    """Format a row for the backtest results table"""
    # Color the action
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.YELLOW,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",  # Total Position Value
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",  # Cash Balance
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",  # Total Value
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",  # Return
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",  # Sharpe Ratio
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",  # Sortino Ratio
            f"{Fore.RED}{max_drawdown:.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",  # Max Drawdown
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{shares_owned:,.0f}{Style.RESET_ALL}",
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
            f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
            f"{Fore.BLUE}{neutral_count}{Style.RESET_ALL}",
        ]
