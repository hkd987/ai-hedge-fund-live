from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json
import os
import logging
import pandas as pd
from utils.caching import cached_analyst

# Import the basic risk management utility functions
from utils.risk_manager import (
    RISK_PARAMS,
    check_position_size_limit,
    should_adjust_for_volatility,
    update_portfolio_value,
    TODAY_STATE
)

# Import enhanced risk management functions 
from utils.enhanced_risk import (
    track_sector_exposure,
    check_sector_limits,
    check_correlation_risk,
    detect_market_regime,
    assess_portfolio_liquidity,
    calculate_portfolio_beta,
    adjust_risk_for_timing,
    calculate_position_risk_parameters,
    enhance_risk_management,
    generate_risk_dashboard
)

# Try to import market data tools
try:
    from utils.market_data import get_current_vix, ALPACA_AVAILABLE as MARKET_DATA_AVAILABLE
except ImportError:
    MARKET_DATA_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('risk_manager_agent')

##### Risk Management Agent #####
@cached_analyst()
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    # Ensure we're working with a dictionary, not the Alpaca client
    # Extract portfolio from state data
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
    
    # Now safely extract portfolio
    data = state_dict.get('data', {})
    portfolio = data.get('portfolio', {})
    tickers = data.get('tickers', [])
    
    # Ensure portfolio is a dictionary
    if not isinstance(portfolio, dict):
        logger.error(f"Portfolio is not a dictionary: {type(portfolio)}")
        # Create a default portfolio if we can't get a valid one
        portfolio = {
            "cash": 10000.0,
            "portfolio_value": 10000.0,
            "positions": {},
            "realized_gains": {}
        }

    # Initialize risk analysis for each ticker
    risk_analysis = {}
    current_prices = {}  # Store prices here to avoid redundant API calls
    prices_data = {}  # Store price dataframes for all tickers
    market_data = None  # Store market data (SPY) for regime detection

    # Get current VIX value for volatility adjustments if available
    vix_value = None
    if MARKET_DATA_AVAILABLE:
        try:
            vix_value = get_current_vix()
            if vix_value:
                logger.info(f"Current VIX value: {vix_value}")
        except Exception as e:
            logger.warning(f"Failed to get VIX data: {e}")

    # Get volatility adjustment factor
    is_high_volatility, volatility_adjustment = should_adjust_for_volatility(vix_value)

    # Calculate total portfolio value
    # Use safer access methods for the portfolio dictionary
    total_portfolio_value = portfolio.get("cash", 0)
    
    # Safely access positions
    positions = portfolio.get("positions", {})
    for ticker in positions:
        position = positions.get(ticker, {})
        
        # Safely get long position value
        long_qty = position.get("long", 0)
        if long_qty > 0:
            long_cost_basis = position.get("long_cost_basis", 0)
            long_value = long_qty * long_cost_basis
            total_portfolio_value += long_value
        
        # Safely get short position value
        short_qty = position.get("short", 0)
        if short_qty > 0:
            # For shorts, we add the margin (typically 50% of position value)
            short_margin = position.get("short_margin_used", 0)
            total_portfolio_value += short_margin

    # Update the risk management system with current portfolio value
    update_portfolio_value(total_portfolio_value)

    # Get market index data (SPY) for market regime detection
    progress.update_status("risk_management_agent", "SPY", "Getting market data for regime detection")
    try:
        spy_prices = get_prices(
            ticker="SPY",
            start_date=data["start_date"],
            end_date=data["end_date"],
        )
        if spy_prices:
            market_data = prices_to_df(spy_prices)
            progress.update_status("risk_management_agent", "SPY", "Market data retrieved successfully")
        else:
            progress.update_status("risk_management_agent", "SPY", "Failed to get market data")
    except Exception as e:
        logger.warning(f"Failed to get market data: {e}")
        progress.update_status("risk_management_agent", "SPY", "Error fetching market data")

    # Get price data for all tickers
    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)
        prices_data[ticker] = prices_df
        
        # Store current price
        current_price = prices_df["close"].iloc[-1]
        current_prices[ticker] = current_price
    
    # Track sector exposure for the portfolio
    progress.update_status("risk_management_agent", None, "Analyzing sector exposure")
    sector_exposures = track_sector_exposure(portfolio)
    over_exposed_sectors = check_sector_limits(sector_exposures, RISK_PARAMS["MAX_SECTOR_EXPOSURE_PCT"])
    
    # Check for correlation risks
    progress.update_status("risk_management_agent", None, "Checking correlation risks")
    correlation_risks = check_correlation_risk(portfolio)
    
    # Calculate portfolio beta
    progress.update_status("risk_management_agent", None, "Calculating portfolio beta")
    portfolio_beta = calculate_portfolio_beta(portfolio)
    
    # Assess portfolio liquidity
    progress.update_status("risk_management_agent", None, "Assessing portfolio liquidity")
    liquidity_risk = assess_portfolio_liquidity(portfolio)
    
    # Detect market regime
    market_regime = {"regime": "Unknown", "risk_adjustment": 1.0}
    if market_data is not None:
        progress.update_status("risk_management_agent", None, "Detecting market regime")
        market_regime = detect_market_regime(market_data)
        logger.info(f"Current market regime: {market_regime['regime']} (risk adjustment: {market_regime['risk_adjustment']})")

    # Generate comprehensive risk dashboard if we have all required data
    if market_data is not None and len(prices_data) > 0:
        progress.update_status("risk_management_agent", None, "Generating risk dashboard")
        risk_dashboard = generate_risk_dashboard(portfolio, tickers, prices_data)
    else:
        risk_dashboard = None
    
    # Process each ticker to determine position limits
    for ticker in tickers:
        if ticker not in prices_data:
            continue

        progress.update_status("risk_management_agent", ticker, "Calculating position limits")
        
        prices_df = prices_data[ticker]
        current_price = current_prices[ticker]
        
        # Calculate current position value for this ticker
        current_position = portfolio.get("positions", {}).get(ticker, {})
        long_position_value = current_position.get("long", 0) * current_position.get("long_cost_basis", 0)
        short_position_value = current_position.get("short", 0) * current_position.get("short_cost_basis", 0)
        current_position_value = long_position_value + short_position_value

        # Calculate basic position limit based on max position size parameter
        # Use the advanced position sizing logic from utils/risk_manager.py
        passes_size_limit, max_position_size = check_position_size_limit(
            ticker, 
            total_portfolio_value * RISK_PARAMS["MAX_POSITION_SIZE_PCT"], 
            total_portfolio_value
        )

        # Apply all enhanced risk management techniques
        if market_data is not None:
            enhanced_risk = enhance_risk_management(
                ticker=ticker,
                position_limit=max_position_size,
                portfolio=portfolio,
                market_data=market_data,
                prices_df=prices_df
            )
            
            # Use the enhanced limit that includes all risk adjustments
            max_position_size = enhanced_risk["enhanced_limit"]
            logger.info(f"Enhanced position limit for {ticker}: ${max_position_size:.2f} (original: ${enhanced_risk['original_limit']:.2f})")
            
            # Check if trading should be halted based on drawdown
            circuit_breaker_active = enhanced_risk.get("halt_trading", False) or TODAY_STATE.get("circuit_breaker_triggered", False)
        else:
            # Apply basic volatility adjustment if enhanced risk isn't available
            if is_high_volatility:
                max_position_size *= volatility_adjustment
                logger.info(f"Adjusting position size for {ticker} due to high volatility (VIX: {vix_value})")
                
            circuit_breaker_active = TODAY_STATE.get("circuit_breaker_triggered", False)
            
            # Apply sector constraint if ticker is in over-exposed sector
            ticker_sector = None
            for sector in over_exposed_sectors:
                if f"({ticker})" in sector:
                    ticker_sector = sector.split(" ")[0]  # Extract sector name
                    break
                    
            if ticker_sector:
                # Reduce position size by 50% for over-exposed sectors
                max_position_size *= 0.5
                logger.info(f"Reducing position size for {ticker} by 50% due to over-exposed sector: {ticker_sector}")

        # For existing positions, subtract current position value from limit
        remaining_position_limit = max_position_size - current_position_value

        # Ensure we don't exceed available cash for new purchases
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        # Calculate maximum shares based on current price
        max_shares = int(max_position_size / current_price) if current_price > 0 else 0

        # If circuit breaker is active, no new trades allowed
        if circuit_breaker_active:
            logger.warning(f"Circuit breaker active - no new trades allowed for {ticker}")
            max_position_size = 0
            max_shares = 0
            
        # Calculate position risk parameters for existing positions
        position_risk_params = None
        entry_price = current_position.get("long_cost_basis", 0) if current_position.get("long", 0) > 0 else current_position.get("short_cost_basis", 0)
        if entry_price > 0 and current_price > 0:
            position_risk_params = calculate_position_risk_parameters(
                ticker, entry_price, current_price, prices_df
            )

        # Calculate risk score from 0-100 (higher is better/less risky)
        # Start with a base score of 75 (moderate risk)
        risk_score = 75.0
        
        # Adjust score based on various risk factors
        if circuit_breaker_active:
            risk_score = 0.0  # Zero confidence if circuit breaker is active
        else:
            # Adjust for market regime
            if market_regime["regime"] == "Bear":
                risk_score -= 20
            elif market_regime["regime"] == "Correction":
                risk_score -= 10
            elif market_regime["regime"] == "Bull":
                risk_score += 10
                
            # Adjust for sector exposure
            if ticker_sector:
                risk_score -= 15  # Penalize for over-exposed sector
                
            # Adjust for volatility
            if is_high_volatility:
                risk_score -= 15
                
            # Ensure score is between 1 and 100 (never use 0 for valid signals)
            risk_score = max(1.0, min(100.0, risk_score))

        # Compile all risk information
        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "max_shares": max_shares,
            "circuit_breaker_active": circuit_breaker_active,
            "sector_exposure": {s: e for s, e in sector_exposures.items() if s != "Cash"},
            "over_exposed_sectors": over_exposed_sectors,
            "market_regime": market_regime.get("regime", "Unknown"),
            "position_risk_params": position_risk_params,
            "confidence": float(risk_score),  # Add explicit confidence score
            "signal": "neutral",  # Risk manager always provides neutral signal
            "reasoning": f"Risk assessment based on market regime ({market_regime.get('regime', 'Unknown')}), " +
                       f"volatility ({is_high_volatility}), and sector exposure. " +
                       f"Max shares allowed: {max_shares}, Position limit: ${max_position_size:.2f}",
            "reasoning_detail": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "max_position_size_pct": float(RISK_PARAMS["MAX_POSITION_SIZE_PCT"]),
                "position_limit": float(max_position_size),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "vix_value": vix_value,
                "high_volatility": is_high_volatility,
                "volatility_adjustment": volatility_adjustment if is_high_volatility else 1.0,
                "portfolio_beta": portfolio_beta.get("portfolio_beta"),
                "market_regime_risk_adjustment": market_regime.get("risk_adjustment", 1.0),
                "liquidity_risk": liquidity_risk.get("portfolio_liquidity_risk", "Unknown"),
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    # Add overall portfolio risk stats to the output
    if len(risk_analysis) > 0:
        risk_analysis["portfolio_stats"] = {
            "sector_exposures": sector_exposures,
            "over_exposed_sectors": over_exposed_sectors,
            "correlation_risks": [f"{r['ticker1']} & {r['ticker2']} ({r['correlation']:.2f})" for r in correlation_risks],
            "portfolio_beta": portfolio_beta.get("portfolio_beta"),
            "market_regime": market_regime.get("regime"),
            "liquidity_risk": liquidity_risk.get("portfolio_liquidity_risk"),
            "circuit_breaker_active": TODAY_STATE.get("circuit_breaker_triggered", False),
            "total_portfolio_value": total_portfolio_value
        }
    
    # Log the number of signals and their confidence scores
    if tickers:
        logger.info(f"Generated risk signals for {len([t for t in risk_analysis if t in tickers])} tickers")
        for ticker in tickers:
            if ticker in risk_analysis:
                logger.info(f"Risk signal for {ticker}: confidence={risk_analysis[ticker].get('confidence', 0)}")
            else:
                logger.warning(f"No risk signal generated for {ticker}")
    
    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
