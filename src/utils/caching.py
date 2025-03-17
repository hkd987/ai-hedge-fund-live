import functools
import logging
import json
import os
from typing import Callable, Dict, Any, List
import inspect
from datetime import datetime
from functools import wraps
from pathlib import Path
import traceback

from utils.run_cache import save_analyst_data, load_analyst_data, should_use_cached_data, ANALYST_CACHE_DIR

logger = logging.getLogger('caching')

def cached_analyst(cache_only_if_success=True):
    """
    Decorator to cache the output of analyst functions.
    Uses the function name and tickers as the cache key.
    
    Args:
        cache_only_if_success: Only cache the output if the function returns successfully
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            # Get the name of the function without the module path
            func_name = func.__name__
            logger.info(f"Running {func_name}")
            
            # Extract tickers from state
            tickers = []
            if isinstance(state, dict) and "data" in state and "tickers" in state["data"]:
                tickers = state["data"]["tickers"]
            elif hasattr(state, "data") and hasattr(state.data, "tickers"):
                tickers = state.data.tickers
                
            if not tickers:
                logger.warning(f"No tickers found in state for {func_name}")
                return func(state, *args, **kwargs)
                
            # Process first ticker (we'll run once for all tickers)
            ticker = tickers[0] if tickers else "unknown"
            logger.info(f"Processing analyst {func_name} for ticker(s): {tickers}")
            
            # Create cache directory if it doesn't exist
            cache_dir = Path("src/cache/analysts")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a cache filename with function name and ticker
            # Normalize function name by removing any "agent" suffix for consistency
            cache_func_name = func_name
            if cache_func_name.endswith("_agent"):
                cache_func_name = cache_func_name.replace("_agent", "")
                
            cache_file = cache_dir / f"{cache_func_name}_{ticker}.json"
            logger.info(f"Cache file: {cache_file}")
            
            # Check if we should skip the cache for this agent
            skip_cache = False
            skip_agents = ["risk_management", "portfolio_management"]
            
            # Normalize func_name before checking if it should be skipped
            normalized_func_name = func_name
            if normalized_func_name.endswith("_agent"):
                normalized_func_name = normalized_func_name.replace("_agent", "")
                
            if normalized_func_name in skip_agents:
                logger.info(f"Skipping cache for {func_name} (always uses fresh data)")
                skip_cache = True
            
            # Check if there's already cached data
            if not skip_cache and cache_file.exists():
                try:
                    logger.info(f"Checking for cached data: {cache_file}")
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Verify the cached data is usable - accept more formats
                    if isinstance(cached_data, dict):
                        # Check for common signal structures or any valid data
                        valid_cache = (
                            "signal" in cached_data or 
                            "decisions" in cached_data or
                            "analyst_signals" in cached_data or
                            "data" in cached_data or
                            len(cached_data) > 0  # Accept any non-empty dict
                        )
                        
                        if valid_cache:
                            logger.info(f"Found cached data for {func_name} ({ticker})")
                            
                            # Check for metadata to log when this was cached
                            cached_time = cached_data.get("cache_timestamp", "unknown")
                            if cached_time != "unknown":
                                try:
                                    cached_dt = datetime.fromisoformat(cached_time)
                                    time_since_cached = datetime.now() - cached_dt
                                    minutes_ago = time_since_cached.total_seconds() / 60
                                    logger.info(f"Using cached data from {minutes_ago:.1f} minutes ago")
                                except Exception as e:
                                    logger.warning(f"Could not parse cache timestamp: {e}")
                            
                            # Check if we should use cached data from run_cache
                            if should_use_cached_data(tickers):
                                logger.info(f"Using cached data for {func_name} ({ticker})")
                                
                                # Ensure we return the cached data in the correct state format
                                # Check if this is an analyst signal meant for the state
                                if "signal" in cached_data or all(isinstance(v, dict) for v in cached_data.values() if isinstance(v, dict)):
                                    # This appears to be signal data that should be added to the state
                                    
                                    # Process signals that are stored as strings instead of objects
                                    if any(isinstance(v, str) and "signal=" in v and "confidence=" in v for v in cached_data.values()):
                                        logger.info("Detected string-formatted signals in cache - converting to proper format")
                                        import re
                                        for k, v in list(cached_data.items()):
                                            if isinstance(v, str) and "signal=" in v and "confidence=" in v:
                                                try:
                                                    signal_match = re.search(r"signal=['\"]([^'\"]+)['\"]", v)
                                                    confidence_match = re.search(r"confidence=([0-9.]+)", v)
                                                    reasoning_match = re.search(r"reasoning=['\"]([^$]*?)['\"](?:\s|$)", v)
                                                    
                                                    if signal_match and confidence_match:
                                                        signal = signal_match.group(1)
                                                        confidence = float(confidence_match.group(1))
                                                        reasoning = reasoning_match.group(1) if reasoning_match else ""
                                                        
                                                        cached_data[k] = {
                                                            "signal": signal,
                                                            "confidence": confidence,
                                                            "reasoning": reasoning
                                                        }
                                                        logger.info(f"Converted string signal for {k} with confidence={confidence}")
                                                except Exception as e:
                                                    logger.error(f"Failed to convert string signal: {e}")
                                    
                                    # Make a shallow copy of the state to avoid modifying the original
                                    if isinstance(state, dict):
                                        state_copy = state.copy()
                                        if "data" not in state_copy:
                                            state_copy["data"] = {}
                                        if "analyst_signals" not in state_copy["data"]:
                                            state_copy["data"]["analyst_signals"] = {}
                                            
                                        # Add the cached data to the analyst signals
                                        norm_name = normalized_func_name or func_name
                                        state_copy["data"]["analyst_signals"][norm_name] = cached_data
                                        
                                        # Create a message with the cached data
                                        from langchain_core.messages import HumanMessage
                                        message = HumanMessage(
                                            content=json.dumps(cached_data),
                                            name=func_name,
                                        )
                                        
                                        if "messages" not in state_copy:
                                            state_copy["messages"] = []
                                        state_copy["messages"].append(message)
                                        
                                        # Return the updated state
                                        return {
                                            "messages": state_copy.get("messages", []),
                                            "data": state_copy.get("data", {}),
                                            "metadata": state_copy.get("metadata", {})
                                        }
                                    # Handle AgentState object
                                    elif hasattr(state, 'data'):
                                        try:
                                            # Extract current data
                                            if hasattr(state.data, "analyst_signals"):
                                                # Add the cached data to analyst_signals
                                                norm_name = normalized_func_name or func_name
                                                state.data.analyst_signals[norm_name] = cached_data
                                            else:
                                                # Create analyst_signals if it doesn't exist
                                                setattr(state.data, "analyst_signals", {
                                                    norm_name: cached_data
                                                })
                                            
                                            # Create a message with the cached data
                                            from langchain_core.messages import HumanMessage
                                            message = HumanMessage(
                                                content=json.dumps(cached_data),
                                                name=func_name,
                                            )
                                            
                                            # Return updated state in correct format
                                            return {
                                                "messages": [message],
                                                "data": state.data
                                            }
                                        except Exception as e:
                                            logger.error(f"Error updating AgentState with cached data: {e}")
                                            # Fall back to original behavior
                                            
                                    # If we couldn't create a proper state, just return the cached data
                                    # (the function that called it will have to handle it properly)
                                    return cached_data
                                else:
                                    logger.info(f"Running {func_name} with fresh data (cache TTL expired)")
                        else:
                            logger.warning(f"Cached data for {func_name} ({ticker}) has unexpected format, will run fresh")
                except json.JSONDecodeError as e:
                    logger.error(f"Error reading cache for {func_name} ({ticker}): Invalid JSON - {e}")
                except Exception as e:
                    logger.error(f"Error reading cache for {func_name} ({ticker}): {e}")
            
            # Run the function since cache is missing or should be skipped
            result = func(state, *args, **kwargs)
            
            # Cache the result if caching is enabled for this agent
            if not skip_cache and (result is not None or not cache_only_if_success):
                try:
                    logger.info(f"Caching result for {func_name} ({ticker})")
                    
                    # Ensure the result is JSON serializable
                    serializable_result = make_json_serializable(result)
                    
                    # For analyst agents, we need to extract the actual signal data
                    # The data might be in different formats:
                    # 1. In the 'data' field under 'analyst_signals'
                    # 2. Directly in the result
                    
                    data_to_cache = serializable_result
                    
                    # Check if result contains analyst_signals - common pattern
                    if isinstance(serializable_result, dict) and "data" in serializable_result:
                        if "analyst_signals" in serializable_result["data"]:
                            analyst_signals = serializable_result["data"]["analyst_signals"]
                            # Check if this agent's results are in analyst_signals
                            if normalized_func_name in analyst_signals:
                                data_to_cache = analyst_signals[normalized_func_name]
                                logger.info(f"Extracted {normalized_func_name} data from analyst_signals")
                            elif func_name in analyst_signals:
                                data_to_cache = analyst_signals[func_name]
                                logger.info(f"Extracted {func_name} data from analyst_signals")
                    
                    # Add metadata
                    if isinstance(data_to_cache, dict):
                        data_to_cache["cache_timestamp"] = datetime.now().isoformat()
                        data_to_cache["cached_ticker"] = ticker
                        data_to_cache["cached_function"] = cache_func_name
                    
                    # Write to cache file
                    with open(cache_file, 'w') as f:
                        json.dump(data_to_cache, f, indent=2)
                    
                    logger.info(f"Successfully cached result for {func_name} ({ticker}) to {cache_file}")
                except Exception as e:
                    logger.error(f"Error caching result for {func_name} ({ticker}): {e}")
                    logger.error(traceback.format_exc())
            
            return result
        
        return wrapper
    return decorator

def make_json_serializable(obj):
    """
    Recursively convert an object to be JSON serializable
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert anything else to string
        try:
            return str(obj)
        except:
            return f"<Unserializable object of type {type(obj).__name__}>" 