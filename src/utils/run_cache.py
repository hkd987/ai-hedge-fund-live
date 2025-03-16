import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

# Set up logging
logger = logging.getLogger('run_cache')

# Cache directories
CACHE_DIR = Path("src/cache")
ANALYST_CACHE_DIR = CACHE_DIR / "analysts"
RUN_HISTORY_FILE = CACHE_DIR / "run_history.json"

# Cache time settings (in minutes)
DEFAULT_CACHE_TTL = 60  # 60 minutes = 1 hour

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ANALYST_CACHE_DIR, exist_ok=True)

def save_run_history(tickers, timestamp=None):
    """
    Save the current run to the run history file.
    
    Args:
        tickers: List of tickers for this run
        timestamp: Optional timestamp for this run (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Create a run history entry
    run_entry = {
        "timestamp": timestamp,
        "tickers": tickers
    }
    
    # Load existing run history
    run_history = []
    if os.path.exists(RUN_HISTORY_FILE):
        try:
            with open(RUN_HISTORY_FILE, 'r') as f:
                run_history = json.load(f)
                
                # Ensure run_history is a list
                if not isinstance(run_history, list):
                    logger.warning(f"Run history file has unexpected format. Resetting.")
                    run_history = []
        except Exception as e:
            logger.error(f"Error reading run history: {e}")
            run_history = []
    
    # Add the new run to history
    run_history.append(run_entry)
    
    # Keep only the last 100 runs to avoid the file growing too large
    if len(run_history) > 100:
        run_history = run_history[-100:]
    
    # Save the updated run history
    try:
        os.makedirs(os.path.dirname(RUN_HISTORY_FILE), exist_ok=True)
        with open(RUN_HISTORY_FILE, 'w') as f:
            json.dump(run_history, f, indent=2)
        logger.info(f"Saved run history with {len(run_history)} entries")
    except Exception as e:
        logger.error(f"Error saving run history: {e}")

def get_last_run_time(tickers=None):
    """
    Get the timestamp of the last run for the given tickers.
    
    Args:
        tickers: Optional list of tickers to find the last run for
        
    Returns:
        datetime: The timestamp of the last run, or None if no runs found
    """
    if not os.path.exists(RUN_HISTORY_FILE):
        logger.info("No run history file found")
        return None
    
    try:
        with open(RUN_HISTORY_FILE, 'r') as f:
            run_history = json.load(f)
            
        if not run_history:
            logger.info("Run history is empty")
            return None
        
        # If tickers are specified, find the latest run that contains all of them
        if tickers:
            # Convert to set for efficient lookups
            ticker_set = set(tickers)
            
            # Go through run history in reverse order (newest first)
            for run in reversed(run_history):
                if "tickers" in run and set(run["tickers"]).issuperset(ticker_set):
                    # This run contains all the requested tickers
                    timestamp = run.get("timestamp")
                    if timestamp:
                        try:
                            return datetime.fromisoformat(timestamp)
                        except Exception as e:
                            logger.error(f"Error parsing timestamp {timestamp}: {e}")
                            continue
            
            # No matching run found
            logger.info(f"No previous run found for tickers: {tickers}")
            return None
        else:
            # No specific tickers, just return the timestamp of the latest run
            if run_history and "timestamp" in run_history[-1]:
                timestamp = run_history[-1]["timestamp"]
                try:
                    return datetime.fromisoformat(timestamp)
                except Exception as e:
                    logger.error(f"Error parsing timestamp {timestamp}: {e}")
                    return None
            else:
                logger.warning("Latest run has no timestamp")
                return None
    except Exception as e:
        logger.error(f"Error getting last run time: {e}")
        return None

def should_use_cached_data(tickers=None, ttl_minutes=DEFAULT_CACHE_TTL):
    """
    Determine if cached data should be used based on the time since the last run.
    
    Args:
        tickers: Optional list of tickers to check
        ttl_minutes: Time-to-live in minutes for cached data
        
    Returns:
        bool: True if cached data should be used, False otherwise
    """
    # Get the last run time
    last_run = get_last_run_time(tickers)
    
    if not last_run:
        logger.info("No previous run found, using fresh data")
        return False
    
    # Check if the cache has expired
    now = datetime.now()
    time_since_last_run = now - last_run
    minutes_since_last_run = time_since_last_run.total_seconds() / 60
    
    # Log the time since the last run
    logger.info(f"Time since last run: {minutes_since_last_run:.1f} minutes (TTL: {ttl_minutes} minutes)")
    
    # Check if we have any cache files for these tickers
    if tickers and not list(ANALYST_CACHE_DIR.glob("*")):
        logger.info("No cache files found, using fresh data")
        return False
    
    # Determine if cache should be used
    use_cache = minutes_since_last_run < ttl_minutes
    
    if use_cache:
        logger.info(f"Using cached data (last run was {minutes_since_last_run:.1f} minutes ago)")
    else:
        logger.info(f"Cache expired (last run was {minutes_since_last_run:.1f} minutes ago)")
    
    return use_cache

def save_analyst_data(agent_name, tickers, data):
    """
    Save analyst data to the cache.
    
    Args:
        agent_name: Name of the analyst agent
        tickers: List of tickers the data is for
        data: Data to save
    """
    # Ensure cache directory exists
    os.makedirs(ANALYST_CACHE_DIR, exist_ok=True)
    
    # Use first ticker for the filename
    if not tickers:
        logger.warning(f"No tickers provided for {agent_name}, cannot cache")
        return
    
    ticker = tickers[0]
    
    # Normalize agent name by removing any "agent" suffix for consistency
    normalized_agent_name = agent_name
    if normalized_agent_name.endswith("_agent"):
        normalized_agent_name = normalized_agent_name.replace("_agent", "")
    
    # Create cache filename 
    cache_file = ANALYST_CACHE_DIR / f"{normalized_agent_name}_{ticker}.json"
    
    try:
        # Ensure data is serializable
        serializable_data = data
        
        # Handle Pydantic models
        if hasattr(data, "model_dump"):
            serializable_data = data.model_dump()
        elif hasattr(data, "dict"):
            serializable_data = data.dict()
        elif hasattr(data, "__dict__"):
            serializable_data = data.__dict__
        
        # If this is a result dict with analyst signals, extract just this agent's data
        if isinstance(serializable_data, dict):
            # Check common patterns
            if "data" in serializable_data and "analyst_signals" in serializable_data["data"]:
                analyst_signals = serializable_data["data"]["analyst_signals"]
                if agent_name in analyst_signals:
                    serializable_data = analyst_signals[agent_name]
                    logger.info(f"Extracted {agent_name} data from analyst_signals")
                elif normalized_agent_name in analyst_signals:
                    serializable_data = analyst_signals[normalized_agent_name]
                    logger.info(f"Extracted {normalized_agent_name} data from analyst_signals")
        
        # Add metadata
        if isinstance(serializable_data, dict):
            serializable_data["cache_timestamp"] = datetime.now().isoformat()
            serializable_data["cache_tickers"] = tickers
            serializable_data["cache_agent"] = normalized_agent_name
        
        # Save to file
        with open(cache_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved cached data for {agent_name} ({ticker}) to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cached data for {agent_name} ({ticker}): {e}")

def load_analyst_data(agent_name, tickers):
    """
    Load analyst data from the cache.
    
    Args:
        agent_name: Name of the analyst agent
        tickers: List of tickers the data is for
        
    Returns:
        The cached data, or None if not found
    """
    # Check if tickers is empty
    if not tickers:
        logger.warning(f"No tickers provided for {agent_name}, cannot load cache")
        return None
    
    # Use first ticker for the filename
    ticker = tickers[0]
    
    # Normalize agent name by removing any "agent" suffix for consistency
    normalized_agent_name = agent_name
    if normalized_agent_name.endswith("_agent"):
        normalized_agent_name = normalized_agent_name.replace("_agent", "")
    
    # Create cache filename
    cache_file = ANALYST_CACHE_DIR / f"{normalized_agent_name}_{ticker}.json"
    
    # Check if cache file exists
    if not os.path.exists(cache_file):
        logger.info(f"No cache file found for {agent_name} ({ticker})")
        # Try alternative name formats
        alt_cache_file = ANALYST_CACHE_DIR / f"{agent_name}_{ticker}.json"
        if not os.path.exists(alt_cache_file):
            return None
        else:
            cache_file = alt_cache_file
            logger.info(f"Found cache file with alternate name: {cache_file}")
    
    try:
        # Load cached data
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Check cache timestamp
        if "cache_timestamp" in cached_data:
            try:
                cache_time = datetime.fromisoformat(cached_data["cache_timestamp"])
                time_since_cache = datetime.now() - cache_time
                minutes_ago = time_since_cache.total_seconds() / 60
                logger.info(f"Loaded cached data for {agent_name} ({ticker}) from {minutes_ago:.1f} minutes ago")
            except Exception as e:
                logger.warning(f"Could not parse cache timestamp: {e}")
        
        # Format the result for agent consumption
        # In some cases, agents expect their results in a specific format
        if agent_name.endswith("_agent"):
            # Many agents expect their data wrapped in the analyst_signals structure
            try:
                # Check if the cached data already has the right structure
                if "analyst_signals" in cached_data:
                    return cached_data
                
                # Construct a result object that mimics the expected state structure
                result = {
                    "messages": [],
                    "data": {
                        "tickers": tickers,
                        "analyst_signals": {
                            agent_name: cached_data
                        }
                    }
                }
                logger.info(f"Wrapped cached data in expected state format for {agent_name}")
                return result
            except Exception as e:
                logger.error(f"Error formatting cached data for {agent_name}: {e}")
                # Fall back to returning the raw cached data
                return cached_data
        
        return cached_data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in cache file for {agent_name} ({ticker}): {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading cached data for {agent_name} ({ticker}): {e}")
        return None

def clear_cache():
    """
    Clear all cached data.
    """
    try:
        # Check if the directory exists
        if os.path.exists(ANALYST_CACHE_DIR):
            # Delete all .json files in the analysts directory
            for cache_file in ANALYST_CACHE_DIR.glob("*.json"):
                try:
                    os.remove(cache_file)
                    logger.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logger.error(f"Error removing cache file {cache_file}: {e}")
            
            logger.info("All analyst cache files have been cleared")
        else:
            logger.info(f"Analyst cache directory does not exist, creating it: {ANALYST_CACHE_DIR}")
            os.makedirs(ANALYST_CACHE_DIR, exist_ok=True)
        
        # Clear the run history file but maintain the file itself with an empty list
        if os.path.exists(RUN_HISTORY_FILE):
            with open(RUN_HISTORY_FILE, 'w') as f:
                json.dump([], f)
            logger.info("Run history has been cleared")
        else:
            # Create an empty run history file
            os.makedirs(os.path.dirname(RUN_HISTORY_FILE), exist_ok=True)
            with open(RUN_HISTORY_FILE, 'w') as f:
                json.dump([], f)
            logger.info("Created empty run history file")
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}") 