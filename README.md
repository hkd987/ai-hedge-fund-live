# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

This system employs several agents working together:

1. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
2. Bill Ackman Agent - An activist investors, takes bold positions and pushes for change
3. Cathie Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
4. Charlie Munger Agent - Warren Buffett's partner, only buys wonderful businesses at fair prices
5. Paul Tudor Jones Agent - Technical trading legend known for swing trades based on price action and momentum
6. Stanley Druckenmiller Agent - Macro trading legend who hunts for asymmetric opportunities with explosive growth potential
7. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
8. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
9. Sentiment Agent - Analyzes market sentiment and generates trading signals
10. Fundamentals Agent - Analyzes fundamental data and generates trading signals
11. Technicals Agent - Analyzes technical indicators and generates trading signals
12. Risk Manager - Calculates risk metrics and sets position limits
13. Portfolio Manager - Makes final trading decisions and generates orders

<img width="1020" alt="Screenshot 2025-03-08 at 4 45 22 PM" src="https://github.com/user-attachments/assets/d8ab891e-a083-4fed-b514-ccc9322a3e57" />

**Note**: the system simulates trading decisions, it does not actually trade unless you enable live trading with Alpaca.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No warranties or guarantees provided
- Past performance does not indicate future results
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Running the Hedge Fund](#running-the-hedge-fund)
  - [Running the Backtester](#running-the-backtester)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Feature Requests](#feature-requests)
- [License](#license)

## Setup

Clone the repository:
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Set up your environment variables:
```bash
# Create .env file for your API keys
cp .env.example .env
```

4. Set your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
# Get your OpenAI API key from https://platform.openai.com/
OPENAI_API_KEY=your-openai-api-key

# For running LLMs hosted by groq (deepseek, llama3, etc.)
# Get your Groq API key from https://groq.com/
GROQ_API_KEY=your-groq-api-key

# For getting financial data to power the hedge fund
# Get your Financial Datasets API key from https://financialdatasets.ai/
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key

# For live trading with Alpaca
# Get your Alpaca API keys from https://app.alpaca.markets/
ALPACA_API_KEY=your-alpaca-api-key
ALPACA_API_SECRET=your-alpaca-api-secret
# Set to "true" to enable live trading (default is "false", which only simulates trades)
LIVE_TRADING=false
```

**Important**: You must set `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY` for the hedge fund to work.  If you want to use LLMs from all providers, you will need to set all API keys.

Financial data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key.

For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file.

For live trading, you will need to set `ALPACA_API_KEY` and `ALPACA_API_SECRET` in the .env file, and set `LIVE_TRADING=true`.

## Usage

### Running the Hedge Fund
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

You can also specify a `--show-reasoning` flag to print the reasoning of each agent to the console.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning
```
You can optionally specify the start and end dates to make decisions for a specific time period.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 
```

### Live Trading with Alpaca
To enable live trading with Alpaca, set the following in your `.env` file:
```
ALPACA_API_KEY=your-alpaca-api-key
ALPACA_API_SECRET=your-alpaca-api-secret
LIVE_TRADING=true
```

When live trading is enabled, the portfolio manager will execute trades through Alpaca based on the agent's decisions. By default, trades are executed on Alpaca's paper trading API, which allows you to test the system with simulated trades before risking real money.

You can also include all your current Alpaca holdings in the analysis using the `--include-alpaca-holdings` flag:
```bash
# Analyze specific tickers plus all current Alpaca holdings
poetry run python src/main.py --ticker AAPL,MSFT --include-alpaca-holdings

# Or analyze only your current Alpaca holdings without specifying additional tickers
poetry run python src/main.py --include-alpaca-holdings
```

This ensures the hedge fund makes decisions about all positions in your portfolio, not just the ones you manually specify.

### Risk Management Features

The AI hedge fund incorporates comprehensive risk management features to protect your capital:

#### Automated Stop Loss and Take Profit Orders
All buy and short orders automatically include:
- Stop loss orders (default: 5% from entry price)
- Take profit orders (default: 20% from entry price)

#### Position Sizing Controls
- Maximum position size limited to 10% of portfolio by default
- Maximum sector exposure limited to 25% of portfolio by default
- Automated size reduction during high market volatility

#### Circuit Breakers
- Daily loss circuit breaker (default: 3% portfolio loss in a day)
- Maximum drawdown protection (default: 10% from peak)
- Trading automatically suspended when circuit breakers trigger

#### Daily Trading Limits
- Maximum trades per day (default: 10)
- Maximum capital deployment per day (default: 20% of portfolio)

#### Emergency Controls
Use the emergency liquidation feature to immediately sell all positions:
```bash
poetry run python src/main.py --emergency-liquidate
```
This "panic button" requires confirmation and will sell all positions at market price.

#### Customizing Risk Parameters
You can customize all risk parameters in your `.env` file:
```
# Stop loss and take profit settings
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.20
TRAILING_STOP_PCT=0.03

# Position sizing constraints
MAX_POSITION_SIZE_PCT=0.10
MAX_SECTOR_EXPOSURE_PCT=0.25

# Daily trading limits
MAX_DAILY_TRADES=10
MAX_DAILY_CAPITAL_PCT=0.20

# Portfolio-wide risk controls
MAX_DRAWDOWN_PCT=0.10
DAILY_LOSS_CIRCUIT_BREAKER_PCT=0.03

# Market condition adjustments
HIGH_VOLATILITY_REDUCTION_PCT=0.50
HIGH_VOLATILITY_VIX_THRESHOLD=25
```

**IMPORTANT**: Before enabling live trading with real funds, thoroughly test the system with paper trading and understand the risks involved.

### Running the Backtester

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />

You can optionally specify the start and end dates to backtest over a specific time period.

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

Like with the main hedge fund mode, you can also include your current Alpaca holdings in the backtest:
```bash
# Backtest specific tickers plus all current Alpaca holdings
poetry run python src/backtester.py --ticker AAPL,MSFT --include-alpaca-holdings

# Or backtest only your current Alpaca holdings without specifying additional tickers
poetry run python src/backtester.py --include-alpaca-holdings
```

## Project Structure 
```
ai-hedge-fund/
├── src/
│   ├── agents/                   # Agent definitions and workflow
│   │   ├── bill_ackman.py        # Bill Ackman agent
│   │   ├── fundamentals.py       # Fundamental analysis agent
│   │   ├── portfolio_manager.py  # Portfolio management agent
│   │   ├── risk_manager.py       # Risk management agent
│   │   ├── sentiment.py          # Sentiment analysis agent
│   │   ├── technicals.py         # Technical analysis agent
│   │   ├── valuation.py          # Valuation analysis agent
│   │   ├── warren_buffett.py     # Warren Buffett agent
│   │   ├── ben_graham.py         # Ben Graham agent
│   │   ├── ...
│   ├── tools/                    # Agent tools
│   │   ├── api.py                # API tools
│   ├── backtester.py             # Backtesting tools
│   ├── main.py                   # Main entry point
├── pyproject.toml
├── ...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
