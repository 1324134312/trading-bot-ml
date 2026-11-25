"""
Configuration settings for the trading bot
"""
import os

# Data directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Default stocks to fetch
DEFAULT_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Google
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "META",   # Meta (Facebook)
    "NVDA",   # NVIDIA
    "SPY",    # S&P 500 ETF
    "QQQ",    # Nasdaq 100 ETF
]

# Data collection settings
DEFAULT_PERIOD = "5y"  # Fetch 5 years of historical data
DEFAULT_INTERVAL = "1d"  # Daily data
VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

# Data validation
MIN_DATA_POINTS = 100  # Minimum data points to consider dataset valid
MAX_MISSING_PERCENTAGE = 5  # Max 5% missing data allowed

