"""
Main entry point for the trading bot
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.data_fetcher import DataFetcher
from src.data_collection.data_validator import DataValidator
from config.settings import (
    DEFAULT_TICKERS,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
    RAW_DATA_DIR
)

def main():
    """Main function to run data collection"""
    
    print("=" * 60)
    print("🚀 Trading Bot ML - Data Collection")
    print("=" * 60)
    print()
    
    # Initialize components
    fetcher = DataFetcher(data_dir=RAW_DATA_DIR)
    validator = DataValidator()
    
    # Fetch data for default tickers
    print(f"📋 Fetching data for: {', '.join(DEFAULT_TICKERS)}\n")
    
    results = fetcher.fetch_multiple_tickers(
        tickers=DEFAULT_TICKERS,
        period=DEFAULT_PERIOD,
        interval=DEFAULT_INTERVAL,
        save=True
    )
    
    print("\n" + "=" * 60)
    print("🔍 Validating Data Quality")
    print("=" * 60)
    print()
    
    # Validate fetched data
    valid_tickers = []
    invalid_tickers = []
    
    for ticker, data in results.items():
        is_valid, errors = validator.validate_data(data, ticker)
        
        if is_valid:
            print(f"✅ {ticker}: Valid")
            valid_tickers.append(ticker)
        else:
            print(f"❌ {ticker}: Invalid")
            for error in errors:
                print(f"   - {error}")
            invalid_tickers.append(ticker)
    
    print()
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"✅ Valid datasets: {len(valid_tickers)}/{len(results)}")
    print(f"❌ Invalid datasets: {len(invalid_tickers)}/{len(results)}")
    
    if valid_tickers:
        print(f"\n🎯 Ready to proceed with: {', '.join(valid_tickers)}")
    else:
        print("\n⚠️  No valid data collected. Please check your internet connection and try again.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

