"""
Fetch 5 years of historical data for better training
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.data_fetcher import DataFetcher
from config.settings import DEFAULT_TICKERS, RAW_DATA_DIR

def fetch_5_year_data():
    """Fetch 5 years of historical data"""
    
    print("=" * 70)
    print("📊 Fetching 5 Years of Historical Data")
    print("=" * 70)
    print()
    print("This will take longer but give us much better training data!")
    print("Fetching 5 years instead of 2 years...")
    print()
    
    # Initialize fetcher
    fetcher = DataFetcher(data_dir=RAW_DATA_DIR)
    
    # Fetch with longer period
    results = fetcher.fetch_multiple_tickers(
        tickers=DEFAULT_TICKERS,
        period="5y",  # 5 years instead of 2y
        interval="1d",
        save=True
    )
    
    print("\n✅ Data fetch complete!")
    print(f"📊 Successfully fetched {len(results)}/{len(DEFAULT_TICKERS)} tickers")
    print("\nNow run: python src/process_features.py")

if __name__ == "__main__":
    fetch_5_year_data()

