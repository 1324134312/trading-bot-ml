"""
UK Stock Data Collector - Intraday Day Trading Data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from typing import List, Optional
from tqdm import tqdm

# UK stock config
UK_TICKERS = [
    "BP.L", "SHEL.L", "GSK.L", "AZN.L", "ULVR.L",
    "DGE.L", "RIO.L", "BHP.L", "VOD.L", "BT-A.L",
]

class UKDataCollector:
    """Collect UK stock data for day trading"""
    
    def __init__(self, data_dir: str = "data/uk/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_uk_intraday_data(
        self, 
        ticker: str, 
        interval: str = "1h",
        period: str = "60d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for UK stock
        
        Args:
            ticker: UK ticker (e.g., 'BP.L')
            interval: '1h', '15m', '1d'
            period: '60d' (about 60 days)
        """
        try:
            print(f"\n📊 Fetching {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️  No data for {ticker}")
                return None
            
            # Add metadata
            data.reset_index(inplace=True)
            data['Ticker'] = ticker
            data['FetchDate'] = datetime.now()
            
            # Ensure Date column exists
            if 'Datetime' in data.columns and 'Date' not in data.columns:
                data['Date'] = data['Datetime']
            elif 'Date' not in data.columns and data.index.name:
                data['Date'] = data.index
            
            print(f"✅ Got {len(data)} bars for {ticker}")
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {str(e)}")
            return None
    
    def fetch_all_uk_stocks(self, tickers: List[str] = None):
        """Fetch data for all UK stocks"""
        
        if tickers is None:
            tickers = UK_TICKERS
        
        print("=" * 70)
        print("🇬🇧 Fetching UK Stock Data for Day Trading")
        print("=" * 70)
        print(f"Interval: 1-hour bars")
        print(f"Period: 60 days (~1500 data points)")
        print(f"Tickers: {len(tickers)}")
        print()
        
        results = {}
        
        for ticker in tickers:
            data = self.fetch_uk_intraday_data(ticker, interval="1h", period="60d")
            
            if data is not None:
                # Save
                filename = f"{ticker}_1h_60d.csv"
                filepath = os.path.join(self.data_dir, filename)
                data.to_csv(filepath, index=False)
                print(f"💾 Saved: {filepath}")
                results[ticker] = data
            
            # Small delay
            time.sleep(1)
        
        print(f"\n✅ Successfully fetched {len(results)}/{len(tickers)} stocks")
        return results

def main():
    collector = UKDataCollector()
    results = collector.fetch_all_uk_stocks()
    
    if results:
        print("\n🎯 Next steps:")
        print("   1. Process intraday features")
        print("   2. Create day trading targets (2-hour predictions)")
        print("   3. Train UK day trading model")

if __name__ == "__main__":
    main()

