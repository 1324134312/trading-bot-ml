"""
Fetch UK Stock Data with Higher Timeframe Resolution
Attempting 5-minute data for recent period and daily for historical
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

UK_TICKERS = [
    "BP.L", "SHEL.L", "GSK.L", "AZN.L", "ULVR.L",
    "DGE.L", "RIO.L", "BHP.L", "VOD.L", "BT-A.L",
    # Add more FTSE 100 stocks
    "BARC.L",  # Barclays
    "LLOY.L",  # Lloyds Banking
    "NXT.L",   # Next
    "BAE.L",   # BAE Systems
    "AAL.L",   # Anglo American
    "PRU.L",   # Prudential
    "ANTO.L",  # Antofagasta
    "AHT.L",   # Ashtead Group
    "REL.L",   # RELX
    "SSE.L",   # SSE
]

def fetch_uk_5min_data():
    """Fetch 5-minute data for UK stocks"""
    
    print("=" * 70)
    print("🇬🇧 Fetching UK 5-Minute Intraday Data")
    print("=" * 70)
    print()
    print("Note: Yahoo Finance limits minute data to ~7-60 days")
    print("Will fetch maximum available and supplement with daily data")
    print()
    
    os.makedirs("data/uk/raw_intraday", exist_ok=True)
    
    results = {}
    
    for ticker in tqdm(UK_TICKERS, desc="Fetching"):
        try:
            print(f"\n📊 {ticker}")
            
            stock = yf.Ticker(ticker)
            
            # Try 5-minute data (limited period)
            print("   Trying 5-minute data...")
            try:
                data = stock.history(period="7d", interval="5m")
                if not data.empty and len(data) > 100:
                    print(f"   ✅ Got {len(data)} 5-minute bars")
                    
                    data.reset_index(inplace=True)
                    filename = f"data/uk/raw_intraday/{ticker}_5m_7d.csv"
                    data.to_csv(filename, index=False)
                    results[ticker] = {"5min": len(data), "period": "7d"}
                    continue
            except Exception as e:
                print(f"   ⚠️  5m data failed: {str(e)[:50]}")
            
            # Try 15-minute data
            print("   Trying 15-minute data...")
            try:
                data = stock.history(period="60d", interval="15m")
                if not data.empty:
                    print(f"   ✅ Got {len(data)} 15-minute bars")
                    
                    data.reset_index(inplace=True)
                    filename = f"data/uk/raw_intraday/{ticker}_15m_60d.csv"
                    data.to_csv(filename, index=False)
                    results[ticker] = {"15min": len(data), "period": "60d"}
                    continue
            except Exception as e:
                print(f"   ⚠️  15m data failed: {str(e)[:50]}")
            
            # Fall back to daily data
            print("   Fetching daily data as backup...")
            data = stock.history(period="5y", interval="1d")
            if not data.empty:
                print(f"   ✅ Got {len(data)} daily bars (5 years)")
                
                data.reset_index(inplace=True)
                filename = f"data/uk/raw_intraday/{ticker}_1d_5y.csv"
                data.to_csv(filename, index=False)
                results[ticker] = {"daily": len(data), "period": "5y"}
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
        
        time.sleep(1)
    
    print("\n" + "=" * 70)
    print("✅ Summary")
    print("=" * 70)
    
    for ticker, info in results.items():
        print(f"{ticker}: {info}")
    
    return results

def supplement_with_daily():
    """Supplement with 5 years of daily data"""
    
    print("\n" + "=" * 70)
    print("📊 Supplementing with 5 years of daily data")
    print("=" * 70)
    
    os.makedirs("data/uk/raw_daily", exist_ok=True)
    
    for ticker in tqdm(UK_TICKERS, desc="Daily data"):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="5y", interval="1d")
            
            if not data.empty:
                data.reset_index(inplace=True)
                filename = f"data/uk/raw_daily/{ticker}_1d_5y.csv"
                data.to_csv(filename, index=False)
                print(f"✅ {ticker}: {len(data)} days")
        except:
            pass
        time.sleep(1)

if __name__ == "__main__":
    fetch_uk_5min_data()
    supplement_with_daily()
    
    print("\n" + "=" * 70)
    print("🎯 Data Collection Complete!")
    print("=" * 70)
    print("\nNote on Yahoo Finance limits:")
    print("- 5-minute data: ~7 days max")
    print("- 15-minute data: ~60 days max")
    print("- Daily data: 5+ years (no limit)")
    print("\nFor true 5-year intraday coverage, consider:")
    print("1. Paid data provider (Bloomberg, Refinitiv)")
    print("2. API aggregator (Alpha Vantage, Polygon.io)")
    print("3. Broker data (Interactive Brokers, etc.)")

