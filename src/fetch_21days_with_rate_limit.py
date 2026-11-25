"""
Fetch 21 Days with Rate Limiting (8 calls per minute)
"""
from twelvedata import TDClient
import pandas as pd
import os
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Missing TWELVE_DATA_API_KEY. Copy .env.example to .env and set it, "
        "or export the variable in your shell."
    )

# Try without .L suffix for UK stocks
UK_STOCKS = [
    "BP", "VOD", "BARC", "LLOY", "BHP", "RIO", "GSK",
    "HSBA", "BT", "DGE", "TSCO", "ULVR", "AZN", "SHEL",
    "REL", "FERG", "GLEN", "CRDA", "ABDN"
]

def fetch_with_delays():
    """Fetch with rate limiting"""
    
    print("=" * 90)
    print("📊 Fetching UK Stock Data via Twelve Data")
    print("=" * 90)
    print()
    print("⚠️  Rate limit: 8 calls/minute (free tier)")
    print("   Adding 8-second delays between calls")
    print()
    
    td = TDClient(apikey=API_KEY)
    os.makedirs("data/uk/twelve_21days", exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    
    for i, ticker in enumerate(tqdm(UK_STOCKS, desc="Fetching")):
        try:
            print(f"\n📊 {ticker}...")
            
            # Try with .L suffix first
            symbol = f"{ticker}.L"
            
            # If that fails, try without
            try:
                ts = td.time_series(
                    symbol=symbol,
                    interval="5min",
                    outputsize=5000
                )
            except:
                # Try without suffix
                ts = td.time_series(
                    symbol=ticker,
                    interval="5min",
                    outputsize=5000
                )
                symbol = ticker
            
            df = ts.as_pandas()
            
            if not df.empty:
                df['Ticker'] = ticker
                df.reset_index(inplace=True)
                if 'datetime' in df.columns:
                    df.rename(columns={'datetime': 'Datetime'}, inplace=True)
                
                output_file = f"data/uk/twelve_21days/{ticker}_5m.csv"
                df.to_csv(output_file, index=False)
                
                print(f"   ✅ {len(df)} bars")
                results.append({
                    'ticker': ticker,
                    'bars': len(df),
                    'start': df['Datetime'].min() if 'Datetime' in df.columns else 'N/A',
                    'end': df['Datetime'].max() if 'Datetime' in df.columns else 'N/A'
                })
                successful += 1
            else:
                print(f"   ⚠️  No data")
                failed += 1
        
        except Exception as e:
            error_str = str(e)
            if "rate limit" in error_str.lower():
                print(f"   ⚠️  Rate limit hit, waiting...")
                time.sleep(70)  # Wait a bit over a minute
                # Retry
                continue
            else:
                print(f"   ❌ Error: {error_str[:100]}")
                failed += 1
        
        # Rate limiting: wait between calls
        if i < len(UK_STOCKS) - 1:  # Don't wait after last call
            time.sleep(8)  # 8 second delay (under 60/8 = 7.5 calls/minute limit)
    
    # Summary
    print("\n" + "=" * 90)
    print("📊 SUMMARY")
    print("=" * 90)
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n✅ Success: {successful} stocks")
        print(f"❌ Failed: {failed} stocks")
        print(f"📊 Total bars: {df_results['bars'].sum():,}")
        print(f"📊 Avg per stock: {df_results['bars'].mean():,.0f}")
        
        df_results.to_csv("data/uk/twelve_21days/SUMMARY.csv", index=False)
    
    print()

if __name__ == "__main__":
    fetch_with_delays()

