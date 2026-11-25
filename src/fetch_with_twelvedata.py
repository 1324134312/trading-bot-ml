"""
Fetch UK Stock Data Using Twelve Data API

This gets MUCH more data than Yahoo (5,000 bars vs 714 bars per call!)
"""
import os
from twelvedata import TDClient
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# You'll need to install: pip install twelvedata
# You'll need to sign up at https://twelvedata.com/ to get an API key

def fetch_twelvedata():
    """Fetch UK stock data using Twelve Data API"""
    
    print("=" * 90)
    print("📊 Fetching UK Data via Twelve Data API")
    print("=" * 90)
    
    # Get API key from environment or user
    API_KEY = os.getenv('TWELVE_DATA_API_KEY')
    
    if not API_KEY:
        print("\n❌ API Key not found!")
        print("\n📝 To get started:")
        print("   1. Sign up at: https://twelvedata.com/")
        print("   2. Get your free API key")
        print("   3. Set environment variable:")
        print("      set TWELVE_DATA_API_KEY=your_key_here")
        print("   4. Or hardcode in this script")
        print("\n✅ Free tier: 8,000 calls/day, 5,000 bars per call!")
        return
    
    # Initialize client
    td = TDClient(apikey=API_KEY)
    
    # UK stocks
    UK_STOCKS = [
        "BP.L", "VOD.L", "BARC.L", "LLOY.L", "BHP.L", "RIO.L", 
        "GSK.L", "HSBA.L", "DGE.L", "TSCO.L", "ULVR.L", "AZN.L",
        "SHEL.L", "REL.L", "FERG.L", "GLEN.L", "CRDA.L", "ABDN.L"
    ]
    
    os.makedirs("data/uk/twelve_data", exist_ok=True)
    
    results = []
    
    for ticker in tqdm(UK_STOCKS, desc="Fetching stocks"):
        try:
            print(f"\n📊 Fetching {ticker}...")
            
            # Get 5-minute data (max 5,000 bars)
            ts = td.time_series(
                symbol=ticker,
                interval="5min",
                outputsize=5000  # Max bars per call!
            )
            
            # Convert to DataFrame
            df = ts.as_pandas()
            
            if not df.empty:
                df['Ticker'] = ticker
                
                # Save
                output_file = f"data/uk/twelve_data/{ticker}_5m.csv"
                df.to_csv(output_file, index=False)
                
                print(f"   ✅ Fetched {len(df)} bars")
                results.append({
                    'ticker': ticker,
                    'bars': len(df),
                    'date_range': f"{df.index.min()} to {df.index.max()}"
                })
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 90)
    print("📊 SUMMARY")
    print("=" * 90)
    
    if results:
        total_bars = sum(r['bars'] for r in results)
        print(f"\n✅ Fetched {len(results)} stocks")
        print(f"📊 Total bars: {total_bars:,}")
        print(f"\n💡 This is MUCH faster than Yahoo!")
        print(f"   Yahoo: ~714 bars per call")
        print(f"   Twelve Data: ~5,000 bars per call (7x more!)")
    
    print()

def main():
    """Main function"""
    
    # Check if API key is set
    API_KEY = os.getenv('TWELVE_DATA_API_KEY')
    
    if not API_KEY:
        print("=" * 90)
        print("🚀 Twelve Data Setup Required")
        print("=" * 90)
        print()
        print("To use Twelve Data API:")
        print()
        print("1. Sign up: https://twelvedata.com/")
        print("2. Get free API key")
        print("3. Install package: pip install twelvedata")
        print("4. Set API key:")
        print("   Windows: set TWELVE_DATA_API_KEY=your_key")
        print("   Linux/Mac: export TWELVE_DATA_API_KEY=your_key")
        print()
        print("OR hardcode in this script (not recommended)")
        print()
        print("✅ Free tier: 8,000 calls/day!")
        print()
    else:
        fetch_twelvedata()

if __name__ == "__main__":
    main()

