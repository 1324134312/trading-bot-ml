"""
Fetch 21 Days of UK Stock Data Using Twelve Data API

Gets ~5,000 bars per stock (vs Yahoo's 714!)
"""
from twelvedata import TDClient
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Twelve Data API key (from the environment)
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Missing TWELVE_DATA_API_KEY. Copy .env.example to .env and set it, "
        "or export the variable in your shell."
    )

# UK stocks to fetch
UK_STOCKS = [
    "BP.L", "VOD.L", "BARC.L", "LLOY.L", "BHP.L", "RIO.L", "GSK.L",
    "HSBA.L", "BT.L", "DGE.L", "TSCO.L", "ULVR.L", "AZN.L", "SHEL.L",
    "REL.L", "FERG.L", "GLEN.L", "CRDA.L", "ABDN.L", "BAE.L"
]

def fetch_with_twelvedata():
    """Fetch 21 days of UK stock data"""
    
    print("=" * 90)
    print("📊 Fetching 21 Days of UK Stock Data via Twelve Data")
    print("=" * 90)
    print()
    print(f"🎯 Target: 5,000 bars per stock (21 days of 5-minute data)")
    print(f"📅 Stocks: {len(UK_STOCKS)}")
    print()
    
    # Initialize client
    td = TDClient(apikey=API_KEY)
    
    # Create output directory
    os.makedirs("data/uk/twelve_21days", exist_ok=True)
    
    results = []
    
    for ticker in tqdm(UK_STOCKS, desc="Fetching stocks"):
        try:
            print(f"\n📊 Fetching {ticker}...")
            
            # Get 5-minute data with maximum outputsize
            ts = td.time_series(
                symbol=ticker,
                interval="5min",
                outputsize=5000  # Max bars (21+ days)
            )
            
            # Convert to DataFrame
            df = ts.as_pandas()
            
            if not df.empty:
                # Add ticker column
                df['Ticker'] = ticker
                
                # Reset index to make Datetime a column
                df.reset_index(inplace=True)
                
                # Rename datetime column
                if 'datetime' in df.columns:
                    df.rename(columns={'datetime': 'Datetime'}, inplace=True)
                
                # Save
                output_file = f"data/uk/twelve_21days/{ticker}_5m.csv"
                df.to_csv(output_file, index=False)
                
                print(f"   ✅ Fetched {len(df)} bars")
                print(f"   📅 Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
                
                results.append({
                    'ticker': ticker,
                    'bars': len(df),
                    'start': df['Datetime'].min(),
                    'end': df['Datetime'].max()
                })
            else:
                print(f"   ⚠️  No data returned")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 90)
    print("📊 SUMMARY")
    print("=" * 90)
    
    if results:
        df_results = pd.DataFrame(results)
        total_bars = df_results['bars'].sum()
        avg_bars = df_results['bars'].mean()
        
        print(f"\n✅ Successfully fetched: {len(results)} stocks")
        print(f"📊 Total bars: {total_bars:,}")
        print(f"📊 Average bars per stock: {avg_bars:,.0f}")
        print(f"📅 Date range: {df_results['start'].min()} to {df_results['end'].max()}")
        
        print(f"\n💡 Comparison with Yahoo:")
        print(f"   Yahoo: ~714 bars per stock (7 days)")
        print(f"   Twelve Data: {avg_bars:,.0f} bars per stock (21+ days)")
        print(f"   ✅ {avg_bars/714:.1f}x more data!")
        
        # Save summary
        df_results.to_csv("data/uk/twelve_21days/SUMMARY.csv", index=False)
    
    print()

if __name__ == "__main__":
    fetch_with_twelvedata()

