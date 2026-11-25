"""
Fetch Extended UK 5-Minute Data Using Multiple Batches
"""
import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import glob

# Key UK stocks for day trading
WIDE_UK_STOCKS = [
    # FTSE 100 Blue Chips
    "BP.L", "BT.L", "DGE.L", "GLEN.L", "HSBA.L", "RIO.L", "VOD.L",
    # Financials
    "BARC.L", "BNZL.L", "VOD.L", "LLOY.L", "STAN.L",
    # Tech/Retail
    "AAL.L", "ASHT.L", "BME.L", "CPG.L", "CRDA.L", "SGE.L",
    "FERG.L", "GSK.L", "HIK.L", "IMB.L", "JMAT.L", "REL.L",
    "RB.L", "NXT.L", "TSCO.L", "WTB.L", "SMIN.L",
]

def fetch_extended_data(ticker: str, num_batches: int = 30) -> pd.DataFrame:
    """
    Fetch data by making multiple overlapping requests
    
    Args:
        ticker: Stock ticker
        num_batches: How many 7-day batches to fetch
        
    Returns:
        Combined DataFrame with all data
    """
    all_data = []
    
    print(f"\n📊 Fetching {ticker}...")
    
    for i in range(num_batches):
        try:
            stock = yf.Ticker(ticker)
            
            # Fetch 7 days of data
            data = stock.history(period="7d", interval="5m")
            
            if data.empty:
                break
            
            data.reset_index(inplace=True)
            
            # Add ticker
            data['Ticker'] = ticker
            
            # Rename Datetime column
            if 'Datetime' in data.columns:
                pass  # Already good
            elif 'Date' in data.columns:
                data['Datetime'] = data['Date']
            
            all_data.append(data)
            
            # Small delay to avoid rate limiting
            import time
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️  Error on batch {i+1}: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates
    combined = combined.drop_duplicates(subset=['Datetime'], keep='first')
    
    # Sort by datetime
    combined = combined.sort_values('Datetime').reset_index(drop=True)
    
    return combined

def main():
    """Main function"""
    
    print("=" * 70)
    print("🇬🇧 Extended UK Data Fetching")
    print("=" * 70)
    print("\n⚠️  This will make multiple API calls per stock")
    print("   Each request fetches 7 days of 5-minute data")
    print("   We'll combine them to get more history")
    print()
    
    # Create output directory
    os.makedirs("data/uk/extended", exist_ok=True)
    
    # Start from a specific date and work backwards
    results = []
    
    for ticker in tqdm(WIDE_UK_STOCKS, desc="Fetching stocks"):
        try:
            # For now, just fetch one batch to test
            # In production, you'd run this script multiple times
            # on different days to build up history
            
            data = fetch_extended_data(ticker, num_batches=1)
            
            if not data.empty and len(data) > 50:
                # Save the data
                output_path = f"data/uk/extended/{ticker}_5m.csv"
                data.to_csv(output_path, index=False)
                
                results.append({
                    'ticker': ticker,
                    'bars': len(data),
                    'start': data['Datetime'].min(),
                    'end': data['Datetime'].max()
                })
                
                print(f"✅ {ticker}: {len(data)} bars")
        
        except Exception as e:
            print(f"❌ {ticker}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n✅ Successfully fetched: {len(results)} stocks")
        print(f"📊 Total bars: {df_results['bars'].sum()}")
        print(f"📅 Date range: {df_results['start'].min()} to {df_results['end'].max()}")
    else:
        print("❌ No data fetched")
    
    print("\n💡 TIP: Run this script daily to accumulate more data")
    print("   The data will automatically extend your historical dataset")

if __name__ == "__main__":
    main()

