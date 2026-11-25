"""
Fetch 1 Year of UK 5-Minute Data Using Sequential Batches
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# UK stocks
UK_TICKERS = [
    "BP.L", "VOD.L", "BARC.L", "LLOY.L", "BHP.L", "RIO.L", "GSK.L",
    "HSBA.L", "BT.L", "DGE.L", "TSCO.L", "ULVR.L", "AVST.L", "AZN.L",
    "SHEL.L", "REL.L", "FERG.L", "GLEN.L", "CRDA.L", "ABDN.L",
]

def fetch_year_of_data(ticker: str, target_days: int = 365) -> pd.DataFrame:
    """
    Fetch 1 year of 5-minute data by running multiple sequential fetches
    
    Strategy:
    - Fetch 7 days at a time (Yahoo limit)
    - Run this script daily for ~52 days (52 weeks × 7 days = 364 days)
    - Data accumulates in CSV files
    
    Args:
        ticker: Stock ticker
        target_days: Target number of days (default 365)
        
    Returns:
        Combined DataFrame
    """
    
    print(f"\n📊 Fetching {ticker}...")
    
    # File to store data
    data_file = f"data/uk/year_5min/{ticker}_1year.csv"
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    
    # Load existing data if it exists
    existing_data = []
    if os.path.exists(data_file):
        try:
            existing_data = pd.read_csv(data_file)
            existing_data['Datetime'] = pd.to_datetime(existing_data['Datetime'])
            print(f"   Loaded {len(existing_data)} existing bars")
        except:
            pass
    
    # Fetch fresh 7 days
    try:
        stock = yf.Ticker(ticker)
        new_data = stock.history(period="7d", interval="5m")
        
        if new_data.empty:
            print(f"   ⚠️  No new data available")
            return pd.DataFrame() if not existing_data else existing_data
        
        new_data.reset_index(inplace=True)
        
        # Rename to Datetime
        if 'Datetime' in new_data.columns:
            pass
        elif 'Date' in new_data.columns:
            new_data['Datetime'] = new_data['Date']
        
        new_data['Ticker'] = ticker
        
        print(f"   Fetched {len(new_data)} new bars")
        
        # Combine with existing
        if existing_data:
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            # Remove duplicates (keep first occurrence)
            combined = combined.drop_duplicates(subset=['Datetime'], keep='first')
            combined = combined.sort_values('Datetime').reset_index(drop=True)
        else:
            combined = new_data
        
        # Save
        combined.to_csv(data_file, index=False)
        
        # Calculate how many days we have
        if len(combined) > 0:
            days_covered = (pd.to_datetime(combined['Datetime'].max()) - 
                          pd.to_datetime(combined['Datetime'].min())).days
            print(f"   ✅ Total: {len(combined)} bars ({days_covered} days)")
        
        return combined
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return existing_data if len(existing_data) > 0 else pd.DataFrame()

def main():
    """Main function"""
    
    print("=" * 70)
    print("🇬🇧 Fetching 1 Year of UK 5-Minute Data")
    print("=" * 70)
    print("\n📝 STRATEGY:")
    print("   1. Fetch 7 days of data (Yahoo limit)")
    print("   2. Run this script DAILY for 52 weeks")
    print("   3. Data accumulates in CSV files")
    print("   4. After ~1 year, you'll have full historical data")
    print()
    print("⚠️  This is a DAILY fetcher - run it once per day!")
    print("   Use Task Scheduler / cron to automate")
    print()
    
    results = []
    
    for ticker in tqdm(UK_TICKERS, desc="Fetching"):
        data = fetch_year_of_data(ticker, target_days=365)
        
        if not data.empty:
            days = (pd.to_datetime(data['Datetime'].max()) - 
                   pd.to_datetime(data['Datetime'].min())).days
            
            results.append({
                'ticker': ticker,
                'bars': len(data),
                'days': days,
                'start': data['Datetime'].min(),
                'end': data['Datetime'].max()
            })
        
        # Rate limiting
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n✅ Successfully fetched: {len(results)} stocks")
        print(f"📊 Total bars: {df_results['bars'].sum():,}")
        print(f"📅 Average days per stock: {df_results['days'].mean():.1f}")
        print(f"📈 Date range: {df_results['start'].min()} to {df_results['end'].max()}")
        
        print("\n💡 TIP:")
        print("   Run this script DAILY to keep building your 1-year dataset")
        print("   After ~52 days, you'll have 1 full year of 5-minute data!")
        
        # Save summary
        df_results.to_csv("data/uk/year_5min/SUMMARY.csv", index=False)
    else:
        print("❌ No data fetched")
    
    print("\n⚙️  TO AUTOMATE:")
    print("   Windows: Task Scheduler (run daily at 5 PM UK time)")
    print("   Linux/Mac: crontab -e")
    print("   0 17 * * 1-5 cd /path/to/project && python src/fetch_1year_5min_data.py")

if __name__ == "__main__":
    main()

