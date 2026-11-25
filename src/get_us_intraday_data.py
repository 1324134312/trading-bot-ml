"""
Get US Intraday Data - Much Easier Than UK!

This gets 60 days of 5-minute data from Yahoo Finance in one call!
"""
import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm

# Popular US stocks for day trading
US_STOCKS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Google
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "NVDA",   # NVIDIA
    "META",   # Facebook/Meta
    "AMD",    # AMD
    "INTC",   # Intel
    "SPY",    # S&P 500 ETF
    "QQQ",    # NASDAQ ETF
    "IWM",    # Russell 2000 ETF
    "DIA",    # Dow Jones ETF
]

def fetch_us_data(ticker, days=60):
    """
    Fetch US stock intraday data
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        days: Number of days (max 60 for Yahoo)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n📊 Fetching {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get 5-minute data (up to 60 days!)
        data = stock.history(period=f"{days}d", interval="5m")
        
        if data.empty:
            print(f"   ⚠️  No data available")
            return pd.DataFrame()
        
        data.reset_index(inplace=True)
        
        # Rename Date to Datetime
        if 'Date' in data.columns:
            data['Datetime'] = data['Date']
        
        data['Ticker'] = ticker
        
        print(f"   ✅ Fetched {len(data)} bars")
        print(f"   📅 Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        
        return data
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()

def main():
    """Main function"""
    
    print("=" * 70)
    print("🇺🇸 Getting US Intraday Data")
    print("=" * 70)
    print()
    print("📝 ADVANTAGES OF US DATA:")
    print("   ✅ 60 days available (vs UK's 7 days)")
    print("   ✅ Free from Yahoo Finance")
    print("   ✅ Easy to get")
    print("   ✅ More data for testing")
    print()
    print("⚠️  WARNING:")
    print("   US day trading requires $25k margin (PDT rule)")
    print("   Use this for testing only!")
    print()
    
    # Create output directory
    os.makedirs("data/us/intraday", exist_ok=True)
    
    results = []
    
    for ticker in tqdm(US_STOCKS, desc="Fetching US stocks"):
        data = fetch_us_data(ticker, days=60)
        
        if not data.empty:
            # Calculate days
            days = (pd.to_datetime(data['Datetime'].max()) - 
                   pd.to_datetime(data['Datetime'].min())).days
            
            # Save
            output_file = f"data/us/intraday/{ticker}_5m.csv"
            data.to_csv(output_file, index=False)
            
            results.append({
                'ticker': ticker,
                'bars': len(data),
                'days': days,
                'start': data['Datetime'].min(),
                'end': data['Datetime'].max()
            })
        
        # Small delay to avoid rate limiting
        import time
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
        print(f"📅 Date range: {df_results['start'].min()} to {df_results['end'].max()}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Process features: python src/process_us_features.py")
        print("   2. Train model: python src/train_us_model.py")
        print("   3. Test backtest: python src/test_us_backtest.py")
        print("\n⚠️  Remember: US day trading needs $25k!")
        print("   Use this for learning/testing, not live trading!")
        
        # Save summary
        df_results.to_csv("data/us/intraday/SUMMARY.csv", index=False)
    else:
        print("❌ No data fetched")

if __name__ == "__main__":
    main()

