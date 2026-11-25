"""
Fetch MAXIMUM 5-Minute Data for US Market

Get the most comprehensive dataset possible:
- 60 days per stock (Yahoo's maximum)
- Hundreds of stocks
- Popular tickers across all sectors
"""
import yfinance as yf
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta

# HUGE range of US stocks - all major sectors
COMPREHENSIVE_US_STOCKS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "INTC", "QCOM", "AVGO", "TXN", "ADI", "LRCX", "KLAC", "AMAT", "ASML",
    
    # Cloud & Enterprise
    "CRM", "ORCL", "NOW", "WDAY", "TEAM", "ZM", "DOCN", "NET", "DOCU",
    
    # Streaming & Media
    "NFLX", "DIS", "PARA", "WBD", "CMCSA", "SPOT",
    
    # E-commerce & Retail
    "SHOP", "ETSY", "EBAY", "ROKU", "AMZN",
    
    # Financials
    "JPM", "BAC", "WFC", "C", "MS", "GS", "BLK", "BK",
    "PNC", "TFC", "USB", "MTB", "SCHW",
    
    # Healthcare
    "JNJ", "PFE", "UNH", "ABT", "TMO", "DHR", "ISRG", "SYK",
    "EW", "ZBH", "BAX", "BSX", "MDT",
    
    # Pharmaceuticals
    "LLY", "MRNA", "GILD", "VRTX", "BIIB", "REGN", "AMGN",
    
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "TGT", "HD", "LOW",
    "NKE", "SBUX", "MCD",
    
    # Industrial
    "BA", "CAT", "DE", "GE", "HON", "LIN", "RTX", "UPS", "FDX",
    
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "OXY", "MPC",
    
    # Utilities
    "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "PEG",
    
    # Telecom
    "VZ", "T", "TMUS", "LUMN",
    
    # Airlines
    "AAL", "DAL", "UAL", "LUV", "JBLU",
    
    # Hotels
    "MAR", "HLT", "H",
    
    # Real Estate
    "AMT", "PLD", "PSA", "EQIX", "WELL", "VICI", "EXPI",
    
    # Gaming
    "TTWO", "EA", "ATVI", "U",
    
    # Cybersecurity
    "PANW", "CRWD", "ZS", "FTNT", "OKTA",
    
    # Electric Vehicles
    "TSLA", "RIVN", "LCID", "F", "GM",
    
    # Semiconductors
    "NVDA", "AMD", "INTC", "AVGO", "TXN", "QCOM", "MRVL",
    
    # Major ETFs
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "XLE",
    "XLF", "XLK", "XLI", "XLV", "XLY", "XLP", "XLB", "XLU",
    
    # FAANG+
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX",
    
    # Crypto-related
    "COIN", "MARA", "RIOT", "BITF",
    
    # DeFi
    "HOOD", "SOFI",
    
    # AI & Automation
    "PLTR", "PATH", "RPD",
    
    # Social Media
    "SNAP", "PINS", "TWTR",
    
    # Cloud Infrastructure
    "SNOW", "MDB", "ESTC", "JAMF",
    
    # Streaming
    "RBLX", "U",
    
    # Fintech
    "SQ", "PYPL", "AFRM", "UPST",
    
    # E-commerce Platforms
    "SHOP", "WMT", "TGT", "AMZN",
    
    # Next-Gen Tech
    "RKT", "FUBO", "FROG",
]

def fetch_stock_data(ticker, days=60):
    """
    Fetch maximum available data for a stock
    
    Args:
        ticker: Stock symbol
        days: Maximum days to fetch (60 is Yahoo's limit)
        
    Returns:
        DataFrame with data or None
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get 5-minute data (max 60 days)
        data = stock.history(period=f"{days}d", interval="5m")
        
        if data.empty:
            return None
        
        data.reset_index(inplace=True)
        
        # Rename Date to Datetime
        if 'Date' in data.columns:
            data['Datetime'] = data['Date']
        
        data['Ticker'] = ticker
        
        # Add fetch timestamp
        data['FetchDate'] = datetime.now()
        
        return data
        
    except Exception as e:
        # Silent fail - just continue
        return None

def main():
    """Main function"""
    
    print("=" * 90)
    print("🇺🇸 FETCHING MAXIMUM US MARKET DATA")
    print("=" * 90)
    print()
    print("📊 STRATEGY:")
    print("   ✅ Fetch 60 days of 5-minute data (maximum allowed)")
    print(f"   ✅ {len(COMPREHENSIVE_US_STOCKS)} stocks across all sectors")
    print("   ✅ Tech, finance, healthcare, energy, consumer, etc.")
    print("   ✅ All major ETFs included")
    print()
    print("⏰ ESTIMATED TIME: ~30-45 minutes")
    print("📦 ESTIMATED DATA: ~5-10M bars")
    print()
    
    # Create output directory
    os.makedirs("data/us/max_data", exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for ticker in tqdm(COMPREHENSIVE_US_STOCKS, desc="Fetching stocks"):
        data = fetch_stock_data(ticker, days=60)
        
        if data is not None:
            # Save to CSV
            output_file = f"data/us/max_data/{ticker}_5m_max.csv"
            data.to_csv(output_file, index=False)
            
            # Calculate days
            days_span = (pd.to_datetime(data['Datetime'].max()) - 
                        pd.to_datetime(data['Datetime'].min())).days
            
            results.append({
                'ticker': ticker,
                'bars': len(data),
                'days': days_span,
                'start': data['Datetime'].min(),
                'end': data['Datetime'].max()
            })
            
            successful += 1
        else:
            failed += 1
        
        # Rate limiting - be nice to Yahoo
        time.sleep(0.5)  # 500ms delay between calls
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 90)
    print("📊 FETCH COMPLETE")
    print("=" * 90)
    
    if results:
        df_results = pd.DataFrame(results)
        total_bars = df_results['bars'].sum()
        avg_bars = df_results['bars'].mean()
        avg_days = df_results['days'].mean()
        
        print(f"\n✅ SUCCESS:")
        print(f"   Stocks fetched: {successful}")
        print(f"   Total bars: {total_bars:,}")
        print(f"   Average bars per stock: {avg_bars:,.0f}")
        print(f"   Average days per stock: {avg_days:.1f}")
        print(f"   Data size: ~{total_bars * 0.5 / 1024:.1f} MB")
        print(f"   Time taken: {elapsed/60:.1f} minutes")
        
        if failed > 0:
            print(f"\n⚠️  FAILED: {failed} stocks")
        
        print(f"\n📅 DATE RANGE:")
        print(f"   From: {df_results['start'].min()}")
        print(f"   To: {df_results['end'].max()}")
        
        print(f"\n💾 DATA LOCATION:")
        print(f"   Directory: data/us/max_data/")
        print(f"   Format: {ticker}_5m_max.csv")
        
        print(f"\n🎯 THIS IS THE MAXIMUM IN-DEPTH DATASET!")
        print(f"   ✅ 60 days per stock (maximum possible)")
        print(f"   ✅ 5-minute bars (highest resolution)")
        print(f"   ✅ {successful} stocks (huge range)")
        print(f"   ✅ All sectors covered")
        
        # Save summary
        df_results.to_csv("data/us/max_data/SUMMARY.csv", index=False)
        
        print(f"\n💾 Summary saved: data/us/max_data/SUMMARY.csv")
        
    else:
        print("\n❌ No data fetched")
    
    print()

if __name__ == "__main__":
    main()

