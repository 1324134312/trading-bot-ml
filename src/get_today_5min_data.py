"""Get 5-minute bars using the Twelve Data API."""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def get_today_5min_data(symbol="AAPL", api_key=None):
    """Get 5-minute data for today using Twelve Data API"""
    if not api_key:
        api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing TWELVE_DATA_API_KEY. Copy .env.example to .env and set it, "
            "or export the variable in your shell."
        )
    
    print(f"📊 Fetching 5-minute data for {symbol} today...")
    
    # Twelve Data API endpoint
    url = "https://api.twelvedata.com/time_series"
    
    params = {
        'symbol': symbol,
        'interval': '5min',
        'outputsize': 500,  # Get more data
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'values' in data:
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Convert price columns to float
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Sort by datetime
            df = df.sort_values('datetime')
            
            # Filter to today only
            today = datetime.now().date()
            df_today = df[df['datetime'].dt.date == today]
            
            print(f"✅ Got {len(df_today)} bars for today")
            
            if not df_today.empty:
                print(f"📅 Date range: {df_today['datetime'].min()} to {df_today['datetime'].max()}")
                print(f"💰 Price range: ${df_today['close'].min():.2f} - ${df_today['close'].max():.2f}")
                print(f"📈 Latest price: ${df_today['close'].iloc[-1]:.2f}")
                
                return df_today
            else:
                print("⚠️  No data for today (market closed or no trades)")
                return df
        else:
            print(f"❌ Error: {data}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def get_multiple_symbols_today(symbols=["AAPL", "MSFT", "TSLA", "GOOGL"]):
    """Get 5-minute data for multiple symbols today"""
    
    print("=" * 60)
    print("📊 Getting 5-Minute Data for Today")
    print("=" * 60)
    
    all_data = {}
    
    for symbol in symbols:
        print(f"\n📈 {symbol}:")
        data = get_today_5min_data(symbol)
        
        if data is not None:
            all_data[symbol] = data
            
            # Show summary
            if not data.empty:
                print(f"   Bars: {len(data)}")
                print(f"   Latest: ${data['close'].iloc[-1]:.2f}")
                print(f"   Volume: {data['volume'].iloc[-1]:,}")
        else:
            print(f"   ❌ Failed to get data")
    
    return all_data

def analyze_today_data(data_dict):
    """Analyze the data we got"""
    
    print("\n" + "=" * 60)
    print("📊 TODAY'S DATA ANALYSIS")
    print("=" * 60)
    
    for symbol, data in data_dict.items():
        if data is not None and not data.empty:
            print(f"\n📈 {symbol}:")
            print(f"   Total bars: {len(data)}")
            print(f"   Time range: {data['datetime'].min().strftime('%H:%M')} - {data['datetime'].max().strftime('%H:%M')}")
            print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print(f"   Latest: ${data['close'].iloc[-1]:.2f}")
            print(f"   Volume: {data['volume'].sum():,}")
            
            # Calculate some basic stats
            price_change = data['close'].iloc[-1] - data['close'].iloc[0]
            price_change_pct = (price_change / data['close'].iloc[0]) * 100
            
            print(f"   Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)")

if __name__ == "__main__":
    # Get data for multiple symbols
    symbols = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
    
    data = get_multiple_symbols_today(symbols)
    
    # Analyze the data
    analyze_today_data(data)
    
    print("\n" + "=" * 60)
    print("✅ Data collection complete!")
    print("=" * 60)
    
    # Show how to use in your bot
    print("\n📚 How to use this data in your bot:")
    print("1. Run this script daily")
    print("2. Feed data to your model")
    print("3. Get predictions")
    print("4. Track performance")
    print("\n🚀 Real-time data for FREE!")

