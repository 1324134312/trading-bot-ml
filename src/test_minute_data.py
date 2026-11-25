"""
Test if we can fetch minute-by-minute data and compare it to daily data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime

def test_minute_data():
    """Test minute-by-minute data fetching"""
    
    print("=" * 70)
    print("🔬 Testing Minute-by-Minute Data Feasibility")
    print("=" * 70)
    print()
    
    # Test with SPY (most liquid ETF)
    ticker = 'SPY'
    
    print(f"Testing with {ticker}...")
    
    # 1. Try to get recent minute data
    print("\n1️⃣ Testing: 1 day of minute data...")
    try:
        spy = yf.Ticker(ticker)
        data_1day = spy.history(period="1d", interval="1m")
        print(f"   ✅ Got {len(data_1day)} minutes of data")
        print(f"   📊 Sample: {data_1day.head(3)}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
    
    # 2. Try to get more data
    print("\n2️⃣ Testing: 5 days of minute data...")
    try:
        data_5days = spy.history(period="5d", interval="1m")
        print(f"   ✅ Got {len(data_5days)} minutes of data")
        print(f"   📅 Coverage: {data_5days.index.min()} to {data_5days.index.max()}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
    
    # 3. Try longer periods
    print("\n3️⃣ Testing: 1 month of minute data...")
    try:
        data_1mo = spy.history(period="1mo", interval="1m")
        print(f"   ✅ Got {len(data_1mo)} minutes of data")
        print(f"   📊 Rows: {len(data_1mo)}")
    except Exception as e:
        print(f"   ⚠️  {str(e)}")
    
    # 4. Calculate potential data volume
    print("\n" + "=" * 70)
    print("📊 Data Volume Analysis")
    print("=" * 70)
    
    # Market trading hours: 9:30 AM - 4:00 PM = 390 minutes per day
    minutes_per_day = 390
    trading_days_per_year = 252
    years = 5
    
    total_minutes = minutes_per_day * trading_days_per_year * years
    print(f"\n📈 Theoretical Data Volume:")
    print(f"   Minutes per trading day: {minutes_per_day}")
    print(f"   Trading days per year: {trading_days_per_year}")
    print(f"   Years: {years}")
    print(f"   Total minutes in 5 years: {total_minutes:,}")
    print(f"   For 9 stocks: {total_minutes * 9:,} data points!")
    
    # Compare to current daily data
    print(f"\n📉 Current Daily Data:")
    print(f"   Days in 5 years: {trading_days_per_year * years}")
    print(f"   For 9 stocks: {trading_days_per_year * years * 9} data points")
    
    print(f"\n📊 Ratio:")
    print(f"   Minute data: {total_minutes * 9:,} points")
    print(f"   Daily data: {trading_days_per_year * years * 9:,} points")
    print(f"   ~{total_minutes // (trading_days_per_year * years)}x MORE data!")
    
    # File size estimate
    bytes_per_row = 200  # approximate
    total_bytes = total_minutes * 9 * bytes_per_row
    print(f"\n💾 Estimated file size:")
    print(f"   Raw data: {total_bytes / 1e6:.1f} MB")
    print(f"   With features (67 features): {total_bytes * 67 / 1e6:.1f} MB")
    print(f"   Total with processed data: ~{total_bytes * 68 / 1e9:.2f} GB")
    
    return data_1day if 'data_1day' in locals() else None

def analyze_minute_data_quality(data):
    """Analyze if minute data is useful or just noisy"""
    
    if data is None or len(data) == 0:
        print("\n⚠️  Cannot analyze - no data")
        return
    
    print("\n" + "=" * 70)
    print("🔬 Analyzing Minute Data Quality")
    print("=" * 70)
    
    # Calculate volatility
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(10).std()
    
    print(f"\n📊 Statistics:")
    print(f"   Average return per minute: {data['Returns'].mean():.6f}")
    print(f"   Std dev: {data['Returns'].std():.6f}")
    print(f"   Min return: {data['Returns'].min():.4f}")
    print(f"   Max return: {data['Returns'].max():.4f}")
    
    # Count unique prices
    unique_prices = data['Close'].nunique()
    print(f"\n📈 Data Quality:")
    print(f"   Unique price levels: {unique_prices}/{len(data)}")
    print(f"   % Duplicate prices: {(1 - unique_prices/len(data))*100:.1f}%")
    
    # Check for gaps (non-trading periods)
    gaps = 0
    if len(data) > 1:
        time_diffs = data.index.to_series().diff()
        gaps = (time_diffs > pd.Timedelta(minutes=2)).sum()
    
    print(f"\n⏰ Data Gaps:")
    print(f"   Detected gaps: {gaps}")
    print(f"   % Complete: {(1 - gaps/len(data))*100:.1f}%")
    
    print(f"\n💡 Noise Analysis:")
    # Minute data is VERY noisy
    noise_level = data['Returns'].std()
    print(f"   Noise level (std): {noise_level:.6f}")
    print(f"   This is {'VERY HIGH' if noise_level > 0.001 else 'HIGH'} noise!")
    print(f"   For comparison, daily volatility is typically ~0.02 (2%)")
    print(f"   Minute volatility: {noise_level:.4%}")

if __name__ == "__main__":
    data = test_minute_data()
    if data is not None and len(data) > 0:
        analyze_minute_data_quality(data)
    
    print("\n" + "=" * 70)
    print("💡 Recommendation")
    print("=" * 70)
    print("""
Minute data pros and cons:

PROS:
✅ MASSIVE data volume (~1.7M points vs 11K)
✅ Could train stock-specific models
✅ See intraday patterns

CONS:
❌ VERY noisy (random minute-to-minute movements)
❌ Hard to download 5 years (rate limits)
❌ Large storage (GBs of data)
❌ Features need redesign for intraday
❌ Market close times, gaps, etc.
❌ Model might learn noise, not signals

RECOMMENDATION:
⚠️  Not worth it for ML trading

Better alternatives:
1. Use current 5-year daily data (already have)
2. Add 15-minute or hourly data (compromise)
3. Focus on ETFs where model works well (77%)
4. Use higher-confidence signals (94% accuracy)
    """)

