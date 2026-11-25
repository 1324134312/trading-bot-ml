"""
Calculate data volume with more stocks at 7-day limit
"""
import yfinance as yf

def calculate_scenario():
    """Calculate different scenarios"""
    
    print("=" * 70)
    print("📊 Data Volume with More Stocks (7-Day Limit)")
    print("=" * 70)
    
    # Current: 19 stocks × 7 days × ~100 bars/day
    current_stocks = 19
    bars_per_day_5min = 100  # Approximate
    days = 7
    
    current_bars = current_stocks * days * bars_per_day_5min
    print(f"\n📊 Current Setup:")
    print(f"   Stocks: {current_stocks}")
    print(f"   Period: {days} days")
    print(f"   Total bars: {current_bars:,}")
    
    # What if we get 50 stocks?
    scenario_50 = 50 * days * bars_per_day_5min
    print(f"\n📊 Scenario: 50 Stocks")
    print(f"   Total bars: {scenario_50:,}")
    print(f"   Increase: {scenario_50 / current_bars:.1f}x more data")
    
    # What if we get 100 stocks?
    scenario_100 = 100 * days * bars_per_day_5min
    print(f"\n📊 Scenario: 100 Stocks")
    print(f"   Total bars: {scenario_100:,}")
    print(f"   Increase: {scenario_100 / current_bars:.1f}x more data")
    
    # What if we get 200 stocks?
    scenario_200 = 200 * days * bars_per_day_5min
    print(f"\n📊 Scenario: 200 Stocks")
    print(f"   Total bars: {scenario_200:,}")
    print(f"   Increase: {scenario_200 / current_bars:.1f}x more data")
    
    # Compare to 5-year daily
    daily_5y = 19 * 1260 * 1
    print(f"\n📊 5-Year Daily Data (Current):")
    print(f"   Total bars: {daily_5y:,}")
    
    print(f"\n💡 Key Finding:")
    print(f"   With 100 stocks × 7 days × 100 bars/day = {scenario_100:,} bars")
    print(f"   vs 19 stocks × 1260 days × 1 bar/day = {daily_5y:,} bars")
    print(f"   Ratio: {scenario_100 / daily_5y:.1f}x more data points!")
    
    print(f"\n✅ Conclusion:")
    print(f"   Getting 100-200 stocks with 7-day limit")
    print(f"   Gives you MORE training data than 5 years of daily!")
    print(f"   This is a valid approach! 🎯")

if __name__ == "__main__":
    calculate_scenario()

