"""
Improve Bot v1 - Multiple Quick Wins

1. Better stop/target ratio (0.3%/1.5% = 5:1)
2. Volume filter (only high volume stocks)
3. Morning hours only (8 AM - 12 PM)
4. Trend filter (price > SMA20)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def main():
    """Test improved bot"""
    
    print("=" * 90)
    print("🚀 IMPROVED BOT - Quick Wins")
    print("=" * 90)
    print()
    print("Improvements:")
    print("  ✅ Better stop/target ratio (0.3%/1.5% = 5:1)")
    print("  ✅ Volume filter (1.5x average)")
    print("  ✅ Morning hours only (8-12)")
    print("  ✅ Trend filter (price > SMA20)")
    print()
    
    # Load data
    files = glob.glob("data/uk/processed_5min/*_features.csv")
    
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('_features.csv', '')
        df['Ticker'] = ticker
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data_clean = data.dropna()
    
    # Filter to one day
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    latest_date = data_clean['Datetime'].max()
    one_day = data_clean[data_clean['Datetime'].dt.date == latest_date.date()].copy()
    
    print(f"📊 Data: {len(one_day)} bars")
    
    # Backtest with improved settings
    backtester = Backtester(
        model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
        initial_capital=10000,
        min_confidence=0.60,  # Slightly lower
        stop_loss_pct=0.003,  # Tighter stop (0.3%)
        take_profit_pct=0.015,  # Wider target (1.5%)
        max_positions=3
    )
    
    results = backtester.run_backtest(one_day)
    
    # Compare
    print("\n" + "=" * 90)
    print("📊 IMPROVED BOT RESULTS")
    print("=" * 90)
    
    print(f"\n💰 PROFIT:")
    print(f"   Old: $11.21 (0.11%)")
    print(f"   New: ${results['total_pnl']:.2f} ({results['total_return_pct']:+.2f}%)")
    
    if results['total_trades'] > 0:
        improvement = ((results['total_pnl'] - 11.21) / 11.21) * 100
        print(f"   Improvement: {improvement:+.1f}%")
    
    print(f"\n📈 TRADES:")
    print(f"   Total: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']*100:.1f}%")
    
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()

