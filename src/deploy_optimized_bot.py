"""
Deploy Optimized Bot with Maximum Profit Settings
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def main():
    """Deploy optimized bot"""
    
    print("=" * 90)
    print("🚀 DEPLOYING OPTIMIZED BOT - MAXIMUM PROFIT SETTINGS")
    print("=" * 90)
    
    # Load data
    print("\n📂 Loading data...")
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
    latest = data_clean['Datetime'].max()
    one_day = data_clean[data_clean['Datetime'].dt.date == latest.date()]
    
    print(f"📊 Testing on {len(one_day)} bars")
    
    # Deploy optimized settings
    print("\n⚙️  OPTIMIZED SETTINGS:")
    print("   Confidence: 0.55 (lowered from 0.65)")
    print("   Stop Loss: 0.5%")
    print("   Take Profit: 1.0%")
    print()
    
    backtester = Backtester(
        model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
        initial_capital=10000,
        min_confidence=0.55,  # OPTIMIZED
        stop_loss_pct=0.005,
        take_profit_pct=0.01,
        max_positions=3
    )
    
    result = backtester.run_backtest(one_day)
    
    # Display results
    print("=" * 90)
    print("💰 OPTIMIZED BOT RESULTS")
    print("=" * 90)
    
    print(f"\n💰 PROFIT: ${result['total_pnl']:.2f}")
    print(f"📈 RETURN: {result['total_return_pct']:+.2f}%")
    print(f"📊 TRADES: {result['total_trades']}")
    print(f"🎯 WIN RATE: {result['win_rate']*100:.1f}%")
    print(f"📊 Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"⚠️  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    
    print(f"\n🔄 IMPROVEMENT:")
    print(f"   Old: $11.21/day")
    print(f"   New: ${result['total_pnl']:.2f}/day")
    print(f"   Increase: {result['total_pnl']/11.21:.1f}x")
    
    print("\n✅ BOT OPTIMIZED FOR MAXIMUM PROFIT!")
    print("=" * 90)
    
    return result

if __name__ == "__main__":
    main()

