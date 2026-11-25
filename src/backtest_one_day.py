"""
Backtest UK Model - ONE DAY Only

This shows realistic intraday trading performance!
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def main():
    """Backtest on just ONE day of data"""
    
    print("=" * 90)
    print("🚀 ONE DAY BACKTEST")
    print("=" * 90)
    
    # Load UK data
    print("\n📂 Loading UK data...")
    files = glob.glob("data/uk/processed_5min/*_features.csv")
    
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('_features.csv', '')
        df['Ticker'] = ticker
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data_clean = data.dropna()
    
    print(f"✅ Total bars: {len(data_clean)}")
    
    # Filter to just ONE day
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    
    # Get the most recent day
    latest_date = data_clean['Datetime'].max()
    one_day_data = data_clean[data_clean['Datetime'].dt.date == latest_date.date()].copy()
    
    print(f"📅 Latest day: {latest_date.date()}")
    print(f"📊 Bars in single day: {len(one_day_data)}")
    print(f"📈 Unique stocks: {one_day_data['Ticker'].nunique()}")
    
    if len(one_day_data) < 10:
        print("❌ Not enough data for one day backtest")
        return
    
    # Backtest - ROBUST SETTINGS (less over-optimized)
    backtester = Backtester(
        model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
        initial_capital=10000,
        min_confidence=0.40,
        stop_loss_pct=0.005,  # 0.5% stop
        take_profit_pct=0.01,  # 1.0% target (more robust)
        max_positions=3  # Conservative (less overfitting)
    )
    
    results = backtester.run_backtest(one_day_data)
    
    # Display results
    print("\n" + "=" * 90)
    print("📊 ONE DAY BACKTEST RESULTS")
    print("=" * 90)
    
    print(f"\n💰 PROFIT (1 DAY):")
    print(f"   Starting: ${results['initial_capital']:,.2f}")
    print(f"   Ending: ${results['final_value']:,.2f}")
    print(f"   Profit: ${results['total_pnl']:,.2f}")
    print(f"   Return: {results['total_return_pct']:+.2f}%")
    
    if results['total_trades'] > 0:
        print(f"\n📈 TRADES:")
        print(f"   Total: {results['total_trades']}")
        print(f"   Winners: {results['winning_trades']}")
        print(f"   Losers: {results['losing_trades']}")
        print(f"   Win Rate: {results['win_rate']*100:.1f}%")
        
        print(f"\n💵 P&L PER TRADE:")
        print(f"   Avg Win: ${results['avg_win']:.2f}")
        print(f"   Avg Loss: ${results['avg_loss']:.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\n⏱️  HOLDING TIME:")
        print(f"   Avg: {results['avg_holding_bars']:.1f} bars ({results['avg_holding_bars']*5:.1f} minutes)")
    
    print(f"\n📊 RISK:")
    print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"   Max DD: {results['max_drawdown']*100:.2f}%")
    
    # Project to full trading day
    if len(one_day_data) > 0:
        bars_per_day = len(one_day_data)
        total_days = len(data_clean['Datetime'].dt.date.unique())
        
        print(f"\n📅 SCALING:")
        print(f"   Bars this day: {bars_per_day}")
        print(f"   Total days available: {total_days}")
        
        if results['total_pnl'] > 0:
            daily_return = results['total_return_pct']
            projected_month = daily_return * 20  # 20 trading days
            projected_year = daily_return * 252  # 252 trading days
            
            print(f"\n📈 PROJECTIONS (if this day repeats):")
            print(f"   Month (20 days): {projected_month:+.2f}%")
            print(f"   Year (252 days): {projected_year:+.2f}%")
    
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()

