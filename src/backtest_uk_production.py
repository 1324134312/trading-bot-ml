"""
Backtest UK Model - Production Ready (Working Version)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def main():
    """Backtest UK model (the working one)"""
    
    print("=" * 90)
    print("🚀 PRODUCTION BACKTEST - UK MODEL")
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
    
    print(f"✅ Loaded {len(data_clean)} UK bars")
    
    # Backtest with UK model - ROBUST SETTINGS
    backtester = Backtester(
        model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
        initial_capital=10000,
        min_confidence=0.40,
        stop_loss_pct=0.005,  # 0.5% stop
        take_profit_pct=0.01,  # 1.0% target (more robust)
        max_positions=3  # Conservative (less overfitting)
    )
    
    results = backtester.run_backtest(data_clean)
    
    # Display results
    print("\n" + "=" * 90)
    print("📊 BACKTEST RESULTS - UK MODEL")
    print("=" * 90)
    
    print(f"\n💰 PROFIT:")
    print(f"   Initial: ${results['initial_capital']:,.2f}")
    print(f"   Final: ${results['final_value']:,.2f}")
    print(f"   Profit: ${results['total_pnl']:,.2f}")
    print(f"   Return: {results['total_return_pct']:+.2f}%")
    
    print(f"\n📈 TRADING:")
    print(f"   Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']*100:.1f}%")
    print(f"   Avg Win: ${results['avg_win']:.2f}")
    print(f"   Avg Loss: ${results['avg_loss']:.2f}")
    
    print(f"\n📊 RISK:")
    print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
    print(f"   Max DD: {results['max_drawdown']*100:.2f}%")
    
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()

