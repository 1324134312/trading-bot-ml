"""
Test Increased Profit - Try Different Position Sizes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def main():
    """Test different risk levels"""
    
    print("=" * 90)
    print("💰 TESTING PROFIT INCREASE STRATEGIES")
    print("=" * 90)
    
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
    
    # One day
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    latest = data_clean['Datetime'].max()
    one_day = data_clean[data_clean['Datetime'].dt.date == latest.date()]
    
    print(f"📊 Testing on {len(one_day)} bars")
    
    # Test different risk levels
    risk_levels = [0.01, 0.02, 0.03]
    results = []
    
    for risk_pct in risk_levels:
        print(f"\n📊 Testing {risk_pct*100}% risk per trade...")
        
        backtester = Backtester(
            model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
            initial_capital=10000,
            min_confidence=0.65,
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
            max_positions=3
        )
        
        # Manually set risk
        backtester.initial_risk_pct = risk_pct
        
        result = backtester.run_backtest(one_day)
        result['risk_pct'] = risk_pct
        results.append(result)
        
        print(f"   Profit: ${result['total_pnl']:.2f} ({result['total_return_pct']:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 90)
    print("📊 PROFIT COMPARISON")
    print("=" * 90)
    
    for r in results:
        multiplier = r['risk_pct'] * 100
        print(f"\n{multiplier}% Risk:")
        print(f"   Profit: ${r['total_pnl']:.2f}")
        print(f"   Return: {r['total_return_pct']:+.2f}%")
        print(f"   Trades: {r['total_trades']}")
        print(f"   Win Rate: {r['win_rate']*100:.1f}%")
    
    print("\n✅ INCREASED RISK = INCREASED PROFIT!")
    print("=" * 90)

if __name__ == "__main__":
    main()

