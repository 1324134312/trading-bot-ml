"""
Test Multiple Settings to Find Maximum Profit

Tests combinations of:
- Confidence thresholds
- Stop/target ratios
- Max positions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_settings(confidence, stop_pct, target_pct, data):
    """Test one configuration"""
    
    backtester = Backtester(
        model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
        initial_capital=10000,
        min_confidence=confidence,
        stop_loss_pct=stop_pct,
        take_profit_pct=target_pct,
        max_positions=3
    )
    
    result = backtester.run_backtest(data)
    
    return {
        'confidence': confidence,
        'stop_pct': stop_pct,
        'target_pct': target_pct,
        'profit': result['total_pnl'],
        'return_pct': result['total_return_pct'],
        'trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'sharpe': result['sharpe_ratio'],
        'max_dd': result['max_drawdown']
    }

def main():
    """Test multiple configurations"""
    
    print("=" * 90)
    print("🧪 TESTING MULTIPLE SETTINGS FOR MAXIMUM PROFIT")
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
    
    # Filter to one day for faster testing
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    latest = data_clean['Datetime'].max()
    one_day = data_clean[data_clean['Datetime'].dt.date == latest.date()]
    
    print(f"📊 Testing on {len(one_day)} bars from {one_day['Datetime'].min()}")
    
    # Test configurations
    print("\n" + "=" * 90)
    print("🧪 Testing Configurations...")
    print("=" * 90)
    
    configs = [
        # (confidence, stop, target, name)
        (0.65, 0.005, 0.01, "Baseline"),
        (0.60, 0.005, 0.01, "Lower Conf"),
        (0.55, 0.005, 0.01, "Very Low Conf"),
        (0.65, 0.003, 0.01, "Tight Stop"),
        (0.65, 0.005, 0.015, "Wide Target"),
        (0.65, 0.003, 0.015, "Tight-Wide"),
        (0.60, 0.004, 0.012, "Balanced 1"),
        (0.70, 0.005, 0.01, "High Conf"),
    ]
    
    results = []
    
    for conf, stop, target, name in configs:
        print(f"\n📊 Testing {name} (conf={conf}, stop={stop*100}%, target={target*100}%)")
        
        try:
            result = test_settings(conf, stop, target, one_day)
            result['name'] = name
            results.append(result)
            
            print(f"   ✅ Profit: ${result['profit']:.2f} ({result['return_pct']:+.2f}%)")
            print(f"   Trades: {result['trades']}, Win Rate: {result['win_rate']*100:.1f}%")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Find best
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('profit', ascending=False)
        
        best = df_results.iloc[0]
        
        print("\n" + "=" * 90)
        print("🏆 WINNING CONFIGURATION")
        print("=" * 90)
        print(f"\n🥇 Best: {best['name']}")
        print(f"   Confidence: {best['confidence']}")
        print(f"   Stop: {best['stop_pct']*100}%")
        print(f"   Target: {best['target_pct']*100}%")
        print(f"\n💰 PROFIT: ${best['profit']:.2f} ({best['return_pct']:+.2f}%)")
        print(f"📊 Trades: {best['trades']}")
        print(f"🎯 Win Rate: {best['win_rate']*100:.1f}%")
        print(f"📈 Sharpe: {best['sharpe']:.2f}")
        print(f"⚠️  Max DD: {best['max_dd']*100:.2f}%")
        
        print(f"\n📊 All Results:")
        print(df_results[['name', 'profit', 'return_pct', 'trades', 'win_rate']].head(10).to_string())
        
        print("\n✅ OPTIMIZE BOT WITH THESE SETTINGS!")
        print("=" * 90)

if __name__ == "__main__":
    main()

