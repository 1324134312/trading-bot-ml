"""
Comprehensive Settings Test - Find the Optimal Configuration

Tests combinations of:
- Confidence thresholds (0.30 - 0.70)
- Stop loss percentages
- Take profit percentages
- Max positions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_config(confidence, stop_pct, target_pct, max_pos, data):
    """Test one configuration"""
    
    try:
        backtester = Backtester(
            model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
            initial_capital=10000,
            min_confidence=confidence,
            stop_loss_pct=stop_pct,
            take_profit_pct=target_pct,
            max_positions=max_pos
        )
        
        result = backtester.run_backtest(data)
        
        return {
            'confidence': confidence,
            'stop_pct': stop_pct,
            'target_pct': target_pct,
            'max_pos': max_pos,
            'profit': result['total_pnl'],
            'return': result['total_return_pct'],
            'trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'sharpe': result['sharpe_ratio'],
            'profit_factor': result['profit_factor'],
            'max_dd': result['max_drawdown'],
            'avg_win': result['avg_win'],
            'avg_loss': result['avg_loss']
        }
    except Exception as e:
        return None

def main():
    """Test comprehensive configuration space"""
    
    print("=" * 90)
    print("🔬 COMPREHENSIVE SETTINGS TEST - FINDING OPTIMAL CONFIG")
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
    
    # One day for speed
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    latest = data_clean['Datetime'].max()
    one_day = data_clean[data_clean['Datetime'].dt.date == latest.date()]
    
    print(f"📊 Testing on {len(one_day)} bars from {latest.date()}")
    
    # Test configurations
    print("\n🔬 Testing comprehensive configuration space...\n")
    
    configs = []
    
    # Test 1: Confidence levels (0.30 - 0.70)
    for conf in [0.30, 0.35, 0.40, 0.45, 0.50]:
        configs.append((conf, 0.005, 0.01, 3, f"Conf={conf}"))
    
    # Test 2: Different stop/target ratios
    for stop, target in [(0.003, 0.01), (0.004, 0.01), (0.005, 0.01), (0.003, 0.015), (0.004, 0.012)]:
        configs.append((0.40, stop, target, 3, f"Stop={stop*100}%, Target={target*100}%"))
    
    # Test 3: More positions
    for max_pos in [5, 7]:
        configs.append((0.40, 0.005, 0.01, max_pos, f"Max Positions={max_pos}"))
    
    # Test 4: Tight stops with higher targets (aggressive)
    for stop, target in [(0.003, 0.015), (0.002, 0.015)]:
        configs.append((0.40, stop, target, 3, f"Aggressive: Stop={stop*100}%, Target={target*100}%"))
    
    results = []
    total_tests = len(configs)
    
    for i, (conf, stop, target, max_pos, name) in enumerate(configs, 1):
        print(f"[{i}/{total_tests}] Testing {name}...")
        
        result = test_config(conf, stop, target, max_pos, one_day)
        
        if result:
            results.append(result)
            print(f"   ✅ Profit: ${result['profit']:.2f}, Trades: {result['trades']}, Win: {result['win_rate']*100:.1f}%")
        else:
            print(f"   ❌ Failed")
        print()
    
    # Find best
    if results:
        df_results = pd.DataFrame(results)
        
        # Sort by profit
        df_sorted = df_results.sort_values('profit', ascending=False)
        
        best = df_sorted.iloc[0]
        
        print("=" * 90)
        print("🏆 WINNING CONFIGURATION")
        print("=" * 90)
        
        print(f"\n🥇 Settings:")
        print(f"   Confidence: {best['confidence']}")
        print(f"   Stop Loss: {best['stop_pct']*100}%")
        print(f"   Take Profit: {best['target_pct']*100}%")
        print(f"   Max Positions: {int(best['max_pos'])}")
        
        print(f"\n💰 RESULTS:")
        print(f"   Profit: ${best['profit']:.2f}")
        print(f"   Return: {best['return']:+.2f}%")
        print(f"   Trades: {int(best['trades'])}")
        print(f"   Win Rate: {best['win_rate']*100:.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        print(f"   Avg Win: ${best['avg_win']:.2f}")
        print(f"   Avg Loss: ${best['avg_loss']:.2f}")
        print(f"   Max DD: {best['max_dd']*100:.2f}%")
        
        print(f"\n📊 TOP 5 CONFIGURATIONS:")
        for i, row in df_sorted.head(5).iterrows():
            print(f"   {row['confidence']:.2f} | ${row['profit']:.2f} | {row['trades']:.0f} trades | {row['win_rate']*100:.1f}% win")
        
        print("\n✅ OPTIMAL CONFIGURATION FOUND!")
        print("=" * 90)

if __name__ == "__main__":
    main()

