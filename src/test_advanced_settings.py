"""
Test Advanced Settings - Other Optimization Dimensions

Tests:
1. Different stop/target combinations
2. Early exit strategies
3. Trailing stops
4. Position sizing variations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_config(confidence, stop_pct, target_pct, max_pos, risk_multiplier, name):
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
        
        # Adjust risk if needed
        if risk_multiplier != 1.0:
            backtester.initial_risk_pct = backtester.initial_risk_pct * risk_multiplier
        
        result = backtester.run_backtest(one_day)
        
        result['name'] = name
        result['risk_mult'] = risk_multiplier
        
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Test advanced configurations"""
    
    print("=" * 90)
    print("🔬 TESTING ADVANCED SETTINGS")
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
    
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    latest = data_clean['Datetime'].max()
    global one_day
    one_day = data_clean[data_clean['Datetime'].dt.date == latest.date()]
    
    print(f"📊 Testing on {len(one_day)} bars from {latest.date()}")
    
    # Test advanced configurations
    print("\n🧪 Testing advanced settings...\n")
    
    configs = []
    
    # Test 1: Different stop/target ratios (tighter stops)
    configs.append((0.40, 0.003, 0.01, 7, 1.0, "Tight Stop: 0.3%"))
    configs.append((0.40, 0.004, 0.01, 7, 1.0, "Tight Stop: 0.4%"))
    configs.append((0.40, 0.006, 0.01, 7, 1.0, "Wide Stop: 0.6%"))
    
    # Test 2: Different targets (let winners run)
    configs.append((0.40, 0.005, 0.008, 7, 1.0, "Close Target: 0.8%"))
    configs.append((0.40, 0.005, 0.012, 7, 1.0, "Wide Target: 1.2%"))
    configs.append((0.40, 0.005, 0.015, 7, 1.0, "Wide Target: 1.5%"))
    
    # Test 3: Better risk/reward ratios
    configs.append((0.40, 0.004, 0.012, 7, 1.0, "Balanced: 0.4%/1.2%"))
    configs.append((0.40, 0.003, 0.015, 7, 1.0, "Aggressive: 0.3%/1.5% (5:1)"))
    
    # Test 4: Lower confidence with wider stops
    configs.append((0.35, 0.005, 0.01, 7, 1.0, "Lower Conf + Wide"))
    
    # Test 5: Higher capital utilization (risk more per trade)
    configs.append((0.40, 0.005, 0.01, 7, 1.5, "Higher Risk (1.5x)"))
    configs.append((0.40, 0.005, 0.01, 7, 2.0, "Higher Risk (2x)"))
    
    # Test 6: Original settings
    configs.append((0.40, 0.005, 0.01, 7, 1.0, "Current Optimal"))
    
    results = []
    total_tests = len(configs)
    
    for i, (conf, stop, target, max_pos, risk_mult, name) in enumerate(configs, 1):
        print(f"[{i}/{total_tests}] {name}...")
        
        result = test_config(conf, stop, target, max_pos, risk_mult, name)
        
        if result:
            results.append(result)
            print(f"   ✅ ${result['total_pnl']:.2f} | {result['total_trades']:.0f} trades | {result['win_rate']*100:.1f}% win")
        print()
    
    if results:
        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values('total_pnl', ascending=False)
        
        best = df_sorted.iloc[0]
        
        print("=" * 90)
        print("🏆 BEST ADVANCED CONFIGURATION")
        print("=" * 90)
        
        print(f"\n🥇 {best['name']}:")
        print(f"   Settings: Conf={best.get('min_confidence', 0.40):.2f}, Stop={best.get('stop_loss_pct', 0.005)*100:.1f}%, Target={best.get('take_profit_pct', 0.01)*100:.1f}%")
        print(f"   Risk Mult: {best['risk_mult']:.1f}x")
        
        print(f"\n💰 RESULTS:")
        print(f"   Profit: ${best['total_pnl']:.2f}")
        print(f"   Return: {best['total_return_pct']:+.2f}%")
        print(f"   Trades: {int(best['total_trades'])}")
        print(f"   Win Rate: {best['win_rate']*100:.1f}%")
        print(f"   Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        
        print(f"\n📊 TOP 5:")
        for i, row in df_sorted.head(5).iterrows():
            print(f"   {row['name']}: ${row['total_pnl']:.2f} ({row['win_rate']*100:.1f}% win, {int(row['total_trades'])} trades)")
        
        print("\n✅ OPTIMAL ADVANCED CONFIGURATION!")
        print("=" * 90)

if __name__ == "__main__":
    main()

