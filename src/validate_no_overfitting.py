"""
Validate Settings Across Multiple Days - Check for Overfitting

If we're over-optimized, performance will drop on different days!
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_config(conf, stop, target, max_pos, data, day_name):
    """Test one configuration"""
    
    try:
        backtester = Backtester(
            model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
            initial_capital=10000,
            min_confidence=conf,
            stop_loss_pct=stop,
            take_profit_pct=target,
            max_positions=max_pos
        )
        
        result = backtester.run_backtest(data)
        
        result['day'] = day_name
        result['conf'] = conf
        result['stop'] = stop
        result['target'] = target
        
        return result
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Test settings across multiple days"""
    
    print("=" * 90)
    print("🔍 VALIDATING SETTINGS - CHECKING FOR OVERFITTING")
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
    
    # Get available days
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    unique_days = data_clean['Datetime'].dt.date.unique()
    unique_days_sorted = sorted(unique_days)
    
    print(f"📅 Available days: {len(unique_days_sorted)}")
    
    if len(unique_days_sorted) < 2:
        print("❌ Need at least 2 days for validation")
        return
    
    # Test on first few days
    test_days = unique_days_sorted[:min(5, len(unique_days_sorted))]
    
    print(f"📊 Testing on {len(test_days)} days...")
    
    # Configuration to test
    # Current "optimal" settings
    test_configs = [
        (0.40, 0.005, 0.008, 7, "Current Optimal (0.8% target)"),
        (0.40, 0.005, 0.01, 7, "Previous Optimal (1.0% target)"),
        (0.40, 0.005, 0.01, 3, "Conservative (3 positions)"),
        (0.55, 0.005, 0.01, 3, "Original Baseline"),
    ]
    
    all_results = []
    
    print("\n🧪 Testing configurations across multiple days...\n")
    
    for conf, stop, target, max_pos, name in test_configs:
        print(f"📊 Testing: {name}")
        
        day_results = []
        
        for day in test_days:
            day_data = data_clean[data_clean['Datetime'].dt.date == day]
            
            if len(day_data) > 0:
                result = test_config(conf, stop, target, max_pos, day_data, str(day))
                
                if result:
                    day_results.append(result)
                    print(f"   Day {day}: ${result['total_pnl']:.2f} ({result['win_rate']*100:.1f}% win, {result['total_trades']:.0f} trades)")
        
        if day_results:
            df_day = pd.DataFrame(day_results)
            avg_profit = df_day['total_pnl'].mean()
            std_profit = df_day['total_pnl'].std()
            avg_winrate = df_day['win_rate'].mean()
            
            print(f"   ✅ Avg: ${avg_profit:.2f} ± ${std_profit:.2f}")
            print(f"   Avg Win Rate: {avg_winrate*100:.1f}%")
            print()
            
            all_results.append({
                'name': name,
                'avg_profit': avg_profit,
                'std_profit': std_profit,
                'avg_winrate': avg_winrate,
                'consistency': std_profit / avg_profit if avg_profit != 0 else 999
            })
    
    # Compare results
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_sorted = df_results.sort_values('avg_profit', ascending=False)
        
        print("=" * 90)
        print("📊 VALIDATION RESULTS")
        print("=" * 90)
        
        print(f"\n🏆 Rankings (by average profit):")
        for i, row in df_sorted.iterrows():
            marker = "🥇" if i == 0 else "   "
            print(f"{marker} {i+1}. {row['name']}:")
            print(f"      Avg Profit: ${row['avg_profit']:.2f}")
            print(f"      Std Dev: ${row['std_profit']:.2f}")
            print(f"      Consistency: {row['consistency']*100:.1f}%")
            print(f"      Avg Win Rate: {row['avg_winrate']*100:.1f}%")
            print()
        
        # Check for overfitting
        best = df_sorted.iloc[0]
        print(f"\n💡 ANALYSIS:")
        
        if best['name'] == "Current Optimal (0.8% target)":
            print("   ✅ Current optimal settings perform best!")
        
        if best['std_profit'] / best['avg_profit'] > 0.5:
            print("   ⚠️  WARNING: High variability suggests possible overfitting!")
            print("   Recommendation: Use more conservative settings")
        
        if best['consistency'] < 0.3:
            print("   ✅ Good consistency across days")
        else:
            print("   ⚠️  Higher variability - consider more conservative settings")
        
        print("\n✅ VALIDATION COMPLETE!")
        print("=" * 90)

if __name__ == "__main__":
    main()

