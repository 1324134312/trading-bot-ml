"""
Test Higher Max Positions - Find the Limit
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_max_positions(max_pos, data):
    """Test one max position level"""
    
    try:
        backtester = Backtester(
            model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
            initial_capital=10000,
            min_confidence=0.40,
            stop_loss_pct=0.005,
            take_profit_pct=0.01,
            max_positions=max_pos
        )
        
        result = backtester.run_backtest(data)
        
        return {
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
        print(f"   ❌ Error: {e}")
        return None

def main():
    """Test different max positions"""
    
    print("=" * 90)
    print("🔬 TESTING HIGHER MAX POSITIONS")
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
    
    # One day
    data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'])
    latest = data_clean['Datetime'].max()
    one_day = data_clean[data_clean['Datetime'].dt.date == latest.date()]
    
    print(f"📊 Testing on {len(one_day)} bars from {latest.date()}")
    
    # Test different max positions
    max_positions_to_test = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    
    print("\n🧪 Testing max positions...\n")
    
    results = []
    
    for max_pos in max_positions_to_test:
        print(f"📊 Testing max positions = {max_pos}...")
        
        result = test_max_positions(max_pos, one_day)
        
        if result:
            results.append(result)
            print(f"   ✅ Profit: ${result['profit']:.2f}")
            print(f"   Trades: {result['trades']}")
            print(f"   Win Rate: {result['win_rate']*100:.1f}%")
            print(f"   Sharpe: {result['sharpe']:.2f}")
            print()
    
    # Find best
    if results:
        df_results = pd.DataFrame(results)
        df_sorted = df_results.sort_values('profit', ascending=False)
        
        best = df_sorted.iloc[0]
        
        print("=" * 90)
        print("🏆 OPTIMAL MAX POSITIONS")
        print("=" * 90)
        
        print(f"\n🥇 Winner: {int(best['max_pos'])} positions")
        print(f"   Profit: ${best['profit']:.2f}")
        print(f"   Return: {best['return']:+.2f}%")
        print(f"   Trades: {int(best['trades'])}")
        print(f"   Win Rate: {best['win_rate']*100:.1f}%")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        print(f"   Max DD: {best['max_dd']*100:.2f}%")
        
        print(f"\n📊 ALL RESULTS:")
        for i, row in df_sorted.iterrows():
            marker = "🥇" if i == 0 else "  "
            print(f"{marker} {int(row['max_pos']):2d} positions: ${row['profit']:.2f} | {int(row['trades']):2d} trades | {row['win_rate']*100:.1f}% win | Sharpe {row['sharpe']:.1f}")
        
        # Find where profit plateaus
        print(f"\n💡 INSIGHTS:")
        
        profits = df_sorted['profit'].values
        positions = df_sorted['max_pos'].values
        
        # Find where returns start diminishing
        for i in range(1, len(df_sorted)):
            current_profit = profits[i]
            previous_profit = profits[i-1]
            improvement = current_profit - previous_profit
            
            if improvement < 10:  # Less than $10 improvement
                print(f"   Profit plateaus after {int(positions[i-1])} positions")
                print(f"   After {int(positions[i-1])} positions: +${improvement:.2f}/position")
                break
        
        print(f"\n✅ RECOMMENDATION:")
        print(f"   Use {int(best['max_pos'])} max positions for maximum profit!")
        
        print("=" * 90)

if __name__ == "__main__":
    main()

