"""
Test Even Lower Confidence Thresholds
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_confidence(confidence, data):
    """Test one confidence level"""
    
    backtester = Backtester(
        model_path="src/models/trained_models/uk_5min_day_trading_model.pkl",
        initial_capital=10000,
        min_confidence=confidence,
        stop_loss_pct=0.005,
        take_profit_pct=0.01,
        max_positions=3
    )
    
    result = backtester.run_backtest(data)
    
    return {
        'confidence': confidence,
        'profit': result['total_pnl'],
        'return': result['total_return_pct'],
        'trades': result['total_trades'],
        'win_rate': result['win_rate'],
        'sharpe': result['sharpe_ratio'],
        'max_dd': result['max_drawdown']
    }

def main():
    """Test multiple confidence levels"""
    
    print("=" * 90)
    print("🧪 TESTING LOWER CONFIDENCE THRESHOLDS")
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
    
    # Test confidence levels from high to low
    conf_levels = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    
    results = []
    
    print("\n🧪 Testing different confidence levels...\n")
    
    for conf in conf_levels:
        print(f"📊 Testing confidence = {conf}...")
        
        try:
            result = test_confidence(conf, one_day)
            results.append(result)
            
            print(f"   Profit: ${result['profit']:.2f} ({result['return']:+.2f}%)")
            print(f"   Trades: {result['trades']}, Win Rate: {result['win_rate']*100:.1f}%")
            print()
        
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    # Find best
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('profit', ascending=False)
        
        best = df_results.iloc[0]
        
        print("=" * 90)
        print("🏆 BEST CONFIDENCE LEVEL")
        print("=" * 90)
        print(f"\n🥇 Confidence: {best['confidence']}")
        print(f"💰 Profit: ${best['profit']:.2f}")
        print(f"📈 Return: {best['return']:+.2f}%")
        print(f"📊 Trades: {best['trades']}")
        print(f"🎯 Win Rate: {best['win_rate']*100:.1f}%")
        print(f"📊 Sharpe: {best['sharpe']:.2f}")
        print(f"⚠️  Max DD: {best['max_dd']*100:.2f}%")
        
        print(f"\n📊 All Results (sorted by profit):")
        for i, row in df_results.iterrows():
            marker = "🥇" if i == 0 else "  "
            print(f"{marker} {row['confidence']:.2f}: ${row['profit']:.2f} ({row['return']:+.2f}%), {row['trades']} trades, {row['win_rate']*100:.1f}% win")
        
        print("\n💡 INSIGHT:")
        if best['confidence'] < 0.50:
            print("   Very low confidence (< 0.50) usually means:")
            print("   ❌ Too many false signals")
            print("   ❌ Lower quality trades")
            print("   ❌ May overwhelm max_positions limit")
            print("   ❌ Lower win rate")
        
        print("\n✅ RECOMMENDATION:")
        if best['confidence'] >= 0.50:
            print(f"   Use confidence = {best['confidence']} for maximum profit!")
        else:
            print(f"   Confidence = {best['confidence']} gives highest profit")
            print("   But consider the trade-off with quality")
        
        print("=" * 90)

if __name__ == "__main__":
    main()

