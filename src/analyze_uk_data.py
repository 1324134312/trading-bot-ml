"""
Analyze UK Day Trading Data Quality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob

def analyze_uk_data():
    """Analyze quality of UK day trading data"""
    
    print("=" * 70)
    print("🔍 UK Day Trading Data Quality Analysis")
    print("=" * 70)
    
    # Find all files
    files = glob.glob("data/uk/processed/*_features.csv")
    
    print(f"\n📁 Files found: {len(files)}")
    print()
    
    all_stats = []
    
    for filepath in files:
        ticker = os.path.basename(filepath).split('_')[0]
        df = pd.read_csv(filepath)
        
        # Basic stats
        total_rows = len(df)
        buy_signals = df['Target_2h'].sum()
        sell_signals = total_rows - buy_signals
        buy_pct = (buy_signals / total_rows) * 100
        
        # Check for missing data
        missing = df.isnull().sum().sum()
        
        # Check for infinity
        inf_count = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count += np.isinf(df[col]).sum().sum()
        
        all_stats.append({
            'Ticker': ticker,
            'Total': total_rows,
            'Buy Signals': buy_signals,
            'Sell Signals': sell_signals,
            'Buy %': buy_pct,
            'Missing': missing,
            'Inf': inf_count
        })
        
        print(f"📊 {ticker}:")
        print(f"   Rows: {total_rows}")
        print(f"   Buy signals: {buy_signals} ({buy_pct:.1f}%)")
        print(f"   Missing data: {missing}")
        print(f"   Infinite values: {inf_count}")
    
    # Summary
    stats_df = pd.DataFrame(all_stats)
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    total_rows_all = stats_df['Total'].sum()
    total_buy = stats_df['Buy Signals'].sum()
    avg_buy_pct = stats_df['Buy %'].mean()
    
    print(f"\nTotal data points: {total_rows_all}")
    print(f"Average per stock: {total_rows_all / len(stats_df):.0f}")
    print(f"\nBuy signal rate: {avg_buy_pct:.1f}%")
    print(f"Overall: {total_buy}/{total_rows_all} ({total_buy/total_rows_all*100:.1f}%)")
    
    # Data quality
    print(f"\nData Quality:")
    print(f"  Missing data: {stats_df['Missing'].sum()}")
    print(f"  Infinite values: {stats_df['Inf'].sum()}")
    print(f"  ✅ Clean data: {stats_df['Missing'].sum() == 0}")
    
    # Coverage
    print(f"\nTime Coverage:")
    print(f"  Period: 60 days")
    print(f"  Hourly bars: ~240 bars per 10 days (approx)")
    print(f"  Expected: 60 days × ~8.5 hours × 10 stocks = ~5,100 bars")
    print(f"  Actual: {total_rows_all} bars")
    
    # Issues
    print("\n⚠️  Issues Identified:")
    if avg_buy_pct < 20:
        print(f"  ❌ SEVERE class imbalance ({avg_buy_pct:.1f}% buy signals)")
        print(f"     Model will struggle with minority class")
    if avg_buy_pct > 30:
        print(f"  ⚠️  Class imbalance ({avg_buy_pct:.1f}% buy signals)")
    
    if total_rows_all < 5000:
        print(f"  ⚠️  Limited data ({total_rows_all} total bars)")
        print(f"      Only ~60 days vs 5 years for US model")
    
    # Comparison to US model
    print("\n" + "=" * 70)
    print("📊 Comparison: UK vs US Model")
    print("=" * 70)
    
    print("\nUK Day Trading Model:")
    print(f"  Data points: {total_rows_all}")
    print(f"  Period: 60 days (hourly)")
    print(f"  Stocks: 10 FTSE 100")
    print(f"  Buy signal rate: {avg_buy_pct:.1f}%")
    print(f"  Model accuracy: 74%")
    
    print("\nUS Swing Trading Model:")
    print(f"  Data points: 11,304")
    print(f"  Period: 5 years (daily)")
    print(f"  Stocks: 9 (US + 2 ETFs)")
    print(f"  Buy signal rate: 58% (balanced)")
    print(f"  Model accuracy: 57% (77% on ETFs)")
    
    print("\n💡 Assessment:")
    if avg_buy_pct < 20:
        print(f"  ❌ UK data quality: POOR")
        print(f"     - Too few buy signals ({avg_buy_pct:.1f}%)")
        print(f"     - Model biased toward 'sell/hold'")
        print(f"     - Limited training data")
        print(f"  ⚠️  Should get more data (extend to 6-12 months)")
    else:
        print(f"  ✅ UK data quality: GOOD")
        print(f"     - Reasonable buy signal rate")
        print(f"     - Clean data (no missing/infinite values)")
        print(f"     - Suitable for training")

if __name__ == "__main__":
    analyze_uk_data()

