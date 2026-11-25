"""
Test UK 5-Minute Day Trading Strategy with Backtest
"""
import pandas as pd
import os
import glob
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.backtester import Backtester

def main():
    """Test backtest on UK 5-minute data"""
    
    print("=" * 70)
    print("🧪 Testing UK 5-Minute Day Trading Strategy")
    print("=" * 70)
    
    # Load processed data
    data_dir = "data/uk/processed_5min"
    
    print(f"\n📁 Loading data from: {data_dir}")
    
    # Load all processed files
    all_data = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*_features.csv"))
    
    if not csv_files:
        print(f"❌ No processed files found in {data_dir}")
        print("   Run: python src/process_uk_5min_features.py")
        return
    
    print(f"📊 Found {len(csv_files)} stocks")
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"✅ {os.path.basename(file)}: {len(df)} rows")
        except Exception as e:
            print(f"⚠️  Error loading {file}: {e}")
    
    if not all_data:
        print("❌ No data loaded")
        return
    
    # Combine all data
    data = pd.concat(all_data, ignore_index=True)
    
    # Extract ticker from filename
    print("\n🔍 Extracting ticker information from filenames...")
    all_data_with_ticker = []
    
    for file in csv_files:
        ticker = os.path.basename(file).replace('_features.csv', '').replace('.L', '')
        df = pd.read_csv(file)
        df['Ticker'] = ticker
        all_data_with_ticker.append(df)
    
    data = pd.concat(all_data_with_ticker, ignore_index=True)
    
    print(f"\n📊 Total bars: {len(data)}")
    print(f"📅 Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    print(f"📈 Unique times: {data['Datetime'].nunique()}")
    
    # Check if we have the target column
    if 'Target_15min' not in data.columns:
        print("\n❌ Missing 'Target_15min' column")
        print("   Run feature engineering first: python src/process_uk_5min_features.py")
        return
    
    # Drop NaN rows
    data_clean = data.dropna()
    print(f"\n📊 Clean data: {len(data_clean)} bars (removed {len(data) - len(data_clean)} NaN rows)")
    
    if len(data_clean) < 100:
        print("❌ Not enough clean data for backtest")
        return
    
    # Backtest
    print("\n" + "=" * 70)
    print("🚀 Running Backtest")
    print("=" * 70)
    
    # Initialize backtester
    model_path = "src/models/trained_models/uk_5min_day_trading_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Run: python src/train_uk_5min_model.py")
        return
    
    backtester = Backtester(
        model_path=model_path,
        initial_capital=10000,
        min_confidence=0.40,  # MAXIMUM PROFIT!
        stop_loss_pct=0.005,  # 0.5% stop loss
        take_profit_pct=0.008,  # 0.8% take profit - OPTIMIZED!
        max_positions=7  # OPTIMAL! (was 3)
    )
    
    # Run backtest
    results = backtester.run_backtest(data_clean)
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n💰 CAPITAL:")
    print(f"   Initial: ${results['initial_capital']:,.2f}")
    print(f"   Final: ${results['final_value']:,.2f}")
    print(f"   Total Return: {results['total_return_pct']:.2f}%")
    
    print(f"\n📈 TRADES:")
    print(f"   Total: {results['total_trades']}")
    print(f"   Winners: {results['winning_trades']}")
    print(f"   Losers: {results['losing_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    
    if results['total_trades'] > 0:
        print(f"\n💵 P&L:")
        print(f"   Total P&L: ${results['total_pnl']:,.2f}")
        print(f"   Avg Win: ${results['avg_win']:,.2f}")
        print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\n⏱️  HOLDING:")
        print(f"   Avg holding: {results['avg_holding_bars']:.1f} bars ({results['avg_holding_bars'] * 5:.1f} minutes)")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']*100:.2f}%")
    
    # Save results
    results_file = "BACKTEST_RESULTS.txt"
    with open(results_file, 'w') as f:
        f.write("BACKTEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Trade history
    if backtester.trade_history:
        trade_df = pd.DataFrame(backtester.trade_history)
        trade_df.to_csv("backtest_trades.csv", index=False)
        print(f"💾 Trade history saved to: backtest_trades.csv")
    
    # Equity curve
    if backtester.equity_curve:
        equity_df = pd.DataFrame(backtester.equity_curve)
        equity_df.to_csv("equity_curve.csv", index=False)
        print(f"💾 Equity curve saved to: equity_curve.csv")

if __name__ == "__main__":
    main()

