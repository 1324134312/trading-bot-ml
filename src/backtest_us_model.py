"""
Backtest US Model with Trading Strategy on US Data

Test the US model on US stocks to see actual trading performance!
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import pickle
from src.backtesting.backtester import Backtester
from tqdm import tqdm

def main():
    """Backtest US model on US data"""
    
    print("=" * 90)
    print("🧪 Backtesting US Model with Trading Strategy")
    print("=" * 90)
    
    # Load US processed data
    print("\n📂 Loading US Max Data...")
    
    data_dir = "data/us/max_processed"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not files:
        print(f"❌ No processed US data found in {data_dir}")
        print("   Run: python src/process_us_max_features.py")
        return
    
    print(f"📁 Found {len(files)} stocks")
    
    all_data = []
    for file in tqdm(files, desc="Loading"):
        try:
            df = pd.read_csv(file)
            ticker = os.path.basename(file).replace('_features.csv', '')
            df['Ticker'] = ticker
            all_data.append(df)
        except:
            pass
    
    data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Loaded {len(data)} bars")
    
    # Check for target
    if 'Target_15min' not in data.columns:
        print("❌ No Target_15min column")
        return
    
    # Clean data
    data_clean = data.dropna()
    print(f"📊 Clean data: {len(data_clean)} bars")
    
    if len(data_clean) < 100:
        print("❌ Not enough data")
        return
    
    # Sort by datetime
    if 'Datetime' in data_clean.columns:
        data_clean['Datetime'] = pd.to_datetime(data_clean['Datetime'], utc=True)
        data_clean = data_clean.sort_values('Datetime')
    
    print(f"📅 Date range: {data_clean['Datetime'].min()} to {data_clean['Datetime'].max()}")
    
    # Initialize backtester
    print("\n" + "=" * 90)
    print("🚀 Running Backtest with US Model")
    print("=" * 90)
    
    # Load US model files
    with open("models/us_max_model.pkl", 'rb') as f:
        us_model = pickle.load(f)
    with open("models/us_max_scaler.pkl", 'rb') as f:
        us_scaler = pickle.load(f)
    with open("models/us_max_features.pkl", 'rb') as f:
        us_features = pickle.load(f)
    
    # Create custom backtester
    from src.backtesting.backtester import Backtester
    
    # Backtester expects ModelPredictor, but we have direct model
    # Let's create a simple backtest without the full predictor
    print("\n⚠️  US model uses different format")
    print("   Using direct prediction method")
    
    # Create a simple test instead
    # We'll use the existing UK backtest method with US model loaded manually
    backtester = Backtester.__new__(Backtester)
    backtester.initial_capital = 10000
    backtester.cash = 10000
    backtester.min_confidence = 0.65
    backtester.stop_loss_pct = 0.005
    backtester.take_profit_pct = 0.01
    backtester.max_positions = 3
    backtester.positions = {}
    backtester.trade_history = []
    backtester.equity_curve = []
    backtester.model = us_model
    backtester.scaler = us_scaler
    backtester.feature_cols = us_features
    
    # Run backtest
    results = backtester.run_backtest(data_clean)
    
    # Display results
    print("\n" + "=" * 90)
    print("📊 BACKTEST RESULTS - US MODEL ON US STOCKS")
    print("=" * 90)
    
    print(f"\n💰 CAPITAL:")
    print(f"   Initial: ${results['initial_capital']:,.2f}")
    print(f"   Final: ${results['final_value']:,.2f}")
    print(f"   Return: {results['total_return_pct']:+.2f}%")
    
    print(f"\n📈 TRADES:")
    print(f"   Total: {results['total_trades']}")
    print(f"   Winners: {results['winning_trades']}")
    print(f"   Losers: {results['losing_trades']}")
    
    if results['total_trades'] > 0:
        print(f"   Win Rate: {results['win_rate']*100:.1f}%")
        
        print(f"\n💵 P&L:")
        print(f"   Total P&L: ${results['total_pnl']:,.2f}")
        print(f"   Avg Win: ${results['avg_win']:,.2f}")
        print(f"   Avg Loss: ${results['avg_loss']:,.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        print(f"\n⏱️  HOLDING:")
        print(f"   Avg: {results['avg_holding_bars']:.1f} bars ({results['avg_holding_bars']*5:.1f} minutes)")
        
        print(f"\n📊 RISK:")
        print(f"   Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"   Max DD: {results['max_drawdown']*100:.2f}%")
    
    # Save results
    results_file = "US_MODEL_BACKTEST.txt"
    with open(results_file, 'w') as f:
        f.write("US MODEL BACKTEST RESULTS\n")
        f.write("=" * 90 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n💾 Results saved: {results_file}")

if __name__ == "__main__":
    main()

