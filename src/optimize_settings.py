"""
Test different settings to find most profitable configuration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.backtesting.backtester import Backtester

def test_configurations(data, model_path):
    """Test multiple configurations"""
    
    # Define configurations to test
    configs = [
        # (name, confidence, risk_pct, stop_pct, target_pct)
        ("Current Baseline", 0.65, 0.01, 0.005, 0.01),
        
        # Lower confidence (more trades)
        ("Low Confidence", 0.55, 0.01, 0.005, 0.01),
        ("Very Low Conf", 0.50, 0.01, 0.005, 0.01),
        
        # Higher risk (bigger positions)
        ("High Risk", 0.65, 0.02, 0.005, 0.01),
        ("Higher Risk", 0.65, 0.03, 0.005, 0.01),
        
        # Tighter stops
        ("Tight Stops", 0.65, 0.01, 0.003, 0.01),
        ("Very Tight", 0.65, 0.01, 0.003, 0.015),
        
        # Wider targets
        ("Wide Targets", 0.65, 0.01, 0.005, 0.015),
        ("Very Wide", 0.65, 0.01, 0.005, 0.02),
        
        # Balanced
        ("Balanced 1", 0.60, 0.015, 0.004, 0.012),
        ("Balanced 2", 0.60, 0.02, 0.006, 0.015),
    ]
    
    results = []
    
    print("=" * 90)
    print("📊 Testing Multiple Configurations")
    print("=" * 90)
    print()
    
    for name, confidence, risk_pct, stop_pct, target_pct in configs:
        print(f"\n{'='*90}")
        print(f"Testing: {name}")
        print(f"  Confidence: {confidence}")
        print(f"  Risk: {risk_pct*100}%")
        print(f"  Stop: {stop_pct*100}%, Target: {target_pct*100}%")
        print(f"{'='*90}")
        
        try:
            backtester = Backtester(
                model_path=model_path,
                initial_capital=10000,
                min_confidence=confidence,
                stop_loss_pct=stop_pct,
                take_profit_pct=target_pct,
                max_positions=3
            )
            
            # Override risk percentage
            backtester.initial_risk_pct = risk_pct
            
            # Run backtest
            results_dict = backtester.run_backtest(data)
            results_dict['config_name'] = name
            results_dict['confidence'] = confidence
            results_dict['risk_pct'] = risk_pct
            results_dict['stop_pct'] = stop_pct
            results_dict['target_pct'] = target_pct
            
            results.append(results_dict)
            
            # Print summary
            print(f"\n✅ {name}:")
            print(f"   Return: {results_dict['total_return_pct']:.2f}%")
            print(f"   Trades: {results_dict['total_trades']}")
            print(f"   Win Rate: {results_dict['win_rate']:.1f}%")
            print(f"   Sharpe: {results_dict['sharpe_ratio']:.2f}")
            print(f"   Max DD: {results_dict['max_drawdown']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Compare results
    print("\n" + "=" * 90)
    print("📊 COMPARISON TABLE")
    print("=" * 90)
    
    df_results = pd.DataFrame(results)
    
    # Sort by total return
    df_results = df_results.sort_values('total_return_pct', ascending=False)
    
    print("\nTop 5 Configurations by Return:")
    print(df_results[['config_name', 'total_return_pct', 'total_trades', 
                     'win_rate', 'sharpe_ratio']].head().to_string())
    
    # Find best Sharpe
    best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    
    print("\n" + "=" * 90)
    print("🏆 BEST CONFIGURATION (Sharpe Ratio)")
    print("=" * 90)
    print(f"Name: {best_sharpe['config_name']}")
    print(f"Confidence: {best_sharpe['confidence']}")
    print(f"Risk: {best_sharpe['risk_pct']*100}%")
    print(f"Stop: {best_sharpe['stop_pct']*100}%, Target: {best_sharpe['target_pct']*100}%")
    print(f"\nResults:")
    print(f"  Return: {best_sharpe['total_return_pct']:.2f}%")
    print(f"  Trades: {best_sharpe['total_trades']}")
    print(f"  Win Rate: {best_sharpe['win_rate']:.1f}%")
    print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
    print(f"  Max DD: {best_sharpe['max_drawdown']*100:.2f}%")
    
    # Save results
    df_results.to_csv("optimization_results.csv", index=False)
    print("\n💾 Results saved to: optimization_results.csv")
    
    return df_results

def main():
    """Main function"""
    
    print("=" * 90)
    print("🔧 Bot Optimization: Testing Multiple Settings")
    print("=" * 90)
    
    # Load data
    print("\n📂 Loading data...")
    data_dir = "data/uk/processed_5min"
    csv_files = glob.glob(os.path.join(data_dir, "*_features.csv"))
    
    all_data_with_ticker = []
    for file in csv_files:
        ticker = os.path.basename(file).replace('_features.csv', '').replace('.L', '')
        df = pd.read_csv(file)
        df['Ticker'] = ticker
        all_data_with_ticker.append(df)
    
    data = pd.concat(all_data_with_ticker, ignore_index=True)
    data_clean = data.dropna()
    
    print(f"✅ Loaded {len(data_clean)} bars from {len(csv_files)} stocks")
    
    # Load model
    model_path = "src/models/trained_models/uk_5min_day_trading_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        return
    
    # Test configurations
    results = test_configurations(data_clean, model_path)
    
    print("\n" + "=" * 90)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    main()

