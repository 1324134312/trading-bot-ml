"""
Analyze what's working and what's not across different stocks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.predictor import ModelPredictor
from config.settings import PROCESSED_DATA_DIR
import matplotlib.pyplot as plt

def analyze_model_performance():
    """Analyze model performance across different stock types"""
    
    print("=" * 70)
    print("📊 Model Performance Analysis Across Stocks")
    print("=" * 70)
    
    # Load model
    model_path = 'src/models/trained_models/multi_ticker_xgb_model.pkl'
    predictor = ModelPredictor(model_path)
    
    # Test on each stock
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
    results = []
    
    for ticker in tickers:
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.csv")
        if not os.path.exists(filepath):
            continue
        
        data = pd.read_csv(filepath)
        predictions = predictor.predict_direction(data.copy())
        
        if 'Target_Direction' not in predictions.columns:
            continue
        
        correct = (predictions['Prediction'] == predictions['Target_Direction']).sum()
        total = len(predictions)
        accuracy = correct / total * 100
        
        results.append({
            'Ticker': ticker,
            'Accuracy': accuracy,
            'Type': 'ETF' if ticker in ['SPY', 'QQQ'] else 'Individual Stock'
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n📈 Results by Stock Type:")
    print("=" * 70)
    print(results_df.groupby('Type')['Accuracy'].agg(['mean', 'min', 'max']))
    
    print("\n📊 Individual Stock Performance:")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['Ticker']:6s}: {row['Accuracy']:5.2f}% ({row['Type']})")
    
    print(f"\n✅ Average: {results_df['Accuracy'].mean():.2f}%")
    print(f"   Best: {results_df['Accuracy'].max():.2f}%")
    print(f"   Worst: {results_df['Accuracy'].min():.2f}%")
    
    print("\n🎯 Key Finding:")
    if 'ETF' in results_df['Type'].values:
        etf_avg = results_df[results_df['Type'] == 'ETF']['Accuracy'].mean()
        stock_avg = results_df[results_df['Type'] == 'Individual Stock']['Accuracy'].mean()
        print(f"   ETFs: {etf_avg:.2f}% vs Individual Stocks: {stock_avg:.2f}%")
        print(f"   Difference: {etf_avg - stock_avg:.2f}%")

if __name__ == "__main__":
    analyze_model_performance()

