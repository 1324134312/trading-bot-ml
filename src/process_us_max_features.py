"""
Process US Max Data - Add Features for Trading
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob
from src.feature_engineering.indicators import TechnicalIndicators
from tqdm import tqdm

def add_5min_features(df):
    """Add features optimized for 5-minute trading"""
    
    indicators = TechnicalIndicators()
    
    # Basic price features
    df['SMA_5'] = indicators.sma(df['Close'], 5)
    df['SMA_10'] = indicators.sma(df['Close'], 10)
    df['SMA_20'] = indicators.sma(df['Close'], 20)
    df['EMA_5'] = indicators.ema(df['Close'], 5)
    df['EMA_10'] = indicators.ema(df['Close'], 10)
    
    # Momentum
    df['RSI_10'] = indicators.rsi(df['Close'], 10)
    df['RSI_5'] = indicators.rsi(df['Close'], 5)
    
    # MACD
    macd = indicators.macd(df['Close'], fast=8, slow=17, signal=6)
    df = pd.concat([df, macd], axis=1)
    
    # Stochastic
    stoch = indicators.stochastic(df['High'], df['Low'], df['Close'], 10)
    df = pd.concat([df, stoch], axis=1)
    
    # Volatility
    df['ATR_10'] = indicators.atr(df['High'], df['Low'], df['Close'], 10)
    df['ATR_Pct'] = (df['ATR_10'] / df['Close']) * 100
    
    # Bollinger Bands
    bb = indicators.bollinger_bands(df['Close'], window=15, std_dev=2)
    df = pd.concat([df, bb], axis=1)
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(15).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    df['Volume_Momentum'] = df['Volume'].pct_change(3)
    
    # Returns
    df['Return_5min'] = df['Close'].pct_change(1)
    df['Return_30min'] = df['Close'].pct_change(6)
    
    # Range
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100
    
    # Candlestick
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Gaps
    df['Gap'] = df['Open'] - df['Close'].shift(1)
    df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
    
    # Time of day (US market: 9:30-16:00 ET)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Is_Morning'] = ((df['Hour'] >= 9) & (df['Hour'] <= 11)).astype(int)
    df['Is_Afternoon'] = ((df['Hour'] >= 14) & (df['Hour'] <= 16)).astype(int)
    df['Is_Start'] = ((df['Hour'] == 9) & (df['Minute'] == 30)).astype(int)
    df['Is_Close'] = ((df['Hour'] == 15) & (df['Minute'] == 59)).astype(int)
    
    # Crossovers
    df['SMA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
    df['EMA_Cross_5_10'] = df['EMA_5'] - df['EMA_10']
    
    # BB Position
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Momentum
    df['RSI_Momentum'] = df['RSI_10'].diff()
    df['MACD_Hist_Momentum'] = df['MACD_Histogram'].diff()
    
    # Lag features
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_3'] = df['Close'].shift(3)
    df['Volume_Lag_1'] = df['Volume'].shift(1)
    df['RSI_Lag_1'] = df['RSI_10'].shift(1)
    
    # Target: price up in 3 bars (15 minutes)
    future_price = df['Close'].shift(-3)
    future_return = (future_price / df['Close']) - 1
    df['Target_15min'] = (future_return > 0.003).astype(int)
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def main():
    """Process all US max data"""
    
    print("=" * 90)
    print("🔧 Processing US Max Data - Adding Features")
    print("=" * 90)
    
    input_dir = "data/us/max_data"
    output_dir = "data/us/max_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    files = [f for f in files if 'SUMMARY' not in f]
    
    print(f"\n📁 Found {len(files)} stocks to process\n")
    
    results = []
    
    for file in tqdm(files, desc="Processing"):
        try:
            ticker = os.path.basename(file).replace('_5m_max.csv', '')
            df = pd.read_csv(file)
            
            # Add features
            df_features = add_5min_features(df)
            
            # Clean
            df_clean = df_features.dropna()
            
            if len(df_clean) > 50:
                # Save
                output_file = os.path.join(output_dir, f"{ticker}_features.csv")
                df_clean.to_csv(output_file, index=False)
                
                results.append({
                    'ticker': ticker,
                    'bars': len(df_clean),
                    'features': len(df_clean.columns)
                })
        
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
    
    print(f"\n✅ Processed {len(results)} stocks")
    print(f"📊 Total bars with features: {sum(r['bars'] for r in results):,}")

if __name__ == "__main__":
    main()

