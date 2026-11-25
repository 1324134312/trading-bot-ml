"""
UK Day Trading Feature Engineering
Intraday-specific features for 1-hour bars
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob
from src.feature_engineering.indicators import TechnicalIndicators
from tqdm import tqdm

class UKIntradayFeatureExtractor:
    """Feature engineering for UK intraday day trading"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def add_intraday_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add features optimized for 1-hour intraday trading
        
        Args:
            data: DataFrame with 1-hour OHLCV bars
            
        Returns:
            DataFrame with intraday features
        """
        df = data.copy()
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            print("⚠️  Missing required OHLCV columns")
            return df
        
        # === INTRADAY MOVING AVERAGES (Hourly periods) ===
        # Short-term for day trading
        df['SMA_5h'] = self.indicators.sma(df['Close'], 5)   # 5 hours
        df['SMA_10h'] = self.indicators.sma(df['Close'], 10)  # ~1 day
        df['SMA_20h'] = self.indicators.sma(df['Close'], 20)  # ~2.5 days
        df['EMA_5h'] = self.indicators.ema(df['Close'], 5)
        df['EMA_10h'] = self.indicators.ema(df['Close'], 10)
        
        # === MOMENTUM FOR INTRADAY ===
        df['RSI_14h'] = self.indicators.rsi(df['Close'], 14)
        df['RSI_7h'] = self.indicators.rsi(df['Close'], 7)
        
        # MACD (slower for intraday)
        macd = self.indicators.macd(df['Close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # Stochastic
        stoch = self.indicators.stochastic(df['High'], df['Low'], df['Close'], 14)
        df = pd.concat([df, stoch], axis=1)
        
        # === VOLATILITY (Critical for Day Trading) ===
        df['ATR_14h'] = self.indicators.atr(df['High'], df['Low'], df['Close'], 14)
        df['ATR_Pct'] = (df['ATR_14h'] / df['Close']) * 100
        
        # === BOLLINGER BANDS ===
        bb = self.indicators.bollinger_bands(df['Close'], window=20, std_dev=2)
        df = pd.concat([df, bb], axis=1)
        
        # === VOLUME FEATURES (Very important for day trading) ===
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Momentum'] = df['Volume'].pct_change(5)
        
        # === INTRADAY PRICE PATTERNS ===
        # Hourly returns
        df['Return_1h'] = df['Close'].pct_change(1)
        df['Return_5h'] = df['Close'].pct_change(5)
        
        # Range (trading range per hour)
        df['Range'] = df['High'] - df['Low']
        df['Range_Pct'] = (df['Range'] / df['Close']) * 100
        
        # === CANDLESTICK PATTERNS ===
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # === GAP ANALYSIS (Between hours) ===
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
        
        # === TIME OF DAY FEATURES (UK Market: 8:00 AM - 4:30 PM GMT) ===
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                df['Hour'] = df['Date'].dt.hour
                df['Is_Morning'] = ((df['Hour'] >= 8) & (df['Hour'] <= 12)).astype(int)
                df['Is_Afternoon'] = ((df['Hour'] >= 13) & (df['Hour'] <= 16)).astype(int)
            except:
                df['Hour'] = 0
                df['Is_Morning'] = 0
                df['Is_Afternoon'] = 0
        
        # === MA CROSSOVERS ===
        df['SMA_Cross_5_20'] = df['SMA_5h'] - df['SMA_20h']
        df['EMA_Cross_5_10'] = df['EMA_5h'] - df['EMA_10h']
        
        # === BB POSITION ===
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # === LAG FEATURES (Previous hour values) ===
        df['Close_Lag_1h'] = df['Close'].shift(1)
        df['Close_Lag_2h'] = df['Close'].shift(2)
        df['Volume_Lag_1h'] = df['Volume'].shift(1)
        df['RSI_14h_Lag_1h'] = df['RSI_14h'].shift(1)
        
        return df
    
    def add_day_trading_target(self, df: pd.DataFrame, hours_forward: int = 2, 
                               threshold: float = 0.005) -> pd.DataFrame:
        """
        Create target for day trading (predict direction in next N hours)
        
        Args:
            df: DataFrame with features
            hours_forward: How many hours to look ahead
            threshold: Minimum price change to signal direction (0.5%)
        """
        # Calculate future price
        future_price = df['Close'].shift(-hours_forward)
        
        # Calculate forward return
        future_return = (future_price / df['Close']) - 1
        
        # Create target: 1 = buy (up), 0 = sell/hold (down)
        df['Target_2h'] = (future_return > threshold).astype(int)
        
        # Clean infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df

def process_uk_data():
    """Process all UK stock data"""
    
    print("=" * 70)
    print("🔧 Processing UK Intraday Data")
    print("=" * 70)
    print()
    
    # Input and output directories
    raw_dir = "data/uk/raw"
    processed_dir = "data/uk/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all files
    pattern = os.path.join(raw_dir, "*_1h_60d.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No UK data files found!")
        return
    
    print(f"📁 Found {len(files)} files to process\n")
    
    extractor = UKIntradayFeatureExtractor()
    processed_count = 0
    
    for filepath in tqdm(files, desc="Processing"):
        try:
            # Load
            ticker = os.path.basename(filepath).split('_')[0]
            df = pd.read_csv(filepath)
            
            print(f"\n📊 Processing {ticker}...")
            
            # Add features
            df_features = extractor.add_intraday_features(df)
            
            # Add target (2-hour prediction)
            df_features = extractor.add_day_trading_target(df_features, hours_forward=2, threshold=0.005)
            
            # Clean infinity and NaN
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_clean = df_features.dropna()
            
            # Save
            output_file = os.path.join(processed_dir, f"{ticker}_features.csv")
            df_clean.to_csv(output_file, index=False)
            
            print(f"   ✓ {len(df_clean)} valid rows")
            print(f"   ✓ Features: {len(df_features.columns)} total")
            print(f"   ✓ Saved: {output_file}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n✅ Processed {processed_count}/{len(files)} stocks")

if __name__ == "__main__":
    process_uk_data()

