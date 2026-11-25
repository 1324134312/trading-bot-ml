"""
Process UK 5-minute data and extract features for day trading
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import glob
from src.feature_engineering.indicators import TechnicalIndicators
from tqdm import tqdm

class UK5MinFeatureExtractor:
    """Feature engineering for 5-minute day trading"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def add_5min_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features optimized for 5-minute trading"""
        
        df = data.copy()
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            return df
        
        # === ULTRA SHORT-TERM MOVING AVERAGES ===
        # For 5-min bars, use shorter periods
        df['SMA_5'] = self.indicators.sma(df['Close'], 5)    # 25 minutes
        df['SMA_10'] = self.indicators.sma(df['Close'], 10)   # 50 minutes
        df['SMA_20'] = self.indicators.sma(df['Close'], 20)  # 100 minutes (~1.5 hours)
        df['EMA_5'] = self.indicators.ema(df['Close'], 5)
        df['EMA_10'] = self.indicators.ema(df['Close'], 10)
        
        # === MOMENTUM (Faster for intraday) ===
        df['RSI_10'] = self.indicators.rsi(df['Close'], 10)
        df['RSI_5'] = self.indicators.rsi(df['Close'], 5)
        
        # MACD (faster for 5-min)
        macd = self.indicators.macd(df['Close'], fast=8, slow=17, signal=6)
        df = pd.concat([df, macd], axis=1)
        
        # Stochastic
        stoch = self.indicators.stochastic(df['High'], df['Low'], df['Close'], 10)
        df = pd.concat([df, stoch], axis=1)
        
        # === VOLATILITY (Critical) ===
        df['ATR_10'] = self.indicators.atr(df['High'], df['Low'], df['Close'], 10)
        df['ATR_Pct'] = (df['ATR_10'] / df['Close']) * 100
        
        # === BOLLINGER BANDS ===
        bb = self.indicators.bollinger_bands(df['Close'], window=15, std_dev=2)
        df = pd.concat([df, bb], axis=1)
        
        # === VOLUME (Very important for 5-min) ===
        df['Volume_SMA'] = df['Volume'].rolling(15).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Momentum'] = df['Volume'].pct_change(3)
        
        # === INTRADAY PATTERNS ===
        df['Return_5min'] = df['Close'].pct_change(1)
        df['Return_30min'] = df['Close'].pct_change(6)
        
        # Range
        df['Range'] = df['High'] - df['Low']
        df['Range_Pct'] = (df['Range'] / df['Close']) * 100
        
        # === CANDLESTICK ===
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # === GAPS (Between 5-min bars) ===
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
        
        # === TIME OF DAY (UK Market 8:00-16:30 GMT) ===
        if 'Datetime' in df.columns:
            try:
                df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
                df['Hour'] = df['Datetime'].dt.hour
                df['Minute'] = df['Datetime'].dt.minute
                df['Is_Morning'] = ((df['Hour'] >= 8) & (df['Hour'] <= 12)).astype(int)
                df['Is_Afternoon'] = ((df['Hour'] >= 13) & (df['Hour'] <= 16)).astype(int)
                df['Is_Start'] = (df['Hour'] == 8).astype(int)
                df['Is_Close'] = (df['Hour'] >= 16).astype(int)
            except:
                df['Hour'] = 0
                df['Minute'] = 0
                df['Is_Morning'] = 0
                df['Is_Afternoon'] = 0
                df['Is_Start'] = 0
                df['Is_Close'] = 0
        
        # === MA CROSSOVERS ===
        df['SMA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
        df['EMA_Cross_5_10'] = df['EMA_5'] - df['EMA_10']
        
        # === BB POSITION ===
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # === MOMENTUM FEATURES ===
        df['RSI_Momentum'] = df['RSI_10'] - df['RSI_10'].shift(1)
        df['MACD_Hist_Momentum'] = df['MACD_Histogram'] - df['MACD_Histogram'].shift(1)
        
        # === LAG FEATURES ===
        df['Close_Lag_1'] = df['Close'].shift(1)
        df['Close_Lag_3'] = df['Close'].shift(3)
        df['Volume_Lag_1'] = df['Volume'].shift(1)
        df['RSI_Lag_1'] = df['RSI_10'].shift(1)
        
        return df
    
    def add_day_trading_target(self, df: pd.DataFrame, bars_forward: int = 3, 
                               threshold: float = 0.003) -> pd.DataFrame:
        """
        Create target for 5-min day trading
        Predict if price goes up in next N bars (15-30 minutes)
        """
        # Calculate future price (3 bars = 15 minutes ahead)
        future_price = df['Close'].shift(-bars_forward)
        
        # Forward return
        future_return = (future_price / df['Close']) - 1
        
        # Target: 1 = buy (0.3%+ move up), 0 = hold/sell
        df['Target_15min'] = (future_return > threshold).astype(int)
        
        # Clean
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df

def process_5min_data():
    """Process all 5-minute UK stock data"""
    
    print("=" * 70)
    print("🔧 Processing UK 5-Minute Day Trading Data")
    print("=" * 70)
    print()
    
    raw_dir = "data/uk/raw_wide"
    processed_dir = "data/uk/processed_5min"
    os.makedirs(processed_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    
    print(f"📁 Found {len(files)} files to process\n")
    
    extractor = UK5MinFeatureExtractor()
    processed_count = 0
    
    for filepath in tqdm(files, desc="Processing"):
        try:
            ticker = os.path.basename(filepath).split('_')[0]
            df = pd.read_csv(filepath)
            
            print(f"\n📊 {ticker}...")
            
            # Add features
            df_features = extractor.add_5min_features(df)
            
            # Add target (15-min ahead prediction)
            df_features = extractor.add_day_trading_target(df_features, bars_forward=3, threshold=0.003)
            
            # Clean
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            df_clean = df_features.dropna()
            
            # Save
            output_file = os.path.join(processed_dir, f"{ticker}_features.csv")
            df_clean.to_csv(output_file, index=False)
            
            print(f"   ✓ {len(df_clean)} valid bars")
            print(f"   ✓ {len(df_features.columns)} features")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
    
    print(f"\n✅ Processed {processed_count}/{len(files)} stocks")

if __name__ == "__main__":
    process_5min_data()

