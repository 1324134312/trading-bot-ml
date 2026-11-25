"""
Add Intraday-Specific Features

These are features specifically designed for 5-minute trading:
- Momentum indicators
- Volume patterns
- Time-based features
- Volatility measures
"""
import pandas as pd
import numpy as np

class IntradayFeatureExtractor:
    """Add features specific to intraday/5-minute trading"""
    
    def __init__(self):
        pass
    
    def add_intraday_features(self, df):
        """Add all intraday-specific features"""
        
        print("🔧 Adding intraday features...")
        
        # Copy dataframe
        df_new = df.copy()
        
        # Ensure Date is datetime
        if 'Date' in df_new.columns:
            df_new['Date'] = pd.to_datetime(df_new['Date'])
        
        # If Date is index, convert to column temporarily
        if isinstance(df_new.index, pd.DatetimeIndex):
            df_new['Date'] = df_new.index
            date_is_index = True
        else:
            date_is_index = False
        
        # Time-based features
        df_new['Hour'] = df_new['Date'].dt.hour
        df_new['Minute'] = df_new['Date'].dt.minute
        df_new['Day_of_Week'] = df_new['Date'].dt.dayofweek
        
        # Market session features
        df_new['Is_Market_Open'] = (df_new['Hour'] >= 9) & (df_new['Hour'] < 16)
        df_new['Is_Morning'] = (df_new['Hour'] >= 9) & (df_new['Hour'] < 12)
        df_new['Is_Midday'] = (df_new['Hour'] >= 12) & (df_new['Hour'] < 14)
        df_new['Is_Afternoon'] = (df_new['Hour'] >= 14) & (df_new['Hour'] < 16)
        df_new['Is_First_Hour'] = df_new['Hour'] == 9
        df_new['Is_Last_Hour'] = df_new['Hour'] == 15
        
        # Price change features (momentum)
        df_new['Price_Change'] = df_new['Close'].diff()
        df_new['Price_Change_Pct'] = df_new['Close'].pct_change()
        df_new['Price_Change_5'] = df_new['Close'].diff(5)  # Last 5 bars
        df_new['Price_Change_10'] = df_new['Close'].diff(10)  # Last 10 bars
        
        # Momentum features
        df_new['ROC_1'] = (df_new['Close'] / df_new['Close'].shift(1) - 1) * 100
        df_new['ROC_5'] = (df_new['Close'] / df_new['Close'].shift(5) - 1) * 100
        df_new['ROC_10'] = (df_new['Close'] / df_new['Close'].shift(10) - 1) * 100
        
        # Volatility features
        df_new['High_Low_Range'] = (df_new['High'] - df_new['Low']) / df_new['Close']
        df_new['High_Low_Range_Pct'] = df_new['High_Low_Range'] * 100
        df_new['Volatility_5'] = df_new['High_Low_Range_Pct'].rolling(5).std()
        df_new['Volatility_10'] = df_new['High_Low_Range_Pct'].rolling(10).std()
        
        # Volume features
        df_new['Volume'] = pd.to_numeric(df_new['Volume'], errors='coerce')
        df_new['Volume_Change'] = df_new['Volume'].pct_change()
        df_new['Volume_MA_5'] = df_new['Volume'].rolling(5).mean()
        df_new['Volume_MA_20'] = df_new['Volume'].rolling(20).mean()
        df_new['Volume_Ratio'] = df_new['Volume'] / df_new['Volume_MA_20']
        df_new['Volume_Momentum'] = df_new['Volume'] / df_new['Volume'].shift(5)
        
        # Candle features
        df_new['Body'] = abs(df_new['Close'] - df_new['Open']) / df_new['Close']
        df_new['Upper_Wick'] = (df_new['High'] - df_new[['Open', 'Close']].max(axis=1)) / df_new['Close']
        df_new['Lower_Wick'] = (df_new[['Open', 'Close']].min(axis=1) - df_new['Low']) / df_new['Close']
        df_new['Is_Green_Candle'] = (df_new['Close'] > df_new['Open']).astype(int)
        df_new['Is_Red_Candle'] = (df_new['Close'] < df_new['Open']).astype(int)
        df_new['Is_Doji'] = (abs(df_new['Open'] - df_new['Close']) / df_new['Close'] < 0.001).astype(int)
        
        # Trend features
        df_new['Price_Above_Open'] = (df_new['Close'] > df_new['Open']).astype(int)
        df_new['Price_Below_Open'] = (df_new['Close'] < df_new['Open']).astype(int)
        
        # MA distance features
        if 'SMA_5' in df_new.columns:
            df_new['Price_vs_SMA5'] = (df_new['Close'] - df_new['SMA_5']) / df_new['SMA_5']
        if 'SMA_10' in df_new.columns:
            df_new['Price_vs_SMA10'] = (df_new['Close'] - df_new['SMA_10']) / df_new['SMA_10']
        if 'EMA_5' in df_new.columns:
            df_new['Price_vs_EMA5'] = (df_new['Close'] - df_new['EMA_5']) / df_new['EMA_5']
        
        # Consecutive features
        df_new['Consecutive_Up'] = (df_new['Close'] > df_new['Close'].shift(1)).astype(int)
        df_new['Consecutive_Down'] = (df_new['Close'] < df_new['Close'].shift(1)).astype(int)
        
        # Gaps and jumps
        df_new['Gap'] = df_new['Open'] - df_new['Close'].shift(1)
        df_new['Gap_Pct'] = df_new['Gap'] / df_new['Close'].shift(1) * 100
        
        # Fill NaN values
        df_new = df_new.fillna(0)
        
        # Replace inf values
        df_new = df_new.replace([np.inf, -np.inf], 0)
        
        # Remove Date column if it was added
        if date_is_index:
            df_new = df_new.set_index('Date')
        
        print(f"✅ Added intraday features. Total features: {len(df_new.columns)}")
        
        return df_new

def test_intraday_features():
    """Test the intraday feature extraction"""
    
    print("=" * 60)
    print("🧪 TESTING INTRADAY FEATURE EXTRACTION")
    print("=" * 60)
    
    # Get some sample data
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.get_today_5min_data import get_today_5min_data
    
    print("📊 Fetching sample data...")
    data = get_today_5min_data("AAPL")
    
    if data is None or data.empty:
        print("❌ No data available")
        return
    
    # Rename columns to match expected format
    data = data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    print(f"✅ Got {len(data)} bars of data")
    
    # Add intraday features
    extractor = IntradayFeatureExtractor()
    data_with_features = extractor.add_intraday_features(data)
    
    print(f"\n📊 Feature Summary:")
    print(f"   Total columns: {len(data_with_features.columns)}")
    print(f"   Sample features: {list(data_with_features.columns[:20])}")
    
    # Show some stats
    print(f"\n📈 Sample Data:")
    print(data_with_features[['Close', 'Hour', 'Minute', 'Is_Morning', 'Volume_Ratio', 'ROC_5']].head(10))
    
    print("\n" + "=" * 60)
    print("✅ INTRADAY FEATURE TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_intraday_features()
