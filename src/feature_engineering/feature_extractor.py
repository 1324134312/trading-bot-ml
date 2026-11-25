"""
Feature Extractor Module
Applies technical indicators to create features for ML models
"""
import pandas as pd
import numpy as np
from typing import Optional
from .indicators import TechnicalIndicators

class FeatureExtractor:
    """Extract features from raw OHLCV data using technical indicators"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def add_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all available technical indicators to the dataset
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added feature columns
        """
        df = data.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print("⚠️  Missing required columns:", required_cols)
            return df
        
        # Calculate Moving Averages
        df['SMA_5'] = self.indicators.sma(df['Close'], 5)
        df['SMA_10'] = self.indicators.sma(df['Close'], 10)
        df['SMA_20'] = self.indicators.sma(df['Close'], 20)
        df['SMA_50'] = self.indicators.sma(df['Close'], 50)
        df['SMA_200'] = self.indicators.sma(df['Close'], 200)
        
        df['EMA_5'] = self.indicators.ema(df['Close'], 5)
        df['EMA_10'] = self.indicators.ema(df['Close'], 10)
        df['EMA_20'] = self.indicators.ema(df['Close'], 20)
        df['EMA_50'] = self.indicators.ema(df['Close'], 50)
        
        # Calculate RSI
        df['RSI_14'] = self.indicators.rsi(df['Close'], 14)
        df['RSI_7'] = self.indicators.rsi(df['Close'], 7)
        
        # Calculate MACD
        macd_data = self.indicators.macd(df['Close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd_data], axis=1)
        
        # Calculate Bollinger Bands
        bb_data = self.indicators.bollinger_bands(df['Close'], window=20, std_dev=2)
        df = pd.concat([df, bb_data], axis=1)
        
        # Calculate Stochastic Oscillator
        stoch_data = self.indicators.stochastic(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, stoch_data], axis=1)
        
        # Calculate ATR
        df['ATR_14'] = self.indicators.atr(df['High'], df['Low'], df['Close'], 14)
        
        # Calculate ADX
        df['ADX_14'] = self.indicators.adx(df['High'], df['Low'], df['Close'], 14)
        
        # Calculate OBV
        df['OBV'] = self.indicators.obv(df['Close'], df['Volume'])
        
        # Calculate CCI
        df['CCI_20'] = self.indicators.cci(df['High'], df['Low'], df['Close'], 20)
        
        # Calculate Williams %R
        df['Williams_R'] = self.indicators.williams_r(df['High'], df['Low'], df['Close'], 14)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/engineered features
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            DataFrame with additional derived features
        """
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(5)
        df['Price_Change_10'] = df['Close'].pct_change(10)
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Range'] = df['High'] - df['Low']
        df['Range_Pct'] = (df['Range'] / df['Close']) * 100
        
        # Moving average crossovers
        df['SMA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
        df['SMA_Cross_10_50'] = df['SMA_10'] - df['SMA_50']
        df['EMA_Cross_5_20'] = df['EMA_5'] - df['EMA_20']
        
        # Bollinger Band position
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # RSI divergence
        df['RSI_Momentum'] = df['RSI_14'] - df['RSI_14'].shift(1)
        
        # MACD histogram momentum
        df['MACD_Hist_Momentum'] = df['MACD_Histogram'] - df['MACD_Histogram'].shift(1)
        
        # Support/Resistance (simplified)
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Higher High, Higher Low pattern
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        
        # Gap
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged features for time series prediction
        
        Args:
            df: DataFrame with features
            lags: List of lag periods to add
            
        Returns:
            DataFrame with lag features added
        """
        key_features = ['Close', 'RSI_14', 'MACD_Histogram', 'Volume']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, forward_period: int = 1, threshold: float = 0.02) -> pd.DataFrame:
        """
        Create target variable for ML prediction
        
        Args:
            df: DataFrame with features
            forward_period: How many periods to look ahead
            threshold: Minimum price change to signal buy (>0) or sell (<0)
            
        Returns:
            DataFrame with 'Target' column added
        """
        # Calculate future return
        future_return = df['Close'].shift(-forward_period) / df['Close'] - 1
        
        # Create target: 1 for buy (increase), -1 for sell (decrease), 0 for hold
        df['Target'] = 0
        df.loc[future_return > threshold, 'Target'] = 1  # Buy signal
        df.loc[future_return < -threshold, 'Target'] = -1  # Sell signal
        
        # Alternative: predict direction only (simpler)
        df['Target_Direction'] = (future_return > 0).astype(int)
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame, feature_cols: Optional[list] = None) -> tuple:
        """
        Prepare data for machine learning
        
        Args:
            df: DataFrame with all features
            feature_cols: List of feature columns to use (None = auto-select)
            
        Returns:
            Tuple of (X, y) for ML training
        """
        # Drop missing values
        df = df.dropna()
        
        # Select features (exclude metadata and target columns)
        exclude_cols = ['Date', 'Ticker', 'FetchDate', 'Target', 'Target_Direction']
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Extract features and target
        X = df[feature_cols].values
        y = df['Target_Direction'].values if 'Target_Direction' in df.columns else df['Target'].values
        
        return X, y, feature_cols

