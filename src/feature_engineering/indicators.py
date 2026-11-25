"""
Technical Indicators Module
Calculates various technical analysis indicators
"""
import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            data: Price series (usually Close prices)
            window: Period for moving average
            
        Returns:
            Series with SMA values
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            data: Price series
            window: Period for EMA
            
        Returns:
            Series with EMA values
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Price series (usually Close prices)
            window: Period for RSI calculation (default: 14)
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Bollinger Bands
        
        Args:
            data: Price series
            window: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Upper, Middle (SMA), and Lower bands
        """
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band
        })
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
        """
        Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for calculation
            
        Returns:
            DataFrame with %K and %D values
        """
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        })
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for ATR calculation
            
        Returns:
            Series with ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average Directional Index
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for ADX calculation
            
        Returns:
            Series with ADX values (0-100)
        """
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = TechnicalIndicators.atr(high, low, close, 1)
        atr = tr.rolling(window=window).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        
        Args:
            close: Close prices
            volume: Trading volume
            
        Returns:
            Series with OBV values
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for CCI calculation
            
        Returns:
            Series with CCI values
        """
        tp = (high + low + close) / 3
        sma = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (tp - sma) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Period for Williams %R
            
        Returns:
            Series with Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return wr

