"""
Data validation module for checking data quality
"""
import pandas as pd
from typing import Tuple, List
import numpy as np

class DataValidator:
    """Validates market data quality"""
    
    def __init__(self, min_data_points: int = 100, max_missing_pct: float = 5.0):
        self.min_data_points = min_data_points
        self.max_missing_pct = max_missing_pct
    
    def validate_data(self, data: pd.DataFrame, ticker: str = "Unknown") -> Tuple[bool, List[str]]:
        """
        Validate market data quality
        
        Args:
            data: DataFrame with market data
            ticker: Ticker symbol for error messages
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check if data is empty
        if data is None or data.empty:
            return False, [f"{ticker}: Empty dataset"]
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            errors.append(
                f"{ticker}: Insufficient data points "
                f"({len(data)} < {self.min_data_points})"
            )
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"{ticker}: Missing required columns: {missing_cols}")
        
        # Check for missing values in critical columns
        for col in required_cols:
            if col in data.columns:
                missing_pct = data[col].isna().sum() / len(data) * 100
                if missing_pct > self.max_missing_pct:
                    errors.append(
                        f"{ticker}: High missing data in {col} "
                        f"({missing_pct:.2f}% > {self.max_missing_pct}%)"
                    )
        
        # Check for invalid price data
        if 'Close' in data.columns:
            invalid_prices = (data['Close'] <= 0).sum()
            if invalid_prices > 0:
                errors.append(
                    f"{ticker}: Found {invalid_prices} invalid prices (≤ 0)"
                )
        
        if 'Volume' in data.columns:
            invalid_volume = (data['Volume'] < 0).sum()
            if invalid_volume > 0:
                errors.append(
                    f"{ticker}: Found {invalid_volume} invalid volumes (< 0)"
                )
        
        # Check for date monotonicity
        if 'Date' in data.columns:
            if not data['Date'].is_monotonic_increasing:
                errors.append(f"{ticker}: Dates are not in ascending order")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing invalid rows
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicate dates
        if 'Date' in data.columns:
            data = data.drop_duplicates(subset=['Date'])
            data = data.sort_values('Date')
        
        # Remove rows with invalid prices
        data = data[(data['Close'] > 0) & (data['High'] > 0) & 
                   (data['Low'] > 0) & (data['Open'] > 0)]
        
        # Remove rows with negative volume
        data = data[data['Volume'] >= 0]
        
        return data
    
    def get_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Get summary statistics of the data
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with summary statistics
        """
        if data is None or data.empty:
            return {}
        
        summary = {
            'total_rows': len(data),
            'date_range': {
                'start': data['Date'].min() if 'Date' in data.columns else None,
                'end': data['Date'].max() if 'Date' in data.columns else None,
            },
            'missing_data': {},
            'price_stats': {},
        }
        
        # Calculate missing data percentages
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                summary['missing_data'][col] = {
                    'count': data[col].isna().sum(),
                    'percentage': data[col].isna().sum() / len(data) * 100
                }
        
        # Calculate price statistics
        if 'Close' in data.columns:
            summary['price_stats'] = {
                'min': data['Close'].min(),
                'max': data['Close'].max(),
                'mean': data['Close'].mean(),
                'std': data['Close'].std(),
            }
        
        return summary

