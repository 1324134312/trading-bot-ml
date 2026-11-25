"""
Data fetcher module for downloading market data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
from typing import List, Optional
import time

class DataFetcher:
    """Handles fetching and saving market data from Yahoo Finance"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_ticker_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        retry_count: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            retry_count: Number of retry attempts on failure
            
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        ticker = ticker.upper()
        
        for attempt in range(retry_count):
            try:
                stock = yf.Ticker(ticker)
                
                # Fetch historical data
                data = stock.history(period=period, interval=interval)
                
                if data.empty:
                    print(f"⚠️  No data available for {ticker}")
                    return None
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Add metadata
                data['Ticker'] = ticker
                data['FetchDate'] = datetime.now()
                
                print(f"✓ Successfully fetched {len(data)} rows for {ticker}")
                return data
                
            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️  Error fetching {ticker}: {str(e)}")
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed to fetch {ticker} after {retry_count} attempts: {str(e)}")
                    return None
        
        return None
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        save: bool = True
    ) -> dict:
        """
        Fetch data for multiple tickers
        
        Args:
            tickers: List of stock symbols
            period: Valid periods
            interval: Valid intervals
            save: Whether to save files to disk
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        failed_tickers = []
        
        print(f"\n🔄 Fetching data for {len(tickers)} tickers...")
        print(f"📊 Period: {period}, Interval: {interval}\n")
        
        for ticker in tqdm(tickers, desc="Downloading"):
            data = self.fetch_ticker_data(ticker, period, interval)
            
            if data is not None:
                results[ticker] = data
                
                # Save to file
                if save:
                    filename = f"{ticker}_{period}_{interval}.csv"
                    filepath = os.path.join(self.data_dir, filename)
                    data.to_csv(filepath, index=False)
                    print(f"💾 Saved: {filepath}")
            else:
                failed_tickers.append(ticker)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        print(f"\n✅ Successfully fetched {len(results)}/{len(tickers)} tickers")
        if failed_tickers:
            print(f"❌ Failed tickers: {', '.join(failed_tickers)}")
        
        return results
    
    def load_ticker_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Load previously saved data from disk
        
        Args:
            ticker: Stock symbol
            period: Period of data
            interval: Interval of data
            
        Returns:
            DataFrame if file exists, None otherwise
        """
        filename = f"{ticker}_{period}_{interval}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return None
    
    def get_ticker_info(self, ticker: str) -> dict:
        """
        Get basic information about a ticker
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with ticker info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'symbol': ticker,
                'name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
            }
        except Exception as e:
            print(f"Error fetching info for {ticker}: {str(e)}")
            return {}

